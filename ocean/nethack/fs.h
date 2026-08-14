// Filesystem plumbing: one private vardir per env (NetHack wants nhdat +
// writable files) under a per-process tmpfs parent, plus the shared options rc.
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <signal.h>

// helpers

static void nethack_touch(const char* path) {
    int fd = open(path, O_CREAT | O_WRONLY, 0644);
    if (fd >= 0) close(fd);
}

static void nethack_rm_rf(const char* path, int depth) {
    if (depth > 3) return; // vardir trees are at most base/env/save/files
    DIR* d = opendir(path);
    if (d) {
        struct dirent* e;
        char p[2048];
        while ((e = readdir(d)) != NULL) {
            if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
            snprintf(p, sizeof(p), "%s/%s", path, e->d_name);
            if (unlink(p) != 0) nethack_rm_rf(p, depth + 1);
        }
        closedir(d);
    }
    rmdir(path);
}

// vardirs

// per-process parent on tmpfs; first use sweeps parents whose owner pid died
static const char* nethack_vardir_base(void) {
    static char base[256] = "";
    if (base[0]) return base;
    const char* root = access("/dev/shm", W_OK) == 0 ? "/dev/shm" : "/tmp";
    DIR* d = opendir(root);
    if (d) {
        struct dirent* e;
        while ((e = readdir(d)) != NULL) {
            long pid = 0;
            if (sscanf(e->d_name, "nle-run-%ld", &pid) != 1 || pid <= 0) continue;
            if (pid == (long)getpid()) continue;
            if (kill((pid_t)pid, 0) == 0 || errno != ESRCH) continue;  // owner alive
            char dead[512];
            snprintf(dead, sizeof(dead), "%s/%s", root, e->d_name);
            nethack_rm_rf(dead, 0);
        }
        closedir(d);
    }
    snprintf(base, sizeof(base), "%s/nle-run-%ld", root, (long)getpid());
    mkdir(base, 0755);
    return base;
}

static int nethack_make_vardir(const char* source_hackdir, char* out_buf, size_t out_cap) {
    char tmpl[512];
    snprintf(tmpl, sizeof(tmpl), "%s/env-XXXXXX", nethack_vardir_base());
    char* dir = mkdtemp(tmpl);
    if (dir == NULL) return -1;
    if ((size_t)snprintf(out_buf, out_cap, "%s", dir) >= out_cap) return -1;

    char abs_source[1024];
    if (source_hackdir[0] == '/') {
        snprintf(abs_source, sizeof(abs_source), "%s", source_hackdir);
    } else {
        char cwd[768];
        if (getcwd(cwd, sizeof(cwd)) == NULL) return -1;
        snprintf(abs_source, sizeof(abs_source), "%s/%s", cwd, source_hackdir);
    }

    char src[1280], dst[1280];
    snprintf(src, sizeof(src), "%s/nhdat", abs_source);
    snprintf(dst, sizeof(dst), "%s/nhdat", dir);

    // fail fast on a dangling nhdat symlink: symlink(2) would succeed and the
    // error surface later as a cryptic init_dungeons panic
    char resolved[4096]; // realpath(3) requires a PATH_MAX buffer
    if (realpath(src, resolved) == NULL || access(resolved, R_OK) != 0) {
        fprintf(stderr,
                "nethack: NETHACKDIR misconfigured — no readable nhdat at %s (%s).\n"
                "Set NETHACKDIR to an absolute dir containing nhdat, e.g. "
                "<repo>/vendor/fast-nle/build/dat.\n", src, strerror(errno));
        exit(1);
    }
    if (symlink(src, dst) != 0) return -1;
    const char* touched[] = {"perm", "record", "logfile", "xlogfile"};
    for (size_t i = 0; i < 4; i++) {
        snprintf(dst, sizeof(dst), "%s/%s", dir, touched[i]);
        nethack_touch(dst);
    }
    snprintf(dst, sizeof(dst), "%s/save", dir);
    mkdir(dst, 0755);
    return 0;
}

static void nethack_rm_vardir(const char* dir) {
    if (dir == NULL || dir[0] == '\0') return;
    // full tree: the game drops level/lock files beyond the fixed set
    nethack_rm_rf(dir, 1);
}

// options rc

// one rc per process: AUTOPICKUP_EXCEPTION is a config-file-only directive, so
// the options string becomes "@<this file>"; written atomically (tmp + rename),
// concurrent env inits write identical content
// content-addressed rc: distinct options content never shares a file
static const char* nethack_rc_path_opts(char* buf, size_t bufsz, const char* options) {
    unsigned h = 2166136261u;
    for (const char* c = options; *c; c++) h = (h ^ (unsigned char)*c) * 16777619u;
    snprintf(buf, bufsz, "%s/nhrc_%08x", nethack_vardir_base(), h);
    if (access(buf, R_OK) != 0) {
        char tmp[560];
        snprintf(tmp, sizeof(tmp), "%s.%p", buf, (void*)&tmp);
        FILE* f = fopen(tmp, "w");
        if (f) {
            fprintf(f, "OPTIONS=%s\n", options);
            fprintf(f, "AUTOPICKUP_EXCEPTION=\">corpse\"\n");
            fclose(f);
            rename(tmp, buf);
        }
    }
    return buf;
}

static const char* nethack_rc_path(const char* default_options) {
    static char path[512] = "";
    if (path[0]) return path;
    char p[512], tmp[560];
    snprintf(p, sizeof(p), "%s/nhrc", nethack_vardir_base());
    if (access(p, R_OK) != 0) {
        snprintf(tmp, sizeof(tmp), "%s.%p", p, (void*)&tmp);
        FILE* f = fopen(tmp, "w");
        if (f) {
            fprintf(f, "OPTIONS=%s\n", default_options);
            // corpses are never AUTO-picked: acquiring one is a deliberate
            // PICKUP, and eating carried corpses stays policy-learnable
            fprintf(f, "AUTOPICKUP_EXCEPTION=\">corpse\"\n");
            fclose(f);
            rename(tmp, p);
        }
    }
    strncpy(path, p, sizeof(path) - 1);
    return path;
}
