#!/usr/bin/env python3
"""Stream a puffer run's metrics jsonl (base.wandb=True) to wandb.

Wrapper (live):  wandb_sync.py --project P [--group G] [--name N] -- ./puffer train ENV ...
                 injects base.wandb=True base.run_id=<name>, follows the file
Import (post):   wandb_sync.py --project P --file logs/ENV/RUN.jsonl [--name N]
Sweep (live):    wandb_sync.py --project P [--group G] --watch logs/ENV
                 one wandb run per new *.jsonl; a run finishes when its file
                 goes idle for --idle seconds
"""
import argparse
import configparser
import json
import os
import subprocess
import sys
import time


def read_config(ini_path):
    cp = configparser.ConfigParser(strict=False)
    try:
        cp.read(ini_path)
        return {f"{s}.{k}": v for s in cp.sections() for k, v in cp[s].items()}
    except Exception:
        return {}


def base_key(configs, key, default):
    cp = configparser.ConfigParser(strict=False)
    cp.read(configs)
    return cp.get("base", key, fallback=default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--entity", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--tags", default=None, help="comma-separated")
    ap.add_argument("--file", default=None, help="import an existing jsonl")
    ap.add_argument("--watch", default=None, help="dir: one wandb run per new jsonl")
    ap.add_argument("--idle", type=float, default=0.0,
                    help="file mode: keep following until idle this long (0 = stop at EOF)")
    ap.add_argument("cmd", nargs="*", help="-- ./puffer train ENV [overrides]")
    args = ap.parse_args()

    if args.watch:
        seen = {}
        try:
            while True:
                for fn in sorted(os.listdir(args.watch)):
                    if fn.endswith(".jsonl") and fn not in seen:
                        fp = os.path.join(args.watch, fn)
                        cmd = [sys.executable, __file__, "--project", args.project,
                               "--file", fp, "--idle", str(args.idle or 180.0)]
                        for flag in ("entity", "group", "tags"):
                            v = getattr(args, flag)
                            if v:
                                cmd += [f"--{flag}", v]
                        print(f"wandb_sync: new trial {fn}")
                        seen[fn] = subprocess.Popen(cmd)
                time.sleep(2.0)
        except KeyboardInterrupt:
            for c in seen.values():
                c.wait()
            return

    child = None
    if args.cmd:
        env_name = next(
            (a for a in args.cmd[1:] if not a.startswith("-") and "=" not in a
             and a not in ("train", "eval", "sweep", "match")), None)
        assert env_name, "could not find ENV in command"
        name = args.name or f"{env_name}_{int(time.time())}"
        log_dir = base_key(
            ["config/default.ini", f"config/{env_name}.ini"], "log_dir", "logs")
        path = os.path.join(log_dir, env_name, f"{name}.jsonl")
        if os.path.exists(path):
            os.remove(path)
        child = subprocess.Popen(args.cmd + ["base.wandb=True", f"base.run_id={name}"])
    else:
        assert args.file, "need --file or a command after --"
        path = args.file
        name = args.name or os.path.basename(path)[:-len(".jsonl")]

    import wandb
    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group, name=name,
        tags=args.tags.split(",") if args.tags else None)
    wandb.define_metric("agent_steps")
    wandb.define_metric("*", step_metric="agent_steps")

    f = None
    config_sent = False
    last = None
    rows = 0
    last_data = time.time()
    while True:
        if f is None:
            if os.path.exists(path):
                f = open(path)
            elif child is not None and child.poll() is not None:
                break
            else:
                time.sleep(0.5)
                continue
        line = f.readline()
        if not line:
            if child is not None:
                if child.poll() is not None:
                    break
            elif args.idle <= 0 or time.time() - last_data > args.idle:
                break
            time.sleep(1.0)
            continue
        if not line.endswith("\n"):  # partial write; rewind and retry
            f.seek(f.tell() - len(line))
            time.sleep(0.2)
            continue
        if not config_sent:
            cfg = read_config(path[:-len(".jsonl")] + ".ini")
            if cfg:
                run.config.update(cfg)
            config_sent = True
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
        wandb.log(last)
        rows += 1
        last_data = time.time()

    if last:
        run.summary.update(last)
    rc = child.wait() if child is not None else 0
    print(f"wandb_sync: {rows} rows -> {run.url}")
    run.finish(exit_code=rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
