#include "onlyfish.h"
#include "puffernet.h"
#include <dirent.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const char* dirpath = "resources/onlyfish/";
    DIR* dir = opendir(dirpath);
    if (dir == NULL) {
        perror("Unable to open directory");
        return 1;
    }

    int num_agents = 0;
    struct dirent* entry;
    while ((entry = readdir(dir))) {
        if (entry->d_type == DT_REG) {
            size_t len = strlen(entry->d_name);
            if (len > 4 && strcmp(entry->d_name + len - 4, ".bin") == 0) {
                num_agents++;
            }
        }
    }
    closedir(dir);

    if (num_agents == 0) {
        fprintf(stderr, "No .bin files found in directory\n");
        return 1;
    }

    LinearLSTM** nets = calloc(num_agents, sizeof(LinearLSTM*));
    char** names = calloc(num_agents, sizeof(char*));

    dir = opendir(dirpath);
    int idx = 0;
    while ((entry = readdir(dir))) {
        if (entry->d_type == DT_REG) {
            size_t len = strlen(entry->d_name);
            if (len > 4 && strcmp(entry->d_name + len - 4, ".bin") == 0) {
                char fullpath[256];
                sprintf(fullpath, "%s%s", dirpath, entry->d_name);
                Weights* weights = load_weights(fullpath, 136847);
                int logit_sizes[2] = {9, 5};
                nets[idx] = make_linearlstm(weights, 1, 21, logit_sizes, 2);

                names[idx] = strdup(entry->d_name);
                char* dot = strrchr(names[idx], '.');
                if (dot) *dot = '\0';

                idx++;
            }
        }
    }
    closedir(dir);

    int num_goals = 4;
    int num_obs = 21;

    OnlyFish env = {
        .width = 1280,
        .height = 720,
        .num_agents = num_agents,
        .num_goals = num_goals,
        .names = names
    };
    init(&env);

    env.observations = calloc(env.num_agents * num_obs, sizeof(float));
    env.actions = calloc(2 * env.num_agents, sizeof(int));
    env.rewards = calloc(env.num_agents, sizeof(float));
    env.terminals = calloc(env.num_agents, sizeof(unsigned char));

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        for (int i = 0; i < num_agents; i++) {
            forward_linearlstm(nets[i], env.observations + i * num_obs, env.actions + 2 * i);
        }
        c_step(&env);
        c_render(&env);
    }

    for (int i = 0; i < num_agents; i++) {
        free_linearlstm(nets[i]);
        free(names[i]);
    }
    free(nets);
    free(names);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}
