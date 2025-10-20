#ifndef COIN_FINDER_H
#define COIN_FINDER_H

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#define GRID_SIZE 10  // (10x10)
#define NUM_COINS 5
#define MAX_STEPS 100

#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3

// Required Log struct (you have this - good!)
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

// Main environment struct
typedef struct {
    Log log;  // Required
    
    int agent_x;
    int agent_y;
    int coin_x[NUM_COINS];
    int coin_y[NUM_COINS];
    bool coin_collected[NUM_COINS];
    int n_collected_coins;
    int n_steps;
    
    // Required buffers (DON'T FORGET THESE!)
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    
} CoinFinder;

void compute_observations(CoinFinder* env) {
    int obs_idx = 0;
    
    // Store agent position normalized.
    env->observations[obs_idx++] = (float)env->agent_x / GRID_SIZE;
    env->observations[obs_idx++] = (float)env->agent_y / GRID_SIZE;
    
    for (int i = 0; i < NUM_COINS; i++) {
        if (env->coin_collected[i]) {
            // If coin IS collected we don't show it
            env->observations[obs_idx++] = -1.0f;
            env->observations[obs_idx++] = -1.0f;
        } else {
            // If coin IS NOT collected we store its position normalized.
            env->observations[obs_idx++] = (float)env->coin_x[i] / GRID_SIZE;
            env->observations[obs_idx++] = (float)env->coin_y[i] / GRID_SIZE;
        }
    }
}

// TODO: Add function declarations
void c_reset(CoinFinder* env){
    env -> agent_x = rand() % GRID_SIZE;
    env -> agent_y = rand() % GRID_SIZE;
    env -> n_collected_coins = 0;
    env -> n_steps = 0;

    for (int i=0; i< NUM_COINS; i++) {
        env -> coin_x[i] = rand() % GRID_SIZE;
        env -> coin_y[i] = rand() % GRID_SIZE;
        env -> coin_collected[i] = 0;
    }

    compute_observations(env);
}

int clip(int val, int min, int max){
    // Function to clip the position to a [0, GRID_SIZE - 1] grid 
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

void c_step(CoinFinder* env) {
    env -> n_steps++;
    
    // 1. Get the action
    int action = env -> actions[0];

    // 2. Move the agent based on the action
    switch (action){
        case UP:
            env -> agent_y --;
            break;
        case DOWN:
            env -> agent_y ++;
            break;
        case LEFT:
            env -> agent_x --;
            break;
        case RIGHT:
            env -> agent_x ++;
            break;
    }

    // 3. Clip the agent to the grid
    env -> agent_x = clip(env -> agent_x, 0, GRID_SIZE - 1);
    env -> agent_y = clip(env -> agent_y, 0, GRID_SIZE - 1);
            
    // 4. Check if the agent collected a coin

    env -> rewards[0] = 0.0f;

    for (int i=0; i < NUM_COINS; i++){
        if (env -> agent_x == env -> coin_x[i] && env -> agent_y == env -> coin_y[i]){
            if (!env->coin_collected[i]) {
                // Update status
                env->coin_collected[i] = 1;
                env->n_collected_coins++;

                // Give reward
                env->rewards[0] = 1.0f;
                
                // Update log
                env->log.perf += 1.0f;
                env->log.score += 1.0f;
                env->log.episode_return += 1.0f;
                env->log.n++;
            }
        }
    }
        
    // 5. Check if episode is finished
    if (env->n_collected_coins == NUM_COINS || env->n_steps >= MAX_STEPS) {
        env->terminals[0] = 1;
        env->log.episode_length = env->n_steps;
    }
    
    // TODO: Compute observations
    compute_observations(env);
}

void c_render(CoinFinder* env) {
    // Clear screen (simple way)
    printf("\n=== Coin Finder ===\n");
    printf("Steps: %d/%d | Coins: %d/%d\n\n", 
           env->n_steps, MAX_STEPS,
           env->n_collected_coins, NUM_COINS);
    
    // Draw the grid
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            // Check if agent is here
            if (env->agent_x == x && env->agent_y == y) {
                printf("A ");  // Agent
            }
            // Check if any coin is here
            else {
                bool coin_here = false;
                for (int i = 0; i < NUM_COINS; i++) {
                    if (!env->coin_collected[i] && 
                        env->coin_x[i] == x && 
                        env->coin_y[i] == y) {
                        printf("$ ");  // Coin
                        coin_here = true;
                        break;
                    }
                }
                if (!coin_here) {
                    printf(". ");  // Empty space
                }
            }
        }
        printf("\n");
    }
    printf("\n");
}

void c_close(CoinFinder* env){
    // Nothing for now until we add raylib.
}

#endif