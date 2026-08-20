#pragma once


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>


#define ACT_NOOP  0
#define ACT_FLAP  1

#define FP_SCREEN_W     480
#define FP_SCREEN_H     720
#define FP_GROUND_H      80
#define FP_PLAYABLE_H   (FP_SCREEN_H - FP_GROUND_H)


#define FP_GRAVITY      0.44f
#define FP_FLAP_VEL    -9.5f
#define FP_PIPE_SPEED   3.0f
#define FP_MAX_VEL     12.0f

#define FP_PIPE_GAP     165
#define FP_PIPE_W        72
#define FP_PIPE_CAP_W    88
#define FP_PIPE_CAP_H    24
#define FP_PIPE_SPACING 280
#define FP_MAX_PIPES      5


#define FP_BIRD_X      115.0f
#define FP_BIRD_R       20.0f  
#define FP_BIRD_DRAW_SZ 52.0f 


#define FP_OBS_SIZE  8

#define FP_REWARD_ALIVE  0.01f  
#define FP_REWARD_PIPE   1.00f   
#define FP_REWARD_DEATH -1.00f   



typedef struct FPLog FPLog;
struct FPLog {
    float perf;           
    float score;          
    float episode_length; 
    float n;              
};

typedef struct FPPipe FPPipe;
struct FPPipe {
    float x;
    float gap_cy;   
    bool  passed;
};


typedef struct FPClient FPClient;

typedef struct FlappyPuffer FlappyPuffer;
struct FlappyPuffer {
    FPClient*    client;   

    int          num_agents;  
    float        bird_y;      
    float        bird_vel;     
    bool         alive;
    int          score;
    int          tick;

    FPPipe       pipes[FP_MAX_PIPES];
    float*       observations; 
    float*       rewards; 
    float*       terminals;
    float*       actions;

    FPLog        log;
    unsigned int rng;
};

void allocate_flappy (FlappyPuffer* env);
void free_flappy     (FlappyPuffer* env);
void c_reset_fp      (FlappyPuffer* env);
void c_step_fp       (FlappyPuffer* env);
int  c_render_fp     (FlappyPuffer* env);