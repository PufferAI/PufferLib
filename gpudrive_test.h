#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#define SELF_OBS 7
#define OTHER_OBS 11
#define MAP_OBS 3000

static int new_map_obs[MAP_OBS];

typedef struct test_struct test_struct;
struct test_struct {
    int* observations;
    int* agent_states;
    int num_agents;
    int active_agents;
};

void print_obs(test_struct* env, int agent_idx) {
    printf("agent %d obs: ", agent_idx);
    int obs_size_per_agent = SELF_OBS + (env->active_agents-1) * (OTHER_OBS);
    for (int i = 0; i < obs_size_per_agent; i++) {
        printf("%d ", env->observations[agent_idx*obs_size_per_agent + i]);
    }
    printf("\n");
}

void print_map_obs(test_struct* env) {
    printf("map obs: ");
    int obs_size_per_agent = SELF_OBS + (env->active_agents-1) * (OTHER_OBS);
    for (int i = 0; i < (MAP_OBS); i++) {
        printf("%d ", env->observations[env->active_agents*obs_size_per_agent + i]);
    }
    printf("\n");
}

void add_obs(test_struct* env) {
    int obs_idx = 0;
    int obs_size_per_agent = SELF_OBS + (env->active_agents-1) * (OTHER_OBS);
    for (int i = 0; i < env->active_agents * obs_size_per_agent; i++) {
        if (i % obs_size_per_agent == 0 && i != 0) {
            obs_idx++;
        }
        env->observations[i] = obs_idx;
    }
    for (int i = 0; i < (MAP_OBS); i++) {
        env->observations[env->active_agents*obs_size_per_agent + i] = rand() % 100;
    }
}

void set_agents(test_struct* env) {
    int obs_size_per_agent = SELF_OBS + (env->active_agents-1) * (OTHER_OBS);
    for(int i=0;i<env->num_agents * (obs_size_per_agent);i++) {
        env->agent_states[i] = rand() % 100;
    }
}

void init(test_struct* env) {
    int obs_size_per_agent = SELF_OBS + (env->active_agents-1) * (OTHER_OBS);
    env->agent_states = (int*)calloc(env->num_agents * obs_size_per_agent, sizeof(int));
    env->observations = (int*)calloc(env->active_agents*obs_size_per_agent + MAP_OBS, sizeof(int));
    add_obs(env);
}

void free_initialized(test_struct* env) {
    free(env->observations);
    free(env->agent_states);
}

void compute_observations(test_struct* env, int* rand_agents){
    int obs_size_per_agent = SELF_OBS + (env->active_agents-1) * (OTHER_OBS);
    for(int i=0;i<env->active_agents;i++) {
        // printf("selected agents %d\n", rand_agents[i]);
        // printf("old obs\n");
        // print_obs(env, i);
        memcpy(env->observations + i*obs_size_per_agent, env->agent_states + rand_agents[i]*obs_size_per_agent, obs_size_per_agent*sizeof(int));
        // printf("new obs\n");
        // print_obs(env, i);
	memcpy(env->observations + env->active_agents*obs_size_per_agent, new_map_obs, (MAP_OBS)*sizeof(int));

    }
      
    
}
void change_world(test_struct* env, int* rand_agents){
    int obs_size_per_agent = SELF_OBS + (env->active_agents-1) * (OTHER_OBS);
    for(int i=0;i<env->active_agents;i++) {
        rand_agents[i] = rand() % env->num_agents;
        for(int j = 0;j<SELF_OBS;j++) {
            env->agent_states[rand_agents[i]*obs_size_per_agent + j] = 5;
        }
        for(int j = 0;j<OTHER_OBS*(env->active_agents-1);j++) {
            env->agent_states[rand_agents[i]*obs_size_per_agent + SELF_OBS + j] = 2;
        }
	
    }
    /*for (int j=0;j<MAP_OBS;j++) {
            new_map_obs[j] = 2;
    	}
	*/
}

void copy_world(test_struct* env, int* rand_agents){
	int obs_size_per_agent = SELF_OBS + (env->active_agents-1) * (OTHER_OBS);
	int* blah = (int*)calloc(obs_size_per_agent, sizeof(int));
	//int* randoworld = (int*)calloc(MAP_OBS, sizeof(int));
	for(int i =0;i<env->active_agents;i++){
		rand_agents[i] = rand() % env->num_agents;
		memcpy(env->agent_states+rand_agents[i]*obs_size_per_agent, blah, obs_size_per_agent*sizeof(int));
	}
	//memcpy(new_map_obs, randoworld, MAP_OBS*sizeof(int));
	free(blah);
	//free(randoworld);
}
void step(test_struct* env) {
    int rand_agents[env->active_agents];
    // do something
    copy_world(env, rand_agents);
    //change_world(env,rand_agents);
    // compute new obs  
    compute_observations(env, rand_agents); 
        // print_obs(env, rand_agents[0]);
    // print_obs(env, rand_agents[1]);
    // print_map_obs(env);
}
