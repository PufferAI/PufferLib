#include <stdlib.h>
#include<unistd.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "raylib.h"

#define TWO_PI 2.0*PI
#define MAX_SIZE 40

#define ATN_PASS 0
#define ATN_FORWARD 1
#define ATN_LEFT 2
#define ATN_RIGHT 3
#define ATN_BACK 4
#define ATN_DROP 5

#define DIR_WEST 0.0;
#define DIR_NORTH PI/2.0;
#define DIR_EAST PI;
#define DIR_SOUTH 3.0*PI/2.0;

#define MAX_MOBS 5

#define EMPTY 0
#define WALL 1
#define ZOMBIE 2
#define GOAL 3
#define REWARD 4
#define OBJECT 5
#define AGENT 6
#define BLOCK 14

// Maximum item number is 32 (from pufferlib/ocean/torch.py , one_hot in encode_observations)

#define LOG_BUFFER_SIZE 4096

typedef struct Log Log;
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
};

// 8 unique agents (we only use 1)
bool is_agent(int idx) {
    //return idx >= AGENT && idx < AGENT + 8;
    return idx == AGENT ;
}


/*int rand_color() {
    return AGENT + rand()%8;
}*/

// 6 unique keys and doors
/*bool is_key(int idx) {
    return idx >= KEY && idx < KEY + 6;
}
bool is_locked_door(int idx) {
    return idx >= DOOR_LOCKED && idx < DOOR_LOCKED + 6;
}
bool is_open_door(int idx) {
    return idx >= DOOR_OPEN && idx <= DOOR_OPEN + 6;
}
bool is_correct_key(int key, int door) {
    return key == door - 6;
}*/



// Every object or NPC (except for wall and
// empty space) is a "Mob". Even those that
// don't move.
typedef struct Mob Mob;
struct Mob{
    float y;
    float x;
    float prev_y;
    float prev_x;
    float spawn_y;
    float spawn_x;
    int color;  // determines the type of the mob
    int alive;
    int mobile;
};

typedef struct Agent Agent;
struct Agent {
    float y;
    float x;
    float prev_y;
    float prev_x;
    float spawn_y;
    float spawn_x;
    int color;
    float direction;
    Mob *held;
};


typedef struct Renderer Renderer;
typedef struct State State;

typedef struct Grixel Grixel;
struct Grixel{
    Renderer* renderer;
    State* levels;
    int num_maps;
    int width;
    int height;
    int num_agents;
    int horizon;
    int vision;
    int tick;
    float speed;
    int obs_diameter;
    int max_size;
    
    // the following values are set in the grixel.py init, passed
    // as params to vec_init in grixel.py init, and 
    // read into these fields by my_init in binding.c:
    int block_size;
    int pixelize;
    int texture_mode;
    int additional_obs_size;
    int nb_object_types;
    
    bool discretize;
    Log log;
    Agent* agents;
    Mob* mobs;
    char choice;
    unsigned char* grid;
    int* counts;
    // block_Textures and observations used to be 
    // unsigned char, but now they are char because 
    // they can have negative values
    char* block_textures;
    // The following arrays are created by the python class (as
    // Gym boxes / numpy arrays), and 
    // bound to the C struct's same-name variables by env_binding.h
    // (better hope that they all agree on the same size!)
    char* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    char* pockets; // if we ever decide to allow agents to hold more than one object at a time...
    char* is_mobile;
    char* is_pickable;
    // For logging, debugging
    float* cum_rewards;
    float* prev_rewards;
};


// init_grid allocates and initializes the memory required by the environment 
// (including a max size grid)
void init_grid(Grixel* env) {
    env->num_agents = 1; // should always be 1
    env->vision = 5;  // assumed to be 5 in other code
    env->speed = 1;
    env->discretize = true;
    // env->block_size and env->pixelize should be set before calling init_grid, like max_size and num_maps.
    env->obs_diameter = 2*env->vision + 1; // this has to be 11 anyway
    /*if (env->block_size != 5){
            printf("block size is not 5, but %d \n", env->block_size);
            puts("Error!");
            exit(1);
    }*/
    if (env->pixelize)
        env->obs_diameter *= env->block_size;
    int env_mem= env->max_size * env->max_size;
    env->grid = calloc(env_mem, sizeof(unsigned char));
    env->counts = calloc(env_mem, sizeof(int));
    env->agents = calloc(env->num_agents, sizeof(Agent));
    env->mobs = calloc(MAX_MOBS, sizeof(Mob));
    env->cum_rewards = calloc(env->num_agents, sizeof(float));
    env->prev_rewards = calloc(env->num_agents, sizeof(float));
    env->pockets = calloc(env->num_agents, sizeof(char) * env->nb_object_types);
    env->is_mobile = calloc(env->nb_object_types, sizeof(char));
    env->is_pickable = calloc(env->nb_object_types, sizeof(char));
    // Allocating and initializing textures
    // Originally, 32 possible objects
    // env->block_textures = calloc(32* env->block_size * env->block_size, sizeof(char));
    env->block_textures = calloc(env->nb_object_types * env->block_size * env->block_size, sizeof(char));
    fill_textures_randomize(env);

}

// Apparently the following is only used in the grixel.c test code
// but note it confirms observations as an unsigned char array...
// DO NOT USE, not updated
Grixel* allocate_grid(int max_size, int num_agents, int horizon,
        int vision, float speed, bool discretize) {
    Grixel* env = (Grixel*)calloc(1, sizeof(Grixel));
    env->max_size = max_size;
    env->num_agents = num_agents;
    env->horizon = horizon;
    env->vision = vision;
    env->speed = speed;
    env->discretize = discretize;
    int obs_diameter = 2*vision + 1;
    // Since we're in the grixel.c test code, the python Grixel class 
    // is not used and we must allocate these arrays ourselves
    env->observations = calloc(
        num_agents*(obs_diameter*obs_diameter+env->additional_obs_size), sizeof(unsigned char));
    env->actions = calloc(num_agents, sizeof(float));
    env->rewards = calloc(num_agents, sizeof(float));
    env->terminals = calloc(num_agents, sizeof(unsigned char));
    init_grid(env);
    return env;
}

void c_close(Grixel* env) {
    free(env->grid);
    free(env->counts);
    free(env->block_textures);
    free(env->agents);
    free(env->cum_rewards);
    free(env->prev_rewards);
    free(env->mobs);  
    free(env);
}

void free_allocated_grid(Grixel* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    c_close(env);
}

bool in_bounds(Grixel* env, int y, int c) {
    return (y >= 0 && y <= env->height
        && c >= 0 && c <= env->width);
}

int grid_offset(Grixel* env, int y, int x) {
    return y*env->max_size + x;
}

void add_log(Grixel* env, int idx) {
    env->log.perf += env->cum_rewards[idx];
    env->log.score += env->cum_rewards[idx];
    env->log.episode_return += env->cum_rewards[idx];
    env->log.episode_length += env->tick;
    env->log.n += 1.0;
}
 
void compute_observations(Grixel* env) {
    memset(env->observations, 0, (env->obs_diameter*env->obs_diameter+env->additional_obs_size)*env->num_agents);
    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        Agent* agent = &env->agents[agent_idx];
        float y = agent->y;
        float x = agent->x;
        int start_r = y - env->vision;
        if (start_r < 0) {
            start_r = 0;
        }

        int start_c = x - env->vision;
        if (start_c < 0) {
            start_c = 0;
        }

        int end_r = y + env->vision;
        if (end_r >= env->max_size) {
            end_r = env->max_size - 1;
        }

        int end_c = x + env->vision;
        if (end_c >= env->max_size) {
            end_c = env->max_size - 1;
        }

        int obs_offset = agent_idx*(env->obs_diameter*env->obs_diameter + env->additional_obs_size);

        if (env->pixelize == 0)
            for (int r = start_r; r <= end_r; r++) {
                for (int c = start_c; c <= end_c; c++) {
                    int r_idx = r - y + env->vision;
                    int c_idx = c - x + env->vision;
                    int obs_adr = obs_offset + r_idx*env->obs_diameter + c_idx;
                    int adr = grid_offset(env, r, c);
                    env->observations[obs_adr] = env->grid[adr];
                }
            }
        else
        {
            int cpt=0;
            int BS2 = env->block_size * env->block_size;
            for (int ii=-env->vision; ii <= env->vision; ii++)
                for (int jj=-env->vision; jj <= env->vision; jj++)
                {
                    cpt ++;
                    // source row and column
                    int r_s = y + ii; int c_s = x + jj;
                    if (r_s < 0) 
                        continue;
                    if (r_s >= env->max_size) 
                        continue;
                    if (c_s < 0) 
                        continue;
                    if (c_s >= env->max_size) 
                        continue;
                    int source_adr = grid_offset(env, r_s, c_s);
                    //for (int pos=0; pos<env->block_size*env->block_size; pos++)
                    //    env->observations[(cpt-1)*BS2*BS2 + pos]= env->block_textures[env->grid[source_adr]*BS2*BS2+pos];
                    // Hopefully the following does the exact same thing:
                    memcpy(env->observations+obs_offset+(cpt-1)*BS2,  env->block_textures+env->grid[source_adr]*BS2, BS2);
                }
        }
 
        // additional observations
        if (env->additional_obs_size> 0)
        {
            // agent sees the previous step's reward (need to test more if current step
            // reward is worse)
            env->observations[obs_offset+env->obs_diameter * env->obs_diameter] = (char)(10.0 * env->prev_rewards[agent_idx]);
            // Note that PufferLib silently clips rewards to -1,1 anyway
            if (((char)(env->rewards[agent_idx]) > 1) || ((char)(env->rewards[agent_idx])<-1))
            {
                printf("OOB value %d\n", (char)env->rewards[agent_idx]);
                puts("Exiting");
                exit(1);
            }
            env->observations[obs_offset+env->obs_diameter * env->obs_diameter +1] = (char)(env->terminals[agent_idx]);
            // for debugging
            //env->observations[obs_offset+env->obs_diameter * env->obs_diameter +2] = (char)(env->choice);
            memcpy(env->observations+obs_offset+env->obs_diameter * env->obs_diameter +3, 
                               env->pockets+agent_idx, env->nb_object_types);
            //env->observations[obs_offset+env->obs_diameter * env->obs_diameter +3] = (char)(env->pockets[agent_idx]);


        }
    }
}

void make_border(Grixel*env) {
    for (int r = 0; r < env->height; r++) {
        int adr = grid_offset(env, r, 0);
        env->grid[adr] = WALL;
        adr = grid_offset(env, r, env->width-1);
        env->grid[adr] = WALL;
    }
    for (int c = 0; c < env->width; c++) {
        int adr = grid_offset(env, 0, c);
        env->grid[adr] = WALL;
        adr = grid_offset(env, env->height-1, c);
        env->grid[adr] = WALL;
    }
}

void spawn_agent(Grixel* env, int idx, int x, int y) {
    Agent* agent = &env->agents[idx];
    int spawn_y = y;
    int spawn_x = x;
    assert(in_bounds(env, spawn_y, spawn_x));
    int adr = grid_offset(env, spawn_y, spawn_x);
    assert(env->grid[adr] == EMPTY);
    agent->spawn_y = spawn_y;
    agent->spawn_x = spawn_x;
    agent->y = agent->spawn_y;
    agent->x = agent->spawn_x;
    agent->prev_y = agent->y;
    agent->prev_x = agent->x;
    agent->color = AGENT; // why was that below?
    env->grid[adr] = agent->color;
    agent->direction = 0;
    agent->held = NULL;
}

void spawn_mob(Grixel* env, int idx, int x, int y, int color) {
    Mob* mob = &env->mobs[idx];
    int spawn_y = y;
    int spawn_x = x;
    assert(in_bounds(env, spawn_y, spawn_x));
    int adr = grid_offset(env, spawn_y, spawn_x);
    assert(env->grid[adr] == EMPTY);
    mob->spawn_y = spawn_y;
    mob->spawn_x = spawn_x;
    mob->y = mob->spawn_y;
    mob->x = mob->spawn_x;
    mob->prev_y = mob->y;
    mob->prev_x = mob->x;
    mob->alive = 1;  // spawning means you're alive
    mob->color = color;
    mob->mobile = env->is_mobile[mob->color];
    env->grid[adr] = mob->color;
}

struct State {
    int width;
    int height;
    int num_agents;
    Agent* agents;
    Mob* mobs;
    unsigned char* grid;
};

// init_state allocates the memory for a single map, with its agents
void init_state(State* state, int max_size, int num_agents) {
    state->agents = calloc(num_agents, sizeof(Agent));
    state->mobs = calloc(MAX_MOBS, sizeof(Mob));
    state->grid = calloc(max_size*max_size, sizeof(unsigned char));
}

void free_state(State* state) {
    free(state->agents);
    free(state->mobs);
    free(state->grid);
    free(state);
}

void get_state(Grixel* env, State* state) {
    state->width = env->width;
    state->height = env->height;
    state->num_agents = env->num_agents;
    // this seems to assume that (previous) state->num_agents >= env->num_agents ??
    memcpy(state->agents, env->agents, env->num_agents*sizeof(Agent));
    memcpy(state->mobs, env->mobs, MAX_MOBS *sizeof(Mob));
    memcpy(state->grid, env->grid, env->max_size*env->max_size);
}

// copies the map and agent data from state to env
void set_state(Grixel* env, State* state) {
    env->width = state->width;
    env->height = state->height;
    // env->horizon should be equal to bptt_horizon !
    env->horizon = 128; //2*env->width*env->height;
    env->num_agents = state->num_agents;
    memcpy(env->agents, state->agents, env->num_agents*sizeof(Agent));
    memcpy(env->mobs, state->mobs, MAX_MOBS * sizeof(Mob));
    memcpy(env->grid, state->grid, env->max_size*env->max_size);
}

void c_reset(Grixel* env) {
    memset(env->grid, 0, env->max_size*env->max_size);
    memset(env->counts, 0, env->max_size*env->max_size*sizeof(int));
    memset(env->cum_rewards, 0, env->num_agents*sizeof(float));
    // memset(env->prev_rewards, 0, env->num_agents*sizeof(float)); // not sure about this
    env->tick = 0;
    fill_textures_randomize(env);
    if (env->renderer != NULL)
        update_renderer_textures(env);
    int idx = rand() % env->num_maps;

    // This one actually initialize most of
    // the data (grid, agent(s), etc.) 
    // Note: levels include agents and mobs
    // (initialized to have one agent and
    // zero alive mobs, see
    // create_maze_level, which is used do
    // create levels in binding.c
    // make_shared)
    set_state(env, &env->levels[idx]);
    
    int width = env->width; int height = env->height;
    int adr, posx, posy;
    int cpt = 0;
    int putrewardfirst = rand() % 2;
    do{
        posx = width/2 + rand() % (width/2);
        //posx = 2 + rand() %  (width-2);
        posy = height/2 + rand() % (height/2);
        adr = grid_offset(env, posy, posx);
        cpt++;
        if (cpt>10000){
            puts("Infinite loop in positioning reward mob");
            exit(1);
        }

    }
    while (env->grid[adr] != EMPTY);
    spawn_mob(env, 0, posx, posy, putrewardfirst?REWARD:ZOMBIE);
    do{
        posx = width/2 + rand() % (width/2);
        //posx = 2 + rand() %  (width-2);
        posy = height/2 + rand() % (height/2);
        adr = grid_offset(env, posy, posx);
        // the joys of debugging in c (assert doesn't work, gdb sucks)
        cpt++;
        if (cpt>10000){
            puts("Infinite loop in positioning zombie mob");
            for (int numr=0; numr<height; numr++){
                for (int numc=0; numc<height; numc++)
                    printf("%d ", env->grid[grid_offset(env, numr, numc)]);
                puts(" ");
            }
            printf("%d %d %d %d\n", posx, posy, width/2, height/2);
            exit(1);
        }
    }
    while (env->grid[adr] != EMPTY);
    spawn_mob(env, 1, posx, posy, putrewardfirst?ZOMBIE:REWARD);
    do{
        posx = width/2 + rand() % (width/2);
        posy = height/2 + rand() % (height/2);
        //posx = 2 + rand() %  (width-5);
        //posy = 2 + rand() %  (height-5);
        adr = grid_offset(env, posy, posx);
        cpt++;
        if (cpt>10000){
            puts("Infinite loop in positioning zombie mob");
            for (int numr=0; numr<height; numr++){
                for (int numc=0; numc<height; numc++)
                    printf("%d ", env->grid[grid_offset(env, numr, numc)]);
                puts(" ");
            }
            printf("%d %d %d %d\n", posx, posy, width/2, height/2);
            exit(1);
        }
    }
    while (env->grid[adr] != EMPTY);
    spawn_mob(env, 2, posx, posy, OBJECT);

    compute_observations(env);
}

int move_to(Grixel* env, int agent_idx, float y, float x) {
    Agent* agent = &env->agents[agent_idx];
    if (!in_bounds(env, y, x)) {
        return 1;
    }

    int adr = grid_offset(env, round(y), round(x));
    int dest = env->grid[adr];
    if (dest == WALL) {
        return 1;
    } else if (dest == REWARD || dest == GOAL || dest == ZOMBIE) {
    
        // REMINDER: REWARDS ARE CLAMPED TO [-1;1] IN
        // PUFFERL.PY !

        if (dest == ZOMBIE)
            env->rewards[agent_idx] = -.7; //-1.0;
        else
            env->rewards[agent_idx] = 1.0; //1.0;
        if (env->renderer != NULL) // i.e. if we're in eval-mode, with display
            printf("Reward: %.2f\n", env->rewards[agent_idx]);
        env->cum_rewards[agent_idx] +=env->rewards[agent_idx]; 
        // Teleporting agent to spawn position
        /*x = agent->spawn_x;
        y = agent->spawn_y;
        adr = grid_offset(env, round(y), round(x));*/
        
        // Teleporting agent to random position:
        int cpt=0;
        do{
                x = 1 + rand() % (env->width-2);
                y = 1 + rand() % (env->height-2);
                adr = grid_offset(env, y, x);
                cpt++;
                if (cpt>10000){
                        puts("Infinite loop in teleporting agent");
                        for (int numr=0; numr<env->height; numr++){
                                for (int numc=0; numc<env->height; numc++)
                                        printf("%d ", env->grid[grid_offset(env, numr, numc)]);
                                puts(" ");
                        }
                        printf("%d %d %d %d\n", x, y, env->width/2, env->height/2);
                        exit(1);
                }
        }
        while (env->grid[adr] != EMPTY);


    }
    else if (env->grid[adr] != EMPTY) 
    {
        if ((!env->is_pickable[dest]) || (agent->held != NULL))
            return 1;
        else
        {
            // We're picking something
            int num_mob = 0; Mob * this_mob;
            // Unspawn mob
            for (num_mob=0; num_mob < MAX_MOBS; num_mob++)
            {
                this_mob = &env->mobs[num_mob];
                if (!this_mob->alive)
                    // Should *not* include not-alive mobs !
                    continue;
                if ((round(this_mob->x) == round(x)) && (round(this_mob->y) == round(y)))
                    break;
            }
            if (num_mob == MAX_MOBS)
            {
                puts("Error! Couldn't find mob to be picked.");
                printf("grid[adr]: %d\n", env->grid[adr]);
                printf("x,y (target): %f %f\n",x,y);
                printf("mob[2].x,y: %f %f\n", env->mobs[2].x, env->mobs[2].y);
                printf("num_mob: %d\n", num_mob);
                for (int nm=0; nm < MAX_MOBS; nm++)
                {
                    this_mob = &env->mobs[nm];
                    printf("mobs[%d]->x,y: %f %f\n", nm, this_mob->x, this_mob->y);
                    printf("mobs[%d].x,y: %f %f\n", nm, env->mobs[nm].x, env->mobs[nm].y);
                }
                exit(1);
            }
            if (env->renderer != NULL) // i.e. if we're in eval-mode, with display
                printf("Picking, type %d\n", this_mob->color);
            agent->held = this_mob;
            this_mob->alive = 0;
            env->grid[adr] = EMPTY;
            // We don't need to move the agent here, this is done below
        }
    }


    int start_y = round(agent->y);
    int start_x = round(agent->x);
    int start_adr = grid_offset(env, start_y, start_x);
    env->grid[start_adr] = EMPTY;

    env->grid[adr] = agent->color;
    agent->y = y;
    agent->x = x;
    return 0;
}
 
bool step_mob(Grixel* env, int idx) {
    Mob* mob = &env->mobs[idx];
    if (!env->is_mobile[mob->color])
    {
            puts("Error! Trying to step a non-mobile mob");
            exit(1);
    }
    mob->prev_y = mob->y;
    mob->prev_x = mob->x;
    
    float x = mob->x;
    float y = mob->y;
    float dx = -1.0 + rand() % 3;
    float dy = -1.0 + rand() % 3;
    float dest_x = x + dx;
    float dest_y = y + dy;
    if (dest_x < 2 && dest_y < 2) {
        return false;
    }
    if (!in_bounds(env, dest_y, dest_x)) {
        return false;
    }
    //int err = move_to(env, idx, dest_y, dest_x);
    int adr = grid_offset(env, round(dest_y), round(dest_x));
    if (env->grid[adr] != EMPTY) {
        return false;
    } 
    int start_y = round(mob->y);
    int start_x = round(mob->x);
    int start_adr = grid_offset(env, start_y, start_x);
    assert(env->grid[start_adr] == mob->color);
    env->grid[start_adr] = EMPTY;
    env->grid[adr] = mob->color;
    mob->y = dest_y;
    mob->x = dest_x;
    return true; // we successfully moved
}



bool step_agent(Grixel* env, int idx) {
    Agent* agent = &env->agents[idx];
    agent->prev_y = agent->y;
    agent->prev_x = agent->x;

    float atn = env->actions[idx];
    int iatn = (int)atn;
    float direction = agent->direction;

        if (iatn == ATN_PASS) {
            return true;
        } else if (iatn == ATN_DROP) {
            if (agent->held == NULL)
                return true;
            // else... nothing, just like for Forward! Actual dropping is dealt with below.

        } else if (iatn == ATN_FORWARD) {
        } else if (iatn == ATN_LEFT) {
            direction -= PI/2.0;
        } else if (iatn == ATN_RIGHT) {
            direction += PI/2.0;
        } else if (iatn == ATN_BACK) {
            direction += PI;
        } else {
            printf("Invalid action: %f\n", atn);
            exit(1);
        }
        if (direction < 0) {
            direction += TWO_PI;
        } else if (direction >= TWO_PI) {
            direction -= TWO_PI;
        }

    float x = agent->x;
    float y = agent->y;
    float dx = env->speed*cos(direction);
    float dy = env->speed*sin(direction);
    agent->direction = direction;
        float dest_x = x + dx;
        float dest_y = y + dy;
        if (!in_bounds(env, dest_y, dest_x)) {
            return false;
        }

        if (iatn == ATN_DROP)
        {
            // We're dropping
            // need to realive the held mob, and spawn it at dest_x,dest_y
            assert( agent->held != NULL);
            int adr = grid_offset(env, round(dest_y), round(dest_x)); 
            if (env->grid[adr] != EMPTY)
                // should check if killable (bumpable?), give approriate reward, unalive killed mob, delete grid[adr]
                return false;
            agent->held->x = dest_x;
            agent->held->y = dest_y;
            agent->held->alive = 1;
            env->grid[adr] = agent->held->color;
            agent->held = NULL;
            if (env->renderer != NULL) // i.e. if we're in eval-mode, with display
                printf("Dropping, type %d\n", env->grid[adr]);
            return true;
        }
        else{
            assert (iatn == ATN_FORWARD);
            int err = move_to(env, idx, dest_y, dest_x);
            if (err) {
                return false;
            }
        }

    int x_int = agent->x;
    int y_int = agent->y;
    int adr = grid_offset(env, y_int, x_int);
    env->counts[adr]++;
    //env->rewards[idx] += 0.01 / (float)env->counts[adr];
    //env->log.episode_return += 0.01 / (float)env->counts[adr];
    return true;
}

void c_step(Grixel* env) {
    memset(env->terminals, 0, env->num_agents);
    memset(env->rewards, 0, env->num_agents*sizeof(float));
    env->tick++;

    for (int i = 0; i < env->num_agents; i++) {
        step_agent(env, i);
    }
    for (int i = 0; i < MAX_MOBS; i++) {
        if (env->mobs[i].alive)
                if (env->mobs[i].mobile)
                    step_mob(env, i);
    }
    // Note: compute_observations occurs *after* step_agent,
    // as it should.
    // Rewards must be computed *before* compute_observations
    compute_observations(env);

    bool done = true;
    for (int i = 0; i < env->num_agents; i++) {
        if (!env->terminals[i]) {
            done = false;
            break;
        }
    }

    // prev_rewards are updated after everything
    memcpy(env->prev_rewards,  env->rewards, env->num_agents*sizeof(float));

    if (env->tick >= env->horizon) {
        done = true;
        //add_log(env, 0);
        for (int i=0; i < env->num_agents; i++){
            env->terminals[i] = 1;  // Truncations not fully implemented
            add_log(env, i);
        }
    }

    if (done) {
        c_reset(env);
        // This is already done in c_reset!
        //int idx = rand() % env->num_maps;
        //set_state(env, &env->levels[idx]);
        //compute_observations(env);
    }
}

// Raylib client
Color COLORS[] = {
    (Color){6, 24, 24, 255},
    (Color){0, 0, 255, 255},
    (Color){0, 128, 255, 255},
    (Color){128, 128, 128, 255},
    (Color){255, 0, 0, 255},
    (Color){255, 255, 255, 255},
    (Color){255, 85, 85, 255},
    (Color){170, 170, 170, 255},
    (Color){0, 255, 255, 255},
    (Color){255, 255, 0, 255},
};

Rectangle UV_COORDS[7] = {
    (Rectangle){0, 0, 0, 0},
    (Rectangle){512, 0, 128, 128},
    (Rectangle){0, 0, 0, 0},
    (Rectangle){0, 0, 128, 128},
    (Rectangle){128, 0, 128, 128},
    (Rectangle){256, 0, 128, 128},
    (Rectangle){384, 0, 128, 128},
};

struct Renderer {
    int cell_size;
    int nb_object_types;
    int width;
    int height;
    Texture2D puffer;
    Texture2D *renderer_textures; // different from env->block_textures (i.e. larger pixel size)
    float* overlay;
};

//Renderer* init_renderer(int cell_size, int width, int height) {
Renderer* init_renderer(Grixel *env,  int width, int height) {
    Renderer* renderer = (Renderer*)calloc(1, sizeof(Renderer));
    renderer->width = width;
    renderer->height = height;
    renderer->nb_object_types = env->nb_object_types;

    int pixel_size = 3; // how big each individual pixel will be shown on screen
    int cell_size = pixel_size * env->block_size;
    renderer->cell_size = cell_size;

    renderer->overlay = (float*)calloc(width*height, sizeof(float));

    InitWindow(width*cell_size, height*cell_size, "PufferLib Grixel");
    SetTargetFPS(60);

    renderer->puffer = LoadTexture("resources/shared/puffers_128.png");

    renderer->renderer_textures = calloc(env->nb_object_types , sizeof(Texture2D));

    for (int numt=0; numt<env->nb_object_types; numt++){
        //Image image = GenImageColor(env->blocksize, env->blocksize, BLACK); 
        Image image = GenImageColor(cell_size, cell_size, BLACK); 
        for (int numc=0; numc< env->block_size; numc ++)
            for (int numr=0; numr< env->block_size; numr ++)
                if (env->block_textures[numt*env->block_size*env->block_size + numr*env->block_size + numc] >0)
                    for (int p1=0; p1 < pixel_size; p1++)
                        for (int p2=0; p2 < pixel_size; p2++)
                            ImageDrawPixel(&image, numc*pixel_size+p1, numr*pixel_size+p2, WHITE);
                else
                    for (int p1=0; p1 < pixel_size; p1++)
                        for (int p2=0; p2 < pixel_size; p2++)
                            ImageDrawPixel(&image, numc*pixel_size+p1, numr*pixel_size+p2, BLACK); //BLUE);
        renderer->renderer_textures[numt] = LoadTextureFromImage(image);
        UnloadImage(image);
    }

    return renderer;
}

void clear_overlay(Renderer* renderer) {
    memset(renderer->overlay, 0, renderer->width*renderer->height*sizeof(float));
}

void close_renderer(Renderer* renderer) {
    CloseWindow();
    free(renderer->overlay);
    UnloadTexture(renderer->puffer); // let's see what happens
    for (int numt=0; numt<renderer->nb_object_types; numt++)
        UnloadTexture(renderer->renderer_textures[numt]);
    // no free(renderer->puffer) ? or UnloadTexture?
    free(renderer);

}

void c_render(Grixel* env) {
    // TODO: fractional rendering
    float frac = 0.0;
    float overlay = 0.0;
    if (env->renderer == NULL) {
        //env->renderer = init_renderer(16, env->max_size, env->max_size); // Renderer* init_renderer(int cell_size, int width, int height) 
        env->renderer = init_renderer(env, env->max_size, env->max_size); // Renderer should compute its own cell size
    }
    Renderer* renderer = env->renderer;
 
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    Agent* agent = &env->agents[0];
    int r = agent->y;
    int c = agent->x;
    int adr = grid_offset(env, r, c);
    //renderer->overlay[adr] = overlay;
    //renderer->overlay[adr] -= 0.1;
    //renderer->overlay[adr] = -1 + 1.0/(float)env->counts[adr];

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});

    int ts = renderer->cell_size;
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++){
            adr = grid_offset(env, r, c);
            int tile = env->grid[adr];
            if (tile == EMPTY) {
                continue;
                // So, the rest of this block is ignored?...
                overlay = renderer->overlay[adr];
                if (overlay == 0) {
                    continue;
                }
                Color color;
                if (overlay < 0) {
                    overlay = -fmaxf(-1.0, overlay);
                    color = (Color){255.0*overlay, 0, 0, 255};
                } else {
                    overlay = fminf(1.0, overlay);
                    color = (Color){0, 255.0*overlay, 0, 255};
                }
                DrawRectangle(c*ts, r*ts, ts, ts, color);
            }

            /*
            Color color;
            if (tile == WALL) {
                color = (Color){128, 128, 128, 255};
            } else if (tile == GOAL) {
                color = GREEN;
            } else if (is_locked_door(tile)) {
                int weight = 40*(tile - DOOR_LOCKED);
                color = (Color){weight, 0, 0, 255};
            } else if (is_open_door(tile)) {
                int weight = 40*(tile - DOOR_OPEN);
                color = (Color){0, weight, 0, 255};
            } else if (is_key(tile)) {
                int weight = 40*(tile - KEY);
                color = (Color){0, 0, weight, 255};
            } else {
                continue;
            }
            DrawRectangle(c*ts, r*ts, ts, ts, color);
            */
            if (tile != AGENT) // Agent is fish!
                DrawTexture(renderer->renderer_textures[tile], c*ts, r*ts, WHITE);
       }
    }

    for (int i = 0; i < env->num_agents; i++) {
        agent = &env->agents[0];
        float y = agent->y + (frac - 1)*(agent->y - agent->prev_y);
        float x = agent->x + (frac - 1)*(agent->x - agent->prev_x);
        int u = 0;
        int v = 0;
        Rectangle source_rect = (Rectangle){u, v, 128, 128};
        Rectangle dest_rect = (Rectangle){x*ts, y*ts, ts, ts};
        DrawTexturePro(renderer->puffer, source_rect, dest_rect,
            (Vector2){0, 0}, 0, WHITE);
    }
 
    EndDrawing();
}


void generate_growing_tree_maze(unsigned char* grid,
        int width, int height, int max_size, float difficulty, int seed) {
    srand(seed);
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, 1, 0, -1};
    int dirs[4] = {0, 1, 2, 3};
    int cells[2*width*height];
    int num_cells = 1;

    bool visited[width*height];
    memset(visited, false, width*height);

    memset(grid, WALL, max_size*height);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int adr = r*max_size + c;
            if (r % 2 == 1 && c % 2 == 1) {
                grid[adr] = EMPTY;
            }
        }
    }

    int x_init = rand() % (width - 1);
    int y_init = rand() % (height - 1);

    if (x_init % 2 == 0) {
        x_init++;
    }
    if (y_init % 2 == 0) {
        y_init++;
    }

    int adr = y_init*height + x_init;
    visited[adr] = true;
    cells[0] = x_init;
    cells[1] = y_init;

    //int cell = 32;
    //InitWindow(width*cell, height*cell, "PufferLib Ray Grixel");
    //SetTargetFPS(60);

    while (num_cells > 0) {
        if (rand() % 1000 > 1000*difficulty) {
            int i = rand() % num_cells;
            int tmp_x = cells[2*num_cells - 2];
            int tmp_y = cells[2*num_cells - 1];
            cells[2*num_cells - 2] = cells[2*i];
            cells[2*num_cells - 1] = cells[2*i + 1];
            cells[2*i] = tmp_x;
            cells[2*i + 1] = tmp_y;
 
        }

        int x = cells[2*num_cells - 2];
        int y = cells[2*num_cells - 1];
 
        int nx, ny;

        // In-place direction shuffle
        for (int i = 0; i < 4; i++) {
            int ii = i + rand() % (4 - i);
            int tmp = dirs[i];
            dirs[i] = dirs[ii];
            dirs[ii] = tmp;
        }

        bool made_path = false;
        for (int dir_i = 0; dir_i < 4; dir_i++) {
            int dir = dirs[dir_i];
            nx = x + 2*dx[dir];
            ny = y + 2*dy[dir];
           
            if (nx <= 0 || nx >= width-1 || ny <= 0 || ny >= height-1) {
                continue;
            }

            int visit_adr = ny*width + nx;
            if (visited[visit_adr]) {
                continue;
            }

            visited[visit_adr] = true;
            cells[2*num_cells] = nx;
            cells[2*num_cells + 1] = ny;

            nx = x + dx[dir];
            ny = y + dy[dir];

            int adr = ny*max_size + nx;
            grid[adr] = EMPTY;
            num_cells++;

            made_path = true;

            /*
            if (IsKeyPressed(KEY_ESCAPE)) {
                exit(0);
            }
            BeginDrawing();
            ClearBackground((Color){6, 24, 24, 255});
            Color color = (Color){128, 128, 128, 255};
            for (int r = 0; r < height; r++) {
                for (int c = 0; c < width; c++){
                    int adr = r*max_size + c;
                    int tile = grid[adr];
                    if (tile == WALL) {
                        DrawRectangle(c*cell, r*cell, cell, cell, color);
                    }
               }
            }
            EndDrawing();
            */

            break;
        }
        if (!made_path) {
            num_cells--;
        }
    }
}

// Map creation / filling.
// This is called by my_shared in binding.c, which generates
// the shared set of pre-generated maps to be used by all
// environments
void create_maze_level(Grixel* env, int width, int height, float difficulty, int seed) {
    env->width = width;
    env->height = height;
    memset(env->grid, EMPTY, env->max_size*env->max_size);
    generate_growing_tree_maze(env->grid, width, height, env->max_size, difficulty, seed);
    int posx, posy, adr;
    for (int myc=0; myc < width; myc++)
        for (int myr=0; myr < height; myr++)
            if (rand() % 2 == 0) {
                adr = grid_offset(env, myr, myc);
                env->grid[adr] = EMPTY;
            }
    make_border(env);
    for (int m=0; m < MAX_MOBS; m++){
        env->mobs[m].alive=0;
    }
    spawn_agent(env, 0, 1, 1);
    // We spawn the agent, but we do not spawn
    // the mobs - we do that at c_reset, rather
    // than here at map creation (so the mob
    // positions are randomized at each reset)

    //int goal_adr = grid_offset(env, env->height - 2, env->width - 2);
    //env->grid[goal_adr] = GOAL;
}


void fill_textures_randomize(Grixel* env)
{
    int choice; 
    int cpt=0;
    choice = 1 - rand() % 2;
    //choice = 0;
    //env->choice = choice;
    if (env->renderer != NULL){  // are we in the eval / visualized env?
        printf("Texture assignment - Choice is %d \n", choice);  
        /*if (choice == 0)     
            printf("reward is cross, zombie is checkers\n");
        else
            printf("reward is checkers, zombie is cross\n");*/
    }

    
    //for (int numt=0; numt<32; numt++)
    for (int numt=0; numt<env->nb_object_types; numt++)
        for (int numr=0; numr < env->block_size; numr++)
            for (int numc=0; numc < env->block_size; numc++)
            {
                int value = 0;
                if (numt==0)
                    value=0;
                else if (numt==1){
                    value = 1;
                    /*if (choice == 0)
                       if (numc==3)
                            if (numr==3)
                                value=0;*/
                }
                else if (is_agent(numt))
                    value = (numc==numr);  // the agent perceives itself as a diagonal slash 
                else if (numt == GOAL || numt == REWARD) {
                    if (env->texture_mode == 0)
                            value = (numr+numc+1) % 2;
                    else if (env->texture_mode == 1){
                      if (choice == 0)
                        value = (numr == env->block_size/2 || numc == env->block_size/2) ? 1: 0;
                    else
                        value = (numr+numc+1) % 2;
                    }
                    else
                            value = rand() % 2; 
                    //value = (numr+numc+1) % 2;
                }
                else if (numt == ZOMBIE) {
                    if (env->texture_mode == 0)
                            value = (numr == env->block_size/ 2 || numc == env->block_size/2) ? 1: 0;
                    else if (env->texture_mode == 1){
                      if (choice == 1)
                        value = (numr == env->block_size/2 || numc == env->block_size/2) ? 1: 0;
                    else
                        value = (numr+numc+1) % 2;
                    }
                    else
                            value = rand() % 2; 
                }
                else 
                    value = rand() % 2;
                env->block_textures[cpt++] = (char) value;
            }
    if (env->renderer != NULL){  // are we in the eval / visualized env?
        puts("Texture assignment done");
        /*if (choice == 0)     
            printf("reward is cross, zombie is checkers\n");
        else
            printf("reward is checkers, zombie is cross\n");*/
    }
}

void update_renderer_textures(Grixel *env){

    Renderer *renderer = env->renderer;
    int cell_size = env->renderer->cell_size;
    int pixel_size = cell_size / env->block_size; // how big each individual pixel will be shown on screen
    for (int numt=0; numt<env->nb_object_types; numt++){
        if (numt != 0 &&  numt != 1 && numt != REWARD && numt != GOAL && numt != ZOMBIE && !(is_agent(numt)))
            continue;
        //Image image = GenImageColor(env->blocksize, env->blocksize, BLACK); 
        Image image = GenImageColor(cell_size, cell_size, BLACK); 
        for (int numc=0; numc< env->block_size; numc ++)
            for (int numr=0; numr< env->block_size; numr ++)
                if (env->block_textures[numt*env->block_size*env->block_size + numr*env->block_size + numc] >0)
                    for (int p1=0; p1 < pixel_size; p1++)
                        for (int p2=0; p2 < pixel_size; p2++)
                            ImageDrawPixel(&image, numc*pixel_size+p1, numr*pixel_size+p2, WHITE);
                else
                    for (int p1=0; p1 < pixel_size; p1++)
                        for (int p2=0; p2 < pixel_size; p2++)
                            ImageDrawPixel(&image, numc*pixel_size+p1, numr*pixel_size+p2, BLACK); //BLUE);
        renderer->renderer_textures[numt] = LoadTextureFromImage(image);
        UnloadImage(image);
    }
}
