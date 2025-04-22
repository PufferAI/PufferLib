#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "raylib.h"

#define EMPTY 0
#define TRASH 1
#define TRASH_BIN 2
#define AGENT 3

#define ACTION_UP 0
#define ACTION_DOWN 1
#define ACTION_LEFT 2
#define ACTION_RIGHT 3

#define LOG_BUFFER_SIZE 1024

typedef struct Log {
    float episode_return;
    float episode_length;
    float trash_collected;
    float score;
} Log;

typedef struct LogBuffer {
    Log* logs;
    int length;
    int idx;
} LogBuffer;

// LogBuffer functions
LogBuffer* allocate_logbuffer(int size) {
    LogBuffer* logs = (LogBuffer*)calloc(1, sizeof(LogBuffer));
    logs->logs = (Log*)calloc(size, sizeof(Log));
    logs->length = size;
    logs->idx = 0;
    return logs;
}

void free_logbuffer(LogBuffer* buffer) {
    free(buffer->logs);
    free(buffer);
}

Log aggregate_and_clear(LogBuffer* logs) {
    Log log = {0};
    if (logs->idx == 0) {
        return log;
    }
    for (int i = 0; i < logs->idx; i++) {
        log.episode_return += logs->logs[i].episode_return;
        log.episode_length += logs->logs[i].episode_length;
        log.trash_collected += logs->logs[i].trash_collected;
        log.score += logs->logs[i].score;
    }
    log.episode_return /= logs->idx;
    log.episode_length /= logs->idx;
    log.trash_collected /= logs->idx;
    log.score /= logs->idx;
    logs->idx = 0;
    return log;
}

void add_log(LogBuffer* logs, Log* log) {
    if (logs->idx == logs->length) {
        return;
    }
    logs->logs[logs->idx] = *log;
    logs->idx += 1;
}

typedef struct {
    int type; // Entity type: EMPTY, TRASH, TRASH_BIN, AGENT
    int pos_x;
    int pos_y;
    bool presence; // Whether or not Entity is present (not applicable to all types)
    bool carrying; // Whether agent is carrying trash (only applicable to Agent types)
} Entity;

typedef struct {
    Entity* entity;
    int index; // Index in the positions array (-1 if not applicable)
} GridCell;

typedef struct {
    // Interface for PufferLib
    char* observations;
    int* actions;
    float* rewards;
    unsigned char* dones;
    LogBuffer* log_buffer;

    int grid_size;
    int num_agents;
    int num_trash;
    int num_bins;
    int max_steps;
    int current_step;

    int total_num_obs;

    int agent_sight_range;

    float positive_reward;
    float negative_reward;
    float total_episode_reward;

    GridCell* grid; // 1D array for grid
    Entity* entities; // Indicies (0 - num_agents) for agents, (num_agents - num_bins) for bins, (num_bins - num_trash) for trash.

    bool do_human_control;
} CTrashPickupEnv;

int get_grid_index(CTrashPickupEnv* env, int x, int y) {
    return (y * env->grid_size) + x;
}

// returns the start index of each type of entity for iteration purposes
int get_entity_type_start_index(CTrashPickupEnv* env, int type)
{
    if (type == AGENT)
        return 0;
    else if (type == TRASH_BIN)
        return env->num_agents;
    else if (type == TRASH)
        return env->num_agents + env->num_bins;
    else
        return -1;
}

// Entity Attribute Based Obs-Space
/*
void compute_observations(CTrashPickupEnv* env) {
    float* obs = env->observations;
    float norm_factor = 1.0f / env->grid_size;

    int obs_index = 0;

    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++){
        float current_norm_pos_x = (float) (env->entities[agent_idx].pos_x) * norm_factor;
        float current_norm_pos_y = (float) (env->entities[agent_idx].pos_y) * norm_factor;

        // Add the observing agent's own position and carrying status
        obs[obs_index++] = current_norm_pos_x;
        obs[obs_index++] = current_norm_pos_y;
        obs[obs_index++] = env->entities[agent_idx].carrying ? 1.0f : 0.0f;

        // Add other observations from other entities (other agents, bins, trash)
        for (int i = 0; i < env->num_agents + env->num_bins + env->num_trash; i++) {
            // skip if current this agent
            if (agent_idx == i)
                continue;

            obs[obs_index++] = ((float) (env->entities[i].pos_x) * norm_factor) - current_norm_pos_x;
            obs[obs_index++] = ((float) (env->entities[i].pos_y) * norm_factor) - current_norm_pos_y;

            if (env->entities[i].type == AGENT) {
                obs[obs_index++] = env->entities[i].carrying ? 1.0f : 0.0f;
            }
            else if (env->entities[i].type == TRASH_BIN) {
                obs[obs_index++] = env->entities[i].presence ? 1.0f : 0.0f;
            }
        }
    }
}
*/

// Local crop version
void compute_observations(CTrashPickupEnv* env) {
    int sight_range = env->agent_sight_range;
    char* obs = env->observations;

    int obs_dim = 2*env->agent_sight_range + 1;
    int channel_offset = obs_dim*obs_dim;
    memset(obs, 0, env->total_num_obs*sizeof(char));

    for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
        // Add obs for whether the agent is carrying or not
        //obs[obs_index++] = env->entities[agent_idx].carrying;

        // Get the agent's position
        int agent_x = env->entities[agent_idx].pos_x;
        int agent_y = env->entities[agent_idx].pos_y;

        // Iterate over the sight range
        for (int dy = -sight_range; dy <= sight_range; dy++) {
            for (int dx = -sight_range; dx <= sight_range; dx++) {
                int cell_x = agent_x + dx;
                int cell_y = agent_y + dy;
                int obs_x = dx + env->agent_sight_range;
                int obs_y = dy + env->agent_sight_range;

                // Check if the cell is within bounds
                if (cell_x < 0 || cell_x >= env->grid_size || cell_y < 0 || cell_y >= env->grid_size) {
                    continue;
                }

                Entity* thisEntity = env->grid[get_grid_index(env, cell_x, cell_y)].entity;
                if (!thisEntity) {
                    continue;
                }

                int offset = agent_idx*5*channel_offset + obs_y*obs_dim + obs_x;
                int obs_idx = offset + thisEntity->type*channel_offset;
                obs[obs_idx] = 1;
                obs_idx = offset + 4*channel_offset;
                obs[obs_idx] = (float)thisEntity->carrying;
            }
        }
    }
}

// Helper functions
void place_random_entities(CTrashPickupEnv* env, int count, int item_type, int gridIndexStart) {
    int placed = 0;
    while (placed < count) 
    {
        int x = rand() % env->grid_size;
        int y = rand() % env->grid_size;

        GridCell* gridCell = &env->grid[get_grid_index(env, x, y)];

        if (gridCell->entity != NULL)
            continue;

        // Allocate and initialize a new Entity
        Entity* newEntity = &env->entities[gridIndexStart];
        newEntity->type = item_type;
        newEntity->pos_x = x;
        newEntity->pos_y = y;
        newEntity->presence = true;
        newEntity->carrying = false;

        gridCell->index = gridIndexStart;
        gridCell->entity = newEntity;

        gridIndexStart++;
        placed++;
    }
}

void add_reward(CTrashPickupEnv* env, int agent_idx, float reward){
    env->rewards[agent_idx] += reward;
    env->total_episode_reward += reward;
}

void move_agent(CTrashPickupEnv* env, int agent_idx, int action) {
    Entity* thisAgent = &env->entities[agent_idx];

    int move_dir_x = 0;
    int move_dir_y = 0;
    if (action == ACTION_UP) move_dir_y = -1;
    else if (action == ACTION_DOWN) move_dir_y = 1;
    else if (action == ACTION_LEFT) move_dir_x = -1;
    else if (action == ACTION_RIGHT) move_dir_x = 1;
    else printf("Undefined action: %d", action);
    
    int new_x = thisAgent->pos_x + move_dir_x;
    int new_y = thisAgent->pos_y + move_dir_y;

    if (new_x < 0 || new_x >= env->grid_size || new_y < 0 || new_y >= env->grid_size)
        return;

    GridCell* currentGridCell = &env->grid[get_grid_index(env, thisAgent->pos_x, thisAgent->pos_y)];
    GridCell* newGridCell = &env->grid[get_grid_index(env, new_x, new_y)];
    int cell_state_type = newGridCell->entity ? newGridCell->entity->type : EMPTY;

    if (cell_state_type == EMPTY) 
    {
        thisAgent->pos_x = new_x;
        thisAgent->pos_y = new_y;

        newGridCell->entity = currentGridCell->entity;
        newGridCell->index = agent_idx;

        currentGridCell->index = -1;
        currentGridCell->entity = NULL;
    } 
    else if (cell_state_type == TRASH && thisAgent->carrying == false) 
    {
        Entity* thisTrash = &env->entities[newGridCell->index];
        thisTrash->presence = false; // Mark as not present
        thisTrash->pos_x = -1;
        thisTrash->pos_y = -1;

        thisAgent->pos_x = new_x;
        thisAgent->pos_y = new_y;
        thisAgent->carrying = true;

        newGridCell->entity = currentGridCell->entity;
        newGridCell->index = currentGridCell->index;

        currentGridCell->entity = NULL;
        currentGridCell->index = -1;

        add_reward(env, agent_idx, env->positive_reward);
    } 
    else if (cell_state_type == TRASH_BIN) 
    {
        if (thisAgent->carrying)
        {
            // Deposit trash into bin
            thisAgent->carrying = false;
            add_reward(env, agent_idx, env->positive_reward);
        }
        else
        {
            int new_bin_x = new_x + move_dir_x;
            int new_bin_y = new_y + move_dir_y;

            if (new_bin_x < 0 || new_bin_x >= env->grid_size || new_bin_y < 0 || new_bin_y >= env->grid_size)
                return;

            GridCell* newGridCellForBin = &env->grid[get_grid_index(env, new_bin_x, new_bin_y)];
            if (newGridCellForBin->entity == NULL) {
                // Move the bin
                Entity* thisBin = newGridCell->entity;
                thisBin->pos_x = new_bin_x;
                thisBin->pos_y = new_bin_y;

                // Move the agent
                thisAgent->pos_x = new_x;
                thisAgent->pos_y = new_y;

                newGridCellForBin->entity = newGridCell->entity;
                newGridCellForBin->index = newGridCell->index;

                newGridCell->entity = currentGridCell->entity;
                newGridCell->index = currentGridCell->index;

                currentGridCell->entity = NULL;
                currentGridCell->index = -1;
            }
            // else don't move the agent
        }
    }
}

bool is_episode_over(CTrashPickupEnv* env) {
    for (int i = 0; i < env->num_agents; i++) 
    {
        if (env->entities[i].carrying) 
            return false;
    }

    int start_index = get_entity_type_start_index(env, TRASH);
    for (int i = start_index; i < start_index + env->num_trash; i++) 
    {
        if (env->entities[i].presence)
            return false;
    }

    return true;
}

void c_reset(CTrashPickupEnv* env) {
    env->current_step = 0;
    env->total_episode_reward = 0;

    for (int i = 0; i < env->grid_size * env->grid_size; i++) 
    {
        env->grid[i].entity = NULL;
        env->grid[i].index = -1;
    }

    // Place trash, bins, and agents randomly across the grid.
    place_random_entities(env, env->num_agents, AGENT, 0);
    place_random_entities(env, env->num_bins, TRASH_BIN, get_entity_type_start_index(env, TRASH_BIN));
    place_random_entities(env, env->num_trash, TRASH, get_entity_type_start_index(env, TRASH));

    compute_observations(env);
}

// Environment functions
void initialize_env(CTrashPickupEnv* env) {
    env->current_step = 0;

    env->positive_reward = 0.5f; // / env->num_trash;
    env->negative_reward = -0.0f; // / (env->max_steps * env->num_agents);

    env->grid = (GridCell*)calloc(env->grid_size * env->grid_size, sizeof(GridCell));
    env->entities = (Entity*)calloc(env->num_agents + env->num_bins + env->num_trash, sizeof(Entity));
    env->total_num_obs = env->num_agents * ((((env->agent_sight_range * 2 + 1) * (env->agent_sight_range * 2 + 1)) * 5));

    c_reset(env);
}

void allocate(CTrashPickupEnv* env) {

    env->observations = (char*)calloc(env->total_num_obs, sizeof(char));
    env->actions = (int*)calloc(env->num_agents, sizeof(int));
    env->rewards = (float*)calloc(env->num_agents, sizeof(float));
    env->dones = (unsigned char*)calloc(env->num_agents, sizeof(unsigned char));
    env->log_buffer = allocate_logbuffer(LOG_BUFFER_SIZE);

    initialize_env(env);
}

void c_step(CTrashPickupEnv* env) {
    // Reset reward for each agent
    memset(env->rewards, 0, sizeof(float) * env->num_agents);
    memset(env->dones, 0, sizeof(unsigned char) * env->num_agents);

    for (int i = 0; i < env->num_agents; i++) {
        move_agent(env, i, env->actions[i]);
        add_reward(env, i, env->negative_reward); // small negative reward to encourage efficiency
    }

    env->current_step++;
    if (env->current_step >= env->max_steps || is_episode_over(env)) 
    {
        memset(env->dones, 1, sizeof(unsigned char) * env->num_agents);

        Log log = {0};

        log.episode_length = env->current_step;
        log.episode_return = env->total_episode_reward;

        int total_trash_not_collected = 0;
        for (int i = env->num_agents + 1; i < env->num_agents + env->num_trash; i++) 
        {
            total_trash_not_collected += env->entities[i].presence;
        }

        log.trash_collected = (float) (env->num_trash - total_trash_not_collected);
        log.score = log.trash_collected - 0.1*log.episode_length;
        add_log(env->log_buffer, &log);

        c_reset(env);
    }

    compute_observations(env);
}

void free_initialized(CTrashPickupEnv* env) {
    free(env->grid);
    free(env->entities);
}

void free_allocated(CTrashPickupEnv* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->dones);
    free_logbuffer(env->log_buffer);
    free_initialized(env);
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_LINES = (Color){50, 50, 50, 255};

typedef struct Client {
    int window_width;
    int window_height;
    int header_offset;
    int cell_size;
    Texture2D agent_texture;
} Client;

// Initialize a rendering client
Client* make_client(CTrashPickupEnv* env) {
    const int CELL_SIZE = 40;
    Client* client = (Client*)malloc(sizeof(Client));
    client->cell_size = CELL_SIZE;
    client->header_offset = 60;
    client->window_width = env->grid_size * CELL_SIZE;
    client->window_height = client->window_width + client->header_offset;

    InitWindow(client->window_width, client->window_height, "Trash Pickup Environment");
    SetTargetFPS(60);

    client->agent_texture = LoadTexture("resources/puffers_128.png");

    return client;
}

// Render the TrashPickup environment
void c_render(Client* client, CTrashPickupEnv* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    // Draw header with current step and total episode reward
    int start_index = get_entity_type_start_index(env, TRASH);
    int total_trash_not_collected = 0;
    for (int i = start_index; i < start_index + env->num_trash; i++){
        total_trash_not_collected += env->entities[i].presence;
    }

    DrawText(
        TextFormat(
            "Step: %d\nTotal Episode Reward: %.2f\nTrash Collected: %d/%d",
            env->current_step,
            env->total_episode_reward,
            env->num_trash - total_trash_not_collected,
            env->num_trash
        ),
        5, 2, 10, PUFF_WHITE
    );

    // Draw the grid and its elements
    for (int x = 0; x < env->grid_size; x++) {
        for (int y = 0; y < env->grid_size; y++) {
            GridCell gridCell = env->grid[get_grid_index(env, x, y)];

            int cell_type;
            if (gridCell.entity)
            {
                cell_type = gridCell.entity->type;
            }
            else
            {
                cell_type = EMPTY;
            }

            int screen_x = x * client->cell_size;
            int screen_y = y * client->cell_size + client->header_offset;

            Rectangle cell_rect = {
                .x = screen_x,
                .y = screen_y,
                .width = client->cell_size,
                .height = client->cell_size
            };

            // Draw grid cell border
            DrawRectangleLines((int)cell_rect.x, (int)cell_rect.y, (int)cell_rect.width, (int)cell_rect.height, PUFF_LINES);

            // Draw grid cell content
            if (cell_type == EMPTY)
                continue;

            if (cell_type == TRASH) {
                DrawRectangle(
                    screen_x + client->cell_size / 4,
                    screen_y + client->cell_size / 4,
                    client->cell_size / 2,
                    client->cell_size / 2,
                    PUFF_CYAN
                );
            } else if (cell_type == TRASH_BIN) {
                DrawRectangle(
                    screen_x + client->cell_size / 8,
                    screen_y + client->cell_size / 8,
                    3 * client->cell_size / 4,
                    3 * client->cell_size / 4,
                    PUFF_RED
                );
            } else if (cell_type == AGENT) {
                Color color;
                if (env->do_human_control && gridCell.index == 0)
                {
                    // Make human controlled agent red
                    color = (Color){255, 128, 128, 255};
                }
                else
                {
                    // Non-human controlled agent
                    color = WHITE;
                }

                DrawTexturePro(
                    client->agent_texture, 
                    (Rectangle) {0, 0, 128, 128},
                    (Rectangle) {
                        screen_x + client->cell_size / 2, 
                        screen_y + client->cell_size / 2,
                        client->cell_size,
                        client->cell_size
                        },
                    (Vector2){client->cell_size / 2, client->cell_size / 2},
                    0,
                    color
                );

                Entity* thisAgent = &env->entities[gridCell.index];
                
                if (thisAgent->carrying)
                {
                    DrawRectangle(
                        screen_x + client->cell_size / 2,
                        screen_y + client->cell_size / 2,
                        client->cell_size / 4,
                        client->cell_size / 4,
                        PUFF_CYAN
                    );
                }
            }
        }
    }

    EndDrawing();
}

// Cleanup and free the rendering client
void close_client(Client* client) {
    UnloadTexture(client->agent_texture);
    CloseWindow();
    free(client);
}
