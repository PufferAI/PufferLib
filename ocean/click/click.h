/* Click: An environment to train an RL agent to click targets fast! 
The agent has to move the mouse from one target to the next and then click. 
Misclicking is penalized.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <raylib.h>

// Constants
#define MAX_TARGETS 5
#define NUM_START_TARGETS 5
#define OBS_SIZE (3 + 3 * MAX_TARGETS + 2)

#define ACTION_SIZE 3
#define TARGET_RADIUS_MIN 1
#define TARGET_RADIUS_MAX 10

// Action space: 2D movement (delta x, delta y) and click status (0 or 1)
#define CONTINUOUS 0 // Set to 1 for continuous actions; 0 is discrete
#define NUM_BINS 5
static float DELTA_X[NUM_BINS] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
static float DELTA_Y[NUM_BINS] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
static const float STATUS[2] = {0, 1};

const Color PUFF_CYAN = (Color){0, 187, 187, 255}; 

// Define structs
typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float n;
} Log;

typedef struct {
    float width;
    float height;
    Texture2D puffer;      // pufferfish sprite
    bool puffer_loaded;
    int frame;             // animation counter for ocean/bubbles
} Client;

typedef struct {
    float x;
    float y;
    float status;
} Agent;

typedef struct {
    float x;
    float y;
    float radius;
    float spawn_time;
} Target;

typedef struct {
    Log log;
    Client* client;
    Agent agent;
    Target targets[MAX_TARGETS];
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    int width;
    int height;
    int target_spawn_duration;
    unsigned int rng;
    int tick;
    int episode_length;
    int episode_return;
    int targets_hit;
    int targets_total;
    int lives;
    Vector2 prev_mouse;
    int human_input;
} ClickEnv;

// Util functions
void add_log(ClickEnv* env) {
    env->log.episode_length += env->episode_length;
    env->log.episode_return += env->targets_hit;
    env->log.score += (float)env->targets_hit / env->targets_total;
    env->log.perf  += (float)env->targets_hit / env->targets_total;
    env->log.n++;
}

int get_human_input(ClickEnv* env) {
    Vector2 mouse = GetMousePosition();

    // If the mouse did not move, return nothing
    if (env->prev_mouse.x < 0) {
        env->prev_mouse = mouse;
        return 0;
    }
    // Set state directly
    float move_x = mouse.x;
    float move_y = mouse.y;
    env->prev_mouse = mouse;

    int moved   = (fabsf(move_x) > 0.5f || fabsf(move_y) > 0.5f);
    int clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);

    if (!moved && !clicked) return 0;

    env->actions[0] = move_x; 
    env->actions[1] = move_y;
    env->actions[2] = clicked ? 1 : 0;
    return 1;

}

void init(ClickEnv* env) {
    env->tick = 0;
}

static void draw_ocean(int w, int h, int frame) {
    // Vertical gradient: lighter near the surface, darker at depth
    Color top    = (Color){ 30, 130, 170, 255 };
    Color bottom = (Color){  3,  35,  70, 255 };
    DrawRectangleGradientV(0, 0, w, h, top, bottom);

    // Wavy horizontal light bands
    for (int b = 0; b < 5; b++) {
        float baseY = h * (0.1f + 0.18f * b);
        for (int x = 0; x < w; x += 8) {
            float yy = baseY + sinf((x * 0.03f) + frame * 0.05f + b) * 6.0f;
            DrawRectangle(x, (int)yy, 8, 2, (Color){255, 255, 255, 18});
        }
    }

    // Rising bubbles
    for (int i = 0; i < 40; i++) {
        float bx = (i * 137) % w;
        float speed = 0.5f + (i % 5) * 0.25f;
        float by = h - fmodf(frame * speed + i * 53, (float)h);
        float br = 1.0f + (i % 3);
        DrawCircleLines((int)bx, (int)by, br, (Color){255, 255, 255, 40});
    }
}

// Environment functions
void compute_observations(ClickEnv* env) {
    // We have one agent and multiple targets.
    env->observations[0] = env->agent.x/env->width;
    env->observations[1] = env->agent.y/env->height;
    env->observations[2] = env->agent.status;

    int obs_idx = 3;
    // For each target, we store its x, y, radius and spawn time
    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            env->observations[obs_idx]     = (env->targets[i].x - env->agent.x)/env->width;
            env->observations[obs_idx + 1] = (env->targets[i].y - env->agent.y)/env->height;
            env->observations[obs_idx + 2] = env->targets[i].radius/TARGET_RADIUS_MAX;
        } else {
            env->observations[obs_idx]     = 0.0f;
            env->observations[obs_idx + 1] = 0.0f;
            env->observations[obs_idx + 2] = 0.0f;
        }
        obs_idx += 3;
    }
}

void c_reset(ClickEnv* env) {
    env->tick = 0;
    env->episode_return = 0;
    env->lives = 3;
    env->targets_hit = 0;
    env->targets_total = NUM_START_TARGETS;
    env->prev_mouse = (Vector2){ -1.0f, -1.0f };  

    // Spawn a number of targets at random locations
    for (int i = 0; i < NUM_START_TARGETS; i++) {
        env->targets[i].x = rand_r(&env->rng) % env->width;
        env->targets[i].y = rand_r(&env->rng) % env->height;
        env->targets[i].radius = TARGET_RADIUS_MIN;
        env->targets[i].spawn_time = 0;
    };

    // Initialize the agent (mouse cursor) near the center of the screen
    env->agent.x = env->width / 2 + rand_r(&env->rng) % 21 - 10; 
    env->agent.y = env->height / 2 + rand_r(&env->rng) % 21 - 10;
    env->agent.status = 0; // Not clicked

    compute_observations(env);
}

static inline float clipf(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

void c_step(ClickEnv* env) {
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    if (env->human_input) {
        env->agent.x = env->actions[0];
        env->agent.y = env->actions[1];
        env->agent.status = env->actions[2];
    } else {
        if (CONTINUOUS) {
            env->agent.x += env->actions[0];
            env->agent.y += env->actions[1];
            env->agent.status = env->actions[2];
        } else {
            env->agent.x += DELTA_X[(int)env->actions[0]];
            env->agent.y += DELTA_Y[(int)env->actions[1]];
            env->agent.status = env->actions[2];
        }
    }
    
    env->agent.x = clipf(env->agent.x, 0, env->width);
    env->agent.y = clipf(env->agent.y, 0, env->height);

    int click_status = env->agent.status;
   
    // Update environment
    // Target spawn times and remove targets that have been on the screen for too long
    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            env->targets[i].spawn_time += 1;
            if (env->targets[i].spawn_time > env->target_spawn_duration) {
                env->targets[i].spawn_time = -1; // Mark target for removal 
            }
        }
        // Spawn new targets if there are less than the max number of targets
        if (env->targets[i].spawn_time == -1 && rand_r(&env->rng) % 100 < 20) { // 20% chance to spawn a new target
            env->targets[i].x = rand_r(&env->rng) % env->width;
            env->targets[i].y = rand_r(&env->rng) % env->height;
            env->targets[i].radius = TARGET_RADIUS_MIN;
            env->targets[i].spawn_time = 0; 
            env->targets_total += 1;
        }
    }
    
    if (click_status == 1) {
        // Check if the agent position (mouse cursor) is within the radius of any target
        int new_targets_hit = 0;
        
        for (int j = 0; j < MAX_TARGETS; j++) {
            if (env->targets[j].spawn_time >= 0) {
                float dist_x = env->agent.x - env->targets[j].x;
                float dist_y = env->agent.y - env->targets[j].y;
                float distance = sqrt(dist_x * dist_x + dist_y * dist_y);
                if (distance <= env->targets[j].radius) {
                    new_targets_hit += 1;
                    env->targets_hit += 1;
                    env->rewards[0] += 1.0f;
                    env->targets[j].spawn_time = -1; // Mark target for removal
                }
            }
        }
        if (new_targets_hit == 0) {
            // Misclicked, give penalty
            env->rewards[0] -= 1;
            env->lives -= 1;
        }
    }
    // Increase target size
    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            env->targets[i].radius += 1;
        }
    }

    if ((env->tick >= env->episode_length) || (env->lives <= -1)) {
        env->terminals[0] = 1; 
        add_log(env);
        c_reset(env);
    } else {
        env->terminals[0] = 0; 
    }

    env->tick += 1;    

    compute_observations(env);
}   

void c_render(ClickEnv* env) {

    if (env->client == NULL) {
        InitWindow(env->width, env->height, "Click environment");
        SetTargetFPS(60);
        env->client = (Client*)malloc(sizeof(Client));
        env->client->width  = env->width;
        env->client->height = env->height;
        env->client->frame  = 0;
        env->client->puffer = LoadTexture("resources/shared/puffer.png");
        env->client->puffer_loaded = (env->client->puffer.id != 0);

        // Smooth scaling when the sprite is drawn small/large
        if (env->client->puffer_loaded) {
            SetTextureFilter(env->client->puffer, TEXTURE_FILTER_BILINEAR);
        }
    }

    Client* client = env->client;
    client->frame++;

    BeginDrawing();
    
    // Background
    draw_ocean(env->width, env->height, client->frame);

    // Draw targets
    for (int i = 0; i < MAX_TARGETS; i++) {
        if (env->targets[i].spawn_time >= 0) {
            float r = env->targets[i].radius;

            if (client->puffer_loaded) {
                // Draw the sprite so its on-screen size grows with the
                // target radius. The fish ~fills the 423px frame, so we
                // want drawn size a bit larger than the hit circle (2*r)
                // for the body to visually cover the target.
                float texW   = (float)client->puffer.width;
                float drawSz  = 2.0f * r * 1.4f;
                float scale   = drawSz / texW;

                // Center the image on the target center.
                Vector2 pos = {
                    env->targets[i].x - drawSz * 0.5f,
                    env->targets[i].y - drawSz * 0.5f
                };

                DrawTextureEx(client->puffer, pos, 0.0f, scale, WHITE);
            } 
        }
    }

    // Agent (mouse cursor)
    Color c = (env->agent.status == 1) ? YELLOW : WHITE;
    float x = env->agent.x;
    float y = env->agent.y;
    Vector2 tip   = { x,      y      };
    Vector2 left  = { x,      y + 16 };
    Vector2 notch = { x + 4,  y + 12 };
    Vector2 right = { x + 11, y + 11 };
    DrawTriangle(tip, left, right, c);
    DrawTriangle(left, notch, right, c);
    DrawTriangleLines(tip, left, right, BLACK);
    DrawTriangleLines(left, notch, right, BLACK);

    DrawText(TextFormat("Timestep: %d", env->tick), 10, 10, 20, RAYWHITE);
    DrawText(TextFormat("Lives: %d", env->lives), 10, 35, 20, RAYWHITE);
    DrawText(TextFormat("Targets hit: %d", env->targets_hit), 200, 10, 20,
             (Color){180, 255, 180, 255});

    EndDrawing();
}

void c_close(ClickEnv* env) {
    if (env->client != NULL) {
        Client* client = env->client;
        if (client->puffer_loaded) {
            UnloadTexture(client->puffer);
        }
        CloseWindow();
        free(client);
    }
}