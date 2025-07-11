#include <stdlib.h>
#include <string.h>
#include <ncurses.h>
#include <stdbool.h>
#include <time.h>
#include <signal.h>

#define PAIR_EMPTY 0
#define PAIR_PUFFER 1
#define PAIR_BACKGROUND 2
#define PAIR_ROAD 3
#define PAIR_WATER 4
#define PAIR_OBSTACLE 5
#define PAIR_LILYPAD 6
#define PAIR_WATERFROG 7
#define PAIR_GOAL 8
#define PAIR_START 9

#define NUM_LANES 5
#define MAX_CARS_PER_LANE 3
#define KEY_ESC 27
#define MAX_MAP_HEIGHT 25
#define MAX_MAP_WIDTH 100

#define CELL_EMPTY 0
#define CELL_ROAD 1
#define CELL_WATER 2
#define CELL_GOAL 3
#define CELL_START 4
#define CELL_OBSTACLE 5
#define CELL_LILYPAD 6

#define BASIC_OBSERVATIONS 5
#define CAR_OBSERVATIONS_PER_CAR 5  // x_pos, y_pos, active, speed, direction
#define TOTAL_CAR_OBSERVATIONS (NUM_LANES * MAX_CARS_PER_LANE * CAR_OBSERVATIONS_PER_CAR)
#define GRID_SIZE 5
#define GRID_OBSERVATIONS (GRID_SIZE * GRID_SIZE * 2)  // Both map cells and car presence
#define TOTAL_OBSERVATIONS (BASIC_OBSERVATIONS + TOTAL_CAR_OBSERVATIONS + GRID_OBSERVATIONS)

typedef struct {
    float lives_remaining;
    float crossings;
    float episode_length;
    float episode_return;
    float n;
} Log;

typedef struct {
    int x;
    int y;
    int active;
} Car;

typedef struct {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    int width;
    int height;
    int episode_length;
    int frog_x;
    int frog_y;
    int last_y;
    int lives;
    int step_count;
    float score;
    int crossings;
    int rendered; 
    Car cars[NUM_LANES][MAX_CARS_PER_LANE];
    int car_speeds[NUM_LANES];
    int car_directions[NUM_LANES]; // 1 right, -1 left
    float car_spawn_timers[NUM_LANES];
    int map[MAX_MAP_HEIGHT][MAX_MAP_WIDTH];
    int map_seed;
    WINDOW* game_panel;
    WINDOW* stats_panel;
} Froggy;

void c_reset(Froggy* env);
void c_step(Froggy* env);
void c_render(Froggy* env );
void c_close(Froggy* env);
void froggy_ui_init(Froggy* env);
int froggy_ui_get_input();
void generate_map(Froggy* env, int seed);

void allocate(Froggy* env) {
    env->observations = (float*)calloc(TOTAL_OBSERVATIONS, sizeof(float));
    env->actions = (int*)calloc(1, sizeof(int));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
}
void free_allocated(Froggy* env) {
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
}
void compute_observations(Froggy* env) {
    float *obs = env->observations;
    obs[0] = (float)env->frog_x;
    obs[1] = (float)env->frog_y;
    obs[2] = (float)env->lives;
    obs[3] = (float)env->crossings;
    obs[4] = (float)env->score;
    const float inv_width = 1.0f / env->width;
    const float inv_height = 1.0f / env->height;
    const float inv_max_speed = 1.0f / 3.0f;
    
    // car obs
    int obs_index = BASIC_OBSERVATIONS;
    for (int lane = 0; lane < NUM_LANES; lane++) {
        for (int car = 0; car < MAX_CARS_PER_LANE; car++) {
            if (obs_index + 5 > TOTAL_OBSERVATIONS) break;  // Prevent buffer overflow
            const Car *c = &env->cars[lane][car];
            const int active = c->active;
            const int in_bounds = (c->x >= 0 && c->x < env->width);
            // car x
            obs[obs_index++] = (active && in_bounds) ? c->x * inv_width : -1.0f;
            // car y
            obs[obs_index++] = active ? c->y * inv_height : -1.0f;
            // is car
            obs[obs_index++] = (float)active;
            // car spd
            obs[obs_index++] = active ? env->car_speeds[lane] * inv_max_speed : 0.0f;
            // car dir
            obs[obs_index++] = active ? (float)env->car_directions[lane] : 0.0f;
        }
        if (obs_index + 5 > TOTAL_OBSERVATIONS) break;
    }

    int car_grid[MAX_MAP_HEIGHT][MAX_MAP_WIDTH];
    memset(car_grid, 0, sizeof(car_grid));
    for (int lane = 0; lane < NUM_LANES; lane++) {
        for (int car = 0; car < MAX_CARS_PER_LANE; car++) {
            const Car *c = &env->cars[lane][car];
            if (c->active && c->x >= 0 && c->x < env->width && c->y >= 0 && c->y < env->height &&
                c->y < MAX_MAP_HEIGHT && c->x < MAX_MAP_WIDTH) {
                car_grid[c->y][c->x] = 1;
            }
        }
    }
    
    // 5x5 obs
    const int grid_radius = GRID_SIZE / 2;
    const float inv_cell_types = 1.0f / 6.0f;
    
    for (int dy = -grid_radius; dy <= grid_radius; dy++) {
        for (int dx = -grid_radius; dx <= grid_radius; dx++) {
            const int world_x = env->frog_x + dx;
            const int world_y = env->frog_y + dy;
            const int in_bounds = (world_x >= 0 && world_x < env->width && world_y >= 0 && world_y < env->height);
            
            // Add bounds check for obs_index
            if (obs_index >= TOTAL_OBSERVATIONS) break;
            obs[obs_index++] = in_bounds ? env->map[world_y][world_x] * inv_cell_types : -1.0f;
        }
        if (obs_index >= TOTAL_OBSERVATIONS) break;
    }
    
    // is car 5x5
    for (int dy = -grid_radius; dy <= grid_radius; dy++) {
        for (int dx = -grid_radius; dx <= grid_radius; dx++) {
            const int world_x = env->frog_x + dx;
            const int world_y = env->frog_y + dy;
            const int in_bounds = (world_x >= 0 && world_x < env->width && 
                                 world_y >= 0 && world_y < env->height);
            
            // Add bounds check for obs_index
            if (obs_index >= TOTAL_OBSERVATIONS) break;
            obs[obs_index++] = (in_bounds && car_grid[world_y][world_x]) ? 1.0f : 0.0f;
        }
        if (obs_index >= TOTAL_OBSERVATIONS) break;
    }
}
void add_log(Froggy* env) {
    env->log.lives_remaining = env->lives;
    env->log.crossings = env->crossings;
    env->log.episode_length = env->step_count;
    env->log.episode_return = env->score;
    env->log.n++;
}

void move_frog(Froggy* env) {
    int new_x = env->frog_x;
    int new_y = env->frog_y;
    if (env->actions[0] == 0 && env->frog_y > 0) {
        new_y = env->frog_y - 1;
    } else if (env->actions[0] == 1 && env->frog_y < env->height - 1) {
        new_y = env->frog_y + 1;
    } else if (env->actions[0] == 2 && env->frog_x > 0) {
        new_x = env->frog_x - 1;
    } else if (env->actions[0] == 3 && env->frog_x < env->width - 1) {
        new_x = env->frog_x + 1;
    }
    env->frog_x = new_x;
    env->frog_y = new_y;
}
void died(Froggy* env) {
    env->lives--;
    env->rewards[0] = -0.5;
    env->score -= 0.5;
    env->frog_x = env->width / 2;
    env->frog_y = env->height - 1;
    env->last_y = env->height - 1;
    add_log(env);
}
void set_stage(Froggy* env) {
    generate_map(env, env->map_seed + 1);

    for (int lane = 0; lane < NUM_LANES; lane++) {
        env->car_speeds[lane] = 1 + (rand() % 3);
        env->car_directions[lane] = (lane % 2 == 0) ? 1 : -1;
        env->car_spawn_timers[lane] = rand() % 30;
        for (int car = 0; car < MAX_CARS_PER_LANE; car++) {
            env->cars[lane][car].active = 0;
        }
    }
    env->frog_x = env->width / 2;
    env->frog_y = env->height - 1;
    env->last_y = env->height - 1;
}


void c_reset(Froggy* env) {
    env->lives = 3;
    env->step_count = 0;
    env->crossings = 0;
    env->score = 0;
    set_stage(env);
    compute_observations(env);
}
void c_step(Froggy* env) {

    // printf("Step: %d, Frog Position: (%d, %d), Lives: %d, Score: %.0f\n",
    //        env->step_count, env->frog_x, env->frog_y, env->lives, env->score);
    env->rewards[0] = 0;
    env->terminals[0] = 0;
    env->step_count++;
    move_frog(env);

    if (env->frog_y < env->last_y) {
        env->rewards[0] = 0.1;
        env->score += 0.1;
        env->last_y = env->frog_y;
        // add_log(env);
    }
    
    // hit water or obstacle
    if (env->frog_y >= 0 && env->frog_y < env->height && env->frog_y < MAX_MAP_HEIGHT &&
        env->frog_x >= 0 && env->frog_x < env->width && env->frog_x < MAX_MAP_WIDTH) {
        if (env->map[env->frog_y][env->frog_x] == CELL_WATER) {
            died(env);
            return;
        }
        if (env->map[env->frog_y][env->frog_x] == CELL_OBSTACLE) {
            died(env);
            return;
        }
    }

    if (env->frog_y <= 0) {
        env->rewards[0] = 1;
        env->score += 1;
        env->crossings += 1;
        add_log(env);
        set_stage(env);
    }
    if (env->lives <= 0) {
        env->terminals[0] = 1;
        env->rewards[0] = -1;
        env->score -= 1;
        add_log(env);
        c_reset(env);
        return;
    }
    if (env->step_count >= env->episode_length) {
        env->terminals[0] = 1;
        add_log(env);
        c_reset(env);
        return;
    }
    
    // move car
    for (int lane = 0; lane < NUM_LANES; lane++) {
        for (int car = 0; car < MAX_CARS_PER_LANE; car++) {
            if (env->cars[lane][car].active) {
                // frog hit
                if (env->cars[lane][car].x == env->frog_x && env->cars[lane][car].y == env->frog_y) {
                    died(env);
                    return;
                }
                
                int old_x = env->cars[lane][car].x;
                env->cars[lane][car].x += env->car_directions[lane] * env->car_speeds[lane];
                if (env->cars[lane][car].x < -2 || env->cars[lane][car].x > env->width + 1) {
                    env->cars[lane][car].active = 0;
                    continue;
                }
                int car_in_bounds = (env->cars[lane][car].x >= 0 && env->cars[lane][car].x < env->width &&
                                   env->cars[lane][car].y >= 0 && env->cars[lane][car].y < env->height &&
                                   env->cars[lane][car].y < MAX_MAP_HEIGHT && env->cars[lane][car].x < MAX_MAP_WIDTH);
                
                if (car_in_bounds && env->map[env->cars[lane][car].y][env->cars[lane][car].x] == CELL_OBSTACLE) {
                    env->cars[lane][car].active = 0; 
                    continue;
                }
                if (car_in_bounds && env->map[env->cars[lane][car].y][env->cars[lane][car].x] != CELL_ROAD) {
                    env->cars[lane][car].x = old_x;
                    if (old_x < 0 || old_x >= env->width || old_x >= MAX_MAP_WIDTH ||
                        env->cars[lane][car].y < 0 || env->cars[lane][car].y >= MAX_MAP_HEIGHT ||
                        env->map[env->cars[lane][car].y][old_x] != CELL_ROAD) {
                        env->cars[lane][car].active = 0;
                    }
                }
                // hit frog
                if (env->cars[lane][car].active && env->cars[lane][car].x == env->frog_x && env->cars[lane][car].y == env->frog_y) {
                    died(env);
                    return;
                }
            }
        }
        // spawn car
        env->car_spawn_timers[lane]--;
        if (env->car_spawn_timers[lane] <= 0) {
            env->car_spawn_timers[lane] = 20 + (rand() % 40);

            int road_rows[MAX_MAP_HEIGHT];
            int road_count = 0;
            for (int y = 1; y < env->height - 1; y++) {
                int is_road_row = 0;
                for (int x = 0; x < env->width; x++) {
                    if (env->map[y][x] == CELL_ROAD) {
                        is_road_row = 1;
                        break;
                    }
                }
                
                if (is_road_row) {
                    road_rows[road_count++] = y;
                }
            }
            if (road_count > 0) {
                int spawn_y = road_rows[lane % road_count];
                int spawn_x = (env->car_directions[lane] > 0) ? -1 : env->width;
                
                // is car
                bool position_occupied = false;
                for (int check_lane = 0; check_lane < NUM_LANES && !position_occupied; check_lane++) {
                    for (int check_car = 0; check_car < MAX_CARS_PER_LANE; check_car++) {
                        if (env->cars[check_lane][check_car].active &&
                            env->cars[check_lane][check_car].x == spawn_x &&
                            env->cars[check_lane][check_car].y == spawn_y) {
                            position_occupied = true;
                            break;
                        }
                    }
                }
                if (!position_occupied) {
                    for (int car = 0; car < MAX_CARS_PER_LANE; car++) {
                        if (!env->cars[lane][car].active) {
                            env->cars[lane][car].active = 1;
                            env->cars[lane][car].y = spawn_y;
                            env->cars[lane][car].x = spawn_x;
                            break;
                        }
                    }
                }
            }
        }
    }

    compute_observations(env);
}
void c_render(Froggy* env) {
    if (env->rendered == 0) {
        froggy_ui_init(env);
    }
    werase(env->game_panel);
    werase(env->stats_panel);
    int game_height, game_width;
    getmaxyx(env->game_panel, game_height, game_width);
    game_height -= 2;
    game_width -= 2;
    
    // center
    int start_y = (game_height - env->height) / 2 + 1;
    int start_x = (game_width - env->width) / 2 + 1;
    if (start_y < 1) start_y = 1;
    if (start_x < 1) start_x = 1;

    for (int y = 0; y < env->height; y++) {
        int panel_y = start_y + y;
        if (panel_y > game_height) {
            continue;
        }
        for (int x = 0; x < env->width; x++) {
            int panel_x = start_x + x;
            if (panel_x > game_width) {
                continue;
            }
            switch (env->map[y][x]) {
                case CELL_GOAL:
                    wattron(env->game_panel, COLOR_PAIR(PAIR_GOAL));
                    mvwprintw(env->game_panel, panel_y, panel_x, "#");  // Goal area
                    wattroff(env->game_panel, COLOR_PAIR(PAIR_GOAL));
                    break;
                case CELL_START:
                    wattron(env->game_panel, COLOR_PAIR(PAIR_START));
                    mvwprintw(env->game_panel, panel_y, panel_x, "S");  // Start area
                    wattroff(env->game_panel, COLOR_PAIR(PAIR_START));
                    break;
                case CELL_ROAD:
                    wattron(env->game_panel, COLOR_PAIR(PAIR_ROAD));
                    mvwprintw(env->game_panel, panel_y, panel_x, "=");  // Road
                    wattroff(env->game_panel, COLOR_PAIR(PAIR_ROAD));
                    break;
                case CELL_WATER:
                    wattron(env->game_panel, COLOR_PAIR(PAIR_WATER));
                    mvwprintw(env->game_panel, panel_y, panel_x, "~");  // Water
                    wattroff(env->game_panel, COLOR_PAIR(PAIR_WATER));
                    break;
                case CELL_OBSTACLE:
                    wattron(env->game_panel, COLOR_PAIR(PAIR_OBSTACLE));
                    mvwprintw(env->game_panel, panel_y, panel_x, "X");  // Obstacle
                    wattroff(env->game_panel, COLOR_PAIR(PAIR_OBSTACLE));
                    break;
                case CELL_LILYPAD:
                    wattron(env->game_panel, COLOR_PAIR(PAIR_LILYPAD));
                    mvwprintw(env->game_panel, panel_y, panel_x, "o");  // Lily pad
                    wattroff(env->game_panel, COLOR_PAIR(PAIR_LILYPAD));
                    break;
                default:
                    wattron(env->game_panel, COLOR_PAIR(PAIR_EMPTY));
                    mvwprintw(env->game_panel, panel_y, panel_x, " ");  // Empty space
                    wattroff(env->game_panel, COLOR_PAIR(PAIR_EMPTY));
            }
        }
    }
    // render cars
    for (int lane = 0; lane < NUM_LANES; lane++) {
        for (int car = 0; car < MAX_CARS_PER_LANE; car++) {
            if (env->cars[lane][car].active && 
                env->cars[lane][car].x >= 0 && 
                env->cars[lane][car].x < env->width) {
                
                int panel_y = start_y + env->cars[lane][car].y;
                int panel_x = start_x + env->cars[lane][car].x;
                
                if (panel_y <= game_height && panel_x <= game_width) {
                    if (env->map[env->cars[lane][car].y][env->cars[lane][car].x] == CELL_ROAD) {
                        wattron(env->game_panel, COLOR_PAIR(PAIR_OBSTACLE));
                        mvwprintw(env->game_panel, panel_y, panel_x, "C");  // Car
                        wattroff(env->game_panel, COLOR_PAIR(PAIR_OBSTACLE));
                    }
                }
            }
        }
    }
    // render frog
    int frog_panel_y = start_y + env->frog_y;
    int frog_panel_x = start_x + env->frog_x;
    if (frog_panel_y <= game_height && frog_panel_x <= game_width) {
        wattron(env->game_panel, COLOR_PAIR(PAIR_PUFFER));
        mvwprintw(env->game_panel, frog_panel_y, frog_panel_x, "@");
        wattroff(env->game_panel, COLOR_PAIR(PAIR_PUFFER));
    }

    add_log(env);
    mvwprintw(env->stats_panel, 1, 2, "Lives: %d", env->lives);
    mvwprintw(env->stats_panel, 2, 2, "Score: %.0f", env->score);
    mvwprintw(env->stats_panel, 3, 2, "Crossings: %d", env->crossings);
    mvwprintw(env->stats_panel, 4, 2, "Path: Always Available");
    int stats_width;
    getmaxyx(env->stats_panel, game_height, stats_width);
    mvwprintw(env->stats_panel, 1, stats_width / 2 - 5, "FROGGY GAME");
    wrefresh(env->game_panel);
    wrefresh(env->stats_panel);
}
void c_close(Froggy* env) {
    endwin();
}
void handle_sigint(int sig) {
    endwin();
    exit(1);
}
void froggy_ui_init(Froggy* env) {
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();
    curs_set(0);
    timeout(100);
    signal(SIGINT, handle_sigint);

    if (has_colors()) {
        start_color();
        // init_color(COLOR_CYAN, 0, 1000, 1000); // Custom blue color
        init_pair(PAIR_EMPTY, COLOR_WHITE, COLOR_WHITE);
        init_pair(PAIR_PUFFER, COLOR_CYAN, COLOR_BLACK);
        init_pair(PAIR_BACKGROUND, COLOR_BLACK, COLOR_BLACK);
        init_pair(PAIR_ROAD, COLOR_WHITE, COLOR_BLACK);
        init_pair(PAIR_WATER, COLOR_BLUE, COLOR_BLUE);
        init_pair(PAIR_OBSTACLE, COLOR_RED, COLOR_BLACK);
        init_pair(PAIR_LILYPAD, COLOR_GREEN, COLOR_BLUE);
        init_pair(PAIR_WATERFROG, COLOR_GREEN, COLOR_BLUE);
        init_pair(PAIR_GOAL, COLOR_YELLOW, COLOR_BLACK);
        init_pair(PAIR_START, COLOR_WHITE, COLOR_BLACK);

    }
    
    clear();
    refresh();
    int max_y, max_x;
    getmaxyx(stdscr, max_y, max_x);
    int game_height = (int)(max_y * 0.8);
    env->game_panel = newwin(game_height, max_x, 0, 0);
    env->stats_panel = newwin(max_y - game_height, max_x, game_height, 0);
    wrefresh(env->game_panel);
    wrefresh(env->stats_panel);
    env->rendered = 1;
}
int froggy_ui_get_input() {
    int ch = getch();
    if (ch == KEY_UP || ch == 'w' || ch == 'W') return 0;
    if (ch == KEY_DOWN || ch == 's' || ch == 'S') return 1;
    if (ch == KEY_LEFT || ch == 'a' || ch == 'A') return 2;
    if (ch == KEY_RIGHT || ch == 'd' || ch == 'D') return 3;
    if (ch == KEY_ESC || ch == 'q' || ch == 'Q') return -1;
    return 4;
}

void create_lilypad_path(Froggy* env, int start_y, int end_y) {
    for (int y = start_y; y <= end_y; y++) {
        if (env->map[y][0] == CELL_WATER) {
            int current_x = rand() % env->width;
            env->map[y][current_x] = CELL_LILYPAD;
            int num_extra_pads = 1 + rand() % 3;
            for (int i = 0; i < num_extra_pads; i++) {
                int pad_x = rand() % env->width;
                if (env->map[y][pad_x] == CELL_WATER) {
                    env->map[y][pad_x] = CELL_LILYPAD;
                }
            }
        }
    }
    
    int path_x = env->width / 2;
    for (int y = start_y; y <= end_y; y++) {
        if (env->map[y][0] == CELL_WATER) {
            int nearest_pad = -1;
            int min_distance = env->width;
            for (int x = 0; x < env->width; x++) {
                if (env->map[y][x] == CELL_LILYPAD) {
                    int distance = abs(x - path_x);
                    if (distance < min_distance) {
                        min_distance = distance;
                        nearest_pad = x;
                    }
                }
            }
            if (nearest_pad == -1 || min_distance > 3) {
                int new_x = path_x + (rand() % 3 - 1);
                if (new_x < 0) new_x = 0;
                if (new_x >= env->width) new_x = env->width - 1;
                env->map[y][new_x] = CELL_LILYPAD;
                path_x = new_x;
            } else {
                path_x = nearest_pad;
            }
        }
    }
}
int is_passable(Froggy* env, int x, int y) {
    if (x < 0 || x >= env->width || y < 0 || y >= env->height) {
        return 0;
    }
    int cell_type = env->map[y][x];
    return cell_type != CELL_WATER && cell_type != CELL_OBSTACLE;
}
int find_path(Froggy* env, int x, int y, int visited[MAX_MAP_HEIGHT][MAX_MAP_WIDTH]) {
    if (x < 0 || x >= env->width || y < 0 || y >= env->height || 
        visited[y][x] || !is_passable(env, x, y)) {
        return 0;
    }
    visited[y][x] = 1;
    if (y == 0) {
        return 1;
    }
    int dx[4] = {0, 1, 0, -1}; // Up, right, down, left
    int dy[4] = {-1, 0, 1, 0}; 
    for (int i = 0; i < 4; i++) {
        if (find_path(env, x + dx[i], y + dy[i], visited)) {
            return 1;
        }
    }
    return 0;
}
void ensure_path_exists(Froggy* env) {
    int visited[MAX_MAP_HEIGHT][MAX_MAP_WIDTH] = {0};
    int start_x = env->width / 2;
    int start_y = env->height - 1;
    if (find_path(env, start_x, start_y, visited)) {
        return;
    }

    int current_x = start_x;
    int current_y = start_y;
    while (current_y > 0) {
        current_y--;
        if (env->map[current_y][current_x] == CELL_WATER) {
            env->map[current_y][current_x] = CELL_LILYPAD;
        } else if (env->map[current_y][current_x] == CELL_OBSTACLE) {
            env->map[current_y][current_x] = CELL_ROAD;
        }
        if (rand() % 100 < 30 && current_y > 1) {
            int direction = (rand() % 2) * 2 - 1; // -1 or 1
            int new_x = current_x + direction;
            
            if (new_x >= 0 && new_x < env->width) {
                current_x = new_x;
                if (env->map[current_y][current_x] == CELL_WATER) {
                    env->map[current_y][current_x] = CELL_LILYPAD;
                } else if (env->map[current_y][current_x] == CELL_OBSTACLE) {
                    env->map[current_y][current_x] = CELL_ROAD;
                }
            }
        }
    }
}
void generate_map(Froggy* env, int seed) {
    if (seed <= 0) {
        seed = (int)time(NULL);
    }
    env->map_seed = seed;
    srand(seed);
    
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            env->map[y][x] = CELL_EMPTY;
        }
    }
    
    for (int x = 0; x < env->width; x++) {
        env->map[0][x] = CELL_GOAL;
    }
    
    for (int x = 0; x < env->width; x++) {
        env->map[env->height-1][x] = CELL_START;
    }
    
    int num_road_sections = 2 + rand() % 3; // 2-4 road sections
    int num_water_sections = 1 + rand() % 2; // 1-2 water sections
    int total_sections = num_road_sections + num_water_sections;
    int available_rows = env->height - 2;
    int section_height = available_rows / total_sections;
    int section_types[10]; // Max 10 sections
    int section_count = 0;
    
    for (int i = 0; i < num_road_sections; i++) {
        section_types[section_count++] = CELL_ROAD;
    }
    
    for (int i = 0; i < num_water_sections; i++) {
        section_types[section_count++] = CELL_WATER;
    }
    
    for (int i = 0; i < section_count; i++) {
        int j = rand() % section_count;
        int temp = section_types[i];
        section_types[i] = section_types[j];
        section_types[j] = temp;
    }
    
    for (int s = 0; s < section_count; s++) {
        int start_y = 1 + s * section_height;
        int end_y = start_y + section_height;
        if (end_y >= env->height - 1) end_y = env->height - 2;
        
        for (int y = start_y; y <= end_y; y++) {
            for (int x = 0; x < env->width; x++) {
                env->map[y][x] = section_types[s];
            }
            
            if (section_types[s] == CELL_ROAD && rand() % 100 < 15) { 
                int obstacles = 1 + rand() % 2;
                for (int o = 0; o < obstacles; o++) {
                    int ox = rand() % env->width;
                    if (env->map[y][ox] != CELL_OBSTACLE) {
                        env->map[y][ox] = CELL_OBSTACLE;
                    }
                }
            }
        }
        
        if (section_types[s] == CELL_WATER) {
            create_lilypad_path(env, start_y, end_y);
        }
    }
    
    srand(time(NULL));
    ensure_path_exists(env);
}
