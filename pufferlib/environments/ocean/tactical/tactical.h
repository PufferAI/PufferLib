#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "raylib.h"
#include "maps.h"


#define CELL_EMPTY 0
#define CELL_GROUND 1
#define CELL_HOLE 2
#define CELL_WALL 3

const Color COLOR_BACKGROUND = {6, 24, 24, 255}; // window background
const Color COLOR_CELL_GRASS = {163, 197, 69, 255}; // top of WALL cells
const Color COLOR_CELL_DIRT = {40, 20, 5, 255}; // side of WALL cells
const Color COLOR_CELL_GROUND = {112, 123, 111, 255}; // GROUND cells
const Color COLOR_CELL_BORDER = RAYWHITE; // border of GROUND cells
const Color COLOR_CELL_MOVE = DARKGREEN;
const Color COLOR_CELL_MOVE_TEXT = RAYWHITE;
const Color COLOR_ACTIVE_PLAYER = RAYWHITE; // border around active player circle
const Color COLOR_PLAYER1 = RED; // player 1 color
const Color COLOR_PLAYER2 = GREEN; // player 2 color
const Color COLOR_TEXT_DEFAULT = RAYWHITE; // main text color


typedef struct Entity {
    unsigned int row;
    unsigned int col;
    Color color;
    unsigned int movement_points;
} Entity;

typedef struct Tactical {
    int num_agents;
    unsigned char* observations;
    int* actions;
    float* rewards;

    unsigned int n_entities;
    Entity* entities;
    // pointers to entities (these won't be allocated), assume 1v1 for now
    Entity* player1;
    Entity* player2;
    Entity* current_player;

    unsigned int map_width;
    unsigned int map_height;
    unsigned int map_size; // width * height
    unsigned int* map;

    unsigned int* movement_path;
    int* movement_distance;
} Tactical;

void free_tactical(Tactical* env) {
    free(env->rewards);
    free(env->observations);
    free(env->actions);
    free(env->map);
    free(env->movement_path);
    free(env->movement_distance);
    free(env->entities);

    free(env); // do this last
}

unsigned int get_cell(Tactical* env, int row, int col) {
    return row * env->map_width + col;
}

unsigned int get_cell_entity(Tactical*env, Entity* entity) {
    return get_cell(env, entity->row, entity->col);
}

void compute_observations(Tactical* env) {

}

void compute_movement(Tactical* env, Entity* entity) {
    // Do a BFS from the entity's current position to find all reachable cells
    // within a distance of the entity's available movement points.
    // Store the result in env->movement_path, where each reachable cell 
    // points to the previous cell in the path, and in env->movement_distance,
    // where each reachable cell stores the distance to the player (or -1 if unreachable).

    // reset array
    memset(env->movement_path, 0, env->map_size);
    memset(env->movement_distance, -1, env->map_size);

    // compute walkable cells mask
    bool* walkable_cells = calloc(env->map_size, sizeof(bool));
    for (int i = 0; i < env->map_size; ++i) {
        // set ground cells to be walkable (TODO this should be pre-computed)
        if (env->map[i] == CELL_GROUND) {
            walkable_cells[i] = true;
        }
        // set all cells with entities to be non-walkable (TODO this should be updated whenever an entity moves or is added/removed)
        for (int j = 0; j < env->n_entities; ++j) {
            const unsigned int cell = get_cell_entity(env, &env->entities[j]);
            walkable_cells[cell] = false;
        }
    }

    // TODO these can be calloc'ed once and reused (memset them to 0 each time this function is called)
    int* queue = calloc(env->map_size, sizeof(int));
    int* visited = calloc(env->map_size, sizeof(int));
    int* distances = calloc(env->map_size, sizeof(int));
    int front = 0;
    int rear = 0;

    // TODO can be pre-computed once
    const int next_row_delta[4] = {1, -1, 0, 0};
    const int next_col_delta[4] = {0, 0, 1, -1};

    int start_pos = get_cell(env, entity->row, entity->col);
    queue[rear++] = start_pos;
    visited[start_pos] = 1;
    distances[start_pos] = 0;

    while (front < rear) {
        int current = queue[front++];
        int row = current / env->map_width;
        int col = current % env->map_width;
        int current_distance = distances[current];

        if (current_distance >= entity->movement_points)
            continue;

        // explore neighbors
        for (int i = 0; i < 4; ++i) {
            int next_row = row + next_row_delta[i];
            int next_col = col + next_col_delta[i];

            // boundary check
            if (next_row < 0 || next_col < 0 || next_row >= env->map_height || next_col >= env->map_width)
                continue;

            int next = next_row * env->map_width + next_col;

            // skip if already visited or not a ground cell
            if (visited[next] || !walkable_cells[next])
                continue;

            // mark as visited and record distance
            visited[next] = 1;
            distances[next] = current_distance + 1;
            env->movement_path[next] = current; // store previous cell in the path
            env->movement_distance[next] = distances[next]; // store previous cell in the path

            // enqueue neighbor
            queue[rear++] = next;
        }
    }

    // cleanup
    free(queue);
    free(visited);
    free(distances);
}

Tactical* init_tactical() {
    Tactical* env = calloc(1, sizeof(Tactical));

    env->num_agents = 1;

    env->rewards = calloc(env->num_agents, sizeof(float));
    env->observations = calloc(env->num_agents*121*121*4, sizeof(unsigned char));
    env->actions = calloc(env->num_agents*1, sizeof(int));

    // init map
    int map_id = 2;
    char* map_str = get_map(map_id);
    env->map_height = get_map_height(map_id);
    env->map_width = get_map_width(map_id);
    env->map_size = env->map_height * env->map_width;
    env->map = calloc(env->map_height * env->map_width, sizeof(unsigned int));
    for (int i = 0; i < env->map_height; i++) {
        for (int j = 0; j < env->map_width; j++) {
            int idx = i * env->map_width + j;
            switch (map_str[idx]) {
                case '_': env->map[idx] = CELL_EMPTY; break;
                case '.': env->map[idx] = CELL_GROUND; break;
                case '|': env->map[idx] = CELL_HOLE; break;
                case '#': env->map[idx] = CELL_WALL; break;
                default: printf("Invalid map character <%c> at row <%i> and column <%i>\n", map1[idx], i, j); exit(1);
            }
        }
    }

    // init players
    env->entities = calloc(2, sizeof(Entity));
    env->n_entities = 2;
    env->player1 = &env->entities[0];
    env->player2 = &env->entities[1];

    env->player1->row = 8;
    env->player1->col = 8;
    env->player1->color = COLOR_PLAYER1;
    env->player1->movement_points = 10;

    env->player2->row = 5;
    env->player2->col = 10;    
    env->player2->color = COLOR_PLAYER2;
    env->player2->movement_points = 10;

    env->current_player = env->player1;

    env->movement_path = calloc(env->map_size, sizeof(unsigned int));
    env->movement_distance = calloc(env->map_size, sizeof(int));
    compute_movement(env, env->current_player);

    return env;
}

void reset(Tactical* env) {
    compute_observations(env);
}

int step(Tactical* env) {
    if (false) {
        reset(env);
        int winner = 2;
        return winner;
    }

    compute_observations(env);
    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// RENDERING ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

typedef struct {
    int width;
    int height;

    bool* movement_cells;
} GameRenderer;


GameRenderer* init_game_renderer(Tactical* env) {
    GameRenderer* renderer = (GameRenderer*)calloc(1, sizeof(GameRenderer));
    renderer->width = 1200;
    renderer->height = 900;

    renderer->movement_cells = malloc(env->map_size * sizeof(bool));

    InitWindow(renderer->width, renderer->height, "Tactical");
    SetTargetFPS(60);

    return renderer;
}

void draw_player(float x, float y, float cw, float ch, Color color, bool is_current_player) {
    // draw the little guy
    if (is_current_player) DrawEllipse(x, y, 0.33 * cw, 0.33 * ch, COLOR_ACTIVE_PLAYER);
    DrawEllipse(x, y, 0.3 * cw, 0.3 * ch, color);
    DrawEllipse(x, y, 0.23 * cw, 0.23 * ch, COLOR_CELL_GROUND);
    DrawLineEx((Vector2){x - 0.1 * cw, y}, (Vector2){x, y - 0.7 * ch}, 2, color);
    DrawLineEx((Vector2){x + 0.1 * cw, y}, (Vector2){x, y - 0.7 * ch}, 2, color);
    DrawLineEx((Vector2){x, y - 0.7 * ch}, (Vector2){x, y - 1.1 * ch}, 2, color);
    DrawLineEx((Vector2){x, y - 1.0 * ch}, (Vector2){x - 0.2 * cw, y - 0.8 * ch}, 2, color);
    DrawLineEx((Vector2){x, y - 1.0 * ch}, (Vector2){x + 0.2 * cw, y - 0.8 * ch}, 2, color);
    DrawCircle(x, y - 1.3 * ch, 0.2 * ch, color);
}

int render_game(GameRenderer* renderer, Tactical* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        return 1;
    }

    BeginDrawing();
    ClearBackground(COLOR_BACKGROUND);

    // TODO fill the screen automatically (these are hardcoded for map 2 and should go in the init / GameRenderer)
    const float cw = 80; // cell width
    const float ch = cw / 2; // cell height
    const float offset_x = 560; // offset for the whole map
    const float offset_y = -200;
    const float dy = ch * 0.4; // vertical offset for wall cells

    // get current cell at cursor position (if any)
    const int mx = GetMouseX();
    const int my = GetMouseY();
    DrawText(TextFormat("Mouse: %i, %i", mx, my), 150, 10, 15, COLOR_TEXT_DEFAULT);
    // to get the formula: we know that cell (row, col) starts at coordinates
    //     x = offset_x + 0.5 * cw * (col - row);
    //     y = offset_y + 0.5 * ch * (col + row + 2);
    // solve this 2x2 linear system to write (row, col)) as a function of (x, y) and we get the formulas below
    const int mrow = floor((my - offset_y) / ch - (mx - offset_x) / cw - 1);
    const int mcol = floor((my - offset_y) / ch + (mx - offset_x) / cw - 1);
    const int mcell = (mrow < 0 || mcol < 0 || mrow >= env->map_height || mcol >= env->map_width) ? -1 : get_cell(env, mrow, mcol);
    DrawText(TextFormat("Cell: %i (row %i, col %i)", mcell, mrow, mcol), 150, 30, 15, COLOR_TEXT_DEFAULT);
    const int mcell_type = mcell == -1 ? -1 : env->map[mcell];
    DrawText(TextFormat("Cell type: %s",
        mcell_type == CELL_EMPTY ? "EMPTY" :
        mcell_type == CELL_GROUND ? "GROUND" :
        mcell_type == CELL_HOLE ? "HOLE" :
        mcell_type == CELL_WALL ? "WALL" : 
        mcell_type == -1 ? "NONE" : "UNKNOWN"), 150, 45, 15, COLOR_TEXT_DEFAULT);

    // movement path display, if applicable
    renderer->movement_cells = calloc(env->map_size, sizeof(bool));
    if (mcell != -1 && env->movement_path[mcell]) {
        unsigned int current_cell = mcell;
        unsigned int end_cell = get_cell_entity(env, env->current_player);
        while (current_cell != end_cell) {
            renderer->movement_cells[current_cell] = true;
            current_cell = env->movement_path[current_cell];
        }
    }

    // draw isometric cells from top-left to bottom-right
    //    (ground)
    //       a
    //   b   e   c  (b<->c = cw)
    //       d
    //    (a<->d = ch)
    float xa, xb, xc, xd, xe, ya, yb, yc, yd, ye;
    for (int row = 0; row < env->map_height; ++row) {
        for (int col = 0; col < env->map_width; ++col) {
            const unsigned int cell = get_cell(env, row, col);
            const unsigned int cell_type = env->map[cell];

            // compute isometrics coordinates (points a, b, c, d, e)
            // TODO this can be pre-computed if we need more rendering performance but it's probably fine
            xa = offset_x + 0.5 * cw * (col - row);
            xb = xa - cw / 2;
            xc = xa + cw / 2;
            xd = xa;
            xe = xa;

            ya = offset_y + 0.5 * ch * (col + row + 2);
            yb = ya + ch / 2;
            yc = ya + ch / 2;
            yd = ya + ch;
            ye = yb;

            // draw cell
            if (cell_type == CELL_WALL) {
                // draw isometric cell (a, b, c, d) shifted up by dy ("grass")
                DrawTriangleStrip((Vector2[]){{xa, ya - dy}, {xb, yb - dy}, {xc, yc - dy}, {xd, yd - dy}}, 4, COLOR_CELL_GRASS);
                // draw connections between (a, b, c, d) and the shifted up cell ("dirt")
                DrawTriangleStrip((Vector2[]){{xc, yc}, {xc, yc - dy}, {xd, yd}, {xd, yd - dy}, {xb, yb}, {xb, yb - dy}}, 6, COLOR_CELL_DIRT);
            } else if (cell_type == CELL_HOLE) {
                // leave empty, as a hole should be
            } else if (cell_type == CELL_GROUND) {
                // draw isometric cell (a, b, c, d)
                Color cell_color = renderer->movement_cells[cell] ? COLOR_CELL_MOVE : COLOR_CELL_GROUND;
                DrawTriangleStrip((Vector2[]){{xa, ya}, {xb, yb}, {xc, yc}, {xd, yd}}, 4, cell_color);
                if (renderer->movement_cells[cell]) {
                    const unsigned int dist = env->movement_distance[cell];
                    const char* text = TextFormat("%i", dist);
                    DrawText(text, xe - MeasureText(text, 12) / 2, ye - 6, 12, COLOR_CELL_MOVE_TEXT);
                }
                // draw white border around cell
                DrawLineStrip((Vector2[]){{xa, ya}, {xb, yb}, {xd, yd}, {xc, yc}, {xa, ya}}, 5, COLOR_CELL_BORDER);

                // draw player if here (TODO loop over a list of entities)
                if (row == env->player1->row && col == env->player1->col) {
                    draw_player(xe, ye, cw, ch, env->player1->color, env->player1 == env->current_player);
                } else if (row == env->player2->row && col == env->player2->col) {
                    draw_player(xe, ye, cw, ch, env->player2->color, env->player2 == env->current_player);
                }

            }
        }
    }

    DrawFPS(10, 10);

    EndDrawing();
    return 0;
}

void close_game_renderer(GameRenderer* renderer) {
    CloseWindow();
    free(renderer->movement_cells);
    free(renderer);
}