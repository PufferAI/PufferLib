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
const Color COLOR_CELL_GRASS = {150, 200, 150, 255}; // top of WALL cells
const Color COLOR_CELL_DIRT = {80, 50, 50, 255}; // side of WALL cells
const Color COLOR_CELL_GROUND = {150, 150, 170, 255}; // GROUND cells
// const Color COLOR_CELL_GRASS = {163, 197, 69, 255}; // top of WALL cells
// const Color COLOR_CELL_DIRT = {40, 20, 5, 255}; // side of WALL cells
// const Color COLOR_CELL_GROUND = {112, 123, 111, 255}; // GROUND cells
const Color COLOR_CELL_BORDER = RAYWHITE; // border of GROUND cells
const Color COLOR_CELL_MOVE = DARKGREEN;
const Color COLOR_CELL_MOVE_TEXT = RAYWHITE;
const Color COLOR_ACTIVE_PLAYER = RAYWHITE; // border around active player circle
const Color COLOR_PLAYER1 = RED; // player 1 color (character and circle)
const Color COLOR_PLAYER2 = GREEN; // player 2 color (character and circle)
const Color COLOR_TEXT_DEFAULT = RAYWHITE; // main text color
const Color COLOR_HEALTH = RED;
const Color COLOR_ACTION_POINTS = SKYBLUE;
const Color COLOR_MOVEMENT_POINTS = LIME;
const Color COLOR_SPELL = GOLD;
const Color COLOR_SPELL_COOLDOWN = BROWN;
const Color COLOR_CELL_SPELL = BEIGE;
const Color COLOR_CELL_ACTIVE_SPELL = ORANGE;
const Color COLOR_ENTITY_NAME = PURPLE;
const Color COLOR_ENTITY_NAME_HOVER = YELLOW;


// TODO many leaks...

// forward declarations
typedef struct Entity Entity;
typedef struct Spell Spell;

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
    int current_player_idx;

    unsigned int map_width;
    unsigned int map_height;
    unsigned int map_size; // width * height
    unsigned int* map;
    Entity** cell_to_entity;

    unsigned int* movement_path;
    int* movement_distance;
} Tactical;

// Entity (player, summoned creature...)
struct Entity {
    const char* name;
    int cell;
    Color color;
    int health_points_total;
    int action_points_total;
    int movement_points_total;
    int health_points_current;
    int action_points_current;
    int movement_points_current;

    // spells
    Spell* spells;
    int spell_count;
};

struct Spell {
    const char* name;
    int ap_cost;
    int cooldown;
    int remaining_cooldown;
    int range; // TODO add different range types (default, in a line, in diagonal...)
    bool line_of_sight; // whether spell can be casted across walls
    bool modifiable_range; // whether spell range can be increased or decreased from other spells
    // TODO add a "zone of effect" shape that lists the deltas that the spell touches around the cell
    void (*effect)(Tactical*, Entity*, int); // pointer to a function that takes in the env, the caster and the target cell
};

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
unsigned int get_row(Tactical* env, int cell) {
    return cell / env->map_width;
}
unsigned int get_col(Tactical* env, int cell) {
    return cell % env->map_width;
}

////////////
// SPELLS //
////////////

void update_cooldowns(Entity* entity) {
    for (int i = 0; i < entity->spell_count; ++i) {
        if (entity->spells[i].remaining_cooldown > 0) {
            entity->spells[i].remaining_cooldown--;
        }
    }
}

void cast_spell(Tactical* env, Entity* caster, Spell* spell, int target_cell) {
    // check if the spell can be cast
    if (caster->action_points_current < spell->ap_cost) {
        printf("Not enough action points to cast %s.\n", spell->name);
        return;
    }
    if (spell->remaining_cooldown > 0) {
        printf("Spell %s is on cooldown for %d more turns.\n", spell->name, spell->remaining_cooldown);
        return;
    }

    // cast the spell
    spell->effect(env, caster, target_cell);
    caster->action_points_current -= spell->ap_cost;
    spell->remaining_cooldown = spell->cooldown;
}

void spell_fire_arrow(Tactical* env, Entity* caster, int target_cell) {
    Entity* target = env->cell_to_entity[target_cell];
    if (target) {
        target->health_points_current -= 200;
    }
}
Spell create_spell_fire_arrow() {
    Spell spell;
    spell.name = "Fire Arrow";
    spell.ap_cost = 4;
    spell.cooldown = 0;
    spell.remaining_cooldown = 0;
    spell.range = 12;
    spell.effect = spell_fire_arrow;
    return spell;
}

void spell_damage_target2(Tactical* env, Entity* caster, int target_cell) {
    //TODO
}
Spell create_spell_damage_target2() {
    Spell spell;
    spell.name = "Fireball 2";
    spell.ap_cost = 6;
    spell.cooldown = 4;
    spell.remaining_cooldown = 0;
    spell.range = 20;
    spell.effect = spell_damage_target2;
    return spell;
}

void assign_spells(Entity* entity) {
    // TODO assign different spells based on class
    entity->spell_count = 5;
    entity->spells = malloc(entity->spell_count * sizeof(Spell));
    entity->spells[0] = create_spell_fire_arrow();
    entity->spells[1] = create_spell_damage_target2();
    entity->spells[2] = create_spell_damage_target2();
    entity->spells[3] = create_spell_damage_target2();
    entity->spells[4] = create_spell_damage_target2();
}

void compute_observations(Tactical* env) {

}

void compute_movement(Tactical* env, Entity* entity) {
    // Do a BFS from the entity's current position to find all reachable cells
    // within a distance of the entity's available movement points.
    // Store the result in env->movement_path, where each reachable cell 
    // points to the previous cell in the path, and in env->movement_distance,
    // where each reachable cell stores the distance to the player (or -1 if unreachable).

    // reset arrays
    for (int i = 0; i < env->map_size; ++i) {
        env->movement_path[i] = 0;
        env->movement_distance[i] = -1;
    }

    // compute walkable cells mask
    bool* walkable_cells = calloc(env->map_size, sizeof(bool));
    for (int i = 0; i < env->map_size; ++i) {
        // set ground cells to be walkable (TODO this should be pre-computed)
        if (env->map[i] == CELL_GROUND) {
            walkable_cells[i] = true;
        }
        // set all cells with entities to be non-walkable (TODO this should be updated whenever an entity moves or is added/removed)
        for (int j = 0; j < env->n_entities; ++j) {
            const unsigned int cell = env->entities[j].cell;
            walkable_cells[cell] = false;
        }
    }

    // TODO these can be calloc'ed once and reused (memset them to 0 each time this function is called)
    // EDIT: no, don't use memset for arrays of int, dangerous
    int* queue = calloc(env->map_size, sizeof(int));
    int* visited = calloc(env->map_size, sizeof(int));
    int* distances = calloc(env->map_size, sizeof(int));
    int front = 0;
    int rear = 0;

    // TODO can be static
    const int next_row_delta[4] = {1, -1, 0, 0};
    const int next_col_delta[4] = {0, 0, 1, -1};

    int start_pos = entity->cell;
    queue[rear++] = start_pos;
    visited[start_pos] = 1;
    distances[start_pos] = 0;

    while (front < rear) {
        int current = queue[front++];
        int row = current / env->map_width;
        int col = current % env->map_width;
        int current_distance = distances[current];

        if (current_distance >= entity->movement_points_current)
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

void move_entity(Tactical* env, Entity* entity, const int cell) {
    env->cell_to_entity[entity->cell] = NULL;
    entity->cell = cell;
    env->cell_to_entity[entity->cell] = entity;
    entity->movement_points_current -= env->movement_distance[cell];
    compute_movement(env, entity);
}

bool try_move_entity(Tactical* env, Entity* entity, const int cell) {
    // TODO i don't like this. Checks should be in game logic, not renderer.
    if (env->movement_path[cell]) {
        move_entity(env, entity, cell);
        return true;
    }
    return false;
}

Tactical* init_tactical() {
    Tactical* env = calloc(1, sizeof(Tactical));

    env->num_agents = 1;

    env->rewards = calloc(env->num_agents, sizeof(float));
    env->observations = calloc(env->num_agents*121*121*4, sizeof(unsigned char));
    env->actions = calloc(env->num_agents*1, sizeof(int));

    // init map
    int map_id = 3; // ok
    char* map_str = get_map(map_id);
    env->map_height = get_map_height(map_id);
    env->map_width = get_map_width(map_id);
    env->map_size = env->map_height * env->map_width;
    env->map = calloc(env->map_height * env->map_width, sizeof(unsigned int));
    for (int i = 0; i < env->map_height; i++) {
        for (int j = 0; j < env->map_width; j++) {
            int idx = i * env->map_width + j;
            switch (map_str[idx]) {
                case '-': env->map[idx] = CELL_EMPTY; break;
                case '.': env->map[idx] = CELL_GROUND; break;
                case '|': env->map[idx] = CELL_HOLE; break;
                case '#': env->map[idx] = CELL_WALL; break;
                default: printf("Invalid map character <%c> at row <%i> and column <%i>\n", map1[idx], i, j); exit(1);
            }
        }
    }

    env->cell_to_entity = (Entity**)calloc(env->map_size, sizeof(Entity*));

    // init players
    env->entities = calloc(2, sizeof(Entity));
    env->n_entities = 2;
    env->player1 = &env->entities[0];
    env->player2 = &env->entities[1];

    env->player1->name = "Player 1";
    env->player1->cell = get_cell(env, 8, 8);
    env->player1->color = COLOR_PLAYER1;
    env->player1->health_points_total = 2500;
    env->player1->action_points_total = 12;
    env->player1->movement_points_total = 10;
    env->player1->health_points_current = 2500;
    env->player1->action_points_current = 12;
    env->player1->movement_points_current = 10;

    env->player2->name = "Player 2";
    env->player2->cell = get_cell(env, 5, 10);
    env->player2->color = COLOR_PLAYER2;
    env->player2->health_points_total = 2500;
    env->player2->action_points_total = 12;
    env->player2->movement_points_total = 10;
    env->player2->health_points_current = 2500;
    env->player2->action_points_current = 12;
    env->player2->movement_points_current = 10;

    env->cell_to_entity[env->player1->cell] = env->player1;
    env->cell_to_entity[env->player2->cell] = env->player2;

    assign_spells(env->player1);
    assign_spells(env->player2);

    // // define a class
    // Class warrior = {
    //     "Warrior",
    //     (Spell[]){push}, // List of spells
    //     1                // Spell count
    // };


    env->current_player_idx = 0;
    env->current_player = &env->entities[env->current_player_idx];

    env->movement_path = calloc(env->map_size, sizeof(unsigned int));
    env->movement_distance = calloc(env->map_size, sizeof(int));
    compute_movement(env, env->current_player);

    return env;
}

void next_player(Tactical* env) {
    // reset current player AP and MP
    env->current_player->movement_points_current = env->current_player->movement_points_total;
    env->current_player->action_points_current = env->current_player->action_points_total;
    // decrease current player cooldowns
    update_cooldowns(env->current_player);
    // switch to next player
    env->current_player_idx = (env->current_player_idx + 1) % env->n_entities;
    env->current_player = &env->entities[env->current_player_idx];
    compute_movement(env, env->current_player);
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

    float cw; // cell width
    float ch; // cell height
    float offset_x; // offset for the whole map
    float offset_y; // offset for the whole map
    float dy; // vertical offset for wall cells

    // current cell (if any) under the mouse cursor
    int mx;
    int my;
    int mrow;
    int mcol;
    int mcell;
    int mcell_type;

    bool* movement_cells;
    Spell* active_spell;
    bool* spell_cells;

    // for drawing
    float *xa, *xb, *xc, *xd, *xe, *ya, *yb, *yc, *yd, *ye;

    // animations
    Entity* move_anim_entity;
    int* move_anim_path;
    int move_anim_path_idx;
    int move_anim_path_length;
    float move_anim_progress;
    float move_anim_dx; // delta in position with respect to the center of the cell
    float move_anim_dy;
    float move_anim_cells_per_second;

    clock_t last_render_time;
    double dt; // in seconds
    float max_fps;

} GameRenderer;


GameRenderer* init_game_renderer(Tactical* env) {
    GameRenderer* renderer = (GameRenderer*)calloc(1, sizeof(GameRenderer));
    renderer->width = 1200;
    renderer->height = 900;

    renderer->movement_cells = malloc(env->map_size * sizeof(bool));
    renderer->spell_cells = malloc(env->map_size * sizeof(bool));
    renderer->active_spell = NULL;

    // TODO fill the screen automatically (these are hardcoded for map 2)
    renderer->cw = 80;
    renderer->ch = renderer->cw / 2;
    renderer->offset_x = 560;
    renderer->offset_y = -200;
    renderer->dy = renderer->ch * 0.4;

    renderer->mcell = -1;
    renderer->mcell_type = -1;

    renderer->move_anim_path = calloc(env->map_size, sizeof(int));
    renderer->move_anim_cells_per_second = 6;
    
    renderer->xa = calloc(env->map_size, sizeof(float));
    renderer->xb = calloc(env->map_size, sizeof(float));
    renderer->xc = calloc(env->map_size, sizeof(float));
    renderer->xd = calloc(env->map_size, sizeof(float));
    renderer->xe = calloc(env->map_size, sizeof(float));
    renderer->ya = calloc(env->map_size, sizeof(float));
    renderer->yb = calloc(env->map_size, sizeof(float));
    renderer->yc = calloc(env->map_size, sizeof(float));
    renderer->yd = calloc(env->map_size, sizeof(float));
    renderer->ye = calloc(env->map_size, sizeof(float));
    for (int row = 0; row < env->map_height; ++row) {
        for (int col = 0; col < env->map_width; ++col) {
            int cell = get_cell(env, row, col);
            renderer->xa[cell] = renderer->offset_x + 0.5 * renderer->cw * (col - row);
            renderer->xb[cell] = renderer->xa[cell] - renderer->cw / 2;
            renderer->xc[cell] = renderer->xa[cell] + renderer->cw / 2;
            renderer->xd[cell] = renderer->xa[cell];
            renderer->xe[cell] = renderer->xa[cell];
            renderer->ya[cell] = renderer->offset_y + 0.5 * renderer->ch * (col + row + 2);
            renderer->yb[cell] = renderer->ya[cell] + renderer->ch / 2;
            renderer->yc[cell] = renderer->ya[cell] + renderer->ch / 2;
            renderer->yd[cell] = renderer->ya[cell] + renderer->ch;
            renderer->ye[cell] = renderer->yb[cell];
        }
    }

    renderer->last_render_time = clock();
    renderer->dt = 0.0f;
    renderer->max_fps = 120;

    InitWindow(renderer->width, renderer->height, "Tactical RL");
    // SetTargetFPS(60);

    return renderer;
}

int get_cell_at_cursor(GameRenderer* renderer, Tactical* env) {
    // to get the formula: we know that cell (row, col) starts at coordinates
    //     x = offset_x + 0.5 * cw * (col - row);
    //     y = offset_y + 0.5 * ch * (col + row + 2);
    // solve this 2x2 linear system to write (row, col)) as a function of (x, y) and we get the formulas below
    const int mx = GetMouseX();
    const int my = GetMouseY();
    const int mrow = floor((my - renderer->offset_y) / renderer->ch - (mx - renderer->offset_x) / renderer->cw - 1);
    const int mcol = floor((my - renderer->offset_y) / renderer->ch + (mx - renderer->offset_x) / renderer->cw - 1);
    renderer->mx = mx;
    renderer->my = my;
    renderer->mrow = mrow;
    renderer->mcol = mcol;
    const int mcell = (mrow < 0 || mcol < 0 || mrow >= env->map_height || mcol >= env->map_width) ? -1 : get_cell(env, mrow, mcol);
    return mcell;
}

void draw_debug_info(GameRenderer* renderer) {
    DrawText(TextFormat("%i FPS", (int)(1 / renderer->dt)), 10, 10, 20, COLOR_TEXT_DEFAULT);
    DrawText(TextFormat("Mouse: %i, %i", renderer->mx, renderer->my), 150, 10, 15, COLOR_TEXT_DEFAULT);
    DrawText(TextFormat("Cell: %i (row %i, col %i)", renderer->mcell, renderer->mrow, renderer->mcol), 150, 30, 15, COLOR_TEXT_DEFAULT);
    DrawText(TextFormat("Cell type: %s",
        renderer->mcell_type == CELL_EMPTY ? "EMPTY" :
        renderer->mcell_type == CELL_GROUND ? "GROUND" :
        renderer->mcell_type == CELL_HOLE ? "HOLE" :
        renderer->mcell_type == CELL_WALL ? "WALL" : 
        renderer->mcell_type == -1 ? "NONE" : "UNKNOWN"), 150, 45, 15, COLOR_TEXT_DEFAULT);
}

void draw_player(GameRenderer* renderer, float x, float y, Color color, bool is_current_player, Color cell_color) {
    // draw the little guy
    if (is_current_player) DrawEllipse(x, y, 0.33 * renderer->cw, 0.33 * renderer->ch, COLOR_ACTIVE_PLAYER);
    DrawEllipse(x, y, 0.3 * renderer->cw, 0.3 * renderer->ch, color);
    DrawEllipse(x, y, 0.23 * renderer->cw, 0.23 * renderer->ch, cell_color);
    DrawLineEx((Vector2){x - 0.1 * renderer->cw, y}, (Vector2){x, y - 0.7 * renderer->ch}, 2, color);
    DrawLineEx((Vector2){x + 0.1 * renderer->cw, y}, (Vector2){x, y - 0.7 * renderer->ch}, 2, color);
    DrawLineEx((Vector2){x, y - 0.7 * renderer->ch}, (Vector2){x, y - 1.1 * renderer->ch}, 2, color);
    DrawLineEx((Vector2){x, y - 1.0 * renderer->ch}, (Vector2){x - 0.2 * renderer->cw, y - 0.8 * renderer->ch}, 2, color);
    DrawLineEx((Vector2){x, y - 1.0 * renderer->ch}, (Vector2){x + 0.2 * renderer->cw, y - 0.8 * renderer->ch}, 2, color);
    DrawCircle(x, y - 1.3 * renderer->ch, 0.2 * renderer->ch, color);
}

void draw_cells_and_entities(GameRenderer* renderer, Tactical* env) {
    // draw isometric cells
    //    (ground)
    //       a
    //   b   e   c  (b<->c = cw)
    //       d
    //    (a<->d = ch)

    // first draw ground cells
    for (int cell = 0; cell < env->map_size; ++cell) {
        int cell_type = env->map[cell];
        if (cell_type == CELL_GROUND) {
            // draw isometric cell (a, b, c, d)
            Color cell_color = COLOR_CELL_GROUND;
            if (renderer->movement_cells[cell]) {
                cell_color = COLOR_CELL_MOVE;
            } else if (renderer->active_spell && renderer->spell_cells[cell]) {
                cell_color = cell == renderer->mcell ? COLOR_CELL_ACTIVE_SPELL : COLOR_CELL_SPELL;
            }
            // DrawTriangleStrip((Vector2[]){{xa, ya}, {xb, yb}, {xc, yc}, {xd, yd}}, 4, cell_color);
            DrawTriangleStrip((Vector2[]){
                {renderer->xa[cell], renderer->ya[cell]},
                {renderer->xb[cell], renderer->yb[cell]},
                {renderer->xc[cell], renderer->yc[cell]},
                {renderer->xd[cell], renderer->yd[cell]}}, 4, cell_color);
            if (renderer->movement_cells[cell]) {
                const unsigned int dist = env->movement_distance[cell];
                const char* text = TextFormat("%i", dist);
                DrawText(text, 
                    renderer->xe[cell] - MeasureText(text, 12) / 2,
                    renderer->ye[cell] - 6, 12, COLOR_CELL_MOVE_TEXT);
            }
            // draw white border around cell
            DrawLineStrip((Vector2[]){
                {renderer->xa[cell], renderer->ya[cell]},
                {renderer->xb[cell], renderer->yb[cell]},
                {renderer->xd[cell], renderer->yd[cell]},
                {renderer->xc[cell], renderer->yc[cell]},
                {renderer->xa[cell], renderer->ya[cell]}}, 5, COLOR_CELL_BORDER);
        }
    }

    // then draw walls and entities alternatively, from top-left to bottom-right, for correct z-order
    bool draw_horizontally = true;  // draw row by row, from top-left to bottom-right (if false: column by column)
    if (renderer->move_anim_entity) {
        int col = get_col(env, renderer->move_anim_path[renderer->move_anim_path_idx]);
        int col_next = get_col(env, renderer->move_anim_path[renderer->move_anim_path_idx + 1]);
        if (col == col_next) {
            draw_horizontally = false; // this is all for correct depth (z-order) rendering
        }
    }
    for (int i = 0; i < env->map_size; ++i) {
        int row, col;
        if (draw_horizontally) {
            row = i / env->map_width;
            col = i % env->map_width;
        } else {
            row = i % env->map_height;
            col = i / env->map_height;
        }
        int cell = get_cell(env, row, col);
        int cell_type = env->map[cell];
        if (cell_type == CELL_WALL) {
            // draw isometric cell (a, b, c, d) shifted up by dy ("grass")
            DrawTriangleStrip((Vector2[]){
                {renderer->xa[cell], renderer->ya[cell] - renderer->dy},
                {renderer->xb[cell], renderer->yb[cell] - renderer->dy},
                {renderer->xc[cell], renderer->yc[cell] - renderer->dy},
                {renderer->xd[cell], renderer->yd[cell] - renderer->dy}}, 4, COLOR_CELL_GRASS);
            // draw connections between (a, b, c, d) and the shifted up cell ("dirt")
            DrawTriangleStrip((Vector2[]){
                {renderer->xc[cell], renderer->yc[cell]},
                {renderer->xc[cell], renderer->yc[cell] - renderer->dy},
                {renderer->xd[cell], renderer->yd[cell]},
                {renderer->xd[cell], renderer->yd[cell] - renderer->dy},
                {renderer->xb[cell], renderer->yb[cell]},
                {renderer->xb[cell], renderer->yb[cell] - renderer->dy}}, 6, COLOR_CELL_DIRT);
        }

        // draw entity at cell (if any)
        Color cell_color = COLOR_CELL_GROUND;
        if (renderer->movement_cells[cell]) {
            cell_color = COLOR_CELL_MOVE;
        } else if (renderer->active_spell && renderer->spell_cells[cell]) {
            cell_color = cell == renderer->mcell ? COLOR_CELL_ACTIVE_SPELL : COLOR_CELL_SPELL;
        }
        Entity* entity = env->cell_to_entity[cell];
        if (entity && entity != renderer->move_anim_entity) {
            draw_player(renderer,
                renderer->xe[cell], renderer->ye[cell],
                entity->color, entity == env->current_player, cell_color);
        }
        // if entity is under move animation, handle it differently
        if (renderer->move_anim_entity && renderer->move_anim_path[renderer->move_anim_path_idx] == cell) {
            draw_player(renderer,
                renderer->xe[cell] + renderer->move_anim_dx, renderer->ye[cell] + renderer->move_anim_dy, 
                renderer->move_anim_entity->color, renderer->move_anim_entity == env->current_player, cell_color);
        }
    }
}

void draw_player_dashboard(GameRenderer* renderer, Entity* dashboard_entity, bool is_current_player) {
    // Health, action points, movement points
    DrawText(dashboard_entity->name, 40, renderer->height - 150, 25,
        is_current_player ? COLOR_ENTITY_NAME : COLOR_ENTITY_NAME_HOVER);
    DrawText(TextFormat("HP: %i / %i", dashboard_entity->health_points_current, dashboard_entity->health_points_total),
             40, renderer->height - 120, 25, COLOR_HEALTH);
    DrawText(TextFormat("AP: %i / %i", dashboard_entity->action_points_current, dashboard_entity->action_points_total),
             40, renderer->height - 90, 25, COLOR_ACTION_POINTS);
    DrawText(TextFormat("MP: %i / %i", dashboard_entity->movement_points_current, dashboard_entity->movement_points_total),
             40, renderer->height - 60, 25, COLOR_MOVEMENT_POINTS);

    // Spells
    DrawText("Spells", 300, renderer->height - 150, 20, COLOR_TEXT_DEFAULT);
    for (int i = 0; i < dashboard_entity->spell_count; ++i) {
        Spell* spell = &dashboard_entity->spells[i];
        if (spell->remaining_cooldown > 0) {
            DrawText(TextFormat("[%i] %s (cooldown: %i)", 
                    i+1, spell->name, spell->remaining_cooldown), 
                300, renderer->height - 125 + i * 20, 20, COLOR_SPELL_COOLDOWN);
        } else {
            DrawText(TextFormat("[%i] %s", i+1, spell->name),
                300, renderer->height - 125 + i * 20, 20, COLOR_SPELL);
        }
    }
}

int render_game(GameRenderer* renderer, Tactical* env) {
    if (IsKeyDown(KEY_Q) || IsKeyDown(KEY_BACKSPACE)) {
        return 1;  // close window
    }

    // cap FPS and compute dt
    clock_t current_time;
    do {
        current_time = clock();
        renderer->dt = (double)(current_time - renderer->last_render_time) / CLOCKS_PER_SEC;
    } while (renderer->dt < 1 / renderer->max_fps);
    renderer->last_render_time = current_time;

    BeginDrawing();
    ClearBackground(COLOR_BACKGROUND);

    int cursor = MOUSE_CURSOR_DEFAULT;

    // get current cell at cursor position (if any), and draw debug info
    renderer->mcell = get_cell_at_cursor(renderer, env);
    renderer->mcell_type = renderer->mcell == -1 ? -1 : env->map[renderer->mcell];
    draw_debug_info(renderer);
    
    const int mcell = renderer->mcell;
    // movement path display, if applicable ; and spells
    memset(renderer->movement_cells, 0, env->map_size * sizeof(bool));
    if (renderer->active_spell) {
        if (mcell != -1 && renderer->spell_cells[mcell]) {
            cursor = MOUSE_CURSOR_POINTING_HAND;
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                cast_spell(env, env->current_player, renderer->active_spell, mcell);
                renderer->active_spell = NULL;
            }
        }
    } else {
        if (mcell != -1 && env->movement_path[mcell] && !renderer->move_anim_entity) {
            int cell = mcell;
            int path_length = env->movement_distance[mcell];
            for (int i = path_length; i >= 0; --i) {
                if (i != 0) {
                    renderer->movement_cells[cell] = true;
                }
                renderer->move_anim_path[i] = cell; // precompute in case it's used, cause after moving the env->movement_path is no longer valid
                cell = env->movement_path[cell];
            }
            cursor = MOUSE_CURSOR_POINTING_HAND;
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                if (try_move_entity(env, env->current_player, mcell)) {
                    // start move animation
                    renderer->move_anim_entity = env->current_player;
                    renderer->move_anim_path_idx = 0;
                    renderer->move_anim_path_length = path_length;
                    renderer->move_anim_progress = 0;
                    renderer->move_anim_dx = 0;
                    renderer->move_anim_dy = 0;
                }
            }
        }
    }

    if (renderer->move_anim_entity) {
        renderer->move_anim_progress += renderer->dt * renderer->move_anim_cells_per_second;
        if (renderer->move_anim_progress >= 1) {
            renderer->move_anim_progress = fmod(renderer->move_anim_progress, 1);
            renderer->move_anim_path_idx += 1;
        }
        if (renderer->move_anim_path_idx == renderer->move_anim_path_length) {
            // reached last cell: stop animation
            renderer->move_anim_entity = NULL;
        } else {
            int current_cell = renderer->move_anim_path[renderer->move_anim_path_idx];
            int next_cell = renderer->move_anim_path[renderer->move_anim_path_idx + 1];
            int current_row = get_row(env, current_cell);
            int next_row = get_row(env, next_cell);
            int current_col = get_col(env, current_cell);
            int next_col = get_col(env, next_cell);
            int move_dx, move_dy;
            if (next_row == current_row + 1) {
                move_dx = -1;
                move_dy = 1;
            } else if (next_row == current_row - 1) {
                move_dx = 1;
                move_dy = -1;
            } else if (next_col == current_col + 1) {
                move_dx = 1;
                move_dy = 1;
            } else if (next_col == current_col - 1) {
                move_dx = -1;
                move_dy = -1;
            } else {
                // should be an impossible case
                move_dx = 0;
                move_dy = 0;
            }
            renderer->move_anim_dx = renderer->move_anim_progress * move_dx * renderer->cw * 0.5;
            renderer->move_anim_dy = renderer->move_anim_progress * move_dy * renderer->ch * 0.5;
        }
    }

    // KEYS
    if (IsKeyPressed(KEY_SPACE)) {
        renderer->active_spell = NULL;
        next_player(env);
    }

    int tentative_spell_id = -1;
    if (IsKeyPressed(KEY_ONE)) tentative_spell_id = 0;
    else if (IsKeyPressed(KEY_TWO)) tentative_spell_id = 1;

    if (tentative_spell_id >= 0 && tentative_spell_id < env->current_player->spell_count) {
        Spell* spell = &env->current_player->spells[tentative_spell_id];
        if (spell->remaining_cooldown == 0 && env->current_player->action_points_current >= spell->ap_cost) {
            renderer->active_spell = spell;

            memset(renderer->spell_cells, 0, env->map_size * sizeof(bool));
            // TODO compute lines of sight (TODO this should be precomputed each time an entity moves)
            for (int i = 0; i < env->map_size; ++i) {
                if (env->map[i] == CELL_GROUND) {
                    renderer->spell_cells[i] = true;
                }
            }
        }
    }

    if (IsKeyPressed(KEY_ESCAPE)) {
        renderer->active_spell = NULL;
    }

    draw_cells_and_entities(renderer, env);

    // Write info about keys
    DrawText("Press Q or Backspace to exit", 600, 10, 15, COLOR_TEXT_DEFAULT);
    DrawText("Press the corresponding key [1-5] to cast a spell", 600, 25, 15, COLOR_TEXT_DEFAULT);
    DrawText("Press Space to skip turn", 600, 40, 15, COLOR_TEXT_DEFAULT);

    // Draw player dashboard (health, action points, movement points, spells)
    Entity* dashboard_entity = env->current_player;
    if (renderer->mcell != -1 && env->cell_to_entity[renderer->mcell] && env->cell_to_entity[renderer->mcell] != env->current_player) {
        dashboard_entity = env->cell_to_entity[mcell];
        cursor = MOUSE_CURSOR_POINTING_HAND;
    }
    draw_player_dashboard(renderer, dashboard_entity, dashboard_entity == env->current_player);

    SetMouseCursor(cursor);

    EndDrawing();
    return 0;
}

void close_game_renderer(GameRenderer* renderer) {
    CloseWindow();
    free(renderer->movement_cells);
    free(renderer);
}