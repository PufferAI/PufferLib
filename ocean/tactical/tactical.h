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

#define MAX_TEXT_ANIMATIONS 100  // max number of text animations

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
const Color COLOR_CELL_ACTIVE_SPELL = {255, 161, 0, 255};
const Color COLOR_CELL_INACTIVE_SPELL = {150, 150, 255, 255};
const Color COLOR_ENTITY_NAME = PURPLE;
const Color COLOR_ENTITY_NAME_HOVER = YELLOW;


// TODO many leaks...

// forward declarations
typedef struct Tactical Tactical;
typedef struct Entity Entity;
typedef struct Spell Spell;
typedef struct Client Client;

// arghh annoying
void add_animation_text(Tactical* env, Client* client, const char* text, int cell, Color color, int font_size, float duration);
void compute_movement(Tactical* env, Entity* entity);

typedef struct Log {
    float score;
    float n;
} Log;

typedef struct Tactical {
    Log log;
    Client* client;
    int num_agents;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;

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
    bool cast_in_line; // whether spell range can be increased or decreased from other spells
    // TODO add a "zone of effect" shape that lists the deltas that the spell touches around the cell
    void (*effect)(Tactical*, Entity*, int, Spell*); // pointer to a function that takes in the env, the caster and the target cell
    bool (*render_animation)(Tactical*, Client*, int, int, float, Spell*); // pointer to a function that takes in the env, the client, the caster cell, the target cell and the progress (in seconds), that renders the spell animation and returns true when the animation is finished
    int damage; // damage dealt by the spell
    int animation_state;
    int aoe_range; // 0: single-target; 1: 5 cells in total (1+4); 2: 13 cells total (1+4+8)
};

struct Client {
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
    bool* active_spell_cells;

    // for drawing
    float *xa, *xb, *xc, *xd, *xe, *ya, *yb, *yc, *yd, *ye;

    // animations (move)
    Entity* move_anim_entity;
    int* move_anim_path;
    int move_anim_path_idx;
    int move_anim_path_length;
    float move_anim_progress;
    float move_anim_dx; // delta in position with respect to the center of the cell
    float move_anim_dy;
    float move_anim_cells_per_second;

    // animations (text)
    char* text_anim_texts[MAX_TEXT_ANIMATIONS];
    float text_anim_x0[MAX_TEXT_ANIMATIONS];
    float text_anim_x1[MAX_TEXT_ANIMATIONS];
    float text_anim_y0[MAX_TEXT_ANIMATIONS];
    float text_anim_y1[MAX_TEXT_ANIMATIONS];
    float text_anim_progress[MAX_TEXT_ANIMATIONS];
    float text_anim_duration[MAX_TEXT_ANIMATIONS]; // in seconds
    Color text_anim_color[MAX_TEXT_ANIMATIONS];
    int text_anim_font_size[MAX_TEXT_ANIMATIONS];
    int text_anim_count;

    // animations (spells) -- only one at a time
    Spell* spell_anim;
    float spell_anim_progress;
    int spell_anim_caster_cell;
    int spell_anim_target_cell;

    clock_t last_render_time;
    double dt; // in seconds
    float max_fps;

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

int get_cell(Tactical* env, int row, int col) {
    if (row < 0 || row >= env->map_height) return -1;
    if (col < 0 || col >= env->map_width) return -1;
    return row * env->map_width + col;
}
int get_row(Tactical* env, int cell) {
    return cell / env->map_width;
}
int get_col(Tactical* env, int cell) {
    return cell % env->map_width;
}
int get_cell_with_delta(Tactical* env, int cell, int delta_row, int delta_col) {
    return get_cell(env, get_row(env, cell) + delta_row, get_col(env, cell) + delta_col);
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
    spell->effect(env, caster, target_cell, spell);
    spell->animation_state = 0;
    caster->action_points_current -= spell->ap_cost;
    spell->remaining_cooldown = spell->cooldown;
}

void spell_explosive_arrow(Tactical* env, Entity* caster, int target_cell, Spell* spell) {
    for (int delta_row = -spell->aoe_range; delta_row <= spell->aoe_range; ++delta_row) {
        for (int delta_col = -(spell->aoe_range - abs(delta_row)); delta_col <= spell->aoe_range - abs(delta_row); ++delta_col) {
            int cell = get_cell_with_delta(env, target_cell, delta_row, delta_col);
            if (env->map[cell] != CELL_GROUND) continue;
            Entity* target = env->cell_to_entity[cell];
            if (target) {
                target->health_points_current -= spell->damage;
            }
        }
    }
}
bool spell_explosive_arrow_anim(Tactical* env, Client* client, int caster_cell, int target_cell, float t, Spell* spell) {
    float xe0 = client->xe[caster_cell];
    float xe1 = client->xe[target_cell];
    float ye0 = client->ye[caster_cell];
    float ye1 = client->ye[target_cell];

    float phase1_duration = 0.5;
    float phase2_duration = 0.2;

    Vector2 vec = GetSplinePointBezierQuad(
        (Vector2){xe0, ye0 - 2 * client->ch},
        (Vector2){(xe0 + xe1) / 2, (ye0 + ye1) / 2 - 200},
        (Vector2){xe1, ye1},
        fmin(t / phase1_duration, 1.0));

    if (t <= phase1_duration) {
        DrawCircle(vec.x, vec.y, 10, (Color){255, 0, 0, 255});
    } else if (t <= phase1_duration + phase2_duration) {
        if (spell->animation_state == 0) {
            spell->animation_state = 1;
            // get all players hit
            for (int delta_row = -spell->aoe_range; delta_row <= spell->aoe_range; ++delta_row) {
                for (int delta_col = -(spell->aoe_range - abs(delta_row)); delta_col <= spell->aoe_range - abs(delta_row); ++delta_col) {
                    int cell = get_cell_with_delta(env, target_cell, delta_row, delta_col);
                    if (env->map[cell] != CELL_GROUND) continue;
                    Entity* target = env->cell_to_entity[cell];
                    if (target) {
                        add_animation_text(env, client, TextFormat("-%i HP", spell->damage),
                            cell, COLOR_HEALTH, 20, 1.2);
                    }
                }
            }
        }
        DrawCircle(vec.x, vec.y, 10 + (t - phase1_duration) * 400,
            (Color){255, 0, 0, 255 * (1 - (t - phase1_duration) / phase2_duration)});
    } else {
        return true;
    }
    return false;
}
Spell create_spell_explosive_arrow() {
    Spell spell;
    spell.name = "Explosive Arrow";
    spell.ap_cost = 4;
    spell.cooldown = 0;
    spell.remaining_cooldown = 0;
    spell.line_of_sight = true;
    spell.cast_in_line = false;
    spell.range = 11;
    spell.damage = 200;
    spell.aoe_range = 2;
    spell.effect = spell_explosive_arrow;
    spell.render_animation = spell_explosive_arrow_anim;
    return spell;
}

void spell_flying_arrow(Tactical* env, Entity* caster, int target_cell, Spell* spell) {
    Entity* target = env->cell_to_entity[target_cell];
    if (target) {
        target->health_points_current -= spell->damage;
    }
}
bool spell_flying_arrow_anim(Tactical* env, Client* client, int caster_cell, int target_cell, float t, Spell* spell) {
    float xe0 = client->xe[caster_cell];
    float xe1 = client->xe[target_cell];
    float ye0 = client->ye[caster_cell];
    float ye1 = client->ye[target_cell];

    float phase1_duration = 0.8;

    Vector2 vec = GetSplinePointBezierQuad(
        (Vector2){xe0, ye0 - 2 * client->ch},
        (Vector2){(xe0 + xe1) / 2, (ye0 + ye1) / 2 - 700},
        (Vector2){xe1, ye1},
        fmin(t / phase1_duration, 1.0));

    if (t <= phase1_duration) {
        DrawCircle(vec.x, vec.y, 10, (Color){0, 255, 0, 255});
    } else {                    
        Entity* target = env->cell_to_entity[target_cell];
        if (target) {
            add_animation_text(env, client, TextFormat("-%i HP", spell->damage),
                target_cell, COLOR_HEALTH, 20, 1.2);
        }
        return true;
    }
    return false;
}
Spell create_spell_flying_arrow() {
    Spell spell;
    spell.name = "Flying Arrow";
    spell.ap_cost = 3;
    spell.cooldown = 0;
    spell.remaining_cooldown = 0;
    spell.line_of_sight = false;
    spell.cast_in_line = false;
    spell.range = 14;
    spell.damage = 100;
    spell.aoe_range = 0;
    spell.effect = spell_flying_arrow;
    spell.render_animation = spell_flying_arrow_anim;
    return spell;
}

void spell_rooting_arrow(Tactical* env, Entity* caster, int target_cell, Spell* spell) {
    Entity* target = env->cell_to_entity[target_cell];
    if (target) {
        target->movement_points_current -= 3;
        if (target == env->current_player)
            compute_movement(env, target);
    }
}
bool spell_rooting_arrow_anim(Tactical* env, Client* client, int caster_cell, int target_cell, float t, Spell* spell) {
    float xe0 = client->xe[caster_cell];
    float xe1 = client->xe[target_cell];
    float ye0 = client->ye[caster_cell];
    float ye1 = client->ye[target_cell];

    float phase1_duration = 0.3;
    float phase2_duration = 0.3;

    if (t <= phase1_duration) {
        float progress = t / phase1_duration;
        DrawLineEx(
            (Vector2){xe0, ye0},
            (Vector2){(1-progress)*xe0 + progress*xe1, (1-progress)*ye0 + progress*ye1},
            4, (Color){200, 100, 0, 255});
    } else if (t <= phase1_duration + phase2_duration) {
        if (spell->animation_state == 0) {
            spell->animation_state = 1;
            // get all players hit
            Entity* target = env->cell_to_entity[target_cell];
            if (target) {
                add_animation_text(env, client, "-3 MP", target_cell, COLOR_MOVEMENT_POINTS, 20, 1.2);
            }
        }
        DrawLineEx((Vector2){xe0, ye0}, (Vector2){xe1, ye1}, 4, (Color){200, 100, 0, 255});
        float progress = (t - phase1_duration) / phase2_duration;
        for (int i = -3; i <= 3; ++i) {
            float angle = -M_PI/2 + i * M_PI/12;
            float new_x = xe1 + cos(angle) * 200;
            float new_y = ye1 + -sin(angle) * 200;

            DrawLineEx(
                (Vector2){xe1, ye1},
                (Vector2){(1-progress)*xe1 + progress*new_x, (1-progress)*ye1 + progress*new_y},
                4, (Color){200, 100, 0, 255});
        }
    } else {
        return true;
    }
    return false;
}
Spell create_spell_rooting_arrow() {
    Spell spell;
    spell.name = "Rooting Arrow";
    spell.ap_cost = 2;
    spell.cooldown = 2;
    spell.remaining_cooldown = 0;
    spell.line_of_sight = true;
    spell.cast_in_line = false;
    spell.range = 8;
    spell.damage = 0;
    spell.aoe_range = 0;
    spell.effect = spell_rooting_arrow;
    spell.render_animation = spell_rooting_arrow_anim;
    return spell;
}

void spell_wind_arrow(Tactical* env, Entity* caster, int target_cell, Spell* spell) {
    Entity* target = env->cell_to_entity[target_cell];
    if (target) {
        target->health_points_current -= spell->damage;
    }
}
bool spell_wind_arrow_anim(Tactical* env, Client* client, int caster_cell, int target_cell, float t, Spell* spell) {
    float xe0 = client->xe[caster_cell];
    float xe1 = client->xe[target_cell];
    float ye0 = client->ye[caster_cell];
    float ye1 = client->ye[target_cell];

    float phase1_duration = 0.4;

    if (t <= phase1_duration) {
        float progress = t / phase1_duration;
        DrawLineEx((Vector2){xe0, ye0-client->ch}, (Vector2){xe1, ye1-client->ch}, 4,
            (Color){50, 200, 50, 125 + 125 * sin(progress * 10)});
    } else {                    
        Entity* target = env->cell_to_entity[target_cell];
        if (target) {
            add_animation_text(env, client, TextFormat("-%i HP", spell->damage),
                target_cell, COLOR_HEALTH, 20, 1.2);
        }
        return true;
    }
    return false;
}
Spell create_spell_wind_arrow() {
    Spell spell;
    spell.name = "Wind Arrow";
    spell.ap_cost = 4;
    spell.cooldown = 0;
    spell.remaining_cooldown = 0;
    spell.line_of_sight = true;
    spell.cast_in_line = true;
    spell.range = 999;
    spell.damage = 400;
    spell.aoe_range = 0;
    spell.effect = spell_wind_arrow;
    spell.render_animation = spell_wind_arrow_anim;
    return spell;
}

void spell_swift_rabbit(Tactical* env, Entity* caster, int target_cell, Spell* spell) {
    Entity* target = env->cell_to_entity[target_cell];
    if (target) {
        target->movement_points_current += 5;
        compute_movement(env, target);
    }
}
bool spell_swift_rabbit_anim(Tactical* env, Client* client, int caster_cell, int target_cell, float t, Spell* spell) {  
    Entity* target = env->cell_to_entity[target_cell];
    if (target && spell->animation_state == 0) {
        add_animation_text(env, client, "+5 MP",
            target_cell, COLOR_MOVEMENT_POINTS, 20, 1.2);
    }
    spell->animation_state = 1;

    float phase1_duration = 0.3;

    if (t <= phase1_duration) {
        DrawCircle(client->xe[caster_cell], client->ye[caster_cell], t * 3000, (Color){0, 255, 0, 255 - t/0.3*255});
    } else {  
        return true;
    }
    return false;
}
Spell create_spell_swift_rabbit() {
    Spell spell;
    spell.name = "Swift Rabbit";
    spell.ap_cost = 2;
    spell.cooldown = 4;
    spell.remaining_cooldown = 0;
    spell.line_of_sight = true;
    spell.cast_in_line = false;
    spell.range = 0;
    spell.damage = 0;
    spell.aoe_range = 0;
    spell.effect = spell_swift_rabbit;
    spell.render_animation = spell_swift_rabbit_anim;
    return spell;
}


void assign_spells(Entity* entity) {
    // TODO assign different spells based on class
    entity->spell_count = 5;
    entity->spells = malloc(entity->spell_count * sizeof(Spell));
    entity->spells[0] = create_spell_explosive_arrow();
    entity->spells[1] = create_spell_flying_arrow();
    entity->spells[2] = create_spell_rooting_arrow();
    entity->spells[3] = create_spell_wind_arrow();
    entity->spells[4] = create_spell_swift_rabbit();
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
    free(walkable_cells);
}

void move_entity(Tactical* env, Entity* entity, const int cell) {
    env->cell_to_entity[entity->cell] = NULL;
    entity->cell = cell;
    env->cell_to_entity[entity->cell] = entity;
    entity->movement_points_current -= env->movement_distance[cell];
    compute_movement(env, entity);
}

bool try_move_entity(Tactical* env, Entity* entity, const int cell) {
    // TODO i don't like this. Checks should be in game logic, not client.
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
    env->player1->cell = get_cell(env, 17, 13);
    env->player1->color = COLOR_PLAYER1;
    env->player1->health_points_total = 2500;
    env->player1->action_points_total = 12;
    env->player1->movement_points_total = 6;
    env->player1->health_points_current = 2500;
    env->player1->action_points_current = 12;
    env->player1->movement_points_current = 6;

    env->player2->name = "Player 2";
    env->player2->cell = get_cell(env, 12, 13);
    env->player2->color = COLOR_PLAYER2;
    env->player2->health_points_total = 2500;
    env->player2->action_points_total = 12;
    env->player2->movement_points_total = 6;
    env->player2->health_points_current = 2500;
    env->player2->action_points_current = 12;
    env->player2->movement_points_current = 6;

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

void c_reset(Tactical* env) {
    compute_observations(env);
}

int c_step(Tactical* env) {
    if (false) {
        c_reset(env);
        int winner = 2;
        return winner;
    }

    compute_observations(env);
    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// RENDERING ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////


Client* init_client(Tactical* env) {
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->width = 1200;
    client->height = 900;

    client->movement_cells = malloc(env->map_size * sizeof(bool));
    client->spell_cells = malloc(env->map_size * sizeof(bool));
    client->active_spell_cells = malloc(env->map_size * sizeof(bool));
    client->active_spell = NULL;

    // TODO fill the screen automatically (these are hardcoded for map 2)
    client->cw = 80;
    client->ch = client->cw / 2;
    client->offset_x = 560;
    client->offset_y = -200;
    client->dy = client->ch * 0.4;

    client->mcell = -1;
    client->mcell_type = -1;

    client->move_anim_path = calloc(env->map_size, sizeof(int));
    client->move_anim_cells_per_second = 6;

    client->text_anim_count = 0;

    client->spell_anim = NULL;
    
    client->xa = calloc(env->map_size, sizeof(float));
    client->xb = calloc(env->map_size, sizeof(float));
    client->xc = calloc(env->map_size, sizeof(float));
    client->xd = calloc(env->map_size, sizeof(float));
    client->xe = calloc(env->map_size, sizeof(float));
    client->ya = calloc(env->map_size, sizeof(float));
    client->yb = calloc(env->map_size, sizeof(float));
    client->yc = calloc(env->map_size, sizeof(float));
    client->yd = calloc(env->map_size, sizeof(float));
    client->ye = calloc(env->map_size, sizeof(float));
    for (int row = 0; row < env->map_height; ++row) {
        for (int col = 0; col < env->map_width; ++col) {
            int cell = get_cell(env, row, col);
            client->xa[cell] = client->offset_x + 0.5 * client->cw * (col - row);
            client->xb[cell] = client->xa[cell] - client->cw / 2;
            client->xc[cell] = client->xa[cell] + client->cw / 2;
            client->xd[cell] = client->xa[cell];
            client->xe[cell] = client->xa[cell];
            client->ya[cell] = client->offset_y + 0.5 * client->ch * (col + row + 2);
            client->yb[cell] = client->ya[cell] + client->ch / 2;
            client->yc[cell] = client->ya[cell] + client->ch / 2;
            client->yd[cell] = client->ya[cell] + client->ch;
            client->ye[cell] = client->yb[cell];
        }
    }

    client->last_render_time = clock();
    client->dt = 0.0f;
    client->max_fps = 120;

    InitWindow(client->width, client->height, "Tactical RL");
    SetTargetFPS(60);

    return client;
}

int get_cell_at_cursor(Client* client, Tactical* env) {
    // to get the formula: we know that cell (row, col) starts at coordinates
    //     x = offset_x + 0.5 * cw * (col - row);
    //     y = offset_y + 0.5 * ch * (col + row + 2);
    // solve this 2x2 linear system to write (row, col)) as a function of (x, y) and we get the formulas below
    const int mx = GetMouseX();
    const int my = GetMouseY();
    const int mrow = floor((my - client->offset_y) / client->ch - (mx - client->offset_x) / client->cw - 1);
    const int mcol = floor((my - client->offset_y) / client->ch + (mx - client->offset_x) / client->cw - 1);
    client->mx = mx;
    client->my = my;
    client->mrow = mrow;
    client->mcol = mcol;
    const int mcell = (mrow < 0 || mcol < 0 || mrow >= env->map_height || mcol >= env->map_width) ? -1 : get_cell(env, mrow, mcol);
    return mcell;
}

void draw_debug_info(Client* client) {
    //DrawText(TextFormat("%i FPS", (int)(1 / client->dt)), 10, 10, 20, COLOR_TEXT_DEFAULT);
    DrawText(TextFormat("Mouse: %i, %i", client->mx, client->my), 150, 10, 15, COLOR_TEXT_DEFAULT);
    DrawText(TextFormat("Cell: %i (row %i, col %i)", client->mcell, client->mrow, client->mcol), 150, 30, 15, COLOR_TEXT_DEFAULT);
    DrawText(TextFormat("Cell type: %s",
        client->mcell_type == CELL_EMPTY ? "EMPTY" :
        client->mcell_type == CELL_GROUND ? "GROUND" :
        client->mcell_type == CELL_HOLE ? "HOLE" :
        client->mcell_type == CELL_WALL ? "WALL" : 
        client->mcell_type == -1 ? "NONE" : "UNKNOWN"), 150, 45, 15, COLOR_TEXT_DEFAULT);
}

void draw_player(Client* client, Entity* player, float x, float y, Color color, bool is_current_player, Color cell_color) {
    // draw the little guy
    if (is_current_player) DrawEllipse(x, y, 0.33 * client->cw, 0.33 * client->ch, COLOR_ACTIVE_PLAYER);
    DrawEllipse(x, y, 0.3 * client->cw, 0.3 * client->ch, color);
    DrawEllipse(x, y, 0.23 * client->cw, 0.23 * client->ch, cell_color);
    DrawLineEx((Vector2){x - 0.1 * client->cw, y}, (Vector2){x, y - 0.7 * client->ch}, 2, color);
    DrawLineEx((Vector2){x + 0.1 * client->cw, y}, (Vector2){x, y - 0.7 * client->ch}, 2, color);
    DrawLineEx((Vector2){x, y - 0.7 * client->ch}, (Vector2){x, y - 1.1 * client->ch}, 2, color);
    DrawLineEx((Vector2){x, y - 1.0 * client->ch}, (Vector2){x - 0.2 * client->cw, y - 0.8 * client->ch}, 2, color);
    DrawLineEx((Vector2){x, y - 1.0 * client->ch}, (Vector2){x + 0.2 * client->cw, y - 0.8 * client->ch}, 2, color);
    DrawCircle(x, y - 1.3 * client->ch, 0.2 * client->ch, color);

    // draw hp, ap and mp above the little guy
    DrawText(TextFormat("%i", player->health_points_current),
             x - MeasureText(TextFormat("%i", player->health_points_current), 15) / 2, y - 2.0 * client->ch, 15, COLOR_HEALTH);
    DrawText(TextFormat("%i", player->action_points_current),
             x - MeasureText(TextFormat("%i", player->action_points_current), 15) - 4, y - 2.4 * client->ch, 15, COLOR_ACTION_POINTS);
    DrawText(TextFormat("%i", player->movement_points_current),
             x + 4, y - 2.4 * client->ch, 15, COLOR_MOVEMENT_POINTS);
}

void draw_cells_and_entities(Client* client, Tactical* env) {
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
            if (client->movement_cells[cell]) {
                cell_color = COLOR_CELL_MOVE;
            } else if (client->active_spell) {
                if (client->active_spell_cells[cell]) {
                    cell_color = COLOR_CELL_ACTIVE_SPELL;
                } else if (client->spell_cells[cell]) {
                    cell_color = COLOR_CELL_SPELL;
                } else {
                    cell_color = COLOR_CELL_INACTIVE_SPELL;
                }
            }
            // DrawTriangleStrip((Vector2[]){{xa, ya}, {xb, yb}, {xc, yc}, {xd, yd}}, 4, cell_color);
            DrawTriangleStrip((Vector2[]){
                {client->xa[cell], client->ya[cell]},
                {client->xb[cell], client->yb[cell]},
                {client->xc[cell], client->yc[cell]},
                {client->xd[cell], client->yd[cell]}}, 4, cell_color);
            if (client->movement_cells[cell]) {
                const unsigned int dist = env->movement_distance[cell];
                const char* text = TextFormat("%i", dist);
                DrawText(text, 
                    client->xe[cell] - MeasureText(text, 12) / 2,
                    client->ye[cell] - 6, 12, COLOR_CELL_MOVE_TEXT);
            }
            // draw white border around cell
            DrawLineStrip((Vector2[]){
                {client->xa[cell], client->ya[cell]},
                {client->xb[cell], client->yb[cell]},
                {client->xd[cell], client->yd[cell]},
                {client->xc[cell], client->yc[cell]},
                {client->xa[cell], client->ya[cell]}}, 5, COLOR_CELL_BORDER);
        }
    }

    // then draw walls and entities alternatively, from top-left to bottom-right, for correct z-order
    bool draw_horizontally = true;  // draw row by row, from top-left to bottom-right (if false: column by column)
    if (client->move_anim_entity) {
        int col = get_col(env, client->move_anim_path[client->move_anim_path_idx]);
        int col_next = get_col(env, client->move_anim_path[client->move_anim_path_idx + 1]);
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
                {client->xa[cell], client->ya[cell] - client->dy},
                {client->xb[cell], client->yb[cell] - client->dy},
                {client->xc[cell], client->yc[cell] - client->dy},
                {client->xd[cell], client->yd[cell] - client->dy}}, 4, COLOR_CELL_GRASS);
            // draw connections between (a, b, c, d) and the shifted up cell ("dirt")
            DrawTriangleStrip((Vector2[]){
                {client->xc[cell], client->yc[cell]},
                {client->xc[cell], client->yc[cell] - client->dy},
                {client->xd[cell], client->yd[cell]},
                {client->xd[cell], client->yd[cell] - client->dy},
                {client->xb[cell], client->yb[cell]},
                {client->xb[cell], client->yb[cell] - client->dy}}, 6, COLOR_CELL_DIRT);
        }

        // draw entity at cell (if any)
        Color cell_color = COLOR_CELL_GROUND;
        if (client->movement_cells[cell]) {
            cell_color = COLOR_CELL_MOVE;
        } else if (client->active_spell && client->spell_cells[cell]) {
            cell_color = cell == client->mcell ? COLOR_CELL_ACTIVE_SPELL : COLOR_CELL_SPELL;
        }
        Entity* entity = env->cell_to_entity[cell];
        if (entity && entity != client->move_anim_entity) {
            draw_player(client, entity,
                client->xe[cell], client->ye[cell],
                entity->color, entity == env->current_player, cell_color);
        }
        // if entity is under move animation, handle it differently
        if (client->move_anim_entity && client->move_anim_path[client->move_anim_path_idx] == cell) {
            draw_player(client, client->move_anim_entity,
                client->xe[cell] + client->move_anim_dx, client->ye[cell] + client->move_anim_dy, 
                client->move_anim_entity->color, client->move_anim_entity == env->current_player, cell_color);
        }
    }
}

void draw_player_dashboard(Client* client, Entity* dashboard_entity, bool is_current_player) {
    // Health, action points, movement points
    DrawText(dashboard_entity->name, 40, client->height - 150, 25,
        is_current_player ? COLOR_ENTITY_NAME : COLOR_ENTITY_NAME_HOVER);
    DrawText(TextFormat("HP: %i / %i", dashboard_entity->health_points_current, dashboard_entity->health_points_total),
             40, client->height - 120, 25, COLOR_HEALTH);
    DrawText(TextFormat("AP: %i / %i", dashboard_entity->action_points_current, dashboard_entity->action_points_total),
             40, client->height - 90, 25, COLOR_ACTION_POINTS);
    DrawText(TextFormat("MP: %i / %i", dashboard_entity->movement_points_current, dashboard_entity->movement_points_total),
             40, client->height - 60, 25, COLOR_MOVEMENT_POINTS);

    // Spells
    DrawText("Spells", 300, client->height - 150, 20, COLOR_TEXT_DEFAULT);
    for (int i = 0; i < dashboard_entity->spell_count; ++i) {
        Spell* spell = &dashboard_entity->spells[i];
        bool cooldown = spell->remaining_cooldown > 0;
        bool spell_active = !cooldown && dashboard_entity->action_points_current >= spell->ap_cost;
        DrawText(TextFormat("[%i]", i+1), 300, client->height - 125 + i * 20, 20, spell_active ? COLOR_SPELL : COLOR_SPELL_COOLDOWN);
        DrawText(TextFormat("(%i AP)", spell->ap_cost), 300 + 30, client->height - 125 + i * 20, 20, cooldown ? COLOR_SPELL_COOLDOWN : COLOR_ACTION_POINTS);
        if (spell->remaining_cooldown > 0) {
            DrawText(TextFormat("%s (cooldown: %i)", spell->name, spell->remaining_cooldown), 
                300 + 100, client->height - 125 + i * 20, 20, COLOR_SPELL_COOLDOWN);
        } else {
            DrawText(spell->name,
                300 + 100, client->height - 125 + i * 20, 20, spell_active ? COLOR_SPELL : COLOR_SPELL_COOLDOWN);
        }
    }
}

void update_animation_move(Tactical* env, Client* client) {
    client->move_anim_progress += client->dt * client->move_anim_cells_per_second;
    if (client->move_anim_progress >= 1) {
        client->move_anim_progress = fmod(client->move_anim_progress, 1);
        client->move_anim_path_idx += 1;
    }
    if (client->move_anim_path_idx == client->move_anim_path_length) {
        // reached last cell: stop animation
        client->move_anim_entity = NULL;
    } else {
        int current_cell = client->move_anim_path[client->move_anim_path_idx];
        int next_cell = client->move_anim_path[client->move_anim_path_idx + 1];
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
        client->move_anim_dx = client->move_anim_progress * move_dx * client->cw * 0.5;
        client->move_anim_dy = client->move_anim_progress * move_dy * client->ch * 0.5;
    }
}

void add_animation_text(Tactical* env, Client* client, const char* text, int cell, Color color, int font_size, float duration) {
    int idx = client->text_anim_count;
    if (idx < MAX_TEXT_ANIMATIONS) {
        client->text_anim_texts[idx] = malloc(strlen(text) + 1);
        strcpy(client->text_anim_texts[idx], text);

        client->text_anim_x0[idx] = client->xe[cell] - MeasureText(text, font_size) / 2;
        client->text_anim_y0[idx] = client->ye[cell] - 2.5 * client->ch;
        client->text_anim_x1[idx] = client->text_anim_x0[idx];
        client->text_anim_y1[idx] = client->text_anim_y0[idx] - 3 * client->ch;

        client->text_anim_progress[idx] = 0;
        client->text_anim_duration[idx] = duration;
        client->text_anim_color[idx] = color;
        client->text_anim_font_size[idx] = font_size;
        client->text_anim_count++;
    } else {
        printf("client->text_anim_texts array is full, cannot add more strings");
    }
}

void remove_animation_text(Client* client, int index) {
    if (index < 0 || index >= client->text_anim_count) {
        printf("Invalid index in remove_animation_text\n");
        return;
    }

    free(client->text_anim_texts[index]);

    // shift all strings after the removed index to the left
    for (int i = index; i < client->text_anim_count - 1; i++) {
        client->text_anim_texts[i] = client->text_anim_texts[i + 1];
        client->text_anim_x0[i] = client->text_anim_x0[i + 1];
        client->text_anim_x1[i] = client->text_anim_x1[i + 1];
        client->text_anim_y0[i] = client->text_anim_y0[i + 1];
        client->text_anim_y1[i] = client->text_anim_y1[i + 1];
        client->text_anim_progress[i] = client->text_anim_progress[i + 1];
        client->text_anim_duration[i] = client->text_anim_duration[i + 1];
        client->text_anim_color[i] = client->text_anim_color[i + 1];
        client->text_anim_font_size[i] = client->text_anim_font_size[i + 1];
    }
    client->text_anim_count--;
}

void update_animations_text(Client* client) {
    for (int i = client->text_anim_count - 1; i >= 0; --i) {
        // backward loop because of the way strings are shifted in remove_animation_text
        client->text_anim_progress[i] += client->dt / client->text_anim_duration[i];
        if (client->text_anim_progress[i] >= 1) {
            remove_animation_text(client, i);
        }
    }
}

void draw_animations_text(Client* client) {
    for (int i = 0; i < client->text_anim_count; ++i) {
        float t = client->text_anim_progress[i];
        float x = (1 - t) * client->text_anim_x0[i] + t * client->text_anim_x1[i];
        float y = (1 - t) * client->text_anim_y0[i] + t * client->text_anim_y1[i];
        Color color = client->text_anim_color[i];
        color.a = (int)((1 - t) * 255);
        DrawText(client->text_anim_texts[i], x, y, client->text_anim_font_size[i], color);
    }
}

void update_animation_spell(Client* client) {
    client->spell_anim_progress += client->dt;
}

void draw_animation_spell(Tactical* env, Client* client) {
    if (client->spell_anim) {
        bool finished = client->spell_anim->render_animation(
            env,
            client,
            client->spell_anim_caster_cell,
            client->spell_anim_target_cell,
            client->spell_anim_progress,
            client->spell_anim
        );
        if (finished) {
            client->spell_anim = NULL;
        }
    }
}

int c_render(Tactical* env) {
    if (IsKeyDown(KEY_Q) || IsKeyDown(KEY_BACKSPACE)) {
        return 1;  // close window
    }

    if (env->client == NULL) {
        env->client = init_client(env);
    }

    Client* client = env->client;
    // cap FPS and compute dt
    /*
    clock_t current_time;
    do {
        current_time = clock();
        client->dt = (double)(current_time - client->last_render_time) / CLOCKS_PER_SEC;
    } while (client->dt < 1 / client->max_fps);
    client->last_render_time = current_time;
    */
    client->dt = 1.0 / 60.0;

    BeginDrawing();
    ClearBackground(COLOR_BACKGROUND);

    int cursor = MOUSE_CURSOR_DEFAULT;

    // get current cell at cursor position (if any), and draw debug info
    client->mcell = get_cell_at_cursor(client, env);
    client->mcell_type = client->mcell == -1 ? -1 : env->map[client->mcell];
    draw_debug_info(client);
    
    const int mcell = client->mcell;
    // movement path display, if applicable ; and spells
    memset(client->movement_cells, 0, env->map_size * sizeof(bool));
    if (client->spell_anim) {
        update_animation_spell(client);
    } else if (client->active_spell) {
        if (mcell != -1 && client->spell_cells[mcell]) {
            cursor = MOUSE_CURSOR_POINTING_HAND;
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                cast_spell(env, env->current_player, client->active_spell, mcell);
                add_animation_text(env, client, TextFormat("-%i AP", client->active_spell->ap_cost),
                    env->current_player->cell, COLOR_ACTION_POINTS, 20, 1.2);
                client->spell_anim = client->active_spell;
                client->spell_anim_caster_cell = env->current_player->cell;
                client->spell_anim_target_cell = mcell;
                client->spell_anim_progress = 0.0f;
                client->active_spell = NULL;
            }
        } else {
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                client->active_spell = NULL;
            }
        }
    } else {
        if (mcell != -1 && env->movement_path[mcell] && !client->move_anim_entity) {
            int cell = mcell;
            int path_length = env->movement_distance[mcell];
            for (int i = path_length; i >= 0; --i) {
                if (i != 0) {
                    client->movement_cells[cell] = true;
                }
                client->move_anim_path[i] = cell; // precompute in case it's used, cause after moving the env->movement_path is no longer valid
                cell = env->movement_path[cell];
            }
            cursor = MOUSE_CURSOR_POINTING_HAND;
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                add_animation_text(env, client, TextFormat("-%i MP", path_length),
                    env->current_player->cell, COLOR_MOVEMENT_POINTS, 20, 1.2);

                if (try_move_entity(env, env->current_player, mcell)) {
                    // start move animation
                    client->move_anim_entity = env->current_player;
                    client->move_anim_path_idx = 0;
                    client->move_anim_path_length = path_length;
                    client->move_anim_progress = 0;
                    client->move_anim_dx = 0;
                    client->move_anim_dy = 0;
                }
            }
        }
    }

    if (client->move_anim_entity) {
        update_animation_move(env, client);
    }
    update_animations_text(client);

    // KEYS

    if (IsKeyPressed(KEY_SPACE)) {
        client->active_spell = NULL;
        next_player(env);
    }

    int tentative_spell_id = -1;
    if (IsKeyPressed(KEY_ONE)) tentative_spell_id = 0;
    else if (IsKeyPressed(KEY_TWO)) tentative_spell_id = 1;
    else if (IsKeyPressed(KEY_THREE)) tentative_spell_id = 2;
    else if (IsKeyPressed(KEY_FOUR)) tentative_spell_id = 3;
    else if (IsKeyPressed(KEY_FIVE)) tentative_spell_id = 4;

    if (tentative_spell_id >= 0 && tentative_spell_id < env->current_player->spell_count) {
        Spell* spell = &env->current_player->spells[tentative_spell_id];
        if (spell->remaining_cooldown == 0 && env->current_player->action_points_current >= spell->ap_cost) {
            client->active_spell = spell;

            // COMPUTE LINES OF SIGHT (this should be in Env, not client)
            // set all ground cells to 1 by default (within castable range of the spell)
            memset(client->spell_cells, 0, env->map_size * sizeof(bool));
            for (int delta_row = -spell->range; delta_row <= spell->range; ++delta_row) {
                for (int delta_col = -(spell->range - abs(delta_row)); delta_col <= spell->range - abs(delta_row); ++delta_col) {
                    int cell = get_cell_with_delta(env, env->current_player->cell, delta_row, delta_col);
                    if (client->active_spell->cast_in_line && delta_row != 0 && delta_col != 0) continue;
                    if (cell != -1 && env->map[cell] == CELL_GROUND) {
                        client->spell_cells[cell] = true;
                    }
                }
            }

            if (spell->line_of_sight) {
                // Bresenham line algorithm
                for (int i = 0; i < env->map_size; ++i) {
                    if (!client->spell_cells[i]) continue;
                    int x0 = get_col(env, env->current_player->cell);
                    int y0 = get_row(env, env->current_player->cell);
                    int x1 = get_col(env, i);
                    int y1 = get_row(env, i);
                    int dx = x1 - x0;
                    int dy = y1 - y0;
                    int nx = abs(dx);
                    int ny = abs(dy);
                    int sign_x = dx > 0 ? 1 : -1;
                    int sign_y = dy > 0 ? 1 : -1;
                    int px = x0;
                    int py = y0;
                    int ix = 0;
                    int iy = 0;
                    while (ix < nx || iy < ny) {
                        int new_cell = get_cell(env, py, px);
                        if (new_cell != -1 && new_cell != env->current_player->cell && (env->map[new_cell] == CELL_WALL || env->cell_to_entity[new_cell] != NULL)) {
                            client->spell_cells[i] = false;
                            break;
                        }
                        int decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx;
                        if (decision == 0) {
                            // next step is diagonal
                            px += sign_x;
                            py += sign_y;
                            ix += 1;
                            iy += 1;
                        } else if (decision < 0) {
                            // next step is horizontal
                            px += sign_x;
                            ix += 1;
                        } else {
                            // next step is vertical
                            py += sign_y;
                            iy += 1;
                        }
                    }
                }
            }
        }
    }        
    memset(client->active_spell_cells, 0, env->map_size * sizeof(bool));
    if (client->active_spell && client->spell_cells[mcell]) {
        Spell* spell = client->active_spell;
        for (int delta_row = -spell->aoe_range; delta_row <= spell->aoe_range; ++delta_row) {
            for (int delta_col = -(spell->aoe_range - abs(delta_row)); delta_col <= spell->aoe_range - abs(delta_row); ++delta_col) {
                int cell = get_cell_with_delta(env, mcell, delta_row, delta_col);
                if (env->map[cell] == CELL_GROUND) {
                    client->active_spell_cells[cell] = true;
                }                
            }
        }
    }

    if (IsKeyPressed(KEY_ESCAPE)) {
        client->active_spell = NULL;
    }

    draw_cells_and_entities(client, env);
    draw_animation_spell(env, client);
    draw_animations_text(client);

    // Write info about keys
    DrawText("Press Q or Backspace to exit", 600, 10, 15, COLOR_TEXT_DEFAULT);
    DrawText("Press the corresponding key [1-5] to cast a spell", 600, 25, 15, COLOR_TEXT_DEFAULT);
    DrawText("Press Space to skip turn", 600, 40, 15, COLOR_TEXT_DEFAULT);

    // Draw player dashboard (health, action points, movement points, spells)
    Entity* dashboard_entity = env->current_player;
    if (client->mcell != -1 && env->cell_to_entity[client->mcell] && env->cell_to_entity[client->mcell] != env->current_player) {
        dashboard_entity = env->cell_to_entity[mcell];
        cursor = MOUSE_CURSOR_POINTING_HAND;
    }
    draw_player_dashboard(client, dashboard_entity, dashboard_entity == env->current_player);

    SetMouseCursor(cursor);

    EndDrawing();
    return 0;
}

void close_client(Client* client) {
    CloseWindow();
    free(client->movement_cells);
    free(client);
}
