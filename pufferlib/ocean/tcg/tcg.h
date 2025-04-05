#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "raylib.h"

#define HAND_SIZE 10
#define BOARD_SIZE 10
#define LAND_SIZE 10
#define DECK_SIZE 60
#define GRAVEYARD_SIZE 60
#define STACK_SIZE 100
#define MAX_LISTENERS 100

#define LAND_ZONE_OFFSET 128
#define GRAVEYARD_ZONE_OFFSET 128

#define ACTION_ENTER 10
#define ACTION_NOOP 11
#define ACTION_SPACE 12

#define TO_USER true
#define TO_STACK false

#define LAND_COLOR GREEN
#define CREATURE_COLOR RED
#define INSTANT_COLOR BLUE
#define SORCERY_COLOR SKYBLUE
#define ARTIFACT_COLOR BLUE
#define ENCHANTMENT_COLOR BLUE
#define PLANESWALKER_COLOR BLUE
#define GRAVEYARD_COLOR GREY
#define TAPPED_COLOR LIGHTGRAY
#define SUMMONING_SICKNESS_COLOR ORANGE

typedef struct TCG TCG;
typedef struct Player Player;
typedef struct Card Card;
typedef struct Event Event;
typedef struct EventListener EventListener;
typedef struct EventManager EventManager;
typedef struct Effect Effect;
typedef struct Stack Stack;
typedef struct StackItem StackItem;
typedef struct CardArray CardArray;
typedef void (*EventHandler)(TCG*, Event*);
typedef bool (*phase_fn)(TCG*, unsigned char);
typedef bool (*target_fn)(TCG*, unsigned char, CardArray*);

bool phase_untap(TCG* env, unsigned char atn);
bool phase_upkeep(TCG* env, unsigned char atn);
bool phase_draw(TCG* env, unsigned char atn);
bool phase_play(TCG* env, unsigned char atn);
bool phase_attack(TCG* env, unsigned char atn);
bool phase_block(TCG* env, unsigned char atn);
bool phase_priority(TCG* env, unsigned char atn);
void reset(TCG* env);

typedef enum {
    STACK_PHASE,
    STACK_EFFECT,
} StackType;

typedef enum {
    TYPE_LAND,
    TYPE_CREATURE,
    TYPE_INSTANT,
    TYPE_SORCERY,
    TYPE_ARTIFACT,
    TYPE_ENCHANTMENT,
    TYPE_PLANESWALKER
} CardType;

typedef enum {
    ZONE_HAND,
    ZONE_BOARD,
    ZONE_LANDS,
    ZONE_DECK,
    ZONE_GRAVEYARD
} Zone;

typedef enum {
    EVENT_CARD_PLAYED,
    EVENT_CARD_DRAWN,
    EVENT_CARD_TAPPED,
    EVENT_CARD_DESTROYED,
    EVENT_CARD_ATTACKED,
    EVENT_CARD_BLOCKED,
    EVENT_PLAYER_HEALTH_CHANGED,
    EVENT_PLAYER_MANA_CHANGED,
    EVENT_PLAYER_LAND_CHANGED,
    EVENT_PLAYER_GRAVEYARD_CHANGED,
} EventType;

struct Event {
    EventType type;
    void *data;
};

struct EventListener {
    EventType type;
    EventHandler handler;
};

struct EventManager {
    EventListener listeners[MAX_LISTENERS];
    int count;
};

void addListener(EventManager *mgr, EventType type, EventHandler handler) {
    if (mgr->count < MAX_LISTENERS) {
        mgr->listeners[mgr->count].type = type;
        mgr->listeners[mgr->count].handler = handler;
        mgr->count++;
    }
    else {
        printf("Max listeners reached\n");
    }
}

void publishEvent(TCG* env, EventManager *mgr, Event *event) {
    for (int i = 0; i < mgr->count; i++) {
        if (mgr->listeners[i].type == event->type) {
            mgr->listeners[i].handler(env, event);
        }
    }
}

void onCardPlayed(TCG* env, Event *event) {
    printf("Card played event triggered.\n");
}

struct Effect {
    char description[256];
    void (*trigger)(TCG* env, Event* event);
    bool (*condition)(TCG* env);
    bool (*target_fn)(TCG* env, unsigned char atn);
    void (*activate)(TCG* env, CardArray* targets);
    EventType event;
};

struct StackItem {
    StackType type;
    union {
        phase_fn phase;
        Effect effect;
    };
};

struct Stack {
    StackItem data[STACK_SIZE];
    int idx;
};

void push(Stack* stack, StackItem item) {
    assert(stack->idx < STACK_SIZE);
    stack->data[stack->idx] = item;
    stack->idx += 1;
}

StackItem pop(Stack* stack) {
    assert(stack->idx > 0);
    stack->idx -= 1;
    return stack->data[stack->idx];
}

StackItem peek(Stack* stack) {
    assert(stack->idx > 0);
    return stack->data[stack->idx - 1];
}

struct Card {
    char name[256];
    CardType type;
    Effect effect;
    Color color;
    int attack;
    int health;
    int defending;
    int cost;
    bool tapped;
    bool summoning_sickness;
    bool attacking;
};

void print_card(Card* card) {
    printf("Name: %s\n", card->name);
    printf("Type: %i\n", card->type);
    printf("Cost: %i\n", card->cost);
    printf("Attack: %i\n", card->attack);
    printf("Health: %i\n", card->health);
}

struct CardArray {
    Card** cards;
    int length;
    int max;
};


void add_card_to(CardArray* to, Card* card) {
    assert(to->length < to->max);
    to->cards[to->length] = card;
    to->length += 1;
}

void remove_card_from(CardArray* from, int idx) {
    for (int i = idx; i < from->length - 1; i++) {
        from->cards[i] = from->cards[i + 1];
    }
    from->length -= 1;
}

void move_card(CardArray* from, CardArray* to, int idx) {
    Card* card = from->cards[idx];
    remove_card_from(from, idx);
    add_card_to(to, card);
}

Card* allocate_creature(char* name, int cost, int attack, int health, Effect effect) {
    Card* card = (Card*)calloc(1, sizeof(Card));
    strncpy(card->name, name, sizeof(card->name) - 1);
    card->type = TYPE_CREATURE;
    card->cost = cost;
    card->tapped = false;
    card->color = CREATURE_COLOR;
    card->effect = effect;
    card->attack = attack;
    card->health = health;
    card->defending = -1;
    card->summoning_sickness = true;
    card->attacking = false;
    return card;
}

Card* allocate_instant(char* name, int cost, Effect effect) {
    Card* card = (Card*)calloc(1, sizeof(Card));
    strncpy(card->name, name, sizeof(card->name) - 1);
    card->type = TYPE_INSTANT;
    card->cost = cost;
    card->tapped = false;
    card->color = INSTANT_COLOR;
    card->effect = effect;
    return card;
}

Card* allocate_sorcery(char* name, int cost, Effect effect) {
    Card* card = (Card*)calloc(1, sizeof(Card));
    strncpy(card->name, name, sizeof(card->name) - 1);
    card->type = TYPE_SORCERY;
    card->cost = cost;
    card->tapped = false;
    card->color = RED;
    card->effect = effect;
    return card;
}

Card* allocate_land(char* name) {
    Card* card = (Card*)calloc(1, sizeof(Card));
    strncpy(card->name, name, sizeof(card->name) - 1);
    card->type = TYPE_LAND;
    card->cost = 0;
    card->tapped = false;
    card->color = LAND_COLOR;
    return card;
}

CardArray* allocate_card_array(int max) {
    CardArray* zone = (CardArray*)calloc(1, sizeof(CardArray));
    zone->cards = (Card**)calloc(max, sizeof(Card*));
    zone->max = max;
    return zone;
}

void free_card_array(CardArray* array) {
    free(array->cards);
    free(array);
}

struct Player {
    CardArray* hand;
    CardArray* board;
    CardArray* lands;
    CardArray* deck;
    CardArray* graveyard;
    int health;
    int mana;
    bool land_played;
};

struct TCG {
    Player* my_player;
    Player* op_player;

    Stack* stack;
    EventManager* event_manager;
    int block_idx;
    int turn;
    int priority;
    bool priority_passed;
    bool participate_in_priority;
    int target_idx;
    CardArray* targets;
};


void no_trigger(TCG* env, Event* event) {}



bool no_condition(TCG* env) { return true; }

CardArray* no_target(TCG* env) { return NULL; }

bool single_target_hand(TCG* env, unsigned char atn) {
    printf("single_target_hand\n");
    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_hand});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Manual confirm\n");
        env->targets->cards[env->target_idx] = env->my_player->hand->cards[env->target_idx];
        return TO_STACK;
    } else if (atn >= env->my_player->hand->length) {
        printf("\t Invalid action %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_hand});
        return TO_USER;
    }
    env->target_idx = atn;
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_hand});
    return TO_USER;
}

bool single_target_op_hand(TCG* env, unsigned char atn) {
    printf("single_target_hand\n");
    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_hand});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Manual confirm\n");
        env->targets->cards[env->target_idx] = env->op_player->hand->cards[env->target_idx];
        return TO_STACK;
    } else if (atn >= env->my_player->hand->length) {
        printf("\t Invalid action %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_hand});
        return TO_USER;
    }
    env->target_idx = atn;
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_hand});
    return TO_USER;
}

bool single_target_board(TCG* env, unsigned char atn) {
    printf("single_target_board\n");
    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_board});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Manual confirm\n");
        env->targets->cards[env->target_idx] = env->my_player->board->cards[env->target_idx];
        return TO_STACK;
    } else if (atn >= env->my_player->board->length) {
        printf("\t Invalid action %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_board});
        return TO_USER;
    }
    env->target_idx = atn;
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_board});
    return TO_USER;
}

bool single_target_op_board(TCG* env, unsigned char atn) {
    printf("single_target_board\n");
    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_board});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Manual confirm\n");
        env->targets->cards[env->target_idx] = env->op_player->board->cards[env->target_idx];
        return TO_STACK;
    } else if (atn >= env->my_player->board->length) {
        printf("\t Invalid action %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_board});
        return TO_USER;
    }
    env->target_idx = atn;
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = single_target_board});
    return TO_USER;
}


void no_activate(TCG* env, CardArray* targets) {}


Player* allocate_player() {
    Player* player = calloc(1, sizeof(Player));
    player->hand = allocate_card_array(HAND_SIZE);
    player->board = allocate_card_array(BOARD_SIZE);
    player->lands = allocate_card_array(LAND_SIZE);
    player->deck = allocate_card_array(DECK_SIZE);
    player->graveyard = allocate_card_array(GRAVEYARD_SIZE);
    return player;
}

void free_player(Player* player) {
    free_card_array(player->hand);
    free_card_array(player->board);
    free_card_array(player->lands);
    free_card_array(player->deck);
    free_card_array(player->graveyard);
    free(player);
}

void allocate_tcg(TCG* env) {
    env->stack = calloc(1, sizeof(Stack));
    env->event_manager = calloc(1, sizeof(EventManager));
    env->my_player = allocate_player();
    env->op_player = allocate_player();
    env->targets = allocate_card_array(1);
}

void free_tcg(TCG* env) {
    free(env->stack);
    free(env->event_manager);
    free_player(env->my_player);
    free_player(env->op_player);
    free_card_array(env->targets);
}

void draw_card(TCG* env, CardArray* deck, CardArray* hand) {
    if (deck->length == 0) {
        reset(env);
        printf("Deck empty. You lose.\n");
        return;
    }
    if (hand->length == hand->max) {
        printf("Hand full. Skipping draw.\n");
        return;
    }
    move_card(deck, hand, deck->length - 1);
}

void randomize_deck(CardArray* deck) {
    // Fisher-Yates shuffle
    for (int i = 0; i < deck->length; i++) {
        int randomIndex = i + rand() % (deck->length - i);
        Card* tmp = deck->cards[i];
        deck->cards[i] = deck->cards[randomIndex];
        deck->cards[randomIndex] = tmp;
    }
}

int tappable_mana(Player* player) {
    int tappable = 0;
    for (int i = 0; i < player->lands->length; i++) {
        Card* card = player->lands->cards[i];
        if (!card->tapped) {
            tappable += 1;
        }
    }
    return tappable;
}

void tap_lands_for_mana(Player* player, int required_mana) {
    for (int i = 0; i < player->lands->length; i++) {
        if (player->mana >= required_mana) {
            break;
        }
        if (!player->lands->cards[i]->tapped) {
            player->lands->cards[i]->tapped = true;
            (player->mana)++;
        }
    }
}

bool can_play_card(TCG* env, Player* player, Card* card) {

    int tappable = tappable_mana(player);
    if (card->cost > player->mana + tappable) {
        return false;
    }

    switch (card->type) {
        case TYPE_LAND:
            return !player->land_played && player->lands->length < LAND_SIZE;
        case TYPE_CREATURE:
            return player->board->length < BOARD_SIZE;
        case TYPE_INSTANT:
            return card->effect.condition(env);
        case TYPE_SORCERY:
            return env->stack->idx == 0 && card->effect.condition(env);
        default:
            printf("Invalid card type: %i\n", card->type);
            return false;
    }
}

bool has_valid_moves(TCG* env, Player* player) {
    // TODO: Check if player can activate abilities

    CardArray* hand = player->hand;

    for (int i = 0; i < hand->length; i++) {
        Card* card = hand->cards[i];
        if (can_play_card(env, player, card)) {
            return true;
        }
    }
    return false;
}


bool phase_untap(TCG* env, unsigned char atn) {
    printf("PHASE_UNTAP\n");

    env->turn = 1 - env->turn;
    env->priority = env->turn;
    
    Player* player = (env->turn == 0) ? env->my_player : env->op_player;
    player->land_played = false;
    player->mana = 0;

    for (int i = 0; i < player->board->length; i++) {
        Card* card = player->board->cards[i];
        if (card->tapped) {
            card->tapped = false;
        }
    }

    for (int i = 0; i < player->lands->length; i++) {
        Card* card = player->lands->cards[i];
        if (card->tapped) {
            card->tapped = false;
        }
    }
    
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_upkeep});
    return TO_STACK;
}

bool phase_upkeep(TCG* env, unsigned char atn) {
    printf("PHASE_UPKEEP\n");

    Player* player = env->turn == 0 ? env->my_player : env->op_player;

    for (int i = 0; i < player->board->length; i++) {
         Card* card = player->board->cards[i];
         if (card->type == TYPE_CREATURE && card->summoning_sickness) {
             card->summoning_sickness = false;
         }
    }
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_draw});
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_priority});
    return TO_STACK;
}

bool phase_draw(TCG* env, unsigned char atn) {
    printf("PHASE_DRAW\n");

    Player* player = env->turn == 0 ? env->my_player : env->op_player;

    draw_card(env, player->deck, player->hand);
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_play});
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_priority});
    return TO_STACK;
}

bool phase_play(TCG* env, unsigned char atn) {
    printf("PHASE_PLAY\n");
    Player* player = (env->turn == 0) ? env->my_player : env->op_player;

    if (!has_valid_moves(env, player)) {
        printf("\t No valid moves. Skip to next phase.\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_attack});
        return TO_STACK;
    }
    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_play});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_attack});
        return TO_STACK;
    } else if (atn == ACTION_SPACE) {
        env->participate_in_priority = !env->participate_in_priority;
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_play});
        return TO_USER;
    } else if (atn >= player->hand->length) {
        printf("\t Invalid action: %i\n. Hand length: %i\n", atn, player->hand->length);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_play});
        return TO_USER;
    }

    Card* card = player->hand->cards[atn];

    if (!can_play_card(env, player, card)) {
        printf("Condition for playing this card not met\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_play});
        return TO_USER;
    }

    tap_lands_for_mana(player, card->cost);
    player->mana -= card->cost;

    Event event = {.type = EVENT_CARD_PLAYED, .data = card};
    publishEvent(env, env->event_manager, &event);

    switch (card->type) {
        case TYPE_LAND:
            move_card(player->hand, player->lands, atn);
            player->land_played = true;
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_play});
            return TO_USER;
        case TYPE_CREATURE:
            move_card(player->hand, player->board, atn);
            addListener(env->event_manager, card->effect.event, card->effect.trigger);
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_play});
            return TO_STACK;
        case TYPE_INSTANT:
            printf("Instant effect\n");
            move_card(player->hand, player->graveyard, atn);
            printf("Moving instant to graveyard\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_play});
            push(env->stack, (StackItem){.type = STACK_EFFECT, .effect = card->effect});
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_priority});
            printf("Pushing target phase\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = card->effect.target_fn});
            return TO_STACK;
        case TYPE_SORCERY:
            move_card(player->hand, player->graveyard, atn);
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_play});
            push(env->stack, (StackItem){.type = STACK_EFFECT, .effect = card->effect});
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_priority});
            return TO_STACK;
        default:
            printf("\t Unknown card type\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_play});
            return TO_USER;
    }
}

bool phase_attack(TCG* env, unsigned char atn) {
    printf("PHASE_ATTACK\n");
    Player* player = (env->turn == 0) ? env->my_player : env->op_player;

    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_attack});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Attacks confirmed. Phase end\n");
        env->turn = 1 - env->turn;
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_block});
        return TO_STACK;
    } else if (atn >= player->board->length || player->board->cards[atn]->type != TYPE_CREATURE) {
        printf("\t Invalid action %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_attack});
        return TO_USER;
    }
    if (player->board->cards[atn]->summoning_sickness) {
            printf("\t Cannot attack with summoning sickness\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_attack});
            return TO_USER;
        }

    printf("\t Setting attacker %i\n", atn);
    player->board->cards[atn]->attacking = !player->board->cards[atn]->attacking;
    player->board->cards[atn]->tapped = !player->board->cards[atn]->tapped;
    Event event = {.type = EVENT_CARD_ATTACKED, .data = player->board->cards[atn]};
    publishEvent(env, env->event_manager, &event);
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_attack});
    return TO_USER;
}

// Function to handle the blocking phase in the game
bool phase_block(TCG* env, unsigned char atn) {
    printf("PHASE_BLOCK\n");

    Player* player = (env->turn == 0) ? env->my_player : env->op_player;
    Player* opponent = (env->turn == 0) ? env->op_player : env->my_player;

    // Skip non-attacking cards
    while (env->block_idx < player->board->length && !player->board->cards[env->block_idx]->attacking) {
        printf("\t Skipping block for %i (not attacking)\n", env->block_idx);
        env->block_idx++;
    }

    bool can_block = false;
    for (int i = 0; i < opponent->board->length; i++) {
        Card* card = opponent->board->cards[i];
        if (card->type == TYPE_CREATURE && !card->tapped && (card->defending == -1 || card->defending == env->block_idx)) {
            can_block = true;
            printf("\t Can block with %i\n", i);
            break;
        }
    }
    if (!can_block) {
        env->block_idx = player->board->length;
    }

    // If no more defenders are available, resolve attacks
    if (env->block_idx == player->board->length) {
        printf("\t Attacker board length: %i\n", player->board->length);

        for (int atk = 0; atk < player->board->length; atk++) {
            printf("\t Resolving %i\n", atk);
            Card* attacker = player->board->cards[atk];

            // Skip non-creature or non-attacking cards
            if (attacker->type != TYPE_CREATURE) {
                printf("\t Not attacking because not a creature\n");
                continue;
            }
            if (!attacker->attacking) {
                printf("\t Not attacking\n");
                continue;
            }

            int attacker_attack = attacker->attack;
            int attacker_health = attacker->health;

            // Resolve combat with each defender
            for (int def = 0; def < opponent->board->length; def++) {
                Card* defender = opponent->board->cards[def];
                if (defender->type != TYPE_CREATURE) {
                    printf("\t Not a creature\n");
                    continue;
                }
                if (defender->defending != atk) {
                    printf("\t Not defending this attacker\n");
                    continue;
                }

                int defender_attack = defender->attack;
                int defender_health = defender->health;

                // Resolve damage between attacker and defender
                if (attacker_attack >= defender_health) {
                    attacker_attack -= defender_health;
                    attacker_health -= defender_attack;

                    defender->health = 0;
                    move_card(opponent->board, opponent->graveyard, def); // Move defender to graveyard
                } else {
                    attacker_health -= defender_attack;
                    attacker_attack = 0;
                }
                // Check if attacker is defeated
                if (attacker_health <= 0) {
                    move_card(player->board, player->graveyard, atk); // Move attacker to graveyard
                    break;
                }
            }

            // Reduce player health if attack goes through
            printf("\t Reducing health by %i\n", attacker_attack);
            opponent->health -= attacker_attack;
        }

        // Handle end of turn logic
        if (opponent->health <= 0) {
            printf("\t Game over\n");
            reset(env); // Reset the game
        }

        // Draw a card for the defender and reset states
        draw_card(env, opponent->deck, opponent->hand);


        // Reset attacking and defending statuses
        for (int i = 0; i < player->board->length; i++) {
            if (player->board->cards[i]->type == TYPE_CREATURE) {
                player->board->cards[i]->attacking = false;
            }
        }
        for (int i = 0; i < opponent->board->length; i++) {
            if (opponent->board->cards[i]->type == TYPE_CREATURE) {
                opponent->board->cards[i]->defending = -1;
            }
        }

        printf("\t Set block idx to 0\n");
        env->block_idx = 0;
        env->turn = 1 - env->turn; // Switch turn
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_untap});
        return TO_STACK;
    }

    // Handle player actions during blocking phase
    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_block});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Manual block confirm %i\n", env->block_idx);
        env->block_idx++;
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_block});
        return TO_STACK;
    } else if (atn >= opponent->board->length) {
        printf("\t Invalid block action %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_block});
        return TO_USER;
    }

    // Validate the selected defender card for blocking
    for (int i = 0; i < env->block_idx; i++) {
        if (opponent->board->cards[atn]->defending == i) {
            printf("\t Already blocked\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_block});
            return TO_USER;
        }
        if (opponent->board->cards[atn]->tapped) {
            printf("\t Cannot block with tapped card\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_block});
            return TO_USER;
        }
        if (opponent->board->cards[atn]->type != TYPE_CREATURE) {
            printf("\t Cannot block with non-creature card\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_block});
            return TO_USER;
        }
    }

    // Set the defending state for the selected card
    printf("\t Blocking index %i with %i\n", env->block_idx, atn);
    Card* card = opponent->board->cards[atn];
    if (card->defending == env->block_idx) {
        card->defending = -1;
    } else {
        card->defending = env->block_idx;
    }
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_block}); // Repeat the blocking phase
    return TO_USER;
}

void pass_priority(TCG* env) {
    env->priority = 1 - env->priority;
    env->priority_passed = true;
}

bool phase_priority(TCG* env, unsigned char atn) {
    printf("phase_priority\n");
    Player* player = (env->priority == 0) ? env->my_player : env->op_player;

    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_priority});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        if (env->priority_passed) {
            printf("Both players passed priority\n");
            env->priority = env->turn;
            env->priority_passed = false;
            return TO_STACK;
        }
        printf("Player %i passed priority\n", env->priority);
        pass_priority(env);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_priority});
        return TO_USER;
    } else if (atn >= player->hand->length) {
        printf("\t Invalid action: %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_priority});
        return TO_USER;
    } else if (player->hand->cards[atn]->type != TYPE_INSTANT) {
        printf("\t Can only play Instant during priority\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_priority});
        return TO_USER;
    } 

    Card* card = player->hand->cards[atn];

    if (!can_play_card(env, player, card)) {
        printf("\t Condition not met\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_priority});
        return TO_USER;
    }

    tap_lands_for_mana(player, card->cost);

    assert(player->mana >= card->cost);
    player->mana -= card->cost;

    printf("\t Activating instant effect\n");
    env->priority_passed = false;
    move_card(player->hand, player->graveyard, atn);
    push(env->stack, (StackItem){
        .type = STACK_EFFECT,
        .effect = card->effect,
    });
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_priority});
    return TO_USER;
}

/* When an effect gets activated, we want to be able to choose any amount of targets. For now, only from a fixed zone. The number of valid targets and the zone will be specified as part of the card effect. The effect has a method "target" that should call choose_targets with the appropriate number of targets and zone. The env has a param target_idx that keeps track of how many targets have been selected yet. */


void step(TCG* env, unsigned char atn) {
    printf("Turn: %i, Action: %i\n", env->turn, atn);
    while (true) {
        StackItem item = pop(env->stack);
        if (item.type == STACK_PHASE) {
            bool return_to_user = item.phase(env, atn);
            if (return_to_user) {
                StackItem next_item = peek(env->stack);
                if (next_item.type == STACK_PHASE &&
                        next_item.phase == phase_priority &&
                        env->priority == 0 &&
                        !env->participate_in_priority) {
                    atn = ACTION_ENTER;
                    continue;
                }
                return;
            }
        } else if (item.type == STACK_EFFECT) {
            item.effect.activate(env, env->targets);
            printf("Effect activations are not yet implemented\n");
        }
        atn = ACTION_NOOP;
    }
}

void reset_player(Player* player) {
    player->hand->length = 0;
    player->board->length = 0;
    player->lands->length = 0;
    player->deck->length = DECK_SIZE;
    player->graveyard->length = 0;
    player->mana = 0;
    player->health = 20;
}

void reset(TCG* env) {
    reset_player(env->my_player);
    reset_player(env->op_player);
    randomize_deck(env->my_player->deck);
    randomize_deck(env->op_player->deck);
    env->turn = rand() % 2;
    env->priority = env->turn;
    env->priority_passed = false;
    env->participate_in_priority = false;
    for (int i = 0; i < 5; i++) {
        draw_card(env, env->my_player->deck, env->my_player->hand);
        draw_card(env, env->op_player->deck, env->op_player->hand);
    }
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase = phase_draw});
    step(env, ACTION_NOOP);
}

void init_client(TCG* env) {
    InitWindow(1080, 720, "Puffer the Schooling TCG");
    SetTargetFPS(60);
}

void close_client(TCG* env) {
    CloseWindow();
}

int card_x(int col, int n) {
    int cards_width = 72*n;
    int offset = 72*col;
    return GetScreenWidth()/2 - cards_width/2 + offset;
}

int card_y(int row) {
    return 64 + (128 + 20)*row;
}

void render_creature(Card* card, int x, int y) {
    Color color = (card->tapped) ? TAPPED_COLOR : card->summoning_sickness ? SUMMONING_SICKNESS_COLOR : CREATURE_COLOR;
    DrawRectangle(x, y, 64, 128, color);
    if (card->attacking) {
        y -= 16;
    }
    DrawText(TextFormat("%i", card->cost), x + 32, y+32, 20, WHITE);
    DrawText(TextFormat("%i", card->attack), x + 32, y + 52, 20, WHITE);
    DrawText(TextFormat("%i", card->health), x + 32, y + 72, 20, WHITE);
}
void render_instant(Card* card, int x, int y) {
    DrawRectangle(x, y, 64, 128, INSTANT_COLOR);
    DrawText(TextFormat("%i", card->cost), x + 32, y+32, 20, WHITE);
}

void render_sorcery(Card* card, int x, int y) {
    DrawRectangle(x, y, 64, 128, SORCERY_COLOR);
    DrawText(TextFormat("%i", card->cost), x + 32, y+32, 20, WHITE);
}

void render_land(Card* card, int x, int y) {
    DrawRectangle(x, y, 64, 128, LAND_COLOR);
}

void render_card(Card* card, int x, int y) {
    if (card->type == TYPE_LAND) {
        render_land(card, x, y);
    } else if (card->type == TYPE_CREATURE) {
        render_creature(card, x, y);
    } else if (card->type == TYPE_INSTANT) {
        render_instant(card, x, y);
    } else if (card->type == TYPE_SORCERY) {
        render_sorcery(card, x, y);
    } else {
        printf("Invalid card type: %i\n", card->type);
    }
    DrawText(TextFormat("%.5s", card->name), x + 8, y + 16, 16, WHITE);
}

void render_label(int x, int y, int idx) {
    DrawText(TextFormat("%i", (idx+1)%10), x+32, y+96, 20, YELLOW);
}

void render_my_hand(TCG* env) {
    for (int i = 0; i < env->my_player->hand->length; i++) {
        Card* card = env->my_player->hand->cards[i];
        int x = card_x(i, env->my_player->hand->length);
        int y = card_y(3);
        render_card(card, x, y);
        if (env->turn == 0) {
            render_label(x, y, i);
        }
    }
}

void render_op_hand(TCG* env) {
    for (int i = 0; i < env->op_player->hand->length; i++) {
        Card* card = env->op_player->hand->cards[i];
        int x = card_x(i, env->op_player->hand->length);
        int y = card_y(0);
        render_card(card, x, y);
    }
}

void render_my_board(TCG* env) {
    for (int i = 0; i < env->my_player->board->length; i++) {
        Card* card = env->my_player->board->cards[i];

        int x = card_x(i, env->my_player->board->length);
        int y = card_y(2);

        render_card(card, x, y);
        if (env->turn == 0) {
            render_label(x, y, i);
        }
    }

    for (int i = 0; i < env->my_player->board->length; i++) {
        Card* card = env->my_player->board->cards[i];
        if (card->type != TYPE_CREATURE || card->defending == -1) {
            continue;
        }
        DrawLineEx(
            (Vector2){32+card_x(i, env->my_player->board->length), 64+card_y(2)},
            (Vector2){32+card_x(card->defending, env->op_player->board->length), 64+card_y(1)},
            3.0f, WHITE
        );
    }
}

void render_op_board(TCG* env) {
    for (int i = 0; i < env->op_player->board->length; i++) {

        Card* card = env->op_player->board->cards[i];
        int x = card_x(i, env->op_player->board->length);
        int y = card_y(1);

        render_card(card, x, y);
    }
}

void render_my_lands(TCG* env) {
    int num_untapped = 0;
    for (int i = 0; i < env->my_player->lands->length; i++) {
        Card* card = env->my_player->lands->cards[i];
        int x = LAND_ZONE_OFFSET;
        int y = card_y(2);
        assert(card->type == TYPE_LAND);
        render_card(card, x, y);
        if (!card->tapped) {
            num_untapped += 1;
        }
    }
    if (env->my_player->lands->length > 0) {
        int x = LAND_ZONE_OFFSET + 4;
        int y = card_y(2) + 4;
        DrawText(TextFormat("%i", num_untapped), x, y, 20, WHITE);
    }
}

void render_op_lands(TCG* env) {
    int num_untapped = 0;
    for (int i = 0; i < env->op_player->lands->length; i++) {
        Card* card = env->op_player->lands->cards[i];
        int x = (GetScreenWidth() - LAND_ZONE_OFFSET);
        int y = card_y(1);
        render_card(card, x, y);
        if (!card->tapped) {
            num_untapped += 1;
        }
    }
    if (env->op_player->lands->length > 0) {
        int x = (GetScreenWidth() - LAND_ZONE_OFFSET) + 4;
        int y = card_y(1) + 4;
        DrawText(TextFormat("%i", num_untapped), x, y, 20, YELLOW);
    }
}

void render_my_graveyard(TCG* env) {
    for (int i = 0; i < env->my_player->graveyard->length; i++) {
        Card* card = env->my_player->graveyard->cards[i];
        int x = (GetScreenWidth() - GRAVEYARD_ZONE_OFFSET);
        int y = card_y(2);
        render_card(card, x, y);
    }
}

void render_op_graveyard(TCG* env) {
    for (int i = 0; i < env->op_player->graveyard->length; i++) {
        Card* card = env->op_player->graveyard->cards[i];
        int x = GRAVEYARD_ZONE_OFFSET;
        int y = card_y(1);
        render_card(card, x, y);
    }
}

void render(TCG* env) {
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
   

render_my_hand(env);

    render_my_board(env);
    render_op_board(env);

    render_my_lands(env);
    render_op_lands(env);

    render_my_graveyard(env);
    render_op_graveyard(env);

    render_op_hand(env);

    // render participate in priority in the bottom right corner
    if (env->participate_in_priority) {
        DrawText("Participate in Priority", GetScreenWidth() - 256, GetScreenHeight() - 64, 20, WHITE);
    } else {
        DrawText("Do not participate in Priority", GetScreenWidth() - 320, GetScreenHeight() - 64, 20, WHITE);
    }


    int x = GetScreenWidth() - 128;
    int y = 32;

    StackItem item = peek(env->stack);
    if (item.type == STACK_PHASE) {
        if (item.phase == phase_draw) {
            DrawText("Draw", x, y, 20, WHITE);
        } else if (item.phase == phase_play) {
            DrawText("Play", x, y, 20, WHITE);
        } else if (item.phase == phase_attack) {
            DrawText("Attack", x, y, 20, WHITE);
        } else if (item.phase == phase_block) {
            DrawText("Block", x, y, 20, WHITE);
        }
    } 

    DrawText(TextFormat("Health: %i", env->my_player->health), 32, 32, 20, WHITE);
    DrawText(TextFormat("Health: %i", env->op_player->health), 32, 64, 20, WHITE);

    EndDrawing();
}
