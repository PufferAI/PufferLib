#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include "raylib.h"

#define HAND_SIZE 10
// TODO: Fix land and board sizes
#define BOARD_SIZE 10
#define LAND_SIZE 10
#define DECK_SIZE 60
#define GRAVEYARD_SIZE 60
#define STACK_SIZE 100

#define LAND_ZONE_OFFSET 128
#define GRAVEYARD_ZONE_OFFSET 128

#define ACTION_ENTER 10
#define ACTION_NOOP 11
#define ACTION_SPACE 12

#define TO_USER true
#define TO_STACK false

#define LAND_COLOR          (Color){139, 69, 19, 255}    // Brown
#define CREATURE_COLOR      (Color){255, 0, 0, 255}      // Red
#define INSTANT_COLOR       (Color){0, 255, 255, 255}    // Cyan
#define SORCERY_COLOR       (Color){128, 0, 128, 255}    // Purple
#define ARTIFACT_COLOR      (Color){192, 192, 192, 255}  // Silver
#define ENCHANTMENT_COLOR   (Color){255, 215, 0, 255}    // Gold
#define PLANESWALKER_COLOR  (Color){0, 100, 255, 255}    // Dark Blue
#define GRAVEYARD_COLOR     (Color){64, 64, 64, 255}     // Dark Grey
#define TAPPED_COLOR        (Color){128, 0, 0, 128}      // Dimmed Red
#define SUMMONING_SICKNESS_COLOR (Color){255, 165, 0, 255} // Orange

typedef struct TCG TCG;
typedef struct Card Card;
typedef struct Effect Effect;
typedef struct Stack Stack;
typedef struct StackItem StackItem;
typedef bool (*call)(TCG*, unsigned char);
bool phase_untap(TCG* env, unsigned char atn);
bool phase_upkeep(TCG* env, unsigned char atn);
bool phase_draw(TCG* env, unsigned char atn);
bool phase_play(TCG* env, unsigned char atn);
bool phase_attack(TCG* env, unsigned char atn);
bool phase_block(TCG* env, unsigned char atn);
bool phase_priority(TCG* env, unsigned char atn);
bool phase_target(TCG* env, unsigned char atn);
void reset(TCG* env);

typedef enum {
    STACK_PHASE,
    STACK_EFFECT,
} StackType;

struct StackItem {
    StackType type;
    union {
        bool (*phase_func)(TCG* env, unsigned char atn);
        struct {
            void (*activate)(TCG* env, int target_idx);
            int target_idx;
        } effect;
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

typedef enum {
    TYPE_LAND,
    TYPE_CREATURE,
    TYPE_INSTANT,
    TYPE_SORCERY,
    TYPE_ARTIFACT,
    TYPE_ENCHANTMENT,
    TYPE_PLANESWALKER
} CardType;


struct Effect {
    char description[256];
    void (*trigger)(TCG* env);
    bool (*condition)(TCG* env);
    void (*activate)(TCG* env, int target_idx);
};

struct Card {
    char name[256];
    CardType type;
    int cost;
    bool tapped;
    Color color;

    union {
        struct { Effect effect; int attack, health; bool summoning_sickness, attacking; int defending; } creature;
        struct { Effect effect; int num_targets; } spell;
    } data;
};

typedef struct Zone Zone;
struct Zone {
    Card* cards;
    int length;
    int max;
};



Card* allocate_creature(int cost, int attack, int health, Effect effect) {
    Card* card = (Card*)calloc(1, sizeof(Card));
    card->type = TYPE_CREATURE;
    card->cost = cost;
    card->tapped = false;
    card->color = CREATURE_COLOR;
    card->data.creature.effect = effect;
    card->data.creature.attack = attack;
    card->data.creature.health = health;
    card->data.creature.summoning_sickness = true;
    card->data.creature.attacking = false;
    card->data.creature.defending = -1;
    return card;
}

Card* allocate_instant(int cost, Effect effect, int num_targets) {
    Card* card = (Card*)calloc(1, sizeof(Card));
    card->type = TYPE_INSTANT;
    card->cost = cost;
    card->tapped = false;
    card->color = INSTANT_COLOR;
    card->data.spell.effect = effect;
    card->data.spell.num_targets = num_targets;
    return card;
}

Card* allocate_sorcery(int cost, Effect effect, int num_targets) {
    Card* card = (Card*)calloc(1, sizeof(Card));
    card->type = TYPE_SORCERY;
    card->cost = cost;
    card->tapped = false;
    card->color = RED;
    card->data.spell.effect = effect;
    card->data.spell.num_targets = num_targets;
    return card;
}

Card* allocate_land() {
    Card* card = (Card*)calloc(1, sizeof(Card));
    card->type = TYPE_LAND;
    card->cost = 0;
    card->tapped = false;
    card->color = LAND_COLOR;
    return card;
}

Zone* allocate_card_array(int max) {
    Zone* hand = (Zone*)calloc(1, sizeof(Zone));
    hand->cards = (Card*)calloc(max, sizeof(Card));
    hand->max = max;
    return hand;
}

void free_card_array(Zone* array) {
    free(array->cards);
    free(array);
}

struct TCG {
    Zone* my_hand;
    Zone* my_board;
    Zone* my_lands;
    Zone* my_deck;
    Zone* my_graveyard;
    int my_health;
    int my_mana;
    bool my_land_played;

    Zone* op_hand;
    Zone* op_board;
    Zone* op_lands;
    Zone* op_deck;
    Zone* op_graveyard;
    int op_health;
    int op_mana;
    bool op_land_played;

    Stack* stack;
    int block_idx;
    int turn;
    int priority;
    bool priority_passed;
    bool participate_in_priority;
};

void add_card_to(TCG* env, Card card, Zone* to) {
    to->cards[to->length] = card;
    to->length += 1;
}

void remove_card_from(TCG* env, Zone* from, int idx) {
    for (int i = idx; i < from->length - 1; i++) {
        from->cards[i] = from->cards[i + 1];
    }
    from->length -= 1;
}

void move_card(TCG* env, Zone* from, Zone* to, int idx) {
    Card card = from->cards[idx];
    remove_card_from(env, from, idx);
    add_card_to(env, card, to);
}

void no_trigger(TCG* env) {}
bool no_condition(TCG* env) { return true; }
void no_activate(TCG* env, int target_idx) {}

bool opponent_has_creature(TCG* env) {
    Zone* op_board = (env->turn == 0) ? env->op_board : env->my_board;
    for (int i = 0; i < op_board->length; i++) {
        if (op_board->cards[i].type == TYPE_CREATURE) {
            return true;
        }
    }
    return false;
}

void lightning_bolt_activate(TCG* env, int target_idx) {
    Zone* op_board = (env->turn == 0) ? env->op_board : env->my_board;
    Zone* op_graveyard = (env->turn == 0) ? env->op_graveyard : env->my_graveyard;
    printf("Lightning bolt activated\n");
    Card* card = &op_board->cards[target_idx];
    assert (card->type == TYPE_CREATURE);
    card->data.creature.health -= 3;
    if (card->data.creature.health <= 0) {
        printf("Lightning Bolt kills creature\n");
        card->data.creature.health = 0;
        move_card(env, op_board, op_graveyard, target_idx);
    }
}


void allocate_tcg(TCG* env) {
    env->stack = calloc(1, sizeof(Stack));
    env->my_hand = allocate_card_array(HAND_SIZE);
    env->op_hand = allocate_card_array(HAND_SIZE);
    env->my_board = allocate_card_array(BOARD_SIZE);
    env->op_board = allocate_card_array(BOARD_SIZE);
    env->my_lands = allocate_card_array(LAND_SIZE);
    env->op_lands = allocate_card_array(LAND_SIZE);
    env->my_deck = allocate_card_array(DECK_SIZE);
    env->op_deck = allocate_card_array(DECK_SIZE);
    env->my_graveyard = allocate_card_array(GRAVEYARD_SIZE);
    env->op_graveyard = allocate_card_array(GRAVEYARD_SIZE);
}

void free_tcg(TCG* env) {
    free_card_array(env->my_hand);
    free_card_array(env->op_hand);
    free_card_array(env->my_board);
    free_card_array(env->op_board);
    free_card_array(env->my_lands);
    free_card_array(env->op_lands);
    free_card_array(env->my_deck);
    free_card_array(env->op_deck);
    free_card_array(env->my_graveyard);
    free_card_array(env->op_graveyard);
}

void draw_card(TCG* env, Zone* deck, Zone* hand) {
    if (deck->length == 0) {
        reset(env);
        printf("Deck empty. You lose.\n");
        return;
    }
    if (hand->length == hand->max) {
        printf("Hand full. Skipping draw.\n");
        return;
    }
    move_card(env, deck, hand, deck->length - 1);
}

bool my_hand_full(TCG* env) {
    Zone* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    return hand->length == HAND_SIZE;
}

bool my_hand_not_full(TCG* env) {
    Zone* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    return hand->length < HAND_SIZE;
}

void draw_card_activate(TCG* env, int target_idx) {
    Zone* deck = (env->turn == 0) ? env->my_deck : env->op_deck;
    Zone* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    draw_card(env, deck, hand);
}

void randomize_deck(Zone* deck) {
    // Fisher-Yates shuffle
    for (int i = 0; i < deck->length; i++) {
        // Generate a random index between i and deck->length - 1
        int randomIndex = i + rand() % (deck->length - i);
        
        // Swap the current card with the card at randomIndex
        Card temp = deck->cards[i];
        deck->cards[i] = deck->cards[randomIndex];
        deck->cards[randomIndex] = temp;
    }
}


int tappable_mana(TCG* env) {
    Zone* lands = (env->turn == 0) ? env->my_lands : env->op_lands;
    int tappable = 0;
    for (int i = 0; i < lands->length; i++) {
        Card card = lands->cards[i];
        if (!card.tapped) {
            tappable += 1;
        }
    }
    return tappable;
}


void tap_lands_for_mana(Zone* lands, int* mana, int required_mana) {
    for (int i = 0; i < lands->length; i++) {
        if (*mana >= required_mana) {
            break;
        }
        if (!lands->cards[i].tapped) {
            lands->cards[i].tapped = true;
            (*mana)++;
        }
    }
}

bool can_play_card(TCG* env, Card card) {
    Zone* board = (env->turn == 0) ? env->my_board : env->op_board;
    Zone* lands = (env->turn == 0) ? env->my_lands : env->op_lands;

    int mana = (env->turn == 0) ? env->my_mana : env->op_mana;
    bool land_played = (env->turn == 0) ? env->my_land_played : env->op_land_played;

    int tappable = tappable_mana(env);
    if (card.cost > mana + tappable) {
        return false;
    }

    switch (card.type) {
        case TYPE_LAND:
            return !land_played && lands->length < LAND_SIZE;
        case TYPE_CREATURE:
            return board->length < BOARD_SIZE;
        case TYPE_INSTANT:
            return card.data.spell.effect.condition(env);
        case TYPE_SORCERY:
            return env->stack->idx == 0 && card.data.spell.effect.condition(env);
        default:
            printf("Invalid card type: %i\n", card.type);
            return false;
    }
}

bool has_valid_moves(TCG* env) {
    Zone* hand = (env->turn == 0) ? env->my_hand : env->op_hand;

    for (int i = 0; i < hand->length; i++) {
        Card card = hand->cards[i];
        if (can_play_card(env, card)) {
            return true;
        }
    }
    return false;
}


bool phase_untap(TCG* env, unsigned char atn) {
    printf("PHASE_UNTAP\n");
    bool* land_played = (env->turn == 0) ? &env->my_land_played : &env->op_land_played;
    *land_played = false;

    env->turn = 1 - env->turn;
    env-> priority = 1 - env->priority;
    Zone* board = (env->turn == 0) ? env->my_board : env->op_board;
    Zone* lands = (env->turn == 0) ? env->my_lands : env->op_lands;

    int* mana = (env->turn == 0) ? &env->my_mana : &env->op_mana;
    *mana = 0;

    printf("Untapping board\n");
    for (int i = 0; i < board->length; i++) {
        Card card = board->cards[i];
        if (card.tapped) {
            board->cards[i].tapped = false;
        }
    }

    printf("Untapping lands\n");
    for (int i = 0; i < lands->length; i++) {
        Card card = lands->cards[i];
        if (card.tapped) {
            lands->cards[i].tapped = false;
        }
    }
    
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_upkeep});
    return TO_STACK;
}

bool phase_upkeep(TCG* env, unsigned char atn) {
    printf("PHASE_UPKEEP\n");
    Zone* board = (env->turn == 0) ? env->my_board : env->op_board;
    for (int i = 0; i < board->length; i++) {
         Card card = board->cards[i];
         if (card.type == TYPE_CREATURE && card.data.creature.summoning_sickness) {
             board->cards[i].data.creature.summoning_sickness = false;
         }
    }
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_draw});
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
    return TO_STACK;
}

bool phase_draw(TCG* env, unsigned char atn) {
    printf("PHASE_DRAW\n");
    Zone* deck = (env->turn == 0) ? env->my_deck : env->op_deck;
    Zone* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    draw_card(env, deck, hand);
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
    return TO_STACK;
}

bool phase_play(TCG* env, unsigned char atn) {
    printf("PHASE_PLAY\n");
    Zone* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    Zone* board = (env->turn == 0) ? env->my_board : env->op_board;
    /*Zone* op_board = (env->turn == 0) ? env->op_board : env->my_board;*/
    Zone* lands = (env->turn == 0) ? env->my_lands : env->op_lands;
    Zone* graveyard = (env->turn == 0) ? env->my_graveyard : env->op_graveyard;
    int* mana = (env->turn == 0) ? &env->my_mana : &env->op_mana;
    bool* land_played = (env->turn == 0) ? &env->my_land_played : &env->op_land_played;


    if (!has_valid_moves(env)) {
        printf("\t No valid moves\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_attack});
        return TO_STACK;
    }

    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_attack});
        return TO_STACK;
    } else if (atn == ACTION_SPACE) {
        printf("Toggling participate in priority\n");
        env->participate_in_priority = !env->participate_in_priority;
        printf("Participate in priority: %i\n", env->participate_in_priority);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
        return TO_USER;
    } else if (atn >= hand->length) {
        printf("\t Invalid action: %i\n. Hand length: %i\n", atn, hand->length);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
        return TO_USER;
    }

    Card card = hand->cards[atn];

    if (!can_play_card(env, card)) {
        printf("Condition for playing card not met\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
        return TO_USER;
    }

    tap_lands_for_mana(lands, mana, card.cost);

    assert(*mana >= card.cost);
    *mana -= card.cost;

    switch (card.type) {
        case TYPE_LAND:
            move_card(env, hand, lands, atn);
            printf("\t Land played\n");
            *land_played = true;
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
            return TO_USER;
        case TYPE_CREATURE:
            move_card(env, hand, board, atn);
            printf("\t Creature played\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
            push(env->stack, (StackItem){.type = STACK_EFFECT, .effect = {.activate = card.data.creature.effect.activate}});
            return TO_STACK;
        case TYPE_INSTANT:
            move_card(env, hand, graveyard, atn);
            printf("\t Instant played\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
            push(env->stack, (StackItem){
                .type = STACK_EFFECT,
                .effect = {
                    .activate = card.data.spell.effect.activate,
                    .target_idx = 0
                }
            });
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
            if (card.data.spell.num_targets > 0) {
                printf("Please choose a target\n");
            }
            return card.data.spell.num_targets == 0 ? TO_STACK : TO_USER;
        case TYPE_SORCERY:
            move_card(env, hand, graveyard, atn);
            printf("\t Sorcery played\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
            push(env->stack, (StackItem){
                .type = STACK_EFFECT,
                .effect = {
                    .activate = card.data.spell.effect.activate,
                    .target_idx = 0
                }
            });
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
            return card.data.spell.num_targets == 0 ? TO_STACK : TO_USER;
        default:
            printf("\t Unknown card type\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
            return TO_USER;
    }
}

bool phase_attack(TCG* env, unsigned char atn) {
    printf("PHASE_ATTACK\n");
    Zone* board = (env->turn == 0) ? env->my_board : env->op_board;

    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_attack});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Attacks confirmed. Phase end\n");
        env->turn = 1 - env->turn;
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
        return TO_STACK;
    } else if (atn >= board->length || board->cards[atn].type != TYPE_CREATURE) {
        printf("\t Invalid action %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_attack});
        return TO_USER;
    } else {
        if (board->cards[atn].data.creature.summoning_sickness) {
            printf("\t Cannot attack with summoning sickness\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_attack});
            return TO_USER;
        }
        printf("\t Setting attacker %i\n", atn);
        board->cards[atn].data.creature.attacking = !board->cards[atn].data.creature.attacking;
        board->cards[atn].tapped = !board->cards[atn].tapped;
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_attack});
        return TO_USER;
    }
}



// Function to handle the blocking phase in the game
bool phase_block(TCG* env, unsigned char atn) {
    printf("PHASE_BLOCK\n");

    // Determine the board, graveyard, and health of the current player and opponent
    Zone* defender_board = (env->turn == 0) ? env->my_board : env->op_board;
    Zone* board = (env->turn == 0) ? env->op_board : env->my_board;
    Zone* defender_graveyard = (env->turn == 0) ? env->my_graveyard : env->op_graveyard;
    Zone* graveyard = (env->turn == 0) ? env->op_graveyard : env->my_graveyard;
    int* health = (env->turn == 0) ? &env->op_health : &env->my_health;

    // Skip non-attacking cards
    while (env->block_idx < board->length && !board->cards[env->block_idx].data.creature.attacking) {
        printf("\t Skipping block for %i (not attacking)\n", env->block_idx);
        env->block_idx++;
    }

    bool can_block = false;
    for (int i = 0; i < defender_board->length; i++) {
        Card card = defender_board->cards[i];
        // If card is not defending yet or already defending the current attacker
        if (card.type == TYPE_CREATURE && !card.tapped && (card.data.creature.defending == -1 || card.data.creature.defending == env->block_idx)) {
            can_block = true;
            printf("\t Can block with %i\n", i);
            break;
        }
    }
    if (!can_block) {
        env->block_idx = board->length;
    }

    // If no more defenders are available, resolve attacks
    if (env->block_idx == board->length) {
        printf("\t Attacker board length: %i\n", board->length);

        for (int atk = 0; atk < board->length; atk++) {
            printf("\t Resolving %i\n", atk);
            Card* attacker = &board->cards[atk];

            // Skip non-creature or non-attacking cards
            if (attacker->type != TYPE_CREATURE) {
                printf("\t Not attacking because not a creature\n");
                continue;
            }
            if (!attacker->data.creature.attacking) {
                printf("\t Not attacking\n");
                continue;
            }

            int attacker_attack = attacker->data.creature.attack;
            int attacker_health = attacker->data.creature.health;

            // Resolve combat with each defender
            for (int def = 0; def < defender_board->length; def++) {
                Card* defender = &defender_board->cards[def];
                if (defender->type != TYPE_CREATURE) {
                    printf("\t Not a creature\n");
                    continue;
                }
                if (defender->data.creature.defending != atk) {
                    printf("\t Not defending this attacker\n");
                    continue;
                }

                int defender_attack = defender->data.creature.attack;
                int defender_health = defender->data.creature.health;

                // Resolve damage between attacker and defender
                if (attacker_attack >= defender_health) {
                    attacker_attack -= defender_health;
                    attacker_health -= defender_attack;

                    defender->data.creature.health = 0;
                    move_card(env, defender_board, defender_graveyard, def); // Move defender to graveyard
                } else {
                    attacker_health -= defender_attack;
                    attacker_attack = 0;
                }
                // Check if attacker is defeated
                if (attacker_health <= 0) {
                    move_card(env, board, graveyard, atk); // Move attacker to graveyard
                    break;
                }
            }

            // Reduce player health if attack goes through
            printf("\t Reducing health by %i\n", attacker_attack);
            *health -= attacker_attack;
        }

        // Handle end of turn logic
        if (*health <= 0) {
            printf("\t Game over\n");
            reset(env); // Reset the game
        }

        // Draw a card for the defender and reset states
        Zone* defender_deck = (env->turn == 0) ? env->my_deck : env->op_deck;
        Zone* defender_hand = (env->turn == 0) ? env->my_hand : env->op_hand;
        draw_card(env, defender_deck, defender_hand);

        // Reset attacking and defending statuses
        for (int i = 0; i < board->length; i++) {
            if (board->cards[i].type == TYPE_CREATURE) {
                board->cards[i].data.creature.attacking = false;
            }
        }
        for (int i = 0; i < defender_board->length; i++) {
            if (defender_board->cards[i].type == TYPE_CREATURE) {
                defender_board->cards[i].data.creature.defending = -1;
            }
        }

        printf("\t Set block idx to 0\n");
        env->block_idx = 0;
        env->turn = 1 - env->turn; // Switch turn
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_untap});
        return TO_STACK;
    }

    // Handle player actions during blocking phase
    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Manual block confirm %i\n", env->block_idx);
        env->block_idx++;
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
        return TO_STACK;
    } else if (atn >= defender_board->length) {
        printf("\t Invalid block action %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
        return TO_USER;
    }

    // Validate the selected defender card for blocking
    for (int i = 0; i < env->block_idx; i++) {
        if (defender_board->cards[atn].data.creature.defending == i) {
            printf("\t Already blocked\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
            return TO_USER;
        }
        if (defender_board->cards[atn].tapped) {
            printf("\t Cannot block with tapped card\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
            return TO_USER;
        }
        if (defender_board->cards[atn].type != TYPE_CREATURE) {
            printf("\t Cannot block with non-creature card\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
            return TO_USER;
        }
    }

    // Set the defending state for the selected card
    printf("\t Blocking index %i with %i\n", env->block_idx, atn);
    Card* card = &defender_board->cards[atn];
    if (card->data.creature.defending == env->block_idx) {
        card->data.creature.defending = -1;
    } else {
        card->data.creature.defending = env->block_idx;
    }
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block}); // Repeat the blocking phase
    return TO_USER;
}

void pass_priority(TCG* env) {
    env->priority = 1 - env->priority;
}

bool phase_priority(TCG* env, unsigned char atn) {
    printf("PHASE_PRIORITY\n");
    Zone* hand = (env->priority == 0) ? env->my_hand : env->op_hand;
    Zone* graveyard = (env->priority == 0) ? env->my_graveyard : env->op_graveyard;
    Zone* lands = (env->priority == 0) ? env->my_lands : env->op_lands;
    int* mana = (env->turn == 0) ? &env->my_mana : &env->op_mana;

    /*if (env->priority == 0 && !env->participate_in_priority) {*/
    /*    atn = ACTION_ENTER;*/
    /*}*/

    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
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
        env->priority_passed = true;
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
        return TO_USER;
    } else if (atn >= hand->length) {
        printf("\t Invalid action: %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
        return TO_USER;
    } else if (hand->cards[atn].type != TYPE_INSTANT) {
        printf("\t Can only play Instant during priority\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
        return TO_USER;
    } else if (!hand->cards[atn].data.spell.effect.condition(env)) {
        printf("\t Instant effect condition not met\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
        return TO_USER;
    }

    Card card = hand->cards[atn];

    if (card.cost > *mana + tappable_mana(env)) {
        printf("\t Not enough mana\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
        return TO_USER;
    }

    tap_lands_for_mana(lands, mana, card.cost);

    // TODO: Remove this debug print after testing
    if (*mana < card.cost) {
        printf("\t Not enough mana\n");
        printf("\t Remaining mana: %i\n", *mana);
        printf("\t Required mana: %i\n", card.cost);
    }
    assert(*mana >= card.cost);
    *mana -= card.cost;

    printf("\t Activating instant effect\n");
    env->priority_passed = false;
    move_card(env, hand, graveyard, atn);
    push(env->stack, (StackItem){
        .type = STACK_EFFECT,
        .effect = {
            .activate = card.data.spell.effect.activate,
            .target_idx = 0
        }
    });
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_priority});
    return TO_USER;
}

/*bool phase_target(TCG* env, unsigned char atn) {*/
/*    printf("PHASE_TARGET\n");*/
/**/
/*    Zone* hand = (env->priority == 0) ? env->my_hand : env->op_hand;*/
/*    Zone* board = (env->priority == 0) ? env->my_board : env->op_board;*/
/*    Zone* graveyard = (env->priority == 0) ? env->my_graveyard : env->op_graveyard;*/
/**/
/*    if (atn == ACTION_NOOP) {*/
/*        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_target});*/
/*        return TO_USER;*/
/*    } else if (atn == ACTION_ENTER) {*/
/*        return TO_STACK;*/
/*    } else if (atn >= board->length) {*/
/*        printf("\t Invalid action: %i\n. Board length: %i\n", atn, board->length);*/
/*        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});*/
/*        return TO_USER;*/
/*    }*/
/*}*/

void step(TCG* env, unsigned char atn) {
    printf("Turn: %i, Action: %i\n", env->turn, atn);
    printf("PRIORITY: %i\n", env->priority);
    while (true) {
        StackItem item = pop(env->stack);
        if (item.type == STACK_PHASE) {
            bool return_to_user = item.phase_func(env, atn);
            if (return_to_user) {
                StackItem next_item = peek(env->stack);
                if (next_item.type == STACK_PHASE &&
                        item.phase_func == phase_priority && next_item.phase_func == phase_priority &&
                        env->priority == 0 &&
                        !env->participate_in_priority) {
                    atn = ACTION_ENTER;
                    continue;
                }
                return;
            }
        } else if (item.type == STACK_EFFECT) {
            printf("ACTIVATING EFFECT FROM STACK");
            item.effect.activate(env, atn);
        }
        atn = ACTION_NOOP;
    }
}

void reset(TCG* env) {
    env->my_deck->length = DECK_SIZE;
    env->op_deck->length = DECK_SIZE;
    env->my_hand->length = 0;
    env->op_hand->length = 0;
    env->my_board->length = 0;
    env->op_board->length = 0;
    env->my_lands->length = 0;
    env->op_lands->length = 0;
    env->my_graveyard->length = 0;
    env->op_graveyard->length = 0;
    env->my_health = 20;
    env->op_health = 20;
    randomize_deck(env->my_deck);
    randomize_deck(env->op_deck);
    env->turn = rand() % 2;
    env->priority = env->turn;
    env->priority_passed = false;
    env->participate_in_priority = false;
    for (int i = 0; i < 5; i++) {
        draw_card(env, env->my_deck, env->my_hand);
        draw_card(env, env->op_deck, env->op_hand);
    }
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_draw});
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

void render_creature(Card card, int x, int y) {
    Color color = (card.tapped) ? TAPPED_COLOR : card.data.creature.summoning_sickness ? SUMMONING_SICKNESS_COLOR : CREATURE_COLOR;
    DrawRectangle(x, y, 64, 128, color);
    if (card.data.creature.attacking) {
        y -= 16;
    }
    DrawText(TextFormat("%i", card.cost), x + 32, y+32, 20, WHITE);
    DrawText(TextFormat("%i", card.data.creature.attack), x + 32, y + 52, 20, WHITE);
    DrawText(TextFormat("%i", card.data.creature.health), x + 32, y + 72, 20, WHITE);
}
void render_instant(Card card, int x, int y) {
    DrawRectangle(x, y, 64, 128, INSTANT_COLOR);
    DrawText(TextFormat("%i", card.cost), x + 32, y+32, 20, WHITE);
}

void render_sorcery(Card card, int x, int y) {
    DrawRectangle(x, y, 64, 128, SORCERY_COLOR);
    DrawText(TextFormat("%i", card.cost), x + 32, y+32, 20, WHITE);
}

void render_land(Card card, int x, int y) {
    DrawRectangle(x, y, 64, 128, LAND_COLOR);
}

void render_card(Card card, int x, int y) {
    if (card.type == TYPE_LAND) {
        render_land(card, x, y);
    } else if (card.type == TYPE_CREATURE) {
        render_creature(card, x, y);
    } else if (card.type == TYPE_INSTANT) {
        render_instant(card, x, y);
    } else if (card.type == TYPE_SORCERY) {
        render_sorcery(card, x, y);
    } else {
        printf("Invalid card type: %i\n", card.type);
    }
    DrawText(TextFormat("%.5s", card.name), x + 8, y + 16, 16, WHITE);
}

void render_label(int x, int y, int idx) {
    DrawText(TextFormat("%i", (idx+1)%10), x+32, y+96, 20, YELLOW);
}

void render_my_hand(TCG* env) {
    for (int i = 0; i < env->my_hand->length; i++) {
        Card card = env->my_hand->cards[i];
        int x = card_x(i, env->my_hand->length);
        int y = card_y(3);
        render_card(card, x, y);
        if (env->turn == 0) {
            render_label(x, y, i);
        }
    }
}

void render_op_hand(TCG* env) {
    for (int i = 0; i < env->op_hand->length; i++) {
        Card card = env->op_hand->cards[i];
        int x = card_x(i, env->op_hand->length);
        int y = card_y(0);
        render_card(card, x, y);
    }
}


void render_my_board(TCG* env) {
    for (int i = 0; i < env->my_board->length; i++) {
        Card card = env->my_board->cards[i];

        int x = card_x(i, env->my_board->length);
        int y = card_y(2);

        render_card(card, x, y);
        if (env->turn == 0) {
            render_label(x, y, i);
        }
    }

    for (int i = 0; i < env->my_board->length; i++) {
        Card card = env->my_board->cards[i];
        if (card.type != TYPE_CREATURE || card.data.creature.defending == -1) {
            continue;
        }
        DrawLineEx(
            (Vector2){32+card_x(i, env->my_board->length), 64+card_y(2)},
            (Vector2){32+card_x(card.data.creature.defending, env->op_board->length), 64+card_y(1)},
            3.0f, WHITE
        );
    }
}

void render_op_board(TCG* env) {
    for (int i = 0; i < env->op_board->length; i++) {

        Card card = env->op_board->cards[i];
        int x = card_x(i, env->op_board->length);
        int y = card_y(1);

        render_card(card, x, y);
    }
}

void render_my_lands(TCG* env) {
    int num_untapped = 0;
    for (int i = 0; i < env->my_lands->length; i++) {
        Card card = env->my_lands->cards[i];
        int x = LAND_ZONE_OFFSET;
        int y = card_y(2);
        assert(card.type == TYPE_LAND);
        render_card(card, x, y);
        if (!card.tapped) {
            num_untapped += 1;
        }
    }
    if (env->my_lands->length > 0) {
        int x = LAND_ZONE_OFFSET + 4;
        int y = card_y(2) + 4;
        DrawText(TextFormat("%i", num_untapped), x, y, 20, WHITE);
    }
}

void render_op_lands(TCG* env) {
    int num_untapped = 0;
    for (int i = 0; i < env->op_lands->length; i++) {
        Card card = env->op_lands->cards[i];
        int x = (GetScreenWidth() - LAND_ZONE_OFFSET);
        int y = card_y(1);
        render_card(card, x, y);
        if (!card.tapped) {
            num_untapped += 1;
        }
    }
    if (env->op_lands->length > 0) {
        int x = (GetScreenWidth() - LAND_ZONE_OFFSET) + 4;
        int y = card_y(1) + 4;
        DrawText(TextFormat("%i", num_untapped), x, y, 20, YELLOW);
    }
}

void render_my_graveyard(TCG* env) {
    for (int i = 0; i < env->my_graveyard->length; i++) {
        Card card = env->my_graveyard->cards[i];
        int x = (GetScreenWidth() - GRAVEYARD_ZONE_OFFSET);
        int y = card_y(2);
        render_card(card, x, y);
    }
}

void render_op_graveyard(TCG* env) {
    for (int i = 0; i < env->op_graveyard->length; i++) {
        Card card = env->op_graveyard->cards[i];
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
        if (item.phase_func == phase_draw) {
            DrawText("Draw", x, y, 20, WHITE);
        } else if (item.phase_func == phase_play) {
            DrawText("Play", x, y, 20, WHITE);
        } else if (item.phase_func == phase_attack) {
            DrawText("Attack", x, y, 20, WHITE);
        } else if (item.phase_func == phase_block) {
            DrawText("Block", x, y, 20, WHITE);
        }
    } 

    DrawText(TextFormat("Health: %i", env->my_health), 32, 32, 20, WHITE);
    DrawText(TextFormat("Health: %i", env->op_health), 32, GetScreenHeight() - 64, 20, WHITE);

    EndDrawing();
}
