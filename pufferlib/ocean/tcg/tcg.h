#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
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
typedef struct Player Player;
typedef struct Card Card;
typedef struct Effect Effect;
typedef struct Stack Stack;
typedef struct StackItem StackItem;
typedef struct CardArray CardArray;
bool phase_untap(TCG* env, unsigned char atn);
bool phase_upkeep(TCG* env, unsigned char atn);
bool phase_draw(TCG* env, unsigned char atn);
bool phase_play(TCG* env, unsigned char atn);
bool phase_attack(TCG* env, unsigned char atn);
bool phase_block(TCG* env, unsigned char atn);
bool resolve_priority(TCG* env, unsigned char atn);
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

struct Effect {
    char description[256];
    void (*trigger)(TCG* env);
    bool (*condition)(TCG* env);
    void (*activate)(TCG* env, CardArray* targets);
    int num_targets;
    CardArray* targets;
};

struct StackItem {
    StackType type;
    union {
        bool (*phase_func)(TCG* env, unsigned char atn);
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
    Effect effect;
    Color color;
    int attack;
    int health;
    int defending;
    int cost;
    CardType type;
    bool tapped;
    bool summoning_sickness;
    bool attacking;
};

struct CardArray {
    Card* cards;
    int length;
    int max;
};

void add_card_to(CardArray* to, Card card) {
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
    Card card = from->cards[idx];
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
    zone->cards = (Card*)calloc(max, sizeof(Card));
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
    int block_idx;
    int turn;
    int priority;
    bool priority_passed;
    bool participate_in_priority;
};


void no_trigger(TCG* env) {}
bool no_condition(TCG* env) { return true; }
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
    env->my_player = allocate_player();
    env->op_player = allocate_player();
}

void free_tcg(TCG* env) {
    free_player(env->my_player);
    free_player(env->op_player);
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
    move_card(env, deck, hand, deck->length - 1);
}

void randomize_deck(CardArray* deck) {
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

int tappable_mana(TCG* env, CardArray* lands) {
    int tappable = 0;
    for (int i = 0; i < lands->length; i++) {
        Card card = lands->cards[i];
        if (!card.tapped) {
            tappable += 1;
        }
    }
    return tappable;
}

void tap_lands_for_mana(CardArray* lands, int* mana, int required_mana) {
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

bool can_play_card(TCG* env, Player* player, Card card) {

    int tappable = tappable_mana(env, player->lands);
    if (card.cost > player->mana + tappable) {
        return false;
    }

    switch (card.type) {
        case TYPE_LAND:
            return !player->land_played && player->lands->length < LAND_SIZE;
        case TYPE_CREATURE:
            return player->board->length < BOARD_SIZE;
        case TYPE_INSTANT:
            return card.effect.condition(env);
        case TYPE_SORCERY:
            return env->stack->idx == 0 && card.effect.condition(env);
        default:
            printf("Invalid card type: %i\n", card.type);
            return false;
    }
}

bool has_valid_moves(TCG* env, Player* player) {
    // TODO: Check if player can activate abilities

    CardArray* hand = player->hand;

    for (int i = 0; i < hand->length; i++) {
        Card card = hand->cards[i];
        if (can_play_card(env, player, card)) {
            return true;
        }
    }
    return false;
}


bool phase_untap(TCG* env, unsigned char atn) {
    printf("PHASE_UNTAP\n");

    // TODO: Maybe move this somewhere else
    env->turn = 1 - env->turn;
    env->priority = env->turn;
    
    Player* player = (env->turn == 0) ? env->my_player : env->op_player;
    player->land_played = false;
    player->mana = 0;

    for (int i = 0; i < player->board->length; i++) {
        Card* card = &player->board->cards[i];
        if (card->tapped) {
            card->tapped = false;
        }
    }

    for (int i = 0; i < player->lands->length; i++) {
        Card* card = &player->lands->cards[i];
        if (card->tapped) {
            card->tapped = false;
        }
    }
    
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_upkeep});
    return TO_STACK;
}

bool phase_upkeep(TCG* env, unsigned char atn) {
    printf("PHASE_UPKEEP\n");

    Player* player = env->turn == 0 ? env->my_player : env->op_player;

    for (int i = 0; i < player->board->length; i++) {
         Card* card = &player->board->cards[i];
         if (card->type == TYPE_CREATURE && card->summoning_sickness) {
             card->summoning_sickness = false;
         }
    }
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_draw});
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = resolve_priority});
    return TO_STACK;
}

bool phase_draw(TCG* env, unsigned char atn) {
    printf("PHASE_DRAW\n");

    Player* player = env->turn == 0 ? env->my_player : env->op_player;

    draw_card(env, player->deck, player->hand);
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = resolve_priority});
    return TO_STACK;
}

bool phase_play(TCG* env, unsigned char atn) {
    printf("PHASE_PLAY\n");
    Player* player = (env->turn == 0) ? env->my_player : env->op_player;

    if (!has_valid_moves(env, player)) {
        printf("\t No valid moves. Skip to next phase.\n");
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
        env->participate_in_priority = !env->participate_in_priority;
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
        return TO_USER;
    } else if (atn >= hand->length) {
        printf("\t Invalid action: %i\n. Hand length: %i\n", atn, hand->length);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
        return TO_USER;
    }

    Card card = player->hand->cards[atn];

    if (!can_play_card(env, player, card)) {
        printf("Condition for playing this card not met\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
        return TO_USER;
    }

    tap_lands_for_mana(player->lands, player->mana, card.cost);
    player->mana -= card.cost;

    switch (card.type) {
        case TYPE_LAND:
            move_card(env, player->hand, player->lands, atn);
            player->land_played = true;
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
            return TO_USER;
        case TYPE_CREATURE:
            move_card(env, player->hand, player->board, atn);
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
            return TO_STACK;
        case TYPE_INSTANT:
            move_card(env, player->hand, player->graveyard, atn);
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
            push(env->stack, (StackItem){.type = STACK_EFFECT, .effect = card.effect});
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = resolve_priority});
            return TO_STACK;
        case TYPE_SORCERY:
            move_card(env, player->hand, player->graveyard, atn);
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
            push(env->stack, (StackItem){.type = STACK_EFFECT, .effect = card.effect});
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = resolve_priority});
            return TO_STACK;
        default:
            printf("\t Unknown card type\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_play});
            return TO_USER;
    }
}

bool phase_attack(TCG* env, unsigned char atn) {
    printf("PHASE_ATTACK\n");
    Player* player = (env->turn == 0) ? env->my_player : env->op_player;

    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_attack});
        return TO_USER;
    } else if (atn == ACTION_ENTER) {
        printf("\t Attacks confirmed. Phase end\n");
        env->turn = 1 - env->turn;
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
        return TO_STACK;
    } else if (atn >= player->board->length || player->board->cards[atn].type != TYPE_CREATURE) {
        printf("\t Invalid action %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_attack});
        return TO_USER;
    }
    if (player->board->cards[atn].summoning_sickness) {
            printf("\t Cannot attack with summoning sickness\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_attack});
            return TO_USER;
        }

    printf("\t Setting attacker %i\n", atn);
    player->board->cards[atn].attacking = !player->board->cards[atn].attacking;
    player->board->cards[atn].tapped = !player->board->cards[atn].tapped;
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_attack});
    return TO_USER;
}

// Function to handle the blocking phase in the game
bool phase_block(TCG* env, unsigned char atn) {
    printf("PHASE_BLOCK\n");

    Player* player = (env->turn == 0) ? env->my_player : env->op_player;
    Player* opponent = (env->turn == 0) ? env->op_player : env->my_player;

    // Skip non-attacking cards
    while (env->block_idx < player->board->length && !player->board->cards[env->block_idx].attacking) {
        printf("\t Skipping block for %i (not attacking)\n", env->block_idx);
        env->block_idx++;
    }

    bool can_block = false;
    for (int i = 0; i < opponent->board->length; i++) {
        Card card = opponent->board->cards[i];
        if (card.type == TYPE_CREATURE && !card.tapped && (card.defending == -1 || card.defending == env->block_idx)) {
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
            Card* attacker = &player->board->cards[atk];

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
                Card* defender = &opponent->board->cards[def];
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
                    move_card(env, opponent->board, opponent->graveyard, def); // Move defender to graveyard
                } else {
                    attacker_health -= defender_attack;
                    attacker_attack = 0;
                }
                // Check if attacker is defeated
                if (attacker_health <= 0) {
                    move_card(env, player->board, player->graveyard, atk); // Move attacker to graveyard
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
            if (player->board->cards[i].type == TYPE_CREATURE) {
                player->board->cards[i].attacking = false;
            }
        }
        for (int i = 0; i < opponent->board->length; i++) {
            if (opponent->board->cards[i].type == TYPE_CREATURE) {
                opponent->board->cards[i].defending = -1;
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
    } else if (atn >= opponent->board->length) {
        printf("\t Invalid block action %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
        return TO_USER;
    }

    // Validate the selected defender card for blocking
    for (int i = 0; i < env->block_idx; i++) {
        if (opponent->board->cards[atn].defending == i) {
            printf("\t Already blocked\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
            return TO_USER;
        }
        if (opponent->board->cards[atn].tapped) {
            printf("\t Cannot block with tapped card\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
            return TO_USER;
        }
        if (opponent->board->cards[atn].type != TYPE_CREATURE) {
            printf("\t Cannot block with non-creature card\n");
            push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block});
            return TO_USER;
        }
    }

    // Set the defending state for the selected card
    printf("\t Blocking index %i with %i\n", env->block_idx, atn);
    Card* card = &opponent->board->cards[atn];
    if (card->defending == env->block_idx) {
        card->defending = -1;
    } else {
        card->defending = env->block_idx;
    }
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = phase_block}); // Repeat the blocking phase
    return TO_USER;
}

void pass_priority(TCG* env) {
    env->priority = 1 - env->priority;
    env->priority_passed = true;
}

bool resolve_priority(TCG* env, unsigned char atn) {
    printf("resolve_priority\n");
    Player* player = (env->priority == 0) ? env->my_player : env->op_player;

    if (atn == ACTION_NOOP) {
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = resolve_priority});
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
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = resolve_priority});
        return TO_USER;
    } else if (atn >= plauyer->hand->length) {
        printf("\t Invalid action: %i\n", atn);
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = resolve_priority});
        return TO_USER;
    } else if (player->hand->cards[atn].type != TYPE_INSTANT) {
        printf("\t Can only play Instant during priority\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = resolve_priority});
        return TO_USER;
    } 

    Card card = player->hand->cards[atn];

    if (!can_play_card(env, card)) {
        printf("\t Condition not met\n");
        push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = resolve_priority});
        return TO_USER;
    }

    tap_lands_for_mana(player->lands, player->mana, card.cost);

    assert(*mana >= card.cost);
    player->mana -= card.cost;

    printf("\t Activating instant effect\n");
    env->priority_passed = false;
    move_card(env, player->hand, player->graveyard, atn);
    push(env->stack, (StackItem){
        .type = STACK_EFFECT,
        .effect = card.data.spell.effect,
    });
    push(env->stack, (StackItem){.type = STACK_PHASE, .phase_func = resolve_priority});
    return TO_USER;
}

void step(TCG* env, unsigned char atn) {
    printf("Turn: %i, Action: %i\n", env->turn, atn);
    while (true) {
        StackItem item = pop(env->stack);
        if (item.type == STACK_PHASE) {
            bool return_to_user = item.phase_func(env, atn);
            if (return_to_user) {
                StackItem next_item = peek(env->stack);
                if (next_item.type == STACK_PHASE &&
                        next_item.phase_func == resolve_priority &&
                        env->priority == 0 &&
                        !env->participate_in_priority) {
                    atn = ACTION_ENTER;
                    continue;
                }
                return;
            }
        } else if (item.type == STACK_EFFECT) {
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
        draw_card(env, env->player->deck, env->player->hand);
        draw_card(env, env->op_player->deck, env->op_player->hand);
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
    for (int i = 0; i < env->my_player->hand->length; i++) {
        Card card = env->my_player->hand->cards[i];
        int x = card_x(i, env->my_player->hand->length);
        int y = card_y(3);
        render_card(card, x, y);
        if (env->turn == 0) {
            render_label(x, y, i);
        }
    }
}

void render_op_hand(TCG* env) {
    for (int i = 0; i < env->->length; i++) {
        Card card = env->op_player->hand->cards[i];
        int x = card_x(i, env->op_player->hand->length);
        int y = card_y(0);
        render_card(card, x, y);
    }
}

void render_my_board(TCG* env) {
    for (int i = 0; i < env->my_player->board->length; i++) {
        Card card = env->board->cards[i];

        int x = card_x(i, env->my_player->board->length);
        int y = card_y(2);

        render_card(card, x, y);
        if (env->turn == 0) {
            render_label(x, y, i);
        }
    }

    for (int i = 0; i < env->my_player->board->length; i++) {
        Card card = env->my_player->board->cards[i];
        if (card.type != TYPE_CREATURE || card.defending == -1) {
            continue;
        }
        DrawLineEx(
            (Vector2){32+card_x(i, env->my_player->board->length), 64+card_y(2)},
            (Vector2){32+card_x(card.defending, env->op_player->board->length), 64+card_y(1)},
            3.0f, WHITE
        );
    }
}

void render_op_board(TCG* env) {
    for (int i = 0; i < env->op_player->board->length; i++) {

        Card card = env->op_player->board->cards[i];
        int x = card_x(i, env->op_player->board->length);
        int y = card_y(1);

        render_card(card, x, y);
    }
}

void render_my_lands(TCG* env) {
    int num_untapped = 0;
    for (int i = 0; i < env->my_player->lands->length; i++) {
        Card card = env->my_player->lands->cards[i];
        int x = LAND_ZONE_OFFSET;
        int y = card_y(2);
        assert(card.type == TYPE_LAND);
        render_card(card, x, y);
        if (!card.tapped) {
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
        Card card = env->op_player->lands->cards[i];
        int x = (GetScreenWidth() - LAND_ZONE_OFFSET);
        int y = card_y(1);
        render_card(card, x, y);
        if (!card.tapped) {
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
        Card card = env->my_player->graveyard->cards[i];
        int x = (GetScreenWidth() - GRAVEYARD_ZONE_OFFSET);
        int y = card_y(2);
        render_card(card, x, y);
    }
}

void render_op_graveyard(TCG* env) {
    for (int i = 0; i < env->op_player->graveyard->length; i++) {
        Card card = env->op_player->graveyard->cards[i];
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

    DrawText(TextFormat("Health: %i", env->my_player->health), 32, 32, 20, WHITE);
    DrawText(TextFormat("Health: %i", env->op_player->health), 32, 64, 20, WHITE);

    EndDrawing();
}
