#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>
#include "raylib.h"

#define HAND_SIZE 10
#define BOARD_SIZE 10
#define DECK_SIZE 60
#define STACK_SIZE 100

#define ACTION_ENTER 10

#define PHASE_DRAW 0
#define PHASE_PLAY 1
#define PHASE_ATTACK 2
#define PHASE_BLOCK 3

typedef struct Card Card;
struct Card {
    int cost;
    int attack;
    int health;
    bool is_land;
    bool remove;
    bool tapped;
};

typedef struct CardArray CardArray;
struct CardArray {
    Card* cards;
    int length;
    int max;
};

typedef void (*call)(TCG*);

typedef struct TCG TCG;
struct TCG {
    int my_health;
    int op_health;

    int my_mana;
    int op_mana;

    bool my_land_played;
    bool op_land_played;

    bool attackers[BOARD_SIZE];
    bool defenders[BOARD_SIZE][BOARD_SIZE];

    int turn;
    int phase;
    int block_idx;

    call stack[STACK_SIZE];
    int stack_idx;

    CardArray* my_hand;
    CardArray* op_hand;
    CardArray* my_board;
    CardArray* op_board;
    CardArray* my_deck;
    CardArray* op_deck;
};

CardArray* allocate_card_array(int max) {
    CardArray* hand = (CardArray*)calloc(1, sizeof(CardArray));
    hand->cards = (Card*)calloc(max, sizeof(Card));
    hand->max = max;
    return hand;
}

void free_card_array(CardArray* hand) {
    free(hand->cards);
    free(hand);
}

void condense_card_array(CardArray* hand) {
    int idx = 0;
    for (int i = 0; i < hand->length; i++) {
        if (!hand->cards[i].remove) {
            hand->cards[idx] = hand->cards[i];
            idx += 1;
        }
    }
    hand->length = idx;
}

void allocate_tcg(TCG* env) {
    env->my_hand = allocate_card_array(HAND_SIZE);
    env->op_hand = allocate_card_array(HAND_SIZE);
    env->my_board = allocate_card_array(BOARD_SIZE);
    env->op_board = allocate_card_array(BOARD_SIZE);
    env->my_deck = allocate_card_array(DECK_SIZE);
    env->op_deck = allocate_card_array(DECK_SIZE);
}

void free_tcg(TCG* env) {
    free_card_array(env->my_hand);
    free_card_array(env->op_hand);
    free_card_array(env->my_board);
    free_card_array(env->op_board);
    free_card_array(env->my_deck);
    free_card_array(env->op_deck);
}

void randomize_deck(CardArray* deck) {
    for (int i = 0; i < deck->length; i++) {
        if (rand() % 3 == 0) {
            deck->cards[i].is_land = true;
        } else {
            int cost = rand() % 6;
            deck->cards[i].cost = cost;
            deck->cards[i].attack = cost + 1;
            deck->cards[i].health = cost + 1;
        }
    }
}

void draw_card(CardArray* deck, CardArray* hand) {
    if (deck->length == 0) {
        return;
    }
    Card card = deck->cards[deck->length - 1];
    hand->cards[hand->length] = card;
    deck->length -= 1;
    hand->length += 1;
}

bool can_attack(CardArray* board) {
    for (int i = 0; i < board->length; i++) {
        if (!board->cards[i].is_land) {
            return true;
        }
    }
    return false;
}

void reset(TCG* env) {
    env->my_deck->length = DECK_SIZE;
    env->op_deck->length = DECK_SIZE;
    env->my_hand->length = 0;
    env->op_hand->length = 0;
    env->my_board->length = 0;
    env->op_board->length = 0;
    env->my_health = 20;
    env->op_health = 20;
    randomize_deck(env->my_deck);
    randomize_deck(env->op_deck);
    env->turn = rand() % 2;
    for (int i = 0; i < 5; i++) {
        draw_card(env->my_deck, env->my_hand);
        draw_card(env->op_deck, env->op_hand);
    }
    env->phase = PHASE_DRAW;
    env->stack[env->stack_idx] = phase_play;
    env->stack_idx += 1;
    step(env, 0);
}

void phase_play(TCG* env) {
    if (atn != ACTION_ENTER && atn >= hand->length) {
        printf("\t Invalid action: %i\n. Hand length: %i\n", atn, hand->length);
        return;
    }

    if (board->length == BOARD_SIZE) {
        printf("\t Board full\n");
        env->phase = PHASE_ATTACK;
        return;
    }

    bool can_play = false;
    int min_cost = 99;
    for (int i = 0; i < hand->length; i++) {
        if (hand->cards[i].is_land && !*land_played) {
            can_play = true;
            break;
        } else if (hand->cards[i].cost < min_cost) {
            min_cost = hand->cards[i].cost;
        }
    }

    int tappable = 0;
    for (int i = 0; i < board->length; i++) {
        Card card = board->cards[i];
        if (card.is_land && !card.tapped) {
            tappable += 1;
        }
    }

    // TODO: Clean up
    if (atn == ACTION_ENTER || (!can_play && *mana + tappable < min_cost)) {
        // Can't play anything. Untap lands
        for (int i = 0; i < board->length; i++) {
            Card card = board->cards[i];
            *mana = 0;
            if (card.is_land && card.tapped) {
                board->cards[i].tapped = false;
            }
        }
        *land_played = false;
        env->phase = PHASE_ATTACK;
        if (atn == ACTION_ENTER) {
            printf("\t Manual confirm. Phase end\n");
        } else {
            printf("\t No valid moves. Phase end\n");
        }
        if (!can_attack(board)) {
            env->phase = PHASE_DRAW;
            env->turn = 1 - env->turn;
            printf("\t No valid attacks. Phase end\n");
        }
        return;
    }

    Card card = hand->cards[atn];
    if (card.is_land) {
        if (*land_played) {
            printf("\t Already played land this turn\n");
            return;
        }
        board->cards[board->length] = card;
        board->length += 1;
        *land_played = true;
        hand->cards[atn].remove = true;
        condense_card_array(hand);
        printf("\t Land played\n");
        return;
    }

    if (card.cost > *mana + tappable) {
        printf("\t Not enough mana\n");
        return;
    }

    // Auto tap lands?
    for (int i = 0; i < board->length; i++) {
        if (card.cost <= *mana) {
            break;
        }
        Card card = board->cards[i];
        if (card.is_land && !card.tapped) {
            *mana += 1;
            board->cards[i].tapped = true;
        }
    }

    assert(*mana >= card.cost);
    board->cards[board->length] = card;
    board->length += 1;
    hand->cards[atn].remove = true;
    condense_card_array(hand);
    printf("\t Card played\n");
    return;
}

void step(TCG* env, unsigned char atn) {
    CardArray* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    CardArray* board = (env->turn == 0) ? env->my_board : env->op_board;
    int* health = (env->turn == 0) ? &env->my_health : &env->op_health;
    int* mana = (env->turn == 0) ? &env->my_mana : &env->op_mana;
    bool* land_played = (env->turn == 0) ? &env->my_land_played : &env->op_land_played;

    while (true) {
        printf("Turn: %i, Phase: %i, Action: %i\n", env->turn, env->phase, atn);
        if (env->phase == PHASE_DRAW) {
            draw_card(env->my_deck, env->my_hand);
            env->phase = PHASE_PLAY;
            return;
        }
        if (env->phase == PHASE_PLAY) {
       }

        if (env->phase == PHASE_ATTACK) {
            if (!can_attack(board)) {
                env->phase = PHASE_DRAW;
                env->turn = 1 - env->turn;
                printf("\t No valid attacks. Phase end\n");
                continue;
            }

            if (atn == ACTION_ENTER) {
                printf("\t Attacks confirmed. Phase end\n");
                CardArray* defender_board = (env->turn == 0) ? env->op_board : env->my_board;
                bool can_block = false;
                for (int i = 0; i < defender_board->length; i++) {
                    if (env->defenders[i][atn]) {
                        can_block = true;
                        break;
                    }
                }
                if (!can_block) {
                    env->turn = 1 - env->turn;
                    hand = (env->turn == 0) ? env->my_hand : env->op_hand;
                    board = (env->turn == 0) ? env->my_board : env->op_board;
                    draw_card(env->my_deck, hand);
                    env->phase = PHASE_BLOCK;
                    printf("\t No valid blocks. Phase end\n");
                    return;
                } else {
                    env->phase = PHASE_BLOCK;
                    return;
                }
            } else if (atn >= board->length) {
                printf("\t Invalid action %i\n", atn);
                return;
            } else if (board->cards[atn].is_land) {
                printf("\t Cannot attack with land\n");
                return;
            } else {
                printf("Setting attacker %i\n", atn);
                env->attackers[atn] = !env->attackers[atn];
                return;
            }
        }
        if (env->phase == PHASE_BLOCK) {
            CardArray* defender_board = (env->turn == 0) ? env->op_board : env->my_board;
            printf("\t Attackers:\n");
            printf("\t\t");
            for (int i = 0; i < BOARD_SIZE; i++) {
                printf("%i ", env->attackers[i]);
            }
            printf("\n");
            printf("\t Defenders:\n");
            for (int i = 0; i < BOARD_SIZE; i++) {
                printf("\t\t");
                for (int j = 0; j < BOARD_SIZE; j++) {
                    printf("%i ", env->defenders[i][j]);
                }
                printf("\n");
            }
            printf("\n");

            while (env->block_idx < board->length && !env->attackers[env->block_idx]) {
                printf("\t Skipping block for %i (not attacking)\n", env->block_idx);
                env->block_idx++;
            }

            if (env->block_idx == board->length) {
                printf("\t Nothing left to block with. Phase end\n");
                env->phase = PHASE_DRAW;
                env->turn = 1 - env->turn;
                return;
            }

            if (atn == ACTION_ENTER) {
                printf("\t Manual block confirm %i\n", env->block_idx);
                env->block_idx++;
                return;
            } else if (atn >= defender_board->length) {
                printf("\t Invalid block action %i\n", atn);
                return;
            } else if (defender_board->cards[atn].is_land) {
                printf("\t Cannot block with land\n");
                return;
            }

            for (int i = 0; i < env->block_idx; i++) {
                if (env->defenders[i][atn]) {
                    printf("Already blocked\n");
                    return;
                }
            }
            printf("Blocking index %i with %i\n", env->block_idx, atn);
            env->defenders[env->block_idx][atn] = !env->defenders[env->block_idx][atn];

            if (env->block_idx == board->length) {
                printf("Attacker board length: %i\n", board->length);
                for (int atk = 0; atk < board->length; atk++) {
                    printf("Resolving %i\n", atk);
                    if (!env->attackers[atk]) {
                        printf("Not attacking\n");
                        continue;
                    }
                    Card* attacker = &board->cards[atk];
                    int attacker_attack = attacker->attack;
                    for (int def = 0; def < defender_board->length; def++) {
                        printf("defense %i\n", def);
                        Card* defender = &defender_board->cards[def];
                        attacker->health -= defender->attack;
                        if (defender->health <= attacker_attack) {
                            printf("Attacker wins\n");
                            attacker_attack -= defender->health;
                            defender->health = 0;
                            defender->remove = true;
                        } else {
                            printf("Defender wins\n");
                            defender->health -= attacker_attack;
                            break;
                        }
                    }
                    printf("Reducing health by %i\n", attacker_attack);
                    *health -= attacker_attack;
                }
                condense_card_array(env->my_board);
                condense_card_array(env->op_board);
                env->phase = PHASE_DRAW;
                env->turn = 1 - env->turn;

                CardArray* defender_deck = (env->turn == 0) ? env->my_deck : env->op_deck;
                CardArray* defender_hand = (env->turn == 0) ? env->my_hand : env->op_hand;
                draw_card(defender_deck, defender_hand);

                for (int i = 0; i < BOARD_SIZE; i++) {
                    env->attackers[i] = false;
                    for (int j = 0; j < BOARD_SIZE; j++) {
                        env->defenders[i][j] = false;
                    }
                }
                env->block_idx = 0;
                env->phase = PHASE_PLAY;
                printf(" Set block idx to 0. Phase: %i\n", env->phase);
            }

        }
    }
}

void init_client(TCG* env) {
    InitWindow(1080, 720, "PufferLib Ray TCG");
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
    return 32*(row+1) + 128*row;
}

void render_card(Card* card, int x, int y, Color color) {
    DrawRectangle(x, y, 64, 128, color);
    if (card->is_land) {
        DrawText("Land", x, y, 20, WHITE);
    } else {
        DrawText(TextFormat("%i", card->cost), x, y, 20, WHITE);
        DrawText(TextFormat("%i", card->attack), x, y + 32, 20, WHITE);
        DrawText(TextFormat("%i", card->health), x, y + 64, 20, WHITE);
    }
}

void render_label(int x, int y, int idx) {
    DrawText(TextFormat("%i", (idx+1)%10), x+32, y+96, 20, YELLOW);
}

void render(TCG* env) {
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    
    //printf("My hand length: %i\n", env->my_hand->length);
    //printf("Op hand length: %i\n", env->op_hand->length);
    //printf("My board length: %i\n", env->my_board->length);
    //printf("Op board length: %i\n", env->op_board->length);

    for (int i = 0; i < env->my_hand->length; i++) {
        Card card = env->my_hand->cards[i];
        int x = card_x(i, env->my_hand->length);
        int y = card_y(3);
        render_card(&card, x, y, RED);
        if (env->turn == 0 && env->phase == PHASE_PLAY) {
            render_label(x, y, i);
        }
    }

    for (int i = 0; i < env->my_board->length; i++) {
        Card card = env->my_board->cards[i];
        int x = card_x(i, env->my_board->length);
        int y = card_y(2);
        if (env->turn == 0 && env->attackers[i]) {
            y -= 16;
        }
        render_card(&card, x, y, RED);
        if (env->turn == 0 && env->phase == PHASE_ATTACK) {
            render_label(x, y, i);
        }
    }

    for (int i = 0; i < env->op_board->length; i++) {
        Card card = env->op_board->cards[i];
        int x = card_x(i, env->op_board->length);
        int y = card_y(1);
        if (env->turn == 1 && env->attackers[i]) {
            y += 16;
        }
        render_card(&card, x, y, BLUE);
    }

    for (int atk = 0; atk < env->my_board->length; atk++) {
        for (int def = 0; def < env->op_board->length; def++) {
            if (env->defenders[atk][def]) {
                DrawLine(
                    32+card_x(atk, env->my_board->length), 64+card_y(2),
                    32+card_x(def, env->op_board->length), 64+card_y(1),
                    WHITE
                );
            }
        }
    }


    for (int i = 0; i < env->op_hand->length; i++) {
        Card card = env->op_hand->cards[i];
        int x = card_x(i, env->op_hand->length);
        int y = card_y(0);
        render_card(&card, x, y, BLUE);
    }

    int phase = env->phase;
    int x = GetScreenWidth() - 256;
    int y = 32;
    if (phase == PHASE_DRAW) {
        DrawText("Draw", x, y, 20, WHITE);
    } else if (phase == PHASE_PLAY) {
        DrawText("Play", x, y, 20, WHITE);
    } else if (phase == PHASE_ATTACK) {
        DrawText("Attack", x, y, 20, WHITE);
    } else if (phase == PHASE_BLOCK) {
        DrawText("Block", x, y, 20, WHITE);
    }

    DrawText(TextFormat("Health: %i", env->my_health), 32, 32, 20, WHITE);
    DrawText(TextFormat("Health: %i", env->op_health), 32, GetScreenHeight() - 64, 20, WHITE);

    EndDrawing();
}








