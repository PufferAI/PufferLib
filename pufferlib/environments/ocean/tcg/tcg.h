#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include "raylib.h"

#define HAND_SIZE 10
#define BOARD_SIZE 10
#define DECK_SIZE 60

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
};

typedef struct CardArray CardArray;
struct CardArray {
    Card* cards;
    int length;
    int max;
};

typedef struct TCG TCG;
struct TCG {
    int my_health;
    int op_health;

    bool attackers[BOARD_SIZE];
    bool defenders[BOARD_SIZE][BOARD_SIZE];

    int turn;
    int phase;
    int block_idx;

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
    if (env->turn == 0) {
        draw_card(env->my_deck, env->my_hand);
    } else {
        draw_card(env->op_deck, env->op_hand);
    }
    env->phase = PHASE_PLAY;
}

void step(TCG* env, unsigned char atn) {
    CardArray* hand = (env->turn == 0) ? env->my_hand : env->op_hand;
    CardArray* board = (env->turn == 0) ? env->my_board : env->op_board;
    int* health = (env->turn == 0) ? &env->my_health : &env->op_health;

    if (env->phase == PHASE_PLAY) {
        if (atn >= hand->length) {
            return;
        }

        if (board->length == BOARD_SIZE) {
            env->phase = PHASE_ATTACK;
            return;
        }

        Card card = hand->cards[atn];
        if (card.is_land) {
            board->cards[board->length] = card;
            board->length += 1;
            hand->cards[atn].remove = true;
            condense_card_array(hand);
            env->phase = PHASE_ATTACK;
            return;
        }
            
        int mana = 0;
        for (int i = 0; i < board->length; i++) {
            if (board->cards[i].is_land) {
                mana += 1;
            }
        }

        if (mana >= card.cost) {
            board->cards[board->length] = card;
            board->length += 1;
            hand->cards[atn].remove = true;
            condense_card_array(hand);
            env->phase = PHASE_ATTACK;
            return;
        }
        return;
    }

    if (env->phase == PHASE_ATTACK) {
        if (atn == ACTION_ENTER) {
            printf("Attacking enter\n");
            env->phase = PHASE_BLOCK;
        } else if (atn >= board->length) {
            return;
        } else if (board->cards[atn].is_land) {
            return;
        } else {
            env->attackers[atn] = !env->attackers[atn];
        }
    }
    if (env->phase == PHASE_BLOCK) {
        CardArray* defender_board = (env->turn == 0) ? env->op_board : env->my_board;
        printf("Blocking %i, board len: %i\n", env->block_idx, defender_board->length);
        if (env->block_idx == defender_board->length) {
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
                printf("Reducing healthy by %i\n", attacker_attack);
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
        }
        if (atn == ACTION_ENTER) {
            printf("Blocking enter\n");
            env->block_idx++;
            return;
        }
        if (atn >= defender_board->length) {
            printf("Blocking out of range. Length: %i\n", defender_board->length);
            return;
        }
        if (defender_board->cards[atn].is_land) {
            printf("Blocking land\n");
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
        if (env->turn == 0) {
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








