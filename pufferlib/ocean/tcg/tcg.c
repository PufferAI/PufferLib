#include "tcg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"

void load_deck_from_json(const char* filename, Zone* deck) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* data = (char*)malloc(length + 1);
    if (!data) {
        perror("Failed to allocate memory");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    fread(data, 1, length, file);
    data[length] = '\0';
    fclose(file);

    cJSON* root = cJSON_Parse(data);
    if (!root) {
        fprintf(stderr, "Error before: %s\n", cJSON_GetErrorPtr());
        free(data);
        exit(EXIT_FAILURE);
    }

    cJSON* card;
    cJSON_ArrayForEach(card, root) {
        const cJSON* name_item = cJSON_GetObjectItem(card, "name");
        const cJSON* type_item = cJSON_GetObjectItem(card, "type");
        const cJSON* count_item = cJSON_GetObjectItem(card, "count");
        const cJSON* cost_item = cJSON_GetObjectItem(card, "cost");
        const cJSON* attack_item = cJSON_GetObjectItem(card, "attack");
        const cJSON* health_item = cJSON_GetObjectItem(card, "health");
        const cJSON* effect_item = cJSON_GetObjectItem(card, "effect");

        if (!name_item || !type_item || !count_item || !cost_item) {
            fprintf(stderr, "Invalid card format. Skipping entry.\n");
            continue;
        }

        const char* name = name_item->valuestring;
        const char* type = type_item->valuestring;
        int count = count_item->valueint;
        double cost = cost_item->valuedouble;

        const char* attack_str = attack_item && cJSON_IsString(attack_item) ? attack_item->valuestring : "";
        const char* health_str = health_item && cJSON_IsString(health_item) ? health_item->valuestring : "";
        const char* effect = effect_item && cJSON_IsString(effect_item) ? effect_item->valuestring : "";

        printf("Loading card %s of type %s with count %i and cost %f\n", name, type, count, cost);

        for (int i = 0; i < count; i++) {
            Card* new_card;
            if (strcmp(type, "creature") == 0) {
                int attack = atoi(attack_str);
                int health = atoi(health_str);

                Effect creature_effect = {0};
                strncpy(creature_effect.description, effect, sizeof(creature_effect.description) - 1);
                creature_effect.trigger = no_trigger;
                creature_effect.condition = no_condition;
                creature_effect.activate = no_activate;

                new_card = allocate_creature(cost, attack, health, creature_effect);
                strncpy(new_card->name, name, sizeof(new_card->name) - 1);

            } else if (strcmp(type, "land") == 0) {
                new_card = allocate_land();
                strncpy(new_card->name, name, sizeof(new_card->name) - 1);

            } else if (strcmp(type, "instant") == 0 || strcmp(type, "sorcery") == 0) {
                Effect spell_effect = {0};
                strncpy(spell_effect.description, effect, sizeof(spell_effect.description) - 1);
                spell_effect.trigger = no_trigger;
                spell_effect.condition = no_condition;
                spell_effect.activate = no_activate;

                if (strcmp(type, "instant") == 0) {
                    new_card = allocate_instant(cost, spell_effect, 0);
                } else {
                    new_card = allocate_sorcery(cost, spell_effect, 0);
                }

                strncpy(new_card->name, name, sizeof(new_card->name) - 1);
            } else {
                fprintf(stderr, "Unknown card type: %s\n", type);
                continue;
            }

            if (deck->length < deck->max) {
                deck->cards[deck->length++] = *new_card;
            } else {
                fprintf(stderr, "Deck is full, cannot add more cards.\n");
                free(new_card);
            }
        }
    }

    printf("Made it to the end of the loop\n");

    cJSON_Delete(root);
    free(data);
}

int main() {
    TCG env = {0}; // MUST ZERO
    allocate_tcg(&env);

    load_deck_from_json("/Users/noahfarr/Documents/PufferLib/pufferlib/ocean/tcg/decks/RoughAndTumble_AFR.json", env.my_deck);
    load_deck_from_json("/Users/noahfarr/Documents/PufferLib/pufferlib/ocean/tcg/decks/RoughAndTumble_AFR.json", env.op_deck);

    printf("Successfully loaded decks\n");

    reset(&env);

    init_client(&env);

    int atn = -1;
    while (!WindowShouldClose()) {
        if (atn != -1) {
            step(&env, atn);
            atn = -1;
        }

        if (env.priority == 0) {
            if (IsKeyPressed(KEY_ONE)) atn = 0;
            if (IsKeyPressed(KEY_TWO)) atn = 1;
            if (IsKeyPressed(KEY_THREE)) atn = 2;
            if (IsKeyPressed(KEY_FOUR)) atn = 3;
            if (IsKeyPressed(KEY_FIVE)) atn = 4;
            if (IsKeyPressed(KEY_SIX)) atn = 5;
            if (IsKeyPressed(KEY_SEVEN)) atn = 6;
            if (IsKeyPressed(KEY_EIGHT)) atn = 7;
            if (IsKeyPressed(KEY_NINE)) atn = 8;
            if (IsKeyPressed(KEY_ZERO)) atn = 9;
            if (IsKeyPressed(KEY_ENTER)) atn = 10;
            if (IsKeyPressed(KEY_SPACE)) atn = 12;
        } else if (env.priority == 1) {
            atn = rand() % 12;
        }
 
        render(&env);
    }
    free_tcg(&env);
    return 0;
}
