/* Overcooked Render: All rendering and texture management functions.
 */

 #ifndef OVERCOOKED_RENDER_H
 #define OVERCOOKED_RENDER_H
 
 #include "overcooked_types.h"
 #include "overcooked_items.h"
 
 static Color get_agent_color(int held_item) {
     switch (held_item) {
         case NO_ITEM:
             return BLUE;      // Blue when empty-handed
         case TOMATO:
             return (Color){200, 50, 50, 255};   // Dark red when holding tomato
         case ONION:
             return (Color){255, 200, 100, 255}; // Light orange when holding onion
         case PLATE:
             return (Color){200, 200, 220, 255}; // Light blue-gray when holding plate
         case SOUP:
             return (Color){255, 140, 0, 255};   // Orange when holding soup
         case PLATED_SOUP:
             return (Color){255, 165, 0, 255};   // Brighter orange when holding plated soup
         default:
             return BLUE;      // Default to blue
     }
 }
 
 static void unload_textures(Client* client) {
     UnloadTexture(client->floor);
     UnloadTexture(client->counter);
     UnloadTexture(client->pot);
     UnloadTexture(client->serve);
     UnloadTexture(client->onions_box);
     UnloadTexture(client->tomatoes_box);
     UnloadTexture(client->dishes_box);
     UnloadTexture(client->wall);
 
     UnloadTexture(client->onion);
     UnloadTexture(client->tomato);
     UnloadTexture(client->dish);
     UnloadTexture(client->soup_onion);
     UnloadTexture(client->soup_tomato);
     UnloadTexture(client->soup_onion_dish);
     UnloadTexture(client->soup_tomato_dish);
 
     UnloadTexture(client->soup_onion_cooking_1);
     UnloadTexture(client->soup_onion_cooking_2);
     UnloadTexture(client->soup_onion_cooking_3);
     UnloadTexture(client->soup_onion_cooked);
     UnloadTexture(client->soup_tomato_cooking_1);
     UnloadTexture(client->soup_tomato_cooking_2);
     UnloadTexture(client->soup_tomato_cooking_3);
     UnloadTexture(client->soup_tomato_cooked);
 
     UnloadTexture(client->chef_north);
     UnloadTexture(client->chef_south);
     UnloadTexture(client->chef_east);
     UnloadTexture(client->chef_west);
     UnloadTexture(client->chef_north_onion);
     UnloadTexture(client->chef_south_onion);
     UnloadTexture(client->chef_east_onion);
     UnloadTexture(client->chef_west_onion);
     UnloadTexture(client->chef_north_tomato);
     UnloadTexture(client->chef_south_tomato);
     UnloadTexture(client->chef_east_tomato);
     UnloadTexture(client->chef_west_tomato);
     UnloadTexture(client->chef_north_dish);
     UnloadTexture(client->chef_south_dish);
     UnloadTexture(client->chef_east_dish);
     UnloadTexture(client->chef_west_dish);
     UnloadTexture(client->chef_north_soup_onion);
     UnloadTexture(client->chef_south_soup_onion);
     UnloadTexture(client->chef_east_soup_onion);
     UnloadTexture(client->chef_west_soup_onion);
     UnloadTexture(client->chef_north_soup_tomato);
     UnloadTexture(client->chef_south_soup_tomato);
     UnloadTexture(client->chef_east_soup_tomato);
     UnloadTexture(client->chef_west_soup_tomato);
 
     CloseWindow();
 }
 
 void c_render(Overcooked* env) {
     if (env->client == NULL) {
         int window_width = env->width * env->grid_size + 350;
         int window_height = env->height * env->grid_size + 80;
         InitWindow(window_width, window_height, "PufferLib Overcooked");
         SetTargetFPS(16);
         env->client = (Client*)calloc(1, sizeof(Client));
 
         env->client->floor = LoadTexture("resources/overcooked/terrain/floor.png");
         env->client->counter = LoadTexture("resources/overcooked/terrain/counter.png");
         env->client->pot = LoadTexture("resources/overcooked/terrain/pot.png");
         env->client->serve = LoadTexture("resources/overcooked/terrain/serve.png");
         env->client->onions_box = LoadTexture("resources/overcooked/terrain/onions.png");
         env->client->tomatoes_box = LoadTexture("resources/overcooked/terrain/tomatoes.png");
         env->client->dishes_box = LoadTexture("resources/overcooked/terrain/dishes.png");
         env->client->wall = LoadTexture("resources/overcooked/terrain/counter.png");
 
         env->client->onion = LoadTexture("resources/overcooked/objects/onion.png");
         env->client->tomato = LoadTexture("resources/overcooked/objects/tomato.png");
         env->client->dish = LoadTexture("resources/overcooked/objects/dish.png");
         env->client->soup_onion = LoadTexture("resources/overcooked/objects/soup-onion-cooked.png");
         env->client->soup_tomato = LoadTexture("resources/overcooked/objects/soup-tomato-cooked.png");
         env->client->soup_onion_dish = LoadTexture("resources/overcooked/objects/soup-onion-dish.png");
         env->client->soup_tomato_dish = LoadTexture("resources/overcooked/objects/soup-tomato-dish.png");
 
         env->client->soup_onion_cooking_1 = LoadTexture("resources/overcooked/objects/soup-onion-1-cooking.png");
         env->client->soup_onion_cooking_2 = LoadTexture("resources/overcooked/objects/soup-onion-2-cooking.png");
         env->client->soup_onion_cooking_3 = LoadTexture("resources/overcooked/objects/soup-onion-3-cooking.png");
         env->client->soup_onion_cooked = LoadTexture("resources/overcooked/objects/soup-onion-cooked.png");
         env->client->soup_tomato_cooking_1 = LoadTexture("resources/overcooked/objects/soup-tomato-1-cooking.png");
         env->client->soup_tomato_cooking_2 = LoadTexture("resources/overcooked/objects/soup-tomato-2-cooking.png");
         env->client->soup_tomato_cooking_3 = LoadTexture("resources/overcooked/objects/soup-tomato-3-cooking.png");
         env->client->soup_tomato_cooked = LoadTexture("resources/overcooked/objects/soup-tomato-cooked.png");
 
         env->client->chef_north = LoadTexture("resources/overcooked/chefs/NORTH.png");
         env->client->chef_south = LoadTexture("resources/overcooked/chefs/SOUTH.png");
         env->client->chef_east = LoadTexture("resources/overcooked/chefs/EAST.png");
         env->client->chef_west = LoadTexture("resources/overcooked/chefs/WEST.png");
         env->client->chef_north_onion = LoadTexture("resources/overcooked/chefs/NORTH-onion.png");
         env->client->chef_south_onion = LoadTexture("resources/overcooked/chefs/SOUTH-onion.png");
         env->client->chef_east_onion = LoadTexture("resources/overcooked/chefs/EAST-onion.png");
         env->client->chef_west_onion = LoadTexture("resources/overcooked/chefs/WEST-onion.png");
         env->client->chef_north_tomato = LoadTexture("resources/overcooked/chefs/NORTH-tomato.png");
         env->client->chef_south_tomato = LoadTexture("resources/overcooked/chefs/SOUTH-tomato.png");
         env->client->chef_east_tomato = LoadTexture("resources/overcooked/chefs/EAST-tomato.png");
         env->client->chef_west_tomato = LoadTexture("resources/overcooked/chefs/WEST-tomato.png");
         env->client->chef_north_dish = LoadTexture("resources/overcooked/chefs/NORTH-dish.png");
         env->client->chef_south_dish = LoadTexture("resources/overcooked/chefs/SOUTH-dish.png");
         env->client->chef_east_dish = LoadTexture("resources/overcooked/chefs/EAST-dish.png");
         env->client->chef_west_dish = LoadTexture("resources/overcooked/chefs/WEST-dish.png");
 
         env->client->chef_north_soup_onion = LoadTexture("resources/overcooked/chefs/NORTH-soup-onion.png");
         env->client->chef_south_soup_onion = LoadTexture("resources/overcooked/chefs/SOUTH-soup-onion.png");
         env->client->chef_east_soup_onion = LoadTexture("resources/overcooked/chefs/EAST-soup-onion.png");
         env->client->chef_west_soup_onion = LoadTexture("resources/overcooked/chefs/WEST-soup-onion.png");
         env->client->chef_north_soup_tomato = LoadTexture("resources/overcooked/chefs/NORTH-soup-tomato.png");
         env->client->chef_south_soup_tomato = LoadTexture("resources/overcooked/chefs/SOUTH-soup-tomato.png");
         env->client->chef_east_soup_tomato = LoadTexture("resources/overcooked/chefs/EAST-soup-tomato.png");
         env->client->chef_west_soup_tomato = LoadTexture("resources/overcooked/chefs/WEST-soup-tomato.png");
     }
 
     if (IsKeyDown(KEY_ESCAPE)) exit(0);
 
     BeginDrawing();
     ClearBackground((Color){240, 240, 240, 255});
 
     DrawText(TextFormat("Correct Dishes: %d", (int)env->log.n), 10, 10, 20, BLACK);
     DrawText(TextFormat("Total Dishes: %d", (int)env->log.dishes_served), 10, 35, 20, BLACK);
     DrawText("Recipe: 3 Onions", 10, 60, 16, DARKGRAY);
 
     int grid_offset_y = 80;
     for (int y = 0; y < env->height; y++) {
         for (int x = 0; x < env->width; x++) {
             int idx = y * env->width + x;
             Rectangle dest = {x * env->grid_size, y * env->grid_size + grid_offset_y, env->grid_size, env->grid_size};
 
             if (env->client->floor.id != 0) {
                 DrawTexturePro(env->client->floor,
                     (Rectangle){0, 0, env->client->floor.width, env->client->floor.height},
                     dest, (Vector2){0, 0}, 0, WHITE);
             }
 
             Texture2D* texture = NULL;
             switch (env->grid[idx]) {
                 case COUNTER:
                     texture = &env->client->counter;
                     break;
                 case STOVE:
                     texture = &env->client->pot;
                     break;
                 case CUTTING_BOARD:
                     texture = &env->client->counter;
                     break;
                 case INGREDIENT_BOX:
                     texture = &env->client->onions_box;
                     break;
                 case SERVING_AREA:
                     texture = &env->client->serve;
                     break;
                 case PLATE_BOX:
                     texture = &env->client->dishes_box;
                     break;
                 case WALL:
                     texture = &env->client->wall;
                     break;
             }
 
             if (texture && texture->id != 0) {
                 DrawTexturePro(*texture,
                     (Rectangle){0, 0, texture->width, texture->height},
                     dest, (Vector2){0, 0}, 0, WHITE);
             }
 
             if (env->grid[idx] == STOVE) {
                 CookingPot* pot = get_pot_at(env, x, y);
                 if (pot && pot->ingredient_count > 0) {
                     Texture2D* cooking_texture = NULL;
 
                     bool is_onion_soup = (pot->num_onions >= pot->num_tomatoes);
                     if (is_onion_soup) {
                         if (pot->ingredient_count <= 1) {
                             cooking_texture = &env->client->soup_onion_cooking_1;
                         } else if (pot->ingredient_count == 2) {
                             cooking_texture = &env->client->soup_onion_cooking_2;
                         } else {
                             cooking_texture = &env->client->soup_onion_cooking_3;
                         }
                     } else {
                         if (pot->ingredient_count <= 1) {
                             cooking_texture = &env->client->soup_tomato_cooking_1;
                         } else if (pot->ingredient_count == 2) {
                             cooking_texture = &env->client->soup_tomato_cooking_2;
                         } else {
                             cooking_texture = &env->client->soup_tomato_cooking_3;
                         }
                     }
 
                     if (pot->cooking_state == COOKING) {
                         float progress = (float)pot->cooking_progress / COOKING_TIME;
 
                         DrawRectangle(x * env->grid_size + 5,
                                     y * env->grid_size + grid_offset_y + env->grid_size - 10,
                                     (env->grid_size - 10) * progress, 3, GREEN);
                         DrawRectangleLines(x * env->grid_size + 5,
                                          y * env->grid_size + grid_offset_y + env->grid_size - 10,
                                          env->grid_size - 10, 3, BLACK);
                     }
                     else if (pot->cooking_state == COOKED) {
                         cooking_texture = is_onion_soup ? &env->client->soup_onion_cooked :
                                                           &env->client->soup_tomato_cooked;
                         DrawText("READY!", x * env->grid_size + 5,
                                y * env->grid_size + grid_offset_y + env->grid_size - 10,
                                8, GREEN);
                     }
                     
                     if (cooking_texture && cooking_texture->id != 0) {
                         Rectangle pot_dest = {
                             x * env->grid_size,
                             y * env->grid_size + grid_offset_y,
                             env->grid_size,
                             env->grid_size
                         };
                         DrawTexturePro(*cooking_texture,
                             (Rectangle){0, 0, cooking_texture->width, cooking_texture->height},
                             pot_dest, (Vector2){0, 0}, 0, WHITE);
                     }
                 }
             }
         }
     }
 
     for (int i = 0; i < env->num_items; i++) {
         Texture2D* texture = NULL;
         switch (env->items[i].type) {
             case TOMATO:
                 texture = &env->client->tomato;
                 break;
             case ONION:
                 texture = &env->client->onion;
                 break;
             case PLATE:
                 texture = &env->client->dish;
                 break;
             case SOUP:
                 texture = &env->client->soup_onion;
                 break;
             case PLATED_SOUP:
                 if (env->items[i].num_onions >= env->items[i].num_tomatoes) {
                     texture = &env->client->soup_onion_dish;
                 } else {
                     texture = &env->client->soup_tomato_dish;
                 }
                 break;
         }
 
         if (texture && texture->id != 0) {
             Rectangle dest = {
                 env->items[i].x * env->grid_size + env->grid_size/4,
                 env->items[i].y * env->grid_size + grid_offset_y + env->grid_size/4,
                 env->grid_size/2,
                 env->grid_size/2
             };
             DrawTexturePro(*texture,
                 (Rectangle){0, 0, texture->width, texture->height},
                 dest, (Vector2){0, 0}, 0, WHITE);
         } else {
             Color item_color = GRAY;
             switch (env->items[i].type) {
                 case TOMATO: item_color = RED; break;
                 case ONION: item_color = YELLOW; break;
                 case PLATE: item_color = WHITE; break;
                 case SOUP: item_color = ORANGE; break;
                 case PLATED_SOUP: item_color = ORANGE; break;
             }
             DrawCircle(
                 env->items[i].x * env->grid_size + env->grid_size/2,
                 env->items[i].y * env->grid_size + grid_offset_y + env->grid_size/2,
                 env->grid_size/4,
                 item_color
             );
         }
     }
 
     for (int agent_idx = 0; agent_idx < env->num_agents; agent_idx++) {
         Agent* agent = &env->agents[agent_idx];
         Texture2D* chef_texture = NULL;
 
         if (agent->held_item == NO_ITEM) {
             switch (agent->facing_direction) {
                 case 0: chef_texture = &env->client->chef_north; break;
                 case 1: chef_texture = &env->client->chef_south; break;
                 case 2: chef_texture = &env->client->chef_west; break;
                 case 3: chef_texture = &env->client->chef_east; break;
             }
         } else if (agent->held_item == ONION) {
             switch (agent->facing_direction) {
                 case 0: chef_texture = &env->client->chef_north_onion; break;
                 case 1: chef_texture = &env->client->chef_south_onion; break;
                 case 2: chef_texture = &env->client->chef_west_onion; break;
                 case 3: chef_texture = &env->client->chef_east_onion; break;
             }
         } else if (agent->held_item == TOMATO) {
             switch (agent->facing_direction) {
                 case 0: chef_texture = &env->client->chef_north_tomato; break;
                 case 1: chef_texture = &env->client->chef_south_tomato; break;
                 case 2: chef_texture = &env->client->chef_west_tomato; break;
                 case 3: chef_texture = &env->client->chef_east_tomato; break;
             }
         } else if (agent->held_item == PLATE) {
             switch (agent->facing_direction) {
                 case 0: chef_texture = &env->client->chef_north_dish; break;
                 case 1: chef_texture = &env->client->chef_south_dish; break;
                 case 2: chef_texture = &env->client->chef_west_dish; break;
                 case 3: chef_texture = &env->client->chef_east_dish; break;
             }
         } else if (agent->held_item == PLATED_SOUP) {
             bool is_onion_soup = (agent->held_soup_onions >= agent->held_soup_tomatoes);
             if (is_onion_soup) {
                 switch (agent->facing_direction) {
                     case 0: chef_texture = &env->client->chef_north_soup_onion; break;
                     case 1: chef_texture = &env->client->chef_south_soup_onion; break;
                     case 2: chef_texture = &env->client->chef_west_soup_onion; break;
                     case 3: chef_texture = &env->client->chef_east_soup_onion; break;
                 }
             } else {
                 switch (agent->facing_direction) {
                     case 0: chef_texture = &env->client->chef_north_soup_tomato; break;
                     case 1: chef_texture = &env->client->chef_south_soup_tomato; break;
                     case 2: chef_texture = &env->client->chef_west_soup_tomato; break;
                     case 3: chef_texture = &env->client->chef_east_soup_tomato; break;
                 }
             }
         }
 
         if (chef_texture && chef_texture->id != 0) {
             Rectangle dest = {
                 agent->x * env->grid_size,
                 agent->y * env->grid_size + grid_offset_y,
                 env->grid_size,
                 env->grid_size
             };
             Color tint = WHITE;
             if (agent_idx == 0) {
                 tint = (Color){255, 255, 255, 255};  // White for player 1
             } else if (agent_idx == 1) {
                 tint = (Color){200, 200, 255, 255};  // Light blue tint for player 2
             } else {
                 tint = (Color){255, 200, 200, 255};  // Light red tint for other players
             }
             DrawTexturePro(*chef_texture,
                 (Rectangle){0, 0, chef_texture->width, chef_texture->height},
                 dest, (Vector2){0, 0}, 0, tint);
         } else {
             Color agent_color = get_agent_color(agent->held_item);
             if (agent_idx == 1) {
                 agent_color = (Color){agent_color.r * 0.8, agent_color.g * 0.8, agent_color.b, agent_color.a};
             }
             DrawRectangle(
                 agent->x * env->grid_size + env->grid_size/4,
                 agent->y * env->grid_size + grid_offset_y + env->grid_size/4,
                 env->grid_size/2,
                 env->grid_size/2,
                 agent_color
             );
 
             int dir_x = agent->x * env->grid_size + env->grid_size/2;
             int dir_y = agent->y * env->grid_size + grid_offset_y + env->grid_size/2;
             int end_x = dir_x, end_y = dir_y;
             switch (agent->facing_direction) {
                 case 0: end_y -= env->grid_size/4; break; // Up
                 case 1: end_y += env->grid_size/4; break; // Down
                 case 2: end_x -= env->grid_size/4; break; // Left
                 case 3: end_x += env->grid_size/4; break; // Right
             }
             DrawLine(dir_x, dir_y, end_x, end_y, BLACK);
 
             DrawText(TextFormat("%d", agent_idx + 1),
                      agent->x * env->grid_size + 2,
                      agent->y * env->grid_size + grid_offset_y + 2,
                      10, BLACK);
         }
     }
 
     int obs_panel_x = env->width * env->grid_size + 10;
     int obs_panel_y = grid_offset_y;
 
     if (env->num_agents > 0) {
         float* obs = &env->observations[0];
 
         DrawText("=== OBSERVATION ARRAY (43 dims) ===", obs_panel_x, obs_panel_y, 11, BLACK);
         obs_panel_y += 18;
 
         DrawText("-- PLAYER (0-33) --", obs_panel_x, obs_panel_y, 10, DARKGREEN);
         obs_panel_y += 13;
 
         DrawText(TextFormat("[0-3] Orient: %.0f %.0f %.0f %.0f",
                  obs[0], obs[1], obs[2], obs[3]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 11;
 
         DrawText(TextFormat("[4-7] Held: %.0f %.0f %.0f %.0f",
                  obs[4], obs[5], obs[6], obs[7]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 11;
 
         DrawText(TextFormat("[8-9] Onion: %.2f, %.2f", obs[8], obs[9]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
         DrawText(TextFormat("[10-11] Dish: %.2f, %.2f", obs[10], obs[11]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
         DrawText(TextFormat("[12-13] Soup: %.2f, %.2f", obs[12], obs[13]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
         DrawText(TextFormat("[14-15] Serve: %.2f, %.2f", obs[14], obs[15]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
         DrawText(TextFormat("[16-17] Empty: %.2f, %.2f", obs[16], obs[17]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
         DrawText(TextFormat("[18-19] Pot: %.2f, %.2f", obs[18], obs[19]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
 
         DrawText(TextFormat("[20-21] PickOnion: %.2f, %.2f", obs[20], obs[21]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
 
         DrawText(TextFormat("[22-23] PickPlate: %.2f, %.2f", obs[22], obs[23]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
 
         DrawText(TextFormat("[24-25] SoupIngr: %.2f, %.2f", obs[24], obs[25]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
 
         DrawText(TextFormat("[26-27] PotIngr: %.2f, %.2f", obs[26], obs[27]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
 
         DrawText(TextFormat("[28] PotExists: %.0f", obs[28]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
 
         DrawText(TextFormat("[29-32] PotState: %.0f %.0f %.0f %.0f",
                  obs[29], obs[30], obs[31], obs[32]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
 
         DrawText(TextFormat("[33] CookTime: %.2f", obs[33]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
 
         DrawText(TextFormat("[34-37] Walls: %.0f %.0f %.0f %.0f",
                  obs[34], obs[35], obs[36], obs[37]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 13;
 
         DrawText("-- TEAMMATE (38-39) --", obs_panel_x, obs_panel_y, 10, DARKBLUE);
         obs_panel_y += 13;
 
         if (env->num_agents > 1) {
             DrawText(TextFormat("[38-39] T.RelPos: %.2f, %.2f", obs[38], obs[39]),
                      obs_panel_x, obs_panel_y, 9, BLACK);
             obs_panel_y += 10;
         } else {
             DrawText("No teammate", obs_panel_x, obs_panel_y, 9, GRAY);
             obs_panel_y += 10;
         }
 
         obs_panel_y += 3;
         DrawText("-- MISC (40-42) --", obs_panel_x, obs_panel_y, 10, DARKGRAY);
         obs_panel_y += 13;
 
         DrawText(TextFormat("[40-41] AbsPos: %.3f, %.3f", obs[40], obs[41]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
 
         DrawText(TextFormat("[42] Reward: %.2f", obs[42]),
                  obs_panel_x, obs_panel_y, 9, BLACK);
         obs_panel_y += 10;
     }
 
     EndDrawing();
 }
 
 #endif // OVERCOOKED_RENDER_H
 