#define _POSIX_C_SOURCE 200809L

#include "tron.h"

int main(void) {
    const float tick_dt = 1.0f / RENDER_TICKS_PER_SECOND;
    const float reset_delay = 0.6f;
    TronGame game = {0};
    TronRenderer renderer = {0};
    BotCache cache[PLAYERS] = {0};
    unsigned int rng = 0x6d2b79f5u;
    float elapsed = 0.0f;
    float crash_age = 1.0f;

    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "PufferLib // Tron");
    SetExitKey(KEY_NULL);
    SetTargetFPS(RENDER_FPS);
    reset(&game);
    renderer_init(&renderer, &game);

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        if (game.outcome != PLAYING) crash_age += dt;
        if (game.outcome != PLAYING && crash_age >= reset_delay) {
            reset(&game);
            renderer.previous = game;
            trails_reset(renderer.trail, &game);
            elapsed = 0.0f;
            crash_age = 1.0f;
        }

        if (game.outcome == PLAYING) elapsed += dt < tick_dt ? dt : tick_dt;
        if (elapsed >= tick_dt) {
            renderer.previous = game;
            step(&game,
                 (TronActions){.player = {
                                    [PLAYER_CYAN] = bot_action(
                                        &game, PLAYER_CYAN, BOT_MINIMAX,
                                        &rng, &cache[PLAYER_CYAN]),
                                    [PLAYER_RED] = bot_action(
                                        &game, PLAYER_RED, BOT_MINIMAX,
                                        &rng, &cache[PLAYER_RED]),
                                }});
            trails_record(renderer.trail, &game);
            elapsed -= tick_dt;
            if (game.outcome != PLAYING) crash_age = 0.0f;
        }

        float lerp = game.tick && game.outcome == PLAYING
                         ? elapsed / tick_dt
                         : 1.0f;
        renderer_draw(&renderer, &game, lerp, crash_age);
    }

    UnloadTexture(renderer.cycle);
    UnloadTexture(renderer.puffer);
    UnloadTexture(renderer.crash);
    CloseWindow();
}
