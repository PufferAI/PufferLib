// 3 checks every immediate response
enum {
    BOT_RANDOM,
    BOT_SURVIVAL,
    BOT_SPACE,
    BOT_MINIMAX,
    BOT_LEVELS
};

static const TronAction BOT_ACTIONS[3] = {STRAIGHT, LEFT, RIGHT};

TronAction bot_action(const TronGame *game, int player, int level,
                      unsigned int *rng, BotCache *cache) {
    if (game->outcome != PLAYING) return STRAIGHT;
    if (level == BOT_RANDOM) return BOT_ACTIONS[rand_r(rng) % 3];

    int opp = other_player(player);
    TronAction best[3];
    int best_n = 0;
    // Level 1 avoids an immediate crash against a straight opponent
    if (level == BOT_SURVIVAL) {
        for (int i = 0; i < 3; i++) {
            TronGame sim = *game;
            TronActions atn = {.player = {
                                   [PLAYER_CYAN] = STRAIGHT,
                                   [PLAYER_RED] = STRAIGHT,
                               }};
            atn.player[player] = BOT_ACTIONS[i];
            step(&sim, atn);
            if (sim.outcome == PLAYING || sim.outcome == player_win(player)) {
                best[best_n++] = BOT_ACTIONS[i];
            }
        }
        if (best_n == 0) return BOT_ACTIONS[rand_r(rng) % 3];
        return best[rand_r(rng) % best_n];
    }

    // One minimax call consumes at most 3 candidates * 3 replies * 2 fills = 18
    // marks. Reset above 230 so uint8_t cannot wrap during a call
    if (cache->mark > 230) {
        memset(cache->seen, 0, sizeof(cache->seen));
        cache->mark = 0;
    }
    uint16_t queue[CELLS];
    int best_value = -(CELLS + 2);
    // Level 2 assumes straight; level 3 checks all replies
    int responses = level == BOT_SPACE ? 1 : 3;
    for (int i = 0; i < 3; i++) {
        int worst = CELLS + 2;
        for (int j = 0; j < responses; j++) {
            TronGame sim = *game;
            TronActions atn = {.player = {
                                   [PLAYER_CYAN] = STRAIGHT,
                                   [PLAYER_RED] = STRAIGHT,
                               }};
            atn.player[player] = BOT_ACTIONS[i];
            atn.player[opp] = BOT_ACTIONS[j];
            step(&sim, atn);

            int value = 0;
            if (sim.outcome != DRAW) {
                if (sim.outcome != PLAYING) {
                    value = sim.outcome == player_win(player)
                                ? CELLS + 1
                                : -(CELLS + 1);
                } else {
                    int own = trail_index(sim.x[player], sim.y[player]);
                    int opponent = trail_index(sim.x[opp], sim.y[opp]);
                    int own_size = flood(&sim, own, opponent, cache->seen, queue,
                                         ++cache->mark);
                    int opponent_size = flood(
                        &sim, opponent, own,
                        cache->seen, queue, ++cache->mark);
                    value = own_size - opponent_size;
                }
            }
            if (value < worst) worst = value;

            if (worst < best_value) break;
        }
        if (worst > best_value) {
            best_value = worst;
            best[0] = BOT_ACTIONS[i];
            best_n = 1;
        } else if (worst == best_value) {
            best[best_n++] = BOT_ACTIONS[i];
        }
    }
    // avoids a straight/left/right bias
    return best[rand_r(rng) % best_n];
}
