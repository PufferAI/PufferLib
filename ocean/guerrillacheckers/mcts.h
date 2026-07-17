// MCTS (UCT) opponent for Guerrilla Checkers, ported from
// nico/guerrillacheckers/src/mcts.nim. Included by guerrillacheckers.h AFTER the
// GuerrillaCheckers struct and the game primitives it relies on are defined:
// gc_rand, gc_enumerate_legal, gc_apply_action, gc_action_capture_score, and the
// GC_* board/action constants. Not meant to be included standalone.
//
// Perfect-information single-tree UCT: wins are stored from the perspective of
// the player who moved into a node, so maximizing a child's win rate selects the
// best move for the player to move at its parent. One tree is built per decision.

#ifndef GUERRILLACHECKERS_MCTS_H
#define GUERRILLACHECKERS_MCTS_H

// MCTS simulation policy: uniform-random (faithful to the reference) or a
// greedy capture-maximizing playout (much stronger on the wide Guerrilla side,
// at a higher per-iteration cost).
enum {
    GC_MCTS_ROLLOUT_RANDOM = 0,
    GC_MCTS_ROLLOUT_GREEDY = 1,
};

// UCB1 exploration constant from the reference engine (~sqrt(2)/2).
#define GC_MCTS_DEFAULT_EXPLORATION 0.7f
typedef struct GcMctsNode {
    int move;              // action leading to this node (-1 at the root)
    int parent;            // pool index of parent (-1 at the root)
    int first_child;       // pool index of first child (-1)
    int next_sibling;      // pool index of next sibling (-1)
    int child_count;
    int player_just_moved; // GC_NONE at the root
    int visits;
    int select_count;      // UCB1 log numerator; same value old siblings shared
    double wins;
} GcMctsNode;

// Collect legal actions on an MCTS clone and, if the side to move is stuck,
// resolve it as a loss (the env applies this in gc_prepare_turn, which MCTS
// does not call). Returns the number of legal actions.
static int gc_mcts_legal(GuerrillaCheckers* s, int* out) {
    if (s->game_over) return 0;
    int n = gc_enumerate_legal(s, out);
    if (n == 0) {
        gc_apply_no_legal_loss(s, n);
    }
    return n;
}

static int gc_mcts_has_child(GcMctsNode* pool, int node, int move) {
    for (int c = pool[node].first_child; c >= 0; c = pool[c].next_sibling) {
        if (pool[c].move == move) return 1;
    }
    return 0;
}

// UCB1 child selection. All children are legal here (perfect information), so we
// consider every child. In this tree all siblings share the same availability
// count, so the parent tracks the old per-sibling value.
static int gc_mcts_ucb_select(GcMctsNode* pool, int node, double exploration) {
    int best = -1;
    double best_score = -1.0e300;
    double select_count = (double)pool[node].select_count;
    for (int c = pool[node].first_child; c >= 0; c = pool[c].next_sibling) {
        double visits = (double)pool[c].visits;
        double score = pool[c].wins / visits +
            exploration * sqrt(log(select_count) / visits);
        if (score > best_score) {
            best_score = score;
            best = c;
        }
    }
    pool[node].select_count += 1;
    return best;
}

// Choose a playout move on state `s`: uniform-random, or (greedy mode) the
// highest immediate-capture move with a random tie-break. `env` supplies rng and
// the rollout mode; `s` is the rollout clone whose side-to-move is scored.
static int gc_mcts_rollout_pick(GuerrillaCheckers* env, GuerrillaCheckers* s,
        int* legal, int n) {
    if (env->mcts_rollout != GC_MCTS_ROLLOUT_GREEDY) {
        return legal[gc_rand(env) % (unsigned int)n];
    }
    return gc_greedy_pick(env, s, legal, n);
}

static void gc_mcts_init_node(GcMctsNode* n, int move, int parent, int sibling,
        int player_just_moved) {
    n->move = move;
    n->parent = parent;
    n->first_child = -1;
    n->next_sibling = sibling;
    n->child_count = 0;
    n->player_just_moved = player_just_moved;
    n->visits = 0;
    n->select_count = 1;
    n->wins = 0.0;
}

static int gc_mcts_action(GuerrillaCheckers* env) {
    int root_legal[GC_ACTIONS];
    int root_n = gc_enumerate_legal(env, root_legal);
    if (root_n <= 1) return root_n == 1 ? root_legal[0] : 0;  // nothing to search

    int itermax = env->mcts_iterations > 0 ? env->mcts_iterations : 1;
    double exploration = env->mcts_exploration > 0.0f ?
        (double)env->mcts_exploration : (double)GC_MCTS_DEFAULT_EXPLORATION;

    // Each iteration expands at most one node, so itermax + 1 nodes suffice.
    GcMctsNode* pool = (GcMctsNode*)malloc((size_t)(itermax + 1) * sizeof(GcMctsNode));
    gc_mcts_init_node(&pool[0], -1, -1, -1, GC_NONE);
    int node_count = 1;

    for (int iter = 0; iter < itermax; iter++) {
        // Clone only the board state; MCTS never touches obs/mask/reward buffers.
        GuerrillaCheckers s = *env;
        memset(s.agents, 0, sizeof(s.agents));
        s.client = NULL;

        int node = 0;
        int legal[GC_ACTIONS];
        int n = gc_mcts_legal(&s, legal);

        // Select: descend while the node is fully expanded and non-terminal.
        while (n > 0) {
            int fully_expanded = pool[node].child_count == n;
            if (!fully_expanded) break;
            node = gc_mcts_ucb_select(pool, node, exploration);
            gc_apply_action(&s, pool[node].move);
            n = gc_mcts_legal(&s, legal);
        }

        // Expand: add one random untried move.
        if (n > 0) {
            int untried[GC_ACTIONS];
            int un = 0;
            for (int i = 0; i < n; i++) {
                if (!gc_mcts_has_child(pool, node, legal[i])) untried[un++] = legal[i];
            }
            int m = untried[gc_rand(env) % (unsigned int)un];
            int player = s.player_to_move;
            gc_apply_action(&s, m);
            int child = node_count++;
            gc_mcts_init_node(&pool[child], m, node, pool[node].first_child, player);
            pool[node].first_child = child;
            pool[node].child_count++;
            node = child;
        }

        // Simulate: rollout to a terminal state (random or greedy playout).
        while (!s.game_over) {
            int rollout[GC_ACTIONS];
            int rn = gc_mcts_legal(&s, rollout);
            if (rn == 0) break;  // game_over was set by gc_mcts_legal
            gc_apply_action(&s, gc_mcts_rollout_pick(env, &s, rollout, rn));
        }

        // Backpropagate the terminal result to the root.
        for (int bn = node; ; bn = pool[bn].parent) {
            pool[bn].visits += 1;
            if (pool[bn].player_just_moved != GC_NONE &&
                    s.winner == pool[bn].player_just_moved) {
                pool[bn].wins += 1.0;
            }
            if (pool[bn].parent < 0) break;
        }
    }

    // Best move = most-visited child of the root.
    int best_move = root_legal[0];
    int best_visits = -1;
    for (int c = pool[0].first_child; c >= 0; c = pool[c].next_sibling) {
        if (pool[c].visits > best_visits) {
            best_visits = pool[c].visits;
            best_move = pool[c].move;
        }
    }
    free(pool);
    return best_move;
}

#endif  // GUERRILLACHECKERS_MCTS_H
