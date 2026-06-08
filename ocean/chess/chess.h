#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <unistd.h> 
#include "raylib.h"

typedef uint64_t Bitboard;
typedef uint64_t Key;
typedef uint32_t Square;
typedef uint32_t Move;
typedef uint32_t Piece;
typedef uint8_t ChessColor;

enum {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE = 64
};

enum { PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };

enum {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING
};

enum { CHESS_WHITE = 0, CHESS_BLACK = 1 };

enum {
    NO_CASTLING = 0,
    WHITE_OO = 1, WHITE_OOO = 2,
    BLACK_OO = 4, BLACK_OOO = 8,
    WHITE_CASTLING = 3, BLACK_CASTLING = 12
};


enum { NORMAL, PROMOTION, ENPASSANT, CASTLING };

enum {
    NORTH = 8, EAST = 1, SOUTH = -8, WEST = -1,
    NORTH_EAST = 9, SOUTH_EAST = -7,
    NORTH_WEST = 7, SOUTH_WEST = -9
};

#define MOVE_NONE 0
#define MOVE_NULL 65

static inline Move make_move(Square from, Square to) {
    return (Move)(to | (from << 6));
}

static inline Move make_promotion(Square from, Square to, int pt) {
    return (Move)(to | (from << 6) | (PROMOTION << 14) | ((pt - KNIGHT) << 12));
}

static inline Move make_enpassant(Square from, Square to) {
    return (Move)(to | (from << 6) | (ENPASSANT << 14));
}

static inline Move make_castling(Square from, Square to) {
    return (Move)(to | (from << 6) | (CASTLING << 14));
}

static inline Square from_sq(Move m) {
    return (Square)((m >> 6) & 0x3f);
}

static inline Square to_sq(Move m) {
    return (Square)(m & 0x3f);
}

static inline int type_of_m(Move m) {
    return (int)(m >> 14);
}

static inline int promotion_type(Move m) {
    return (int)(((m >> 12) & 3) + KNIGHT);
}

static inline Square make_square(int f, int r) {
    return (Square)((r << 3) + f);
}

static inline int file_of(Square s) {
    return (int)(s & 7);
}

static inline int rank_of(Square s) {
    return (int)(s >> 3);
}

static inline Piece make_piece(int c, int pt) {
    return (Piece)((c << 3) + pt);
}

static inline int type_of_p(Piece p) {
    return (int)(p & 7);
}

static inline int color_of(Piece p) {
    return (int)(p >> 3);
}
#define MAX_GAME_PLIES 2048

#define FileABB 0x0101010101010101ULL
#define FileBBB (FileABB << 1)
#define FileCBB (FileABB << 2)
#define FileDBB (FileABB << 3)
#define FileEBB (FileABB << 4)
#define FileFBB (FileABB << 5)
#define FileGBB (FileABB << 6)
#define FileHBB (FileABB << 7)

#define Rank1BB 0xFFULL
#define Rank2BB (Rank1BB << 8)
#define Rank3BB (Rank1BB << 16)
#define Rank4BB (Rank1BB << 24)
#define Rank5BB (Rank1BB << 32)
#define Rank6BB (Rank1BB << 40)
#define Rank7BB (Rank1BB << 48)
#define Rank8BB (Rank1BB << 56)

#define SQ_FEATURES 15

static const char* PIECE_CHARS[] = {
    "",
    "P", "N", "B", "R", "Q", "K",
    "", "",
    "p", "n", "b", "r", "q", "k"
};

static const char* PIECE_FILLED[] = {
    "",
    "♟", "♞", "♝", "♜", "♛", "♚",
    "", "",
    "♟", "♞", "♝", "♜", "♛", "♚"
};


static uint64_t prng_state = 1070372;
static inline uint64_t prng_rand(void) {
    prng_state ^= prng_state >> 12;
    prng_state ^= prng_state << 25;
    prng_state ^= prng_state >> 27;
    return prng_state * 2685821657736338717ULL;
}

extern Bitboard SquareBB[65];
extern Bitboard PawnAttacks[2][64];
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];
extern Bitboard BetweenBB[64][64];
extern Bitboard LineBB[64][64];

static Bitboard BishopMasks[64];
static uint64_t BishopMagics[64];
static int BishopShifts[64];
static Bitboard BishopTable[64 * 512];
static Bitboard* BishopAttacks[64];
static const uint64_t BISHOP_MAGICS[64] = {
    9368648609924554880ULL, 9009475591934976ULL,     4504776450605056ULL,
    1130334595844096ULL,    1725202480235520ULL,     288516396277699584ULL,
    613618303369805920ULL,  10168455467108368ULL,    9046920051966080ULL,
    36031066926022914ULL,   1152925941509587232ULL,  9301886096196101ULL,
    290536121828773904ULL,  5260205533369993472ULL,  7512287909098426400ULL,
    153141218749450240ULL,  9241386469758076456ULL,  5352528174448640064ULL,
    2310346668982272096ULL, 1154049638051909890ULL,  282645627930625ULL,
    2306405976892514304ULL, 11534281888680707074ULL, 72339630111982113ULL,
    8149474640617539202ULL, 2459884588819024896ULL,  11675583734899409218ULL,
    1196543596102144ULL,    5774635144585216ULL,     145242600416216065ULL,
    2522607328671633440ULL, 145278609400071184ULL,   5101802674455216ULL,
    650979603259904ULL,     9511646410653040801ULL,  1153493285013424640ULL,
    18016048314974752ULL,   4688397299729694976ULL,  9226754220791842050ULL,
    4611969694574863363ULL, 145532532652773378ULL,   5265289125480634376ULL,
    288239448330604544ULL,  2395019802642432ULL,     14555704381721968898ULL,
    2324459974457168384ULL, 23652833739932677ULL,    282583111844497ULL,
    4629880776036450560ULL, 5188716322066279440ULL,  146367151686549765ULL,
    1153170821083299856ULL, 2315697107408912522ULL,  2342448293961403408ULL,
    2309255902098161920ULL, 469501395595331584ULL,   4615626809856761874ULL,
    576601773662552642ULL,  621501155230386208ULL,   13835058055890469376ULL,
    3748138521932726784ULL, 9223517207018883457ULL,  9237736128969216257ULL,
    1127068154855556ULL,
};

static Bitboard RookMasks[64];
static uint64_t RookMagics[64];
static int RookShifts[64];
static Bitboard RookTable[64 * 4096];  
static Bitboard* RookAttacks[64];
static const uint64_t ROOK_MAGICS[64] = {
    612498416294952992ULL,  2377936612260610304ULL,  36037730568766080ULL,
    72075188908654856ULL,   144119655536003584ULL,   5836666216720237568ULL,
    9403535813175676288ULL, 1765412295174865024ULL,  3476919663777054752ULL,
    288300746238222339ULL,  9288811671472386ULL,     146648600474026240ULL,
    3799946587537536ULL,    704237264700928ULL,      10133167915730964ULL,
    2305983769267405952ULL, 9223634270415749248ULL,  10344480540467205ULL,
    9376496898355021824ULL, 2323998695235782656ULL,  9241527722809755650ULL,
    189159985010188292ULL,  2310421375767019786ULL,  4647717014536733827ULL,
    5585659813035147264ULL, 1442911135872321664ULL,  140814801969667ULL,
    1188959108457300100ULL, 288815318485696640ULL,   758869733499076736ULL,
    234750139167147013ULL,  2305924931420225604ULL,  9403727128727390345ULL,
    9223970239903959360ULL, 309094713112139074ULL,   38290492990967808ULL,
    3461016597114651648ULL, 181289678366835712ULL,   4927518981226496513ULL,
    1155212901905072225ULL, 36099167912755202ULL,    9024792514543648ULL,
    4611826894462124048ULL, 291045264466247688ULL,   83880127713378308ULL,
    1688867174481936ULL,    563516973121544ULL,      9227888831703941123ULL,
    703691741225216ULL,     45203259517829248ULL,    693563138976596032ULL,
    4038638777286134272ULL, 865817582546978176ULL,   13835621555058516608ULL,
    11541041685463296ULL,   288511853443695360ULL,   283749161902275ULL,
    176489098445378ULL,     2306124759338845321ULL,  720584805193941061ULL,
    4977040710267061250ULL, 10097633331715778562ULL, 325666550235288577ULL,
    1100057149646ULL,
};

typedef struct {
    Key psq[16][64];
    Key enpassant[8];
    Key castling[16];
    Key side;
} Zobrist;

extern Zobrist zob;

typedef struct {
    Bitboard byTypeBB[7];    // [0]=all, [1-6]=PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING
    Bitboard byColorBB[2];
    uint8_t board[64];
    uint8_t pieceCount[16];
    ChessColor sideToMove;
    uint8_t castlingRights;
    uint8_t epSquare;
    uint8_t rule50;
    Key key;
} Position;

static inline Bitboard pieces(const Position* pos) {
    return pos->byTypeBB[0];
}

static inline Bitboard pieces_p(const Position* pos, int p) {
    return pos->byTypeBB[p];
}

static inline Bitboard pieces_c(const Position* pos, int c) {
    return pos->byColorBB[c];
}

static inline Bitboard pieces_cp(const Position* pos, int c, int p) {
    return pieces_p(pos, p) & pieces_c(pos, c);
}

static inline Piece piece_on(const Position* pos, Square s) {
    return (Piece)pos->board[s];
}

typedef struct {
    Move move;
} ExtMove;

typedef struct {
    ExtMove moves[256];
    int count;
} MoveList;
/*
enum {
    // Relational NNUE tokens
    O_TOKEN_COUNT = 0,                    
    O_TOKEN_DATA  = 2,                  
    // Meta data
    O_SIDE = 130,                      
    O_CASTLE = 132,                   
    O_EP = 148,                      
    O_PICK_PHASE = 213,             
    O_SELECTED_PIECE = 215,        
    O_VALID_PIECES = 279,         
    O_VALID_DESTS = 343,         
    O_VALID_PROMOS = 407,       

    O_SELF_CHECK = 439,
    O_OPP_CHECK = 440,
    O_RULE50 = 441,
    O_REPETITION = 442,
    O_PASS_VALID = 443,

    OBS_SIZE = 444
};
*/
/*enum {
    O_SQUARES = 0,
    O_VALID_PROMOS = 960,
    O_CASTLE = 992,
    O_EP = 993,
    O_PICK_PHASE = 994,
    O_RULE50 = 995,
    O_REPETITION = 996,
    O_PASS_VALID = 997,
    OBS_SIZE = 998
};
*/

/*
Selfplay branch obs layout before embedding approach
enum {
    O_BOARD = 0,
    O_SIDE = 768,
    O_CASTLE = 770,
    O_EP = 786,
    O_PICK_PHASE = 851,
    O_SELECTED_PIECE = 853,
    O_VALID_PIECES = 917,
    O_VALID_DESTS = 981,
    O_VALID_PROMOS = 1045,
    O_SELF_CHECK = 1077,
    O_OPP_CHECK = 1078,
    O_RULE50 = 1079,
    O_REPETITION = 1080,
    O_PASS_VALID = 1081,
    OBS_SIZE = 1082
};
*/
enum {
    O_BOARD = 0,
    O_SIDE = 64,
    O_CASTLE = 65,
    O_EP = 69,
    O_RULE50 = 78,
    O_REPETITION = 79,
    O_SELF_CHECK = 80,
    O_OPP_CHECK = 81,
    O_PICK_PHASE = 82,
    O_SELECTED_PIECE = 83,
    O_VALID_FROM_COUNT = 84,
    O_VALID_FROM = 85,
    O_VALID_TO_COUNT = 101,
    O_VALID_TO = 102,
    O_VALID_PROMOS = 134,
    O_PASS_VALID = 166,
    OBS_SIZE = 167
};

#define CHESS_MAX_VALID_FROM 16
#define CHESS_MAX_VALID_TO   32
#define CHESS_NULL_SQ        64

#define PASS_ACTION 96
#define NUM_ACTIONS 97
enum {
    CHESS_MODE_RANDOM = 0,
    CHESS_MODE_SELFPLAY = 1,
    CHESS_MODE_HUMAN = 2,
    CHESS_MODE_HUMAN_RANDOM = 3,
    CHESS_MODE_MAIA = 4
};

#define CHESS_TAG_SELFPLAY 0
#define CHESS_TAG_HISTORICAL 1
// Multi-bank selfplay: env tags are 1..CHESS_MAX_BANKS, one tag value per
// frozen bank. tag = 0 means pure selfplay env (no historical opponent).
// Backward compat: with num_frozen_banks=1, tag=1 still means "play bank 0".
#define CHESS_MAX_BANKS 8

typedef struct {
    float perf;
    float score;
    float draw_rate;
    float timeout_rate;
    float chess_moves;
    float episode_length;
    float episode_return;
    float invalid_action_rate;
    // Per-slot scores (selfplay only). In match: slot 0 = primary policy A,
    // slot 1 = frozen policy B. In selfplay training both should average ~0.5.
    float slot_0_score;
    float slot_1_score;
    // Per-bank historical tracking. hist_score_bank[b] sums primary's score
    // (1.0 win / 0.5 draw / 0 loss) on historical envs tagged b+1; hist_n_bank
    // counts those games. Python recovers per-bank winrate as score/n. The
    // legacy aggregates hist_score / hist_n sum across all banks for backward
    // compat with single-bank dashboards.
    float hist_score;
    float hist_n;
    float hist_score_bank[CHESS_MAX_BANKS];
    float hist_n_bank[CHESS_MAX_BANKS];
    float n;
    // Eval diagnostics (non-selfplay modes). Per-color score/games let Python
    // compute per-color win rate and surface obs-flip / perspective bugs as
    // lopsided splits. maia_failures counts how often maia_get_move returned
    // MOVE_NONE and we fell back to a random legal move — non-zero means part
    // of the eval is degraded.
    float wins_as_white;
    float wins_as_black;
    float games_as_white;
    float games_as_black;
    float maia_failures;
} Log;

typedef struct {
    int cell_size;
    Font piece_font;
    int use_unicode_pieces;
} Client;

typedef struct {
    Piece captured;
    uint8_t castlingRights;
    uint8_t epSquare;
    uint8_t rule50;
    Key key;
    uint8_t pliesFromNull;
} UndoInfo;

typedef struct {
    Log log;
    Client* client;
    uint8_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;   // (97,) — NULL unless MY_ACTION_MASK is defined

    // Per-slot pointers used by the env body. Identity perm = same addresses as
    // base+stride; non-identity perm = where this slot actually lives in vec
    // global buffers. Populated by my_setup_perm.
    uint8_t* obs_ptr[2];
    unsigned char* action_mask_ptr[2];
    float* action_ptr[2];
    float* reward_ptr[2];
    float* terminal_ptr[2];

    unsigned int rng;
    int num_agents;

    Position pos;
    MoveList legal_moves;
    int legal_dirty;
    int game_result;
    int tick;
    int chess_moves;
    int max_moves;
    float reward_draw;
    float episode_reward;
    int render_fps;
    int mode;
    
    char starting_fen[128];
    char** fen_curriculum;
    float fen_curric_pct;
    int num_fens;
    int random_fen;
    
    UndoInfo undo_stack[MAX_GAME_PLIES];
    int undo_stack_ptr;
    uint8_t repetition_matches;
    
    int invalid_actions_this_episode;
    
    int pick_phase[2];
    Square selected_square[2];
    MoveList valid_destinations[2];
    Bitboard valid_from_mask[2];
    Bitboard valid_to_mask[2];
    Bitboard obs_selected_view_mask[2];
    Bitboard obs_valid_from_view_mask[2];
    Bitboard obs_valid_to_view_mask[2];
    uint32_t obs_valid_promo_mask[2];
    float reward_invalid_piece;
    float reward_invalid_move;
    float reward_repetition;
    
    int enable_50_move_rule;
    int enable_threefold_repetition;
    
    int learner_color;
    // Selfplay-pool tagging. tag = 0 selfplay, tag = 1 historical (slot 0 =
    // primary, slot 1 = frozen). boundary_reached set on game-end so Python can
    // detect when historical envs have all completed at least one game since
    // the last swap arm.
    int tag;
    int boundary_reached;
    // Selfplay only: slot_for_color[c] = which slot (0 or 1) plays color c.
    // Default (slot 0 = WHITE, slot 1 = BLACK); randomized per env to remove
    // white-bias when running matched policies in different slots.
    int slot_for_color[2];
    int human_color;
    float white_score;
    float black_score;
    float learner_wins;
    float learner_losses; 
    float learner_draws;
    char last_result[32];
    
    Move pgn_moves[MAX_GAME_PLIES];
    int pgn_move_count;
    int show_game_end_popup;
    
    int log_pgn;
    int log_pgn_choice_made;
    char pgn_filename[128];
    int pgn_game_number;
    
    int white_captured[6];
    int black_captured[6];
    int render_paused;
    int has_last_move_highlight;
    Square last_move_from;
    Square last_move_to;

    // CHESS_MODE_MAIA: per-env lc0 subprocess pipes. Initialized lazily on the
    // first opponent move. -1 / 0 means "not yet spawned".
    int maia_pid;
    int maia_stdin_fd;
    int maia_stdout_fd;
    // Maia commits its move in one UCI round-trip, but selfplay training shows
    // the learner a 2-step opponent wait (pick + place phases). Splitting
    // Maia's move into 2 c_steps (no-op + commit) keeps the learner's LSTM in
    // the training distribution. 0 = next c_step is the no-op phase, 1 = the
    // commit phase.
    int maia_phase;
} Chess;

static inline Bitboard sq_bb(Square s) {
    return SquareBB[s];
}

static inline int popcount(Bitboard b) {
    return __builtin_popcountll(b);
}

static inline Square lsb(Bitboard b) {
    assert(b);
    return __builtin_ctzll(b);
}

static inline Square pop_lsb(Bitboard* b) {
    Square s = lsb(*b);
    *b &= *b - 1;
    return s;
}

static inline Bitboard shift_bb(int Direction, Bitboard b) {
    return Direction == NORTH ? b << 8
         : Direction == SOUTH ? b >> 8
         : Direction == EAST ? (b & ~FileHBB) << 1
         : Direction == WEST ? (b & ~FileABB) >> 1
         : Direction == NORTH_EAST ? (b & ~FileHBB) << 9
         : Direction == SOUTH_EAST ? (b & ~FileHBB) >> 7
         : Direction == NORTH_WEST ? (b & ~FileABB) << 7
         : Direction == SOUTH_WEST ? (b & ~FileABB) >> 9
         : 0;
}

static inline Bitboard pawn_attacks_bb(ChessColor c, Square s) {
    return PawnAttacks[c][s];
}

static inline Bitboard knight_attacks_bb(Square s) {
    return KnightAttacks[s];
}

static inline Bitboard king_attacks_bb(Square s) {
    return KingAttacks[s];
}


static inline Bitboard rook_attacks_bb(Square s, Bitboard occupied) {
    occupied &= RookMasks[s];
    return RookAttacks[s][(occupied * RookMagics[s]) >> RookShifts[s]];
}

static inline Bitboard bishop_attacks_bb(Square s, Bitboard occupied) {
    occupied &= BishopMasks[s];
    return BishopAttacks[s][(occupied * BishopMagics[s]) >> BishopShifts[s]];
}

static inline Bitboard queen_attacks_bb(Square s, Bitboard occupied) {
    return rook_attacks_bb(s, occupied) | bishop_attacks_bb(s, occupied);
}


Bitboard SquareBB[65];
Bitboard PawnAttacks[2][64];
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];
Zobrist zob;

static bool bitboards_initialized = false;

static Bitboard index_to_occupancy(int index, Bitboard mask) {
    Bitboard occ = 0;
    int bits = popcount(mask);
    for (int i = 0; i < bits; i++) {
        Square sq = lsb(mask);
        mask &= mask - 1;
        if (index & (1 << i)) occ |= sq_bb(sq);
    }
    return occ;
}
static Bitboard compute_bishop_mask(Square s) {
    Bitboard mask = 0;
    int r = rank_of(s), f = file_of(s);
    for (int rr = r + 1, ff = f + 1; rr < 7 && ff < 7; rr++, ff++) mask |= sq_bb(make_square(ff, rr));
    for (int rr = r - 1, ff = f + 1; rr > 0 && ff < 7; rr--, ff++) mask |= sq_bb(make_square(ff, rr));
    for (int rr = r - 1, ff = f - 1; rr > 0 && ff > 0; rr--, ff--) mask |= sq_bb(make_square(ff, rr));
    for (int rr = r + 1, ff = f - 1; rr < 7 && ff > 0; rr++, ff--) mask |= sq_bb(make_square(ff, rr));
    return mask;
}

static void init_bishop_magics(void) {
    Bitboard* table_ptr = BishopTable;
    
    for (Square sq = 0; sq < 64; sq++) {
        BishopMasks[sq] = compute_bishop_mask(sq);
        BishopMagics[sq] = BISHOP_MAGICS[sq];
        
        int bits = popcount(BishopMasks[sq]);
        BishopShifts[sq] = 64 - bits;
        BishopAttacks[sq] = table_ptr;
        
        int num_entries = 1 << bits;
        memset(table_ptr, 0, num_entries * sizeof(Bitboard));
        
        for (int i = 0; i < num_entries; i++) {
            Bitboard occ = index_to_occupancy(i, BishopMasks[sq]);
            
            Bitboard attacks = 0;
            int r = rank_of(sq), f = file_of(sq);
            for (int rr = r + 1, ff = f + 1; rr < 8 && ff < 8; rr++, ff++) {
                Square tsq = make_square(ff, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int rr = r - 1, ff = f + 1; rr >= 0 && ff < 8; rr--, ff++) {
                Square tsq = make_square(ff, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int rr = r - 1, ff = f - 1; rr >= 0 && ff >= 0; rr--, ff--) {
                Square tsq = make_square(ff, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int rr = r + 1, ff = f - 1; rr < 8 && ff >= 0; rr++, ff--) {
                Square tsq = make_square(ff, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            
            uint64_t idx = (occ * BishopMagics[sq]) >> BishopShifts[sq];
            table_ptr[idx] = attacks;
        }
        table_ptr += num_entries;
    }
}

static Bitboard compute_rook_mask(Square s) {
    Bitboard mask = 0;
    int r = rank_of(s), f = file_of(s);
    for (int rr = r + 1; rr < 7; rr++) mask |= sq_bb(make_square(f, rr));
    for (int rr = r - 1; rr > 0; rr--) mask |= sq_bb(make_square(f, rr));
    for (int ff = f + 1; ff < 7; ff++) mask |= sq_bb(make_square(ff, r));
    for (int ff = f - 1; ff > 0; ff--) mask |= sq_bb(make_square(ff, r));
    return mask;
}

static void init_rook_magics(void) {
    Bitboard* table_ptr = RookTable;
    
    for (Square sq = 0; sq < 64; sq++) {
        RookMasks[sq] = compute_rook_mask(sq);
        RookMagics[sq] = ROOK_MAGICS[sq];
        
        int bits = popcount(RookMasks[sq]);
        RookShifts[sq] = 64 - bits;
        RookAttacks[sq] = table_ptr;
        
        int num_entries = 1 << bits;
        memset(table_ptr, 0, num_entries * sizeof(Bitboard));
        
        for (int i = 0; i < num_entries; i++) {
            Bitboard occ = index_to_occupancy(i, RookMasks[sq]);
            
            Bitboard attacks = 0;
            int r = rank_of(sq), f = file_of(sq);
            for (int rr = r + 1; rr < 8; rr++) {
                Square tsq = make_square(f, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int rr = r - 1; rr >= 0; rr--) {
                Square tsq = make_square(f, rr);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int ff = f + 1; ff < 8; ff++) {
                Square tsq = make_square(ff, r);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            for (int ff = f - 1; ff >= 0; ff--) {
                Square tsq = make_square(ff, r);
                attacks |= sq_bb(tsq);
                if (occ & sq_bb(tsq)) break;
            }
            
            uint64_t idx = (occ * RookMagics[sq]) >> RookShifts[sq];
            table_ptr[idx] = attacks;
        }
        table_ptr += num_entries;
    }
}

void init_bitboards(void) {
    if (bitboards_initialized) return;
    
    for (int c = 0; c < 2; c++) {
        for (int pt = PAWN; pt <= KING; pt++) {
            for (int s = 0; s < 64; s++) {
                zob.psq[make_piece(c, pt)][s] = prng_rand();
            }
        }
    }
    for (int f = 0; f < 8; f++) {
        zob.enpassant[f] = prng_rand();
    }
    for (int cr = 0; cr < 16; cr++) {
        zob.castling[cr] = prng_rand();
    }
    zob.side = prng_rand();
    
    for (int i = 0; i < 64; i++) {
        SquareBB[i] = 1ULL << i;
    }
    SquareBB[64] = 0;
    
    for (int s = 0; s < 64; s++) {
        Bitboard bb = sq_bb(s);
        PawnAttacks[CHESS_WHITE][s] = shift_bb(NORTH_WEST, bb) | shift_bb(NORTH_EAST, bb);
        PawnAttacks[CHESS_BLACK][s] = shift_bb(SOUTH_WEST, bb) | shift_bb(SOUTH_EAST, bb);
    }
    
    int knight_dirs[] = {-17, -15, -10, -6, 6, 10, 15, 17};
    for (int s = 0; s < 64; s++) {
        Bitboard attack = 0;
        int file = file_of(s);
        int rank = rank_of(s);
        
        for (int i = 0; i < 8; i++) {
            int to = s + knight_dirs[i];
            if (to >= 0 && to < 64) {
                int to_file = file_of(to);
                int to_rank = rank_of(to);
                if (abs(to_file - file) <= 2 && abs(to_rank - rank) <= 2) {
                    attack |= sq_bb(to);
                }
            }
        }
        KnightAttacks[s] = attack;
    }
    
    int king_dirs[] = {-9, -8, -7, -1, 1, 7, 8, 9};
    for (int s = 0; s < 64; s++) {
        Bitboard attack = 0;
        int file = file_of(s);
        
        for (int i = 0; i < 8; i++) {
            int to = s + king_dirs[i];
            if (to >= 0 && to < 64) {
                int to_file = file_of(to);
                if (abs(to_file - file) <= 1) {
                    attack |= sq_bb(to);
                }
            }
        }
        KingAttacks[s] = attack;
    }
    
    for (int s1 = 0; s1 < 64; s1++) {
        for (int s2 = 0; s2 < 64; s2++) {
            BetweenBB[s1][s2] = 0;
            LineBB[s1][s2] = 0;
            
            if (s1 == s2) continue;
            
            int f1 = file_of(s1), r1 = rank_of(s1);
            int f2 = file_of(s2), r2 = rank_of(s2);
            int df = f2 - f1, dr = r2 - r1;
            
            if (df == 0 || dr == 0 || abs(df) == abs(dr)) {
                int step_f = df == 0 ? 0 : (df > 0 ? 1 : -1);
                int step_r = dr == 0 ? 0 : (dr > 0 ? 1 : -1);
                
                // BetweenBB: squares strictly between s1 and s2
                int f = f1 + step_f;
                int r = r1 + step_r;
                while (f != f2 || r != r2) {
                    Square sq = make_square(f, r);
                    BetweenBB[s1][s2] |= sq_bb(sq);
                    f += step_f;
                    r += step_r;
                }

                f = f1;
                r = r1;
                while (f - step_f >= 0 && f - step_f < 8 && r - step_r >= 0 && r - step_r < 8) {
                    f -= step_f;
                    r -= step_r;
                }
                while (f >= 0 && f < 8 && r >= 0 && r < 8) {
                    LineBB[s1][s2] |= sq_bb(make_square(f, r));
                    f += step_f;
                    r += step_r;
                }
            }
        }
    }
    init_bishop_magics(); 
    init_rook_magics();
    bitboards_initialized = true;
}

static void pos_set(Position* pos, const char* fen) {
    memset(pos, 0, sizeof(Position));
    
    int rank = 7, file = 0;
    const char* ptr = fen;
    
    while (*ptr && *ptr != ' ') {
        char c = *ptr++;
        
        if (c == '/') {
            rank--;
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += c - '0';
        } else {
            Square sq = make_square(file, rank);
            Piece pc = NO_PIECE;
            int pt = 0, color = 0;
            
            switch (c) {
                case 'P': pc = W_PAWN; pt = PAWN; color = CHESS_WHITE; break;
                case 'N': pc = W_KNIGHT; pt = KNIGHT; color = CHESS_WHITE; break;
                case 'B': pc = W_BISHOP; pt = BISHOP; color = CHESS_WHITE; break;
                case 'R': pc = W_ROOK; pt = ROOK; color = CHESS_WHITE; break;
                case 'Q': pc = W_QUEEN; pt = QUEEN; color = CHESS_WHITE; break;
                case 'K': pc = W_KING; pt = KING; color = CHESS_WHITE; break;
                case 'p': pc = B_PAWN; pt = PAWN; color = CHESS_BLACK; break;
                case 'n': pc = B_KNIGHT; pt = KNIGHT; color = CHESS_BLACK; break;
                case 'b': pc = B_BISHOP; pt = BISHOP; color = CHESS_BLACK; break;
                case 'r': pc = B_ROOK; pt = ROOK; color = CHESS_BLACK; break;
                case 'q': pc = B_QUEEN; pt = QUEEN; color = CHESS_BLACK; break;
                case 'k': pc = B_KING; pt = KING; color = CHESS_BLACK; break;
            }
            
            if (pc != NO_PIECE) {
                pos->board[sq] = pc;
                pos->byTypeBB[pt] |= sq_bb(sq);
                pos->byColorBB[color] |= sq_bb(sq);
                pos->byTypeBB[0] |= sq_bb(sq);
                pos->pieceCount[pc]++;
            }
            file++;
        }
    }
    
    if (*ptr == ' ') ptr++;
    
    pos->sideToMove = (*ptr == 'w') ? CHESS_WHITE : CHESS_BLACK;
    ptr += 2;
    
    pos->castlingRights = NO_CASTLING;
    while (*ptr && *ptr != ' ') {
        if (*ptr == 'K') pos->castlingRights |= WHITE_OO;
        else if (*ptr == 'Q') pos->castlingRights |= WHITE_OOO;
        else if (*ptr == 'k') pos->castlingRights |= BLACK_OO;
        else if (*ptr == 'q') pos->castlingRights |= BLACK_OOO;
        ptr++;
    }
    

    if (*ptr == ' ') ptr++;
    
    pos->epSquare = SQ_NONE;
    if (*ptr != '-') {
        int ep_file = ptr[0] - 'a';
        int ep_rank = ptr[1] - '1';
        pos->epSquare = make_square(ep_file, ep_rank);
    }
    
    pos->key = 0;
    for (Square sq = SQ_A1; sq <= SQ_H8; sq++) {
        Piece pc = pos->board[sq];
        if (pc != NO_PIECE) {
            pos->key ^= zob.psq[pc][sq];
        }
    }
    if (pos->sideToMove == CHESS_BLACK) {
        pos->key ^= zob.side;
    }
    if (pos->castlingRights) {
        pos->key ^= zob.castling[pos->castlingRights];
    }
    if (pos->epSquare != SQ_NONE) {
        pos->key ^= zob.enpassant[file_of(pos->epSquare)];
    }
}

static void do_move(Position* pos, Move m, UndoInfo* undo_stack, int* undo_stack_ptr) {
    if (m == MOVE_NULL) {
        undo_stack[*undo_stack_ptr].captured = NO_PIECE;
        undo_stack[*undo_stack_ptr].castlingRights = pos->castlingRights;
        undo_stack[*undo_stack_ptr].epSquare = pos->epSquare;
        undo_stack[*undo_stack_ptr].rule50 = pos->rule50;
        undo_stack[*undo_stack_ptr].key = pos->key;
        undo_stack[*undo_stack_ptr].pliesFromNull = 0;
        (*undo_stack_ptr)++;
        
        if (pos->epSquare != SQ_NONE) {
            pos->key ^= zob.enpassant[file_of(pos->epSquare)];
            pos->epSquare = SQ_NONE;
        }
        pos->sideToMove = !pos->sideToMove;
        pos->key ^= zob.side;
        return;
    }
    
    Square from = from_sq(m);
    Square to = to_sq(m);
    int move_type = type_of_m(m);
    Piece pc = piece_on(pos, from);
    Piece captured = piece_on(pos, to);
    int pt = type_of_p(pc);
    ChessColor us = pos->sideToMove;
    ChessColor them = !us;
    
    undo_stack[*undo_stack_ptr].captured = captured;
    undo_stack[*undo_stack_ptr].castlingRights = pos->castlingRights;
    undo_stack[*undo_stack_ptr].epSquare = pos->epSquare;
    undo_stack[*undo_stack_ptr].rule50 = pos->rule50;
    undo_stack[*undo_stack_ptr].key = pos->key;
    undo_stack[*undo_stack_ptr].pliesFromNull = (*undo_stack_ptr > 0) ? undo_stack[*undo_stack_ptr - 1].pliesFromNull + 1 : 0;
    (*undo_stack_ptr)++;
    
    if (pt == PAWN || captured != NO_PIECE) {
        pos->rule50 = 0;
        undo_stack[*undo_stack_ptr - 1].pliesFromNull = 0;
    }
    else {
        pos->rule50++;
    }
    
    if (pos->epSquare != SQ_NONE) {
        pos->key ^= zob.enpassant[file_of(pos->epSquare)];
    }
    pos->epSquare = SQ_NONE;
    
    switch (move_type) {
        case CASTLING: {
            pos->key ^= zob.psq[pc][from];
            
            pos->board[from] = NO_PIECE;
            pos->board[to] = pc;
            pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
            pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
            pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
            pos->key ^= zob.psq[pc][to];
            
            Square rook_from, rook_to;
            if (to > from) { 
                rook_from = from + 3;
                rook_to = from + 1;
            } else {
                rook_from = from - 4;
                rook_to = from - 1;
            }
            
            Piece rook = piece_on(pos, rook_from);
            pos->key ^= zob.psq[rook][rook_from];
            pos->board[rook_from] = NO_PIECE;
            pos->board[rook_to] = rook;
            pos->byTypeBB[ROOK] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
            pos->byColorBB[us] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
            pos->byTypeBB[0] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
            pos->key ^= zob.psq[rook][rook_to];
            break;
        }
        case ENPASSANT: {
            pos->key ^= zob.psq[pc][from];
            
            pos->board[from] = NO_PIECE;
            pos->board[to] = pc;
            pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
            pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
            pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
            pos->key ^= zob.psq[pc][to];
            
            Square cap_sq = to + (us == CHESS_WHITE ? SOUTH : NORTH);
            Piece cap_pawn = piece_on(pos, cap_sq);
            pos->key ^= zob.psq[cap_pawn][cap_sq];
            pos->board[cap_sq] = NO_PIECE;
            pos->byTypeBB[PAWN] ^= sq_bb(cap_sq);
            pos->byColorBB[them] ^= sq_bb(cap_sq);
            pos->byTypeBB[0] ^= sq_bb(cap_sq);
            pos->pieceCount[cap_pawn]--;
            break;
        }
        case NORMAL:
        case PROMOTION: {
            pos->key ^= zob.psq[pc][from];
            
            if (captured != NO_PIECE) {
                pos->key ^= zob.psq[captured][to];
                int cap_pt = type_of_p(captured);
                pos->byTypeBB[cap_pt] ^= sq_bb(to);
                pos->byColorBB[them] ^= sq_bb(to);
                pos->byTypeBB[0] ^= sq_bb(to);
                pos->pieceCount[captured]--;
            }
            
            pos->board[from] = NO_PIECE;
            pos->board[to] = pc;
            pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
            pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
            pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
            pos->key ^= zob.psq[pc][to];
            
            if (move_type == PROMOTION) {
                int promo_pt = promotion_type(m);
                Piece promo_pc = make_piece(us, promo_pt);
                pos->key ^= zob.psq[pc][to];
                pos->board[to] = promo_pc;
                pos->byTypeBB[pt] ^= sq_bb(to);
                pos->byTypeBB[promo_pt] ^= sq_bb(to);
                pos->pieceCount[pc]--;
                pos->pieceCount[promo_pc]++;
                pos->key ^= zob.psq[promo_pc][to];
            }
            
            if (pt == PAWN) {
                int diff = to - from;
                if (diff == 16 || diff == -16) {
                    Square ep_sq = (from + to) / 2;
                    if (pawn_attacks_bb(us, ep_sq) & pieces_cp(pos, them, PAWN)) {
                        pos->epSquare = ep_sq;
                        pos->key ^= zob.enpassant[file_of(ep_sq)];
                    }
                }
            }
            break;
        }
        default:
            break;
    }
    
    uint8_t old_castling = pos->castlingRights;
    if (pt == KING) {
        pos->castlingRights &= us == CHESS_WHITE ? ~WHITE_CASTLING : ~BLACK_CASTLING;
    }
    if (from == SQ_A1 || to == SQ_A1) pos->castlingRights &= ~WHITE_OOO;
    if (from == SQ_H1 || to == SQ_H1) pos->castlingRights &= ~WHITE_OO;
    if (from == SQ_A8 || to == SQ_A8) pos->castlingRights &= ~BLACK_OOO;
    if (from == SQ_H8 || to == SQ_H8) pos->castlingRights &= ~BLACK_OO;
    
    if (old_castling != pos->castlingRights) {
        pos->key ^= zob.castling[old_castling];
        pos->key ^= zob.castling[pos->castlingRights];
    }
    
    pos->sideToMove = them;
    pos->key ^= zob.side;
}

static void undo_move(Position* pos, Move m, UndoInfo* undo_stack, int* undo_stack_ptr) {
    (*undo_stack_ptr)--;
    UndoInfo* undo = &undo_stack[*undo_stack_ptr];
    
    if (m == MOVE_NULL) {
        pos->castlingRights = undo->castlingRights;
        pos->epSquare = undo->epSquare;
        pos->rule50 = undo->rule50;
        pos->key = undo->key;
        pos->sideToMove = !pos->sideToMove;
        return;
    }
    
    Square from = from_sq(m);
    Square to = to_sq(m);
    int move_type = type_of_m(m);
    ChessColor us = !pos->sideToMove;
    ChessColor them = pos->sideToMove;
    
    Piece pc = piece_on(pos, to);
    int pt = type_of_p(pc);
    
    pos->castlingRights = undo->castlingRights;
    pos->epSquare = undo->epSquare;
    pos->rule50 = undo->rule50;
    pos->key = undo->key;
    pos->sideToMove = us;
    
    switch (move_type) {
        case CASTLING: {
            pos->board[to] = NO_PIECE;
            pos->board[from] = pc;
            pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
            pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
            pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
            
            Square rook_from, rook_to;
            if (to > from) {
                rook_from = from + 3;
                rook_to = from + 1;
            } else {
                rook_from = from - 4;
                rook_to = from - 1;
            }
            
            Piece rook = piece_on(pos, rook_to);
            pos->board[rook_to] = NO_PIECE;
            pos->board[rook_from] = rook;
            pos->byTypeBB[ROOK] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
            pos->byColorBB[us] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
            pos->byTypeBB[0] ^= sq_bb(rook_from) ^ sq_bb(rook_to);
            break;
        }
        case ENPASSANT: {
            pos->board[to] = NO_PIECE;
            pos->board[from] = pc;
            pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
            pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
            pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);

            Square cap_sq = to + (us == CHESS_WHITE ? SOUTH : NORTH);
            Piece cap_pawn = make_piece(them, PAWN);
            pos->board[cap_sq] = cap_pawn;
            pos->byTypeBB[PAWN] ^= sq_bb(cap_sq);
            pos->byColorBB[them] ^= sq_bb(cap_sq);
            pos->byTypeBB[0] ^= sq_bb(cap_sq);
            pos->pieceCount[cap_pawn]++;
            break;
        }
        case NORMAL:
        case PROMOTION: {
            if (move_type == PROMOTION) {
                int promo_pt = promotion_type(m);
                Piece promo_pc = make_piece(us, promo_pt);
                pc = make_piece(us, PAWN);
                pt = PAWN;
                pos->board[to] = NO_PIECE;
                pos->byTypeBB[promo_pt] ^= sq_bb(to);
                pos->byTypeBB[pt] ^= sq_bb(to);
                pos->pieceCount[promo_pc]--;
                pos->pieceCount[pc]++;
            }
            
            pos->board[to] = undo->captured;
            pos->board[from] = pc;
            pos->byTypeBB[pt] ^= sq_bb(from) ^ sq_bb(to);
            pos->byColorBB[us] ^= sq_bb(from) ^ sq_bb(to);
            
            if (undo->captured != NO_PIECE) {
                int cap_pt = type_of_p(undo->captured);
                pos->byTypeBB[cap_pt] ^= sq_bb(to);
                pos->byColorBB[them] ^= sq_bb(to);
                pos->byTypeBB[0] ^= sq_bb(from);
                pos->pieceCount[undo->captured]++;
            } else {
                pos->byTypeBB[0] ^= sq_bb(from) ^ sq_bb(to);
            }
            break;
        }
        default:
            break;
    }
}

static inline void add_move(MoveList* ml, Move m) {
    ml->moves[ml->count].move = m;
    ml->count++;
}

static void generate_pawn_moves(Position* pos, MoveList* ml, ChessColor us) {
    ChessColor them = !us;
    int up = (us == CHESS_WHITE) ? NORTH : SOUTH;
    Bitboard rank7 = (us == CHESS_WHITE) ? Rank7BB : Rank2BB;
    Bitboard rank3 = (us == CHESS_WHITE) ? Rank3BB : Rank6BB;
    
    Bitboard pawns = pieces_cp(pos, us, PAWN);
    Bitboard pawnsOn7 = pawns & rank7;
    Bitboard pawnsNotOn7 = pawns & ~rank7;
    
    Bitboard enemies = pieces_c(pos, them);
    Bitboard empty = ~pieces(pos);
    
    Bitboard b1 = shift_bb(up, pawnsNotOn7) & empty;
    Bitboard b2 = shift_bb(up, b1 & rank3) & empty;
    
    while (b1) {
        Square to = pop_lsb(&b1);
        add_move(ml, make_move(to - up, to));
    }
    
    while (b2) {
        Square to = pop_lsb(&b2);
        add_move(ml, make_move(to - up - up, to));
    }
    
    if (pawnsOn7) {
        Bitboard b3 = shift_bb(up, pawnsOn7) & empty;
        while (b3) {
            Square to = pop_lsb(&b3);
            Square from = to - up;
            add_move(ml, make_promotion(from, to, QUEEN));
            add_move(ml, make_promotion(from, to, ROOK));
            add_move(ml, make_promotion(from, to, BISHOP));
            add_move(ml, make_promotion(from, to, KNIGHT));
        }
    }
    
    Bitboard b4 = shift_bb(up + WEST, pawnsNotOn7) & enemies;
    Bitboard b5 = shift_bb(up + EAST, pawnsNotOn7) & enemies;
    
    while (b4) {
        Square to = pop_lsb(&b4);
        add_move(ml, make_move(to - up - WEST, to));
    }
    
    while (b5) {
        Square to = pop_lsb(&b5);
        add_move(ml, make_move(to - up - EAST, to));
    }
    
    if (pawnsOn7) {
        Bitboard b6 = shift_bb(up + WEST, pawnsOn7) & enemies;
        Bitboard b7 = shift_bb(up + EAST, pawnsOn7) & enemies;
        
        while (b6) {
            Square to = pop_lsb(&b6);
            Square from = to - up - WEST;
            add_move(ml, make_promotion(from, to, QUEEN));
            add_move(ml, make_promotion(from, to, ROOK));
            add_move(ml, make_promotion(from, to, BISHOP));
            add_move(ml, make_promotion(from, to, KNIGHT));
        }
        
        while (b7) {
            Square to = pop_lsb(&b7);
            Square from = to - up - EAST;
            add_move(ml, make_promotion(from, to, QUEEN));
            add_move(ml, make_promotion(from, to, ROOK));
            add_move(ml, make_promotion(from, to, BISHOP));
            add_move(ml, make_promotion(from, to, KNIGHT));
        }
    }
    
    if (pos->epSquare != SQ_NONE) {
        Bitboard ep_pawns = pawnsNotOn7 & pawn_attacks_bb(them, pos->epSquare);
        while (ep_pawns) {
            Square from = pop_lsb(&ep_pawns);
            add_move(ml, make_enpassant(from, pos->epSquare));
        }
    }
}

static void generate_castling(Position* pos, MoveList* ml, ChessColor us) {
    Bitboard occupied = pieces(pos);
    
    if (us == CHESS_WHITE) {
        if (pos->castlingRights & WHITE_OO) {
            if (!(occupied & (sq_bb(SQ_F1) | sq_bb(SQ_G1)))) {
                add_move(ml, make_castling(SQ_E1, SQ_G1));
            }
        }
        if (pos->castlingRights & WHITE_OOO) {
            if (!(occupied & (sq_bb(SQ_D1) | sq_bb(SQ_C1) | sq_bb(SQ_B1)))) {
                add_move(ml, make_castling(SQ_E1, SQ_C1));
            }
        }
    } else {
        if (pos->castlingRights & BLACK_OO) {
            if (!(occupied & (sq_bb(SQ_F8) | sq_bb(SQ_G8)))) {
                add_move(ml, make_castling(SQ_E8, SQ_G8));
            }
        }
        if (pos->castlingRights & BLACK_OOO) {
            if (!(occupied & (sq_bb(SQ_D8) | sq_bb(SQ_C8) | sq_bb(SQ_B8)))) {
                add_move(ml, make_castling(SQ_E8, SQ_C8));
            }
        }
    }
}

static Bitboard attackers_to_sq(Position* pos, Square sq, Bitboard occupied) {
    return (pawn_attacks_bb(CHESS_WHITE, sq) & pieces_cp(pos, CHESS_BLACK, PAWN) & occupied)
         | (pawn_attacks_bb(CHESS_BLACK, sq) & pieces_cp(pos, CHESS_WHITE, PAWN) & occupied)
         | (knight_attacks_bb(sq) & pieces_p(pos, KNIGHT) & occupied)
         | (king_attacks_bb(sq) & pieces_p(pos, KING) & occupied)
         | (bishop_attacks_bb(sq, occupied) & (pieces_p(pos, BISHOP) | pieces_p(pos, QUEEN)))
         | (rook_attacks_bb(sq, occupied) & (pieces_p(pos, ROOK) | pieces_p(pos, QUEEN)));
}

static bool is_check(Position* pos, ChessColor c) {
    Bitboard king_bb = pieces_cp(pos, c, KING);
    if (!king_bb) return false;
    Square king_sq = lsb(king_bb);
    return (attackers_to_sq(pos, king_sq, pieces(pos)) & pieces_c(pos, !c)) != 0;
}

static Bitboard compute_pinned(Position* pos, ChessColor c) {
    Bitboard pinned = 0;
    Bitboard our_pieces = pieces_c(pos, c);
    Bitboard king_bb = pieces_cp(pos, c, KING);
    if (!king_bb) return 0;
    
    Square ksq = lsb(king_bb);
    ChessColor them = !c;
    Bitboard occupied = pieces(pos);
    
    Bitboard diag_pinners = (pieces_cp(pos, them, BISHOP) | pieces_cp(pos, them, QUEEN)) 
                          & bishop_attacks_bb(ksq, 0);
    
    while (diag_pinners) {
        Square pinner_sq = pop_lsb(&diag_pinners);
        Bitboard between = BetweenBB[ksq][pinner_sq] & occupied;
        if (popcount(between) == 1) {
            pinned |= between & our_pieces;
        }
    }
    
    Bitboard rook_pinners = (pieces_cp(pos, them, ROOK) | pieces_cp(pos, them, QUEEN)) 
                          & rook_attacks_bb(ksq, 0);
    
    while (rook_pinners) {
        Square pinner_sq = pop_lsb(&rook_pinners);
        Bitboard between = BetweenBB[ksq][pinner_sq] & occupied;
        if (popcount(between) == 1) {
            pinned |= between & our_pieces;
        }
    }
    
    return pinned;
}

static inline bool is_legal_move_fast(Position* pos, Move m, Bitboard pinned, Square ksq, ChessColor us) {
    Square from = from_sq(m);
    Square to = to_sq(m);
    int mt = type_of_m(m);
    
    if (from == ksq) {
        if (mt == CASTLING) {
            ChessColor them = !us;
            if (is_check(pos, us)) return false;
            Square mid = (from + to) / 2;
            Bitboard occ = pieces(pos) ^ sq_bb(from);
            if (attackers_to_sq(pos, mid, occ) & pieces_c(pos, them)) return false;
            if (attackers_to_sq(pos, to, occ) & pieces_c(pos, them)) return false;
            return true;
        }
        Bitboard occ = pieces(pos) ^ sq_bb(from);
        return !(attackers_to_sq(pos, to, occ) & pieces_c(pos, !us));
    }
    
    if (mt == ENPASSANT) {
        Bitboard occ = pieces(pos) ^ sq_bb(from) ^ sq_bb(to);
        Square capsq = to + (us == CHESS_WHITE ? -8 : 8);
        occ ^= sq_bb(capsq);
        return !(attackers_to_sq(pos, ksq, occ) & pieces_c(pos, !us));
    }
    
    if (!(pinned & sq_bb(from))) {
        return true;
    }
    
    return LineBB[ksq][from] & sq_bb(to);
}

static inline bool is_legal_move(Position* pos, Move m) {
    ChessColor us = pos->sideToMove;
    ChessColor them = (ChessColor)!us;
    int mt = type_of_m(m);
    if (mt == CASTLING) {
        if (is_check(pos, us)) return false;
        Square from = from_sq(m), to = to_sq(m);
        Square mid = (from + to) / 2;
        Bitboard occ = pieces(pos);
        if ((attackers_to_sq(pos, mid, occ) & pieces_c(pos, them))
         || (attackers_to_sq(pos, to, occ) & pieces_c(pos, them))) return false;
        return true;
    }
    if (mt == ENPASSANT) {
        Bitboard king_bb = pieces_cp(pos, us, KING);
        if (!king_bb) return false;
        Square ksq = lsb(king_bb);
        Square from = from_sq(m), to = to_sq(m);
        Square capsq = (us == CHESS_WHITE) ? (to - 8) : (to + 8);
        Bitboard occ = pieces(pos) ^ sq_bb(from) ^ sq_bb(capsq) ^ sq_bb(to);
        return (attackers_to_sq(pos, ksq, occ) & pieces_c(pos, them)) == 0;
    }
    UndoInfo u[1]; int p = 0;
    do_move(pos, m, u, &p);
    bool ok = !is_check(pos, us);
    undo_move(pos, m, u, &p);
    return ok;
}

static inline void generate_pseudo_legal(Position* pos, MoveList* ml, ChessColor us) {
    ml->count = 0;
    generate_pawn_moves(pos, ml, us);

    Bitboard occupied = pieces(pos);
    Bitboard target = ~pieces_c(pos, us);
    Bitboard bb = pieces_cp(pos, us, KNIGHT);
    while (bb) {
        Square from = pop_lsb(&bb);
        Bitboard attacks = knight_attacks_bb(from) & target;
        while (attacks) {
            Square to = pop_lsb(&attacks);
            add_move(ml, make_move(from, to));
        }
    }

    bb = pieces_cp(pos, us, BISHOP);
    while (bb) {
        Square from = pop_lsb(&bb);
        Bitboard attacks = bishop_attacks_bb(from, occupied) & target;
        while (attacks) {
            Square to = pop_lsb(&attacks);
            add_move(ml, make_move(from, to));
        }
    }

    bb = pieces_cp(pos, us, ROOK);
    while (bb) {
        Square from = pop_lsb(&bb);
        Bitboard attacks = rook_attacks_bb(from, occupied) & target;
        while (attacks) {
            Square to = pop_lsb(&attacks);
            add_move(ml, make_move(from, to));
        }
    }

    bb = pieces_cp(pos, us, QUEEN);
    while (bb) {
        Square from = pop_lsb(&bb);
        Bitboard attacks = queen_attacks_bb(from, occupied) & target;
        while (attacks) {
            Square to = pop_lsb(&attacks);
            add_move(ml, make_move(from, to));
        }
    }

    bb = pieces_cp(pos, us, KING);
    while (bb) {
        Square from = pop_lsb(&bb);
        Bitboard attacks = king_attacks_bb(from) & target;
        while (attacks) {
            Square to = pop_lsb(&attacks);
            add_move(ml, make_move(from, to));
        }
    }

    generate_castling(pos, ml, us);
}

static void generate_legal(Position* pos, MoveList* ml, UndoInfo* undo_stack, int* undo_stack_ptr) {
    generate_pseudo_legal(pos, ml, pos->sideToMove);
    ChessColor us = pos->sideToMove;
    ChessColor them = (ChessColor)!us;
    Bitboard king_bb = pieces_cp(pos, us, KING);
    Square ksq = king_bb ? lsb(king_bb) : SQ_NONE;
    Bitboard pinned = compute_pinned(pos, us);
    bool in_check = is_check(pos, us);
    int check_count = 0;
    Bitboard evasion_mask = 0;
    if (in_check) {
        Bitboard checkers = attackers_to_sq(pos, ksq, pieces(pos)) & pieces_c(pos, them);
        check_count = popcount(checkers);
        if (check_count == 1) {
            Square checker_sq = lsb(checkers);
            evasion_mask = sq_bb(checker_sq);
            Piece checker = piece_on(pos, checker_sq);
            if (checker != NO_PIECE) {
                int checker_type = type_of_p(checker);
                if (checker_type == BISHOP || checker_type == ROOK || checker_type == QUEEN) {
                    evasion_mask |= BetweenBB[ksq][checker_sq];
                }
            }
        }
    }
    
    int write = 0;
    for (int i = 0; i < ml->count; i++) {
        Move m = ml->moves[i].move;
        Square from = from_sq(m);
        bool legal;
        if (!in_check) {
            legal = is_legal_move_fast(pos, m, pinned, ksq, us);
        } else if (from == ksq) {
            legal = is_legal_move_fast(pos, m, pinned, ksq, us);
        } else if (check_count > 1) {
            legal = false;
        } else if (type_of_m(m) == ENPASSANT) {
            legal = is_legal_move(pos, m);
        } else if ((sq_bb(to_sq(m)) & evasion_mask) == 0) {
            legal = false;
        } else {
            legal = is_legal_move_fast(pos, m, pinned, ksq, us);
        }
        if (legal) {
            ml->moves[write++] = ml->moves[i];
        }
    }
    ml->count = write;
}

static inline bool is_insufficient_material(const Position* pos) {
    if (pieces_p(pos, PAWN) | pieces_p(pos, ROOK) | pieces_p(pos, QUEEN))
        return false;

    int wN = popcount(pieces_cp(pos, CHESS_WHITE, KNIGHT));
    int bN = popcount(pieces_cp(pos, CHESS_BLACK, KNIGHT));
    int wB = popcount(pieces_cp(pos, CHESS_WHITE, BISHOP));
    int bB = popcount(pieces_cp(pos, CHESS_BLACK, BISHOP));
    int totalMinors = wN + bN + wB + bB;

    if (totalMinors == 0)
        return true;

    if (totalMinors == 1)
        return true;

    if (totalMinors == 2) {
        if ((wN == 2 && wB == 0 && bN == 0 && bB == 0) || (bN == 2 && bB == 0 && wN == 0 && wB == 0))
            return true;
        if ((wN + wB) == 1 && (bN + bB) == 1)
            return true;
    }

    return false;
}

static void clear_player_selection(Chess* env, int side) {
    env->pick_phase[side] = 0;
    env->selected_square[side] = SQ_NONE;
    env->valid_destinations[side].count = 0;
    env->valid_to_mask[side] = 0;
}

static void rebuild_legal_state(Chess* env) {
    generate_legal(&env->pos, &env->legal_moves, env->undo_stack, &env->undo_stack_ptr);
    env->legal_dirty = 0;
    env->valid_from_mask[0] = 0;
    env->valid_from_mask[1] = 0;
    env->valid_to_mask[0] = 0;
    env->valid_to_mask[1] = 0;
    int side = (int)env->pos.sideToMove;
    Bitboard from_mask = 0;
    for (int i = 0; i < env->legal_moves.count; i++) {
        from_mask |= sq_bb(from_sq(env->legal_moves.moves[i].move));
    }
    env->valid_from_mask[side] = from_mask;
}
void populate_observations(Chess* env) {
    Position* pos = &env->pos;

    int num_players = env->mode == CHESS_MODE_SELFPLAY ? 2 : 1;
    for (int player_iter = 0; player_iter < num_players; player_iter++) {
        int player = env->mode == CHESS_MODE_SELFPLAY ? player_iter : env->learner_color;
        // Selfplay: slot ↔ color mapping is randomized per env (slot_for_color).
        // Single-agent modes: only the learner has a slot (idx 0).
        int buffer_idx = (env->mode == CHESS_MODE_SELFPLAY) ? env->slot_for_color[player] : 0;
        uint8_t* player_obs = env->obs_ptr[buffer_idx];
        memset(player_obs, 0, OBS_SIZE);
        uint8_t* board_planes = player_obs + O_BOARD;

        // Selfplay: each iteration writes into its own per-slot mask.
        // Single-agent modes: only the learner iter writes into the (single) mask.
        unsigned char* my_mask = NULL;
        bool fill_mask = false;
        if (env->action_mask != NULL) {
            if (env->mode == CHESS_MODE_SELFPLAY) {
                my_mask = env->action_mask_ptr[buffer_idx];
                fill_mask = true;
            } else if (player == env->learner_color) {
                my_mask = env->action_mask_ptr[0];
                fill_mask = true;
            }
        }
        if (fill_mask) {
            memset(my_mask, 0, NUM_ACTIONS * sizeof(unsigned char));
        }

        ChessColor us = (ChessColor)player;  // 0=White, 1=Black
        ChessColor them = (ChessColor)!us;

        int flip = player * 56;


        // Compact ego-centric board: one byte per square. 0 = empty,
        // own P..K = 1..6, enemy P..K = 7..12.
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = pieces_cp(pos, player, pt);
            while (bb) {
                Square sq = pop_lsb(&bb);
                board_planes[sq ^ flip] = (uint8_t)pt;
            }
        }
        
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = pieces_cp(pos, them, pt);
            while (bb) {
                Square sq = pop_lsb(&bb);
                board_planes[sq ^ flip] = (uint8_t)(6 + pt);
            }
        }
               
        ChessColor side_to_move = pos->sideToMove;
        
        player_obs[O_SIDE] = (pos->sideToMove == us) ? 1 : 0;
        
        uint8_t castle_rights = pos->castlingRights;
        if (player == 1) {
            uint8_t flipped = 0;
            if (castle_rights & BLACK_OO) flipped |= WHITE_OO;
            if (castle_rights & BLACK_OOO) flipped |= WHITE_OOO;
            if (castle_rights & WHITE_OO) flipped |= BLACK_OO;
            if (castle_rights & WHITE_OOO) flipped |= BLACK_OOO;
            castle_rights = flipped;
        }
        player_obs[O_CASTLE + 0] = (castle_rights & WHITE_OO) ? 1 : 0;
        player_obs[O_CASTLE + 1] = (castle_rights & WHITE_OOO) ? 1 : 0;
        player_obs[O_CASTLE + 2] = (castle_rights & BLACK_OO) ? 1 : 0;
        player_obs[O_CASTLE + 3] = (castle_rights & BLACK_OOO) ? 1 : 0;

        if (pos->epSquare < 64) {
            int ep_sq = (player == 1) ? (pos->epSquare ^ 56) : pos->epSquare;
            player_obs[O_EP + file_of((Square)ep_sq)] = 1;
        } else {
            player_obs[O_EP + 8] = 1;
        }
    
        uint8_t* valid_from_indices = player_obs + O_VALID_FROM;
        uint8_t* valid_to_indices = player_obs + O_VALID_TO;
        for (int k = 0; k < CHESS_MAX_VALID_FROM; k++) valid_from_indices[k] = CHESS_NULL_SQ;
        for (int k = 0; k < CHESS_MAX_VALID_TO; k++)   valid_to_indices[k]   = CHESS_NULL_SQ;
        int valid_from_count = 0;
        int valid_to_count = 0;

        int player_idx = (int)us;

        if (side_to_move == us) {
            if (env->pick_phase[player_idx] == 0) {
                Bitboard added = 0;
                for (int i = 0; i < env->legal_moves.count; i++) {
                    Square from = from_sq(env->legal_moves.moves[i].move);
                    int view_from = (player == 1) ? (from ^ 56) : from;
                    Bitboard bit = sq_bb((Square)view_from);
                    if (!(added & bit)) {
                        added |= bit;
                        if (valid_from_count < CHESS_MAX_VALID_FROM) {
                            valid_from_indices[valid_from_count++] = (uint8_t)view_from;
                        }
                        if (fill_mask) my_mask[view_from] = 1;
                    }
                }
            } else {
                Bitboard added = 0;
                for (int i = 0; i < env->valid_destinations[player_idx].count; i++) {
                    Square to = to_sq(env->valid_destinations[player_idx].moves[i].move);
                    int view_to = (player == 1) ? (to ^ 56) : to;
                    Bitboard bit = sq_bb((Square)view_to);
                    if (!(added & bit)) {
                        added |= bit;
                        if (valid_to_count < CHESS_MAX_VALID_TO) {
                            valid_to_indices[valid_to_count++] = (uint8_t)view_to;
                        }
                        if (fill_mask) my_mask[view_to] = 1;
                    }
                }
            }
        }
        player_obs[O_VALID_FROM_COUNT] = (uint8_t)valid_from_count;
        player_obs[O_VALID_TO_COUNT]   = (uint8_t)valid_to_count;
        player_obs[O_PASS_VALID] = (side_to_move != us) ? 255 : 0;
        if (fill_mask && side_to_move != us) {
            my_mask[PASS_ACTION] = 1;
        }

        player_obs[O_PICK_PHASE] = env->pick_phase[player_idx] ? 1 : 0;

        uint8_t selected_byte = (uint8_t)CHESS_NULL_SQ;
        if (env->pick_phase[player_idx] == 1 && env->selected_square[player_idx] != SQ_NONE) {
            int view_selected = (player == 1)
                ? (env->selected_square[player_idx] ^ 56)
                : env->selected_square[player_idx];
            selected_byte = (uint8_t)view_selected;
        }
        player_obs[O_SELECTED_PIECE] = selected_byte;
        
        uint8_t* valid_promos = player_obs + O_VALID_PROMOS;
        
        if (env->pick_phase[player_idx] == 1 && env->valid_destinations[player_idx].count > 0) {
            for (int i = 0; i < env->valid_destinations[player_idx].count; i++) {
                Move m = env->valid_destinations[player_idx].moves[i].move;
                if (type_of_m(m) == PROMOTION) {
                    int type_idx = QUEEN - promotion_type(m);
                    int file_idx = file_of(to_sq(m));
                    valid_promos[type_idx * 8 + file_idx] = 1;
                    if (fill_mask) my_mask[64 + type_idx * 8 + file_idx] = 1;
                }
            }
        }
        
        player_obs[O_SELF_CHECK] = is_check(pos, us) ? 255 : 0;
        player_obs[O_OPP_CHECK] = is_check(pos, them) ? 255 : 0;
        
        int rule50 = pos->rule50;
        if (rule50 > 100) rule50 = 100;
        player_obs[O_RULE50] = (uint8_t)((rule50 * 255) / 100);
        
        uint8_t rep_val = 0;
        if (env->undo_stack_ptr >= 4) {
            uint8_t plies = env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull;
            if (plies >= 4) {
                int repetitions = 0;
                for (int i = 4; i <= plies; i += 2) {
                    int idx = env->undo_stack_ptr - i;
                    if (idx >= 0 && env->undo_stack[idx].key == pos->key) {
                        repetitions++;
                    }
                }
                if (repetitions >= 2) {
                    rep_val = 255;
                } else if (repetitions == 1) {
                    rep_val = 128;
                }
            }
        }
        player_obs[O_REPETITION] = rep_val;
    }
}

static int move_to_san(Position* pos, Move m, char* buf, UndoInfo* undo_stack, int* undo_stack_ptr) {
    const char files[] = "abcdefgh";
    const char ranks[] = "12345678";
    const char piece_chars[] = ".PNBRQK";
    char* ptr = buf;
    
    Square from = from_sq(m);
    Square to = to_sq(m);
    int move_type = type_of_m(m);
    Piece pc = piece_on(pos, from);
    int pt = type_of_p(pc);
    ChessColor us = pos->sideToMove;
    
    if (move_type == CASTLING) {
        if (to > from) {
            strcpy(ptr, "O-O");
            ptr += 3;
        } else {
            strcpy(ptr, "O-O-O");
            ptr += 5;
        }
    } else {
        if (pt != PAWN) {
            *ptr++ = piece_chars[pt];
            
            Bitboard same_pieces = pieces_cp(pos, us, pt) & ~sq_bb(from);
            Bitboard attackers = 0;
            
            if (pt == KNIGHT) {
                attackers = knight_attacks_bb(to) & same_pieces;
            } else if (pt == BISHOP) {
                attackers = bishop_attacks_bb(to, pieces(pos)) & same_pieces;
            } else if (pt == ROOK) {
                attackers = rook_attacks_bb(to, pieces(pos)) & same_pieces;
            } else if (pt == QUEEN) {
                attackers = (bishop_attacks_bb(to, pieces(pos)) | rook_attacks_bb(to, pieces(pos))) & same_pieces;
            } else if (pt == KING) {
                attackers = king_attacks_bb(to) & same_pieces;
            }
            
            Bitboard legal_attackers = 0;
            while (attackers) {
                Square attacker_sq = pop_lsb(&attackers);
                Move test_move = make_move(attacker_sq, to);
                if (is_legal_move(pos, test_move)) {
                    legal_attackers |= sq_bb(attacker_sq);
                }
            }
            
            if (legal_attackers) {
                int same_file = 0, same_rank = 0;
                Bitboard temp = legal_attackers;
                while (temp) {
                    Square s = pop_lsb(&temp);
                    if (file_of(s) == file_of(from)) same_file++;
                    if (rank_of(s) == rank_of(from)) same_rank++;
                }
                
                if (same_file == 0) {
                    *ptr++ = files[file_of(from)];
                } else if (same_rank == 0) {
                    *ptr++ = ranks[rank_of(from)];
                } else {
                    *ptr++ = files[file_of(from)];
                    *ptr++ = ranks[rank_of(from)];
                }
            }
        }
        
        Piece captured = piece_on(pos, to);
        bool is_capture = (captured != NO_PIECE) || (move_type == ENPASSANT);
        
        if (is_capture) {
            if (pt == PAWN) {
                *ptr++ = files[file_of(from)];
            }
            *ptr++ = 'x';
        }
        
        *ptr++ = files[file_of(to)];
        *ptr++ = ranks[rank_of(to)];
        
        if (move_type == PROMOTION) {
            *ptr++ = '=';
            const char promo_pieces[] = "..NBRQ";
            *ptr++ = promo_pieces[promotion_type(m)];
        }
    }
    
    do_move(pos, m, undo_stack, undo_stack_ptr);
    
    ChessColor them = pos->sideToMove;
    if (is_check(pos, them)) {
        MoveList ml;
        generate_legal(pos, &ml, undo_stack, undo_stack_ptr);
        if (ml.count == 0) {
            *ptr++ = '#';
        } else {
            *ptr++ = '+';
        }
    }
    
    undo_move(pos, m, undo_stack, undo_stack_ptr);
    
    *ptr = '\0';
    return ptr - buf;
}

static void export_pgn_append(Chess* env, const char* filename, int append) {
    FILE* f = fopen(filename, append ? "a" : "w");
    if (!f) return;
    
    if (env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM) {
        const char* opponent_name = env->mode == CHESS_MODE_HUMAN ? "AI" : "Random";
        const char* event_name = env->mode == CHESS_MODE_HUMAN ? "Human vs AI" : "Human vs Random";
        fprintf(f, "[Event \"%s\"]\n", event_name);
        fprintf(f, "[White \"%s\"]\n", env->human_color == CHESS_WHITE ? "Human" : opponent_name);
        fprintf(f, "[Black \"%s\"]\n", env->human_color == CHESS_BLACK ? "Human" : opponent_name);
    } else {
        fprintf(f, "[Event \"Selfplay Eval Game %d\"]\n", env->pgn_game_number);
        fprintf(f, "[White \"%s\"]\n", env->learner_color == CHESS_BLACK ? "Learner" : "Opponent");
        fprintf(f, "[Black \"%s\"]\n", env->learner_color == CHESS_BLACK ? "Opponent" : "Learner");
    }
    fprintf(f, "[Site \"PufferLib\"]\n");
    fprintf(f, "[Result \"%s\"]\n\n", env->last_result);
    
    Position replay_pos;
    pos_set(&replay_pos, env->starting_fen);
    
    UndoInfo replay_undo[MAX_GAME_PLIES];
    int replay_undo_ptr = 0;
    
    char san_buf[16];
    
    for (int i = 0; i < env->pgn_move_count; i++) {
        if (i % 2 == 0) {
            fprintf(f, "%d. ", i/2 + 1);
        }
        
        Move m = env->pgn_moves[i];
        move_to_san(&replay_pos, m, san_buf, replay_undo, &replay_undo_ptr);
        fprintf(f, "%s ", san_buf);
        
        do_move(&replay_pos, m, replay_undo, &replay_undo_ptr);
        
        if ((i + 1) % 8 == 0) fprintf(f, "\n");
    }
    
    if (strcmp(env->last_result, "White Wins") == 0) {
        fprintf(f, "1-0");
    } else if (strcmp(env->last_result, "Black Wins") == 0) {
        fprintf(f, "0-1");
    } else {
        fprintf(f, "1/2-1/2");
    }
    
    fprintf(f, "\n\n");
    fclose(f);
}

static void generate_random_fen(Chess* env, char* fen_out) {
    char board[64];
    memset(board, '.', 64);

    int wk_sq, bk_sq;
    do {
        wk_sq = rand_r(&env->rng) % 64;
        bk_sq = rand_r(&env->rng) % 64;
        int wk_rank = wk_sq / 8, wk_file = wk_sq % 8;
        int bk_rank = bk_sq / 8, bk_file = bk_sq % 8;
        int rank_diff = abs(wk_rank - bk_rank);
        int file_diff = abs(wk_file - bk_file);
        if (wk_sq != bk_sq && (rank_diff > 1 || file_diff > 1)) break;
    } while (1);

    board[wk_sq] = 'K';
    board[bk_sq] = 'k';

    const char* white_pieces = "QRRNNBBPP";
    const char* black_pieces = "qrrnnbbpp";
    int num_white = rand_r(&env->rng) % 16;
    int num_black = rand_r(&env->rng) % 16;

    for (int i = 0; i < num_white; i++) {
        int sq, rank;
        char piece;
        do {
            sq = rand_r(&env->rng) % 64;
            rank = sq / 8;
            piece = white_pieces[rand_r(&env->rng) % 9];
        } while (board[sq] != '.' || (piece == 'P' && (rank == 0 || rank == 7)));
        board[sq] = piece;
    }

    for (int i = 0; i < num_black; i++) {
        int sq, rank;
        char piece;
        do {
            sq = rand_r(&env->rng) % 64;
            rank = sq / 8;
            piece = black_pieces[rand_r(&env->rng) % 9];
        } while (board[sq] != '.' || (piece == 'p' && (rank == 0 || rank == 7)));
        board[sq] = piece;
    }
    
    char* ptr = fen_out;
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            char piece = board[rank * 8 + file];
            if (piece == '.') {
                empty++;
            } else {
                if (empty > 0) {
                    *ptr++ = '0' + empty;
                    empty = 0;
                }
                *ptr++ = piece;
            }
        }
        if (empty > 0) *ptr++ = '0' + empty;
        if (rank > 0) *ptr++ = '/';
    }
    strcpy(ptr, " w - - 0 1");
}

static inline int apply_move_to_env(Chess* env, Move chosen, int* is_timeout) {
    env->chess_moves++;
    env->last_move_from = from_sq(chosen);
    env->last_move_to = to_sq(chosen);
    env->has_last_move_highlight = 1;

    if ((env->mode == CHESS_MODE_HUMAN
            || env->mode == CHESS_MODE_HUMAN_RANDOM
            || env->log_pgn) && env->pgn_move_count < MAX_GAME_PLIES) {
        env->pgn_moves[env->pgn_move_count++] = chosen;
    }
    
    ChessColor side_before = env->pos.sideToMove;
    do_move(&env->pos, chosen, env->undo_stack, &env->undo_stack_ptr);
    env->legal_dirty = 1;
    clear_player_selection(env, (int)env->pos.sideToMove);
    
    if (env->undo_stack_ptr > 0) {
        Piece cap = env->undo_stack[env->undo_stack_ptr - 1].captured;
        if (cap != NO_PIECE) {
            int pt = type_of_p(cap) - 1;
            if (pt >= 0 && pt < 6) {
                if (color_of(cap) == CHESS_WHITE) env->white_captured[pt]++;
                else env->black_captured[pt]++;
            }
        } else if ((int)type_of_m(chosen) == ENPASSANT) {
            Piece cap_pawn = (side_before == CHESS_WHITE) ? B_PAWN : W_PAWN;
            int pt = type_of_p(cap_pawn) - 1;
            if (pt >= 0 && pt < 6) {
                if (color_of(cap_pawn) == CHESS_WHITE) env->white_captured[pt]++;
                else env->black_captured[pt]++;
            }
        }
        if (env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull > 99) {
            env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull = 99;
        }
    }

    env->repetition_matches = 0;
    if (env->undo_stack_ptr >= 4) {
        int max_back = env->undo_stack[env->undo_stack_ptr - 1].pliesFromNull;
        if (max_back > env->undo_stack_ptr) max_back = env->undo_stack_ptr;
        for (int i = 4; i <= max_back; i += 2) {
            if (env->undo_stack[env->undo_stack_ptr - i].key != env->pos.key) continue;
            env->repetition_matches++;
            if (env->repetition_matches == 2) break;
        }
    }
    
    rebuild_legal_state(env);

    int game_result = 0;
    *is_timeout = 0;
    if (env->chess_moves >= env->max_moves || env->undo_stack_ptr >= MAX_GAME_PLIES - 2) {
        *is_timeout = 1;
        game_result = 3;
    } else if (env->legal_moves.count == 0) {
        if (is_check(&env->pos, env->pos.sideToMove)) {
            game_result = env->pos.sideToMove == CHESS_WHITE ? 1 : 2;
        } else {
            game_result = 3;
        }
    } else if (is_insufficient_material(&env->pos)) {
        game_result = 3;
    } else if (env->enable_50_move_rule && env->pos.rule50 >= 100) {
        game_result = 3;
    } else if (env->enable_threefold_repetition && env->repetition_matches >= 2) {
        game_result = 3;
    }

    return game_result;
}

// ---- CHESS_MODE_MAIA: external lc0/Maia UCI engine ----
// Each env owns one lc0 child process. Communication is line-based UCI:
//   parent → child: "position fen <FEN>\ngo nodes <N>\n"
//   child  → parent: ... "info ..." ... "bestmove <uci>\n"
// Configured via env vars: MAIA_LC0_PATH, MAIA_WEIGHTS_PATH, MAIA_NODES,
// MAIA_BACKEND. MAIA_NODES=1 ≈ weakest Maia setting; raise for stronger play.

#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>

static void position_to_fen(const Position* pos, char* out) {
    char* p = out;
    // Indexed by Piece enum: W_* are 1..6, B_* are 9..14. Gap at 7..8.
    static const char pchars[16] = ".PNBRQK??pnbrqk?";
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            Piece pc = piece_on(pos, make_square(file, rank));
            if (pc == NO_PIECE) {
                empty++;
            } else {
                if (empty > 0) { *p++ = '0' + empty; empty = 0; }
                *p++ = pchars[pc];
            }
        }
        if (empty > 0) *p++ = '0' + empty;
        if (rank > 0) *p++ = '/';
    }
    *p++ = ' ';
    *p++ = (pos->sideToMove == CHESS_WHITE) ? 'w' : 'b';
    *p++ = ' ';
    int wrote_castle = 0;
    if (pos->castlingRights & 1) { *p++ = 'K'; wrote_castle = 1; }
    if (pos->castlingRights & 2) { *p++ = 'Q'; wrote_castle = 1; }
    if (pos->castlingRights & 4) { *p++ = 'k'; wrote_castle = 1; }
    if (pos->castlingRights & 8) { *p++ = 'q'; wrote_castle = 1; }
    if (!wrote_castle) *p++ = '-';
    *p++ = ' ';
    if (pos->epSquare != SQ_NONE && (int)pos->epSquare < 64) {
        *p++ = 'a' + (int)file_of(pos->epSquare);
        *p++ = '1' + (int)rank_of(pos->epSquare);
    } else {
        *p++ = '-';
    }
    *p++ = ' ';
    // Fullmove number isn't tracked on Position; hardcode 1. Halfmove (rule50)
    // matters for the 50-move rule and is passed through.
    p += sprintf(p, "%d 1", pos->rule50);
    *p = '\0';
}

// Parse a UCI move like "e2e4" or "g7g8q" against the env's current legal list.
// Returns MOVE_NONE if not found.
static Move uci_to_move(const char* uci, const MoveList* legal) {
    if (uci[0] < 'a' || uci[0] > 'h' || uci[2] < 'a' || uci[2] > 'h') return MOVE_NONE;
    int from_file = uci[0] - 'a';
    int from_rank = uci[1] - '1';
    int to_file   = uci[2] - 'a';
    int to_rank   = uci[3] - '1';
    if (from_rank < 0 || from_rank > 7 || to_rank < 0 || to_rank > 7) return MOVE_NONE;
    Square from = make_square(from_file, from_rank);
    Square to   = make_square(to_file,   to_rank);
    int promo_pt = -1;
    if (uci[4] == 'q') promo_pt = QUEEN;
    else if (uci[4] == 'r') promo_pt = ROOK;
    else if (uci[4] == 'b') promo_pt = BISHOP;
    else if (uci[4] == 'n') promo_pt = KNIGHT;
    for (int i = 0; i < legal->count; i++) {
        Move m = legal->moves[i].move;
        if (from_sq(m) != from || to_sq(m) != to) continue;
        if (promo_pt >= 0) {
            if (type_of_m(m) != PROMOTION) continue;
            if ((int)promotion_type(m) != promo_pt) continue;
        }
        return m;
    }
    return MOVE_NONE;
}

static int maia_write_all(int fd, const char* buf, int len) {
    int n = 0;
    while (n < len) {
        int w = (int)write(fd, buf + n, (size_t)(len - n));
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        n += w;
    }
    return 0;
}

// Read one line (up to '\n' or buf-1 chars). Returns # bytes (excluding NUL) or
// -1 on error / EOF. Blocks until a line is available.
static int maia_read_line(int fd, char* buf, int bufsz) {
    int n = 0;
    while (n < bufsz - 1) {
        char c;
        int r = (int)read(fd, &c, 1);
        if (r == 0) return -1;  // EOF
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (c == '\n') break;
        buf[n++] = c;
    }
    buf[n] = '\0';
    return n;
}

static void maia_close(Chess* env);

static void maia_init(Chess* env) {
    if (env->maia_pid > 0) return;  // already spawned

    const char* lc0_path     = getenv("MAIA_LC0_PATH");
    const char* weights_path = getenv("MAIA_WEIGHTS_PATH");
    const char* backend_arg  = getenv("MAIA_BACKEND");
    if (lc0_path == NULL)     lc0_path     = "./lc0";
    if (weights_path == NULL) weights_path = "lc0/maia-1100.pb.gz";

    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) < 0 || pipe(out_pipe) < 0) {
        fprintf(stderr, "maia_init: pipe() failed\n");
        env->maia_pid = -1;
        return;
    }
    pid_t pid = fork();
    if (pid < 0) {
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        fprintf(stderr, "maia_init: fork() failed\n");
        env->maia_pid = -1;
        return;
    }
    if (pid == 0) {
        // Child: stdin←in_pipe[0], stdout→out_pipe[1].
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        // Redirect stderr to /dev/null so info logs don't spam the parent.
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        close(in_pipe[0]);  close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        char weights_arg[512];
        snprintf(weights_arg, sizeof(weights_arg), "--weights=%s", weights_path);
        if (backend_arg) {
            char backend_buf[128];
            snprintf(backend_buf, sizeof(backend_buf), "--backend=%s", backend_arg);
            execlp(lc0_path, "lc0", weights_arg, backend_buf, (char*)NULL);
        } else {
            execlp(lc0_path, "lc0", weights_arg, (char*)NULL);
        }
        _exit(127);
    }
    // Parent.
    close(in_pipe[0]);  // we write to in_pipe[1]
    close(out_pipe[1]); // we read from out_pipe[0]
    env->maia_pid       = (int)pid;
    env->maia_stdin_fd  = in_pipe[1];
    env->maia_stdout_fd = out_pipe[0];

    // UCI handshake: send "uci", drain until "uciok"; then "isready", drain
    // until "readyok". Engine init also loads weights so this can take a few
    // seconds the first time.
    char line[1024];
    if (maia_write_all(env->maia_stdin_fd, "uci\n", 4) < 0) { maia_close(env); return; }
    while (maia_read_line(env->maia_stdout_fd, line, sizeof(line)) >= 0) {
        if (strncmp(line, "uciok", 5) == 0) break;
    }
    if (maia_write_all(env->maia_stdin_fd, "isready\n", 8) < 0) { maia_close(env); return; }
    while (maia_read_line(env->maia_stdout_fd, line, sizeof(line)) >= 0) {
        if (strncmp(line, "readyok", 7) == 0) break;
    }
}

static void maia_close(Chess* env) {
    if (env->maia_pid > 0) {
        if (env->maia_stdin_fd >= 0) {
            (void)maia_write_all(env->maia_stdin_fd, "quit\n", 5);
            close(env->maia_stdin_fd);
            env->maia_stdin_fd = -1;
        }
        if (env->maia_stdout_fd >= 0) {
            close(env->maia_stdout_fd);
            env->maia_stdout_fd = -1;
        }
        int status;
        if (waitpid((pid_t)env->maia_pid, &status, WNOHANG) == 0) {
            kill((pid_t)env->maia_pid, SIGTERM);
            waitpid((pid_t)env->maia_pid, &status, 0);
        }
    }
    env->maia_pid = 0;
}

// Ask Maia for the best move at the current env position. Returns a Move that
// is guaranteed to be in env->legal_moves (or MOVE_NONE on engine failure).
static Move maia_get_move(Chess* env) {
    if (env->maia_pid <= 0) {
        maia_init(env);
        if (env->maia_pid <= 0) return MOVE_NONE;
    }

    int nodes = 1;
    const char* nodes_str = getenv("MAIA_NODES");
    if (nodes_str) nodes = atoi(nodes_str);
    if (nodes < 1) nodes = 1;

    char fen[128];
    position_to_fen(&env->pos, fen);
    char cmd[256];
    int n = snprintf(cmd, sizeof(cmd), "position fen %s\ngo nodes %d\n", fen, nodes);
    if (maia_write_all(env->maia_stdin_fd, cmd, n) < 0) {
        maia_close(env);
        return MOVE_NONE;
    }

    char line[1024];
    while (1) {
        int len = maia_read_line(env->maia_stdout_fd, line, sizeof(line));
        if (len < 0) { maia_close(env); return MOVE_NONE; }
        if (strncmp(line, "bestmove ", 9) != 0) continue;
        char uci[8] = {0};
        int j = 9, k = 0;
        while (line[j] && line[j] != ' ' && line[j] != '\n' && line[j] != '\r' && k < 7) {
            uci[k++] = line[j++];
        }
        return uci_to_move(uci, &env->legal_moves);
    }
}

void c_reset(Chess* env) {
    env->tick = 0;
    env->chess_moves = 0;
    env->game_result = 0;
    env->undo_stack_ptr = 0;
    env->repetition_matches = 0;
    env->invalid_actions_this_episode = 0;
    env->episode_reward = 0.0f;
    env->pgn_move_count = 0;
    env->show_game_end_popup = 0;
    env->has_last_move_highlight = 0;
    clear_player_selection(env, 0);
    clear_player_selection(env, 1);
    env->valid_from_mask[0] = 0;
    env->valid_from_mask[1] = 0;
    
    memset(env->white_captured, 0, sizeof(env->white_captured));
    memset(env->black_captured, 0, sizeof(env->black_captured));
    
    if (env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM) {
        env->human_color = -1;
    } else if (env->mode != CHESS_MODE_SELFPLAY) {
        env->learner_color = 1 - env->learner_color;
    }
    env->maia_phase = 0;
    
    if (env->fen_curriculum != NULL && env->num_fens > 0) {
        float randvalue = (float)rand_r(&env->rng) / (float)(RAND_MAX);
        if(env->fen_curric_pct >= randvalue){
            int idx = rand_r(&env->rng) % env->num_fens;
            pos_set(&env->pos, env->fen_curriculum[idx]);
        }
        else {
            pos_set(&env->pos, env->starting_fen);
        }

    } else if (env->random_fen) {
        char fen_buf[128];
        generate_random_fen(env, fen_buf);
        pos_set(&env->pos, fen_buf);
    } else {
        pos_set(&env->pos, env->starting_fen);
    }
    
    rebuild_legal_state(env);
    populate_observations(env);

}

void c_step(Chess* env) {
    if (env->render_paused && env->client != NULL) {
        return;
    }
    if ((env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
            && env->human_color == -1) {
        return;
    }
    
    if (env->mode == CHESS_MODE_SELFPLAY && !env->log_pgn_choice_made) {
        if (env->client != NULL) {
            return;
        }
        env->log_pgn = 0;
        env->log_pgn_choice_made = 1;
    }

    if ((env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
            && env->show_game_end_popup) {
        *env->reward_ptr[0] = 0.0f;
        *env->terminal_ptr[0] = 0;
        return;
    }

    if (env->legal_dirty) {
        rebuild_legal_state(env);
    }
    
    env->tick++;
    int move_completed = 0;
    ChessColor mover = env->pos.sideToMove;
    int mover_idx = (int)mover;
    int game_result = 0;
    int is_timeout = 0;

    if ((env->mode == CHESS_MODE_RANDOM && env->pos.sideToMove != env->learner_color)
            || (env->mode == CHESS_MODE_HUMAN_RANDOM && env->pos.sideToMove != env->human_color)) {
        if (env->legal_moves.count > 0) {
            int idx = rand_r(&env->rng) % env->legal_moves.count;
            clear_player_selection(env, mover_idx);
            game_result = apply_move_to_env(env, env->legal_moves.moves[idx].move, &is_timeout);
            move_completed = 1;
        }
    } else if (env->mode == CHESS_MODE_MAIA && env->pos.sideToMove != env->learner_color) {
        if (env->legal_moves.count > 0) {
            if (env->maia_phase == 0) {
                // First c_step of Maia's "move": no-op so the learner's LSTM
                // sees a 2-step opponent wait (matches selfplay's pick+place
                // cadence the policy was trained on).
                env->maia_phase = 1;
                move_completed = 1;
            } else {
                Move maia_mv = maia_get_move(env);
                if (maia_mv == MOVE_NONE) {
                    // Engine failure / unparseable bestmove: fall back to random
                    // so the trial completes. Logged via log.maia_failures so
                    // Python can flag a degraded eval.
                    env->log.maia_failures += 1.0f;
                    int idx = rand_r(&env->rng) % env->legal_moves.count;
                    maia_mv = env->legal_moves.moves[idx].move;
                }
                clear_player_selection(env, mover_idx);
                game_result = apply_move_to_env(env, maia_mv, &is_timeout);
                env->maia_phase = 0;
                move_completed = 1;
            }
        }
    } else {
        // Selfplay: side-to-move's action lives in whichever slot plays that color.
        int action = (env->mode == CHESS_MODE_SELFPLAY)
            ? *env->action_ptr[env->slot_for_color[mover_idx]]
            : *env->action_ptr[0];
        if ((env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
                && env->pos.sideToMove == env->human_color) {
            action = -1;
            *env->action_ptr[0] = -1;
        }

        mover = env->pos.sideToMove;
        mover_idx = (int)mover;

        // In selfplay both players train, so charge whichever slot moved.
        // In single-agent modes, only the learner's slot exists.
        int penalty_slot = (env->mode == CHESS_MODE_SELFPLAY) ? env->slot_for_color[mover_idx]
                         : (mover == env->learner_color) ? 0 : -1;

        if (env->legal_moves.count == 0) {
            clear_player_selection(env, mover_idx);
        } else if (action < 0 || action >= PASS_ACTION) {
            if (penalty_slot >= 0) {
                *env->reward_ptr[penalty_slot] += (env->pick_phase[mover_idx] == 0)
                    ? env->reward_invalid_piece : env->reward_invalid_move;
                env->invalid_actions_this_episode++;
            }
            if (env->pick_phase[mover_idx] == 1) {
                clear_player_selection(env, mover_idx);
            }
        } else {
            bool is_promo = (action >= 64 && action < 96);

            if (env->pick_phase[mover_idx] == 0) {
                clear_player_selection(env, mover_idx);

                bool valid_pick = !is_promo;
                Square picked_sq = SQ_NONE;
                if (valid_pick) {
                    picked_sq = (mover == CHESS_BLACK) ? (Square)(action ^ 56) : (Square)action;
                    Piece pc = piece_on(&env->pos, picked_sq);
                    valid_pick = (pc != NO_PIECE && color_of(pc) == mover);
                }

                if (valid_pick) {
                    MoveList* dests = &env->valid_destinations[mover_idx];
                    dests->count = 0;
                    Bitboard to_mask = 0;
                    for (int i = 0; i < env->legal_moves.count; i++) {
                        if (from_sq(env->legal_moves.moves[i].move) == picked_sq) {
                            dests->moves[dests->count++] = env->legal_moves.moves[i];
                            to_mask |= sq_bb(to_sq(env->legal_moves.moves[i].move));
                        }
                    }

                    if (dests->count > 0) {
                        env->selected_square[mover_idx] = picked_sq;
                        env->pick_phase[mover_idx] = 1;
                        env->valid_to_mask[mover_idx] = to_mask;
                    } else {
                        valid_pick = false;
                        clear_player_selection(env, mover_idx);
                    }
                }

                if (!valid_pick && penalty_slot >= 0) {
                    *env->reward_ptr[penalty_slot] += env->reward_invalid_piece;
                    env->invalid_actions_this_episode++;
                }
            } else {
                if (env->selected_square[mover_idx] == SQ_NONE || env->valid_destinations[mover_idx].count == 0) {
                    fprintf(stderr, "c_step: pick_phase=1 but selected_square=%u, valid_destinations.count=%d (mover=%d)\n",
                            env->selected_square[mover_idx], env->valid_destinations[mover_idx].count, mover_idx);
                    exit(1);
                }

                Square target_sq = SQ_NONE;
                Move chosen_move = MOVE_NONE;
                int desired_promo = -1;
                int desired_file = -1;

                if (is_promo) {
                    int promo_row = (action - 64) / 8;
                    desired_file = (action - 64) % 8;
                    desired_promo = QUEEN - promo_row;
                } else {
                    target_sq = (mover == CHESS_BLACK) ? (Square)(action ^ 56) : (Square)action;
                }

                for (int i = 0; i < env->valid_destinations[mover_idx].count; i++) {
                    Move m = env->valid_destinations[mover_idx].moves[i].move;
                    if (!is_promo) {
                        if ((int)to_sq(m) == (int)target_sq) {
                            chosen_move = m;
                            break;
                        }
                    } else {
                        if ((int)type_of_m(m) == PROMOTION
                                && (int)promotion_type(m) == desired_promo
                                && (int)file_of(to_sq(m)) == desired_file) {
                            chosen_move = m;
                            break;
                        }
                    }
                }

                if (chosen_move == MOVE_NONE) {
                    if (penalty_slot >= 0) {
                        *env->reward_ptr[penalty_slot] += env->reward_invalid_move;
                        env->invalid_actions_this_episode++;
                    }
                    clear_player_selection(env, mover_idx);
                } else {
                    game_result = apply_move_to_env(env, chosen_move, &is_timeout);
                    if (env->reward_repetition != 0.0f
                            && penalty_slot >= 0
                            && env->repetition_matches >= 1) {
                        *env->reward_ptr[penalty_slot] += env->reward_repetition;
                    }
                    move_completed = 1;
                }
            }
        }
    }

    if (!move_completed) {
        if (env->chess_moves >= env->max_moves || env->undo_stack_ptr >= MAX_GAME_PLIES - 2) {
            game_result = 3;
            is_timeout = 1;
        } else {
            if (env->legal_moves.count == 0) {
                if (is_check(&env->pos, env->pos.sideToMove)) {
                    game_result = env->pos.sideToMove == CHESS_WHITE ? 1 : 2;
                } else {
                    game_result = 3;
                }
            } else if (is_insufficient_material(&env->pos)) {
                game_result = 3;
            } else if (env->enable_50_move_rule && env->pos.rule50 >= 100) {
                game_result = 3;
            } else if (env->enable_threefold_repetition && env->repetition_matches >= 2) {
                game_result = 3;
            }
        }
    }

    if (game_result != 0) {
        *env->terminal_ptr[0] = 1;
        if (env->mode == CHESS_MODE_SELFPLAY) {
            *env->terminal_ptr[1] = 1;
        }
        env->game_result = game_result;
        float win_value = 0.0f;

        switch (game_result) {
            case 3:
                *env->reward_ptr[0] = env->reward_draw;
                if (env->mode == CHESS_MODE_SELFPLAY) {
                    *env->reward_ptr[1] = env->reward_draw;
                    env->log.slot_0_score += 0.5f;
                    env->log.slot_1_score += 0.5f;
                }
                win_value = 0.5f;
                env->log.draw_rate += 1.0f;
                if (is_timeout) {
                    env->log.timeout_rate += 1.0f;
                }
                env->white_score += 0.5f;
                env->black_score += 0.5f;
                env->learner_draws += 1.0f;
                strcpy(env->last_result, "Draw");
                break;
            case 1:
                env->black_score += 1.0f;
                if (env->mode == CHESS_MODE_SELFPLAY) {
                    *env->reward_ptr[env->slot_for_color[CHESS_WHITE]] = -1.0f;
                    *env->reward_ptr[env->slot_for_color[CHESS_BLACK]] = 1.0f;
                    if (env->slot_for_color[CHESS_BLACK] == 0) env->log.slot_0_score += 1.0f;
                    else env->log.slot_1_score += 1.0f;
                    win_value = 0.5f;             // zero-sum: averaged across both slots
                } else if (env->learner_color == CHESS_WHITE) {
                    *env->reward_ptr[0] = -1.0f;
                    env->learner_losses += 1.0f;
                } else {
                    *env->reward_ptr[0] = 1.0f;
                    win_value = 1.0f;
                    env->learner_wins += 1.0f;
                }
                strcpy(env->last_result, "Black Wins");
                break;
            case 2:
                env->white_score += 1.0f;
                if (env->mode == CHESS_MODE_SELFPLAY) {
                    *env->reward_ptr[env->slot_for_color[CHESS_WHITE]] = 1.0f;
                    *env->reward_ptr[env->slot_for_color[CHESS_BLACK]] = -1.0f;
                    if (env->slot_for_color[CHESS_WHITE] == 0) env->log.slot_0_score += 1.0f;
                    else env->log.slot_1_score += 1.0f;
                    win_value = 0.5f;
                } else if (env->learner_color == CHESS_WHITE) {
                    *env->reward_ptr[0] = 1.0f;
                    win_value = 1.0f;
                    env->learner_wins += 1.0f;
                } else {
                    *env->reward_ptr[0] = -1.0f;
                    env->learner_losses += 1.0f;
                }
                strcpy(env->last_result, "White Wins");
                break;
            default:
                break;
        }

        if (env->mode == CHESS_MODE_SELFPLAY) {
            env->episode_reward += *env->reward_ptr[0] + *env->reward_ptr[1];
        } else {
            env->episode_reward += *env->reward_ptr[0];
        }
        env->log.episode_return += env->episode_reward;
        env->log.perf += win_value;
        env->log.score += win_value;
        env->log.chess_moves += env->chess_moves;
        env->log.episode_length += env->tick;
        env->log.invalid_action_rate += (env->tick > 0)
            ? ((float)env->invalid_actions_this_episode / (float)env->tick) : 0.0f;

        env->log.n += 1.0f;

        // Per-color split for non-selfplay eval. learner_color is the color the
        // learner just played; win_value already accounts for whether it won.
        // A lopsided split (e.g. high white win rate, near-zero black) is the
        // signature of a broken obs flip / wrong perspective for one color.
        if (env->mode != CHESS_MODE_SELFPLAY) {
            if (env->learner_color == CHESS_WHITE) {
                env->log.wins_as_white += win_value;
                env->log.games_as_white += 1.0f;
            } else {
                env->log.wins_as_black += win_value;
                env->log.games_as_black += 1.0f;
            }
        }

        // Per-bank historical tracking. Tag = 1..CHESS_MAX_BANKS picks the bank
        // index (tag-1) this env was assigned to play. Tag = 0 is pure selfplay
        // (skip historical accounting). Backward compat: tag=1 with single bank
        // still routes into hist_score_bank[0] / hist_n_bank[0].
        if (env->tag > 0 && env->tag <= CHESS_MAX_BANKS) {
            int bank_idx = env->tag - 1;
            float primary_score;
            if (game_result == 3) {
                primary_score = 0.5f;
            } else if (game_result == 2) {  // White wins
                primary_score = (env->slot_for_color[CHESS_WHITE] == 0) ? 1.0f : 0.0f;
            } else {  // Black wins
                primary_score = (env->slot_for_color[CHESS_BLACK] == 0) ? 1.0f : 0.0f;
            }
            env->log.hist_score_bank[bank_idx] += primary_score;
            env->log.hist_n_bank[bank_idx] += 1.0f;
            // Legacy aggregate fields — sum across all banks.
            env->log.hist_score += primary_score;
            env->log.hist_n += 1.0f;
            env->boundary_reached = 1;
        }

        if (env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM) {
            env->show_game_end_popup = 1;
        } else {
            if (env->log_pgn && env->pgn_filename[0] != '\0') {
                env->pgn_game_number++;
                export_pgn_append(env, env->pgn_filename, 1);
            }
            c_reset(env);
        }
    } else {
        if (env->mode == CHESS_MODE_SELFPLAY) {
            env->episode_reward += *env->reward_ptr[0] + *env->reward_ptr[1];
        } else {
            env->episode_reward += *env->reward_ptr[0];
        }
    }

    populate_observations(env);
}
static Font load_piece_font(int cell_size, int* loaded) {
    const char* candidates[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Apple Symbols.ttf",
        "C:\\Windows\\Fonts\\seguisym.ttf"
    };

    int codepoints[] = {0x2654, 0x2655, 0x2656, 0x2657, 0x2658, 0x2659, 0x265A, 0x265B, 0x265C, 0x265D, 0x265E, 0x265F};
    Font font = (Font){0};
    size_t candidate_count = sizeof(candidates) / sizeof(candidates[0]);
    size_t codepoint_count = sizeof(codepoints) / sizeof(codepoints[0]);

    for (size_t i = 0; i < candidate_count; i++) {
        if (!FileExists(candidates[i])) {
            continue;
        }
        font = LoadFontEx(candidates[i], cell_size, codepoints, (int)codepoint_count);
        if (font.texture.id != 0) {
            if (loaded) {
                *loaded = 1;
            }
            SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);
            return font;
        }
    }

    if (loaded) {
        *loaded = 0;
    }
    return GetFontDefault();
}

static void draw_piece(Chess* env, Piece pc, int file, int rank, int cell_size) {
    if (pc == NO_PIECE) {
        return;
    }
    
    Color pc_color = color_of(pc) == CHESS_WHITE 
        ? (Color){255, 255, 255, 255}
        : (Color){0, 0, 0, 255};
    
    Color outline = (color_of(pc) == CHESS_WHITE) 
        ? (Color){0, 0, 0, 220} 
        : (Color){255, 255, 255, 180};

    int draw_x = file * cell_size;
    int draw_y = (7 - rank) * cell_size;

    if (env->client && env->client->use_unicode_pieces) {
        float icon_size = cell_size * 0.85f;
        Vector2 pos = (Vector2){
            draw_x + (cell_size - icon_size) / 2.0f,
            draw_y + (cell_size - icon_size) / 2.0f - cell_size * 0.05f
        };
        const char* str = PIECE_FILLED[pc];
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx != 0 || dy != 0) {
                    Vector2 opos = (Vector2){pos.x + dx, pos.y + dy};
                    DrawTextEx(env->client->piece_font, str, opos, icon_size, 0, outline);
                }
            }
        }
        DrawTextEx(env->client->piece_font, str, pos, icon_size, 0, pc_color);
    } else {
        int x = draw_x + cell_size / 4;
        int y = draw_y + cell_size / 8;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx != 0 || dy != 0) {
                    DrawText(PIECE_CHARS[pc], x + dx, y + dy, cell_size / 2, outline);
                }
            }
        }
        DrawText(PIECE_CHARS[pc], x, y, cell_size / 2, pc_color);
    }
}

static void init_chess_client(Chess* env, int cell_size) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    int board_size = 8 * cell_size;
    InitWindow(board_size, board_size + 140, "PufferLib Chess - AI vs Opponent");
    SetTargetFPS(env->render_fps > 0 ? env->render_fps : 30);
    env->client = (Client*)calloc(1, sizeof(Client));
    env->client->cell_size = cell_size;
    int font_loaded = 0;
    env->client->piece_font = load_piece_font(cell_size, &font_loaded);
    env->client->use_unicode_pieces = font_loaded;
    if (env->mode == CHESS_MODE_SELFPLAY) env->log_pgn_choice_made = 0;
}

void c_render(Chess* env) {
    const int cell_size = 64;
    const int board_size = 8 * cell_size;
    const int scoreboard_y = board_size + 10;
    static int speed_idx = 3;
    static const int SPEED_FPS[] = {2, 5, 10, 30, 60, 120, 0};
    static const int NUM_SPEEDS = 7;
    static int selected_sq = -1;
    
    if (env->client == NULL) {
        init_chess_client(env, cell_size);
    }

human_wait_retry:
    if (IsKeyDown(KEY_ESCAPE) || WindowShouldClose()) { CloseWindow(); exit(0); }
    
    int flip_board = ((env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
        && env->human_color == CHESS_BLACK) ? 1 : 0;
    Vector2 mouse = GetMousePosition();
    int clicked = IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
    
    if (IsKeyPressed(KEY_SPACE)) env->render_paused = !env->render_paused;
    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
        if (speed_idx < NUM_SPEEDS - 1) { speed_idx++; SetTargetFPS(SPEED_FPS[speed_idx]); }
    }
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
        if (speed_idx > 0) { speed_idx--; SetTargetFPS(SPEED_FPS[speed_idx]); }
    }
    
    if (!env->render_paused
            && (env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
            && env->human_color != -1
            && !env->show_game_end_popup
            && clicked) {
        int file = (int)(mouse.x) / cell_size;
        int rank = 7 - ((int)(mouse.y) / cell_size);
        if (flip_board) { file = 7 - file; rank = 7 - rank; }
        if (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
            int clicked_sq = (int)make_square(file, rank);
            if (selected_sq == -1) {
                if (env->pos.sideToMove == env->human_color) {
                    Piece pc = piece_on(&env->pos, (Square)clicked_sq);
                    if (pc != NO_PIECE && color_of(pc) == env->human_color) {
                        bool has_from = false;
                        for (int i = 0; i < env->legal_moves.count; i++) {
                            if ((int)from_sq(env->legal_moves.moves[i].move) == clicked_sq) { has_from = true; break; }
                        }
                        if (has_from) selected_sq = clicked_sq;
                    }
                }
            } else {
                Move chosen = MOVE_NONE;
                for (int i = 0; i < env->legal_moves.count; i++) {
                    Move m = env->legal_moves.moves[i].move;
                    if ((int)from_sq(m) == selected_sq && (int)to_sq(m) == clicked_sq) { chosen = m; break; }
                }
                if (chosen != MOVE_NONE) {
                    int is_timeout = 0;
                    apply_move_to_env(env, chosen, &is_timeout);
                    *env->action_ptr[0] = -1;
                }
                selected_sq = -1;
            }
        }
    }

    BeginDrawing();
    ClearBackground((Color){40, 40, 40, 255});
    
    if ((env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
            && env->show_game_end_popup) {
        int pw = 300, ph = 200;
        int px = (board_size - pw) / 2, py = (board_size - ph) / 2;
        DrawRectangle(px, py, pw, ph, (Color){60, 60, 60, 255});
        DrawRectangleLines(px, py, pw, ph, WHITE);
        DrawText("Game Over!", px + 70, py + 20, 24, WHITE);
        DrawText(env->last_result, px + 80, py + 55, 18, YELLOW);
        
        Rectangle save_btn = {px + 20, py + 110, 120, 35};
        Rectangle new_btn  = {px + 160, py + 110, 120, 35};
        DrawRectangleRec(save_btn, DARKGREEN);
        DrawRectangleLinesEx(save_btn, 2, WHITE);
        DrawText("Save PGN", px + 35, py + 120, 16, WHITE);
        DrawRectangleRec(new_btn, DARKBLUE);
        DrawRectangleLinesEx(new_btn, 2, WHITE);
        DrawText("New Game", px + 175, py + 120, 16, WHITE);
        
        if (clicked) {
            if (CheckCollisionPointRec(mouse, save_btn)) {
                char filename[64];
                snprintf(filename, sizeof(filename), "game_%d.pgn", (int)time(NULL));
                export_pgn_append(env, filename, 0);
                printf("Saved PGN to %s\n", filename);
            } else if (CheckCollisionPointRec(mouse, new_btn)) {
                c_reset(env);
            }
        }
    } else if (env->mode == CHESS_MODE_SELFPLAY && !env->log_pgn_choice_made) {
        int cx = board_size / 2;
        DrawText("Log PGN Files?", cx - 80, 180, 24, WHITE);
        DrawText("Games will be appended to a timestamped file", cx - 160, 220, 14, LIGHTGRAY);
        
        Rectangle yes_btn = {cx - 70, 270, 140, 40};
        Rectangle no_btn  = {cx - 70, 330, 140, 40};
        DrawRectangleRec(yes_btn, DARKGREEN);
        DrawRectangleLinesEx(yes_btn, 2, WHITE);
        DrawText("Yes, Log PGN", cx - 55, 282, 16, WHITE);
        DrawRectangleRec(no_btn, MAROON);
        DrawRectangleLinesEx(no_btn, 2, WHITE);
        DrawText("No Logging", cx - 45, 342, 16, WHITE);
        
        if (clicked) {
            if (CheckCollisionPointRec(mouse, yes_btn)) {
                env->log_pgn = 1;
                env->log_pgn_choice_made = 1;
                env->pgn_game_number = 0;
                snprintf(env->pgn_filename, sizeof(env->pgn_filename), "run_%d_pgns.pgn", (int)time(NULL));
                printf("PGN logging enabled: %s\n", env->pgn_filename);
            } else if (CheckCollisionPointRec(mouse, no_btn)) {
                env->log_pgn = 0;
                env->log_pgn_choice_made = 1;
                printf("PGN logging disabled\n");
            }
        }
    } else if ((env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
            && env->human_color == -1) {
        int cx = board_size / 2;
        DrawText("Choose Your Color", cx - 100, 200, 24, WHITE);
        
        Rectangle white_btn = {cx - 60, 280, 120, 40};
        Rectangle black_btn = {cx - 60, 340, 120, 40};
        DrawRectangleRec(white_btn, LIGHTGRAY);
        DrawRectangleLinesEx(white_btn, 2, BLACK);
        DrawText("Play White", cx - 45, 292, 18, BLACK);
        DrawRectangleRec(black_btn, GRAY);
        DrawRectangleLinesEx(black_btn, 2, BLACK);
        DrawText("Play Black", cx - 45, 352, 18, WHITE);
        
        if (clicked) {
            if (CheckCollisionPointRec(mouse, white_btn)) {
                env->human_color = CHESS_WHITE;
                env->learner_color = CHESS_BLACK;
            } else if (CheckCollisionPointRec(mouse, black_btn)) {
                env->human_color = CHESS_BLACK;
                env->learner_color = CHESS_WHITE;
            }
        }
    } else {
        Bitboard selected_destinations = 0;
        if (selected_sq != -1) {
            for (int i = 0; i < env->legal_moves.count; i++) {
                Move m = env->legal_moves.moves[i].move;
                if ((int)from_sq(m) == selected_sq) {
                    selected_destinations |= sq_bb(to_sq(m));
                }
            }
        }
        int selected_file = -1;
        int selected_rank = -1;
        if (selected_sq != -1) {
            selected_file = file_of((Square)selected_sq);
            selected_rank = rank_of((Square)selected_sq);
        }
        for (int rank = 0; rank < 8; rank++) {
            for (int file = 0; file < 8; file++) {
                Color sq_color = ((rank + file) % 2 == 1) ? (Color){240, 217, 181, 255} : (Color){181, 136, 99, 255};
                int draw_file = flip_board ? (7 - file) : file;
                int draw_rank = flip_board ? (7 - rank) : rank;
                int draw_x = draw_file * cell_size;
                int draw_y = (7 - draw_rank) * cell_size;
                DrawRectangle(draw_x, draw_y, cell_size, cell_size, sq_color);

                if (env->has_last_move_highlight) {
                    Square lf = env->last_move_from;
                    Square lt = env->last_move_to;
                    if ((file == (int)file_of(lf) && rank == (int)rank_of(lf))
                            || (file == (int)file_of(lt) && rank == (int)rank_of(lt))) {
                        Color last_mv = (Color){247, 247, 105, 255};
                        DrawRectangle(draw_x, draw_y, cell_size, cell_size, Fade(last_mv, 0.52f));
                    }
                }

                if (selected_sq != -1 && selected_file == file && selected_rank == rank) {
                    DrawRectangleLines(draw_x, draw_y, cell_size, cell_size, (Color){255, 215, 0, 255});
                }
                if (selected_sq != -1 && (selected_destinations & sq_bb(make_square(file, rank)))) {
                    DrawRectangleLines(draw_x + 2, draw_y + 2, cell_size - 4, cell_size - 4, (Color){0, 200, 0, 255});
                }
            }
        }
        for (int pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = pieces_p(&env->pos, pt);
            while (bb) {
                Square sq = pop_lsb(&bb);
                Piece pc = piece_on(&env->pos, sq);
                int f = file_of(sq), r = rank_of(sq);
                int draw_f = flip_board ? (7 - f) : f;
                int draw_r = flip_board ? (7 - r) : r;
                draw_piece(env, pc, draw_f, draw_r, cell_size);
            }
        }
        
        char buf[128];
        snprintf(buf, sizeof(buf), "White: %.1f  Black: %.1f", env->white_score, env->black_score);
        DrawText(buf, 10, scoreboard_y, 20, WHITE);
        
        snprintf(buf, sizeof(buf), "Learner: %.0f-%.0f-%.0f (W-L-D)", env->learner_wins, env->learner_losses, env->learner_draws);
        DrawText(buf, 10, scoreboard_y + 22, 16, GREEN);
        
        snprintf(buf, sizeof(buf), "Move: %d", env->chess_moves);
        DrawText(buf, board_size - 100, scoreboard_y, 18, LIGHTGRAY);
        
        if (env->mode != CHESS_MODE_HUMAN && env->mode != CHESS_MODE_HUMAN_RANDOM) {
            DrawText(env->learner_color == CHESS_WHITE ? "Learner: White" : "Learner: Black",
                     board_size - 120, scoreboard_y + 22, 16, LIGHTGRAY);
        }
        
        int cap_y = scoreboard_y + 42;
        int cap_x_start = 10;
        Color white_cap_color = (Color){240, 217, 181, 255};
        Color black_cap_color = (Color){100, 100, 100, 255};
        int white_x = cap_x_start;
        int black_x = cap_x_start;

        for (int pt = 0; pt < 6; pt++) {
            int wc = env->white_captured[pt];
            if (wc > 0) {
                Piece wpc = (Piece)(W_PAWN + pt);
                if (env->client && env->client->use_unicode_pieces) {
                    DrawTextEx(env->client->piece_font, PIECE_FILLED[wpc],
                        (Vector2){(float)white_x, (float)(cap_y - 1)}, 16.0f, 0.0f, white_cap_color);
                    white_x += 16;
                } else {
                    DrawText(PIECE_CHARS[wpc], white_x, cap_y, 14, white_cap_color);
                    white_x += 12;
                }
                if (wc > 1) {
                    char mult[8];
                    snprintf(mult, sizeof(mult), "x%d", wc);
                    DrawText(mult, white_x, cap_y + 2, 10, white_cap_color);
                    white_x += MeasureText(mult, 10) + 4;
                } else {
                    white_x += 4;
                }
            }

            int bc = env->black_captured[pt];
            if (bc > 0) {
                Piece bpc = (Piece)(B_PAWN + pt);
                Color outline = (Color){255, 255, 255, 180};
                if (env->client && env->client->use_unicode_pieces) {
                    Vector2 pos = {(float)black_x, (float)(cap_y + 17)};
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            if (dx != 0 || dy != 0) {
                                DrawTextEx(env->client->piece_font, PIECE_FILLED[bpc],
                                    (Vector2){pos.x + dx, pos.y + dy}, 16.0f, 0.0f, outline);
                            }
                        }
                    }
                    DrawTextEx(env->client->piece_font, PIECE_FILLED[bpc], pos, 16.0f, 0.0f, black_cap_color);
                    black_x += 16;
                } else {
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            if (dx != 0 || dy != 0)
                                DrawText(PIECE_CHARS[bpc], black_x + dx, cap_y + 18 + dy, 14, outline);
                        }
                    }
                    DrawText(PIECE_CHARS[bpc], black_x, cap_y + 18, 14, black_cap_color);
                    black_x += 12;
                }
                if (bc > 1) {
                    char mult[8];
                    snprintf(mult, sizeof(mult), "x%d", bc);
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            if (dx != 0 || dy != 0)
                                DrawText(mult, black_x + dx, cap_y + 20 + dy, 10, outline);
                        }
                    }
                    DrawText(mult, black_x, cap_y + 20, 10, black_cap_color);
                    black_x += MeasureText(mult, 10) + 4;
                } else {
                    black_x += 4;
                }
            }
        }

        if (env->last_result[0] != '\0') {
            Color rc = YELLOW;
            if (strstr(env->last_result, "White")) rc = (Color){240, 217, 181, 255};
            else if (strstr(env->last_result, "Black")) rc = (Color){100, 100, 100, 255};
            DrawText(env->last_result, 10, cap_y + 40, 18, rc);
        }
        
        int btn_w = 36;
        int btn_h = 24;
        int btn_y = scoreboard_y + 100;
        int btn_x = (env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
            ? board_size / 2 - 100 : board_size / 2 - 70;
        Rectangle minus_btn = {btn_x, btn_y, btn_w, btn_h};
        Rectangle pause_btn = {btn_x + btn_w + 5, btn_y, btn_w + 10, btn_h};
        Rectangle plus_btn = {btn_x + 2 * btn_w + 20, btn_y, btn_w, btn_h};
        DrawRectangleRec(minus_btn, DARKGRAY);
        DrawRectangleLinesEx(minus_btn, 2, LIGHTGRAY);
        DrawText("-", btn_x + 14, btn_y + 4, 20, WHITE);
        DrawRectangleRec(pause_btn, env->render_paused ? MAROON : DARKGREEN);
        DrawRectangleLinesEx(pause_btn, 2, LIGHTGRAY);
        DrawText(env->render_paused ? ">" : "||", btn_x + btn_w + 14, btn_y + 4, 18, WHITE);
        DrawRectangleRec(plus_btn, DARKGRAY);
        DrawRectangleLinesEx(plus_btn, 2, LIGHTGRAY);
        DrawText("+", btn_x + 2 * btn_w + 32, btn_y + 4, 20, WHITE);
        char speed_buf[32];
        if (SPEED_FPS[speed_idx] == 0) {
            snprintf(speed_buf, sizeof(speed_buf), "max");
        } else {
            snprintf(speed_buf, sizeof(speed_buf), "%dfps", SPEED_FPS[speed_idx]);
        }
        DrawText(speed_buf, btn_x + 3 * btn_w + 30, btn_y + 4, 14, env->render_paused ? RED : LIGHTGRAY);
        
        Rectangle restart_btn = {0, 0, 0, 0};
        if (env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM) {
            restart_btn = (Rectangle){board_size - 60, minus_btn.y, 55, minus_btn.height};
            DrawRectangleRec(restart_btn, MAROON);
            DrawRectangleLinesEx(restart_btn, 2, LIGHTGRAY);
            DrawText("Exit", board_size - 53, minus_btn.y + 4, 16, WHITE);
        }
        
        if (env->render_paused) {
            DrawRectangle(0, 0, board_size, board_size, (Color){0, 0, 0, 120});
            DrawText("PAUSED", board_size / 2 - 60, board_size / 2 - 15, 30, RED);
        }
        
        if (clicked) {
            if (CheckCollisionPointRec(mouse, minus_btn)) {
                if (speed_idx > 0) { speed_idx--; SetTargetFPS(SPEED_FPS[speed_idx]); }
            }
            if (CheckCollisionPointRec(mouse, pause_btn)) env->render_paused = !env->render_paused;
            if (CheckCollisionPointRec(mouse, plus_btn)) {
                if (speed_idx < NUM_SPEEDS - 1) { speed_idx++; SetTargetFPS(SPEED_FPS[speed_idx]); }
            }
            if ((env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
                    && CheckCollisionPointRec(mouse, restart_btn)) c_reset(env);
        }
    }
    
    EndDrawing();

    // Human-mode only: stay in c_render (on the window-owning thread) until
    // the human commits a move via mouse clicks. Re-poll input + redraw each
    // iteration. Non-human modes fall through to a single c_render call as
    // before. Refresh obs after the commit so the next rollout inference sees
    // the post-human-move state instead of the stale "human's turn" obs.
    if ((env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
            && env->human_color != -1
            && env->pos.sideToMove == env->human_color
            && !env->show_game_end_popup
            && !env->render_paused
            && !WindowShouldClose()) {
        goto human_wait_retry;
    }
    if ((env->mode == CHESS_MODE_HUMAN || env->mode == CHESS_MODE_HUMAN_RANDOM)
            && env->human_color != -1
            && env->pos.sideToMove != env->human_color
            && !env->show_game_end_popup) {
        if (env->legal_dirty) rebuild_legal_state(env);
        populate_observations(env);
    }
}

void c_close(Chess* env) {
    if (env->client != NULL) {
        if (env->client->use_unicode_pieces && env->client->piece_font.texture.id != 0) {
            UnloadFont(env->client->piece_font);
        }
        if (IsWindowReady()) {
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
    maia_close(env);
    env->fen_curriculum = NULL;
    env->num_fens = 0;
}
