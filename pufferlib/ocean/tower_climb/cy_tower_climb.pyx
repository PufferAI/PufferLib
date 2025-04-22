from libc.stdlib cimport calloc, free, rand
from libc.stdint cimport uint64_t
import numpy as np
cdef extern from "tower_climb.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float rows_cleared;
        float levels_completed;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Level:
        const int* map;
        int rows;
        int cols;
        int size;
        int total_length;
        int goal_location;
        int spawn_location;

    ctypedef struct PuzzleState:
        unsigned char* blocks;
        int robot_position;
        int robot_orientation;
        int robot_state;
        int block_grabbed;

    ctypedef struct VisitedNode:
        PuzzleState state;
        uint64_t hashVal;
        VisitedNode* next;

    ctypedef struct BFSNode:
        PuzzleState state;
        int depth;
        int parent;
        int action;

    ctypedef struct CTowerClimb:
        unsigned char* observations;
        int* actions;
        float* rewards;
        unsigned char* dones;
        LogBuffer* log_buffer;
        Log log;
        float score;
        Level* level;
        PuzzleState* state;
        int distance_to_goal;
        float reward_climb_row;
        float reward_fall_row;
        float reward_illegal_move;
        float reward_move_block;

    ctypedef struct Client:
        int enable_animations;
        int isMoving;

    void init(CTowerClimb* env)
    void free_allocated(CTowerClimb* env)
    void init_level(Level* level)
    void init_puzzle_state(PuzzleState* state)
    void init_random_level(CTowerClimb* env, int goal_height, int max_moves, int min_moves,  int seed)
    void cy_init_random_level(Level* level, int goal_height, int max_moves, int min_moves, int seed)
    void levelToPuzzleState(Level* level, PuzzleState* state)
    void setPuzzle(CTowerClimb* dest, PuzzleState* src, Level* lvl)


    Client* make_client(CTowerClimb* env)
    void close_client(Client* client)
    void c_render(Client* client, CTowerClimb* env)
    void c_reset(CTowerClimb* env)
    int c_step(CTowerClimb* env)

cdef class CyTowerClimb:
    cdef:
        CTowerClimb* envs
        Level* levels
        PuzzleState* puzzle_states
        Client* client
        LogBuffer* logs
        int num_envs
        int num_maps

    def __init__(self, unsigned char[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            int num_maps, float reward_climb_row, float reward_fall_row,
            float reward_illegal_move, float reward_move_block):

        self.client = NULL
        self.num_envs = num_envs
        self.num_maps = num_maps
        self.levels = <Level*> calloc(num_maps, sizeof(Level))
        self.puzzle_states = <PuzzleState*> calloc(num_maps, sizeof(PuzzleState))
        self.envs = <CTowerClimb*> calloc(num_envs, sizeof(CTowerClimb))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)
        cdef int i
        for i in range(num_envs):
            self.envs[i] = CTowerClimb(
                observations=&observations[i, 0],
                actions=&actions[i],
                rewards=&rewards[i],
                dones=&terminals[i],
                log_buffer=self.logs,
                reward_climb_row=reward_climb_row,
                reward_fall_row=reward_fall_row,
                reward_illegal_move=reward_illegal_move,
                reward_move_block=reward_move_block,
            )
            init(&self.envs[i])
            self.client = NULL

        cdef int goal_height
        cdef int max_moves
        for i in range(num_maps):
            goal_height = np.random.randint(5,9)
            min_moves = 10
            max_moves = 15
            init_level(&self.levels[i])
            init_puzzle_state(&self.puzzle_states[i])
            cy_init_random_level(&self.levels[i], goal_height, max_moves, min_moves, i)
            levelToPuzzleState(&self.levels[i], &self.puzzle_states[i])
            if (i + 1 ) % 50 == 0:
                print(f"Created {i+1} maps..")


    def reset(self):
        cdef int i, idx
        for i in range(self.num_envs):
            idx = np.random.randint(0, self.num_maps)
            c_reset(&self.envs[i])
            setPuzzle(&self.envs[i], &self.puzzle_states[idx], &self.levels[idx])

    def step(self):
        cdef int i, idx, done
        for i in range(self.num_envs):
            done = c_step(&self.envs[i])
            if (done):
                idx = np.random.randint(0, self.num_maps) 
                c_reset(&self.envs[i])
                setPuzzle(&self.envs[i], &self.puzzle_states[idx], &self.levels[idx])

    def render(self):
        cdef CTowerClimb* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)
            self.client.enable_animations = 1
        cdef int isMoving
        while True:
            c_render(self.client, &self.envs[0])
            isMoving = self.client.isMoving
            if not isMoving:
                break
    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
