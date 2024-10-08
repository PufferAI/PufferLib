cimport numpy as cnp
from libc.stdlib cimport calloc, free

cdef extern from "connect4.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float score;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct CConnect4:
        float* observations
        unsigned char* actions
        float* rewards
        unsigned char* dones
        LogBuffer* log_buffer;
        Log log;
        int game_over
        int piece_width;
        int piece_height;
        float* board_x;
        float* board_y;
        float board_states[6][7];
        int* longest_connected;
        int width;
        int height;
        int pieces_placed;

    ctypedef struct Client

    CConnect4* init_cconnect4(CConnect4* env)
    void free_cconnect4(CConnect4* env)

    Client* make_client(float width, float height)
    void close_client(Client* client)
    void render(Client* client, CConnect4* env)
    void reset(CConnect4* env)
    void step(CConnect4* env)

cdef class CyConnect4:
    cdef:
        CConnect4* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, cnp.ndarray observations, cnp.ndarray actions,
            cnp.ndarray rewards, cnp.ndarray terminals, int num_envs,
            int width, int height, int piece_width, int piece_height):

        self.num_envs = num_envs
        self.client = NULL
        self.envs = <CConnect4*> calloc(num_envs, sizeof(CConnect4))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef:
            cnp.ndarray observations_i
            cnp.ndarray actions_i
            cnp.ndarray rewards_i
            cnp.ndarray terminals_i

        cdef int i
        for i in range(num_envs):
            observations_i = observations[i:i+1]
            actions_i = actions[i:i+1]
            rewards_i = rewards[i:i+1]
            terminals_i = terminals[i:i+1]
            self.envs[i] = CConnect4(
                observations = <float*> observations_i.data,
                actions = <unsigned char*> actions_i.data,
                rewards = <float*> rewards_i.data,
                dones = <unsigned char*> terminals_i.data,
                log_buffer=self.logs,
                piece_width=piece_width,
                piece_height=piece_height,
                width=width,
                height=height,
            )
            init_cconnect4(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def render(self):
        cdef CConnect4* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env.width, env.height)

        render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        # TODO: free
        free_cconnect4(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
