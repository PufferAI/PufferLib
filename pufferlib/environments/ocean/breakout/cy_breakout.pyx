from libc.stdlib cimport calloc, free

cdef extern from "breakout.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float score;

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Breakout:
        float* observations
        int* actions;
        float* rewards
        unsigned char* dones
        LogBuffer* log_buffer;
        Log log;
        int score
        float episode_return
        float paddle_x
        float paddle_y
        float ball_x
        float ball_y
        float ball_vx
        float ball_vy
        float* brick_x
        float* brick_y
        float* brick_states
        int balls_fired
        float paddle_width
        float paddle_height
        float ball_speed
        int hits
        int width
        int height
        int num_bricks
        int brick_rows
        int brick_cols
        int ball_width
        int ball_height
        int brick_width
        int brick_height
        int num_balls
        int frameskip

    ctypedef struct Client

    void init(Breakout* env)
    void free_initialized(Breakout* env)

    Client* make_client(Breakout* env)
    void close_client(Client* client)
    void render(Client* client, Breakout* env)
    void reset(Breakout* env)
    void step(Breakout* env)

cdef class CyBreakout:
    cdef:
        Breakout* envs
        Client* client
        LogBuffer* logs
        int num_envs

    def __init__(self, float[:, :] observations, int[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,  int frameskip,
            int width, int height, float paddle_width, float paddle_height,
            int ball_width, int ball_height, int brick_width, int brick_height,
            int brick_rows, int brick_cols):

        self.client = NULL
        self.num_envs = num_envs
        self.envs = <Breakout*> calloc(num_envs, sizeof(Breakout))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            self.envs[i] = Breakout(
                observations=&observations[i, 0],
                actions=&actions[i],
                rewards=&rewards[i],
                dones=&terminals[i],
                log_buffer=self.logs,
                width=width,
                height=height,
                paddle_width=paddle_width,
                paddle_height=paddle_height,
                ball_width=ball_width,
                ball_height=ball_height,
                brick_width=brick_width,
                brick_height=brick_height,
                brick_rows=brick_rows,
                brick_cols=brick_cols,
                frameskip=frameskip,
            )
            init(&self.envs[i])
            self.client = NULL

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            step(&self.envs[i])

    def render(self):
        cdef Breakout* env = &self.envs[0]
        if self.client == NULL:
            self.client = make_client(env)

        render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
