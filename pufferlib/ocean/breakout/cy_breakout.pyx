from libc.stdlib cimport calloc, free

cdef extern from "breakout.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return
        float episode_length
        float score

    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Breakout:
        float* observations
        float* actions
        float* rewards
        unsigned char* terminals
        LogBuffer* log_buffer
        Log log
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
        int continuous

    void init(Breakout* env)
    void c_reset(Breakout* env)
    void c_step(Breakout* env)
    void c_render(Breakout* env)

cdef class CyBreakout:
    cdef:
        Breakout* envs
        LogBuffer* logs
        int num_envs
        float width
        float height
        float paddle_width
        float paddle_height
        float ball_width
        float ball_height

    def __init__(self, float[:, :] observations, float[:] actions,
            float[:] rewards, unsigned char[:] terminals, int num_envs,
            int frameskip, int width, int height,
            float paddle_width, float paddle_height,
            int ball_width, int ball_height,
            int brick_width, int brick_height,
            int brick_rows, int brick_cols, int continuous):

        self.num_envs = num_envs
        self.envs = <Breakout*> calloc(num_envs, sizeof(Breakout))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)

        cdef int i
        for i in range(num_envs):
            self.envs[i] = Breakout(
                observations=&observations[i, 0],
                actions=&actions[i],
                rewards=&rewards[i],
                terminals=&terminals[i],
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
                continuous=continuous,  
            )
            init(&self.envs[i])

    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i

        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef Breakout* env = &self.envs[0]
        c_render(env)

    def close(self):
        free(self.envs)
        free(self.logs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
