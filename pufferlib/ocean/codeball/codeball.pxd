cdef extern from "codeball.h":

    cdef struct Log:
        double episode_return_side
        double episode_return_total
        double episode_return_abs
        double episode_return_one
        int episode_length
        double winner

    cdef struct LogBuffer:
        Log* logs
        int length
        int idx

    LogBuffer* allocate_logbuffer(int size)

    void free_logbuffer(LogBuffer* buffer)

    void add_log(LogBuffer* logs, Log* log)

    Log aggregate(LogBuffer* logs)

    Log aggregate_and_clear(LogBuffer* logs)

    ctypedef struct CodeBallArena:
        float width
        float height
        float depth
        float bottom_radius
        float top_radius
        float corner_radius
        float goal_top_radius
        float goal_width
        float goal_depth
        float goal_height
        float goal_side_radius

    CodeBallArena arena

    ctypedef struct Vec3D:
        float x
        float y
        float z

    ctypedef struct Action:
        Vec3D target_velocity
        float jump_speed
        bool use_nitro

    ctypedef struct Entity:
        Vec3D position
        Vec3D velocity
        float radius
        float radius_change_speed
        float mass
        float arena_e
        bool touch
        Vec3D touch_normal
        float nitro
        Action action
        bool side

    ctypedef struct NitroPack:
        Vec3D position
        bool alive
        int respawn_ticks
        float radius

    float vec3d_length(Vec3D v)

    Vec3D vec3d_normalize(Vec3D v)

    float vec3d_dot(Vec3D a, Vec3D b)

    Vec3D vec3d_subtract(Vec3D a, Vec3D b)

    Vec3D vec3d_add(Vec3D a, Vec3D b)

    Vec3D vec3d_multiply(Vec3D v, float s)

    float clamp(float val, float min, float max)

    Vec3D vec3d_clamp(Vec3D v, float max_length)

    ctypedef struct DistanceAndNormal:
        float distance
        Vec3D normal

    DistanceAndNormal dan_to_plane(Vec3D point, Vec3D point_on_plane, Vec3D plane_normal)

    DistanceAndNormal dan_to_sphere_inner(Vec3D point, Vec3D sphere_center, float sphere_radius)

    DistanceAndNormal dan_to_sphere_outer(Vec3D point, Vec3D sphere_center, float sphere_radius)

    DistanceAndNormal dan_to_arena_quarter(Vec3D point)

    DistanceAndNormal dan_to_arena(Vec3D point)

    void collide_entities(Entity* a, Entity* b)

    Vec3D collide_with_arena(Entity* e)

    void move(Entity* e, float delta_time)

    cpdef enum BaselineType:
        DO_NOTHING
        RANDOM_ACTIONS
        RUN_TO_BALL

    cdef struct CodeBall:
        Entity ball
        int n_robots
        Entity* robots
        int n_nitros
        NitroPack* nitro_packs
        int tick
        float* actions
        float* rewards
        bool terminal
        int frame_skip
        Log log
        LogBuffer* log_buffer
        bool is_single
        BaselineType baseline
        float goal_scored_reward
        float loiter_penalty
        float ball_reward

    void allocate(CodeBall* env)

    void free_allocated(CodeBall* env)

    void reset_positions(CodeBall* env)

    void goal_scored(CodeBall* env, bool side)

    void reset(CodeBall* env)

    void update(float delta_time, CodeBall* env)

    float goal_potential(Vec3D position, CodeBallArena* arena, bool side)

    void step(CodeBall* env)

    float goal_potential(Vec3D position, CodeBallArena* arena, bool side)

    void make_observation(CodeBall* env, float* buffer)
