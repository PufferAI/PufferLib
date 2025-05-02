from libc.stdlib cimport calloc, free
import numpy as np
cdef extern from "gpudrive.h":
    int LOG_BUFFER_SIZE

    ctypedef struct Log:
        float episode_return;
        float episode_length;
        float score;
        float offroad_rate;
        float collision_rate;
        float dnf_rate;
    ctypedef struct LogBuffer
    LogBuffer* allocate_logbuffer(int)
    void free_logbuffer(LogBuffer*)
    Log aggregate_and_clear(LogBuffer*)

    ctypedef struct Entity:
        int type;
        int road_object_id;
        int road_point_id;
        int array_size;
        float* traj_x;
        float* traj_y;
        float* traj_z;
        float* traj_vx;
        float* traj_vy;
        float* traj_vz;
        float* traj_heading;
        int* traj_valid;
        float width;
        float length;
        float height;
        float goal_position_x;
        float goal_position_y;
        float goal_position_z;
        int mark_as_expert;
        int collision_state;
        float x;
        float y;
        float z;
        float vx;
        float vy;
        float vz;
        float heading;
        int valid;

    ctypedef struct GPUDrive:
        float* observations;
        int* actions;
        float* rewards;
        unsigned char* masks;
        unsigned char* dones;
        LogBuffer* log_buffer;
        Log* logs;
        int num_agents;
        int active_agent_count;
        int* active_agent_indices;
        int human_agent_idx;
        Entity* entities;
        int num_entities;
        int num_cars;
        int num_objects;
        int num_roads;
        int static_car_count;
        int* static_car_indices;
        int expert_static_car_count;
        int* expert_static_car_indices;
        int timestep;
        int dynamics_model;
        float* fake_data;
        char* goal_reached;
        float* map_corners;
        int* grid_cells; 
        int grid_cols;
        int grid_rows;
        int vision_range;
        int* neighbor_offsets;
        int* neighbor_cache_entities;
        int* neighbor_cache_indices;
        float reward_vehicle_collision;
        float reward_offroad_collision;
        char* map_name;
        char* reached_goal_this_turn;

    ctypedef struct Client

    void init(GPUDrive* env)
    void free_allocated(GPUDrive* env)
    void free_entity(Entity* entity)

    Client* make_client(GPUDrive* env)
    void close_client(Client* client)
    void c_render(Client* client, GPUDrive* env)
    void c_reset(GPUDrive* env)
    void c_step(GPUDrive* env)
    Entity* load_map_binary(char* name, GPUDrive* env)
    void set_active_agents(GPUDrive *env)

cpdef entity_dtype():
    '''Make a dummy entity to get the dtype'''
    # Create a numpy structured dtype that matches the Entity struct
    return np.dtype([
        ('type', np.int32),
        ('road_object_id', np.int32),
        ('road_point_id', np.int32),
        ('array_size', np.int32),
        # For pointer fields, we use intp (integer large enough to hold a pointer)
        ('traj_x', np.intp),
        ('traj_y', np.intp),
        ('traj_z', np.intp),
        ('traj_vx', np.intp),
        ('traj_vy', np.intp),
        ('traj_vz', np.intp),
        ('traj_heading', np.intp),
        ('traj_valid', np.intp),
        ('width', np.float32),
        ('length', np.float32),
        ('height', np.float32),
        ('goal_position_x', np.float32),
        ('goal_position_y', np.float32),
        ('goal_position_z', np.float32),
        ('collision_state', np.int32),
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('vx', np.float32),
        ('vy', np.float32),
        ('vz', np.float32),
        ('heading', np.float32),
        ('valid', np.int32)
    ])

cdef class CyGPUDrive:
    cdef:
        GPUDrive* envs
        Client* client
        LogBuffer* logs
        int num_envs
        int* agent_offsets
        int agent_count

    @staticmethod
    def get_total_agent_count(int num_envs, int human_agent_idx, float reward_vehicle_collision, float reward_offroad_collision):
        """Static method to count total agents across all environments"""
        cdef int* agent_offsets = <int*> calloc(num_envs + 1, sizeof(int))
        cdef int total_count = 0
        cdef GPUDrive* temp_envs = <GPUDrive*> calloc(num_envs, sizeof(GPUDrive))
        cdef int i
        for i in range(num_envs):
            temp_envs[i].human_agent_idx = human_agent_idx
            temp_envs[i].reward_vehicle_collision = reward_vehicle_collision
            temp_envs[i].reward_offroad_collision = reward_offroad_collision
                
            map_file = f"resources/gpudrive/binaries/map_{i:03d}.bin".encode('utf-8')
            temp_envs[i].entities = load_map_binary(map_file, &temp_envs[i])
            set_active_agents(&temp_envs[i])
                
            agent_offsets[i] = total_count
            total_count += temp_envs[i].active_agent_count
            if (temp_envs[i].active_agent_count ==0 ):
                print("No active agents: ", map_file)
            
        agent_offsets[num_envs] = total_count
        py_offsets = [agent_offsets[i] for i in range(num_envs + 1)]
        for i in range(num_envs):
            for x in range(temp_envs[i].num_entities):
                free_entity(&temp_envs[i].entities[x])
            free(temp_envs[i].entities)
            free(temp_envs[i].active_agent_indices)
            free(temp_envs[i].static_car_indices)
        free(temp_envs)
        free(agent_offsets)
        return total_count, py_offsets  
    def __init__(self, float[:, :] observations, int[:,:] actions,
            float[:] rewards, unsigned char[:] masks, unsigned char[:] terminals, int num_envs,
            int human_agent_idx, reward_vehicle_collision, 
            reward_offroad_collision, offsets):

        self.client = NULL
        self.num_envs = num_envs
        self.envs = <GPUDrive*> calloc(num_envs, sizeof(GPUDrive))
        self.agent_offsets = <int*> calloc(num_envs + 1, sizeof(int))
        self.logs = allocate_logbuffer(LOG_BUFFER_SIZE)
        cdef int i
        for i in range(num_envs + 1):
            self.agent_offsets[i] = offsets[i]
        cdef int inc
        for i in range(num_envs):
            inc = self.agent_offsets[i]
            print(inc)
            map_file = f"resources/gpudrive/binaries/map_{i:03d}.bin".encode('utf-8')
            print("cython map_name", map_file)
            self.envs[i] = GPUDrive(
                observations=&observations[inc, 0],
                actions=&actions[inc,0],
                rewards=&rewards[inc],
                masks=&masks[inc],
                dones=&terminals[inc],
                log_buffer=self.logs,
                human_agent_idx=human_agent_idx,
                reward_vehicle_collision=reward_vehicle_collision,
                reward_offroad_collision=reward_offroad_collision,
                map_name = map_file
            )
            print("init")
            init(&self.envs[i])
            self.client = NULL


    def reset(self):
        cdef int i
        for i in range(self.num_envs):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.num_envs):
            c_step(&self.envs[i])

    def render(self):
        cdef GPUDrive* env = &self.envs[211]
        if self.client == NULL:
            import os
            cwd = os.getcwd()
            os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
            self.client = make_client(env)
            os.chdir(cwd)

        c_render(self.client, env)

    def close(self):
        if self.client != NULL:
            close_client(self.client)
            self.client = NULL

        free(self.envs)

    def log(self):
        cdef Log log = aggregate_and_clear(self.logs)
        return log
