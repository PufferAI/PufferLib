# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3
# cython: boundscheck=True
# cython: initializedcheck=True
# cython: wraparound=True
# cython: cdivision=True
# cython: nonecheck=True
# cython: profile=False

from libc.stdlib cimport rand, RAND_MAX
cimport numpy as cnp
import numpy as np

cdef:
    int EMPTY = 0
    int TOWER = 1
    int WALL = 2
    int AGENT_1 = 3
    int AGENT_2 = 4
    int CREEP_1 = 5
    int CREEP_2 = 6

    int PASS = 0
    int NORTH = 1
    int SOUTH = 2
    int EAST = 3
    int WEST = 4

    int TOWER_VISION = 5
    int CREEP_VISION = 5

    int ENTITY_PLAYER = 0
    int ENTITY_CREEP = 1
    int ENTITY_TOWER = 2

cdef struct Entity:
    int pid
    int team
    int type
    float health
    float max_health
    float mana
    float max_mana
    float y
    float x
    float spawn_y
    float spawn_x
    float damage
    int lane
    int waypoint
    float move_speed
    float move_modifier
    int stun_timer
    int move_timer
    int q_timer
    int w_timer
    int e_timer
    bint is_hit

cpdef entity_dtype():
    '''Make a dummy entity to get the dtype'''
    cdef Entity entity
    return np.asarray(<Entity[:1]>&entity).dtype

cdef inline float clip(float x):
    # Clip to [-1, 1]
    return max(-1, min(x, 1))

cdef inline float l2_distance(float x1, float y1, float x2, float y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

cdef class Environment:
    cdef:
        int width
        int height
        int num_agents
        int num_creeps
        int num_towers
        int vision_range
        float agent_speed
        bint discretize
        int obs_size
        int creep_idx
        int tick

        unsigned char[:, :] grid
        unsigned char[:, :, :] observations

        float[:] rewards
        int[:, :] pid_map
        Entity[:, :] player_obs
        Entity[:] entities

        float[:, :, :] waypoints 

    def __init__(self, grid, pids, cnp.ndarray entities, cnp.ndarray player_obs,
            observations, rewards, int num_agents, int num_creeps, int num_towers,
            int vision_range, float agent_speed, bint discretize):
        self.height = grid.shape[0]
        self.width = grid.shape[1]
        self.num_agents = num_agents
        self.num_creeps = num_creeps
        self.num_towers = num_towers
        self.vision_range = vision_range
        self.agent_speed = agent_speed
        self.discretize = discretize
        self.obs_size = 2*vision_range + 1
        self.creep_idx = 0

        self.grid = grid
        self.observations = observations
        self.rewards = rewards

        self.pid_map = pids
        self.entities = entities
        self.player_obs = player_obs

        self.waypoints = np.array([
            [[97, 25], [72, 25], [43, 25], [128-104, 128-96], [128-106, 128-66], [128-104, 128-33]],
            [[99, 29], [85, 46], [74, 55], [128-74, 128-55], [128-85, 128-46], [128-99, 128-29]],
            [[104, 33], [106, 66], [104, 96], [128-43, 128-25], [128-72, 128-25], [128-97, 128-25]],


            [[128-97, 128-25], [128-72, 128-25], [128-43, 128-25], [104, 96], [106, 66], [104, 33]],
            [[128-99, 128-29], [128-85, 128-46], [128-74, 128-55], [99, 29], [85, 46], [74, 55]],
            [[128-104, 128-33], [128-106, 128-66], [128-104, 128-96], [97, 25], [72, 25], [43, 25]],
        ], dtype=np.float32)

        self.spawn_tower(0, 0, 1, 43, 25)
        self.spawn_tower(1, 0, 1, 74, 55)
        self.spawn_tower(2, 0, 1, 104, 96)
        self.spawn_tower(3, 0, 2, 72, 25)
        self.spawn_tower(4, 0, 2, 85, 46)
        self.spawn_tower(5, 0, 2, 106, 66)
        self.spawn_tower(6, 0, 3, 97, 25)
        self.spawn_tower(7, 0, 3, 99, 29)
        self.spawn_tower(8, 0, 3, 104, 33)
        self.spawn_tower(9, 1, 1, 128-43, 128-25)
        self.spawn_tower(10, 1, 1, 128-74, 128-55)
        self.spawn_tower(11, 1, 1, 128-104, 128-96)
        self.spawn_tower(12, 1, 2, 128-72, 128-25)
        self.spawn_tower(13, 1, 2, 128-85, 128-46)
        self.spawn_tower(14, 1, 2, 128-106, 128-66)
        self.spawn_tower(15, 1, 3, 128-97, 128-25)
        self.spawn_tower(16, 1, 3, 128-99, 128-29)
        self.spawn_tower(17, 1, 3, 128-104, 128-33)
     
    cdef Entity* get_player_ob(self, int pid, int idx):
        return &self.player_obs[pid, idx]

    cdef Entity* get_entity(self, int pid):
        return &self.entities[pid]

    cdef Entity* get_tower(self, int idx):
        return &self.entities[idx + self.num_agents]

    cdef Entity* get_creep(self, int idx):
        return &self.entities[idx + self.num_agents + self.num_towers]

    cdef void compute_observations(self):
        cdef:
            float y
            float x
            int r
            int c
            int dx
            int dy
            int agent_idx
            int idx
            int pid
            int target_pid
            Entity* player
            Entity* target
            Entity* target_ob

        # TODO: Figure out how to zero data
        #self.player_obs[:, :] = 0

        for pid in range(self.num_agents):
            for idx in range(10):
                self.get_player_ob(pid, idx).pid = -1

            player = self.get_entity(pid)
            y = player.y
            x = player.x
            r = int(y)
            c = int(x)
            self.observations[pid, :] = self.grid[
                r-self.vision_range:r+self.vision_range+1,
                c-self.vision_range:c+self.vision_range+1
            ]

            idx = 0
            for dy in range(-self.vision_range, self.vision_range+1):
                r = int(player.y) + dy
                for dx in range(-self.vision_range, self.vision_range+1):
                    c = int(player.x) + dx
                    target_pid = self.pid_map[r, c]
                    if target_pid == -1:
                        continue

                    target = self.get_entity(target_pid)
                    # TODO: figure out how to copy all
                    target_ob = self.get_player_ob(pid, idx)
                    target_ob.pid = target_pid
                    target_ob.y = target.y
                    target_ob.x = target.x
                    target_ob.health = target.health
                    target_ob.team = target.team
                    target_ob.type = target.type

                    idx += 1
                    if idx == 10:
                        break

                if idx == 10:
                    break

    cdef void spawn_tower(self, int idx, int team, int tier, int y, int x):
        cdef Entity* tower = self.get_tower(idx)
        tower.type = ENTITY_TOWER
        tower.pid = idx + self.num_agents
        tower.team = team

        if tier == 1:
            tower.health = 1800
            tower.max_health = 1800
            tower.damage = 3
        elif tier == 2:
            tower.health = 2500
            tower.max_health = 2500
            tower.damage = 7
        elif tier == 3:
            tower.health = 2500
            tower.max_health = 2500
            tower.damage = 10
        else:
            raise ValueError('Invalid tier')

        self.move_to(tower, y, x)

    def reset(self, seed=0):
        cdef:
            Entity* player
            Entity* tower
            int pid
            int y
            int x

        self.tick = 0
        for pid in range(self.num_agents):
            player = self.get_entity(pid)
            player.pid = pid
            player.type = ENTITY_PLAYER
            player.max_health = 500
            player.max_mana = 100
            player.move_speed = self.agent_speed
            player.move_modifier = 0
            player.move_timer = 0
            player.stun_timer = 0

            if pid < self.num_agents//2:
                player.team = 0
            else:
                player.team = 1

            self.respawn(player)

        self.spawn_tower(0, 0, 1, 43, 25)
        self.spawn_tower(1, 0, 1, 74, 55)
        self.spawn_tower(2, 0, 1, 104, 96)
        self.spawn_tower(3, 0, 2, 72, 25)
        self.spawn_tower(4, 0, 2, 85, 46)
        self.spawn_tower(5, 0, 2, 106, 66)
        self.spawn_tower(6, 0, 3, 97, 25)
        self.spawn_tower(7, 0, 3, 99, 29)
        self.spawn_tower(8, 0, 3, 104, 33)
        self.spawn_tower(9, 1, 1, 128-43, 128-25)
        self.spawn_tower(10, 1, 1, 128-74, 128-55)
        self.spawn_tower(11, 1, 1, 128-104, 128-96)
        self.spawn_tower(12, 1, 2, 128-72, 128-25)
        self.spawn_tower(13, 1, 2, 128-85, 128-46)
        self.spawn_tower(14, 1, 2, 128-106, 128-66)
        self.spawn_tower(15, 1, 3, 128-97, 128-25)
        self.spawn_tower(16, 1, 3, 128-99, 128-29)
        self.spawn_tower(17, 1, 3, 128-104, 128-33)
        
        self.compute_observations()

    cdef int move_to(self, Entity* player, float dest_y, float dest_x):
        cdef:
            int disc_y = int(player.y)
            int disc_x = int(player.x)
            int disc_dest_y = int(dest_y)
            int disc_dest_x = int(dest_x)
            int agent_type

        if (self.grid[disc_dest_y, disc_dest_x] != EMPTY and
                self.pid_map[disc_dest_y, disc_dest_x] != player.pid):
            return -1

        if player.type == ENTITY_TOWER:
            agent_type = TOWER
        elif player.type == ENTITY_CREEP:
            if player.team == 0:
                agent_type = CREEP_1
            else:
                agent_type = CREEP_2
        elif player.type == ENTITY_PLAYER:
            if player.team == 0:
                agent_type = AGENT_1
            else:
                agent_type = AGENT_2

        self.grid[disc_y, disc_x] = EMPTY
        self.grid[disc_dest_y, disc_dest_x] = agent_type

        self.pid_map[disc_y, disc_x] = -1
        self.pid_map[disc_dest_y, disc_dest_x] = player.pid

        player.y = dest_y
        player.x = dest_x
        return 1

    cdef int move_near(self, Entity* entity, Entity* target):
        if self.move_to(entity, target.y+1.0, target.x) == 1:
            return 1
        elif self.move_to(entity, target.y-1.0, target.x) == 1:
            return 1
        elif self.move_to(entity, target.y, target.x+1.0) == 1:
            return 1
        elif self.move_to(entity, target.y, target.x-1.0) == 1:
            return 1
        elif self.move_to(entity, target.y+1.0, target.x+1.0) == 1:
            return 1
        elif self.move_to(entity, target.y+1.0, target.x-1.0) == 1:
            return 1
        elif self.move_to(entity, target.y-1.0, target.x+1.0) == 1:
            return 1
        elif self.move_to(entity, target.y-1.0, target.x-1.0) == 1:
            return 1
        else:
            return -1

    cdef void kill(self, Entity* entity):
        cdef:
            int x = int(entity.x)
            int y = int(entity.y)

        self.grid[y, x] = EMPTY
        self.pid_map[y, x] = -1
        entity.pid = -1

    cdef void creep_path(self, Entity* creep, float dest_y, float dest_x):
        cdef:
            float dy = creep.move_modifier*self.agent_speed*clip(dest_y - creep.y)
            float dx = creep.move_modifier*self.agent_speed*clip(dest_x - creep.x)
            float move_dest_y = creep.y + dy
            float move_dest_x = creep.x + dx
            int disc_y = int(move_dest_y)
            int disc_x = int(move_dest_x)

        if (self.grid[disc_y, disc_x] != EMPTY and
                self.pid_map[disc_y, disc_x] != creep.pid):
            dx = 2 * self.agent_speed * (rand()/(RAND_MAX + 1.0) - 0.5)
            dy = 2 * self.agent_speed * (rand()/(RAND_MAX + 1.0) - 0.5)
            move_dest_y = creep.y + dy
            move_dest_x = creep.x + dx

        self.move_to(creep, move_dest_y, move_dest_x)

    cdef int creep_target(self, Entity* creep):
        cdef:
            Entity* target
            int y = int(creep.y)
            int x = int(creep.x)
            int dy, dx, target_pid

        for dy in range(-CREEP_VISION, CREEP_VISION+1):
            for dx in range(-CREEP_VISION, CREEP_VISION+1):
                target_pid = self.pid_map[y + dy, x + dx]

                if target_pid == -1:
                    continue

                target = self.get_entity(target_pid)
                if target.team != creep.team:
                    return target_pid

        return -1

    cdef void creep_ai(self, Entity* creep):
        cdef:
            int waypoint = creep.waypoint
            int lane = creep.lane
            float dest_y = self.waypoints[lane, waypoint, 0]
            float dest_x = self.waypoints[lane, waypoint, 1]
            int target_pid
            Entity* target

        # Aggro check
        target_pid = self.creep_target(creep)
            
        if target_pid != -1:
            target = self.get_entity(target_pid)
            dest_y = target.y
            dest_x = target.x

            if l2_distance(creep.y, creep.x, dest_y, dest_x) < 2:
                self.attack(creep, target, 2)

            self.creep_path(creep, dest_y, dest_x)
        else:
            self.creep_path(creep, dest_y, dest_x)

            if l2_distance(creep.y, creep.x, dest_y, dest_x) < 2:
                creep.waypoint += 1

            if creep.waypoint > 5:
                creep.waypoint = 5

    cdef void spawn_creep(self, int idx, int lane):
        cdef:
            int pid = idx + self.num_agents + self.num_towers
            Entity* creep = self.get_entity(pid)
            int team 

        if lane < 3:
            team = 0
        else:
            team = 1

        creep.pid = pid
        creep.type = ENTITY_CREEP
        creep.health = 200
        creep.max_health = 200
        creep.team = team
        creep.lane = lane
        creep.waypoint = 0

        self.respawn(creep)

    cdef void spawn_creep_wave(self):
        cdef int lane, creep
        for lane in range(6):
            for creep in range(5):
                self.spawn_creep(self.creep_idx, lane)
                self.creep_idx = (self.creep_idx + 1) % self.num_creeps

    cdef void respawn(self, Entity* entity):
        cdef:
            bint valid_pos = False
            int spawn_y
            int spawn_x
            int min_y
            int max_y
            int min_x
            int max_x

        if entity.team == 0:
            min_y = 128-30
            max_y = 128-22
            min_x = 22
            max_x = 30
        else:
            min_y = 22
            max_y = 30
            min_x = 128-30
            max_x = 128-22

        while not valid_pos:
            spawn_y = rand() % (max_y - min_y) + min_y
            spawn_x = rand() % (max_x - min_x) + min_x
            if self.grid[spawn_y, spawn_x] == EMPTY:
                valid_pos = True
                break

        self.move_to(entity, spawn_y, spawn_x)
        entity.health = entity.max_health
        entity.mana = entity.max_mana

    cdef void attack(self, Entity* player, Entity* target, float damage):
        if target.pid == -1:
            return

        if target.team == player.team:
            return

        target.is_hit = True
        target.health -= damage
        if target.health > 0:
            return

        if target.type == ENTITY_PLAYER:
            self.respawn(target)
        elif target.type == ENTITY_TOWER or target.type == ENTITY_CREEP:
            self.kill(target)

    cdef void heal(self, Entity* player, Entity* target, float amount):
        if target.pid == -1:
            return

        if target.team != player.team:
            return

        target.health += amount
        if target.health > target.max_health:
            target.health = target.max_health

    cdef void aoe(self, Entity* player, Entity* target, int radius, float damage):
        cdef int y, x, dy, dx, target_pid

        # Must be centered on player
        y = int(target.y)
        x = int(target.x)

        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                target_pid = self.pid_map[y + dy, x + dx]
                if target_pid == -1:
                    continue

                target = self.get_entity(target_pid)
                if damage > 0:
                    self.attack(player, target, damage)
                else:
                    self.heal(player, target, -damage)

    cdef void push(self, Entity* player, Entity* target, float amount):
        cdef:
            float dx = target.x - player.x
            float dy = target.y - player.y
            float dist = l2_distance(target.x, target.y, player.x, player.y)
            int valid_move

        if dist == 0.0:
            return

        # Norm to unit vector
        dx /= dist
        dy /= dist

        while amount > 1.0:
            valid_move = self.move_to(target, target.y + dy, target.x + dx)
            amount -= 1.0

            if valid_move == -1:
                return

        if amount > 0.05:
            self.move_to(target, target.y + amount*dy, target.x + amount*dx)

    cdef void pull(self, Entity* player, Entity* target, float amount):
        self.push(target, player, -amount)

    cdef void update_status(self, Entity* entity):
        if entity.stun_timer > 0:
            entity.stun_timer -= 1

        if entity.move_timer > 0:
            entity.move_timer -= 1

        if entity.move_timer == 0:
            entity.move_modifier = 1.0

    cdef void update_cooldowns(self, Entity* entity):
        if entity.q_timer > 0:
            entity.q_timer -= 1

        if entity.w_timer > 0:
            entity.w_timer -= 1

        if entity.e_timer > 0:
            entity.e_timer -= 1

    cdef void skill_support_hook(self, Entity* player, Entity* target):
        if player.mana < 15:
            return

        self.pull(player, target, 1.0)
        player.mana -= 15

    cdef void skill_support_aoe_heal(self, Entity* player, Entity* target):
        self.aoe(player, player, 4, -200)

    cdef void skill_support_stun(self, Entity* player, Entity* target):
        if target.pid == -1:
            return

        if player.mana < 60:
            return

        self.attack(player, target, 50)
        target.stun_timer = 15

        player.mana -= 60
        player.e_timer = 50

    cdef void skill_burst_nuke(self, Entity* player, Entity* target):
        if target.pid == -1:
            return

        if player.mana < 60:
            return

        self.attack(player, target, 500)

        player.mana -= 60
        player.q_timer = 70

    cdef void skill_burst_aoe(self, Entity* player, Entity* target):
        if target.pid == -1:
            return

        if player.mana < 40:
            return

        self.aoe(player, target, 2, 200)

        player.mana -= 40
        player.w_timer = 40

    cdef void skill_burst_aoe_stun(self, Entity* player, Entity* target):
        if target.pid == -1:
            return
        
        if player.mana < 60:
            return

        self.aoe(player, target, 2, 0)

        player.mana -= 60
        player.e_timer = 80

    cdef void skill_tank_aoe_dot(self, Entity* player, Entity* target):
        if player.mana < 5:
            return

        self.aoe(player, player, 2, 20)
        player.mana -= 5

    cdef void skill_tank_self_heal(self, Entity* player, Entity* target):
        if player.mana < 30:
            return

        self.heal(player, player, 250)
        player.mana -= 30
        player.w_timer = 60

    cdef void skill_tank_engage_dot(self, Entity* player, Entity* target):
        if target.pid == -1:
            return

        if player.mana < 60:
            return

        if self.move_near(player, target) == -1:
            return

        self.attack(player, target, 50)
        player.mana -= 60
        player.e_timer = 70

    cdef void skill_carry_retreat_slow(self, Entity* player, Entity* target):
        if target.pid == -1:
            return

        if player.mana < 5:
            return

        self.push(target, player, 1.0)
        target.move_timer = 15
        target.move_modifier = 0.6

        player.mana -= 5

    cdef void skill_carry_slow_damage(self, Entity* player, Entity* target):
        if target.pid == -1:
            return
        
        if player.mana < 40:
            return

        self.attack(player, target, 100)
        target.move_timer = 10
        target.move_modifier = 0.7

        player.mana -= 40
        player.w_timer = 40

    cdef void skill_carry_aoe(self, Entity* player, Entity* target):
        if target.pid == -1:
            return

        if player.mana < 40:
            return

        self.aoe(player, target, 2, 200)
        
        player.mana -= 40
        player.e_timer = 40

    cdef void skill_assassin_aoe_minions(self, Entity* player, Entity* target):
        if target.pid == -1:
            return

        if target.type != ENTITY_CREEP:
            return

        if player.mana < 40:
            return

        # Targeted on minions, splashes to players
        self.aoe(player, target, 3, 300)

        player.mana -= 40
        player.q_timer = 40

    cdef void skill_assassin_tp_damage(self, Entity* player, Entity* target):
        if target.pid == -1:
            return

        if player.mana < 60:
            return

        if self.move_near(player, target) == -1:
            return

        self.attack(player, target, 600)

        player.mana -= 60
        player.w_timer = 60

    cdef void skill_assassin_move_buff(self, Entity* player, Entity* target):
        if player.mana < 5:
            return

        player.move_modifier = 2.0
        player.move_timer = 2
        player.mana -= 5

    def step(self, np_actions):
        cdef:
            float[:, :] actions_continuous
            unsigned int[:, :] actions_discrete
            int agent_idx
            float y
            float x
            float vel_y
            float vel_x
            int attack
            int disc_y
            int disc_x
            int disc_dest_y
            int disc_dest_x
            Entity* player
            Entity* target
            Entity* tower
            Entity* creep
            int pid
            int target_pid
            float damage
            int dy
            int dx
            bint use_q
            bint use_w
            bint use_e

        for pid in range(self.num_agents + self.num_towers + self.num_creeps):
            player = self.get_entity(pid)
            player.is_hit = False

        if self.discretize:
            actions_discrete = np_actions
        else:
            actions_continuous = np_actions

        # Creep AI
        if self.tick % 1200 == 0:
            self.spawn_creep_wave()

        for idx in range(self.num_creeps):
            creep = self.get_creep(idx)
            if creep.pid == -1:
                continue

            if creep.stun_timer > 0:
                continue

            self.update_status(creep)
            self.creep_ai(creep)

        # Tower AI
        for tower_idx in range(self.num_towers):
            tower = self.get_tower(tower_idx)
            if tower.pid == -1:
                continue

            self.aoe(tower, tower, TOWER_VISION, tower.damage)

        # Player Logic
        for pid in range(self.num_agents):
            player = self.get_entity(pid)

            if player.mana < player.max_mana:
                player.mana += 1

            if player.stun_timer > 0:
                continue

            self.update_status(player)
            self.update_cooldowns(player)

            # Attacks
            if self.discretize:
                # Convert [0, 1, 2] to [-1, 0, 1]
                vel_y = float(actions_discrete[pid, 0]) - 1.0
                vel_x = float(actions_discrete[pid, 1]) - 1.0
                attack = actions_discrete[pid, 2]
                use_q = actions_discrete[pid, 3]
                use_w = actions_discrete[pid, 4]
                use_e = actions_discrete[pid, 5]
            else:
                vel_y = actions_continuous[pid, 0]
                vel_x = actions_continuous[pid, 1]
                attack = int(actions_continuous[pid, 2])
                use_q = int(actions_continuous[pid, 3]) > 0.5
                use_w = int(actions_continuous[pid, 4]) > 0.5
                use_e = int(actions_continuous[pid, 5]) > 0.5

            # This is a copy. Have to get the real one
            target = self.get_player_ob(pid, attack)
            target = self.get_entity(target.pid)

            if player.pid == 0 or player.pid == 5:
                if use_q:
                    #self.skill_carry_retreat_slow(player, target)
                    #self.skill_support_hook(player, target)
                    self.skill_assassin_tp_damage(player, target)
                elif use_w:
                    self.skill_support_aoe_heal(player, target)
                elif use_e:
                    #self.skill_support_stun(player, target)
                    self.skill_assassin_move_buff(player, target)
                else:
                    self.attack(player, target, 5)
            elif player.pid == 1 or player.pid == 6:
                if use_q:
                    self.skill_assassin_aoe_minions(player, target)
                elif use_w:
                    self.skill_assassin_tp_damage(player, target)
                elif use_e:
                    self.skill_assassin_move_buff(player, target)
                else:
                    self.attack(player, target, 5)
            elif player.pid == 2 or player.pid == 7:
                if use_q:
                    self.skill_burst_nuke(player, target)
                elif use_w:
                    self.skill_burst_aoe(player, target)
                elif use_e:
                    self.skill_burst_aoe_stun(player, target)
                else:
                    self.attack(player, target, 5)
            elif player.pid == 3 or player.pid == 8:
                if use_q:
                    self.skill_tank_aoe_dot(player, target)
                elif use_w:
                    self.skill_tank_self_heal(player, target)
                elif use_e:
                    self.skill_tank_engage_dot(player, target)
                else:
                    self.attack(player, target, 5)
            elif player.pid == 4 or player.pid == 9:
                if use_q:
                    self.skill_carry_retreat_slow(player, target)
                elif use_w:
                    self.skill_carry_slow_damage(player, target)
                elif use_e:
                    self.skill_carry_aoe(player, target)
                else:
                    self.attack(player, target, 5)

            dest_y = player.y + player.move_modifier*self.agent_speed*vel_y
            dest_x = player.x + player.move_modifier*self.agent_speed*vel_x
            self.move_to(player, dest_y, dest_x)

        self.tick += 1
        self.compute_observations()
