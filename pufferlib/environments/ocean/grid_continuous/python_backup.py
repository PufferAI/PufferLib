def _compute_observations(self):
    for agent_idx in range(self.num_agents):
        y = self.agent_positions[agent_idx, 0]
        x = self.agent_positions[agent_idx, 1]
        r = int(y)
        c = int(x)
        self.buf.observations[agent_idx, :, :] = self.grid[
            r-self.vision_range:r+self.vision_range+1,
            c-self.vision_range:c+self.vision_range+1
        ]

def python_reset(self):
    # Add borders
    left = self.vision_range
    right = self.render_size - self.vision_range - 1
    self.grid[:left, :] = WALL
    self.grid[right:, :] = WALL
    self.grid[:, :left] = WALL
    self.grid[:, right:] = WALL

    # Agent spawning
    agent_idx = 0
    for spawn_idx in range(self.map_size**2):
        y = self.spawn_position_cands[spawn_idx, 0]
        x = self.spawn_position_cands[spawn_idx, 1]
        disc_y = int(self.scale * y)
        disc_x = int(self.scale * x)

        if self.grid[disc_y, disc_x] == 0:
            self.grid[disc_y, disc_x] = AGENT
            self.agent_positions[agent_idx, 0] = y
            self.agent_positions[agent_idx, 1] = x
            agent_idx += 1
            if agent_idx == self.num_agents:
                break

    self._compute_observations()

def python_step(self, actions):
    for agent_idx in range(self.num_agents):
        atn = actions[agent_idx]
        vel_y = atn[0]
        vel_x = atn[1]

        y = self.agent_positions[agent_idx, 0]
        x = self.agent_positions[agent_idx, 1]
        dest_y = y + vel_y
        dest_x = x + vel_x

        # Discretize
        disc_y = int(self.scale * y)
        disc_x = int(self.scale * x)
        disc_dest_y = int(self.scale * dest_y)
        disc_dest_x = int(self.scale * dest_x)

        if self.grid[disc_dest_y, disc_dest_x] == 0:
            self.grid[disc_y, disc_x] = EMPTY
            self.grid[disc_dest_y, disc_dest_x] = AGENT

            # Continuous position update
            self.agent_positions[agent_idx, 0] = dest_y
            self.agent_positions[agent_idx, 1] = dest_x

    self._compute_observations()


