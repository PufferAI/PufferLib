
import torch
import torch.utils.cpp_extension
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')

# Note: In Python/CUDA interop via PyTorch, we'll use integers directly
NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4

if __name__ == '__main__':
    # THIS IS HARDCODED IN CUDA. DO NOT CHANGE
    num_envs = 2048
    steps = 10000
    grid_size = 9
    dummy = torch.zeros(5).cuda()
    indices = torch.arange(num_envs).int()
    envs, obs, actions, rewards, terminals = _C.create_squared_environments(num_envs, grid_size)#, dummy)
    _C.reset_environments(envs, indices)

    import time
    start = time.time()
    torch.cuda.synchronize()

    for i in range(steps):
        # Get agent and goal positions from obs
        agent_pos = torch.nonzero(obs == 1, as_tuple=False)  # [N, 3] -> (env_idx, y, x)
        goal_pos = torch.nonzero(obs == 2, as_tuple=False)   # [N, 3] -> (env_idx, y, x)

        # Extract environment indices and coordinates
        agent_envs = agent_pos[:, 0]
        agent_y = agent_pos[:, 1]
        agent_x = agent_pos[:, 2]

        goal_envs = goal_pos[:, 0]
        goal_y = goal_pos[:, 1]
        goal_x = goal_pos[:, 2]

        # Since both are sorted by env index, we can assume alignment
        # But we need to map both to the same batch dimension (num_envs)
        # Create tensors to hold coords per env
        device = obs.device
        full_agent_y = torch.zeros(num_envs, dtype=torch.long, device=device)
        full_agent_x = torch.zeros(num_envs, dtype=torch.long, device=device)
        full_goal_y = torch.zeros(num_envs, dtype=torch.long, device=device)
        full_goal_x = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Scatter the detected positions into full arrays
        full_agent_y[agent_envs] = agent_y
        full_agent_x[agent_envs] = agent_x
        full_goal_y[goal_envs] = goal_y
        full_goal_x[goal_envs] = goal_x

        # Now compute desired actions
        move_y = full_goal_y - full_agent_y
        move_x = full_goal_x - full_agent_x

        # Default action is NOOP
        atns = torch.full((num_envs,), NOOP, dtype=torch.long, device=device)

        up_mask = move_y < 0
        down_mask = move_y > 0
        atns[up_mask] = UP
        atns[down_mask] = DOWN

        noop_mask = move_y == 0
        left_mask = noop_mask & (move_x < 0)
        right_mask = noop_mask & (move_x > 0)
        atns[left_mask] = LEFT
        atns[right_mask] = RIGHT

        # Assign actions
        actions[:] = atns

        # Step environment
        _C.step_environments(envs, indices)

    torch.cuda.synchronize()
    end = time.time()

    logs = _C.log_environments(envs, indices)
    print('perf', logs.perf)
    print('score', logs.score)
    print('episode_return', logs.episode_return)
    print('episode_length', logs.episode_length)
    print('n', logs.n)

    print('Steps/sec:', num_envs * steps / (end - start))
