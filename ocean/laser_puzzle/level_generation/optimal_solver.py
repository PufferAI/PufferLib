"""
Optimal Solver time complexity:

branching factor: at most 16 * 2 = 32 initially
depth: 7
work per state: at most 16 * 4 = 64 (4 lasers)

4x4 board, max 7 mirrors
states: sum(C(16, k) * 2^k for k in 0..7) = 2,150,721

Total work = states * work per state = 2,150,721 * 64 = 137,646,144 --> severly reduced though in practice due to laser path pruning

Will implement idffs + pruning based on laser beams path. This means realistically, we should be finding the solution significantly quicker than the worst case
"""
from generate_level import laser_direction
from generate_level import walk_laser

def iddfs(grid, max_depth):
    optimal = None
    def dfs(grid, visited, cur_depth, goal_depth):
        nonlocal optimal

        if cur_depth > goal_depth:
            return

        # found the solution already, exit
        if optimal is not None:
            return
        
        ROWS, COLS = len(grid), len(grid[0])

        lasers = {(r, c, grid[r][c][1:]) for r in range(ROWS) for c in range(COLS) if grid[r][c][0] == 'L'}
        
        valid = set()
        need_sinks = len(lasers)

        for laser in lasers:
            r,c,idx = laser
            cur_dir = laser_direction((r,c), grid)

            visited_laser = set([((r,c), cur_dir)])
            for next_pos, next_dir in walk_laser(grid, (r,c)):
                if next_dir is None:
                    if grid[next_pos[0]][next_pos[1]] == f"S{idx}":
                        need_sinks -= 1
                    break

                next_state = (next_pos, next_dir)

                # see if laser forms a cycle
                if next_state in visited_laser:
                    break

                if grid[next_pos[0]][next_pos[1]] == '*':
                    valid.add(next_pos)

                visited_laser.add(next_state)

        # check if we have put the lasers all in thier place
        if need_sinks == 0:
            # store the optimal solution
            optimal = [[[grid[r][c] for c in range(len(grid[r]))] for r in range(len(grid))], cur_depth]
            return

        # we now have all the valid spots, choose each valid spot with both configurations and branch out
        for r,c in valid:
            for orientation in ['ML', 'MR']:
                grid[r][c] = orientation
                state = tuple(tuple(grid[r][c] for c in range(len(grid[r]))) for r in range(len(grid)))
                if state in visited: 
                    # backtrack
                    grid[r][c] = '*'
                    continue
                
                visited.add(state)
                dfs(grid, visited, cur_depth + 1, goal_depth)

                # backtrack
                grid[r][c] = '*'

    for depth in range(max_depth + 1):
        visited = set([tuple(tuple(grid[r][c] for c in range(len(grid[r]))) for r in range(len(grid)))])
        dfs(grid, visited, 0, depth)
        if optimal is not None:
            return optimal
    
    return [None, None]
