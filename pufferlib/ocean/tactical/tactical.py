import numpy as np
import gymnasium
import os
#from raylib import rl
#import heapq

import pufferlib
from pufferlib.ocean.tactical import binding
# from pufferlib.environments.ocean import render

EMPTY = 0
GROUND = 1
HOLE = 2
WALL = 3

MAP_DICT = {
    '_': EMPTY,
    '.': GROUND,
    '|': HOLE,
    '#': WALL,
}


class Tactical:
    def __init__(self, num_envs=200, render_mode='human', seed=0):
        self.num_envs = num_envs
        self.render_mode = render_mode

        # env spec (TODO)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(10,), dtype=np.uint8)
        self.action_space = gymnasium.spaces.Discrete(4)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = self.num_envs
        self.render_mode = render_mode
        self.emulated = None
        self.done = False
        self.buf = pufferlib.namespace(
            observations = np.zeros(
                (num_envs, 10), dtype=np.uint8),
            rewards = np.zeros(num_envs, dtype=np.float32),
            terminals = np.zeros(num_envs, dtype=bool),
            truncations = np.zeros(num_envs, dtype=bool),
            masks = np.ones(num_envs, dtype=bool),
        )
        self.actions = np.zeros(num_envs, dtype=np.uint32)
        
        self.c_envs = binding.vec_init(
            self.buf.observations,
            self.actions,
            self.buf.rewards,
            self.buf.terminals,
            self.buf.truncations,
            num_envs,
            seed,
            num_obs=self.buf.observations.shape[1]
        )

        # render
        # if render_mode == 'human':
        #     self.client = RaylibClient()

        # map_path = 'pufferlib/environments/ocean/tactical/map.txt'
        # map_path = 'pufferlib/environments/ocean/tactical/map_test.txt'
        # print(map_path)
        # self.load_map(map_path)
    
    def load_map(self, filename):
        with open(filename, 'r') as f:
            self.map_str = [line.strip() for line in f.read().strip().split('\n') if line[0] != ';']
        self.map_width = len(self.map_str[0])
        self.map_height = len(self.map_str)
        self.map = np.zeros((self.map_height, self.map_width), dtype=np.uint8)
        for i, row in enumerate(self.map_str):
            for j, cell in enumerate(row):
                self.map[i, j] = MAP_DICT[cell]

    def reset(self, seed=None):
        self.c_envs = []
        for i in range(self.num_envs):
            self.c_envs.append(binding.vec_init(
                self.buf.observations[i],
                self.actions[i:i+1],
                self.buf.rewards[i:i+1]))
            binding.vec_reset(self.c_envs[i])

        return self.buf.observations, {}

    def step(self, actions):
        self.actions[:] = actions
        for c_env in self.c_envs:
            binding.vec_step(c_env)
        
        info = {}

        return (self.buf.observations, self.buf.rewards,
            self.buf.terminals, self.buf.truncations, info)

    def render(self):
        return binding.vec_render(self.c_envs, 0)
        # if self.render_mode == 'human':
        #     return self.client.render(self.map)

    def close(self):
        for c_env in self.c_envs:
            binding.vec_close(c_env)

'''
def a_star_search(map, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))

    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while len(frontier) > 0:
        current = heapq.heappop(frontier)[1]
        
        if current == goal:
            break
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next = (current[0] + dx, current[1] + dy)
            if next[0] < 0 or next[1] < 0 or next[0] >= map.shape[0] or next[1] >= map.shape[1] or map[next] != GROUND:
                continue
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + abs(next[0] - goal[0]) + abs(next[1] - goal[1])
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current
    
    # return came_from, cost_so_far
    # reconstruct path
    path = []
    if goal not in came_from:  # no path was found
        return []
    assert current == goal
    while current != start:
        path.append(current)
        current = came_from[current]
    # path.append(start)
    path.reverse()
    return path


class RaylibClient:
    def __init__(self):
        self.screenw = 1200
        self.screenh = 900
        rl.InitWindow(self.screenw, self.screenh, "Puffer Tactical".encode())
        rl.SetTargetFPS(60)

        self.row = 12
        self.col = 12

        self.anim = False
        self.anim_type = None
        self.anim_path = None
        self.anim_path_progress = None

        self.spell_mode = False

        self.cra_bottom = rl.LoadTexture('pufferlib/environments/ocean/tactical/sacri_bottom.png'.encode())
        self.cra_top = rl.LoadTexture('pufferlib/environments/ocean/tactical/sacri_top.png'.encode())
        self.cra_left = rl.LoadTexture('pufferlib/environments/ocean/tactical/sacri_left.png'.encode())
        self.cra_right = rl.LoadTexture('pufferlib/environments/ocean/tactical/sacri_right.png'.encode())
        self.cra_tex = self.cra_bottom

    def render(self, map):
        # TODO : rather than compute isometric coordinates
        # could be easier to do all cartesian and use a coordinate conversion (linear algebra, some cos/sin)
        # to go back and forth between the two coordinate systems?
        # see https://en.wikipedia.org/wiki/Isometric_projection
        if rl.IsKeyDown(rl.KEY_ESCAPE):
            exit(0)

        if rl.IsKeyDown(rl.KEY_E) and not self.anim:
            self.spell_mode = True
        if rl.IsKeyDown(rl.KEY_R) and not self.anim:
            self.spell_mode = False

        nrows, ncols = map.shape

        # figure out dimensions so the map scales to fit on the screen

        # map width = 14, map height = 16
        # find map width (longest bottomleft-topright diagonal)

        mapw = -1
        for i in range(nrows):
            horizontal_line = [map[i-k,k] for k in range(min(i + 1, ncols))]
            if set(horizontal_line) == {EMPTY}: continue
            i0, i1 = 0, len(horizontal_line) - 1
            while horizontal_line[i0] == EMPTY: i0 += 1
            while horizontal_line[i1] == EMPTY: i1 -= 1
            mapw = max(mapw, i1 - i0 + 1)
        maph = -1
        for i in range(ncols):
            vertical_line = [map[k,i+k] for k in range(min(ncols - i, nrows))]
            if set(vertical_line) == {EMPTY}: continue
            i0, i1 = 0, len(vertical_line) - 1
            while vertical_line[i0] == EMPTY: i0 += 1
            while vertical_line[i1] == EMPTY: i1 -= 1
            maph = max(maph, i1 - i0 + 1)


        padding_top = 100
        padding_bottom = 100
        cw_max = (self.screenw) / mapw
        ch_max = (self.screenh - padding_top - padding_bottom) / maph
        # we want ch = cw / 2 -> pick the best ratio
        if ch_max > cw_max / 2:
            cw = cw_max
            ch = cw / 2
        else:
            ch = ch_max
            cw = ch * 2

        # figure out correct offset to center the game
        xmin = 1e9
        ymin = 1e9
        for i, row in enumerate(map):
            for j, cell in enumerate(row):
                # todo not the most efficient + avoid code repetition
                if cell != EMPTY:
                    xa = 0.5 * (j + 1) * cw - 0.5 * (i + 1) * cw
                    ya = 0.5 * (j + 1) * ch + 0.5 * (i + 1) * ch
                    xmin = min(xmin, xa - cw / 2)
                    ymin = min(ymin, ya)

        # import sys; sys.exit(0)

        offset_x = -xmin + (self.screenw-cw*mapw)/2  # center
        offset_y = -ymin + padding_top
        # cw = 80
        # ch = cw / 2

        rl.BeginDrawing()
        rl.ClearBackground(render.PUFF_BACKGROUND)

        # get mouse pos
        mx, my = rl.GetMouseX(), rl.GetMouseY()
        rl.DrawText(f"Mouse: {mx}, {my}".encode(), 15, 10, 20, render.PUFF_TEXT)
        # get corresponding cell (if any)
        # to get the formula: we know that cell (row, col) = (i, j) starts at coordinates
        #   x = offset_x + 0.5 * (j + 1) * cw - 0.5 * (i + 1) * cw
        #   y = offset_y + 0.5 * (j + 1) * ch + 0.5 * (i + 1) * ch
        # Solve this to write i and j as a function of x and y and we get the formulas below
        ci = int((offset_x - mx) / cw + (my - offset_y) / ch - 1)
        cj = int((mx - offset_x) / cw + (my - offset_y) / ch - 1)
        cell = None if ci < 0 or cj < 0 or ci >= nrows or cj >= ncols else (ci, cj)
        rl.DrawText(f"Cell: {cell}".encode(), 15, 35, 20, render.PUFF_TEXT)


        # movement
        movement = np.zeros_like(map)

        if not self.anim and not self.spell_mode:
            if cell is not None:
                # draw movement path
                path = a_star_search(map, start=(self.row, self.col), goal=(ci, cj))
                if path:
                    path_rows, path_cols = zip(*path)
                    movement[path_rows, path_cols] = 1

                    if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT):
                        if cell is not None and map[cell] == GROUND:
                            # self.row = ci
                            # self.col = cj
                            self.anim = True
                            self.anim_type = 'move'
                            self.anim_path = [(self.row, self.col)] + path
                            self.anim_path_progress = 0

        # line of sight
        los = np.ones_like(map)

        for i in range(nrows):
            for j in range(ncols):
                cell = map[i, j]
                if cell != GROUND:
                    los[i, j] = 0
                elif (i, j) == (self.row, self.col):
                    los[i, j] = 0
                else:
                    # use bresenham-based supercover line algorithm
                    # http://eugen.dedu.free.fr/projects/bresenham/
                    # note: bresenham alone doesnt find all cells covered by the lines
                    # implementation from https://www.redblobgames.com/grids/line-drawing/#supercover (covers all quadrants) <- here it is explained very well, the algo is pretty simple
                    # now we could precompute this on the map for every pair of points
                    # the question is: if we add one obstacle, how does it change lines of sight? mb its fast enough to just simulate in real time? 
                    # ONE OTHER APPROACH: for every pair of points, assume one point is the observer and the other is a wall (so, ignoring the geometry of the map). then, what lines of sight do we have? then we just need to do a logical and for all lines of sight. not sure its even faster though, it doesnt seem to be. 
                    # an optimization: instead of doing lines of sight for all pair of points, we could check between observer and all border cells of the map? then, we set all cells to line of sight true and as soon as we hit an obstacle, we'll set all subsequent cells to line of sight false. this should hit all the cells?
                    # bressenham: check all points between character and (i, j), if any is an obstacle then cancel the line of sight
                    x0 = self.col
                    y0 = self.row
                    x1 = j
                    y1 = i
                    ###
                    dx = x1 - x0
                    dy = y1 - y0
                    nx = abs(dx)
                    ny = abs(dy)
                    sign_x = 1 if dx > 0 else -1 
                    sign_y = 1 if dy > 0 else -1
                    px = x0
                    py = y0
                    ix = 0
                    iy = 0
                    while ix < nx or iy < ny:
                        if map[py, px] == WALL:
                            los[i, j] = 0
                            break
                        decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
                        if decision == 0:
                            # next step is diagonal
                            px += sign_x
                            py += sign_y
                            ix += 1
                            iy += 1
                        elif decision < 0:
                            # next step is horizontal
                            px += sign_x
                            ix += 1
                        else:
                            # next step is vertical
                            py += sign_y
                            iy += 1



    # bool IsMouseButtonPressed(int button);   


        # naive (O(n^3)) for each pair of cell A, B
        # we draw the line from the center of cell A to the center of cell B
        # then we use bressenham's algo https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
        # to find all the cells that the line goes through
        # if any of these is an obstacle, then there is no line of sight between A and B. Otherwise there is.

        # maybe better: for each obstacle, directly find all the cells this obstacle hides and mask them
        



        # draw cells from top-left to bottom-right
        #  isometric cell               link to bottom   link to top
        #    (ground)                                         4    
        #       a                             a           5   a   6
        #   b   e   c  (b<->c = cw)       b   0   c       b   7   c
        #       d                         1   d   2           d    
        #     (a<->d = ch)                    3                    
        # cell dimensions (as per drawing above)
        for i, row in enumerate(map):
            for j, cell in enumerate(row):
                # compute isometrics coordinates (points a,b,c,d) -- TODO of course all this should be precomputed
                xa = offset_x + 0.5 * (j + 1) * cw - 0.5 * (i + 1) * cw
                xb, xc, xd = xa - cw / 2, xa + cw / 2, xa
                ya = offset_y + 0.5 * (j + 1) * ch + 0.5 * (i + 1) * ch
                yb, yc, yd = ya + ch / 2, ya + ch / 2, ya + ch
                xe, ye = xa, yb
                # draw cell
                if cell == WALL:
                    dy = ch * 0.4
                    x4, x5, x6, x7 = xa, xb, xc, xd
                    y4, y5, y6, y7 = ya - dy, yb - dy, yc - dy, yd - dy
                    rl.DrawTriangleStrip([(x4, y4), (x5, y5), (x6, y6), (x7, y7)], 4, [163, 197, 69, 255])  # top square
                    rl.DrawTriangleStrip([(xc, yc), (x6, y6), (xd, yd), (x7, y7), (xb, yb), (x5, y5)], 6, [40, 20, 5, 255])  # connection with ground
                elif cell == HOLE:
                    pass  # leave empty, as a hole should be
                elif cell == GROUND:
                    if movement[(i, j)]:
                        col = [0, 180, 0, 255]
                    elif self.spell_mode:
                        #elif abs(i - self.row) + abs(j - self.col) <= 10 and abs(i - self.row) + abs(j - self.col) > 0:
                        if los[(i, j)]:
                            if (i, j) == (ci, cj):
                                col = [255, 165, 0, 255]                            
                            else:
                                col = [68, 109, 153, 255]
                        else:
                            col = [112, 123, 111, 255]
                    else:
                        col = [189, 205, 125, 255] if (i + j) % 2 == 0 else [180, 195, 118, 255]
                    rl.DrawTriangleStrip([(xa, ya), (xb, yb), (xc, yc), (xd, yd)], 4, col)

                    # draw white border around cell
                    rl.DrawLineStrip([(xa, ya), (xb, yb), (xd, yd), (xc, yc), (xa, ya)], 5, (255, 255, 255, 255))
                # Draw dirt below the cell
                if cell == GROUND or cell == WALL:
                    # here we only draw what will be seen ; maybe it's faster to draw everything and not do any checks
                    dy = ch * 0.7
                    x0, x1, x2, x3 = xa, xb, xc, xd
                    y0, y1, y2, y3 = ya + dy, yb + dy, yc + dy, yd + dy
                    if i == len(map) - 1 or map[i+1,j] in [HOLE, EMPTY]:
                        rl.DrawTriangleStrip([(xb, yb), (x1, y1), (xd, yd), (x3, y3)], 4, [68, 48, 10, 255])  # left side (b-1-3-d boundary)
                    if j == len(row) - 1 or map[i,j+1] in [HOLE, EMPTY]:
                        rl.DrawTriangleStrip([(xd, yd), (x3, y3), (xc, yc), (x2, y2)], 4, [95, 77, 21, 255])  # right side (d-3-2-c boundary)



        # draw character

        xe = offset_x + 0.5 * (self.col + 1) * cw - 0.5 * (self.row + 1) * cw
        ye = offset_y + 0.5 * (self.col + 1) * ch + 0.5 * (self.row + 1) * ch + ch / 2

        xe_m = offset_x + 0.5 * (cj + 1) * cw - 0.5 * (ci + 1) * cw
        ye_m = offset_y + 0.5 * (cj + 1) * ch + 0.5 * (ci + 1) * ch + ch / 2

        # 465*1129
        cra_tex_w = 465
        cra_tex_h = 1129
        cra_tex_desired_h = 1.6 * ch
        scale = cra_tex_desired_h / cra_tex_h
        cra_tex_desired_w = cra_tex_w * scale
        cra_x = xe - cra_tex_desired_w / 2
        cra_y = ye - cra_tex_desired_h + 0.1 * ch

        if self.anim and self.anim_type == "move":
            # cur is updated when we arrive at the center of a new cell
            cur = self.anim_path[int(self.anim_path_progress)]
            self.row, self.col = cur
            transition_progress = self.anim_path_progress - int(self.anim_path_progress)
            if cur == self.anim_path[-1]:
                self.anim = False
            else:
                next = self.anim_path[int(self.anim_path_progress)+1]
                # use correct facing of the texture
                if next[0] == cur[0] + 1:
                    self.cra_tex = self.cra_bottom
                    self.movx, self.movy = -1, 1
                elif next[0] == cur[0] - 1:
                    self.cra_tex = self.cra_top
                    self.movx, self.movy = 1, -1
                elif next[1] == cur[1] + 1:
                    self.cra_tex = self.cra_right
                    self.movx, self.movy = 1, 1
                elif next[1] == cur[1] - 1:
                    self.cra_tex = self.cra_left
                    self.movx, self.movy = -1, -1
            # add a delta to the x,y texture position for continuous movement
            delta_x = (transition_progress) * cw * 0.5 * self.movx
            delta_y = (transition_progress) * ch * 0.5 * self.movy
            self.anim_path_progress += 0.1
            cur = self.anim_path[int(self.anim_path_progress)]
            self.row, self.col = cur
        else:
            delta_x = delta_y = 0

        coef = 0.35
        thickness = 2
        if self.anim and self.anim_type == 'move':
            col = [189, 205, 125, 255] if (self.anim_path[0][0] + self.anim_path[0][1]) % 2 == 0 else [180, 195, 118, 255]
        else:
            col = [189, 205, 125, 255] if (self.row + self.col) % 2 == 0 else [180, 195, 118, 255]
        rl.DrawEllipse(int(xe + delta_x), int(ye + delta_y), cw * coef, ch * coef, [255, 0, 0, 255])
        rl.DrawEllipse(int(xe + delta_x), int(ye + delta_y), cw * coef - thickness, ch * coef - thickness, col)

        rl.DrawTextureEx(self.cra_tex, (cra_x + delta_x, cra_y + delta_y), 0, scale, [255, 255, 255, 255])

        # void DrawSplineLinear(Vector2 *points, int pointCount, float thick, Color color);                  // Draw spline: Linear, minimum 2 points
        # rl.DrawSplineLinear([(xe, ye), (mx, my)], 10, 5, [255, 0, 0, 255])
        # rl.DrawSplineBezierQuadratic([(xe, ye-cra_tex_desired_h/2), ((xe+mx)/2,(ye+my)/2-200), (mx, my)], 3, 5, [255, 0, 0, 255])

        if rl.IsMouseButtonPressed(rl.MOUSE_BUTTON_LEFT) and self.spell_mode and los[ci,cj]:
            self.anim = True
            self.anim_type = "spell"
            self.spell_mode = False

            self.anim_path = [(xe, ye-cra_tex_desired_h/2), ((xe+mx)/2,(ye+my)/2-200), (xe_m, ye_m)]
            self.anim_path_progress = 0.01

        if self.anim and self.anim_type == "spell":
            self.anim_path_progress += 0.025
            pt = rl.GetSplinePointBezierQuad(*self.anim_path, min(self.anim_path_progress, 1.0))

            if self.anim_path_progress <= 1.0:
                rl.DrawCircle(int(pt.x), int(pt.y), 10, [255, 0, 0, 255])
            else:
                rl.DrawCircle(int(pt.x), int(pt.y), 10 + (self.anim_path_progress - 1.0) * 100, [255, 0, 0, int(255 - (self.anim_path_progress - 1.0) * 1200)])

            if self.anim_path_progress >= 1.2:
                self.anim = False

        rl.EndDrawing()
        return render.cdata_to_numpy()
'''


if __name__ == '__main__':
    PROFILE = False
    env = Tactical(num_envs=1, render_mode='human')
    env.reset()
    import time
    t0 = time.time()
    steps = 0
    while not PROFILE or time.time() - t0 < 10:
        env.step([0] * env.num_envs)
        if not PROFILE:
            if env.render() == 1:  # exit code
                break
        steps += 1
    print('SPS:', 1 * steps / (time.time() - t0))
    
