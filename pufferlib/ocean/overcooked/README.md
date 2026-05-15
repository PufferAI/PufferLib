# Overcooked Environment

A multi-agent cooking coordination environment where agents cooperate to prepare and serve onion soup. Based on the popular Overcooked video game, this environment tests agents' ability to coordinate, divide labor, and work together efficiently.

## File Structure

```
overcooked/
├── overcooked.h           # Main entry point (init, reset, step, close)
├── overcooked_types.h     # Constants, enums, and struct definitions
├── overcooked_items.h     # Item and cooking pot management
├── overcooked_obs.h       # Observation computation
├── overcooked_logic.h     # Game logic (interaction, movement, cooking)
├── overcooked_render.h    # Rendering and texture management
├── binding.c              # Python bindings
└── overcooked.py          # Python environment wrapper
```

## Observation Space

**39-dimensional vector per agent** — *see [compute_observations](overcooked_obs.h#L81)*

### Player Features (34 dims)
- **Orientation** (4): One-hot encoding of facing direction — [overcooked_obs.h:101-103](overcooked_obs.h#L101-L103)
- **Held Object** (4): One-hot encoding (onion, plated_soup, plate, empty) — [overcooked_obs.h:105-116](overcooked_obs.h#L105-L116)
- **Proximity Features** (12): Normalized (dx, dy) to nearest — [overcooked_obs.h:118-167](overcooked_obs.h#L118-L167):
  - Onion source (ingredient box)
  - Dish source (plate box)
  - Plated soup on counter
  - Serving area
  - Empty counter
  - Pot (stove)
- **Nearest Soup Ingredients** (2): Onion/tomato counts in nearest plated soup or held soup (normalized) — [overcooked_obs.h:169-179](overcooked_obs.h#L169-L179)
- **Pot Soup Ingredients** (2): Onion/tomato counts in nearest pot (normalized) — [overcooked_obs.h:181-202](overcooked_obs.h#L181-L202)
- **Pot Existence** (1): Binary flag for reachable pot — [overcooked_obs.h:205](overcooked_obs.h#L205)
- **Pot State** (4): Binary flags (empty, full, cooking, ready) — [overcooked_obs.h:207-215](overcooked_obs.h#L207-L215)
- **Cooking Time** (1): Remaining cook time (normalized) — [overcooked_obs.h:217-223](overcooked_obs.h#L217-L223)
- **Wall Detection** (4): Binary flags for walls/obstacles (up, down, left, right) — [overcooked_obs.h:225-235](overcooked_obs.h#L225-L235)

### Spatial Features (4 dims)
- **Teammate Relative Position** (2): Normalized (dx, dy) to other agent — [overcooked_obs.h:237-248](overcooked_obs.h#L237-L248)
- **Absolute Position** (2): Normalized (x, y) coordinates — [overcooked_obs.h:250-252](overcooked_obs.h#L250-L252)

### Context (1 dim)
- **Reward** (1): Current step reward — [overcooked_obs.h:255](overcooked_obs.h#L255)

## Action Space

**6 discrete actions** — *see [c_step](overcooked.h#L77)*
- 0: No-op — [ACTION_NOOP](overcooked_types.h#L43)
- 1: Move up — [ACTION_UP](overcooked_types.h#L44)
- 2: Move down — [ACTION_DOWN](overcooked_types.h#L45)
- 3: Move left — [ACTION_LEFT](overcooked_types.h#L46)
- 4: Move right — [ACTION_RIGHT](overcooked_types.h#L47)
- 5: Interact (pick up/place items, use equipment) — [ACTION_INTERACT](overcooked_types.h#L48)

## Reward System

*See [evaluate_dish_served](overcooked_logic.h#L229) and [handle_interaction](overcooked_logic.h#L106)*

### Main Rewards
- **Correct dish served** (3 onions): +1.0 (shared), +0.0 (server bonus) — [overcooked_logic.h:237-241](overcooked_logic.h#L237-L241)
- **Wrong dish served** (incorrect recipe): +0.0 (shared) — [overcooked_logic.h:252-258](overcooked_logic.h#L252-L258)
- **Step penalty**: 0.0 — [overcooked.h:80](overcooked.h#L80)

### Intermediate Rewards
- **Pick up ingredient**: +0.05 — [overcooked_logic.h:221](overcooked_logic.h#L221)
- **Add onion to pot**: +0.15 — [overcooked_logic.h:133](overcooked_logic.h#L133)
- **Start cooking** (3 onions in pot): +0.15 — [overcooked_logic.h:145-147](overcooked_logic.h#L145-L147)
- **Plate cooked soup**: +0.20 — [overcooked_logic.h:159](overcooked_logic.h#L159)

## Recipe

The correct recipe requires **exactly 3 onions** in the soup. Agents must:
1. Pick up onions from ingredient boxes
2. Add 3 onions to a pot
3. Start cooking (interact with pot when empty-handed)
4. Wait for soup to cook (20 steps)
5. Pick up a plate from plate box
6. Plate the cooked soup (interact with pot while holding plate)
7. Deliver plated soup to serving area

## Configuration

*See [Overcooked class](overcooked.py#L14)*

```python
env = Overcooked(
    num_envs=1,                          # Number of parallel environments
    layout="cramped_room",               # Layout name (see Available Layouts)
    num_agents=2,                        # Agents per environment
    render_mode=None,                    # Set to enable rendering
    log_interval=128,                    # Steps between log aggregation
    grid_size=32,                        # Render tile size in pixels

    # Reward configuration (from config/ocean/overcooked.ini)
    reward_dish_served_whole_team=1.0,   # Shared reward for correct dish
    reward_dish_served_agent=0.0,        # Bonus for serving agent
    reward_pot_started=0.15,             # Starting correct recipe
    reward_ingredient_added=0.15,        # Adding onion to pot
    reward_ingredient_picked=0.05,       # Picking up ingredient
    reward_soup_plated=0.20,             # Plating cooked soup
    reward_wrong_dish_served=0.0,        # Serving incorrect dish
    reward_step_penalty=0.0,             # Per-step penalty
)
```

## Game Constants

- **Cooking time**: 20 steps — [COOKING_TIME](overcooked_types.h#L39)
- **Max ingredients per pot**: 3 — [MAX_INGREDIENTS](overcooked_types.h#L40)
- **Max episode steps**: 400 (default)
- **Max dynamic items**: 20 — [overcooked.h:19](overcooked.h#L19)

## Available Layouts

*See [LAYOUTS](overcooked_types.h#L244-L259)*

### cramped_room (5x5)

```
+---+---+---+---+---+
| W | C | P | C | W |   W = Wall
+---+---+---+---+---+   C = Counter
| I |   |   |   | I |   P = Pot (Stove)
+---+---+---+---+---+   I = Ingredient Box (Onions)
| C |   |   |   | C |   D = Dish/Plate Box
+---+---+---+---+---+   S = Serving Area
| C |   |   |   | C |
+---+---+---+---+---+
| W | D | C | S | W |
+---+---+---+---+---+
```
Spawns: (1,2) and (3,2)

### asymmetric_advantages (9x5)

```
+---+---+---+---+---+---+---+---+---+
| W | C | W | W | W | W | W | C | W |
+---+---+---+---+---+---+---+---+---+
| I |   | C | S | W | I | C |   | S |
+---+---+---+---+---+---+---+---+---+
| C |   |   |   | P |   |   |   | C |
+---+---+---+---+---+---+---+---+---+
| C |   |   |   | P |   |   |   | C |
+---+---+---+---+---+---+---+---+---+
| W | C | C | D | W | D | C | C | W |
+---+---+---+---+---+---+---+---+---+
```
Spawns: (1,2) and (7,2)

### forced_coordination (5x5)

```
+---+---+---+---+---+
| W | C | W | P | W |   W = Wall
+---+---+---+---+---+   C = Counter
| I |   | C |   | P |   P = Pot (Stove)
+---+---+---+---+---+   I = Ingredient Box (Onions)
| I |   | C |   | C |   D = Dish/Plate Box
+---+---+---+---+---+   S = Serving Area
| D |   | C |   | C |
+---+---+---+---+---+
| W | C | W | S | W |
+---+---+---+---+---+
```
Spawns: (1,2) and (3,2)

A challenging layout with a center wall dividing the kitchen. Agents must coordinate through limited passage points.

### coordination_ring (5x5)

```
+---+---+---+---+---+
| W | C | C | P | W |   W = Wall
+---+---+---+---+---+   C = Counter
| C |   |   |   | P |   P = Pot (Stove)
+---+---+---+---+---+   I = Ingredient Box (Onions)
| D |   | C |   | C |   D = Dish/Plate Box
+---+---+---+---+---+   S = Serving Area
| I |   |   |   | C |
+---+---+---+---+---+
| W | I | S | C | W |
+---+---+---+---+---+
```
Spawns: (1,2) and (3,2)

Ring-shaped layout with a center counter obstacle. Agents must navigate around the center to coordinate ingredient pickup and soup delivery.

### counter_circuit (8x5)

```
+---+---+---+---+---+---+---+---+
| W | C | C | P | P | C | C | W |
+---+---+---+---+---+---+---+---+
| C |   |   |   |   |   |   | C |
+---+---+---+---+---+---+---+---+
| D |   | C | C | C | C |   | S |
+---+---+---+---+---+---+---+---+
| C |   |   |   |   |   |   | C |
+---+---+---+---+---+---+---+---+
| W | C | C | I | I | C | C | W |
+---+---+---+---+---+---+---+---+
```
Spawns: (1,1) and (6,3)

Circuit-shaped layout with a center counter island. Agents must coordinate around the obstacle to efficiently transport ingredients and serve dishes. Features dual pots and dual ingredient boxes for parallel cooking.

## Logging Metrics

*See [Log struct](overcooked_types.h#L65-L78)*

| Metric | Description |
|--------|-------------|
| perf | Normalized performance (correct dishes served) |
| score | Raw score (correct dishes served) |
| episode_return | Sum of rewards over episode |
| episode_length | Number of steps in episode |
| dishes_served | Total dishes served (correct + wrong) |
| correct_dishes | Number of 3-onion dishes served |
| wrong_dishes | Number of incorrect dishes served |
| ingredients_picked | Total ingredients picked up |
| pots_started | Number of cooking sessions started |
| items_dropped | Number of items placed on counters |
| agent_collisions | Number of agent collision attempts |

## Agent Reset Mechanism

If an agent goes 512 steps without receiving a reward, it is automatically reset to its starting position with no held item. This prevents agents from getting stuck — [c_step](overcooked.h#L114-L133)

## Building

```bash
# Build the environment
python setup.py build_overcooked --inplace

# Run standalone test
python pufferlib/ocean/overcooked/overcooked.py

# Run standalone demo with specific layout
./overcooked cramped_room
./overcooked asymmetric_advantages
./overcooked forced_coordination
./overcooked coordination_ring
./overcooked counter_circuit
```
