#define BASE_ROAD_SPEED 1.0f/13.0f // inverse number of seconds to go from the left to the right (slowest enemy)
#define MULT_ROAD_SPEED 1.35 // Factor of increase for the road speed (approximates the ratio between min speed and max speed of lvl1 (13/2.5)^(1/5))
#define BASE_PLAYER_SPEED 1.0f/3.5f // inverse number of seconds to go from the bottom to the top of the screen for the player
#define MAX_ENEMIES_PER_LANE 3
#define NUM_LEVELS 8
#define NUM_LANES 10

const float SPEED0 =  BASE_ROAD_SPEED;
const float SPEED1 =  MULT_ROAD_SPEED*BASE_ROAD_SPEED;
const float SPEED2 =  MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED;
const float SPEED3 =  MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED;
const float SPEED4 =  MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED;
const float SPEED5 =  MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*MULT_ROAD_SPEED*BASE_ROAD_SPEED;

const float SPEED_VALUES[6] = {SPEED0, SPEED1, SPEED2, SPEED3, SPEED4, SPEED5};

const int HUMAN_HIGH_SCORE[] = {25, 20, 18, 20, 25, 18, 15, 18};

const int ENEMIES_PER_LANE[NUM_LEVELS][NUM_LANES] = {
    // Level 0
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    // Level 1
    {1, 2, 2, 3, 2, 1, 2, 3, 2, 2},
    // Level 2
    {3, 3, 1, 3, 1, 1, 3, 1, 3, 1},
    // Level 3
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    // Level 4
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    // Level 5
    {1, 2, 2, 3, 2, 1, 2, 3, 2, 2},
    // Level 6
    {3, 3, 1, 3, 1, 1, 3, 1, 3, 1},
    // Level 7
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
};

const float ENEMIES_TYPES[NUM_LEVELS][NUM_LANES] = {
    // Level 0
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    // Level 1
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
    // Level 2
    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    // Level 3
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    // Level 4
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    // Level 5
    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
    // Level 6
    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    // Level 7
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
};

const int SPEED_RANDOMIZATION[NUM_LEVELS]= {0,0,0,0,1,1,1,1};


const float ENEMIES_INITIAL_X[NUM_LEVELS][NUM_LANES][MAX_ENEMIES_PER_LANE] = {
    // Level 0
    {
        {0.0, 0.0, 0.0}, // lane 0 to 9
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
    },
    {
        // Level 1
        {0.0, 0.0, 0.0}, // lane 0 to 9
        {0.0, 0.1, 0.0},
        {0.0, 0.2, 0.0},
        {0.0, 0.1, 0.2},
        {0.0, 0.4, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.4, 0.0},
        {0.0, 0.1, 0.2},
        {0.0, 0.2, 0.0},
        {0.0, 0.1, 0.0},
    },
    {
        // Level 2
        {0.0, 0.2, 0.4}, // lane 0 to 9
        {0.0, 0.2, 0.4},
        {0.0, 0.0, 0.0},
        {0.0, 0.2, 0.4},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.2, 0.4},
        {0.0, 0.0, 0.0},
        {0.0, 0.2, 0.4},
        {0.0, 0.2, 0.4},
    },
    {
        // Level 3
        {0.0, 0.0, 0.0}, // lane 0 to 9
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
    },
        // Level 4
    {
        {0.0, 0.0, 0.0}, // lane 0 to 9
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
    },
    {
        // Level 5
        {0.0, 0.0, 0.0}, // lane 0 to 9
        {0.0, 0.1, 0.0},
        {0.0, 0.2, 0.0},
        {0.0, 0.1, 0.2},
        {0.0, 0.4, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.4, 0.0},
        {0.0, 0.1, 0.2},
        {0.0, 0.2, 0.0},
        {0.0, 0.1, 0.0},
    },
    {
        // Level 6
        {0.0, 0.2, 0.4}, // lane 0 to 9
        {0.0, 0.2, 0.4},
        {0.0, 0.0, 0.0},
        {0.0, 0.2, 0.4},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.2, 0.4},
        {0.0, 0.0, 0.0},
        {0.0, 0.2, 0.4},
        {0.0, 0.2, 0.4},
    },
    {
        // Level 7
        {0.0, 0.0, 0.0}, // lane 0 to 9
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
    }
};

const float ENEMIES_INITIAL_SPEED_IDX[NUM_LEVELS][NUM_LANES] = {
    // Level 0
    {0,1,2,3,4,4,3,2,1,0},
    // Level 1
    {0,1,3,4,5,5,4,3,1,0},
    // Level 2
    {0,1,3,4,5,5,4,3,1,0},
    // Level 3
    {5,4,2,4,5,5,4,2,4,5},
    // Level 4
    {0,1,2,3,4,4,3,2,1,0},
    // Level 5
    {0,1,3,4,5,5,4,3,1,0},
    // Level 6
    {0,1,3,4,5,5,4,3,1,0},
    // Level 7
    {5,4,2,4,5,5,4,2,4,5},
};