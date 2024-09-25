// _ = EMPTY (not walkable, not necessarily rendered)
// . = GROUND (walkable)
// | = HOLE (not walkable, blocks view)
// # = WALL (not walkable)

unsigned int map1_height = 12;
unsigned int map1_width = 12;
char map1[] =
    "............"
    ".#|..#.#|..#"
    "............"
    "..||....||.."
    "|##..#|##..#"
    "............"
    "............"
    ".#|..#.#|..#"
    "............"
    "..||....||.."
    "|##..#|##..#"
    "............";

unsigned int map2_height = 28;
unsigned int map2_width = 29;
char map2[] = 
    "_____________#.______________"
    "___________.#..______________"
    "__________......_____________"
    "_________.|.#|...____________"
    "________........#.___________"
    "_______............._________"
    "______...............________"
    "______..............#._______"
    "_____..................______"
    "___....................._____"
    "__.|.....................____"
    "_..|......................___"
    ".................|#|.......__"
    "_#..............||||.......__"
    "__..............#|#|........."
    "___...............#........._"
    "____.......................__"
    "_____.....................___"
    "______...................____"
    "_______................._____"
    "________...............______"
    "_________............._______"
    "__________...........________"
    "___________........._________"
    "____________.......__________"
    "_____________.....___________"
    "______________|.#____________"
    "_______________._____________";

char* get_map(int map_id) {
    switch (map_id) {
        case 1: return map1; break;
        case 2: return map2; break;
        default: printf("Invalid map id <%i>\n", map_id); exit(1);
    }
}

unsigned int get_map_height(int map_id) {
    switch (map_id) {
        case 1: return map1_height; break;
        case 2: return map2_height; break;
        default: printf("Invalid map id <%i>\n", map_id); exit(1);
    }
}

unsigned int get_map_width(int map_id) {
    switch (map_id) {
        case 1: return map1_width; break;
        case 2: return map2_width; break;
        default: printf("Invalid map id <%i>\n", map_id); exit(1);
    }
}
