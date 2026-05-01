#ifndef PUZZLE_TYPES_H
#define PUZZLE_TYPES_H

#define INIT_ROWS 6
#define INIT_COLS 6
#define MAX_LASERS 8

typedef enum {
    EMPTY,
    LASER,
    SENSOR
} CellType;

typedef enum {
    MIRROR_NONE,
    MIRROR_RIGHT,
    MIRROR_LEFT
} MirrorState;

typedef struct {
    CellType type;
    MirrorState mirror;
    int id;
} Cell;

#endif
