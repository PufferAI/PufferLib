#include <stdlib.h>
#include <stdbool.h>

typedef struct Position {
    int x;
    int y;
} Position;


#define DOWN (0)
#define UP (1)
#define RIGHT (2)
#define LEFT (3)

static inline int reverse_direction(int direction) {
  return direction ^ 1;
}

static inline bool pos_equal(Position a, Position b) {
    return a.x == b.x && a.y == b.y;
}

static inline int pos_distance_squared(Position pos, Position target) {
    int dx = pos.x - target.x;
    int dy = pos.y - target.y;
    return dx * dx + dy * dy;
}

static inline Position pos_move(Position pos, int direction, int distance) {
    switch (direction) {
    case UP:
      pos.y -= distance;
      break;
    case DOWN:
      pos.y += distance;
      break;
    case LEFT:
      pos.x -= distance;
      break;
    case RIGHT:
      pos.x += distance;
      break;
    }

    return pos;
}

static inline int rand_range(int min, int max) {
  if (min == max) {
    return min;
  }

  return min + (rand() % (max - min));
}
