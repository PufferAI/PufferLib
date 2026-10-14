#pragma once
#include "ghost_eval.cuh"

// Negative means unhandled; the trainer then dispatches its standard commands.
static int racing_trainer_command(int argc, char** argv) {
#ifdef RACING_MULTI
    if (strcmp(argv[1], "race") == 0) return racing_ghost_main(argc, argv);
#else
    if (strcmp(argv[1], "ghosts") == 0) return racing_ghost_main(argc, argv);
#endif
    return -1;
}
