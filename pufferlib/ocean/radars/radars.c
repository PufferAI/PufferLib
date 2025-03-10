#include "radars.h"

#include "puffernet.h"

int main() {
  Radars env = {.max_trackers = 300, .initial_targets = 300};

  allocate(&env);
  c_reset(&env);

  Client *client = make_client(2);  // 1 small, 2 for mid, 4 for 1080p, 8 for 4k

  env.actions[0] = SEARCH;
  for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
    c_step(&env);
  }
  int last_tracker = 0;
  while (!WindowShouldClose()) {
    if (last_tracker >= env.max_trackers) {
      last_tracker = 0;
    }
    last_tracker += 1;
    env.actions[0] = last_tracker;

    if (env.observations[MAX_AZ_SLICES * MAX_EL_SLICES + env.max_trackers * FEATURES_PER_TRACKER] ==
        0) {
      for (int i = 0; i < MAX_AZ_SLICES * MAX_EL_SLICES; i++) {
        if (env.observations[i] <= MAX_AZ_SLICES * MAX_EL_SLICES * SEARCH_DWELL_TIME / 4) {
          last_tracker -= 1;
          env.actions[0] = SEARCH;
          break;
        }
      }
    }

    c_step(&env);
    c_render(client, &env);
  }
  free_allocated(&env);
  close_client(client);
}
