#pragma once

// Static gates uploaded once; timing advances only with fixed physics steps.
struct RacingGate {
    PfVec3 left, right, forward;
    float s; // Route coordinate for guidance; not a mandatory checkpoint.
};
struct RacingTrial {
    double clock, started, last, best, travel;
    unsigned long long passed[2]; // Optional guide counts, once per lap.
    int active, next, invalid, laps, checkpoints;
};

// Lap timing uses continuous signed route travel and the finite start/finish line.
__device__ static void racing_trial_step(
    RacingTrial *t, const RacingGate *gates, int count, PfVec3 before, PfVec3 after,
    float route_s, float route_delta, float length, bool route_jump) {
    double begin = t->clock;
    t->clock += 1.0 / 240.0;
    float delta = route_jump ? 0 : route_delta;
    // A discontinuity breaks this lap's traversal proof, but never resets the car.
    if (route_jump && t->active && !t->invalid) t->invalid = 2;
    if (t->active) t->travel += delta;
    // Guidance follows the route even when a gate aperture was missed.
    t->next = 0;
    for (int i = 1; i < count; i++) {
        if (gates[i].s > route_s) { t->next = i; break; }
    }
    for (int i = 0; i < count; i++) {
        RacingGate g = gates[i];
        PfVec3 span = pf_sub(g.right, g.left);
        float a = pf_dot(pf_sub(before, g.left), g.forward);
        float b = pf_dot(pf_sub(after, g.left), g.forward);
        bool forward = a < 0 && b >= 0;
        if (!forward)
            continue;
        float fraction = a / (a - b);
        PfVec3 point = pf_add(before, pf_scale(pf_sub(after, before), fraction));
        float width2 = span.x * span.x + span.z * span.z;
        float across = ((point.x - g.left.x) * span.x + (point.z - g.left.z) * span.z) / width2;
        float height = g.left.y + across * span.y;
        // Finite gate aperture, not an infinite plane crossing elsewhere on the circuit.
        if (across < 0 || across > 1 || fabsf(point.y - height) > 2.0f)
            continue;
        double crossed = begin + fraction / 240.0;
        if (i == 0) {
            // Allow small projection differences across the line's finite width.
            double travel = t->travel - (1 - fraction) * delta;
            float tolerance = fminf(length * 0.01f, sqrtf(width2) + 2);
            if (route_jump || (t->invalid && t->invalid != 2)) continue;
            bool complete = t->active && !t->invalid && travel >= length - tolerance;
            if (t->active && !complete && !t->invalid) continue;
            if (complete) {
                t->last = crossed - t->started;
                if (t->best == 0 || t->last < t->best) t->best = t->last;
                t->laps++;
                t->checkpoints++;
            }
            // A finish or a traversal-invalid lap arms a clean attempt at this line.
            t->active = 1;
            t->started = crossed;
            t->travel = (1 - fraction) * delta;
            t->passed[0] = t->passed[1] = 0;
            t->invalid = 0;
        } else if (t->active && !t->invalid) {
            unsigned long long bit = 1ull << (i % 64);
            if (!(t->passed[i / 64] & bit)) {
                t->passed[i / 64] |= bit;
                t->checkpoints++;
            }
        }
    }
}
