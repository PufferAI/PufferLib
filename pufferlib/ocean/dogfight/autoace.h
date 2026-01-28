/**
 * autoace.h - Intelligent Adversarial Opponent for Dogfight
 *
 * AutoAce uses real air combat tactics, energy management, and predictive
 * decision-making to challenge trained RL agents. Based on:
 *   - Boyd's E-M Theory (energy state determines tactical options)
 *   - BFM Doctrine (lead/lag/pure pursuit, yo-yos, scissors, break turns)
 *   - DARPA AlphaDogfight hierarchical RL concepts
 *
 * Architecture:
 *   1. Tactical State Computation (every frame)
 *   2. Engagement Classifier (OFFENSIVE/NEUTRAL/DEFENSIVE/WEAPONS/EXTEND)
 *   3. Maneuver Selector (state machine with persistence)
 *   4. Maneuver Executor (PID controllers)
 */

#ifndef AUTOACE_H
#define AUTOACE_H

#include <math.h>
#include <stdbool.h>
#include <string.h>

// Note: autoace.h requires flightlib.h and autopilot.h to be included BEFORE this file

// ============================================================================
// TACTICAL STATE
// ============================================================================
// Computed every frame to assess the engagement geometry and energy state

typedef struct TacticalState {
    // Geometry (computed from positions and orientations)
    float aspect_angle;      // 0 = behind target, 180 = head-on (degrees)
    float angle_off;         // Track crossing angle - velocity alignment (degrees)
    float antenna_train;     // Target bearing from our nose, 0 = dead ahead (degrees)
    float range;             // Distance in meters
    float closure_rate;      // Positive = closing (m/s)

    // Energy state
    float specific_energy;   // Own Es = 0.5*v^2 + g*h (m^2/s^2)
    float target_energy;     // Target Es
    float energy_delta;      // Own Es - Target Es (positive = advantage)
    float own_speed;         // Current airspeed (m/s)
    float target_speed;      // Target airspeed (m/s)

    // Derived tactical indicators
    float time_to_intercept; // range / closure_rate (seconds, 999 if not closing)
    bool in_gun_envelope;    // range < 500m && antenna_train < 5 deg
    bool target_in_front;    // antenna_train < 90 deg
    bool we_are_faster;      // own_speed > target_speed + 5 m/s
    bool closing;            // closure_rate > 0

    // Target lead point (for gun tracking)
    Vec3 lead_pos;           // Predicted target position at bullet TOF
} TacticalState;

// ============================================================================
// ENGAGEMENT STATE CLASSIFIER
// ============================================================================

typedef enum {
    ENGAGE_OFFENSIVE,    // Behind target, closing, have energy - ATTACK
    ENGAGE_NEUTRAL,      // Neither has clear advantage - MANEUVER FOR POSITION
    ENGAGE_DEFENSIVE,    // Target behind us, closing - SURVIVE
    ENGAGE_WEAPONS,      // In firing solution - TRACK AND SHOOT
    ENGAGE_EXTEND,       // Low energy, need to disengage and rebuild
} EngagementState;

// ============================================================================
// AUTOACE STATE (extends AutopilotState)
// ============================================================================
// Additional state needed for tactical decision-making

typedef struct AutoAceState {
    // Current engagement assessment
    TacticalState tactical;
    EngagementState engagement;

    // Maneuver state machine
    int mode_timer;          // Ticks remaining in current mode (for persistence)
    int maneuver_phase;      // Phase within multi-phase maneuvers (0, 1, 2...)
    float yoyo_apex_alt;     // Target altitude for high yo-yo apex

    // Scissors maneuver state
    int scissors_timer;      // Ticks until next reversal
    int scissors_direction;  // +1 or -1 (current turn direction)

    // PID state for tracking
    float prev_heading_error;
    float prev_pitch_error;
    float prev_bank_error;

    // Statistics
    int shots_fired;
    int hits;
} AutoAceState;

// ============================================================================
// CONSTANTS
// ============================================================================

#define AUTOACE_GUN_RANGE 500.0f        // Gun effective range (m)
#define AUTOACE_BULLET_SPEED 850.0f     // ~WW2 .50 cal muzzle velocity (m/s)
#define AUTOACE_GUN_CONE 5.0f           // Firing cone half-angle (degrees)
#define AUTOACE_MIN_MODE_TIME 25        // Minimum ticks per mode (~0.5s at 50Hz)
#define AUTOACE_FIRE_COOLDOWN 10        // Ticks between shots

// Energy thresholds (specific energy in m^2/s^2)
#define AUTOACE_ENERGY_LOW -5000.0f     // Significant energy deficit
#define AUTOACE_ENERGY_ADVANTAGE 3000.0f // Clear energy advantage

// Speed thresholds (m/s)
#define AUTOACE_SPEED_LOW 60.0f         // Approaching stall
#define AUTOACE_SPEED_FAST_DIFF 5.0f    // Speed difference considered significant

// Closure rate thresholds (m/s)
#define AUTOACE_CLOSURE_FAST 50.0f      // Closing too fast (overshoot risk)
#define AUTOACE_CLOSURE_SLOW -10.0f     // Falling behind

// ============================================================================
// TACTICAL STATE COMPUTATION
// ============================================================================

static inline void compute_tactical_state(Plane* self, Plane* target, TacticalState* ts) {
    // === Geometry ===
    Vec3 to_target = sub3(target->pos, self->pos);
    ts->range = norm3(to_target);

    // Avoid division by zero for very close range
    if (ts->range < 1.0f) {
        ts->range = 1.0f;
    }

    Vec3 los = normalize3(to_target);  // Line of sight

    // Aspect angle: angle between LOS (from us to target) and target's forward
    // 0 = directly behind target (LOS aligns with target's fwd), 180 = head-on
    Vec3 tgt_fwd = quat_rotate(target->ori, vec3(1, 0, 0));
    float aspect_cos = dot3(los, tgt_fwd);
    aspect_cos = clampf(aspect_cos, -1.0f, 1.0f);
    ts->aspect_angle = acosf(aspect_cos) * RAD_TO_DEG;

    // Antenna train angle: target bearing from our nose
    // 0 = dead ahead, 90 = to our side, 180 = behind us
    Vec3 self_fwd = quat_rotate(self->ori, vec3(1, 0, 0));
    float train_cos = dot3(los, self_fwd);
    train_cos = clampf(train_cos, -1.0f, 1.0f);
    ts->antenna_train = acosf(train_cos) * RAD_TO_DEG;

    // Angle-off: track crossing angle (how parallel are our velocities?)
    // 0 = same direction, 180 = opposite directions
    float self_speed = norm3(self->vel);
    float tgt_speed = norm3(target->vel);

    if (self_speed > 1.0f && tgt_speed > 1.0f) {
        Vec3 self_vel_n = normalize3(self->vel);
        Vec3 tgt_vel_n = normalize3(target->vel);
        float angle_off_cos = dot3(self_vel_n, tgt_vel_n);
        angle_off_cos = clampf(angle_off_cos, -1.0f, 1.0f);
        ts->angle_off = acosf(angle_off_cos) * RAD_TO_DEG;
    } else {
        ts->angle_off = 0.0f;
    }

    // Closure rate: positive = closing
    // This is the rate of change of range (negative range_dot = closing)
    Vec3 rel_vel = sub3(self->vel, target->vel);
    ts->closure_rate = dot3(rel_vel, los);

    // === Energy State ===
    ts->own_speed = self_speed;
    ts->target_speed = tgt_speed;

    // Specific energy: Es = 0.5*v^2 + g*h (kinetic + potential per unit mass)
    ts->specific_energy = 0.5f * ts->own_speed * ts->own_speed + GRAVITY * self->pos.z;
    ts->target_energy = 0.5f * ts->target_speed * ts->target_speed + GRAVITY * target->pos.z;
    ts->energy_delta = ts->specific_energy - ts->target_energy;

    // === Derived Indicators ===
    ts->time_to_intercept = (ts->closure_rate > 1.0f) ?
                            ts->range / ts->closure_rate : 999.0f;
    ts->in_gun_envelope = (ts->range < AUTOACE_GUN_RANGE &&
                          ts->antenna_train < AUTOACE_GUN_CONE);
    ts->target_in_front = (ts->antenna_train < 90.0f);
    ts->we_are_faster = (ts->own_speed > ts->target_speed + AUTOACE_SPEED_FAST_DIFF);
    ts->closing = (ts->closure_rate > 0.0f);

    // === Lead Point Computation ===
    // Predict where target will be when bullet arrives
    float bullet_tof = ts->range / AUTOACE_BULLET_SPEED;
    ts->lead_pos = add3(target->pos, mul3(target->vel, bullet_tof));
}

// ============================================================================
// ENGAGEMENT CLASSIFIER
// ============================================================================

static inline EngagementState classify_engagement(TacticalState* ts) {
    // WEAPONS: In gun envelope - shoot!
    if (ts->in_gun_envelope) {
        return ENGAGE_WEAPONS;
    }

    // DEFENSIVE: Target behind us (aspect > 135 from target's POV means
    // they're behind us) and closing
    // Actually, if OUR aspect < 45 means we're behind THEM
    // If THEIR aspect (to us) > 135, they're behind us
    // Easier: if antenna_train > 135 degrees, target is behind us
    if (ts->antenna_train > 135.0f && ts->closure_rate > 10.0f) {
        // Wait, antenna_train > 135 means target is behind us? No.
        // antenna_train is target bearing FROM our nose
        // If target is behind us, antenna_train > 90
        // Let's think about aspect_angle instead:
        // aspect_angle = 0 means we're behind target
        // What we need: is TARGET behind US?
        // We can compute this from the reverse perspective:
        // If target were computing aspect on us, what would it be?
        // Simplified: if our antenna_train > 120, target has angular advantage
        // OR if our aspect_angle > 120 (we're in front of target = bad)
        return ENGAGE_DEFENSIVE;
    }

    // More defensive check: target behind us means high antenna_train
    if (ts->antenna_train > 120.0f && ts->closing) {
        return ENGAGE_DEFENSIVE;
    }

    // EXTEND: We're slow and/or low on energy
    if (ts->own_speed < AUTOACE_SPEED_LOW ||
        ts->energy_delta < AUTOACE_ENERGY_LOW) {
        return ENGAGE_EXTEND;
    }

    // OFFENSIVE: Behind target (low aspect angle) with reasonable energy
    // aspect_angle < 60 means we're in the rear quarter
    if (ts->aspect_angle < 60.0f &&
        ts->energy_delta > AUTOACE_ENERGY_LOW &&
        ts->target_in_front) {
        return ENGAGE_OFFENSIVE;
    }

    // Default: NEUTRAL - need to maneuver for advantage
    return ENGAGE_NEUTRAL;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Get heading angle to a world-space point
static inline float get_heading_to_point(Plane* self, Vec3 point) {
    Vec3 to_point = sub3(point, self->pos);
    return atan2f(to_point.y, to_point.x);
}

// Get pitch angle to a world-space point
static inline float get_pitch_to_point(Plane* self, Vec3 point) {
    Vec3 to_point = sub3(point, self->pos);
    float horiz_dist = sqrtf(to_point.x * to_point.x + to_point.y * to_point.y);
    return atan2f(to_point.z, horiz_dist);
}

// Get current heading (yaw angle around Z)
static inline float get_current_heading(Plane* p) {
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    return atan2f(fwd.y, fwd.x);
}

// Get current pitch angle
static inline float get_current_pitch(Plane* p) {
    Vec3 fwd = quat_rotate(p->ori, vec3(1, 0, 0));
    return asinf(clampf(fwd.z, -1.0f, 1.0f));
}

// Get current bank angle (positive = right wing down)
static inline float get_current_bank(Plane* p) {
    Vec3 up = quat_rotate(p->ori, vec3(0, 0, 1));
    float bank = acosf(clampf(up.z, -1.0f, 1.0f));
    // Sign: positive when right wing down (up.y < 0)
    return (up.y < 0) ? bank : -bank;
}

// Normalize angle to [-PI, PI]
static inline float normalize_angle(float angle) {
    while (angle > PI) angle -= 2.0f * PI;
    while (angle < -PI) angle += 2.0f * PI;
    return angle;
}

// Compute bank angle needed to turn toward a heading
static inline float compute_bank_for_heading(Plane* self, float target_heading, float max_bank) {
    float current_heading = get_current_heading(self);
    float heading_error = normalize_angle(target_heading - current_heading);

    // Proportional bank: more error = more bank
    float bank_command = heading_error * 1.5f;  // Gain of 1.5
    return clampf(bank_command, -max_bank, max_bank);
}

// ============================================================================
// MANEUVER IMPLEMENTATIONS
// ============================================================================

// Gun tracking: lead pursuit with firing solution
static inline void execute_gun_track(AutopilotState* ap, AutoAceState* ace,
                                     Plane* self, Plane* target, float* actions) {
    // Aim at lead point
    Vec3 to_lead = sub3(ace->tactical.lead_pos, self->pos);
    float desired_heading = atan2f(to_lead.y, to_lead.x);
    float horiz_dist = sqrtf(to_lead.x * to_lead.x + to_lead.y * to_lead.y);
    float desired_pitch = atan2f(to_lead.z, horiz_dist);

    float current_heading = get_current_heading(self);
    float current_pitch = get_current_pitch(self);

    float heading_error = normalize_angle(desired_heading - current_heading);
    float pitch_error = desired_pitch - current_pitch;

    // Bank to turn toward target
    // Positive heading_error (target left) → negative target_bank (bank left) → turn left
    float target_bank = clampf(heading_error * -2.0f, -1.2f, 1.2f);  // ~70 deg max
    float current_bank = get_current_bank(self);
    float bank_error = target_bank - current_bank;

    // Aileron: roll to target bank (positive gain)
    actions[2] = clampf(bank_error * 5.0f, -1.0f, 1.0f);

    // Elevator: pitch to track
    // In a bank, we need to pull to change heading, not just pitch
    float load_factor = 1.0f / fmaxf(cosf(fabsf(current_bank)), 0.3f);
    float base_pull = -0.2f * load_factor;  // Base pull to maintain altitude in turn
    float pitch_correction = -pitch_error * 3.0f;
    actions[1] = clampf(base_pull + pitch_correction, -1.0f, 1.0f);

    // Throttle: maintain energy
    actions[0] = 0.8f * 2.0f - 1.0f;  // 80% throttle -> action space

    // Rudder: coordinate turn
    actions[3] = clampf(heading_error * 0.5f, -0.3f, 0.3f);

    // Fire when on target
    if (ace->tactical.antenna_train < 3.0f && ace->tactical.range < AUTOACE_GUN_RANGE) {
        if (self->fire_cooldown == 0) {
            actions[4] = 1.0f;  // FIRE!
            ace->shots_fired++;
        }
    } else {
        actions[4] = -1.0f;
    }
}

// Lag pursuit: nose behind target for controlled pursuit
static inline void execute_pursuit_lag(AutopilotState* ap, AutoAceState* ace,
                                       Plane* self, Plane* target, float* actions) {
    // Aim at where target WAS (lag behind)
    // Effectively aim at target position but don't lead
    Vec3 to_target = sub3(target->pos, self->pos);
    float desired_heading = atan2f(to_target.y, to_target.x);
    float horiz_dist = sqrtf(to_target.x * to_target.x + to_target.y * to_target.y);
    float desired_pitch = atan2f(to_target.z, horiz_dist);

    float current_heading = get_current_heading(self);
    float current_pitch = get_current_pitch(self);

    float heading_error = normalize_angle(desired_heading - current_heading);
    float pitch_error = desired_pitch - current_pitch;

    // Bank to turn toward target
    // Positive heading_error (target left) → negative target_bank (bank left) → turn left
    // Sign convention matches autopilot.h: negative bank = left wing down = turn left
    float target_bank = clampf(heading_error * -1.5f, -1.0f, 1.0f);  // ~60 deg max
    float current_bank = get_current_bank(self);
    float bank_error = target_bank - current_bank;

    // Positive gain: positive bank_error → positive aileron → roll right
    // This matches autopilot.h roll_kp = -5.0 (but we changed target_bank sign)
    actions[2] = clampf(bank_error * 5.0f, -1.0f, 1.0f);  // Aileron
    actions[1] = clampf(-pitch_error * 2.0f, -0.5f, 0.5f); // Elevator (gentle)
    actions[0] = 0.9f * 2.0f - 1.0f;  // High throttle to close
    actions[3] = clampf(heading_error * 0.3f, -0.2f, 0.2f);  // Rudder
    actions[4] = -1.0f;  // Don't fire in lag pursuit
}

// Lead pursuit: nose ahead of target for gun attack
static inline void execute_pursuit_lead(AutopilotState* ap, AutoAceState* ace,
                                        Plane* self, Plane* target, float* actions) {
    // Aim at lead point
    Vec3 to_lead = sub3(ace->tactical.lead_pos, self->pos);
    float desired_heading = atan2f(to_lead.y, to_lead.x);
    float horiz_dist = sqrtf(to_lead.x * to_lead.x + to_lead.y * to_lead.y);
    float desired_pitch = atan2f(to_lead.z, horiz_dist);

    float current_heading = get_current_heading(self);
    float current_pitch = get_current_pitch(self);

    float heading_error = normalize_angle(desired_heading - current_heading);
    float pitch_error = desired_pitch - current_pitch;

    // Aggressive turn toward lead point
    // Positive heading_error → negative target_bank → turn left
    float target_bank = clampf(heading_error * -2.0f, -1.2f, 1.2f);
    float current_bank = get_current_bank(self);
    float bank_error = target_bank - current_bank;

    actions[2] = clampf(bank_error * 5.0f, -1.0f, 1.0f);  // Aileron
    actions[1] = clampf(-pitch_error * 3.0f, -0.7f, 0.7f); // Elevator (aggressive)
    actions[0] = 0.7f * 2.0f - 1.0f;  // Moderate throttle (manage closure)
    actions[3] = clampf(heading_error * 0.4f, -0.3f, 0.3f);  // Rudder
    actions[4] = -1.0f;
}

// Break turn: maximum rate defensive turn away from threat
static inline void execute_break_turn(AutopilotState* ap, AutoAceState* ace,
                                      Plane* self, Plane* target, float* actions) {
    // Turn AWAY from target - determine which side target is on
    Vec3 to_target = sub3(target->pos, self->pos);
    Vec3 right = quat_rotate(self->ori, vec3(0, 1, 0));
    float dot_right = dot3(normalize3(to_target), right);

    // Turn away (opposite side from target)
    // If target is to our right (dot_right > 0), bank left (negative) to turn away
    // If target is to our left (dot_right < 0), bank right (positive) to turn away
    float target_bank = (dot_right > 0) ? -1.3f : 1.3f;  // Max bank ~75 deg

    float current_bank = get_current_bank(self);
    float bank_error = target_bank - current_bank;

    actions[2] = clampf(bank_error * 6.0f, -1.0f, 1.0f);  // Aggressive aileron
    actions[1] = -0.7f;  // Pull hard!
    actions[0] = 1.0f;   // Full throttle (max action = 1.0)
    actions[3] = 0.0f;   // No rudder in break
    actions[4] = -1.0f;
}

// High yo-yo: climb to bleed closure rate, then dive back
static inline void execute_high_yoyo(AutopilotState* ap, AutoAceState* ace,
                                     Plane* self, Plane* target, float* actions) {
    float current_bank = get_current_bank(self);

    if (ace->maneuver_phase == 0) {
        // Phase 1: Reduce bank, pull up to climb
        if (ace->yoyo_apex_alt == 0.0f) {
            // Set apex altitude 150-200m above current
            ace->yoyo_apex_alt = self->pos.z + 150.0f + rndf(0, 50);
        }

        // Shallow bank, climb - reduce current bank toward zero
        float target_bank = current_bank * 0.3f;
        float bank_error = target_bank - current_bank;

        actions[2] = clampf(bank_error * 3.0f, -1.0f, 1.0f);
        actions[1] = -0.5f;  // Pull up moderately
        actions[0] = 0.8f * 2.0f - 1.0f;
        actions[3] = 0.0f;
        actions[4] = -1.0f;

        // Transition when reaching apex
        if (self->pos.z > ace->yoyo_apex_alt) {
            ace->maneuver_phase = 1;
        }
    } else {
        // Phase 2: Roll back in, dive toward target
        Vec3 to_target = sub3(target->pos, self->pos);
        float desired_heading = atan2f(to_target.y, to_target.x);
        float current_heading = get_current_heading(self);
        float heading_error = normalize_angle(desired_heading - current_heading);

        // Positive heading_error → negative target_bank → turn left
        float target_bank = clampf(heading_error * -2.0f, -1.0f, 1.0f);
        float bank_error = target_bank - current_bank;

        actions[2] = clampf(bank_error * 5.0f, -1.0f, 1.0f);
        actions[1] = 0.3f;   // Push over slightly to dive
        actions[0] = 0.5f * 2.0f - 1.0f;  // Reduced throttle
        actions[3] = clampf(heading_error * 0.3f, -0.2f, 0.2f);
        actions[4] = -1.0f;
    }
}

// Scissors: reversing breaks to force overshoot
static inline void execute_scissors(AutopilotState* ap, AutoAceState* ace,
                                    Plane* self, Plane* target, float* actions) {
    // Initialize direction if needed
    if (ace->scissors_direction == 0) {
        ace->scissors_direction = (rndf(0, 1) > 0.5f) ? 1 : -1;
        ace->scissors_timer = 40;  // ~0.8 seconds per reversal
    }

    // Check for reversal
    ace->scissors_timer--;
    if (ace->scissors_timer <= 0) {
        ace->scissors_direction *= -1;  // Reverse!
        ace->scissors_timer = 35 + (int)rndf(0, 10);  // Vary timing
    }

    // target_bank: +1.4 = bank right = turn right, -1.4 = bank left = turn left
    float target_bank = ace->scissors_direction * 1.4f;  // ~80 deg
    float current_bank = get_current_bank(self);
    float bank_error = target_bank - current_bank;

    actions[2] = clampf(bank_error * 6.0f, -1.0f, 1.0f);  // Aggressive roll
    actions[1] = -0.5f;   // Pull through each reversal
    actions[0] = 0.3f * 2.0f - 1.0f;  // Low throttle to slow down
    actions[3] = 0.0f;
    actions[4] = -1.0f;
}

// Extend: disengage and rebuild energy
static inline void execute_extend(AutopilotState* ap, AutoAceState* ace,
                                  Plane* self, Plane* target, float* actions) {
    // Fly straight away from target
    Vec3 from_target = sub3(self->pos, target->pos);
    float away_heading = atan2f(from_target.y, from_target.x);

    float current_heading = get_current_heading(self);
    float heading_error = normalize_angle(away_heading - current_heading);

    // Gentle turn to face away
    // Positive heading_error → negative target_bank → turn left
    float target_bank = clampf(heading_error * -1.0f, -0.5f, 0.5f);
    float current_bank = get_current_bank(self);
    float bank_error = target_bank - current_bank;

    actions[2] = clampf(bank_error * 4.0f, -1.0f, 1.0f);
    actions[1] = -0.1f;   // Slight climb to gain energy
    actions[0] = 1.0f;    // Full throttle!
    actions[3] = 0.0f;
    actions[4] = -1.0f;
}

// Pure pursuit: nose directly at target
static inline void execute_pursuit_pure(AutopilotState* ap, AutoAceState* ace,
                                        Plane* self, Plane* target, float* actions) {
    Vec3 to_target = sub3(target->pos, self->pos);
    float desired_heading = atan2f(to_target.y, to_target.x);
    float horiz_dist = sqrtf(to_target.x * to_target.x + to_target.y * to_target.y);
    float desired_pitch = atan2f(to_target.z, horiz_dist);

    float current_heading = get_current_heading(self);
    float current_pitch = get_current_pitch(self);

    float heading_error = normalize_angle(desired_heading - current_heading);
    float pitch_error = desired_pitch - current_pitch;

    // Positive heading_error → negative target_bank → turn left
    float target_bank = clampf(heading_error * -2.0f, -1.0f, 1.0f);
    float current_bank = get_current_bank(self);
    float bank_error = target_bank - current_bank;

    actions[2] = clampf(bank_error * 5.0f, -1.0f, 1.0f);
    actions[1] = clampf(-pitch_error * 2.5f, -0.6f, 0.6f);
    actions[0] = 0.8f * 2.0f - 1.0f;
    actions[3] = clampf(heading_error * 0.3f, -0.2f, 0.2f);
    actions[4] = -1.0f;
}

// Hard turn (generic): execute hard turn left or right
static inline void execute_hard_turn(AutopilotState* ap, AutoAceState* ace,
                                     Plane* self, int direction, float* actions) {
    // direction: +1 = right (positive bank), -1 = left (negative bank)
    float target_bank = direction * 1.2f;  // ~70 deg
    float current_bank = get_current_bank(self);
    float bank_error = target_bank - current_bank;

    actions[2] = clampf(bank_error * 5.0f, -1.0f, 1.0f);
    actions[1] = -0.5f;  // Pull to turn
    actions[0] = 0.9f * 2.0f - 1.0f;
    actions[3] = 0.0f;
    actions[4] = -1.0f;
}

// ============================================================================
// TACTICAL DECISION FSM
// ============================================================================

static inline AutopilotMode select_tactical_mode(TacticalState* ts, AutoAceState* ace, Plane* self) {
    EngagementState engage = classify_engagement(ts);
    ace->engagement = engage;

    switch (engage) {
        case ENGAGE_WEAPONS:
            return AP_GUN_TRACK;

        case ENGAGE_OFFENSIVE:
            // Behind target, closing
            if (ts->closure_rate > AUTOACE_CLOSURE_FAST && ts->range < 400.0f) {
                return AP_HIGH_YOYO;  // Too fast, will overshoot
            }
            if (ts->closure_rate < AUTOACE_CLOSURE_SLOW) {
                return AP_PURSUIT_LEAD;  // Falling behind, cut inside
            }
            return AP_PURSUIT_LAG;  // Default: controlled pursuit

        case ENGAGE_NEUTRAL:
            // Turn fight for position
            if (ts->energy_delta > AUTOACE_ENERGY_ADVANTAGE) {
                return AP_HIGH_YOYO;  // Convert energy to position
            }
            // Turn toward target
            if (ts->antenna_train > 90.0f) {
                // Target behind us, turn to face
                return (rndf(0, 1) > 0.5f) ? AP_HARD_TURN_LEFT : AP_HARD_TURN_RIGHT;
            }
            return AP_PURSUIT_PURE;

        case ENGAGE_DEFENSIVE:
            // Threat behind, need to survive
            if (self->pos.z > 1500.0f && ts->closure_rate > 30.0f) {
                // High altitude and fast closure - could split-s but we don't have that
                return AP_SCISSORS;  // Force overshoot
            }
            if (ts->range < 300.0f) {
                return AP_SCISSORS;  // Force overshoot when close
            }
            return AP_BREAK_TURN;  // Hard turn away

        case ENGAGE_EXTEND:
            return AP_EXTEND;  // Run away, rebuild energy
    }

    return AP_LEVEL;  // Fallback
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================

static inline void autoace_init(AutoAceState* ace) {
    memset(ace, 0, sizeof(AutoAceState));
    ace->scissors_direction = 0;
    ace->mode_timer = 0;
    ace->maneuver_phase = 0;
    ace->yoyo_apex_alt = 0.0f;
}

static inline void autoace_step(AutopilotState* ap, AutoAceState* ace,
                                Plane* self, Plane* target, float* actions, float dt) {
    // Initialize actions
    actions[0] = 0.0f;   // throttle
    actions[1] = 0.0f;   // elevator
    actions[2] = 0.0f;   // ailerons
    actions[3] = 0.0f;   // rudder
    actions[4] = -1.0f;  // trigger (default: don't fire)

    // Compute tactical state
    compute_tactical_state(self, target, &ace->tactical);

    // Decrement mode timer
    if (ace->mode_timer > 0) {
        ace->mode_timer--;
    }

    // Select new mode if timer expired or maneuver complete
    bool maneuver_done = false;

    // Check if current maneuver is complete
    switch (ap->mode) {
        case AP_HIGH_YOYO:
            // Done when back at target altitude and closure rate reasonable
            if (ace->maneuver_phase == 1 &&
                ace->tactical.closure_rate < 30.0f &&
                ace->tactical.closure_rate > -10.0f) {
                maneuver_done = true;
            }
            break;
        case AP_BREAK_TURN:
            // Done when target no longer behind us
            if (ace->tactical.antenna_train < 100.0f) {
                maneuver_done = true;
            }
            break;
        case AP_EXTEND:
            // Done when we have energy advantage or good separation
            if (ace->tactical.energy_delta > 0.0f ||
                ace->tactical.range > 800.0f) {
                maneuver_done = true;
            }
            break;
        default:
            break;
    }

    // Mode selection with persistence
    if (ace->mode_timer <= 0 || maneuver_done) {
        AutopilotMode new_mode = select_tactical_mode(&ace->tactical, ace, self);
        if (new_mode != ap->mode) {
            ap->mode = new_mode;
            ace->mode_timer = AUTOACE_MIN_MODE_TIME;
            ace->maneuver_phase = 0;
            ace->yoyo_apex_alt = 0.0f;
            ace->scissors_direction = 0;
        }
    }

    // Execute current maneuver
    switch (ap->mode) {
        case AP_GUN_TRACK:
            execute_gun_track(ap, ace, self, target, actions);
            break;
        case AP_PURSUIT_LAG:
            execute_pursuit_lag(ap, ace, self, target, actions);
            break;
        case AP_PURSUIT_LEAD:
            execute_pursuit_lead(ap, ace, self, target, actions);
            break;
        case AP_PURSUIT_PURE:
            execute_pursuit_pure(ap, ace, self, target, actions);
            break;
        case AP_HIGH_YOYO:
            execute_high_yoyo(ap, ace, self, target, actions);
            break;
        case AP_SCISSORS:
            execute_scissors(ap, ace, self, target, actions);
            break;
        case AP_BREAK_TURN:
            execute_break_turn(ap, ace, self, target, actions);
            break;
        case AP_EXTEND:
            execute_extend(ap, ace, self, target, actions);
            break;
        case AP_HARD_TURN_LEFT:
            execute_hard_turn(ap, ace, self, -1, actions);
            break;
        case AP_HARD_TURN_RIGHT:
            execute_hard_turn(ap, ace, self, +1, actions);
            break;

        default:
            // Fall back to existing autopilot behavior
            autopilot_step(ap, self, actions, dt);
            break;
    }

    // Handle fire cooldown
    if (self->fire_cooldown > 0) {
        self->fire_cooldown--;
    }
    if (actions[4] > 0.5f && self->fire_cooldown == 0) {
        self->fire_cooldown = AUTOACE_FIRE_COOLDOWN;
    }

    // Debug output
    #if DEBUG >= 3
    static int debug_counter = 0;
    if (debug_counter++ % 50 == 0) {  // Every second
        const char* mode_names[] = {
            "STRAIGHT", "LEVEL", "TURN_L", "TURN_R",
            "CLIMB", "DESCEND", "HARD_L", "HARD_R",
            "WEAVE", "EVASIVE", "RANDOM",
            "PURSUIT_LEAD", "PURSUIT_LAG", "PURSUIT_PURE",
            "HIGH_YOYO", "LOW_YOYO", "SCISSORS", "BREAK",
            "SPLIT_S", "EXTEND", "BARREL_ATK", "GUN_TRACK"
        };
        const char* engage_names[] = {
            "OFFENSIVE", "NEUTRAL", "DEFENSIVE", "WEAPONS", "EXTEND"
        };
        printf("[AUTOACE] mode=%s engage=%s range=%.0f aspect=%.0f train=%.0f closure=%.0f\n",
               mode_names[ap->mode], engage_names[ace->engagement],
               ace->tactical.range, ace->tactical.aspect_angle,
               ace->tactical.antenna_train, ace->tactical.closure_rate);
    }
    #endif
}

#endif // AUTOACE_H
