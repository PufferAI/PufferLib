#pragma once
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "contact_traits.cuh"

// Persistent feature-keyed impulse solver with compliant normal rows, split
// correction and patch torsion. Supply contacts each substep; retain Cache and
// clear it on reset or topology changes. Geometry and pair policy are caller-owned.
// R supplies optional live articulation reactions, including proxy displacement.
// The default policy handles independent rigid bodies without articulation state.
template<class T = PfContactTraits, class R = PfNoContactReaction,
         int MaxManifolds = 48, int MaxCandidates = 20, int MaxCache = 192>
struct PfImpulseSolverT {
    using Vec3 = typename T::Vec3;
    using Contact = typename T::Contact;
    using Body = typename T::Body;
    using State = typename R::State;
    using Reaction = typename R::Reaction;
    using AngularReaction = typename R::AngularReaction;
    static constexpr int max_manifolds = MaxManifolds, max_candidates = MaxCandidates;
    static constexpr int max_cache = MaxCache, max_points = 4, max_iterations = 64;
    static constexpr float epsilon = 1.0e-8f;
    static_assert(MaxManifolds > 0 && MaxCandidates > 0 && MaxCache > 0, "positive contact capacities required");
    typedef struct Config {
        int velocity_iterations;
        int position_iterations;
        float velocity_impulse_tolerance; // 0 disables; does not lower the hard cap
        int warm_start;
        int split_position;
        float position_beta;
        float slop;
        float speculative_margin;
        float static_friction;
        float dynamic_friction;
        float restitution;
        float restitution_threshold;
        float max_normal_impulse;
        float max_position_correction;
        float max_position_impulse;
        int cache_max_age;
    } Config;

    typedef struct Patch {
        float area; // 0 disables torsion
        Vec3 centroid;
        float second_11;
        float second_22;
        float second_12;
    } Patch;

    typedef struct Candidate {
        Contact contact;
        uint32_t feature;
        uint32_t patch_group;
        Patch patch;
    } Candidate;

    typedef struct Point {
        uint32_t feature;
        uint32_t patch_group;
        Vec3 point_a;
        Vec3 point_b;
        Vec3 local_a;
        Vec3 local_b;
        float separation;
        float normal_mass;
        float tangent_1_mass;
        float tangent_2_mass;
        float normal_impulse;
        float tangent_1_impulse;
        float tangent_2_impulse;
        float velocity_bias;
        float pre_normal_velocity;
        float normal_erp; // erp [1/s]; cfm = 1/(dt*(c+dt*k)); 0 = hard row
        float normal_cfm;
        float prescribed_separation_offset; // split correction on FK proxy
        Patch patch;
        Reaction reaction;
    } Point;

    typedef struct Manifold {
        int body_a;
        int body_b;
        uint32_t pair_key;
        Vec3 normal;
        Vec3 tangent_1;
        Vec3 tangent_2;
        float static_friction;
        float dynamic_friction;
        float restitution;
        float patch_area;
        Vec3 patch_centroid;
        float patch_second_11;
        float patch_second_22;
        float patch_second_12;
        float torsional_radius;
        float patch_second_moment;
        float torsional_impulse;
        float torsional_mass;
        AngularReaction angular_reaction;
        uint32_t angular_cache_feature; // independent of point[0].feature
        int point_count;
        Point points[max_points];
    } Manifold;

    typedef struct CacheEntry {
        int body_a;
        int body_b;
        uint32_t pair_key;
        uint32_t feature;
        uint32_t stamp;
        Vec3 normal;
        Vec3 tangent_1;
        Vec3 tangent_2;
        float normal_impulse;
        float tangent_1_impulse;
        float tangent_2_impulse;
        float torsional_impulse;
    } CacheEntry;

    typedef struct Cache {
        uint32_t tick;
        int count;
        CacheEntry entries[max_cache];
    } Cache;

    __device__ static __forceinline__ int finite(float value) {
        return value == value && value < 1.0e30f && value > -1.0e30f; // no isfinite()
    }

    __device__ static __forceinline__ float vector_length(Vec3 value) {
        float length_squared = T::dot(value, value);
        return length_squared > epsilon * epsilon ? sqrtf(length_squared) : 0.0f;
    }

    __device__ static __forceinline__ Vec3 normalize(Vec3 value, Vec3 fallback) {
        float length = vector_length(value);
        if (length > 0.0f) {
            return T::scale(value, 1.0f / length);
        }
        length = vector_length(fallback);
        if (length > 0.0f) {
            return T::scale(fallback, 1.0f / length);
        }
        return T::v3(1.0f, 0.0f, 0.0f);
    }

    __device__ static __forceinline__ void tangents(Vec3 normal, Vec3* tangent_1, Vec3* tangent_2) {
        Vec3 reference = fabsf(normal.y) < 0.90f ? T::v3(0.0f, 1.0f, 0.0f) : T::v3(1.0f, 0.0f, 0.0f);
        *tangent_1 = normalize(T::cross(reference, normal), T::v3(0.0f, 0.0f, 1.0f));
        *tangent_2 = normalize(T::cross(normal, *tangent_1), T::v3(0.0f, 1.0f, 0.0f));
    }

    __device__ static __forceinline__ uint32_t hash_pair(int body_a, int body_b) {
        uint32_t value = 2166136261u;
        value = (value ^ (uint32_t)(body_a + 1)) * 16777619u;
        value = (value ^ (uint32_t)(body_b + 1)) * 16777619u;
        return value;
    }

    __device__ static __forceinline__ Vec3 midpoint(const Candidate* candidate) {
        return T::scale(T::add(candidate->contact.point_a, candidate->contact.point_b), 0.5f);
    }

    __device__ static __forceinline__ int candidate_before(const Candidate* left, const Candidate* right) {
        if (left->feature != right->feature) {
            return left->feature < right->feature;
        }
        if (left->contact.separation != right->contact.separation) {
            return left->contact.separation < right->contact.separation;
        }
        Vec3 left_point = midpoint(left);
        Vec3 right_point = midpoint(right);
        if (left_point.x != right_point.x) {
            return left_point.x < right_point.x;
        }
        if (left_point.y != right_point.y) {
            return left_point.y < right_point.y;
        }
        return left_point.z < right_point.z;
    }

    __device__ static __forceinline__ int candidate_usable(const Candidate* candidate, float margin) {
        const Contact* contact = &candidate->contact;
        if (!finite(contact->separation) || !finite(contact->normal.x) || !finite(contact->normal.y)
            || !finite(contact->normal.z)) {
            return 0;
        }
        if (!finite(contact->point_a.x) || !finite(contact->point_a.y) || !finite(contact->point_a.z)
            || !finite(contact->point_b.x) || !finite(contact->point_b.y)
            || !finite(contact->point_b.z)) {
            return 0;
        }
        return contact->hit || contact->separation <= margin;
    }

    __device__ static __forceinline__ int manifold(int body_a, int body_b, const Candidate* candidates,
        int candidate_count, float margin, float static_friction, float dynamic_friction,
        float restitution, Manifold* manifold) {
        manifold->body_a = body_a;
        manifold->body_b = body_b;
        manifold->pair_key = hash_pair(body_a, body_b);
        manifold->angular_cache_feature = 0xa0000000u | (manifold->pair_key & 0x0fffffffu);
        manifold->normal = T::v3(1.0f, 0.0f, 0.0f);
        manifold->tangent_1 = T::v3(0.0f, 1.0f, 0.0f);
        manifold->tangent_2 = T::v3(0.0f, 0.0f, 1.0f);
        manifold->static_friction = T::max(static_friction, 0.0f);
        manifold->dynamic_friction = T::clamp(dynamic_friction, 0.0f, manifold->static_friction);
        manifold->restitution = T::clamp(restitution, 0.0f, 1.0f);
        manifold->point_count = 0;

        Candidate work[max_candidates];
        int count = T::min(T::max(candidate_count, 0), max_candidates);
        int usable = 0;
        for (int index = 0; index < count; ++index) {
            if (!candidate_usable(&candidates[index], margin)) {
                continue;
            }
            work[usable] = candidates[index];
            ++usable;
        }
        for (int index = 1; index < usable; ++index) {
            Candidate value = work[index];
            int cursor = index;
            while (cursor > 0 && candidate_before(&value, &work[cursor - 1])) {
                work[cursor] = work[cursor - 1];
                --cursor;
            }
            work[cursor] = value;
        }
        if (usable == 0) {
            return 0;
        }

        int unique = 0;
        for (int index = 0; index < usable; ++index) {
            if (unique > 0 && work[index].feature == work[unique - 1].feature) {
                continue;
            }
            work[unique++] = work[index];
        }
        usable = unique;
        int selected[max_points];
        int selected_count = 0;
        int deepest = 0;
        for (int index = 1; index < usable; ++index) {
            if (work[index].contact.separation < work[deepest].contact.separation) {
                deepest = index;
            }
        }
        selected[selected_count++] = deepest;
        while (selected_count < max_points && selected_count < usable) {
            int best = -1;
            float best_score = -1.0f;
            for (int index = 0; index < usable; ++index) {
                int already = 0;
                for (int slot = 0; slot < selected_count; ++slot) {
                    if (selected[slot] == index) {
                        already = 1;
                    }
                }
                if (already) {
                    continue;
                }
                Vec3 point = midpoint(&work[index]);
                float score = 1.0e30f;
                for (int slot = 0; slot < selected_count; ++slot) {
                    Vec3 other = midpoint(&work[selected[slot]]);
                    Vec3 delta = T::sub(point, other);
                    score = T::min(score, T::dot(delta, delta));
                }
                if (best < 0 || score > best_score
                    || (score == best_score && work[index].feature < work[best].feature)) {
                    best = index;
                    best_score = score;
                }
            }
            if (best < 0) {
                break;
            }
            selected[selected_count++] = best;
        }
        for (int index = 1; index < selected_count; ++index) {
            int value = selected[index];
            int cursor = index;
            while (cursor > 0 && work[value].feature < work[selected[cursor - 1]].feature) {
                selected[cursor] = selected[cursor - 1];
                --cursor;
            }
            selected[cursor] = value;
        }

        Vec3 normal = work[deepest].contact.normal;
        Vec3 fallback = T::sub(work[deepest].contact.point_a, work[deepest].contact.point_b);
        manifold->normal = normalize(normal, fallback);
        tangents(manifold->normal, &manifold->tangent_1, &manifold->tangent_2);
        manifold->torsional_radius = 0.0f;
        manifold->patch_area = 0.0f;
        manifold->patch_centroid = T::v3(0.0f, 0.0f, 0.0f);
        manifold->patch_second_11 = 0.0f;
        manifold->patch_second_22 = 0.0f;
        manifold->patch_second_12 = 0.0f;
        manifold->patch_second_moment = 0.0f;
        manifold->torsional_impulse = 0.0f;
        manifold->torsional_mass = 0.0f;
        memset(&manifold->angular_reaction, 0, sizeof(manifold->angular_reaction));
        manifold->point_count = selected_count;
        for (int slot = 0; slot < selected_count; ++slot) {
            const Candidate* candidate = &work[selected[slot]];
            Point* point = &manifold->points[slot];
            point->feature = candidate->feature;
            point->patch_group = candidate->patch_group;
            point->point_a = candidate->contact.point_a;
            point->point_b = candidate->contact.point_b;
            point->local_a = T::v3(0.0f, 0.0f, 0.0f);
            point->local_b = T::v3(0.0f, 0.0f, 0.0f);
            point->separation = candidate->contact.separation;
            point->normal_mass = 0.0f;
            point->tangent_1_mass = 0.0f;
            point->tangent_2_mass = 0.0f;
            point->normal_impulse = 0.0f;
            point->tangent_1_impulse = 0.0f;
            point->tangent_2_impulse = 0.0f;
            point->velocity_bias = 0.0f;
            point->pre_normal_velocity = 0.0f;
            point->normal_erp = 0.0f;
            point->normal_cfm = 0.0f;
            point->prescribed_separation_offset = 0.0f;
            point->patch = candidate->patch;
            memset(&point->reaction, 0, sizeof(point->reaction));
        }
        return selected_count;
    }

    __device__ static __forceinline__ void sort(Manifold* manifolds, int manifold_count) {
        int count = T::min(T::max(manifold_count, 0), max_manifolds);
        for (int index = 1; index < count; ++index) {
            Manifold value = manifolds[index];
            int cursor = index;
            while (cursor > 0) {
                const Manifold* left = &value;
                const Manifold* right = &manifolds[cursor - 1];
                int before = 0;
                if (left->pair_key != right->pair_key) {
                    before = left->pair_key < right->pair_key;
                } else if (left->body_a != right->body_a) {
                    before = left->body_a < right->body_a;
                } else if (left->body_b != right->body_b) {
                    before = left->body_b < right->body_b;
                } else if (left->point_count != right->point_count) {
                    before = left->point_count < right->point_count;
                } else {
                    int point_count = T::min(T::max(left->point_count, 0), max_points);
                    for (int point = 0; point < point_count; ++point) {
                        if (left->points[point].feature != right->points[point].feature) {
                            before = left->points[point].feature < right->points[point].feature;
                            break;
                        }
                    }
                }
                if (!before) {
                    break;
                }
                manifolds[cursor] = manifolds[cursor - 1];
                --cursor;
            }
            manifolds[cursor] = value;
        }
    }

    __host__ __device__ static __forceinline__ void clear_cache(Cache* cache) {
        cache->tick = 0;
        cache->count = 0;
    }

    __device__ static __forceinline__ int find_cache(
        const Cache* cache, int body_a, int body_b, uint32_t feature) {
        uint32_t pair_key = hash_pair(body_a, body_b);
        for (int index = 0; index < cache->count; ++index) {
            const CacheEntry* entry = &cache->entries[index];
            if (entry->pair_key == pair_key && entry->body_a == body_a && entry->body_b == body_b
                && entry->feature == feature) {
                return index;
            }
        }
        return -1;
    }

    __device__ static __forceinline__ int cache_slot(
        const Cache* cache, int body_a, int body_b, uint32_t feature) {
        int existing = find_cache(cache, body_a, body_b, feature);
        if (existing >= 0) {
            return existing;
        }
        if (cache->count < max_cache) {
            return cache->count;
        }
        int best = 0;
        for (int index = 1; index < cache->count; ++index) {
            const CacheEntry* left = &cache->entries[index];
            const CacheEntry* right = &cache->entries[best];
            if (left->stamp < right->stamp
                || (left->stamp == right->stamp
                    && (left->pair_key < right->pair_key
                        || (left->pair_key == right->pair_key
                            && (left->feature < right->feature
                                || (left->feature == right->feature && index < best)))))) {
                best = index;
            }
        }
        return best;
    }

    __device__ static __forceinline__ Vec3 anchor(const Body* body, Vec3 local) {
        return T::add(T::position(*body), T::rotate(T::rotation(*body), local));
    }

    __device__ static __forceinline__ float effective_mass(const Body* body_a, Vec3 point_a,
        const Body* body_b, Vec3 point_b, Vec3 direction, const Reaction* reaction,
        int direction_index) {
        float denominator = T::impulse_denominator(body_a, point_a, direction) + T::impulse_denominator(body_b, point_b, direction);
        if (reaction != NULL && reaction->active && direction_index >= 0 && direction_index < 3) {
            denominator += T::max(reaction->inverse_mass[direction_index], 0.0f);
        }
        return denominator;
    }

    __device__ static __forceinline__ void apply_pair(
        Body* body_a, Vec3 point_a, Body* body_b, Vec3 point_b, Vec3 impulse) {
        T::apply_impulse(body_a, point_a, impulse);
        T::apply_impulse(body_b, point_b, T::scale(impulse, -1.0f));
    }

    __device__ static __forceinline__ void apply_reaction(const Reaction* reaction, State* reaction_state,
        int direction_index, float impulse) {
        if (reaction == NULL || !reaction->active || reaction_state == NULL || direction_index < 0
            || direction_index >= 3) {
            return;
        }
        R::apply(*reaction, *reaction_state, direction_index, impulse);
    }

    __device__ static __forceinline__ void update_offsets(Manifold* manifolds, int manifold_count,
        const typename R::Displacement& delta) {
        int count = T::min(T::max(manifold_count, 0), max_manifolds);
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < manifolds[i].point_count; ++j) {
                auto& point = manifolds[i].points[j];
                if (point.reaction.active)
                    point.prescribed_separation_offset -= R::displacement(point.reaction, delta);
            }
        }
    }

    __device__ static __forceinline__ float body_velocity(const Body* body, Vec3 point, Vec3 direction) {
        Vec3 velocity = T::add(body->linear_velocity,
            T::cross(body->angular_velocity, T::sub(point, T::position(*body))));
        return T::dot(velocity, direction);
    }

    __device__ static __forceinline__ float reaction_velocity(const Body* body, const Reaction* reaction,
        const State* reaction_state, int direction_index, Vec3 point, Vec3 direction) {
        if (reaction != NULL && reaction->active && reaction_state != NULL && direction_index >= 0
            && direction_index < 3) {
            return R::velocity(*reaction, *reaction_state, direction_index);
        }
        return body_velocity(body, point, direction);
    }

    __device__ static __forceinline__ float angular_mass(const Body* body, Vec3 axis) {
        if (!T::dynamic(*body)) {
            return 0.0f;
        }
        return T::dot(axis, T::inverse_inertia(body, axis));
    }

    __device__ static __forceinline__ void apply_angular_pair(
        Body* body_a, Body* body_b, Vec3 axis, float moment) {
        if (T::dynamic(*body_a)) {
            body_a->angular_velocity =
                T::add(body_a->angular_velocity, T::scale(T::inverse_inertia(body_a, axis), moment));
        }
        if (T::dynamic(*body_b)) {
            body_b->angular_velocity =
                T::add(body_b->angular_velocity, T::scale(T::inverse_inertia(body_b, axis), -moment));
        }
    }

    __device__ static __forceinline__ void apply_angular_reaction(
        const AngularReaction* reaction, State* reaction_state, float moment) {
        if (reaction == NULL || !reaction->active || reaction_state == NULL) {
            return;
        }
        R::apply_angular(*reaction, *reaction_state, moment);
    }

    __device__ static __forceinline__ void apply_position(Body* body, Vec3 point, Vec3 impulse) {
        if (!T::dynamic(*body)) {
            return;
        }
        Vec3 lever = T::sub(point, T::position(*body));
        T::position(*body) =
            T::add(T::position(*body), T::scale(impulse, T::inverse_mass(*body)));
        Vec3 angular_delta = T::inverse_inertia(body, T::cross(lever, impulse));
        T::rotation(*body) = T::integrate_rotation(T::rotation(*body), angular_delta, 1.0f);
    }

    __device__ static __forceinline__ void refresh(Body* body_a, Body* body_b,
        const Manifold* manifold, Point* point) {
        point->point_a = anchor(body_a, point->local_a);
        point->point_b = anchor(body_b, point->local_b);
        point->separation = T::dot(T::sub(point->point_a, point->point_b), manifold->normal)
            + point->prescribed_separation_offset;
    }

    __device__ static __forceinline__ void load_cache(const Cache* cache, Manifold* manifold,
        Point* point, int warm_start, int max_age) {
        point->normal_impulse = 0.0f;
        point->tangent_1_impulse = 0.0f;
        point->tangent_2_impulse = 0.0f;
        if (!warm_start) {
            return;
        }
        assert(cache != NULL);
        int slot = find_cache(cache, manifold->body_a, manifold->body_b, point->feature);
        if (slot < 0) {
            return;
        }
        const CacheEntry* entry = &cache->entries[slot];
        if (max_age >= 0 && cache->tick - entry->stamp > (uint32_t)max_age) {
            return;
        }
        point->normal_impulse = T::max(entry->normal_impulse, 0.0f);
        Vec3 old_tangent = T::add(T::scale(entry->tangent_1, entry->tangent_1_impulse),
            T::scale(entry->tangent_2, entry->tangent_2_impulse));
        point->tangent_1_impulse = T::dot(old_tangent, manifold->tangent_1);
        point->tangent_2_impulse = T::dot(old_tangent, manifold->tangent_2);
        float tangent_length = hypotf(point->tangent_1_impulse, point->tangent_2_impulse);
        float limit = T::max(manifold->static_friction * point->normal_impulse, 0.0f);
        if (tangent_length > limit) {
            float scale = limit / T::max(tangent_length, epsilon);
            point->tangent_1_impulse *= scale;
            point->tangent_2_impulse *= scale;
        }
    }

    __device__ static __forceinline__ void load_angular_cache(
        const Cache* cache, Manifold* manifold, int warm_start, int max_age) {
        manifold->torsional_impulse = 0.0f;
        if (!warm_start || cache == NULL) {
            return;
        }
        int slot = find_cache(cache, manifold->body_a, manifold->body_b, manifold->angular_cache_feature);
        if (slot < 0) {
            return;
        }
        const CacheEntry* entry = &cache->entries[slot];
        if (max_age >= 0 && cache->tick - entry->stamp > (uint32_t)max_age) {
            return;
        }
        Vec3 old_moment = T::scale(entry->normal, entry->torsional_impulse);
        manifold->torsional_impulse = T::dot(old_moment, manifold->normal);
    }

    __device__ static __forceinline__ void prepare(Body* bodies, int body_count,
        Manifold* manifolds, int manifold_count, const Cache* cache,
        const Config* config) {
        int count = T::min(T::max(manifold_count, 0), max_manifolds);
        for (int manifold_index = 0; manifold_index < count; ++manifold_index) {
            Manifold* manifold = &manifolds[manifold_index];
            if (manifold->body_a < 0 || manifold->body_b < 0 || manifold->body_a >= body_count
                || manifold->body_b >= body_count || manifold->body_a == manifold->body_b) {
                manifold->point_count = 0;
                continue;
            }
            Body* body_a = &bodies[manifold->body_a];
            Body* body_b = &bodies[manifold->body_b];
            manifold->normal =
                normalize(manifold->normal, T::sub(T::position(*body_a), T::position(*body_b)));
            tangents(manifold->normal, &manifold->tangent_1, &manifold->tangent_2);
            manifold->static_friction = T::max(manifold->static_friction, 0.0f);
            manifold->dynamic_friction =
                T::clamp(manifold->dynamic_friction, 0.0f, manifold->static_friction);
            manifold->restitution = T::clamp(manifold->restitution, 0.0f, 1.0f);
            int point_count = T::min(T::max(manifold->point_count, 0), max_points);
            manifold->point_count = point_count;
            for (int point_index = 0; point_index < point_count; ++point_index) {
                Point* point = &manifold->points[point_index];
                point->local_a = T::rotate(
                    T::conjugate(T::rotation(*body_a)), T::sub(point->point_a, T::position(*body_a)));
                point->local_b = T::rotate(
                    T::conjugate(T::rotation(*body_b)), T::sub(point->point_b, T::position(*body_b)));
                refresh(body_a, body_b, manifold, point);
                point->normal_mass = 1.0f
                    / T::max(effective_mass(body_a, point->point_a, body_b, point->point_b, manifold->normal,
                                 &point->reaction, 0)
                            + point->normal_cfm,
                        epsilon);
                point->tangent_1_mass = 1.0f
                    / T::max(effective_mass(body_a, point->point_a, body_b, point->point_b,
                                 manifold->tangent_1, &point->reaction, 1),
                        epsilon);
                point->tangent_2_mass = 1.0f
                    / T::max(effective_mass(body_a, point->point_a, body_b, point->point_b,
                                 manifold->tangent_2, &point->reaction, 2),
                        epsilon);
                load_cache(cache, manifold, point, config->warm_start != 0, config->cache_max_age);
            }
            load_angular_cache(cache, manifold, config->warm_start != 0, config->cache_max_age);
            if (manifold->torsional_radius > 0.0f) {
                float denominator = angular_mass(body_a, manifold->normal) + angular_mass(body_b, manifold->normal)
                    + manifold->angular_reaction.inverse_mass;
                manifold->torsional_mass = 1.0f / T::max(denominator, epsilon);
            }
        }
    }

    __device__ static __forceinline__ void solve_positions(Body* bodies, int body_count,
        Manifold* manifolds, int manifold_count, const Config* config,
        State* reaction_state) {
        if (!config->split_position) {
            return;
        }
        int count = T::min(T::max(manifold_count, 0), max_manifolds);
        int iterations = T::min(T::max(config->position_iterations, 0), max_iterations);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            float maximum_penetration = 0.0f;
            for (int manifold_index = 0; manifold_index < count; ++manifold_index) {
                Manifold* manifold = &manifolds[manifold_index];
                if (manifold->point_count <= 0 || manifold->body_a < 0 || manifold->body_b < 0
                    || manifold->body_a >= body_count || manifold->body_b >= body_count) {
                    continue;
                }
                Body* body_a = &bodies[manifold->body_a];
                Body* body_b = &bodies[manifold->body_b];
                for (int point_index = 0; point_index < manifold->point_count; ++point_index) {
                    Point* point = &manifold->points[point_index];
                    refresh(body_a, body_b, manifold, point);
                    float penetration = T::max(-point->separation - config->slop, 0.0f);
                    maximum_penetration = T::max(maximum_penetration, penetration);
                    if (penetration <= 0.0f) {
                        continue;
                    }
                    float correction = config->position_beta * penetration;
                    correction = T::min(correction, T::max(config->max_position_correction, 0.0f));
                    float denominator = effective_mass(body_a, point->point_a, body_b, point->point_b,
                                            manifold->normal, &point->reaction, 0)
                        + point->normal_cfm;
                    if (denominator <= epsilon) {
                        continue;
                    }
                    float magnitude = correction / denominator;
                    magnitude = T::min(magnitude, T::max(config->max_position_impulse, 0.0f));
                    apply_position(body_a, point->point_a, T::scale(manifold->normal, magnitude));
                    apply_position(body_b, point->point_b, T::scale(manifold->normal, -magnitude));
                    typename R::Displacement delta = {};
                    if (reaction_state != NULL && point->reaction.active)
                        delta = R::correct(*reaction_state, point->reaction, magnitude);
                    update_offsets(manifolds, manifold_count, delta);
                }
            }
            if (maximum_penetration <= config->slop) {
                break;
            }
        }
    }

    __device__ static __forceinline__ void prepare_bias(Body* bodies, int body_count,
        Manifold* manifolds, int manifold_count, float dt, const Config* config,
        State* reaction_state) {
        float safe_dt = T::max(dt, epsilon);
        int count = T::min(T::max(manifold_count, 0), max_manifolds);
        for (int manifold_index = 0; manifold_index < count; ++manifold_index) {
            Manifold* manifold = &manifolds[manifold_index];
            if (manifold->body_a < 0 || manifold->body_b < 0 || manifold->body_a >= body_count
                || manifold->body_b >= body_count) {
                continue;
            }
            Body* body_a = &bodies[manifold->body_a];
            Body* body_b = &bodies[manifold->body_b];
            for (int point_index = 0; point_index < manifold->point_count; ++point_index) {
                Point* point = &manifold->points[point_index];
                refresh(body_a, body_b, manifold, point);
                point->normal_mass = 1.0f
                    / T::max(effective_mass(body_a, point->point_a, body_b, point->point_b, manifold->normal,
                                 &point->reaction, 0)
                            + point->normal_cfm,
                        epsilon);
                point->tangent_1_mass = 1.0f
                    / T::max(effective_mass(body_a, point->point_a, body_b, point->point_b,
                                 manifold->tangent_1, &point->reaction, 1),
                        epsilon);
                point->tangent_2_mass = 1.0f
                    / T::max(effective_mass(body_a, point->point_a, body_b, point->point_b,
                                 manifold->tangent_2, &point->reaction, 2),
                        epsilon);
                float velocity_a = body_velocity(body_a, point->point_a, manifold->normal);
                float velocity_b = reaction_velocity(
                    body_b, &point->reaction, reaction_state, 0, point->point_b, manifold->normal);
                point->pre_normal_velocity = velocity_a - velocity_b;
                float restitution_target = 0.0f;
                if (point->pre_normal_velocity < -config->restitution_threshold
                    && point->separation <= config->speculative_margin) {
                    restitution_target = -manifold->restitution * point->pre_normal_velocity;
                }
                float speculative_target = point->separation > config->slop
                    ? -point->separation / safe_dt
                    : 0.0f; // do not clamp vn to 0
                float compliance_target = 0.0f;
                if (point->normal_erp > 0.0f && point->separation < 0.0f) {
                    compliance_target = point->normal_erp * (-point->separation);
                }
                float active_bias = T::max(compliance_target, restitution_target);
                if (point->separation > config->slop && active_bias <= 0.0f) {
                    active_bias = speculative_target;
                }
                point->velocity_bias = active_bias;
            }
        }
    }

    __device__ static __forceinline__ float torsional_limit(const Manifold* manifold, float friction) {
        float limit = 0.0f;
        for (int point_index = 0; point_index < manifold->point_count; ++point_index) {
            const Point* point = &manifold->points[point_index];
            if (point->patch_group == 0) {
                continue;
            }
            int first_group = 1;
            for (int previous = 0; previous < point_index; ++previous) {
                if (manifold->points[previous].patch_group == point->patch_group) {
                    first_group = 0;
                    break;
                }
            }
            if (!first_group) {
                continue;
            }
            float area = point->patch.area;
            if (area <= epsilon) {
                continue;
            }
            float group_normal_impulse = 0.0f;
            float group_tangent_squared = 0.0f;
            for (int member = point_index; member < manifold->point_count; ++member) {
                const Point* group_point = &manifold->points[member];
                if (group_point->patch_group != point->patch_group) {
                    continue;
                }
                group_normal_impulse += T::max(group_point->normal_impulse, 0.0f);
                group_tangent_squared += group_point->tangent_1_impulse * group_point->tangent_1_impulse
                    + group_point->tangent_2_impulse * group_point->tangent_2_impulse;
            }
            float full_capacity = T::max(friction, 0.0f) * group_normal_impulse;
            float remaining_squared = full_capacity * full_capacity - group_tangent_squared;
            float remaining_capacity = sqrtf(T::max(remaining_squared, 0.0f));
            float intrinsic_moment = point->patch.second_11 + point->patch.second_22;
            float radius = sqrtf(T::max(intrinsic_moment / area, 0.0f));
            limit += remaining_capacity * radius;
        }
        return limit;
    }

    __device__ static __forceinline__ void warm_angular(Body* body_a, Body* body_b,
        Manifold* manifold, State* reaction_state) {
        if (R::require_angular_reaction && manifold->angular_reaction.active == 0) {
            return;
        }
        float angular_limit = torsional_limit(manifold, manifold->static_friction);
        float angular_length = sqrtf(manifold->torsional_impulse * manifold->torsional_impulse);
        if (angular_length > angular_limit) {
            float scale = angular_limit / T::max(angular_length, epsilon);
            manifold->torsional_impulse *= scale;
        }
        apply_angular_pair(body_a, body_b, manifold->normal, manifold->torsional_impulse);
        apply_angular_reaction(&manifold->angular_reaction, reaction_state, manifold->torsional_impulse);
    }

    __device__ static __forceinline__ void solve_angular(Body* body_a, Body* body_b,
        Manifold* manifold, State* reaction_state) {
        if ((R::require_angular_reaction && manifold->angular_reaction.active == 0) || manifold->torsional_radius <= 0.0f) {
            return;
        }
        float normal = 0.0f;
        for (int point = 0; point < manifold->point_count; ++point) {
            normal += T::max(manifold->points[point].normal_impulse, 0.0f);
        }
        if (normal <= epsilon) {
            manifold->torsional_impulse = 0.0f;
            return;
        }
        float static_friction = T::max(manifold->static_friction, 0.0f);
        float dynamic_friction = T::clamp(manifold->dynamic_friction, 0.0f, static_friction);
        float prescribed_velocity = manifold->angular_reaction.active && reaction_state != NULL
            ? R::angular_velocity(manifold->angular_reaction, *reaction_state)
            : T::dot(body_b->angular_velocity, manifold->normal);
        float relative_normal =
            T::dot(body_a->angular_velocity, manifold->normal) - prescribed_velocity;
        float old_torsion = manifold->torsional_impulse;
        float candidate_torsion = old_torsion - relative_normal * manifold->torsional_mass;
        float torsion_limit = torsional_limit(manifold, static_friction);
        if (fabsf(candidate_torsion) > torsion_limit) {
            float dynamic_limit = torsional_limit(manifold, dynamic_friction);
            candidate_torsion = T::clamp(candidate_torsion, -dynamic_limit, dynamic_limit);
        }
        manifold->torsional_impulse = candidate_torsion;
        float torsion_delta = candidate_torsion - old_torsion;
        apply_angular_pair(body_a, body_b, manifold->normal, torsion_delta);
        apply_angular_reaction(&manifold->angular_reaction, reaction_state, torsion_delta);
    }

    __device__ static __forceinline__ void warm_start(Body* bodies, int body_count, Manifold* manifolds,
        int manifold_count, const Config* config, State* reaction_state) {
        if (!config->warm_start) {
            return;
        }
        int count = T::min(T::max(manifold_count, 0), max_manifolds);
        for (int manifold_index = 0; manifold_index < count; ++manifold_index) {
            Manifold* manifold = &manifolds[manifold_index];
            if (manifold->body_a < 0 || manifold->body_b < 0 || manifold->body_a >= body_count
                || manifold->body_b >= body_count) {
                continue;
            }
            Body* body_a = &bodies[manifold->body_a];
            Body* body_b = &bodies[manifold->body_b];
            for (int point_index = 0; point_index < manifold->point_count; ++point_index) {
                Point* point = &manifold->points[point_index];
                Vec3 impulse = T::add(T::scale(manifold->normal, point->normal_impulse),
                    T::add(T::scale(manifold->tangent_1, point->tangent_1_impulse),
                        T::scale(manifold->tangent_2, point->tangent_2_impulse)));
                apply_pair(body_a, point->point_a, body_b, point->point_b, impulse);
                apply_reaction(&point->reaction, reaction_state, 0, point->normal_impulse);
                apply_reaction(&point->reaction, reaction_state, 1, point->tangent_1_impulse);
                apply_reaction(&point->reaction, reaction_state, 2, point->tangent_2_impulse);
            }
            warm_angular(body_a, body_b, manifold, reaction_state);
        }
    }

    __device__ static __forceinline__ void solve_velocities(Body* bodies, int body_count,
        Manifold* manifolds, int manifold_count, const Config* config,
        State* reaction_state) {
        int count = T::min(T::max(manifold_count, 0), max_manifolds);
        int iterations = T::min(T::max(config->velocity_iterations, 0), max_iterations);
        warm_start(bodies, body_count, manifolds, count, config, reaction_state);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            float maximum_impulse_delta = 0.0f;
            for (int manifold_index = 0; manifold_index < count; ++manifold_index) {
                Manifold* manifold = &manifolds[manifold_index];
                if (manifold->point_count <= 0 || manifold->body_a < 0 || manifold->body_b < 0
                    || manifold->body_a >= body_count || manifold->body_b >= body_count) {
                    continue;
                }
                Body* body_a = &bodies[manifold->body_a];
                Body* body_b = &bodies[manifold->body_b];
                float dynamic_friction =
                    T::clamp(manifold->dynamic_friction, 0.0f, manifold->static_friction);
                for (int point_index = 0; point_index < manifold->point_count; ++point_index) {
                    Point* point = &manifold->points[point_index];
                    float velocity_a_normal = body_velocity(body_a, point->point_a, manifold->normal);
                    float velocity_b_normal = reaction_velocity(
                        body_b, &point->reaction, reaction_state, 0, point->point_b, manifold->normal);
                    float normal_velocity = velocity_a_normal - velocity_b_normal;
                    float old_normal = point->normal_impulse;
                    float candidate_normal = old_normal
                        + (point->velocity_bias - normal_velocity - point->normal_cfm * old_normal)
                            * point->normal_mass;
                    candidate_normal =
                        T::clamp(candidate_normal, 0.0f, T::max(config->max_normal_impulse, 0.0f));
                    point->normal_impulse = candidate_normal;
                    maximum_impulse_delta =
                        T::max(maximum_impulse_delta, fabsf(candidate_normal - old_normal));
                    apply_pair(body_a, point->point_a, body_b, point->point_b,
                        T::scale(manifold->normal, candidate_normal - old_normal));
                    apply_reaction(&point->reaction, reaction_state, 0, candidate_normal - old_normal);

                    float old_tangent_1 = point->tangent_1_impulse;
                    float old_tangent_2 = point->tangent_2_impulse;
                    float relative_tangent_1 = body_velocity(body_a, point->point_a, manifold->tangent_1)
                        - reaction_velocity(body_b, &point->reaction, reaction_state, 1, point->point_b,
                            manifold->tangent_1);
                    float relative_tangent_2 = body_velocity(body_a, point->point_a, manifold->tangent_2)
                        - reaction_velocity(body_b, &point->reaction, reaction_state, 2, point->point_b,
                            manifold->tangent_2);
                    float candidate_tangent_1 =
                        old_tangent_1 - relative_tangent_1 * point->tangent_1_mass;
                    float candidate_tangent_2 =
                        old_tangent_2 - relative_tangent_2 * point->tangent_2_mass;
                    float tangent_length = hypotf(candidate_tangent_1, candidate_tangent_2);
                    float static_limit = manifold->static_friction * candidate_normal;
                    if (tangent_length > static_limit) {
                        float dynamic_limit = dynamic_friction * candidate_normal;
                        float scale = dynamic_limit / T::max(tangent_length, epsilon);
                        candidate_tangent_1 *= scale;
                        candidate_tangent_2 *= scale;
                    }
                    point->tangent_1_impulse = candidate_tangent_1;
                    point->tangent_2_impulse = candidate_tangent_2;
                    maximum_impulse_delta = T::max(maximum_impulse_delta,
                        T::max(fabsf(candidate_tangent_1 - old_tangent_1),
                            fabsf(candidate_tangent_2 - old_tangent_2)));
                    Vec3 friction_delta =
                        T::add(T::scale(manifold->tangent_1, candidate_tangent_1 - old_tangent_1),
                            T::scale(manifold->tangent_2, candidate_tangent_2 - old_tangent_2));
                    apply_pair(body_a, point->point_a, body_b, point->point_b, friction_delta);
                    apply_reaction(&point->reaction, reaction_state, 1, candidate_tangent_1 - old_tangent_1);
                    apply_reaction(&point->reaction, reaction_state, 2, candidate_tangent_2 - old_tangent_2);
                }
                float old_torsion = manifold->torsional_impulse;
                solve_angular(body_a, body_b, manifold, reaction_state);
                maximum_impulse_delta =
                    T::max(maximum_impulse_delta, fabsf(manifold->torsional_impulse - old_torsion));
            }
            if (config->velocity_impulse_tolerance > 0.0f
                && maximum_impulse_delta <= config->velocity_impulse_tolerance) {
                break;
            }
        }
    }

    __device__ static __forceinline__ void write_cache(
        Cache* cache, const Manifold* manifolds, int manifold_count) {
        int count = T::min(T::max(manifold_count, 0), max_manifolds);
        for (int manifold_index = 0; manifold_index < count; ++manifold_index) {
            const Manifold* manifold = &manifolds[manifold_index];
            for (int point_index = 0; point_index < manifold->point_count; ++point_index) {
                const Point* point = &manifold->points[point_index];
                int slot = cache_slot(cache, manifold->body_a, manifold->body_b, point->feature);
                CacheEntry* entry = &cache->entries[slot];
                if (slot == cache->count) {
                    ++cache->count;
                }
                entry->body_a = manifold->body_a;
                entry->body_b = manifold->body_b;
                entry->pair_key = manifold->pair_key;
                entry->feature = point->feature;
                entry->stamp = cache->tick;
                entry->normal = manifold->normal;
                entry->tangent_1 = manifold->tangent_1;
                entry->tangent_2 = manifold->tangent_2;
                entry->normal_impulse = point->normal_impulse;
                entry->tangent_1_impulse = point->tangent_1_impulse;
                entry->tangent_2_impulse = point->tangent_2_impulse;
                entry->torsional_impulse = 0.0f;
            }
            if (manifold->patch_area > epsilon && manifold->torsional_radius > 0.0f) {
                int slot = cache_slot(
                    cache, manifold->body_a, manifold->body_b, manifold->angular_cache_feature);
                CacheEntry* entry = &cache->entries[slot];
                if (slot == cache->count) {
                    ++cache->count;
                }
                entry->body_a = manifold->body_a;
                entry->body_b = manifold->body_b;
                entry->pair_key = manifold->pair_key;
                entry->feature = manifold->angular_cache_feature;
                entry->stamp = cache->tick;
                entry->normal = manifold->normal;
                entry->tangent_1 = manifold->tangent_1;
                entry->tangent_2 = manifold->tangent_2;
                entry->normal_impulse = 0.0f;
                entry->tangent_1_impulse = 0.0f;
                entry->tangent_2_impulse = 0.0f;
                entry->torsional_impulse = manifold->torsional_impulse;
            }
        }
    }

    __device__ static __forceinline__ void solve(Body* bodies, int body_count,
        Manifold* manifolds, int manifold_count, float dt, const Config* config,
        Cache* cache, State* reaction_state = nullptr) {
        assert(config != NULL);
        assert(cache != NULL);
        cache->tick += 1u;
        if (cache->tick == 0u) {
            cache->tick = 1u;
        }
        int write = 0;
        for (int read = 0; read < cache->count; ++read) {
            CacheEntry* entry = &cache->entries[read];
            uint32_t age = cache->tick - entry->stamp;
            if (age > (uint32_t)config->cache_max_age) {
                continue;
            }
            if (write != read) {
                cache->entries[write] = *entry;
            }
            ++write;
        }
        cache->count = write;
        prepare(bodies, body_count, manifolds, manifold_count, cache, config);
        solve_positions(bodies, body_count, manifolds, manifold_count, config, reaction_state);
        prepare_bias(bodies, body_count, manifolds, manifold_count, dt, config, reaction_state);
        solve_velocities(bodies, body_count, manifolds, manifold_count, config, reaction_state);
        write_cache(cache, manifolds, manifold_count);
    }

};
using PfImpulseSolver = PfImpulseSolverT<>;
