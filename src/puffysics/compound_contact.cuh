#pragma once
#include "sat_manifold.cuh"
#include "impulse_solver.cuh"

// Stable feature namespaces are supplied by the caller. Patch ownership is
// carried separately in patch_group, so it imposes no feature bit layout.
struct PfCompoundFeatures {
    __device__ static __forceinline__ uint32_t box(uint32_t scope, int owner, int x, int z, uint32_t vertex) {
        uint32_t h=2166136261u;
        h=(h^scope)*16777619u; h=(h^(uint32_t)owner)*16777619u;
        h=(h^(uint32_t)x)*16777619u; h=(h^(uint32_t)z)*16777619u;
        // The impulse cache reserves the 0xa... namespace for angular rows.
        return 0xd0000000u | (((h^vertex)*16777619u)&0x0fffffffu);
    }
    __device__ static __forceinline__ uint32_t sphere(uint32_t scope, int owner) {
        return box(scope,owner,0,0,0);
    }
};

// Contacts on the exposed +/-local-Y surface of a union of aligned boxes.
// All components share X/Z axes; inward is their unit outward surface normal.
// Buffers and motion are caller-owned. No scene, robot or material constants.
template<class T=PfContactTraits, class Solver=PfImpulseSolverT<T>,
         int MaxComponents=8, int MaxCandidates=128, class F=PfCompoundFeatures>
struct PfCompoundContactT {
    using Vec3=typename T::Vec3;
    using Shape=typename T::Shape;
    using Body=typename T::Body;
    using Contact=typename T::Contact;
    using Candidate=typename Solver::Candidate;
    using Manifold=typename Solver::Manifold;
    using Patch=typename Solver::Patch;
    using Sat=PfSatCollisionT<T>;
    static constexpr int max_clip_vertices=Sat::max_clip_vertices;
    static constexpr int component_capacity=MaxComponents, candidate_capacity=MaxCandidates;
    static_assert(MaxComponents>0 && MaxComponents<=32 && MaxCandidates>0, "invalid compound capacities");
    enum Status { no_contact, contact, capacity, invalid };
    struct Result { Status status; int point_count; uint32_t component_mask; };
    struct Options {
        float dt=0.002f, margin=0.0005f;
        float static_friction=0.8f, dynamic_friction=0.8f, restitution=0;
        float boundary_epsilon=2.0e-7f, support_tolerance=2.0e-6f;
        float min_normal_alignment=0.45f;
        uint32_t feature_namespace=0, group_base=0;
    };
    struct ClipVertex { Vec3 point; uint32_t feature; };
    __device__ static __forceinline__ bool valid(const Shape* components, int count, const Options& o) {
        if (!components || count<=0 || count>MaxComponents || o.group_base>UINT32_MAX-(uint32_t)count)
            return false;
        for (int i=0;i<count;++i) if (components[i].type!=T::box_kind) return false;
        return true;
    }
    __device__ static __forceinline__ void measure_patch(const Vec3* polygon, int count, Vec3 plane_origin,
        Vec3 surface_normal, Vec3 tangent_1, Vec3 tangent_2, Patch* patch) {
        memset(patch, 0, sizeof(*patch));
        if (count <= 0) {
            return;
        }
        float x[max_clip_vertices];
        float z[max_clip_vertices];
        int vertex_count = T::min(count, max_clip_vertices);
        for (int index = 0; index < vertex_count; ++index) {
            Vec3 projected = polygon[index];
            float signed_distance = T::dot(T::sub(projected, plane_origin), surface_normal);
            projected = T::sub(projected, T::scale(surface_normal, signed_distance));
            Vec3 offset = T::sub(projected, plane_origin);
            x[index] = T::dot(offset, tangent_1);
            z[index] = T::dot(offset, tangent_2);
        }
        if (vertex_count < 3) {
            return;
        }

        float cross_sum = 0.0f;
        float centroid_x_sum = 0.0f;
        float centroid_z_sum = 0.0f;
        float second_11_sum = 0.0f;
        float second_22_sum = 0.0f;
        float second_12_sum = 0.0f;
        for (int index = 0; index < vertex_count; ++index) {
            int next = (index + 1) % vertex_count;
            float cross = x[index] * z[next] - x[next] * z[index];
            cross_sum += cross;
            centroid_x_sum += (x[index] + x[next]) * cross;
            centroid_z_sum += (z[index] + z[next]) * cross;
            second_11_sum += (x[index] * x[index] + x[index] * x[next] + x[next] * x[next]) * cross;
            second_22_sum += (z[index] * z[index] + z[index] * z[next] + z[next] * z[next]) * cross;
            second_12_sum += (2.0f * x[index] * z[index] + x[index] * z[next] + x[next] * z[index]
                                 + 2.0f * x[next] * z[next])
                * cross;
        }
        float signed_area = 0.5f * cross_sum;
        float area = fabsf(signed_area);
        if (area <= 1.0e-12f) {
            return;
        }
        float centroid_x = centroid_x_sum / (6.0f * signed_area);
        float centroid_z = centroid_z_sum / (6.0f * signed_area);
        float orientation = signed_area < 0.0f ? -1.0f : 1.0f;
        float raw_second_11 = orientation * second_11_sum / 12.0f;
        float raw_second_22 = orientation * second_22_sum / 12.0f;
        float raw_second_12 = orientation * second_12_sum / 24.0f;
        patch->area = area;
        patch->centroid = T::add(
            plane_origin, T::add(T::scale(tangent_1, centroid_x), T::scale(tangent_2, centroid_z)));
        patch->second_11 = T::max(raw_second_11 - area * centroid_x * centroid_x, 0.0f);
        patch->second_22 = T::max(raw_second_22 - area * centroid_z * centroid_z, 0.0f);
        patch->second_12 = raw_second_12 - area * centroid_x * centroid_z;
    }

    __device__ static __forceinline__ int clip(const ClipVertex* input, int count,
        ClipVertex* output, Vec3 normal, float offset, uint32_t plane_index) {
        if (count <= 0) {
            return 0;
        }
        int output_count = 0;
        ClipVertex previous = input[count - 1];
        float previous_distance = T::dot(previous.point, normal) - offset;
        int previous_inside = previous_distance <= 0.0f;
        for (int index = 0; index < count; ++index) {
            ClipVertex current = input[index];
            float current_distance = T::dot(current.point, normal) - offset;
            int current_inside = current_distance <= 0.0f;
            if (current_inside != previous_inside && output_count < max_clip_vertices) {
                float fraction = previous_distance / (previous_distance - current_distance);
                ClipVertex intersection;
                intersection.point = T::lerp(previous.point, current.point, fraction);
                uint32_t lower =
                    previous.feature < current.feature ? previous.feature : current.feature;
                uint32_t upper =
                    previous.feature < current.feature ? current.feature : previous.feature;
                intersection.feature = 0x80000000u | ((plane_index & 0xffu) << 16)
                    | ((lower & 0xffu) << 8) | (upper & 0xffu);
                output[output_count++] = intersection;
            }
            if (current_inside && output_count < max_clip_vertices) {
                output[output_count++] = current;
            }
            previous = current;
            previous_distance = current_distance;
            previous_inside = current_inside;
        }
        return output_count;
    }

    __device__ static __forceinline__ Result box(const Shape& object_shape, const Body& object_body,
            const Shape* components, const Body* component_bodies, int component_count, Vec3 inward,
            const Options& options, Candidate* candidates, int body_a, int body_b, Manifold* manifold) {
        if (!valid(components, component_count, options) || object_shape.type != T::box_kind)
            return {invalid, 0, 0};
        float margin = options.margin;
        float static_friction = options.static_friction;
        float dynamic_friction = options.dynamic_friction;
        float restitution = options.restitution;
        int candidate_count = 0;
        const Shape* object = &object_shape;
        const Body* object_state = &object_body;
        typename Sat::Obb object_obb = Sat::obb(object);
        Vec3 object_axes[3];
        T::axes(object->pose.rotation, object_axes);
        const Vec3 object_half = object->half_extents;
        const Vec3 frame_origin = components[0].pose.position;
        Vec3 component_axes[3];
        T::axes(components[0].pose.rotation, component_axes);
        float frame_axis_x = T::dot(frame_origin, component_axes[0]);
        float frame_axis_z = T::dot(frame_origin, component_axes[2]);
        float rect_min_x[MaxComponents];
        float rect_max_x[MaxComponents];
        float rect_min_z[MaxComponents];
        float rect_max_z[MaxComponents];
        float support_plane[MaxComponents];
        Vec3 inner_surface[MaxComponents];
        float temporal_plane[MaxComponents];
        float surface_normal_velocity[MaxComponents];
        float component_angular_bound[MaxComponents];
        int component_active[MaxComponents];
        float x_bounds[MaxComponents * 2];
        float z_bounds[MaxComponents * 2];
        int x_bound_count = 0;
        int z_bound_count = 0;
        for (int component = 0; component < component_count; ++component) {
            const Shape* component_shape = &components[component];
            Vec3 rectangle_delta = T::sub(component_shape->pose.position, frame_origin);
            float center_x = T::dot(rectangle_delta, component_axes[0]);
            float center_z = T::dot(rectangle_delta, component_axes[2]);
            rect_min_x[component] = center_x - component_shape->half_extents.x;
            rect_max_x[component] = center_x + component_shape->half_extents.x;
            rect_min_z[component] = center_z - component_shape->half_extents.z;
            rect_max_z[component] = center_z + component_shape->half_extents.z;
            x_bounds[x_bound_count++] = rect_min_x[component];
            x_bounds[x_bound_count++] = rect_max_x[component];
            z_bounds[z_bound_count++] = rect_min_z[component];
            z_bounds[z_bound_count++] = rect_max_z[component];
            inner_surface[component] =
                T::add(component_shape->pose.position, T::scale(inward, component_shape->half_extents.y));
            support_plane[component] = T::dot(inner_surface[component], inward);
            component_active[component] = 0;
            temporal_plane[component] = support_plane[component];
            surface_normal_velocity[component] = 0.0f;
            component_angular_bound[component] = 0.0f;
        }
        for (int index = 1; index < x_bound_count; ++index) {
            float value = x_bounds[index];
            int cursor = index;
            while (cursor > 0 && value < x_bounds[cursor - 1]) {
                x_bounds[cursor] = x_bounds[cursor - 1];
                --cursor;
            }
            x_bounds[cursor] = value;
        }
        for (int index = 1; index < z_bound_count; ++index) {
            float value = z_bounds[index];
            int cursor = index;
            while (cursor > 0 && value < z_bounds[cursor - 1]) {
                z_bounds[cursor] = z_bounds[cursor - 1];
                --cursor;
            }
            z_bounds[cursor] = value;
        }
        int unique_x = 0;
        for (int index = 0; index < x_bound_count; ++index) {
            if (unique_x == 0
                || fabsf(x_bounds[index] - x_bounds[unique_x - 1]) > options.boundary_epsilon) {
                x_bounds[unique_x++] = x_bounds[index];
            }
        }
        int unique_z = 0;
        for (int index = 0; index < z_bound_count; ++index) {
            if (unique_z == 0
                || fabsf(z_bounds[index] - z_bounds[unique_z - 1]) > options.boundary_epsilon) {
                z_bounds[unique_z++] = z_bounds[index];
            }
        }
        for (int component = 0; component < component_count; ++component) {
            const Shape* component_shape = &components[component];
            typename Sat::Query sat_query = Sat::query(object, component_shape, margin);

            Vec3 delta = T::sub(object->pose.position, component_shape->pose.position);
            if (T::dot(delta, inward) < 0.0f) {
                continue;
            }
            Vec3 object_surface = Sat::support(&object_obb, T::scale(inward, -1.0f));
            float face_separation = T::dot(T::sub(object_surface, inner_surface[component]), inward);
            const Body* component_state = &component_bodies[component];
            float angular_bound = T::length(object_state->angular_velocity) * T::radius(object)
                + T::length(component_state->angular_velocity) * T::radius(component_shape);
            Vec3 sat_normal = Solver::normalize(
                sat_query.contact.normal, T::sub(sat_query.contact.point_a, sat_query.contact.point_b));
            Vec3 sat_velocity_a = T::add(object_state->linear_velocity,
                T::cross(object_state->angular_velocity,
                    T::sub(sat_query.contact.point_a, T::position(*object_state))));
            Vec3 sat_velocity_b = T::add(component_state->linear_velocity,
                T::cross(component_state->angular_velocity,
                    T::sub(sat_query.contact.point_b, T::position(*component_state))));
            float sat_projected_separation = sat_query.contact.separation
                + T::dot(T::sub(sat_velocity_a, sat_velocity_b), sat_normal) * options.dt
                - angular_bound * options.dt;
            if (!sat_query.contact.hit && sat_query.contact.separation > margin
                && sat_projected_separation > margin) {
                continue;
            }
            float object_radius_x = T::box_radius(object_axes, object_half, component_axes[0]);
            float object_radius_z = T::box_radius(object_axes, object_half, component_axes[2]);
            float local_x = T::dot(delta, component_axes[0]);
            float local_z = T::dot(delta, component_axes[2]);
            if (fabsf(local_x) > component_shape->half_extents.x + object_radius_x + margin
                || fabsf(local_z) > component_shape->half_extents.z + object_radius_z + margin) {
                continue;
            }

            Vec3 object_velocity = T::add(object_state->linear_velocity,
                T::cross(object_state->angular_velocity,
                    T::sub(object_surface, T::position(*object_state))));
            Vec3 component_velocity = T::add(component_state->linear_velocity,
                T::cross(component_state->angular_velocity,
                    T::sub(inner_surface[component], T::position(*component_state))));
            float normal_velocity = T::dot(T::sub(object_velocity, component_velocity), inward);
            float projected_separation =
                face_separation + (normal_velocity - angular_bound) * options.dt;
            if (face_separation > margin && projected_separation > margin) {
                continue;
            }
            component_active[component] = 1;
            surface_normal_velocity[component] = normal_velocity;
            component_angular_bound[component] = angular_bound;
            temporal_plane[component] =
                support_plane[component] + margin - (normal_velocity - angular_bound) * options.dt;
        }
        Vec3 patch_tangent_1, patch_tangent_2;
        Solver::tangents(inward, &patch_tangent_1, &patch_tangent_2);
        float patch_area_acc[MaxComponents] = {0.0f};
        float patch_first_1[MaxComponents] = {0.0f};
        float patch_first_2[MaxComponents] = {0.0f};
        float patch_raw_11[MaxComponents] = {0.0f};
        float patch_raw_22[MaxComponents] = {0.0f};
        float patch_raw_12[MaxComponents] = {0.0f};
        int candidate_overflow = 0;

        int incident_axis = Sat::aligned_axis(&object_obb, T::scale(inward, -1.0f));
        float incident_sign =
            T::dot(object_obb.axis[incident_axis], T::scale(inward, -1.0f)) < 0.0f ? -1.0f : 1.0f;
        Vec3 incident_face[4];
        Sat::face(&object_obb, incident_axis, incident_sign, incident_face);
        float face_x_offset = frame_axis_x;
        float face_z_offset = frame_axis_z;
        for (int x_cell = 0; x_cell + 1 < unique_x; ++x_cell) {
            float cell_min_x = x_bounds[x_cell];
            float cell_max_x = x_bounds[x_cell + 1];
            if (cell_max_x - cell_min_x <= options.boundary_epsilon) {
                continue;
            }
            for (int z_cell = 0; z_cell + 1 < unique_z; ++z_cell) {
                float cell_min_z = z_bounds[z_cell];
                float cell_max_z = z_bounds[z_cell + 1];
                if (cell_max_z - cell_min_z <= options.boundary_epsilon) {
                    continue;
                }
                float cell_center_x = 0.5f * (cell_min_x + cell_max_x);
                float cell_center_z = 0.5f * (cell_min_z + cell_max_z);
                int owner = -1;
                float best_support = 0.0f;
                for (int component = 0; component < component_count; ++component) {
                    if (cell_center_x < rect_min_x[component] - options.boundary_epsilon
                        || cell_center_x > rect_max_x[component] + options.boundary_epsilon
                        || cell_center_z < rect_min_z[component] - options.boundary_epsilon
                        || cell_center_z > rect_max_z[component] + options.boundary_epsilon) {
                        continue;
                    }
                    if (owner < 0 || support_plane[component] > best_support + options.support_tolerance
                        || (fabsf(support_plane[component] - best_support) <= options.support_tolerance
                            && component < owner)) {
                        owner = component;
                        best_support = support_plane[component];
                    }
                }
                if (owner < 0 || !component_active[owner]) {
                    continue;
                }
                ClipVertex clipped[max_clip_vertices];
                ClipVertex scratch[max_clip_vertices];
                for (int vertex = 0; vertex < 4; ++vertex) {
                    clipped[vertex].point = incident_face[vertex];
                    clipped[vertex].feature = (uint32_t)vertex;
                }
                int count = 4;
                int plane_base = 5;
                count = clip(
                    clipped, count, scratch, component_axes[0], face_x_offset + cell_max_x, plane_base);
                count = clip(scratch, count, clipped, T::scale(component_axes[0], -1.0f),
                    -face_x_offset - cell_min_x, plane_base + 1);
                count = clip(
                    clipped, count, scratch, component_axes[2], face_z_offset + cell_max_z, plane_base + 2);
                count = clip(scratch, count, clipped, T::scale(component_axes[2], -1.0f),
                    -face_z_offset - cell_min_z, plane_base + 3);
                count = clip(clipped, count, scratch, inward, temporal_plane[owner], 10 + owner);
                for (int point_index = 0; point_index < count; ++point_index) {
                    clipped[point_index] = scratch[point_index];
                }
                if (count <= 0) {
                    continue;
                }
                Vec3 polygon[max_clip_vertices];
                for (int point_index = 0; point_index < count; ++point_index) {
                    polygon[point_index] = clipped[point_index].point;
                }
                Patch cell_patch;
                measure_patch(polygon, count, inner_surface[owner], inward, patch_tangent_1, patch_tangent_2,
                    &cell_patch);
                if (cell_patch.area > 1.0e-12f) {
                    float centroid_1 =
                        T::dot(T::sub(cell_patch.centroid, frame_origin), patch_tangent_1);
                    float centroid_2 =
                        T::dot(T::sub(cell_patch.centroid, frame_origin), patch_tangent_2);
                    patch_area_acc[owner] += cell_patch.area;
                    patch_first_1[owner] += cell_patch.area * centroid_1;
                    patch_first_2[owner] += cell_patch.area * centroid_2;
                    patch_raw_11[owner] +=
                        cell_patch.second_11 + cell_patch.area * centroid_1 * centroid_1;
                    patch_raw_22[owner] +=
                        cell_patch.second_22 + cell_patch.area * centroid_2 * centroid_2;
                    patch_raw_12[owner] +=
                        cell_patch.second_12 + cell_patch.area * centroid_1 * centroid_2;
                }
                for (int point_index = 0; point_index < count; ++point_index) {
                    Vec3 point_a = clipped[point_index].point;
                    float point_x = T::dot(T::sub(point_a, frame_origin), component_axes[0]);
                    float point_z = T::dot(T::sub(point_a, frame_origin), component_axes[2]);
                    int point_owner = -1;
                    float point_support = 0.0f;
                    for (int component = 0; component < component_count; ++component) {
                        if (point_x < rect_min_x[component] - options.boundary_epsilon
                            || point_x > rect_max_x[component] + options.boundary_epsilon
                            || point_z < rect_min_z[component] - options.boundary_epsilon
                            || point_z > rect_max_z[component] + options.boundary_epsilon) {
                            continue;
                        }
                        if (point_owner < 0
                            || support_plane[component] > point_support + options.support_tolerance
                            || (fabsf(support_plane[component] - point_support)
                                    <= options.support_tolerance
                                && component < point_owner)) {
                            point_owner = component;
                            point_support = support_plane[component];
                        }
                    }
                    if (point_owner != owner) {
                        continue;
                    }
                    float separation = T::dot(T::sub(point_a, inner_surface[owner]), inward);
                    float projected = separation
                        + (surface_normal_velocity[owner] - component_angular_bound[owner]) * options.dt;
                    if (separation > margin + options.boundary_epsilon
                        && projected > margin + options.boundary_epsilon) {
                        continue;
                    }
                    if (candidate_count >= MaxCandidates) {
                        candidate_overflow = 1;
                        continue;
                    }
                    Vec3 point_b = T::sub(point_a, T::scale(inward, separation));
                    Candidate candidate;
                    memset(&candidate, 0, sizeof(candidate));
                    candidate.contact.hit = 1;
                    candidate.contact.iterations = 15;
                    candidate.contact.normal = inward;
                    candidate.contact.point_a = point_a;
                    candidate.contact.point_b = point_b;
                    candidate.contact.separation = separation;
                    candidate.feature = F::box(options.feature_namespace, owner, x_cell, z_cell,
                        clipped[point_index].feature);
                    candidate.patch_group = options.group_base + (uint32_t)(owner + 1);
                    candidates[candidate_count++] = candidate;
                }
            }
        }
        if (candidate_overflow) {
            return {capacity, 0, 0};
        }
        Patch component_patch[MaxComponents];
        for (int component = 0; component < component_count; ++component) {
            memset(&component_patch[component], 0, sizeof(component_patch[component]));
            if (patch_area_acc[component] <= 1.0e-12f) {
                continue;
            }
            float centroid_1 = patch_first_1[component] / patch_area_acc[component];
            float centroid_2 = patch_first_2[component] / patch_area_acc[component];
            component_patch[component].area = patch_area_acc[component];
            component_patch[component].centroid = T::add(frame_origin,
                T::add(T::scale(patch_tangent_1, centroid_1), T::scale(patch_tangent_2, centroid_2)));
            component_patch[component].second_11 =
                T::max(patch_raw_11[component] - patch_area_acc[component] * centroid_1 * centroid_1, 0.0f);
            component_patch[component].second_22 =
                T::max(patch_raw_22[component] - patch_area_acc[component] * centroid_2 * centroid_2, 0.0f);
            component_patch[component].second_12 =
                patch_raw_12[component] - patch_area_acc[component] * centroid_1 * centroid_2;
        }
        for (int index = 0; index < candidate_count; ++index) {
            int component = (int)(candidates[index].patch_group - options.group_base - 1);
            candidates[index].patch = component_patch[component];
        }
        if (candidate_count <= 0) {
            return {no_contact, 0, 0};
        }
        int unique_count = 0;
        for (int index = 0; index < candidate_count; ++index) {
            Vec3 point = Solver::midpoint(&candidates[index]);
            int duplicate = 0;
            for (int previous = 0; previous < unique_count; ++previous) {
                Vec3 delta = T::sub(point, Solver::midpoint(&candidates[previous]));
                if (T::dot(delta, delta) <= 1.0e-12f) {
                    duplicate = 1;
                    break;
                }
            }
            if (!duplicate) {
                candidates[unique_count++] = candidates[index];
            }
        }
        candidate_count = unique_count;
        if (candidate_count <= 0) {
            return {no_contact, 0, 0};
        }
        int selected[Solver::max_points];
        int selected_count = 0;
        int deepest = 0;
        for (int index = 1; index < candidate_count; ++index) {
            if (candidates[index].contact.separation < candidates[deepest].contact.separation
                || (candidates[index].contact.separation == candidates[deepest].contact.separation
                    && candidates[index].feature < candidates[deepest].feature)) {
                deepest = index;
            }
        }
        selected[selected_count++] = deepest;
        while (selected_count < Solver::max_points && selected_count < candidate_count) {
            int best = -1;
            float best_score = -1.0f;
            for (int index = 0; index < candidate_count; ++index) {
                int already = 0;
                for (int slot = 0; slot < selected_count; ++slot) {
                    if (selected[slot] == index) {
                        already = 1;
                    }
                }
                if (already) {
                    continue;
                }
                Vec3 point = Solver::midpoint(&candidates[index]);
                float score = 3.402823466e+38f;
                for (int slot = 0; slot < selected_count; ++slot) {
                    Vec3 other = Solver::midpoint(&candidates[selected[slot]]);
                    Vec3 delta = T::sub(point, other);
                    score = T::min(score, T::dot(delta, delta));
                }
                if (best < 0 || score > best_score
                    || (score == best_score && candidates[index].feature < candidates[best].feature)) {
                    best = index;
                    best_score = score;
                }
            }
            if (best < 0) {
                break;
            }
            selected[selected_count++] = best;
        }
        Candidate reduced[Solver::max_points];
        for (int index = 0; index < selected_count; ++index) {
            reduced[index] = candidates[selected[index]];
        }
        int made_count = Solver::manifold(body_a, body_b, reduced, selected_count, margin, static_friction,
            dynamic_friction, restitution, manifold);
        int selected_manifold_count = made_count;
        if (selected_manifold_count <= 0) {
            return {no_contact, 0, 0};
        }
        uint32_t component_mask = 0;
        float patch_area = 0.0f;
        Vec3 patch_centroid_sum = T::v3(0, 0, 0);
        for (int point = 0; point < selected_manifold_count; ++point) {
            component_mask |= 1u << (manifold->points[point].patch_group - options.group_base - 1);
            const Patch* patch = &manifold->points[point].patch;
            int first_group = 1;
            for (int previous = 0; previous < point; ++previous) {
                if (manifold->points[previous].patch_group == manifold->points[point].patch_group) {
                    first_group = 0;
                    break;
                }
            }
            if (!first_group) {
                continue;
            }
            patch_area += patch->area;
            patch_centroid_sum = T::add(patch_centroid_sum, T::scale(patch->centroid, patch->area));
        }
        if (patch_area > 1.0e-12f) {
            manifold->patch_area = patch_area;
            manifold->patch_centroid = T::scale(patch_centroid_sum, 1.0f / patch_area);
            float second_11 = 0.0f;
            float second_22 = 0.0f;
            float second_12 = 0.0f;
            for (int point = 0; point < selected_manifold_count; ++point) {
                const Patch* patch = &manifold->points[point].patch;
                int first_group = 1;
                for (int previous = 0; previous < point; ++previous) {
                    if (manifold->points[previous].patch_group == manifold->points[point].patch_group) {
                        first_group = 0;
                        break;
                    }
                }
                if (!first_group) {
                    continue;
                }
                if (patch->area <= 0.0f) {
                    continue;
                }
                Vec3 offset = T::sub(patch->centroid, manifold->patch_centroid);
                float offset_1 = T::dot(offset, manifold->tangent_1);
                float offset_2 = T::dot(offset, manifold->tangent_2);
                second_11 += patch->second_11 + patch->area * offset_1 * offset_1;
                second_22 += patch->second_22 + patch->area * offset_2 * offset_2;
                second_12 += patch->second_12 + patch->area * offset_1 * offset_2;
            }
            manifold->patch_second_11 = T::max(second_11, 0.0f);
            manifold->patch_second_22 = T::max(second_22, 0.0f);
            manifold->patch_second_12 = second_12;
            manifold->patch_second_moment = manifold->patch_second_11 + manifold->patch_second_22;
        } else {
            manifold->patch_area = 0.0f;
            manifold->patch_centroid = T::v3(0, 0, 0);
            manifold->patch_second_11 = 0.0f;
            manifold->patch_second_22 = 0.0f;
            manifold->patch_second_12 = 0.0f;
            manifold->patch_second_moment = 0.0f;
        }
        float effective_radius = patch_area > 1.0e-12f
            ? sqrtf(T::max(manifold->patch_second_moment / patch_area, 0.0f))
            : 0.0f;
        manifold->torsional_radius = effective_radius;
        return {contact, selected_manifold_count, component_mask};
    }

    __device__ static __forceinline__ Result sphere(const Shape& object_shape, const Body& object_body,
            const Shape* components, const Body* component_bodies, int component_count, Vec3 inward,
            const Options& options, int body_a, int body_b, Manifold* manifold) {
        if (!valid(components, component_count, options) || object_shape.type != T::sphere_kind)
            return {invalid, 0, 0};
        Candidate best;
        memset(&best, 0, sizeof(best));
        float best_separation = 3.402823466e+38f;
        int best_component = -1;
        for (int component = 0; component < component_count; ++component) {
            typename Sat::Query query =
                Sat::query(&object_shape, &components[component], options.margin);
            if (T::dot(query.contact.normal, inward) < options.min_normal_alignment) {
                continue;
            }
            const Body* sphere_body = &object_body;
            const Body* component_body = &component_bodies[component];
            Vec3 sphere_velocity = T::add(sphere_body->linear_velocity,
                T::cross(sphere_body->angular_velocity, T::sub(query.contact.point_a, T::position(*sphere_body))));
            Vec3 component_velocity = T::add(component_body->linear_velocity,
                T::cross(component_body->angular_velocity,
                    T::sub(query.contact.point_b, T::position(*component_body))));
            float projected = query.contact.separation
                + T::dot(T::sub(sphere_velocity, component_velocity), query.contact.normal) * options.dt;
            if (!query.contact.hit && projected > options.margin) {
                continue;
            }
            if (query.contact.separation < best_separation) {
                best_separation = query.contact.separation;
                best.contact = query.contact;
                best.contact.hit = 1;
                best.feature = F::sphere(options.feature_namespace, component);
                best_component = component;
            }
        }
        if (best_component < 0) {
            return {no_contact, 0, 0};
        }
        int count = Solver::manifold(body_a, body_b + best_component, &best, 1, options.margin, options.static_friction,
            options.dynamic_friction, options.restitution, manifold);
        if (count <= 0) {
            return {no_contact, 0, 0};
        }
        return {contact, count, 1u << best_component};
    }

};
using PfCompoundContact=PfCompoundContactT<>;
