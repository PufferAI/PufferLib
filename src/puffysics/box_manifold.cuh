#pragma once

#include "collision_box.cuh"

// A clipped point counts as in contact at this separation. It matches the
// tolerance pf_box_query_axis already uses to treat two SAT features as the
// same feature, so the narrow phase and the patch agree on what "touching"
// means, and a point lying exactly on the reference plane is not dropped by
// float rounding around zero.
#define PF_BOX_CONTACT_MARGIN 1.0e-4f
// Clips the convex polygon `input` to the half-space dot(p, axis) <= offset,
// or to >= offset when `keep_positive` is set. A convex quad clipped by four
// half-spaces reaches at most eight vertices, which is what bounds the
// `output_count < 8` guards below.
__device__ static inline int pf_clip_plane(
        const PfVec3* input, int count, PfVec3* output, PfVec3 axis,
        float offset, bool keep_positive) {
    if (count <= 0) {
        return 0;
    }
    int output_count = 0;
    PfVec3 previous = input[count - 1];
    float previous_distance = pf_dot(previous, axis) - offset;
    bool previous_inside = keep_positive
        ? previous_distance >= 0.0f : previous_distance <= 0.0f;
    for (int index = 0; index < count; ++index) {
        PfVec3 current = input[index];
        float current_distance = pf_dot(current, axis) - offset;
        bool current_inside = keep_positive
            ? current_distance >= 0.0f : current_distance <= 0.0f;
        if (previous_inside != current_inside && output_count < 8) {
            float denominator = previous_distance - current_distance;
            float t = fabsf(denominator) > 1.0e-20f
                ? previous_distance / denominator : 0.0f;
            output[output_count++] = pf_add(previous,
                pf_scale(pf_sub(current, previous), t));
        }
        if (current_inside && output_count < 8) {
            output[output_count++] = current;
        }
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }
    return output_count;
}

// Reduces a clipped patch to PF_MAX_MANIFOLD_POINTS vertices. The deepest
// point is always retained because it is the one the position solver
// corrects, and the remaining slots go to the vertices that best span the
// patch. Taking the first four vertices instead silently drops the deepest
// one whenever clipping runs long, and four points clustered on one side
// under-resolve a rotation about the normal. Strict comparisons with a
// forward scan make the choice deterministic under ties.
__device__ static inline int pf_box_patch_reduce(
        const PfVec3* polygon, const float* separations, int count,
        int* keep) {
    int deepest = 0;
    for (int index = 1; index < count; ++index) {
        if (separations[index] < separations[deepest]) {
            deepest = index;
        }
    }
    keep[0] = deepest;
    int kept = 1;
    while (kept < PF_MAX_MANIFOLD_POINTS && kept < count) {
        int best = -1;
        float best_value = -1.0f;
        for (int index = 0; index < count; ++index) {
            bool taken = false;
            for (int slot = 0; slot < kept; ++slot) {
                taken = taken || keep[slot] == index;
            }
            if (taken) {
                continue;
            }
            PfVec3 edge = pf_sub(polygon[index], polygon[keep[0]]);
            float value = 0.0f;
            if (kept == 1) {
                value = pf_length_squared(edge);
            } else if (kept == 2) {
                value = pf_length_squared(pf_cross(edge,
                    pf_sub(polygon[keep[1]], polygon[keep[0]])));
            } else {
                value = pf_length(pf_cross(
                    pf_sub(polygon[keep[1]], polygon[keep[0]]), edge))
                    + pf_length(pf_cross(edge,
                        pf_sub(polygon[keep[2]], polygon[keep[0]])));
            }
            if (value > best_value) {
                best_value = value;
                best = index;
            }
        }
        if (best < 0) {
            break;
        }
        keep[kept++] = best;
    }
    return kept;
}

__device__ static inline bool pf_box_manifold(
        const PfBody* a, const PfBody* b, const PfVec3 axes_a[3],
        const PfVec3 axes_b[3], const PfBoxQuery* query, PfManifold* out) {
    if (query->separation > 0.0f) {
        return false;
    }
    if (query->feature >= 6) {
        // Edge contact. The manifold normal points from B toward A, so A's
        // contacting edge is the one extremal against it and B's the one
        // extremal along it. The support point of the whole box is only the
        // midpoint of that edge, which coincides with the closest point only
        // when the two edges cross in projection; on a partial overlap the
        // closest points have to be solved on the finite segments.
        int feature = query->feature - 6;
        int edge_axis_a = feature / 3;
        int edge_axis_b = feature % 3;
        PfVec3 center_a = a->position;
        PfVec3 center_b = b->position;
        for (int axis = 0; axis < 3; ++axis) {
            if (axis == edge_axis_a) {
                continue;
            }
            float sign = pf_dot(axes_a[axis], query->normal) >= 0.0f
                ? -1.0f : 1.0f;
            center_a = pf_add(center_a,
                pf_scale(axes_a[axis], sign * pf_box_half(a, axis)));
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (axis == edge_axis_b) {
                continue;
            }
            float sign = pf_dot(axes_b[axis], query->normal) >= 0.0f
                ? 1.0f : -1.0f;
            center_b = pf_add(center_b,
                pf_scale(axes_b[axis], sign * pf_box_half(b, axis)));
        }
        float half_a = pf_box_half(a, edge_axis_a);
        float half_b = pf_box_half(b, edge_axis_b);
        PfVec3 start_a = pf_add(center_a,
            pf_scale(axes_a[edge_axis_a], -half_a));
        PfVec3 start_b = pf_add(center_b,
            pf_scale(axes_b[edge_axis_b], -half_b));
        PfVec3 direction_a = pf_scale(axes_a[edge_axis_a], 2.0f * half_a);
        PfVec3 direction_b = pf_scale(axes_b[edge_axis_b], 2.0f * half_b);
        float length_a = pf_length_squared(direction_a);
        float length_b = pf_length_squared(direction_b);
        float along_b = pf_dot(direction_b, pf_sub(start_a, start_b));
        float along_a = pf_dot(direction_a, pf_sub(start_a, start_b));
        float s = 0.0f;
        float t = 0.0f;
        if (length_a > 1.0e-20f && length_b > 1.0e-20f) {
            float between = pf_dot(direction_a, direction_b);
            float denominator = length_a * length_b - between * between;
            s = denominator > 1.0e-20f ? fmaxf(0.0f, fminf(1.0f,
                (between * along_b - along_a * length_b) / denominator)) : 0.0f;
            t = (between * s + along_b) / length_b;
            if (t < 0.0f) {
                t = 0.0f;
                s = fmaxf(0.0f, fminf(1.0f, -along_a / length_a));
            } else if (t > 1.0f) {
                t = 1.0f;
                s = fmaxf(0.0f, fminf(1.0f, (between - along_a) / length_a));
            }
        } else if (length_a > 1.0e-20f) {
            s = fmaxf(0.0f, fminf(1.0f, -along_a / length_a));
        }
        PfVec3 point_a = pf_add(start_a, pf_scale(direction_a, s));
        PfVec3 point_b = pf_add(start_b, pf_scale(direction_b, t));
        out->normal = query->normal;
        out->point_count = 1;
        out->points[0].point_a = point_a;
        out->points[0].point_b = point_b;
        // Measured from the two points rather than from the SAT axis, so the
        // reported depth is the real segment distance on a partial overlap.
        out->points[0].separation =
            pf_dot(pf_sub(point_a, point_b), query->normal);
        return true;
    }
    // Face contact. The manifold normal points from B toward A, so B lies on
    // the -normal side of A and the two touched faces face each other: the
    // reference face's outward normal is -normal when A is the reference, and
    // +normal when B is. That asymmetry is the whole reason the sign cannot be
    // derived from the normal alone. Taking -normal unconditionally - which is
    // what this did - picks A's far face correctly but B's FAR face, and a
    // far-face reference then measures the incident point against the plane
    // on the wrong side of the body entirely, reporting a penetration of
    // roughly the sum of both half extents and building the patch against the
    // wrong surface.
    bool reference_is_a = query->feature < 3;
    const PfBody* reference = reference_is_a ? a : b;
    const PfBody* incident = reference_is_a ? b : a;
    const PfVec3* reference_axes = reference_is_a ? axes_a : axes_b;
    const PfVec3* incident_axes = reference_is_a ? axes_b : axes_a;
    int reference_axis = reference_is_a ? query->feature : query->feature - 3;
    float alignment = pf_dot(reference_axes[reference_axis], query->normal);
    float reference_sign = alignment >= 0.0f
        ? (reference_is_a ? -1.0f : 1.0f)
        : (reference_is_a ? 1.0f : -1.0f);
    PfVec3 outward = pf_scale(reference_axes[reference_axis], reference_sign);
    PfVec3 face_center = pf_add(reference->position,
        pf_scale(outward, pf_box_half(reference, reference_axis)));
    // The incident face is the incident box face whose outward normal is most
    // opposed to the reference outward normal, which is the one whose own axis
    // is most aligned with it. Its two tangent axes are the incident box's
    // own remaining axes, so each half extent belongs to the axis it scales.
    int incident_axis = 0;
    float aligned = fabsf(pf_dot(incident_axes[0], outward));
    for (int axis = 1; axis < 3; ++axis) {
        float candidate = fabsf(pf_dot(incident_axes[axis], outward));
        if (candidate > aligned) {
            aligned = candidate;
            incident_axis = axis;
        }
    }
    float incident_sign = pf_dot(incident_axes[incident_axis], outward) >= 0.0f
        ? -1.0f : 1.0f;
    PfVec3 incident_center = pf_add(incident->position,
        pf_scale(incident_axes[incident_axis],
            incident_sign * pf_box_half(incident, incident_axis)));
    int incident_tangent_1 = (incident_axis + 1) % 3;
    int incident_tangent_2 = (incident_axis + 2) % 3;
    PfVec3 tangent_1 = incident_axes[incident_tangent_1];
    PfVec3 tangent_2 = incident_axes[incident_tangent_2];
    float tangent_1_half = pf_box_half(incident, incident_tangent_1);
    float tangent_2_half = pf_box_half(incident, incident_tangent_2);
    PfVec3 polygon[8];
    polygon[0] = pf_add(incident_center,
        pf_add(pf_scale(tangent_1, -tangent_1_half),
            pf_scale(tangent_2, -tangent_2_half)));
    polygon[1] = pf_add(incident_center,
        pf_add(pf_scale(tangent_1, tangent_1_half),
            pf_scale(tangent_2, -tangent_2_half)));
    polygon[2] = pf_add(incident_center,
        pf_add(pf_scale(tangent_1, tangent_1_half),
            pf_scale(tangent_2, tangent_2_half)));
    polygon[3] = pf_add(incident_center,
        pf_add(pf_scale(tangent_1, -tangent_1_half),
            pf_scale(tangent_2, tangent_2_half)));
    int count = 4;
    PfVec3 scratch[8];
    for (int tangent = 0; tangent < 2; ++tangent) {
        int axis = (reference_axis + 1 + tangent) % 3;
        float half = pf_box_half(reference, axis);
        float offset = pf_dot(face_center, reference_axes[axis]);
        count = pf_clip_plane(polygon, count, scratch, reference_axes[axis],
            offset + half, false);
        if (count == 0) {
            return false;
        }
        count = pf_clip_plane(scratch, count, polygon, reference_axes[axis],
            offset - half, true);
        if (count == 0) {
            return false;
        }
    }
    if (count <= 0) {
        return false;
    }
    // Separation is measured from the reference face plane, and the incident
    // point stays on the incident body: the reference point is the one
    // projected along the normal, which is what makes the pair satisfy
    // separation == dot(point_a - point_b, normal) whatever the body order is.
    float separations[8];
    for (int index = 0; index < count; ++index) {
        separations[index] = pf_dot(pf_sub(polygon[index], face_center), outward);
    }
    // Drop the points that are not in contact. A tilted incident face straddles
    // the reference plane, so the clipped patch legitimately contains corners
    // sitting ABOVE it; keeping them drags the patch centroid off the region
    // that is actually touching, and the solver collapses a patch to one
    // centroid impulse, so the impulse then lands where nothing touches.
    // The margin is the same 1e-4 the SAT already uses to decide that two
    // features are the same feature, so "in contact" means the same thing in
    // the narrow phase and here; a point exactly on the reference plane
    // carries float rounding around zero and would otherwise be dropped at
    // random. The deepest point always survives because the query separation
    // is non-positive, so a contact never loses its whole patch.
    PfVec3 contact_points[8];
    float contact_separations[8];
    int contact_count = 0;
    for (int index = 0; index < count; ++index) {
        if (separations[index] > PF_BOX_CONTACT_MARGIN) {
            continue;
        }
        contact_points[contact_count] = polygon[index];
        contact_separations[contact_count] = separations[index];
        ++contact_count;
    }
    if (contact_count == 0) {
        return false;
    }
    int keep[PF_MAX_MANIFOLD_POINTS];
    int kept = contact_count;
    if (contact_count > PF_MAX_MANIFOLD_POINTS) {
        kept = pf_box_patch_reduce(contact_points, contact_separations,
            contact_count, keep);
    } else {
        for (int index = 0; index < contact_count; ++index) {
            keep[index] = index;
        }
    }
    out->normal = query->normal;
    out->point_count = kept;
    for (int slot = 0; slot < kept; ++slot) {
        int index = keep[slot];
        float separation = contact_separations[index];
        out->points[slot].separation = separation;
        if (reference_is_a) {
            out->points[slot].point_b = contact_points[index];
            out->points[slot].point_a = pf_add(contact_points[index],
                pf_scale(query->normal, separation));
        } else {
            out->points[slot].point_a = contact_points[index];
            out->points[slot].point_b = pf_sub(contact_points[index],
                pf_scale(query->normal, separation));
        }
    }
    return true;
}
