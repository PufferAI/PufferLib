#pragma once

static Color racing_gate_color(int index, const RacingCarFrame *frame) {
    if (index == frame->next_gate) return YELLOW;
    if (frame->lap_active && index > 0 && (frame->next_gate == 0 || index < frame->next_gate))
        return GREEN;
    return index == 0 ? WHITE : SKYBLUE;
}

static void racing_gates_draw(const RacingGate *gates, int count, const RacingCarFrame *frame) {
    // The counting test accepts the chassis centre within these edges and +/-2 metres vertically.
    rlDrawRenderBatchActive();
    rlDisableDepthMask();
    rlDisableBackfaceCulling();
    for (int i = 0; i < count; i++) {
        const RacingGate &g = gates[i];
        Color color = racing_gate_color(i, frame);
        Vector3 left = {g.left.x, g.left.y, g.left.z};
        Vector3 right = {g.right.x, g.right.y, g.right.z};
        Vector3 a = Vector3Add(left, (Vector3){0, -2, 0});
        Vector3 b = Vector3Add(right, (Vector3){0, -2, 0});
        Vector3 c = Vector3Add(right, (Vector3){0, 2, 0});
        Vector3 d = Vector3Add(left, (Vector3){0, 2, 0});
        DrawTriangle3D(a, b, c, Fade(color, i == frame->next_gate ? 0.22f : 0.08f));
        DrawTriangle3D(a, c, d, Fade(color, i == frame->next_gate ? 0.22f : 0.08f));
        DrawLine3D(a, b, color);
        DrawLine3D(b, c, color);
        DrawLine3D(c, d, color);
        DrawLine3D(d, a, color);
        DrawLine3D(Vector3Add(left, (Vector3){0, 0.04f, 0}),
            Vector3Add(right, (Vector3){0, 0.04f, 0}), color);
        Vector3 center = Vector3Add(Vector3Scale(Vector3Add(left, right), 0.5f), (Vector3){0, 0.12f, 0});
        Vector3 forward = Vector3Normalize((Vector3){g.forward.x, g.forward.y, g.forward.z});
        Vector3 tip = Vector3Add(center, Vector3Scale(forward, 4));
        Vector3 side = {forward.z, 0, -forward.x};
        Vector3 tail = Vector3Subtract(tip, Vector3Scale(forward, 1.2f));
        DrawLine3D(center, tip, color);
        DrawLine3D(tip, Vector3Add(tail, side), color);
        DrawLine3D(tip, Vector3Subtract(tail, side), color);
    }
    rlDrawRenderBatchActive();
    rlEnableBackfaceCulling();
    rlEnableDepthMask();
}
