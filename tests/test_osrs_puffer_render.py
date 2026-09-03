import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    "ocean/osrs_inferno/osrs_inferno.h",
    "ocean/osrs_colosseum/osrs_colosseum.h",
    "ocean/osrs_zulrah/osrs_zulrah.h",
    "ocean/osrs_pvp/osrs_pvp.h",
)


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_puf_render_owns_renderer_lifecycle(tmp_path: Path, wrapper: str) -> None:
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    raylib_include = next(REPOSITORY_ROOT.glob("raylib-*/include"), None)
    if raylib_include is None:
        pytest.skip("raylib headers unavailable")

    source = tmp_path / "render_lifecycle.c"
    executable = tmp_path / "render_lifecycle"
    source.write_text(
        f"""
#include <assert.h>
#include <stddef.h>

static int create_calls;
static int draw_calls;
static int destroy_calls;
static int renderer_token;

void* osrs_puffer_render_create(const void* encounter_def, void* encounter_state, void* encounter_context) {{
    assert(encounter_def != NULL);
    assert(encounter_state != NULL);
    assert(encounter_context != NULL);
    create_calls++;
    return &renderer_token;
}}

void osrs_puffer_render_draw(void* renderer) {{
    assert(renderer == &renderer_token);
    draw_calls++;
}}

void osrs_puffer_render_destroy(void* renderer) {{
    assert(renderer == &renderer_token);
    destroy_calls++;
}}

#define OSRS_PUFFER_RENDER
#include "{wrapper}"

int main(void) {{
    Env env = {{0}};

    puf_render(&env);
    puf_render(&env);

    assert(create_calls == 1);
    assert(draw_calls == 2);

    puf_close(&env);
    assert(destroy_calls == 1);
    Env unrendered_env = {{0}};
    puf_close(&unrendered_env);
    assert(destroy_calls == 1);

    Log log = {{0}};
    Dict metrics = {{0}};
    puf_log(&log, &metrics);
    assert(dict_find(&metrics, "perf") != NULL);
    dict_clear(&metrics);
    return 0;
}}
"""
    )

    subprocess.run(
        [
            compiler,
            "-D_POSIX_C_SOURCE=200809L",
            "-std=c11",
            "-I",
            str(REPOSITORY_ROOT),
            "-I",
            str(REPOSITORY_ROOT / "src"),
            "-I",
            str(raylib_include),
            str(source),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
    )
    subprocess.run([executable], check=True, cwd=REPOSITORY_ROOT)
