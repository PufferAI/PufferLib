from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import pytest

import pufferlib
import pufferlib.emulation
from pufferlib.pytorch import NativeDType, nativize_dtype, nativize_tensor


# TODO: align=True for dtype
@pytest.mark.parametrize(
    "observation_dtype,emulated_dtype,expected",
    [
        (
            np.dtype((np.uint8, (4,)), align=True),
            np.dtype(
                [
                    ("x", np.uint8, (4,)),
                ],
                align=True,
            ),
            {"x": (torch.uint8, (4,), 0, 4)},
        ),
        (
            np.dtype((np.uint8, (4, 5)), align=True),
            np.dtype(
                [
                    ("x", np.uint8, (4, 5)),
                ],
                align=True,
            ),
            {"x": (torch.uint8, (4, 5), 0, 20)},
        ),
        (
            np.dtype((np.uint8, (4,)), align=True),
            np.dtype([("x", np.uint32, (1,))], align=True),
            {"x": (torch.uint32, (1,), 0, 4)},
        ),
        (
            np.dtype((np.uint8, (12,)), align=True),
            np.dtype([("foo", np.int32, (1,)), ("bar", np.int32, (2,))], align=True),
            {"foo": (torch.int32, (1,), 0, 4), "bar": (torch.int32, (2,), 4, 8)},
        ),
        (
            np.dtype((np.uint8, (16,)), align=True),
            np.dtype(
                [
                    ("foo", np.int32, (1,)),
                    ("bar", [("a", np.int32, (2,)), ("b", np.int32, (1,))]),
                ],
                align=True,
            ),
            {
                "foo": (torch.int32, (1,), 0, 4),
                "bar": {
                    "a": (torch.int32, (2,), 4, 8),
                    "b": (torch.int32, (1,), 12, 4),
                },
            },
        ),
        (
            np.dtype((np.float32, (4,)), align=True),
            np.dtype(
                [
                    ("foo", np.float32, (1,)),
                    ("bar", [("a", np.float32, (2,)), ("b", np.float32, (1,))]),
                ],
                align=True,
            ),
            {
                "foo": (torch.float32, (1,), 0, 1),
                "bar": {
                    "a": (torch.float32, (2,), 1, 2),
                    "b": (torch.float32, (1,), 3, 1),
                },
            },
        ),
        (
            np.dtype((np.int32, (4,)), align=True),
            np.dtype(
                [
                    ("foo", np.int32, (1,)),
                    (
                        "bar",
                        [
                            ("a", [("y", np.int32, (1,)), ("z", np.int32, (1,))]),
                            ("b", np.int32, (1,)),
                        ],
                    ),
                ],
                align=True,
            ),
            {
                "foo": (torch.int32, (1,), 0, 1),
                "bar": {
                    "a": {
                        "y": (torch.int32, (1,), 1, 1),
                        "z": (torch.int32, (1,), 2, 1),
                    },
                    "b": (torch.int32, (1,), 3, 1),
                },
            },
        ),
        (
            np.dtype((np.uint8, (84,)), align=True),
            np.dtype(
                [
                    ("xx", np.float32, (1, 2)),
                    ("yy", [("aa", np.uint8, (7, 7)), ("bb", np.int32, (2, 3))],),
                ],
                align=True,
            ),
            {
                "xx": (torch.float32, (1, 2), 0, 8),
                "yy": {
                    "aa": (torch.uint8, (7, 7), 8, 49),
                    "bb": (torch.int32, (2, 3), 60, 24),
                },
            },
        ),
    ],
)
def test_nativize_dtype(
    observation_dtype: np.array, emulated_dtype: np.array, expected: NativeDType
):
    assert expected == nativize_dtype(
        pufferlib.namespace(
            observation_dtype=observation_dtype,
            emulated_observation_dtype=emulated_dtype,
        )
    )


@pytest.mark.parametrize(
    "space,sample_dtype",
    [
        (
            gym.spaces.Dict(
                {
                    "x": gym.spaces.Box(-1.0, 1.0, (1, 2), dtype=np.float32),
                    "y": gym.spaces.Dict(
                        {
                            "a": gym.spaces.Box(0, 255, (7, 7), dtype=np.uint8),
                            "b": gym.spaces.Box(-1024, 1024, (2, 3), dtype=np.int32),
                        }
                    ),
                }
            ),
            np.dtype(np.uint8),
        ),
        (
            gym.spaces.Dict(
                {
                    "xx": gym.spaces.Box(-1.0, 1.0, (1, 2), dtype=np.float32),
                    "yy": gym.spaces.Box(-1.0, 1.0, (4, 5), dtype=np.float32),
                }
            ),
            np.dtype(np.float32),
        ),
        (
            gym.spaces.Dict(
                {
                    "screen": gym.spaces.Box(0, 255, (18, 20), dtype=np.uint8),
                }
            ),
            np.dtype(np.uint8),
        ),
    ],
)
def test_nativize_tensor(space: gym.spaces.Space, sample_dtype: np.dtype):
    emulated_dtype = pufferlib.emulation.dtype_from_space(space)
    observation_space, observation_dtype = (
        pufferlib.emulation.emulate_observation_space(space)
    )
    native_dtype = nativize_dtype(
        pufferlib.namespace(
            observation_dtype=sample_dtype,
            emulated_observation_dtype=emulated_dtype,
        )
    )
    flat = np.zeros(observation_space.shape, dtype=observation_space.dtype).view(
        observation_dtype
    )
    structured = space.sample()
    pufferlib.emulation.emulate(flat, structured)

    def flatten(inp: Any | Dict[str, Any]) -> List[Any | Tuple[str, Any]]:
        result = []

        for k, v in inp.items():
            if isinstance(v, dict):
                result.extend(flatten(v))
            elif isinstance(v, np.ndarray):
                result.append((k, v))
            elif isinstance(v, torch.Tensor):
                result.append((k, v.numpy()))
            else:
                raise
        return result

    observation = torch.tensor(flat.view(observation_space.dtype)).unsqueeze(0)
    nativized_tensor = nativize_tensor(observation, native_dtype)
    assert all(
        nx == ny and np.all(vx == vy)
        for (nx, vx), (ny, vy) in zip(flatten(nativized_tensor), flatten(structured))
    )
    explain_out = torch._dynamo.explain(nativize_tensor)(observation, native_dtype)
    assert len(explain_out.break_reasons) == 0
