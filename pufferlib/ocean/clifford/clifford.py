import gymnasium
import numpy as np

import pufferlib
from . import binding
from .reference import _build_actions, _is_symplectic

_GATE_KIND = {
    "h": 0,
    "s": 1,
    "v": 2,
    "hs": 3,
    "hv": 4,
    "cz": 5,
}
_REWARD_MODE = {
    "gate_cost": 0,
    "hamming_left": 1,
}


class Clifford(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        n_qubits=6,
        difficulty=10,
        max_steps=200,
        single_qubit_cost=0.01,
        goal_bonus=0.0,
        reward_mode="gate_cost",
        hamming_left_scale=0.5,
        use_reset_pool=True,
        log_interval=128,
        buf=None,
        seed=0,
        render_mode=None,
    ):
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if 2 * n_qubits > 64:
            raise ValueError("native Clifford env requires 2*n_qubits <= 64")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if difficulty < 0:
            raise ValueError("difficulty must be non-negative")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if goal_bonus < 0.0:
            raise ValueError("goal_bonus must be non-negative")
        if reward_mode not in _REWARD_MODE:
            raise ValueError(f"reward_mode must be one of {sorted(_REWARD_MODE)}")
        if hamming_left_scale < 0.0:
            raise ValueError("hamming_left_scale must be non-negative")

        self.n_qubits = int(n_qubits)
        self.dim = 2 * self.n_qubits
        self._actions = _build_actions(self.n_qubits)
        self.log_interval = int(log_interval)
        self.render_mode = render_mode
        self.num_agents = int(num_envs)
        self.agents_per_batch = self.num_agents

        obs_size = self.dim * self.dim
        self.single_observation_space = gymnasium.spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.uint8,
        )
        self.single_action_space = gymnasium.spaces.Discrete(len(self._actions))
        super().__init__(buf)

        gate_kinds = np.empty(len(self._actions), dtype=np.int32)
        action_q0 = np.empty(len(self._actions), dtype=np.int32)
        action_q1 = np.empty(len(self._actions), dtype=np.int32)
        for idx, (gate, q0, q1) in enumerate(self._actions):
            gate_kinds[idx] = _GATE_KIND[gate]
            action_q0[idx] = q0
            action_q1[idx] = q1

        self.tick = 0
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            self.num_agents,
            int(seed),
            n_qubits=self.n_qubits,
            difficulty=float(difficulty),
            max_steps=int(max_steps),
            gate_kinds=gate_kinds,
            action_q0=action_q0,
            action_q1=action_q1,
            single_qubit_cost=float(single_qubit_cost),
            goal_bonus=float(goal_bonus),
            reward_mode=int(_REWARD_MODE[reward_mode]),
            hamming_left_scale=float(hamming_left_scale),
            use_reset_pool=bool(use_reset_pool),
        )

    def reset(self, seed=0):
        self.tick = 0
        binding.vec_reset(self.c_envs, -1 if seed is None else int(seed))
        return self.observations, []

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int32)
        if actions.shape != (self.num_agents,):
            raise ValueError(f"Expected actions with shape ({self.num_agents},), got {actions.shape}")

        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.log_interval == 0:
            metrics = binding.vec_log(self.c_envs)
            if metrics:
                info.append(metrics)

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def close(self):
        if getattr(self, "c_envs", None) is not None:
            binding.vec_close(self.c_envs)
            self.c_envs = None

    def flush_logs(self):
        metrics = binding.vec_log(self.c_envs)
        return metrics or None

    def set_difficulty(self, difficulty):
        difficulty = float(difficulty)
        if difficulty < 0.0:
            raise ValueError("difficulty must be non-negative")
        binding.vec_set_difficulty(self.c_envs, difficulty)

    def set_max_steps(self, max_steps):
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        binding.vec_set_max_steps(self.c_envs, max_steps)

    def set_matrix(self, matrix, env_index=0):
        matrix_u8 = np.ascontiguousarray(np.asarray(matrix, dtype=np.uint8))
        expected_shape = (self.dim, self.dim)
        if matrix_u8.shape != expected_shape:
            raise ValueError(f"Expected matrix shape {expected_shape}, got {matrix_u8.shape}")
        if not _is_symplectic(matrix_u8):
            raise ValueError("matrix must be symplectic")
        binding.vec_set_matrix(self.c_envs, int(env_index), matrix_u8)


def test_performance(
    timeout=10,
    atn_cache=1024,
    num_envs=2048,
    n_qubits=6,
    difficulty=10,
    max_steps=200,
    use_reset_pool=True,
):
    env = Clifford(
        num_envs=num_envs,
        n_qubits=n_qubits,
        difficulty=difficulty,
        max_steps=max_steps,
        use_reset_pool=use_reset_pool,
    )
    env.reset()
    tick = 0

    actions = np.random.randint(
        0,
        env.single_action_space.n,
        (atn_cache, num_envs),
        dtype=np.int32,
    )

    import time
    start = time.time()
    while time.time() - start < timeout:
        env.step(actions[tick % atn_cache])
        tick += 1

    sps = num_envs * tick / (time.time() - start)
    print(
        f"Clifford SPS: {sps:,.0f} "
        f"(num_envs={num_envs}, n_qubits={n_qubits}, difficulty={difficulty}, "
        f"max_steps={max_steps}, use_reset_pool={use_reset_pool})"
    )
    env.close()


if __name__ == "__main__":
    test_performance()
