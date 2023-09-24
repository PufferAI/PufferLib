from pufferlib import namespace


def create_precheck(env_creator, env_args, env_kwargs):
    if env_args is None:
        env_args = []
    if env_kwargs is None:
        env_kwargs = {}

    assert callable(env_creator)
    assert isinstance(env_args, list)
    #assert isinstance(env_kwargs, dict)

    return env_args, env_kwargs

def init(self,
        env_creator: callable = None,
        env_args: list = [],
        env_kwargs: dict = {},
        n: int = 1,
        ) -> None:
    env_args, env_kwargs = create_precheck(env_creator, env_args, env_kwargs)
    return namespace(self,
        envs = [env_creator(*env_args, **env_kwargs) for _ in range(n)],
        preallocated_obs = None,
    )

def put(state, *args, **kwargs):
    for e in state.envs:
        e.put(*args, **kwargs)
    
def get(state, *args, **kwargs):
    return [e.get(*args, **kwargs) for e in state.envs]

def close(state):
    for env in state.envs:
        env.close()

def profile(state):
    return [e.timers for e in state.envs]
