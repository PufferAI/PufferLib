from pdb import set_trace as T
from pufferlib.environments import atari


def test_atari_reset():
    '''Common way to bug the wrappers can be detected
    by checking that the environment properly resets
    after hitting 0 lives'''
    env = atari.make_env('BreakoutNoFrameskip-v4', 4)

    obs, info = env.reset()
    prev_lives = 5

    lives = []
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)

        if info['lives'] != prev_lives:
            lives.append(i)
            prev_lives = info['lives']

        if terminal or truncated:
            obs = env.reset()

    assert len(lives) > 10

if __name__ == '__main__':
    test_atari_reset()
