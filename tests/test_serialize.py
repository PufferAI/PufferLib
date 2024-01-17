import ray

def test_serialize():
    import ray

    ray.init()

    @ray.remote
    def remote_env():
        import griddly, gym
        env = gym.make('GDY-Spiders-v0') 
        return env.reset()

    remote = remote_env.remote()
    ray.get(remote)

if __name__  ==  '__main__' :
    test_serialize()