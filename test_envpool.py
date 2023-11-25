from pdb import set_trace as T
import numpy as np
import time

import gymnasium

import pufferlib
from pufferlib.vectorization import Serial, Multiprocessing, Ray


# This is about 1 second on a good CPU core. It is quite difficult to
# find good sources of a 1 second delay without using a timer that can swap
# on sleep
WORK_ITERATIONS = 150_000_000


class PerformanceEnv:
    def __init__(self, delay_mean, delay_std):
        np.random.seed(time.time_ns() % 2**32)

        self.observation_space = gymnasium.spaces.Box(
            low=-2**20, high=2**20,
            shape=(1,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(2)
        self.observation = self.observation_space.sample()

        self.delay_mean = delay_mean
        self.delay_std = delay_std


    def reset(self, seed=None):
        return self.observation, {}

    def step(self, action):
        start = time.process_time()
        idx = 0
        target_time = self.delay_mean + self.delay_std*np.random.randn()
        while time.process_time() - start < target_time:
            idx += 1

        return self.observation, 0, False, False, {}

    def close(self):
        pass


def test_performance(vectorization, workers, envs_per_worker,
        delay_mean, delay_std, batch_size=None, timeout=1):
    def make_env():
        return pufferlib.emulation.GymnasiumPufferEnv(
            env_creator=PerformanceEnv, env_args=(delay_mean, delay_std))

    if batch_size is None:
        batch_size = workers * envs_per_worker

    actions = np.array([make_env().action_space.sample() for _ in range(batch_size)])

    if vectorization in (Serial, Multiprocessing, 'SyncMultiprocessing', 'SyncRay', Ray):
        synchronous = False
        if vectorization == 'SyncMultiprocessing':
            vectorization = Multiprocessing
            synchronous = True
        if vectorization == 'SyncRay':
            vectorization = Ray
            synchronous = True

        envs = vectorization(
            make_env,
            num_workers=workers,
            envs_per_worker=envs_per_worker,
            batch_size=batch_size,
            synchronous=synchronous,
        )
    else:
        envs = vectorization([make_env for _ in range(workers)])

    envs.reset()
    num_steps = 0
    start = time.time()
    while time.time() - start < timeout:
        obs = envs.step(actions)[0]
        num_steps += obs.shape[0]

    end = time.time()
    envs.close()

    return num_steps, end - start


def sweep_performance_tests():
    backends = (
        gymnasium.vector.SyncVectorEnv, Serial,
        gymnasium.vector.AsyncVectorEnv, 'SyncMultiprocessing',
        Multiprocessing, 
        'SyncRay', Ray,
    )
    results = {}
    delay_means = (1e-2, 1e-2, 1e-3, 1e-3, 1e-4, 1e-4)
    delay_stds = (1e-3, 1e-2, 1e-4, 1e-3, 1e-5, 1e-4)
    for mean, std in zip(delay_means, delay_stds):
        results[(mean, std)] = {}
        print('Environment delay: ', mean, std)
        for workers in (1, 6, 24, 96, 192):
            resul = {}
            results[(mean, std)][workers] = resul
            print('\t', workers)
            for vec in backends:
                res = {}
                if type(vec) != str:
                    name = vec.__name__
                else:
                    name = vec

                resul[name] = res
                print(2*'\t', name)

                for envs_per_worker in (1, 2, 4):
                    batch_sizes=[workers * envs_per_worker]
                    if vec in (Multiprocessing, Ray) and workers != 1:
                        batch_sizes.append(workers * envs_per_worker // 2)
                        batch_sizes.append(workers * envs_per_worker // 3)

                    for batch in batch_sizes:
                        steps, duration = test_performance(
                                vec, workers, envs_per_worker, mean, std, batch)

                        res[(envs_per_worker, batch)] = (steps, duration)

                        print('SPS, envs/worker, batch size: ',
                              steps / duration, envs_per_worker, batch)

    #np.save('envpool_results.npy', results, allow_pickle=True)

def plot_performance_tests():
    data = np.load('envpool_results.npy', allow_pickle=True).item()
    n_envs = len(data)

    inner_data = list(data.items())[0][1]
    n_cores, cores = len(inner_data), list(inner_data.keys())

    inner_inner_data = list(inner_data.items())[0][1]
    n_backends, backends = len(inner_inner_data), list(inner_inner_data.keys())

    from matplotlib import pyplot as plt
    import matplotlib.colors as mcolors

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 5))  # Adjust size as needed
    #plt.yscale('log')

    # Bar settings
    bar_width = 0.15
    group_width = n_backends * bar_width * n_cores
    index = np.arange(n_envs) * (group_width + bar_width * 2)  # Adding more space between environments

    # Grayscale colors for backends
    grayscale_colors = np.linspace(0.4, 1, n_backends)
    backend_colors = [str(g) for g in grayscale_colors]

    # Hue colors for cores
    hue_colors = 255*plt.cm.hsv(np.linspace(0, 0.6, n_cores))[:, :3]
    bars_data = []

    grayscale_colors = np.linspace(0.4, 1, n_cores)
    hue_colors = 255*plt.cm.hsv(np.linspace(0, 0.6, n_backends))[:, :3]

    import plotly.graph_objects as go
    import dash
    import dash_core_components as dcc
    import dash_html_components as html

    # Plotting the bars
    pos = 0

    x_labels = [f'{mean}±{std}' for mean, std in data.keys()]
    tick_vals = np.linspace(0, bar_width*n_envs*n_cores*(n_backends+1), n_envs)

    # Set up layout configuration
    layout = go.Layout(
        title=dict(
            text='Performance of Vectorization Backends on Various Workloads (24 core machine)',
            y=0.9
        ),
        width=2000,# 1000,
        height=500,
        yaxis=dict(title='Speedup over Expected Serial Performance'),
        plot_bgcolor='rgba(6, 26, 26, 1)',  # Dark cyan background
        paper_bgcolor='rgba(6, 26, 26, 1)',
        font=dict(color='rgba(241, 241, 241, 1)'),  # Light text
        barmode='group',
        xaxis = dict(
            title='Test Environment Delays (mean/std) and Process Counts',
            tickmode='array',
            tickvals = tick_vals,
            ticktext = x_labels,
        ),
        legend=dict(
            y=1.20,
            x=0.9,#0.80
        ),
    )

    fig = go.Figure(data=bars_data, layout=layout)
    x = 0
    for env_idx, (mean, std) in enumerate(data):
        env = data[(mean, std)]
        label = ('mean = %.1e, std = %.1e' % (mean, std))
        for workers_idx, workers in enumerate(env):
            runs = env[workers]
            for vec_idx, vec in enumerate(runs):
                results = runs[vec].values()
                best_sps = max(steps / duration for steps, duration in results)
                speedup = best_sps * mean

                color = hue_colors[vec_idx] * grayscale_colors[workers_idx]
                color = f'rgb{tuple(color[:3])}'  # Convert to RGB string
                fig.add_trace(go.Bar(
                    x=[x],
                    y=[speedup],  # Y value
                    marker_color=color,  # Color
                    text=label,
                    showlegend=False,
                ))
                x += bar_width
                label = ''
            x += bar_width
        x += 3*bar_width

    # Create figure with the collected bar data and layout
    for idx, vec in enumerate(backends):
        if vec == 'Serial':
            vec = 'Puffer Serial'
        elif vec == 'SyncMultiprocessing':
            vec = 'Puffer Multiproc.'
        elif vec == 'Multiprocessing':
            vec = 'Puffer Pool'

        color = f'rgb{tuple(hue_colors[idx])}'  # Convert to RGB string
        fig.add_trace(go.Bar(
            x=[None],  # No x value
            y=[None],  # No y value
            name=vec,  # Name for the legend entry
            marker_color=color,  # Transparent color
            showlegend=True,  # Show in legend
        ))

    for idx, core in enumerate(cores):
        color = f'rgb{tuple(3*[grayscale_colors[idx]])}'
        fig.add_trace(go.Bar(
            x=[None],  # No x value
            y=[None],  # No y value
            name=core,  # Name for the legend entry
            marker_color=color,  # Transparent color
            showlegend=True,  # Show in legend
        ))

    # Save the figure to a file
    fig.write_image('../docker/envpool_sps.png', scale=3)


if __name__ == '__main__':
    #sweep_performance_tests()
    plot_performance_tests()
