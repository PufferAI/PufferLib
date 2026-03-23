from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
import glob
import os

FONT_FAMILY = 'Arial'
FONT_SIZE_TITLE = 28
FONT_SIZE_AXIS = 22
FONT_SIZE_TICK = 20
FONT_SIZE_TICK_3D = 14
FONT_SIZE_LEGEND = 18
FONT_COLOR = '#f1f1f1'
PLOT_BG_COLOR = '#061a1a'
PAPER_BG_COLOR = '#061a1a'
LINE_WIDTH = 4
LINE_COLORS = ["#0000b3", "#0010d9", "#0020ff", "#0040ff", "#0060ff", "#0080ff", "#009fff", "#00bfff", "#00ffff"][::-1]
roygbiv = np.random.permutation(['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'])
#roygbiv = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
TITLE_FONT = dict(
    family=FONT_FAMILY,
    size=FONT_SIZE_TITLE,
    color=FONT_COLOR
)
AXIS_FONT = dict(
    family=FONT_FAMILY,
    size=FONT_SIZE_AXIS,
    color=FONT_COLOR
)
TICK_FONT = dict(
    family=FONT_FAMILY,
    size=FONT_SIZE_TICK,
    color=FONT_COLOR
)
GRID_COLOR = '#00f1f1'
TICK_FONT_3D = dict(
    family=FONT_FAMILY,
    size=FONT_SIZE_TICK_3D,
    color=FONT_COLOR
)
LEGEND_FONT = dict(
    family=FONT_FAMILY,
    size=FONT_SIZE_LEGEND,
    color=FONT_COLOR
)
HYPERS = [
    'train/learning_rate',
    'train/ent_coef',
    'train/gamma',
    'train/gae_lambda',
    'train/vtrace_rho_clip',
    'train/vtrace_c_clip',
    'train/clip_coef',
    'train/vf_clip_coef',
    'train/vf_coef',
    'train/max_grad_norm',
    'train/adam_beta1',
    'train/adam_beta2',
    'train/adam_eps',
    'train/prio_alpha',
    'train/prio_beta0',
    'train/bptt_horizon',
    'train/num_minibatches',
    'train/minibatch_size',
    'policy/hidden_size',
    'env/num_envs',
]
ALL_KEYS = [
    'agent_steps',
    'cost',
    'environment/score',
    'environment/perf'
] + HYPERS

SCATTER_COLOR = ['env_name'] + ALL_KEYS

import colorsys
import numpy as np

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex string."""
    return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def generate_distinct_palette(n):
    """
    Generate a palette with n maximally distinct colors across the hue spectrum.
    
    Parameters:
    n (int): Number of colors to generate.
    
    Returns:
    list: List of hex color strings.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    
    # Generate hues evenly spaced across the spectrum (0 to 1)
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = []
    for hue in hues:
        # Use full saturation and value for vivid colors
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(rgb)
    hex_colors = [rgb_to_hex(color) for color in colors]
    return hex_colors

def pareto_idx(steps, costs, scores):
    idxs = []
    for i in range(len(steps)):
        better = [scores[j] >= scores[i] and
            costs[j] < costs[i] and steps[j] < steps[i]
            for j in range(len(scores))]
        if not any(better):
            idxs.append(i)

    return idxs

def build_dataset(dataframe):
    dataset = []
    for hyper in HYPERS:
        dat = dataframe[hyper]
        #mmin = dataframe[f'sweep/{hyper}/min']
        #mmax = dataframe[f'sweep/{hyper}/max']
        #distribution = dataframe[f'sweep/{hyper}/distribution']



def load_sweep_data(path):
    data = {}
    keys = None
    for fpath in glob.glob(path):
        with open(fpath, 'r') as f:
            exp = json.load(f)

        if not data:
            for kk in exp.keys():
                if kk == 'data':
                    for k, v in exp[kk][-1].items():
                        data[k] = []
                else:
                    data[kk] = []

        discard = False
        for kk in list(data.keys()):
            if kk not in exp and kk not in exp['data'][-1]:
                discard = True
                break

        if discard:
            continue

        for kk in list(data.keys()):
            if kk in exp:
                v = exp[kk]
                sweep_key = f'sweep/{kk}/distribution'
                if sweep_key in data and exp[sweep_key] == 'logit_normal':
                    v = 1 - v
                elif kk in ('train/vtrace_rho_clip', 'train/vtrace_c_clip'):
                    v = max(v, 0.1)

                data[kk].append(v)
            else:
                data[kk].append(exp['data'][-1][kk])

    return data

def cached_sweep_load(path, env_name):
    cache_file = os.path.join(path, 'cache.json')
    if not os.path.exists(cache_file):
        data = load_sweep_data(os.path.join(path, '*.json'))
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    with open(cache_file, 'r') as f:
        data = json.load(f)

    steps = data['agent_steps']
    costs = data['cost']
    scores = data['environment/score']

    idxs = pareto_idx(steps, costs, scores)
    
    # Create a DataFrame for this environment
    df_data = {}
    for k in data:
        df_data[k] = [data[k][i] for i in idxs]
    
    # Apply performance cap
    df_data['environment/perf'] = [min(e, 1.0) for e in df_data['environment/perf']]
    
    # Adjust steps by frameskip if present
    if 'env/frameskip' in df_data:
        skip = df_data['env/frameskip']
        df_data['agent_steps'] = [n*m for n, m in zip(df_data['agent_steps'], skip)]
    
    # Add environment name
    df_data['env_name'] = [env_name] * len(idxs)
    
    return pd.DataFrame(df_data)

def compute_tsne():
    dataset = EXPERIMENTS[HYPERS].copy()  # Create a copy to avoid modifying the original

    # Normalize each hyperparameter column using its corresponding min and max columns
    for hyper in HYPERS:
        min_col = f'sweep/{hyper}/min'
        max_col = f'sweep/{hyper}/max'

        mmin = min(EXPERIMENTS[min_col])
        mmax = max(EXPERIMENTS[max_col])

        distribution = EXPERIMENTS[f'sweep/{hyper}/distribution']
        if 'log' in distribution or 'pow2' in distribution:
            mmin = np.log(mmin)
            mmax = np.log(mmax)
            normed = np.log(dataset[hyper])
        else:
            normed = dataset[hyper]

        dataset[hyper] = (normed - mmin) / (mmax - mmin)
        # Normalize: (value - min) / (max - min) for each row

        #dataset[hyper] = (dataset[hyper] - EXPERIMENTS[min_col]) / (EXPERIMENTS[max_col] - EXPERIMENTS[min_col])

    # Filter dataset based on performance threshold
    # Apply TSNE
    from sklearn.manifold import TSNE
    proj = TSNE(n_components=2)
    reduced = proj.fit_transform(dataset)
    EXPERIMENTS['tsne1'] = reduced[:, 0]
    EXPERIMENTS['tsne2'] = reduced[:, 1]

env_names = ['tripletriad', 'grid', 'moba', 'tower_climb', 'tetris', 'breakout', 'pong', 'g2048', 'snake', 'pacman']
env_all = ['all'] + env_names
#env_names = ['grid', 'breakout', 'g2048']
#env_names = ['grid']

roygbiv = generate_distinct_palette(len(env_names))

# Create a list of DataFrames for each environment
dfs = [cached_sweep_load(f'experiments/logs/puffer_{name}', name) for name in env_names]

# Concatenate all DataFrames into a single DataFrame
EXPERIMENTS = pd.concat(dfs, ignore_index=True)
#EXPERIMENTS.set_index('env_name', inplace=True)
compute_tsne()

app = Dash()
app.css.append_css({'external_stylesheets': 'dash.css'})
app.layout = html.Div([
    html.H1('Puffer Constellation', style={'textAlign': 'center'}),
    html.Br(),

    html.Label([
        "X: ",
        dcc.Dropdown(
            id="optimal-dropdown-x",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="cost",
            style={"width": "50%"}
        )
    ]),
    html.Label([
        "Y: ",
        dcc.Dropdown(
            id="optimal-dropdown-y",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="agent_steps",
            style={"width": "50%"}
        )
    ]),
    html.Label([
        "Z: ",
        dcc.Dropdown(
            id="optimal-dropdown-z",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="environment/perf",
            style={"width": "50%"}
        )
    ]),
    dcc.Graph(id='optimal'),
    html.Br(),

    html.Label([
        "Environment: ",
        dcc.Dropdown(
            id="scatter-dropdown-env",
            options=[{"label": key, "value": key} for key in env_all],
            value="all",
            style={"width": "50%"}
        )
    ]),
    html.Br(),
    html.Label([
        "X: ",
        dcc.Dropdown(
            id="scatter-dropdown-x",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="train/learning_rate",
            style={"width": "50%"}
        ),
        dcc.Checklist(
            id="scatter-checkbox-logx",
            options=[{"label": "Log", "value": "log"}],
            value=["log"],
            style={"display": "inline-block", "margin-left": "10px"}
        ),
    ]),
    html.Br(),
    html.Label([
        "Y: ",
        dcc.Dropdown(
            id="scatter-dropdown-y",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="environment/perf",
            style={"width": "50%"}
        ),
        dcc.Checklist(
            id="scatter-checkbox-logy",
            options=[{"label": "Log", "value": "log"}],
            value=[],
            style={"display": "inline-block", "margin-left": "10px"}
        ),
       
    ]),
    html.Br(),
    html.Label([
        "Color: ",
        dcc.Dropdown(
            id="scatter-dropdown-color",
            options=[{"label": key, "value": key} for key in SCATTER_COLOR],
            value="env_name",
            style={"width": "50%"}
        )
    ]),
    html.Br(),
    html.Label([
        "Range 1: ",
        dcc.Dropdown(
            id="scatter-dropdown-range-1",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="agent_steps",
            style={"width": "50%"}
        ),
        dcc.RangeSlider(
            id='scatter-range-1',
            min=0.0,
            max=1.0,
            step=0.05,
            value=[0.0, 0.25]
        ),
    ]),
    html.Br(),
    html.Label([
        "Range 2: ",
        dcc.Dropdown(
            id="scatter-dropdown-range-2",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="cost",
            style={"width": "50%"}
        ),
        dcc.RangeSlider(
            id='scatter-range-2',
            min=0.0,
            max=1.0,
            step=0.05,
            value=[0.0, 0.95]
        ),
    ]),
    dcc.Graph(id='scatter'),
    html.Br(),

    #html.Label([
    #    "X Axis: ",
    #    dcc.Dropdown(
    #        id="hyper-box-x",
    #        options=[{"label": key, "value": key} for key in ['cost', 'agent_steps']],
    #        value="agent_steps",
    #        style={"width": "50%"}
    #    )
    #]),
    #dcc.Graph(id='hyper-box'),

    html.Br(),
    html.Label([
        "Range 1: ",
        dcc.Dropdown(
            id="hyper-dropdown-range-1",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="environment/perf",
            style={"width": "50%"}
        ),
        dcc.RangeSlider(
            id='hyper-range-1',
            min=0.0,
            max=1.0,
            step=0.05,
            value=[0.8, 1.0]
        ),
    ]),
    html.Br(),
    html.Label([
        "Range 2: ",
        dcc.Dropdown(
            id="hyper-dropdown-range-2",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="agent_steps",
            style={"width": "50%"}
        ),
        dcc.RangeSlider(
            id='hyper-range-2',
            min=0.0,
            max=1.0,
            step=0.05,
            value=[0.0, 1.0]
        ),
    ]),
    dcc.Graph(id='hyper'),


    html.Br(),
    html.Label([
        "Range 1: ",
        dcc.Dropdown(
            id="tsnee-dropdown-range-1",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="environment/perf",
            style={"width": "50%"}
        ),
        dcc.RangeSlider(
            id='tsnee-range-1',
            min=0.0,
            max=1.0,
            step=0.05,
            value=[0.5, 1.0]
        ),
    ]),
    html.Br(),
    html.Label([
        "Range 2: ",
        dcc.Dropdown(
            id="tsnee-dropdown-range-2",
            options=[{"label": key, "value": key} for key in ALL_KEYS],
            value="cost",
            style={"width": "50%"}
        ),
        dcc.RangeSlider(
            id='tsnee-range-2',
            min=0.0,
            max=1.0,
            step=0.05,
            value=[0.0, 1.0]
        ),
    ]),
    dcc.Graph(id='tsnee'),

],
style={"width": 1280}
)

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.distance import cdist

# Assuming EXPERIMENTS is your pandas DataFrame, and xkey, ykey, zkey are defined.
# Also assuming percentages for cutoffs, e.g.:
percentage1 = 5.0  # Percentage for XYZ distance threshold relative to plot diagonal in transformed space
percentage2 = 0.5  # Percentage for PCA distance threshold relative to PCA diagonal

@app.callback(
    Output("optimal", "figure"),
    Input("optimal-dropdown-x", "value"),
    Input("optimal-dropdown-y", "value"),
    Input("optimal-dropdown-z", "value")
)
def update_optimal_plot(xkey, ykey, zkey):
    all_x = EXPERIMENTS[xkey].values
    all_y = EXPERIMENTS[ykey].values
    all_z = EXPERIMENTS[zkey].values
    all_pca1 = EXPERIMENTS['tsne1'].values
    all_pca2 = EXPERIMENTS['tsne2'].values
    all_env = EXPERIMENTS['env_name'].values# Handle transformed coordinates for XYZ (accounting for log axes)
    trans_x = np.log10(all_x)  # Assuming all_x > 0
    trans_y = np.log10(all_y)  # Assuming all_y > 0
    trans_z = all_z
    points_trans_xyz = np.column_stack((trans_x, trans_y, trans_z))

    # Compute ranges in transformed space
    range_tx = np.max(trans_x) - np.min(trans_x)
    range_ty = np.max(trans_y) - np.min(trans_y)
    range_tz = np.max(trans_z) - np.min(trans_z)
    diagonal_xyz = np.sqrt(range_tx**2 + range_ty**2 + range_tz**2)
    delta1 = (percentage1 / 100.0) * diagonal_xyz

    # For PCA (assuming linear scales)
    points_pca = np.column_stack((all_pca1, all_pca2))
    range_p1 = np.max(all_pca1) - np.min(all_pca1)
    range_p2 = np.max(all_pca2) - np.min(all_pca2)
    diagonal_pca = np.sqrt(range_p1**2 + range_p2**2)
    delta2 = (percentage2 / 100.0) * diagonal_pca

    # Create the base scatter plot
    f = px.scatter_3d(
        x=all_x,
        y=all_y,
        z=all_z,
        color=all_env,
        log_x=True,
        log_y=True,
        log_z=False,
        color_discrete_sequence=roygbiv
    )

    # Compute pairwise L2 distances in transformed spaces
    dists_xyz = cdist(points_trans_xyz, points_trans_xyz)
    dists_pca = cdist(points_pca, points_pca)

    # Create boolean masks
    xyz_mask = dists_xyz < delta1
    pca_mask = dists_pca < delta2
    # Use boolean array for upper triangle to avoid type mismatch
    triu_mask = np.triu(np.ones_like(dists_xyz, dtype=bool), k=1)

    # Combine masks with boolean operations
    mask = xyz_mask & pca_mask & triu_mask

    # Get indices of valid pairs
    i, j = np.where(mask)

    # Collect line segment coordinates (in original space)
    line_x = []
    line_y = []
    line_z = []
    for k in range(len(i)):
        line_x.extend([all_x[i[k]], all_x[j[k]], None])
        line_y.extend([all_y[i[k]], all_y[j[k]], None])
        line_z.extend([all_z[i[k]], all_z[j[k]], None])

    # Add the lines as a single trace
    if line_x:
        f.add_trace(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode='lines',
                line=dict(color='rgba(255,255,255,0.25)', width=2),
                showlegend=False
            )
        )

    # Show the figure
    f.show()

    layout_dict = {
        'title': dict(text='Pareto', font=TITLE_FONT),
        'showlegend': True,
        'legend': dict(font=LEGEND_FONT),
        'plot_bgcolor': PLOT_BG_COLOR,
        'paper_bgcolor': PAPER_BG_COLOR,
        'width': 1280,
        'height': 720,
        'autosize': False,
        'scene': dict(
            xaxis=dict(
                title=dict(text=xkey, font=AXIS_FONT),
                tickfont=TICK_FONT_3D,
                type='log',
                showgrid=True,
                gridcolor=GRID_COLOR,
                backgroundcolor=PLOT_BG_COLOR,
                zeroline=False
            ),
            yaxis=dict(
                title=dict(text=ykey, font=AXIS_FONT),
                tickfont=TICK_FONT_3D,
                type='log',
                showgrid=True,
                gridcolor=GRID_COLOR,
                backgroundcolor=PLOT_BG_COLOR,
                zeroline=False
            ),
            zaxis=dict(
                title=dict(text=zkey, font=AXIS_FONT),
                tickfont=TICK_FONT_3D,
                type='linear',
                showgrid=True,
                gridcolor=GRID_COLOR,
                backgroundcolor=PLOT_BG_COLOR,
                zeroline=False
            ),
            bgcolor=PLOT_BG_COLOR,
        )
    }
    f.update_layout(**layout_dict)
    return f


@app.callback(
    Output("scatter", "figure"),
    Input("scatter-dropdown-env", "value"),
    Input("scatter-dropdown-x", "value"),
    Input("scatter-checkbox-logx", "value"),
    Input("scatter-dropdown-y", "value"),
    Input("scatter-checkbox-logy", "value"),
    Input("scatter-dropdown-color", "value"),
    Input("scatter-dropdown-range-1", "value"),
    Input("scatter-range-1", "value"),
    Input("scatter-dropdown-range-2", "value"),
    Input("scatter-range-2", "value"),
)
def update_scatter(env, xkey, logx, ykey, logy, zkey, range1_key, range1, range2_key, range2):
    #env_data = EXPERIMENTS.loc[env]
    if env == 'all':
        env_data = EXPERIMENTS
    else:
        env_data = EXPERIMENTS[EXPERIMENTS['env_name'] == env]

    range1_mmin = min(EXPERIMENTS[range1_key])
    range1_mmax = max(EXPERIMENTS[range1_key])
    norm_range1 = (EXPERIMENTS[range1_key] - range1_mmin) / (range1_mmax - range1_mmin)

    range2_mmin = min(EXPERIMENTS[range2_key])
    range2_mmax = max(EXPERIMENTS[range2_key])
    norm_range2 = (EXPERIMENTS[range2_key] - range2_mmin) / (range2_mmax - range2_mmin)

    mask = (norm_range1 >= range1[0]) & (norm_range1 <= range1[1]) & (norm_range2 >= range2[0]) & (norm_range2 <= range2[1])

    env_data = env_data[mask]

    x = env_data[xkey]
    y = env_data[ykey]
    z = env_data[zkey]

    if zkey == 'env_name':
        f = px.scatter(x=x, y=y, color=z, color_discrete_sequence=roygbiv)
    else:
        mmin = min(z)
        mmax = max(z)
        thresh = np.geomspace(mmin, mmax, 8)
        all_fx = []
        all_fy = []
        bin_label = []
        for j in range(7):
            filter = (thresh[j] < z) & (z < thresh[j+1])
            if filter.sum() <= 2:
                continue

            fx = x[filter]
            fy = y[filter]
            all_fx += fx.tolist()
            all_fy += fy.tolist()
            bin_label += [str(thresh[j])] * len(fx)

        f = px.scatter(x=all_fx, y=all_fy, color=bin_label, color_discrete_sequence=roygbiv)

    f.update_traces(marker_size=10)
    layout_dict = {
        'title': dict(text='Experiments', font=TITLE_FONT),
        'showlegend': True,
        'legend': dict(font=LEGEND_FONT),
        'plot_bgcolor': PLOT_BG_COLOR,
        'paper_bgcolor': PAPER_BG_COLOR,
        'width': 1280,
        'height': 720,
        'autosize': False,
        'xaxis': dict(
            title=dict(text=xkey, font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False,
            type='log' if 'log' in logx else 'linear',
        ),
        'yaxis': dict(
            title=dict(text=ykey, font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False,
            type='log' if 'log' in logy else 'linear',
        )
    }
    f.update_layout(**layout_dict)
    return f

@app.callback(
    Output("hyper-box", "figure"),
    Input("hyper-box-x", "value")
)
def update_hyper_box(x):
    buckets = 4
    env_data = {}
    for env in env_names:
        #data = EXPERIMENTS.loc[env]
        data = EXPERIMENTS[EXPERIMENTS['env_name'] == env]
        steps = data['agent_steps']
        costs = data['cost']
        scores = data['environment/score']
        x_data = costs if x == 'cost' else steps
        hyper_data = {}
        env_data[env] = {'x': x_data, 'hypers': hyper_data}
        for h in HYPERS:
            hyper_data[h] = data[h]
    all_x = [x for env in env_data for x in env_data[env]['x']]
    x_min, x_max = min(all_x), max(all_x)
    bucket_edges = np.linspace(x_min, x_max, buckets + 1)
    bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2
    heatmap_data = np.zeros((len(HYPERS), buckets))
    for i, hyper in enumerate(HYPERS):
        for j in range(buckets):
            bucket_means = []
            for env in env_data:
                if hyper not in env_data[env]['hypers']:
                    continue
                x_vals = np.array(env_data[env]['x'])
                hyper_vals = np.array(env_data[env]['hypers'][hyper])
                idxs = (x_vals >= bucket_edges[j]) & (x_vals < bucket_edges[j+1])
                if np.any(idxs):
                    bucket_means.append(np.mean(hyper_vals[idxs]))
            heatmap_data[i, j] = np.mean(bucket_means) if bucket_means else np.nan
    heatmap_data = np.log(heatmap_data)
    heatmap_data -= heatmap_data[:, 0, None] # Normalize
    f = px.imshow(heatmap_data, x=bucket_centers, y=HYPERS, color_continuous_scale='Viridis', zmin=np.nanmin(heatmap_data), zmax=np.nanmax(heatmap_data), labels=dict(color="Value"))
    layout_dict = {
        'title': dict(text="Hyperparameter Drift", font=TITLE_FONT),
        'showlegend': True,
        'legend': dict(font=LEGEND_FONT),
        'plot_bgcolor': PLOT_BG_COLOR,
        'paper_bgcolor': PAPER_BG_COLOR,
        'width': 1280,
        'height': 720,
        'autosize': False,
        'xaxis': dict(
            title=dict(text=x.capitalize(), font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False
        ),
        'yaxis': dict(
            title=dict(text="Hyperparameters", font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False
        )
    }
    f.update_layout(**layout_dict)
    return f

@app.callback(
    Output("hyper", "figure"),
    Input("hyper-dropdown-range-1", "value"),
    Input("hyper-range-1", "value"),
    Input("hyper-dropdown-range-2", "value"),
    Input("hyper-range-2", "value"),
)
def update_hyper_plot(xkey, range1, ykey, range2):
    # Initialize figure
    f = go.Figure()
    f.update_layout(
        title=dict(text='Hyperparameter Stable Range', font=TITLE_FONT),
        xaxis=dict(title=dict(text='Value', font=AXIS_FONT), tickfont=TICK_FONT),
        yaxis=dict(title=dict(text='Hyper', font=AXIS_FONT), tickfont=TICK_FONT),
        showlegend=True,
        legend=dict(font=LEGEND_FONT),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=PAPER_BG_COLOR,
        width=1280,
        height=720,
        autosize=False,
        xaxis_type='log',
        barmode='overlay',  # Overlay bars instead of stacking
    )
    f.update_xaxes(showgrid=False)
    f.update_yaxes(showgrid=False)

    range1_mmin = min(EXPERIMENTS[xkey])
    range1_mmax = max(EXPERIMENTS[xkey])
    norm_x = (EXPERIMENTS[xkey] - range1_mmin) / (range1_mmax - range1_mmin)
    range2_mmin = min(EXPERIMENTS[ykey])
    range2_mmax = max(EXPERIMENTS[ykey])
    norm_y = (EXPERIMENTS[ykey] - range2_mmin) / (range2_mmax - range2_mmin)
    mask = (norm_x >= range1[0]) & (norm_x <= range1[1]) & (norm_y >= range2[0]) & (norm_y <= range2[1])
    filtered = EXPERIMENTS[mask]

    for i, env in enumerate(env_names):
        #env_data = EXPERIMENTS.loc[env]
        env_data = filtered[filtered['env_name'] == env]
        if len(env_data) < 2:
            continue

        steps = env_data['agent_steps']
        costs = env_data['cost']
        scores = env_data['environment/score']

        max_score = max(scores)
        max_steps = max(steps)
        n = len(scores)

       
        for k, hyper in enumerate(HYPERS):
            y = env_data[hyper]

            ymin = min(y)
            ymax = max(y)
            f.add_trace(
                go.Bar(
                    x=[ymax - ymin],
                    y=[hyper],  # Hyperparameter as x-axis
                    base=ymin,
                    showlegend=False,
                    marker_color='#00f1f1',
                    opacity=0.25,
                    width=1.0,
                    orientation='h'
                )
            )

    return f


@app.callback(
    Output("tsnee", "figure"),
    Input("tsnee-dropdown-range-1", "value"),
    Input("tsnee-range-1", "value"),
    Input("tsnee-dropdown-range-2", "value"),
    Input("tsnee-range-2", "value"),
)
def update_pca_plot(xkey, range1, ykey, range2):
    # Initialize figure
    f = go.Figure()
    f.update_layout(
        title=dict(text='Hyperparameter Stable Range', font=TITLE_FONT),
        xaxis=dict(title=dict(text='Value', font=AXIS_FONT), tickfont=TICK_FONT),
        yaxis=dict(title=dict(text='Hyper', font=AXIS_FONT), tickfont=TICK_FONT),
        showlegend=True,
        legend=dict(font=LEGEND_FONT),
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=PAPER_BG_COLOR,
        width=1280,
        height=720,
        autosize=False,
        xaxis_type='log',
        barmode='overlay',  # Overlay bars instead of stacking
    )
    f.update_xaxes(showgrid=False)
    f.update_yaxes(showgrid=False)

    range1_mmin = min(EXPERIMENTS[xkey])
    range1_mmax = max(EXPERIMENTS[xkey])
    norm_x = (EXPERIMENTS[xkey] - range1_mmin) / (range1_mmax - range1_mmin)
    range2_mmin = min(EXPERIMENTS[ykey])
    range2_mmax = max(EXPERIMENTS[ykey])
    norm_y = (EXPERIMENTS[ykey] - range2_mmin) / (range2_mmax - range2_mmin)
    mask = (norm_x >= range1[0]) & (norm_x <= range1[1]) & (norm_y >= range2[0]) & (norm_y <= range2[1])
    filtered = EXPERIMENTS[mask]

    f = px.scatter(
        x=filtered['tsne1'],
        y=filtered['tsne2'],
        color=filtered['env_name'],
        color_discrete_sequence=roygbiv
    )
 
    f.update_traces(marker_size=10)
    layout_dict = {
        'title': dict(text='Experiments', font=TITLE_FONT),
        'showlegend': True,
        'legend': dict(font=LEGEND_FONT),
        'plot_bgcolor': PLOT_BG_COLOR,
        'paper_bgcolor': PAPER_BG_COLOR,
        'width': 1280,
        'height': 720,
        'autosize': False,
        'xaxis': dict(
            title=dict(text='TSNE-1', font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False
        ),
        'yaxis': dict(
            title=dict(text='TSNE-2', font=AXIS_FONT),
            tickfont=TICK_FONT,
            showgrid=False
        )
    }
    f.update_layout(**layout_dict)
    return f


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
