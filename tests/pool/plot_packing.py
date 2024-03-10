import plotly.graph_objects as go
import numpy as np

# Parameters
n_bars = 24
mu = 0.002
std = 0.002

background = '#061a1a'
forground = '#f1f1f1'

# Sampling from the normal distribution
bar_heights = mu + np.clip(np.random.normal(mu, std, n_bars), 0, np.inf)

# Creating the bar chart
fig = go.Figure(go.Bar(
    x=[i for i in range(n_bars)],
    y=bar_heights,
    marker_line_width=0,
    marker_color=forground,
))

# Updating the layout
fig.update_layout({
    'plot_bgcolor': background,
    'paper_bgcolor': background,
    'showlegend': False,
    'xaxis': {'visible': False},
    'yaxis': {'visible': False, 'range': [0, max(bar_heights)]},
    'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
    'height': 400,
    'width': 800,
    'bargap': 0.0,
    'bargroupgap': 0.0,
})


fig.show()
fig.write_image('../docker/env_variance.png', scale=3)
