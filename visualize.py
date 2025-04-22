from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.plotting import figure, show
from bokeh.palettes import Turbo256

import numpy as np
import sys
path = sys.argv[1]

data = np.load(path+'.npy', allow_pickle=True).item()
costs = data['costs']
scores = data['scores']

sorted_costs = np.sort(costs)
aoc = np.max(scores) * np.cumsum(sorted_costs) / np.sum(costs)

# Create a ColumnDataSource that includes the 'order' for each point
source = ColumnDataSource(data=dict(
    x=costs,
    y=scores,
    order=list(range(len(scores)))  # index/order for each point
))

curve = ColumnDataSource(data=dict(
    x=sorted_costs,
    y=aoc,
    order=list(range(len(scores)))  # index/order for each point
))

# Define a color mapper across the range of point indices
mapper = LinearColorMapper(
    palette=Turbo256,
    low=0,
    high=len(scores)
)

# Set up the figure
p = figure(title='Synthetic Hyperparam Test', 
           x_axis_label='Cost', 
           y_axis_label='Score')

# Use the 'order' field for color -> mapped by 'mapper'
p.scatter(x='x', 
          y='y', 
          color={'field': 'order', 'transform': mapper}, 
          size=10, 
          source=source)

p.line(x='x', 
       y='y', 
       color='purple',
       source=curve)

show(p)


