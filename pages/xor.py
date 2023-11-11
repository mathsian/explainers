from dash import html, dcc, callback, Output, Input, register_page, ALL, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly

register_page(__name__)

# data set
xx, yy = np.mgrid[-1:1.1:0.1, -1:1.1:0.1]
df = pd.DataFrame({'x': xx.flatten(), 'y': yy.flatten()})
df['class'] = df.eval('(x < 0 and y < 0) or (x >= 0 and y >= 0)').replace({False: 'red', True: 'blue'})

# graph area
graph_area = dcc.Graph(id='graph')

# input layer
input_layer = ['x', 'y']
input_layer_panel = dbc.Container([
    dbc.Row([dbc.Col(dcc.Slider(id={'layer': 'input', 'label': label}, min=-1, max=1, value=0))])
    for label in input_layer
])

# first layer sliders
first_layer = ['u_x0', 'u_x1', 'u_y0', 'u_y1']
first_layer_panel = dbc.Container([
    dbc.Row([dbc.Col(dcc.Slider(id={'layer': 'first', 'label': label}, min=-2, max=2, value=0))])
    for label in first_layer
])

# second layer sliders
second_layer = ['w0', 'w1']
second_layer_panel = dbc.Container([
    dbc.Row([dbc.Col(dcc.Slider(id={'layer': 'second', 'label': label}, min=-2, max=2, value=0))])
    for label in second_layer
])

# output layer
output_layer_panel = dbc.Container([
    dbc.Row([dbc.Col(html.Div(id='z'))])
])


def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(input_layer_panel),
            dbc.Col(first_layer_panel),
            dbc.Col(second_layer_panel),
            dbc.Col(output_layer_panel)
        ]),
        dbc.Row([
            dbc.Col([graph_area])
        ]),
 #       dbc.Row([dbc.Col([html.Div(id='text-area')])])
    ])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feedforward(x, y, ux0, ux1, uy0, uy1, w0, w1):
    v0 = x * ux0 + y * uy0
    v1 = x * ux1 + y * uy1

    z = sigmoid(w0 * v0 + w1 * v1)

    return z


@callback(
    [Output('z', 'children'),
     Output('graph', 'figure'),
     # Output('text-area', 'children')
     ],
    [
        Input({'layer': 'input', 'label': ALL}, 'value'),
        Input({'layer': 'first', 'label': ALL}, 'value'),
        Input({'layer': 'second', 'label': ALL}, 'value'),
    ]
)
def update_output_panel(input_values, first_values, second_values):
    x, y = input_values
    ux0, ux1, uy0, uy1 = first_values
    w0, w1 = second_values
    z = feedforward(*input_values, *first_values, *second_values)

    df['v0'] = df['x'].mul(ux0).add(df['y'].mul(uy0)).apply(sigmoid)
    df['v1'] = df['x'].mul(ux1).add(df['y'].mul(uy1)).apply(sigmoid)
    df['z'] = df['v0'].mul(w0).add(df['v1'].mul(w1)).apply(sigmoid)

    df['prediction'] = df.eval('z >= 0.5').replace({False: 'circle-open', True: 'circle'})

    fig = go.Figure(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(color=df['class'],
                    symbol=df['prediction'],
                    size=5)
    ))

    # table = dbc.Table.from_dataframe(df)

    return 0 if z < 0.5 else 1, fig
