from dash import html, dcc, callback, Output, Input, register_page, ALL, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

np.seterr(all='ignore')

register_page(__name__)

xx1 = np.random.normal(loc=0.3, scale=0.1, size=100)
yy1 = np.random.normal(loc=0.4, scale=0.1, size=100)
xx2 = np.random.normal(loc=0.7, scale=0.1, size=100)
yy2 = np.random.normal(loc=0.5, scale=0.1, size=100)

df1 = pd.DataFrame({'x': xx1, 'y': yy1})
df1['class'] = "circle"

df2 = pd.DataFrame({'x': xx2, 'y': yy2})
df2['class'] = "square"

df = pd.concat([df1, df2], axis='index')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def layout():
    b_slider = dcc.Slider(id='perceptron b', min=-1, max=1, value=0)
    wx_slider = dcc.Slider(id='perceptron wx', min=-1, max=1, value=1)
    wy_slider = dcc.Slider(id='perceptron wy', min=-1, max=1, value=-1)
    graph = dcc.Graph(id='perceptron graph', responsive=True)
    lyt = dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Row(dbc.Col(b_slider)),
                dbc.Row(dbc.Col(wx_slider)),
                dbc.Row(dbc.Col(wy_slider)),
            ], width=3),
            dbc.Col(graph, width=9)
        ])
    ], fluid=True)
    return lyt


@callback(
    Output('perceptron graph', 'figure'),
    [
        Input('perceptron b', 'value'),
        Input('perceptron wx', 'value'),
        Input('perceptron wy', 'value'),
    ]
)
def update_graph(b, wx, wy):

    df['z'] = df.eval('x * @wx +y * @wy + @b').apply(sigmoid)
    # df['prediction'] = (df['z'] > 0.5).replace({False: 'circle-open', True: 'circle'})

    fig = go.Figure(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(color=df['z'],
                    cmin=0,
                    cmax=1,
                    colorscale='icefire',
                    colorbar=dict(title='Probability'),
                    symbol=df['class'],
                    line=dict(color='midnightblue', width=2),
                    size=15)
    ))

    x_0 = (0, np.divide(-b, wy, ))
    x_1 = (1, np.divide(-(wx + b), wy))
    y_0 = (np.divide(-b, wx), 0)
    y_1 = (np.divide(-(wy + b), wx), 1)
    pts = [x_0, x_1, y_0, y_1]
    box_pts = [(x, y) for x, y in pts if 0 <= x <= 1 and 0 <= y <= 1]
    if len(box_pts) >= 2:
        x0, y0 = box_pts[0]
        x1, y1 = box_pts[1]
        fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color='midnightblue'))

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig
