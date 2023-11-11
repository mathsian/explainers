from dash import html, dcc, callback, Output, Input, register_page
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from os import path

HAWKS_CSV = path.join(path.dirname(__file__), '../data/Hawks.csv')

hawks_data = pd.read_csv(HAWKS_CSV).dropna(subset=['Wing', 'Weight'])

hawks_data['RT'] = (hawks_data['Species'] == 'RT').map({True: 'RT', False: 'Not RT'})

register_page(__name__)

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1(children='Hawk classification', style={'textAlign': 'center'})
            ])
        ]),
    dbc.Row([
        dbc.Col([
            html.H2("Wing length distribution"),
            dcc.Graph(id={'page': 'hawks_one', 'type': 'graph', 'name': 'wing-graph'}),
            html.Br(),
            dcc.Slider(0, hawks_data['Wing'].max(), value=hawks_data['Wing'].mean().round(), id={'page': 'hawks_one', 'type': 'slider', 'name': 'wing-slider'}),
            html.Br(),
            html.H2("Weight distribution"),
            dcc.Graph(id={'page': 'hawks_one', 'type': 'graph', 'name': 'weight-graph'}),
            html.Br(),
            dcc.Slider(0, hawks_data['Weight'].max(), value=hawks_data['Weight'].mean().round(), id={'page': 'hawks_one', 'type': 'slider', 'name': 'weight-slider'}),
            ], width=8),
        dbc.Col([
            html.H2("Confusion Matrix"),
            html.Div(id={'page': 'hawks_one', 'type': 'div', 'name': 'confusion-matrix-div'})
            ],
            width=4)
        ])
    ])

@callback(
        [
        Output({'page': 'hawks_one', 'type': 'graph', 'name': 'wing-graph'}, 'figure'),
        Output({'page': 'hawks_one', 'type': 'graph', 'name': 'weight-graph'}, 'figure'),
        ],[
        Input({'page': 'hawks_one', 'type': 'slider', 'name': 'wing-slider'}, 'value'),
        Input({'page': 'hawks_one', 'type': 'slider', 'name': 'weight-slider'}, 'value'),
        ]
)
def update_graphs(wing_value, weight_value):
    wing_graph = px.strip(hawks_data, x='Wing', color='Species')
    wing_graph.add_vline(wing_value)
    weight_graph = px.strip(hawks_data.query('Wing > @wing_value'), x='Weight', color='Species')
    weight_graph.add_vline(weight_value)
    return wing_graph, weight_graph

@callback(
        Output({'page': 'hawks_one', 'type': 'div', 'name': 'confusion-matrix-div'}, 'children'),
        [
        Input({'page': 'hawks_one', 'type': 'slider', 'name': 'wing-slider'}, 'value'),
        Input({'page': 'hawks_one', 'type': 'slider', 'name': 'weight-slider'}, 'value'),
        ]
        )
def update_crosstab(wing_value, weight_value):
    predictions = hawks_data.eval('Wing > @wing_value and Weight > @weight_value').map({True: 'RT', False: 'Not RT'})
    crosstab = pd.crosstab(hawks_data['RT'], predictions, colnames=['Predicted'], rownames=['Actual'], margins=True)
    crosstab.columns = pd.MultiIndex.from_tuples([('Predicted', c) for c in crosstab.columns])
    confusion_matrix = dbc.Table.from_dataframe(crosstab, index=True)
    return confusion_matrix
