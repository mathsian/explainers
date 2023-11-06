from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

hawks_data = pd.read_csv('Hawks.csv').dropna(subset=['Wing', 'Weight'])

hawks_data['RT'] = (hawks_data['Species'] == 'RT').map({True: 'RT', False: 'Not RT'})

app = Dash(__name__, url_base_pathname='/hawks/', external_stylesheets=[dbc.themes.CERULEAN], title='Hawk classification')

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1(children='Hawk classification', style={'textAlign': 'center'})
            ])
        ]),
    dbc.Row([
        dbc.Col([
            html.H2("Wing length distribution"),
            dcc.Graph(id='wing-graph'),
            html.Br(),
            dcc.Slider(0, hawks_data['Wing'].max(), value=hawks_data['Wing'].mean().round(), id='wing-slider'),
            html.Br(),
            html.H2("Weight distribution"),
            dcc.Graph(id='weight-graph'),
            html.Br(),
            dcc.Slider(0, hawks_data['Weight'].max(), value=hawks_data['Weight'].mean().round(), id='weight-slider'),
            ], width=8),
        dbc.Col([
            html.H2("Confusion Matrix"),
            html.Div(id="confusion-matrix")
            ],
            width=4)
        ])
    ])

@callback(
        [
        Output('wing-graph', 'figure'),
        Output('weight-graph', 'figure'),
        ],[
        Input('wing-slider', 'value'),
        Input('weight-slider', 'value'),
        ]
)
def update_graphs(wing_value, weight_value):
    wing_graph = px.strip(hawks_data, x='Wing', color='Species')
    wing_graph.add_vline(wing_value)
    weight_graph = px.strip(hawks_data.query('Wing > @wing_value'), x='Weight', color='Species')
    weight_graph.add_vline(weight_value)
    return wing_graph, weight_graph

@callback(
        Output('confusion-matrix', 'children'),
        [
            Input('wing-slider', 'value'),
            Input('weight-slider', 'value')
        ]
        )
def update_crosstab(wing_value, weight_value):
    predictions = hawks_data.eval('Wing > @wing_value and Weight > @weight_value').map({True: 'RT', False: 'Not RT'})
    crosstab = pd.crosstab(hawks_data['RT'], predictions, colnames=['Predicted'], rownames=['Actual'], margins=True)
    crosstab.columns = pd.MultiIndex.from_tuples([('Predicted', c) for c in crosstab.columns])
    confusion_matrix = dbc.Table.from_dataframe(crosstab, index=True)
    return confusion_matrix

if __name__ == '__main__':
    app.run(debug=True)


