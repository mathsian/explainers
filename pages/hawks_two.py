from dash import html, dcc, callback, Output, Input, State, register_page
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from os import path

numeric_features = ['Wing', 'Weight', 'Culmen', 'Hallux', 'Tail']
species = ['RT', 'CH', 'SS']

HAWKS_CSV = path.join(path.dirname(__file__), '../data/Hawks.csv')

hawks_data = pd.read_csv(HAWKS_CSV).dropna(subset=numeric_features)

register_page(__name__)

for s in species:
    hawks_data[s] = (hawks_data['Species'] == s).map({True: s, False: f'Not {s}'})

def node_card_factory(id_stub):
    header = dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    dcc.Markdown(id=f'{id_stub}-label'),
                    ], width='auto', align='baseline'),
                dbc.Col([
                    dbc.Select(id=f'{id_stub}-select',
                        options=[{'label': f, 'value': f} for f in numeric_features],
                        value=numeric_features[0]
                    ),
                ], width='auto', align='baseline'),
             ], justify='center'),
            ])
    body = dbc.CardBody([
            dbc.Col([
                dcc.Graph(id=f'{id_stub}-graph', config={'displayModeBar': False, 'staticPlot': True}),
                dcc.Slider(0, 100, id=f'{id_stub}-slider'),
            ]),
            ])
    return dbc.Card([header, body])

def leaf_card_factory(id_stub):
    header = dbc.CardHeader([
        dbc.Row([
            dbc.Col([
            dcc.Markdown(id=f'{id_stub}-label')
            ], width='auto', align='baseline')
        ], justify='center')
        ])
    body = dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dcc.Markdown(id=f'{id_stub}-table')
                ], width='auto', align='baseline')
                ], justify='center')
            ])
    return dbc.Card([header, body])

first_card = node_card_factory("first-feature")
second_cards = [node_card_factory(f'second-{side}') for side in ['left', 'right']]
leaf_cards = [leaf_card_factory(f'leaf-{n}') for n in [1, 2, 3, 4]]

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1(children='Hawk classification', style={'textAlign': 'center'})
            ])
        ]),
    dbc.Row([
        dbc.Col([
            first_card
            ])
        ]),
    dbc.Row([
        dbc.Col([
            dbc.CardGroup(second_cards)
       ]),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.CardGroup(leaf_cards)
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Select(id='species-select',
                options=[{'label': s, 'value': s} for s in species],
                value=species[0]
                ),
            html.H2("Confusion Matrix"),
            html.Div(id="confusion-matrix")
            ],
            width=6)
    ])
])
# ], fluid=True)

@callback(
        [
            Output('first-feature-label', 'children'),
            Output('first-feature-slider', 'min'),
            Output('first-feature-slider', 'max'),
            Output('first-feature-slider', 'value'),
        ],
        Input('first-feature-select', 'value')
        )
def update_first_feature_slider(first_feature):
    first_feature_heading = f'{first_feature} distribution'
    first_feature_min = hawks_data[first_feature].min().round()
    first_feature_max = hawks_data[first_feature].max().round()
    first_feature_value = hawks_data[first_feature].mean().round()
    return "First look at ", first_feature_min, first_feature_max, first_feature_value 

@callback(
        [
        Output('first-feature-graph', 'figure'),
        Output('second-left-label', 'children'),
        Output('second-right-label', 'children'),
        ],
        Input('first-feature-slider', 'value'),
        State('first-feature-select', 'value')
        )
def update_first_feature_graph(first_feature_value, first_feature):
    first_feature_figure = px.strip(hawks_data, x=first_feature, color='Species', category_orders={'Species': species}, title=f'{first_feature} distribution')
    first_feature_figure.add_vline(first_feature_value)
    second_left_label = f'Then if {first_feature} &le; {first_feature_value}, look at '
    second_right_label = f'And if {first_feature} &gt; {first_feature_value}, look at '
    return first_feature_figure, second_left_label, second_right_label

@callback(
        [
            Output('second-left-slider', 'min'),
            Output('second-left-slider', 'max'),
            Output('second-left-slider', 'value'),
        ], [ 
        Input('second-left-select', 'value'),
        Input('first-feature-slider', 'value')
        ],
        State('first-feature-select', 'value')
        )
def update_second_left_slider(second_left_feature, first_feature_value, first_feature):
    second_left_feature_min = hawks_data[hawks_data[first_feature] <= first_feature_value][second_left_feature].min().round()
    second_left_feature_max = hawks_data[hawks_data[first_feature] <= first_feature_value][second_left_feature].max().round()
    second_left_feature_value = hawks_data[hawks_data[first_feature] <= first_feature_value][second_left_feature].mean().round()
    return second_left_feature_min, second_left_feature_max, second_left_feature_value 

@callback(
        [
            Output('second-right-slider', 'min'),
            Output('second-right-slider', 'max'),
            Output('second-right-slider', 'value'),
        ], [ 
        Input('second-right-select', 'value'),
        Input('first-feature-slider', 'value')
        ],
        State('first-feature-select', 'value')
        )
def update_second_right_slider(second_right_feature, first_feature_value, first_feature):
    second_right_feature_min = hawks_data[hawks_data[first_feature] > first_feature_value][second_right_feature].min().round()
    second_right_feature_max = hawks_data[hawks_data[first_feature] > first_feature_value][second_right_feature].max().round()
    second_right_feature_value = hawks_data[hawks_data[first_feature] > first_feature_value][second_right_feature].mean().round()
    return second_right_feature_min, second_right_feature_max, second_right_feature_value 

@callback(
        Output('second-left-graph', 'figure'),
        [
        Input('second-left-slider', 'value'),
        Input('first-feature-slider', 'value'),
        ], [
        State('second-left-select', 'value'),
        State('first-feature-select', 'value'),
        ]
        )
def update_second_left_graph(second_left_value, first_feature_value, second_left_feature, first_feature):
    second_left_figure = px.strip(hawks_data[hawks_data[first_feature] <= first_feature_value],
            x=second_left_feature, 
            color='Species', 
            category_orders={'Species': species})
    second_left_figure.add_vline(second_left_value)
    return second_left_figure

@callback(
        Output('second-right-graph', 'figure'),
        [
        Input('second-right-slider', 'value'),
        Input('first-feature-slider', 'value'),
        ], [
        State('second-right-select', 'value'),
        State('first-feature-select', 'value'),
        ]
        )
def update_second_right_graph(second_right_value, first_feature_value, second_right_feature, first_feature):
    second_right_figure = px.strip(hawks_data[hawks_data[first_feature] > first_feature_value], 
            x=second_right_feature, 
            color='Species', 
            category_orders={'Species': species})
    second_right_figure.add_vline(second_right_value)
    return second_right_figure


@callback(
        [
        Output('confusion-matrix', 'children'),
        Output('leaf-1-label', 'children'),
        Output('leaf-2-label', 'children'),
        Output('leaf-3-label', 'children'),
        Output('leaf-4-label', 'children'),
        Output('leaf-1-table', 'children'),
        Output('leaf-2-table', 'children'),
        Output('leaf-3-table', 'children'),
        Output('leaf-4-table', 'children'),
        ], [
            Input('first-feature-select', 'value'),
            Input('first-feature-slider', 'value'),
            Input('second-left-select', 'value'),
            Input('second-left-slider', 'value'),
            Input('second-right-select', 'value'),
            Input('second-right-slider', 'value'),
            Input('species-select', 'value'),
        ]
        )
def update_crosstab(first_feature, first_feature_value, second_left_feature, second_left_value, second_right_feature, second_right_value, species):
    predictions = (hawks_data[first_feature] > first_feature_value).map({True: species, False: f'Not {species}'})
    crosstab = pd.crosstab(hawks_data[species], predictions, colnames=['Predicted'], rownames=['Actual'], margins=True)
    crosstab.columns = pd.MultiIndex.from_tuples([('Predicted', c) for c in crosstab.columns])
    confusion_matrix = dbc.Table.from_dataframe(crosstab, index=True)

    leaf_1_label = f'{first_feature} &le; {first_feature_value} and {second_left_feature} &le; {second_left_value}'
    leaf_1_table = hawks_data[(hawks_data[first_feature] <= first_feature_value) & (hawks_data[second_left_feature] <= second_left_value)].value_counts('Species').rename('Count').to_markdown()
    leaf_2_label = f'{first_feature} &le; {first_feature_value} and {second_left_feature} &gt; {second_left_value}'
    leaf_2_table = hawks_data[(hawks_data[first_feature] <= first_feature_value) & (hawks_data[second_left_feature] > second_left_value)].value_counts('Species').rename('Count').to_markdown()
    leaf_3_label = f'{first_feature} &gt; {first_feature_value} and {second_right_feature} &le; {second_right_value}'
    leaf_3_table = hawks_data[(hawks_data[first_feature] > first_feature_value) & (hawks_data[second_right_feature] <= second_right_value)].value_counts('Species').rename('Count').to_markdown()
    leaf_4_label = f'{first_feature} &gt; {first_feature_value} and {second_right_feature} &gt; {second_right_value}'
    leaf_4_table = hawks_data[(hawks_data[first_feature] > first_feature_value) & (hawks_data[second_right_feature] > second_right_value)].value_counts('Species').rename('Count').to_markdown()
    return confusion_matrix, leaf_1_label, leaf_2_label, leaf_3_label, leaf_4_label, leaf_1_table, leaf_2_table, leaf_3_table, leaf_4_table

