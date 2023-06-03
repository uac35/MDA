import pandas as pd
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import os


dash.register_page(__name__, path="/")  # path argument to make this the home page

options = [
    {"label": "Shouting", "value": "Human voice - Shouting"},
    {"label": "Singing", "value": "Human voice - Singing"},
    {"label": "Music", "value": "Music non-amplified"},
    {"label": "Wind", "value": "Nature elements - Wind"},
    {"label": "Passenger Car", "value": "Transport road - Passenger car"},
    {"label": "Siren", "value": "Transport road - Siren"},
    {"label": "Unsupported", "value": "Unsupported"},
]

layout = html.Div(
    [
        dbc.Container(
            [
                html.H1(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Dropdown(
                                    id="dropdown",
                                    options=options,
                                    value="Human voice - Shouting",
                                    style={"width": "70%", "padding-left": "20px"},
                                )
                            ],
                            width=4,
                            align="center",
                        ),
                        dbc.Col(
                            [dcc.Graph(id="map", style={"width": "100%", "padding-left": "20px"})],
                            align="center",
                            width=10,
                        ),
                    ]
                ),
            ]
        )
    ]
)
