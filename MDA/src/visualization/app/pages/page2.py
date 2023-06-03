import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime

dash.register_page(__name__)

layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [dcc.Graph(id="map", style={"width": "100%", "padding-left": "20px"})],
                            width=12,
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Date"),
                                dcc.DatePickerSingle(id="date-picker", date=datetime.now()),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Temperature"),
                                dcc.Slider(
                                    id="temperature-slider", min=0, max=50, step=0.5, value=25
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Rainfall"),
                                dcc.Slider(id="rainfall-slider", min=0, max=500, step=5, value=100),
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Windspeed"),
                                dcc.Slider(id="windspeed-slider", min=0, max=30, step=1, value=10),
                            ],
                            width=3,
                        ),
                    ]
                ),
            ]
        )
    ]
)
