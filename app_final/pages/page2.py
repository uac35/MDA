import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime

dash.register_page(__name__)

layout = html.Div(
    [
        html.H1("Forecasting Model"),
        html.Label("Date and Hour (format: YYYY-MM-DD HH):"),
        dcc.Input(id="date-input", type="text", style={'height':'25px', 'width':'120px'}),
        html.Label("T [°C]:"),
        dcc.Input(id="temp-input", type="number", style={'height':'25px', 'width':'50px'}),
        html.Label("Rain [m]:"),
        dcc.Input(id="rain-input", type="number", style={'height':'25px', 'width':'50px'}),
        html.Label("Wind [m/s]:"),
        dcc.Input(id="wind-input", type="number", style={'height':'25px', 'width':'50px'}),
        html.Button("Forecast", id="predict-button", n_clicks=0),
        html.Div(id="prediction-output"),
    ]
)

