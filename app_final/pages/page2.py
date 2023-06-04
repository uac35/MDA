import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime

dash.register_page(__name__)

layout = html.Div(
    [
        html.H1("Predictive Model"),
        html.Label("Date and Hour (format: YYYY-MM-DD HH):"),
        dcc.Input(id="date-input", type="text"),
        html.Label("LC_TEMP value:"),
        dcc.Input(id="temp-input", type="number"),
        html.Label("LC_RAIN value:"),
        dcc.Input(id="rain-input", type="number"),
        html.Label("LC_WIND value:"),
        dcc.Input(id="wind-input", type="number"),
        html.Button("Predict", id="predict-button", n_clicks=0),
        html.Div(id="prediction-output"),
    ]
)
