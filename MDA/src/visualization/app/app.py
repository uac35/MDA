import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from skforecast.utils import load_forecaster
from datetime import datetime
import pytz
import pandas
import numpy as np
import holidays

from lightgbm import LGBMClassifier

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import recall_score, precision_score, f1_score

from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import backtesting_forecaster_multiseries
from skforecast.model_selection_multiseries import grid_search_forecaster_multiseries
from skforecast.utils import save_forecaster, load_forecaster

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB])

app.layout = html.Div(
    [
        # Framework of the main app
        html.Div(
            "Noise Events in the city of Leuven", style={"fontSize": 50, "textAlign": "center"}
        ),
        html.Div(
            [
                dcc.Link(children=page["name"] + " | ", href=page["path"])
                for page in dash.page_registry.values()
            ]
        ),
        html.Hr(),
        # Content of each page
        dash.page_container,
    ]
)

df = pd.read_csv("/Users/pjcnudde/Desktop/MDA/Datasets/export_41/df_final.csv")
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")


@app.callback(
    dash.dependencies.Output("map", "figure"), [dash.dependencies.Input("dropdown", "value")]
)
def update_map(event_type):
    if event_type in df.columns:
        filtered_df = df[df[event_type] > 0]
        filtered_df = filtered_df.sort_values(by=["date"])
        filtered_df["Date"] = filtered_df["date"].dt.strftime("%Y-%m-%d")
        fig = px.density_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            z=event_type,
            radius=10,
            hover_data=["description"],
            animation_frame="Date",
            mapbox_style="carto-positron",
        )
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, mapbox_zoom=13)
        return fig
    else:
        return dash.no_update


if __name__ == "__main__":
    app.run(debug=False)
