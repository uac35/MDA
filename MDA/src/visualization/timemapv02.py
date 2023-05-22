import pandas as pd
import plotly.express as px
import dash
import dash_bootstrap_components as dbc 
from dash import dcc
from dash import html
import os

df = pd.read_csv("./df_final.csv")
df["date"] = pd.to_datetime(df['date'], format='%Y-%m-%d')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB]) 

options = [{"label": "Shouting", "value": "Human voice - Shouting"},
           {"label": "Singing", "value": "Human voice - Singing"},
           {"label": "Music", "value": "Music non-amplified"},
           {"label": "Wind", "value": "Nature elements - Wind"},
           {"label": "Passenger Car", "value": "Transport road - Passenger car"},
           {"label": "Siren", "value": "Transport road - Siren"},
           {"label": "Unsupported", "value": "Unsupported"}]

app.layout = dbc.Container([
    html.H1(),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id="dropdown", options=options, value="Human voice - Shouting", style={"width": "70%","padding-left": "20px"})
        ], width=4, align="center"),
        dbc.Col([
            dcc.Graph(id="map",style={"width": "100%","padding-left": "20px"})
        ], align="center",width=10)
    ])
])


@app.callback(dash.dependencies.Output("map", "figure"),
              [dash.dependencies.Input("dropdown", "value")])
def update_map(event_type):
    if event_type in df.columns:
        filtered_df = df[df[event_type] > 0]
        filtered_df = filtered_df.sort_values(by=['date'])
        filtered_df["Date"] = filtered_df["date"].dt.strftime("%Y-%m-%d")
        fig = px.density_mapbox(filtered_df,
                                lat="latitude",
                                lon="longitude",
                                z=event_type,
                                radius=10,
                                hover_data=["description"],
                                animation_frame="Date",
                                mapbox_style="carto-positron")
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0},mapbox_zoom=13)
        return fig
    else:
        return dash.no_update

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=False)
