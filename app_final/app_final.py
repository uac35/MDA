import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from skforecast.utils import load_forecaster
from datetime import datetime
import numpy as np
import holidays
import os
import pickle
from holidays import Belgium
#os.chdir('C:\\Users\\uygar\\Desktop\\DataAnalytics\\woork\\newestofthenew')

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

df = pd.read_csv("./df_final.csv")
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")


@app.callback(
    dash.dependencies.Output("map", "figure"), [dash.dependencies.Input("dropdown", "value")]
)
def update_map(event_type):
    if event_type in df.columns:
        filtered_df = df[df[event_type] > 0]
        filtered_df = filtered_df.sort_values(by=["date"])
        filtered_df["Date"] = filtered_df["date"].dt.strftime("%Y-%m-%d")
        filtered_df["marker_size"]=1
        max_value = filtered_df[event_type].max()
        fig = px.density_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            z=event_type,
            hover_data=["description"],
            animation_frame="Date",
            range_color=[0, max_value],
            mapbox_style="carto-positron",
            zoom=13,
        )
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        return fig
    else:
        return dash.no_update

# All the code for the page of the forecaster
# Load classifier model from pickle, google trends data
with open("./classifier_trends_daily.pkl", "rb") as file:
    classifier = pickle.load(file)
    


def create_heatmap(df):
    df['Predicted'] = df['prediction'].apply(lambda x: 'Expect noise' if x == 1 else 'Quiet')
    df['marker_size']=3
    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        color='prediction',
        size='marker_size',
        hover_data={'Predicted': True},
        center=dict(lat=50.875, lon=4.700),
        color_continuous_scale='Jet',
        range_color=[0,1],
        zoom=14,  
        mapbox_style="carto-positron",
        size_max=10,
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig

last_known_timestamp = datetime.strptime("2022-12-31 23", "%Y-%m-%d %H")
LOCATIONS = [
    (
        "255439_mp-01-naamsestraat-35-maxim.csv",
        (50.877160828051615, 4.7006992525313445),
    ),
    ("255440_mp-02-naamsestraat-57-xior.csv", (50.87649057167669, 4.700691639839767)),
    ("255441_mp-03-naamsestraat-62-taste.csv", (50.875850541602986, 4.700192168675618)),
    (
        "255442_mp-05-calvariekapel-ku-leuven.csv",
        (50.87448678827803, 4.699889625684066),
    ),
    (
        "255443_mp-06-parkstraat-2-la-filosovia.csv",
        (50.87409073433792, 4.700018371714718),
    ),
    ("255444_mp-07-naamsestraat-81.csv", (50.87381822269834, 4.70010170607578)),
]

forecasters = [
    "./forecaster_h12_255439_mp-01-naamsestraat-35-maxim.pkl",
    "./forecaster_h12_255440_mp-02-naamsestraat-57-xior.pkl",
    "./forecaster_h12_255441_mp-03-naamsestraat-62-taste.pkl",
    "./forecaster_h12_255442_mp-05-calvariekapel-ku-leuven.pkl",
    "./forecaster_h12_255443_mp-06-parkstraat-2-la-filosovia.pkl",
    "./forecaster_h12_255444_mp-07-naamsestraat-81.pkl",
]


def get_custom_weights(index):
    ts_train = ts_data.loc[start_train:end_val, level]
    ts = ts_data.loc[start_train:, level]
    minority_weight = ts_train.value_counts()[0] / ts_train.value_counts()[1]
    weights = np.where(ts == 1, minority_weight, 1)
    w = weights[: len(index)]
    return w


def to_cat(x):
    x = x.astype("category")
    return x


def prepare_new_data(new_data):
    # Define the academic calendar data for 2023
    kul_ac_year = pd.DataFrame(
        {
            "begin_date": [
                "2023-01-10",
                "2023-05-30",
                "2023-01-10",
                "2023-05-30",
                "2023-01-31",
                "2023-06-27",
                "2023-01-01",
                "2023-02-05",
                "2023-04-02",
                "2023-07-02",
                "2023-12-24",
            ],
            "end_date": [
                "2023-02-04",
                "2023-07-01",
                "2023-01-30",
                "2023-06-26",
                "2023-02-04",
                "2023-07-01",
                "2023-01-13",
                "2023-02-13",
                "2023-04-18",
                "2023-09-25",
                "2023-12-31",
            ],
            "type": [
                "exams",
                "exams",
                "first_exam_weeks",
                "first_exam_weeks",
                "final_exam_week",
                "final_exam_week",
                "vacation",
                "vacation",
                "vacation",
                "vacation",
                "vacation",
            ],
        }
    )

    kul_ac_year["begin_date"] = pd.to_datetime(kul_ac_year["begin_date"])
    kul_ac_year["end_date"] = pd.to_datetime(kul_ac_year["end_date"])
    kul_ac_year["end_date"] = kul_ac_year["end_date"] + pd.Timedelta("23:59:59")

    # Prepare features
    new_data["is_weekend"] = new_data.index.weekday.isin([5, 6]).astype(int)
    new_data["day_of_week"] = new_data.index.day_name()
    new_data["hour_of_day"] = new_data.index.hour
    new_data["time_of_day"] = pd.cut(
        new_data.index.hour,
        bins=[-1, 6, 12, 18, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"],
    )
    new_data["season"] = pd.cut(
        new_data.index.month,
        bins=[0, 3, 6, 9, 12],
        labels=["Winter", "Spring", "Summer", "Fall"],
    )
    new_data["month"] = new_data.index.month_name()
    new_data["day_of_month"] = new_data.index.day
    new_data["quarter"] = "Q" + new_data.index.quarter.astype(str)
    be_holidays = holidays.BE()
    new_data["is_be_holiday"] = (
        pd.Series(new_data.index.date).astype("datetime64").isin(be_holidays).astype(int)
    )
    new_data["is_business_day"] = ~(new_data.index.weekday.isin([5, 6]) | new_data["is_be_holiday"])

    new_data["exams"] = new_data.index.to_series().apply(
        lambda t: any(
            (kul_ac_year["type"] == "exams")
            & (kul_ac_year["begin_date"] <= t)
            & (kul_ac_year["end_date"] >= t)
        )
    )
    new_data["first_exam_weeks"] = new_data.index.to_series().apply(
        lambda t: any(
            (kul_ac_year["type"] == "first_exam_weeks")
            & (kul_ac_year["begin_date"] <= t)
            & (kul_ac_year["end_date"] >= t)
        )
    )
    new_data["final_exam_week"] = new_data.index.to_series().apply(
        lambda t: any(
            (kul_ac_year["type"] == "final_exam_week")
            & (kul_ac_year["begin_date"] <= t)
            & (kul_ac_year["end_date"] >= t)
        )
    )
    new_data["student_vacation"] = new_data.index.to_series().apply(
        lambda t: any(
            (kul_ac_year["type"] == "vacation")
            & (kul_ac_year["begin_date"] <= t)
            & (kul_ac_year["end_date"] >= t)
        )
    )
    new_data["missing"] = 0  # Add the missing column
    new_data = new_data.drop(["day_of_month", "quarter", "exams"], axis=1)
    return new_data


# Define the callback function to handle button clicks and make predictions
@app.callback(
    dash.dependencies.Output("prediction-output", "children"),
    dash.dependencies.Input("predict-button", "n_clicks"),
    [
        dash.dependencies.State("date-input", "value"),
        dash.dependencies.State("temp-input", "value"),
        dash.dependencies.State("rain-input", "value"),
        dash.dependencies.State("wind-input", "value"),
    ],
)
def update_output(n_clicks, date, temp, rain, wind):
    if n_clicks > 0:
        try:
            timestamp = pd.to_datetime(date, format='%Y-%m-%d %H')
            steps_ahead = int((timestamp - last_known_timestamp).total_seconds() // 3600)
            # Calculate the starting timestamp for exog data
            exog_start = last_known_timestamp + pd.DateOffset(hours=1)
            # Create exog data with default values, assuming default values are zeros
            future_data = pd.DataFrame({
                'timestamp': [exog_start + pd.DateOffset(hours=i) for i in range(12)],
                'LC_TEMP_QCL3': [0] * 12,
                'LC_RAININ': [0] * 12,
                'LC_WINDSPEED': [0] * 12
            })

            future_data.set_index('timestamp', inplace=True)
            future_data = prepare_new_data(future_data)
            future_data = future_data.fillna(method='pad')  # Or method='backfill'
            future_data.index = pd.DatetimeIndex(future_data.index, freq='H')

            # Replace the exog data at the input hour with user-provided values
            future_data.loc[timestamp, ['LC_TEMP_QCL3', 'LC_RAININ', 'LC_WINDSPEED']] = [temp, rain, wind]

            # Load the pre-trained models and make predictions
            predictions = []
            for forecaster in forecasters:
                with open(forecaster, 'rb') as file:
                    model = load_forecaster(file)
                prediction = model.predict(steps=steps_ahead, exog=future_data)
                prediction_value = prediction.iloc[-1, 0]  # Get the value of the prediction, not the index
                predictions.append(prediction_value)          

            df = pd.DataFrame({
                'location': [loc[0].split('_')[-1].replace('.csv', '') for loc in LOCATIONS],  # extracting location name from the CSV filename
                'prediction': predictions,
                'latitude': [loc[1][0] for loc in LOCATIONS],
                'longitude': [loc[1][1] for loc in LOCATIONS],
            })


            
            
            # Add police_activity to the 'df' dataframe
    
            fig = create_heatmap(df)
            return dcc.Graph(figure=fig)
        except Exception as e:
            return f"Error: Check the input, potentially out of forecast range. Error message: {str(e)}"
        
        
@app.callback(
    dash.dependencies.Output("guess-output", "children"),
    dash.dependencies.Input("guess-button", "n_clicks"),
    [
        dash.dependencies.State("date-input", "value"),
        dash.dependencies.State("noise-input", "value"),
    ],
)
def update_output2(n_clicks, date, noise_human):
    if n_clicks > 0:
        try:       
            timestamp = pd.to_datetime(date, format='%Y-%m-%d %H')   
            day_of_week = timestamp.dayofweek
            is_weekend = 1 if day_of_week > 4 else 0  # 5 and 6 corresponds to Saturday and Sunday
            belgium_holidays = Belgium()
            is_holiday = 1 if timestamp in belgium_holidays else 0
            classifier_features = pd.DataFrame([{
                'human_noise': noise_human, # Add noise predictions
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
            }])
                    # Use the classifier to predict police activity
            police_activity = classifier.predict(classifier_features)
            police_activity = 'low' if police_activity == 0 else 'high' # Map binary output to 'High'/'Low'
            
            return f"Expect {police_activity} police activity"
        except Exception as e:
            return f"Error message: {str(e)}"

if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=False)
