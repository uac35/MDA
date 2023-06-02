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


#TODO how do we get this information? there is no API ? Web scraping?
kul_ac_year = pd.DataFrame({"begin_date": ["2023-01-10", "2023-05-30", "2023-01-10", "2023-05-30",
                                           "2023-01-31", "2023-06-27", "2023-01-01", "2023-02-05",
                                           "2023-04-02", "2023-07-02", "2023-12-24"],
                    "end_date": ["2023-02-04", "2023-07-01", "2023-01-30", "2023-06-26", 
                                 "2023-02-04", "2023-07-01", "2023-01-13", "2023-02-13", 
                                 "2023-04-18", "2023-09-25", "2023-12-31"],
                    "type": ["exams", "exams", "first_exam_weeks", "first_exam_weeks", 
                             "final_exam_week", "final_exam_week", "vacation", "vacation", 
                             "vacation", "vacation", "vacation"]})
kul_ac_year.head()

kul_ac_year["begin_date"] = pd.to_datetime(kul_ac_year["begin_date"]).dt.tz_localize('UTC')
kul_ac_year["end_date"] = pd.to_datetime(kul_ac_year["end_date"]).dt.tz_localize('UTC') + pd.Timedelta('23:59:59')

kul_ac_year["end_date"] = kul_ac_year["end_date"] + pd.Timedelta('23:59:59')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
 
def get_custom_weights(index):
    ts_train = ts_data.loc[start_train:end_val, level]
    ts = ts_data.loc[start_train:, level]
    minority_weight = ts_train.value_counts()[0] / ts_train.value_counts()[1]
    weights = np.where(ts == 1, minority_weight, 1)
    w = weights[:len(index)]
    return w
def to_cat(x):
  x = x.astype('category')
  return x

forecasters = ["C:\\Users\\uygar\\Desktop\\DataAnalytics\\fcast\\forecaster_h12_255439_mp-01-naamsestraat-35-maxim.pkl", "C:\\Users\\uygar\\Desktop\\DataAnalytics\\fcast\\forecaster_h12_255440_mp-02-naamsestraat-57-xior.pkl", "C:\\Users\\uygar\\Desktop\\DataAnalytics\\fcast\\forecaster_h12_255441_mp-03-naamsestraat-62-taste.pkl",
               "C:\\Users\\uygar\\Desktop\\DataAnalytics\\fcast\\forecaster_h12_255442_mp-05-calvariekapel-ku-leuven.pkl", "C:\\Users\\uygar\\Desktop\\DataAnalytics\\fcast\\forecaster_h12_255443_mp-06-parkstraat-2-la-filosovia.pkl", "C:\\Users\\uygar\\Desktop\\DataAnalytics\\fcast\\forecaster_h12_255444_mp-07-naamsestraat-81.pkl"]

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="map", style={"width": "100%", "padding-left": "20px"})
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Date'),
            dcc.DatePickerSingle(id='date-picker', date=datetime.now())
        ], width=3),
        dbc.Col([
            html.Label('Temperature'),
            dcc.Slider(id='temperature-slider', min=0, max=50, step=0.5, value=25)  
        ], width=3),
        dbc.Col([
            html.Label('Rainfall'),
            dcc.Slider(id='rainfall-slider', min=0, max=500, step=5, value=100)
        ], width=3),
        dbc.Col([
            html.Label('Windspeed'),
            dcc.Slider(id='windspeed-slider', min=0, max=30, step=1, value=10)
        ], width=3),
    ])
])

@app.callback(
    Output("map", "figure"),
    [Input('date-picker', 'date'),
     Input('temperature-slider', 'value'),
     Input('rainfall-slider', 'value'),
     Input('windspeed-slider', 'value')]
)
def update_map(date, temperature, rainfall, windspeed):
        # Initialize an empty DataFrame for the results
    df = pd.DataFrame()

    # Convert date string to datetime object
    date = pd.to_datetime(date)

    # Start date is set to 2023-01-01
    start_date = pd.Timestamp('2023-01-01 00:00:00', tz='UTC')
    end_date = pd.Timestamp(date, tz='UTC')
    exog = pd.DataFrame(index=pd.date_range(start=start_date, freq='H', periods= 12))

    # Set temperature, rainfall, and windspeed to the values provided by the user
    exog['LC_TEMP_QCL3'] = temperature
    exog['LC_RAININ'] = rainfall
    exog['LC_WINDSPEED'] = windspeed

    # Compute the other exogenous variables for each hour in the DataFrame
    exog['is_weekend'] = (exog.index.weekday >= 5).astype(int)
    exog['day_of_week'] = exog.index.day_name()
    exog['hour_of_day'] = exog.index.hour
    bins = pd.cut(exog['hour_of_day'], bins=[-1, 6, 12, 18, 24], labels=False)
    exog['time_of_day'] = bins.apply(lambda x: ["Night", "Morning", "Afternoon", "Evening"][int(x)])
    months = pd.Series(exog.index.month)
    season_bins = pd.cut(months, bins=[0, 3, 6, 9, 12], labels=False)
    exog['season'] = season_bins.map(lambda x: ["Winter", "Spring", "Summer", "Fall"][x])

    exog['month'] = exog.index.month_name()

    # Compute holiday and business day features for each hour in the DataFrame
    be_holidays = holidays.BE()
    exog['is_be_holiday'] = exog.index.isin(be_holidays).astype(int)
    exog['is_business_day'] = (~exog['is_weekend'] & ~exog['is_be_holiday']).astype(int)

    # Compute academic calendar features for each hour in the DataFrame
    for i, row in kul_ac_year.iterrows():
        exog.loc[(exog.index >= row.begin_date) & (exog.index <= row.end_date), row.type] = 1

    # Fill NaN values in the academic calendar features
    kul_ac_year_cols = kul_ac_year['type'].unique()
    exog[kul_ac_year_cols] = exog[kul_ac_year_cols].fillna(0)

    # Set the missing data indicator to 0
    exog['missing'] = 0
    # Loop over all forecasters
    for forecaster_pkl in forecasters:
        # Load the forecaster
        forecaster = load_forecaster(forecaster_pkl)

        # Build the exog DataFrame based on user inputs
        exog = pd.DataFrame(
            data={
                'timestamp': [pd.Timestamp(date, tz=pytz.UTC)],
                'is_weekend': [(exog.index.weekday >= 5).astype(int)],
                'day_of_week': [exog.index.day_name()],
                'time_of_day': [bins.apply(lambda x: ["Night", "Morning", "Afternoon", "Evening"][int(x)])],
                'season': [season_bins.map(lambda x: ["Winter", "Spring", "Summer", "Fall"][x])],
                'month': [exog.index.month_name()],
                'is_be_holiday': [exog.index.isin(be_holidays).astype(int)],
                'is_business_day': [(~exog['is_weekend'] & ~exog['is_be_holiday']).astype(int)],
                'first_exam_weeks': [exog.index.to_series().apply(lambda t: any((kul_ac_year["type"] == "first_exam_weeks") & (kul_ac_year["begin_date"] <= t) & (kul_ac_year["end_date"] >= t)))],
                'final_exam_week': [exog.index.to_series().apply(lambda t: any((kul_ac_year["type"] == "final_exam_weeks") & (kul_ac_year["begin_date"] <= t) & (kul_ac_year["end_date"] >= t)))],
                'student_vacation': [exog.index.to_series().apply(lambda t: any((kul_ac_year["type"] == "student_vacation") & (kul_ac_year["begin_date"] <= t) & (kul_ac_year["end_date"] >= t)))],
                'hour_of_day': [exog.index.hour],
                'LC_TEMP_QCL3': [temperature],
                'LC_RAININ': [rainfall],
                'LC_WINDSPEED': [windspeed],
                'missing': [0] #TODO HOW?
            }
).set_index('timestamp')
        
        exog.index = pd.DatetimeIndex(exog.index, freq='H')
        # Perform the prediction
        predicted_noise = forecaster.predict(steps=1,exog=exog)
        print(f'Type of predicted_noise: {type(predicted_noise)}')
        print(f'Value of predicted_noise: {predicted_noise}')

        # Extract the longitude and latitude from the forecaster
        latitude = forecaster.data['latitude'].iloc[0]
        longitude = forecaster.data['longitude'].iloc[0]

        # Append the data to the main DataFrame
        df = df.append(pd.DataFrame({
            'latitude': [latitude],
            'longitude': [longitude],
            'predicted_human_noise': [predicted_noise.iloc[0]],
        }), ignore_index=True)


    fig = px.density_mapbox(df,
                            lat="latitude",
                            lon="longitude",
                            z='predicted_human_noise',
                            radius=10,
                            hover_data=["predicted_human_noise"],
                            mapbox_style="carto-positron")

    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, mapbox_zoom=13)

    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
