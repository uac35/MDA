import pickle
import pandas as pd
import numpy as np
import joblib
import holidays
import dash
import dash_html_components as html
import dash_core_components as dcc
from skforecast.utils import load_forecaster
from datetime import datetime
last_known_timestamp = datetime.strptime('2022-12-31 23', '%Y-%m-%d %H')

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

def prepare_new_data(new_data):
    # Define the academic calendar data for 2023
    kul_ac_year = pd.DataFrame({
        "begin_date": ["2023-01-10", "2023-05-30", "2023-01-10", "2023-05-30",
                       "2023-01-31", "2023-06-27", "2023-01-01", "2023-02-05",
                       "2023-04-02", "2023-07-02", "2023-12-24"],
        "end_date": ["2023-02-04", "2023-07-01", "2023-01-30", "2023-06-26", 
                     "2023-02-04", "2023-07-01", "2023-01-13", "2023-02-13", 
                     "2023-04-18", "2023-09-25", "2023-12-31"],
        "type": ["exams", "exams", "first_exam_weeks", "first_exam_weeks", 
                 "final_exam_week", "final_exam_week", "vacation", "vacation", 
                 "vacation", "vacation", "vacation"]
    })

    kul_ac_year["begin_date"] = pd.to_datetime(kul_ac_year["begin_date"])
    kul_ac_year["end_date"] = pd.to_datetime(kul_ac_year["end_date"])
    kul_ac_year["end_date"] = kul_ac_year["end_date"] + pd.Timedelta('23:59:59')

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

    new_data["exams"] = new_data.index.to_series().apply(lambda t: any((kul_ac_year["type"] == "exams") & (kul_ac_year["begin_date"] <= t) & (kul_ac_year["end_date"] >= t)))
    new_data["first_exam_weeks"] = new_data.index.to_series().apply(lambda t: any((kul_ac_year["type"] == "first_exam_weeks") & (kul_ac_year["begin_date"] <= t) & (kul_ac_year["end_date"] >= t)))
    new_data["final_exam_week"] = new_data.index.to_series().apply(lambda t: any((kul_ac_year["type"] == "final_exam_week") & (kul_ac_year["begin_date"] <= t) & (kul_ac_year["end_date"] >= t)))
    new_data["student_vacation"] = new_data.index.to_series().apply(lambda t: any((kul_ac_year["type"] == "vacation") & (kul_ac_year["begin_date"] <= t) & (kul_ac_year["end_date"] >= t)))
    new_data["missing"] = 0  # Add the missing column
    new_data = new_data.drop(["day_of_month", "quarter", "exams"], axis=1) 
    return new_data


# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the web app
app.layout = html.Div(
    children=[
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
        timestamp = pd.to_datetime(date, format='%Y-%m-%d %H')
        new_data = pd.DataFrame({
            'timestamp': [pd.to_datetime(date, format='%Y-%m-%d %H')],
            'LC_TEMP_QCL3': [temp],
            'LC_RAININ': [rain],
            'LC_WINDSPEED': [wind]
        })
        new_data.set_index('timestamp', inplace=True)
        new_data = prepare_new_data(new_data)
        new_data = new_data.fillna(method='pad')  # Or method='backfill'
        new_data.index = pd.DatetimeIndex(new_data.index, freq='H')


        # Load the pre-trained model
        with open('C:\\Users\\uygar\\Desktop\\DataAnalytics\\fcast\\forecaster_255439_mp-01-naamsestraat-35-maxim.pkl', 'rb') as file:
            model = load_forecaster(file)
        steps_ahead = int((timestamp - last_known_timestamp).total_seconds() // 3600)

        predictions = []

        for i in range(steps_ahead):
            # Perform a one-step ahead forecast
            prediction = model.predict(steps=1, exog=new_data)

            # Store the forecast
            predictions.append(prediction)

            # Update the input data for the next forecast
            new_row = pd.DataFrame({
                'timestamp': [timestamp + pd.DateOffset(hours=i)],
                'LC_TEMP_QCL3': [temp],
                'LC_RAININ': [rain],
                'LC_WINDSPEED': [wind]
            })
            new_row.set_index('timestamp', inplace=True)
            new_row = prepare_new_data(new_row)
            new_row = new_row.fillna(method='pad')  # Or method='backfill'
            new_data = pd.concat([new_data[:-1], new_row])
        return html.Div(f"The predicted values are {predictions}.")
# Run the application
if __name__ == "__main__":
    app.run_server(debug=True)