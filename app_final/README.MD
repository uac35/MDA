# Noise Events Forecasting in the City of Leuven

This project uses data from noise sensors and a machine learning model to forecast human noise in the city of Leuven, Belgium. The forecasting results are visualized on a heat map.

## Project Repository

The GitHub repository for this project can be found [here](https://github.com/uac35/MDA).

## Live Demo

A live demo of the application is running on Google Cloud Run and can be accessed [here](https://dash-mdamal-g6oxupb6zq-lz.a.run.app/). Please keep in mind this demo can only predict 12 hours ahead of the last window (2023-01-01 and between 00 and 11). If this were deployed as an actual product, we'd ideally get realtime data from the event sensors, and use the last_window argument in the predict function to generate predictions for other timestamps (but always max 12 hours ahead of current date).

## Requirements

The following libraries are required for the final app:

- dash
- dash_bootstrap_components
- plotly
- pandas
- skforecast
- datetime
- numpy
- holidays
- pickle

## Data

The `forecaster` uses data from several noise sensors located at different points in the city. This data is stored in csv files. The model also uses a subset of weather data (temperature, rainfall, and wind speed) and date/time data (time of day, day of week, month, season, etc.) as additional features. Additional domain expertise such as the academic calendar of KU Leuven is also taken into account.

The `classifier` model uses features of date/time (hour of day, day of week), targets Google trends data for the search 'leuven politie' to replace for police calls/complaints data, obtained with pytrends package.

## Usage

To run the *Dash* application:

1. Install the required libraries.
2. Clone the [completed app container](https://github.com/uac35/MDA/tree/main/app_final) and change your working directory to the project folder.
3. Run the script using a Python interpreter. The Dash app will start and can be accessed from a web browser.
4. It is possible to deploy this container directly with Google Cloud Run.

Command to run:

```sh
python app_final.py
```

Then, open a web browser and go to http://localhost:8080 to access the app.

In the app, users can select a date and time and input weather conditions. Upon clicking the 'Predict' button, the app will make a human noise event forecast for the selected date/time and weather conditions. The results will be shown on a heatmap, which also indicates the predicted police activity level for each location. There is also separate page where users can view the previous noise event data of Leuven. Page 3 has the classifier model that uses date and noise occurence data for predicting possible police activity.

## Model

A time series forecasting model that predicts human noise event. See [here](https://github.com/uac35/MDA/tree/Forecast/MDA/notebooks) and [here](https://github.com/uac35/MDA/tree/Forecast/MDA/models/forecasters)
Additionally, a complementary binary classifier that predicts the level of police activity ('High' or 'Low') based on the human noise activity data. Located [here](https://github.com/uac35/MDA/tree/main/MDA/models/Classifier)

The models were trained separately using a *direct multi-step forecast strategy* and saved to indvidual pickle files, which are loaded when the app runs.
