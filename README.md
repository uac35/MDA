# Noise Events Forecasting in the City of Leuven

This project uses data from noise sensors and a machine learning model to forecast human noise in the city of Leuven, Belgium. The forecasting results are visualized on a heat map. The project also uses an additional machine learning model to predict the level of police activity (google trends search: 'leuven politie') based on the forecasted noise levels and other features.

## Project Repository

The GitHub repository for this project can be found [here](https://github.com/uac35/MDA).

## Live Demo

A live demo of the application is running on Google Cloud Run and can be accessed [here](https://dash-mdamal-g6oxupb6zq-lz.a.run.app/).

## Requirements

The following libraries are required:

- dash
- dash_bootstrap_components
- plotly
- pandas
- skforecast
- datetime
- numpy
- holidays
- os
- pickle

## Data

The model uses data from several noise sensors located at different points in the city. This data is stored in csv files. The model also uses weather data (temperature, rainfall, and wind speed) and date/time data (time of day, day of week, month, season, etc.) as additional features. The academic calendar of the KU Leuven university is also taken into account.

## Usage

To run the Dash application:

1. Install the required libraries.
2. Clone the [completed app container](https://github.com/uac35/MDA/app_final) and change your working directory to the project folder.
3. Run the script using a Python interpreter. The Dash app will start and can be accessed from a web browser.
4. It is possible to deploy this container directly with Google Cloud Run.

Command to run:

python app_final.py

Then, open a web browser and go to http://localhost:8080 to access the app.

In the app, users can select a date and time and input weather conditions. Upon clicking the 'Predict' button, the app will make a human noise event forecast for the selected date/time and weather conditions. The results will be shown on a heatmap, which also indicates the predicted police activity level for each location. There is also separate page where users can view the previous noise event data of Leuven.

## Model

Two machine learning models are used in this project:

1. A time series forecasting model that predicts human noise event.
2. A binary classifier that predicts the level of police activity ('High' or 'Low').

The models were trained separately and saved to pickle files, which are loaded when the app runs.
