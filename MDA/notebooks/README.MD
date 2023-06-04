# Forecasting Noise Events

This project focuses on forecasting noise events using a multivariate time series approach. The goal is to predict the occurrence of noise events at different locations based on various features such as weather conditions, day of the week, time of day, holidays, and other contextual factors.

## Data

The data used in this project consists of hourly observations of noise events at different locations. The dataset includes the following columns:

- `timestamp`: The timestamp of the observation.
- `location_csv`: The location identifier where the noise event was recorded.
- `human_noise`: The target variable indicating the occurrence of a noise event.
- Other features such as weather conditions, day of the week, time of day, holidays, etc.

## Feature Engineering

To improve the forecasting model's performance, several additional features were engineered based on the timestamp:

- `is_weekend`: A binary feature indicating whether the day is a weekend.
- `day_of_week`: The name of the day of the week.
- `hour_of_day`: The hour of the day.
- `time_of_day`: A categorical feature indicating the time of day (night, morning, afternoon, evening).
- `season`: The season based on the month (winter, spring, summer, fall).
- `month`: The name of the month.
- `day_of_month`: The day of the month.
- `quarter`: The quarter of the year.
- `is_be_holiday`: A binary feature indicating whether it is a Belgian holiday.
- `is_business_day`: A binary feature indicating whether it is a business day.
- `missing`: A binary feature indicating whether the response variable is missing.
- `exams`: A binary feature indicating whether it is an exam period.
- `first_exam_weeks`: A binary feature indicating whether it is the first exam period.
- `final_exam_week`: A binary feature indicating whether it is the final exam week.
- `student_vacation`: A binary feature indicating whether it is a student vacation period.

## Forecasting Model

The forecasting model used in this project is based on the LightGBM algorithm. It is a gradient boosting framework that uses tree-based learning algorithms. The model is trained on the historical data and exogenous features to predict the occurrence of noise events in the future.

The model was trained and evaluated for each location separately. Grid search was used to find the best hyperparameters for each location, considering metrics such as Cohen's kappa, F1-score, recall, and precision. The best model for each location was then tested using backtesting, and the performance metrics were calculated.

The final models were trained on all available data and saved for future use.

## Requirements

The following Python libraries are required to run the code:

- `numpy`
- `pandas`
- `scikit-learn`
- `lightgbm`
- `holidays`
