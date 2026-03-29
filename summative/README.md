# Life Expectancy Predictor - Women's Health Mission

## Mission

Predict women's life expectancy using socio-economic and health indicators to support policy decisions for improving global women's health outcomes. The model identifies key factors like education, HIV prevalence, GDP, and healthcare access that influence longevity differences across countries.

## Data Source

WHO Life Expectancy Dataset (Kaggle: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who).  
This dataset contains multivariate country-level features (2011-2015). Target: `Life expectancy ` (years). Features include schooling, GDP, HIV/AIDS, adult mortality, BMI/thinness, vaccination rates, and additional engineered indices.

## Model Implementation

This project implements all required regression models:

- Linear Regression (baseline implementation)
- Gradient Descent (`SGDRegressor` with loss curve visualization)
- Decision Trees
- Random Forest

While linear regression was our starting point, Random Forest achieved superior performance due to non-linear relationships in life-expectancy drivers.

## API Endpoint

- Live API: https://linear-regression-model-x6f4.onrender.com
- Swagger UI: https://linear-regression-model-x6f4.onrender.com/docs

Endpoints:
- `POST /predict`: returns `life_expectancy_years`
- `POST /retrain`: retrains on the base dataset (hot-swaps the best model)
- `POST /retrain_upload`: retrains using an uploaded CSV

## Video Demo

https://youtu.be/PBoF1a4d92c

## Mobile App Setup

1. Run the mobile app:
   `cd summative/FlutterApp/life_predictor_app`  
   `flutter pub get`  
   `flutter run`

App inputs: the app calls `/predict` using the required fields and `status` (must be `Developing` or `Developed`).

