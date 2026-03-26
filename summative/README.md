# Life Expectancy Predictor - Women's Health Mission

Mission
Predict women's life expectancy using socio-economic & health indicators to support policy decisions for improving global women's health outcomes. The model identifies key factors like education, HIV prevalence, GDP, and healthcare access that influence longevity differences across countries.

Data Source
WHO Life Expectancy Dataset (Kaggle: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
This dataset contains multivariate country-level features (2011–2015). Target: `Life expectancy ` (years). Features include schooling, GDP, HIV/AIDS, adult mortality, BMI/thinness, vaccination rates, and additional engineered indices.

Model Implementation
This project implements ALL required regression models:

    Linear Regression (baseline; uses standardized features)
    Gradient Descent (SGDRegressor; loss curve visualization on train/test)
    Decision Trees
    Random Forest

Feature engineering and interpretation are prioritized (creation of indices like `womens_empowerment`, `healthcare_index`, and `disease_burden`), then data is standardized for linear/SGD models. The best-performing model is saved and deployed via the API.

API Endpoint
Live API: https://YOUR_RENDER_SERVICE.onrender.com/predict (replace `YOUR_RENDER_SERVICE`)
Swagger UI: https://YOUR_RENDER_SERVICE.onrender.com/docs

Endpoints:
- `POST /predict`: returns `life_expectancy_years`
- `POST /retrain`: retrains on the base dataset (hot-swaps the best model)
- `POST /retrain_upload`: retrains using an uploaded CSV

Video Demo
https://youtu.be/REPLACE_WITH_YOUR_VIDEO_ID

Mobile App Setup
Steps to run and test (also used for the demo recording):

    1. Start the API locally:
       cd summative/API
       pip install -r requirements.txt
       uvicorn prediction:app --host 0.0.0.0 --port 8000 --reload
    2. Open Swagger UI (datatype/range checks):
       http://localhost:8000/docs
    3. Update the Flutter API URL:
       In `summative/FlutterApp/life_predictor_app/lib/main.dart`, replace:
       'https://REPLACE_WITH_RENDER_URL/predict'
       with your Render prediction URL (or use localhost if you are testing locally).
    4. Run the mobile app:
       cd summative/FlutterApp/life_predictor_app
       flutter pub get
       flutter run

App inputs:
The app calls `/predict` using the required fields and `status` (must be `Developing` or `Developed`).

