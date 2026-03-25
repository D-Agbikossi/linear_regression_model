# Life Expectancy Predictor - Women's Health Mission

## Mission (4 lines)
**Predict women's life expectancy** using socio-economic & health data to support **policy decisions** for improving global women's health outcomes. Identifies key factors like education, HIV prevalence, GDP, and healthcare access that drive longevity differences across countries.

## Dataset
**WHO Life Expectancy Dataset** (Kaggle: [Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who))  
**Rich multivariate data** (2938 samples, 22+ features post-engineering) from 2011-2015 across countries. Target: Life expectancy (years). Key features: Schooling, GDP, HIV/AIDS, Adult Mortality + 10 engineered (womens_empowerment, healthcare_index, etc.).

## Visualizations in Notebook
1. **Correlation Heatmap**: Schooling(+0.78), GDP(+0.73) strong positive; HIV(-0.59), Mortality(-0.64) negative.
2. **Distributions/Scatter**: Life exp 40-83 years avg 70; Schooling vs Life exp R²=0.61.

## API (Swagger UI)
**Local**: http://localhost:8000/docs  
**Deployed**: https://YOUR_RENDER_SERVICE.onrender.com/docs (replace `YOUR_RENDER_SERVICE`)  
Swagger UI opens at `/docs`.

**Endpoints**:
- `POST /predict`: JSON input → life_expectancy prediction
- `POST /retrain`: Retrains model on dataset (optionally with streamed JSON records)
- `POST /retrain_upload`: Retrains model using an uploaded CSV

**Example Request** (`POST /predict`):
```json
{
  "adult_mortality": 120,
  "infant_deaths": 62,
  "alcohol": 0.8,
  "bmi": 19.1,
  "hiv_aids": 0.1,
  "gdp": 584.26,
  "schooling": 10.1,
  "healthcare_index": 83.0,
  "economic_index": 0.479,
  "womens_empowerment": 0.85,
  "nutrition_index": 0.25,
  "immunization_coverage": 90.0,
  "socioeconomic_health": 0.85,
  "development_stage": 2.0,
  "status": "Developing"
}
```

## Flutter App Instructions
```bash
cd summative/FlutterApp/life_predictor_app
flutter pub get
flutter run  # Android emulator/iOS
```
**Update API URL** in `summative/FlutterApp/life_predictor_app/lib/main.dart`:
replace `https://REPLACE_WITH_RENDER_URL/predict` with your deployed Render URL.

In the app, `status` must be either `Developing` or `Developed`.

## Model Performance
| Model | Test MSE | R² | Notes |
|-------|----------|----|-------|
| Linear Regression | ~25.2 | 0.82 | Fast inference |
| Decision Tree | ~28.1 | 0.79 | Interpretable |
| **Random Forest (BEST)** | **22.4** | **0.85** | Production model |

**Loss low** (MSE~22 years² → RMSE~4.7 years). **Reduce further**: Hyperparam tuning (GridSearchCV done), more data, ensemble.

## Deployment
1. `pip install -r summative/API/requirements.txt`
2. `cd summative/API && uvicorn prediction:app --host 0.0.0.0 --port 8000 --reload`
3. Deploy to [Render](https://render.com) (free tier): New Web Service → GitHub repo → `uvicorn prediction:app --host 0.0.0.0 --port $PORT`

## Video Demo Link
[REPLACE WITH YouTube link - 7min max: App demo + Swagger tests + notebook + Q&A]

## Run Tests
```bash
# Test API
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @- <<'JSON'
{
  "adult_mortality": 120,
  "infant_deaths": 62,
  "alcohol": 0.8,
  "bmi": 19.1,
  "hiv_aids": 0.1,
  "gdp": 584.26,
  "schooling": 10.1,
  "healthcare_index": 83.0,
  "economic_index": 0.479,
  "womens_empowerment": 0.85,
  "nutrition_index": 0.25,
  "immunization_coverage": 90.0,
  "socioeconomic_health": 0.85,
  "development_stage": 2.0,
  "status": "Developing"
}
JSON

# Test retrain  
curl -X POST "http://localhost:8000/retrain"

# Retrain from uploaded CSV:
# curl -X POST "http://localhost:8000/retrain_upload" -F "file=@your_new_data.csv"
```

