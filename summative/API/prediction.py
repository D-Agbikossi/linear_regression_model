from __future__ import annotations

from enum import Enum
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

LINEAR_DIR = Path(__file__).resolve().parent.parent / "linear_regression"
MODEL_PATH = LINEAR_DIR / "best_womenlife_model.pkl"
SCALER_PATH = LINEAR_DIR / "scaler.pkl"
FEATURE_COLS_PATH = LINEAR_DIR / "feature_columns.pkl"
LABEL_ENC_PATH = LINEAR_DIR / "label_encoders.pkl"
DATA_PATH = LINEAR_DIR / "Life Expectancy Data.csv"

# Globals (hot-swapped on retrain)
model: Any = None
scaler: StandardScaler | None = None
feature_columns: list[str] | None = None
label_encoders: dict[str, LabelEncoder] | None = None


def load_artifacts() -> None:
    global model, scaler, feature_columns, label_encoders
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLS_PATH)
    # label_encoders is required to encode the categorical 'Status' feature
    label_encoders = joblib.load(LABEL_ENC_PATH) if LABEL_ENC_PATH.exists() else {}


load_artifacts()

app = FastAPI(title="Life Expectancy Predictor API", version="1.1.0")

# CORS - specific origins (no wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "http://localhost:8081",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class StatusEnum(str, Enum):
    Developing = "Developing"
    Developed = "Developed"


class PredictionInput(BaseModel):
    """Inputs must match the exact notebook-trained feature space."""

    adult_mortality: float = Field(..., ge=0, le=1000)
    infant_deaths: float = Field(..., ge=0, le=200)
    alcohol: float = Field(..., ge=0, le=20)
    bmi: float = Field(..., ge=10, le=50)
    hiv_aids: float = Field(..., ge=0, le=50)
    gdp: float = Field(..., ge=0, le=5_000_000)
    schooling: float = Field(..., ge=0, le=25)

    healthcare_index: float = Field(..., ge=0, le=100)
    economic_index: float = Field(..., ge=0, le=1.0)
    womens_empowerment: float = Field(..., ge=0, le=1.0)
    # notebook computes nutrition_index as mean(BMI and thinness) / 100
    nutrition_index: float = Field(..., ge=0, le=1.0)
    immunization_coverage: float = Field(..., ge=0, le=100)
    socioeconomic_health: float = Field(..., ge=0, le=1.0)
    # (GDP quartile code + schooling quartile code) / 2  -> range [0, 3]
    development_stage: float = Field(..., ge=0, le=3.0)

    status: StatusEnum


def _encode_status(status: StatusEnum) -> float:
    if not label_encoders or "Status" not in label_encoders:
        # Fallback: try to keep it stable with a simple mapping
        return 1.0 if status == StatusEnum.Developed else 0.0
    le = label_encoders["Status"]
    return float(le.transform([status.value])[0])


def build_feature_row(data: PredictionInput) -> pd.DataFrame:
    """
    Create a single-row DataFrame matching `feature_columns` order exactly.
    The notebook used original dataset column names, including whitespace.
    """

    assert feature_columns is not None

    row: dict[str, float] = {
        "Adult Mortality": float(data.adult_mortality),
        "infant deaths": float(data.infant_deaths),
        "Alcohol": float(data.alcohol),
        " BMI ": float(data.bmi),
        " HIV/AIDS": float(data.hiv_aids),
        "GDP": float(data.gdp),
        "Schooling": float(data.schooling),
        "healthcare_index": float(data.healthcare_index),
        "economic_index": float(data.economic_index),
        "womens_empowerment": float(data.womens_empowerment),
        "nutrition_index": float(data.nutrition_index),
        "immunization_coverage": float(data.immunization_coverage),
        "socioeconomic_health": float(data.socioeconomic_health),
        "development_stage": float(data.development_stage),
        "Status": _encode_status(data.status),
    }

    input_df = pd.DataFrame([row])

    # Safety: ensure every expected feature exists (model expects fixed width)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0.0

    return input_df[feature_columns]


def predict_life_expectancy_row(input_df: pd.DataFrame) -> float:
    """Run best-model prediction; apply scaling only when the model is linear."""
    assert scaler is not None
    assert feature_columns is not None

    # Tree-based models were trained on unscaled features in the notebook.
    tree_like = hasattr(model, "feature_importances_")

    X = input_df.values if tree_like else scaler.transform(input_df[feature_columns])
    pred = model.predict(X)[0]
    return float(np.clip(pred, 40, 90))


def predict_life_expectancy(features: dict[str, Any]) -> float:
    """
    Public prediction function used by the API (and available for Task 2).
    Expects the same snake_case keys as `PredictionInput`.
    """
    data = PredictionInput(**features)
    input_df = build_feature_row(data)
    return predict_life_expectancy_row(input_df)


@app.post("/predict", summary="Predict life expectancy for a country")
def predict_endpoint(data: PredictionInput):
    try:
        input_df = build_feature_row(data)
        prediction = predict_life_expectancy_row(input_df)
        return {
            "life_expectancy_years": prediction,
            "model_used": type(model).__name__,
            "features_used": len(feature_columns or []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


class RetrainRecord(PredictionInput):
    """New record in the model's feature space + target for supervised retraining."""

    life_expectancy_years: float = Field(..., ge=0, le=120)


class RetrainRequest(BaseModel):
    records: list[RetrainRecord] | None = None


def _engineer_features_like_notebook(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering kept consistent with the training notebook.
    NOTE: we intentionally mirror the notebook's exact column-name checks
    (including whitespace) so retraining stays compatible with the saved
    `feature_columns.pkl`.
    """

    global label_encoders
    df = df_raw.copy()

    if "Country" in df.columns:
        df = df.drop(columns=["Country"], errors="ignore")

    # notebook dropped NaNs early, then also filled numeric medians as a safety check
    df = df.dropna()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Encode categoricals (Status) using saved label_encoders when possible
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if label_encoders and col in label_encoders:
            le = label_encoders[col]
            try:
                df[col] = le.transform(df[col].astype(str))
            except Exception:
                # New categories in uploaded data; refit and hot-swap encoder.
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            if label_encoders is None:
                label_encoders = {}
            label_encoders[col] = le

    # 1. Healthcare Access Index (combination of immunization rates)
    healthcare_cols = []
    if "Polio" in df.columns:
        healthcare_cols.append("Polio")
    # mirror notebook's (possibly-wrong) check: dataset has 'Diphtheria ' with trailing space
    if "Diphtheria" in df.columns:
        healthcare_cols.append("Diphtheria")
    if "Hepatitis B" in df.columns:
        healthcare_cols.append("Hepatitis B")
    if healthcare_cols:
        df["healthcare_index"] = df[healthcare_cols].mean(axis=1)

    # 2. Economic Development Index
    if "GDP" in df.columns and "Total expenditure" in df.columns:
        gdp_std = (df["GDP"] - df["GDP"].min()) / (df["GDP"].max() - df["GDP"].min())
        exp_std = (df["Total expenditure"] - df["Total expenditure"].min()) / (
            df["Total expenditure"].max() - df["Total expenditure"].min()
        )
        df["economic_index"] = (gdp_std + exp_std) / 2

    # 3. Child Health Index
    # mirror notebook's exact check: dataset uses 'under-five deaths ' (trailing space)
    if "infant deaths" in df.columns and "under-five deaths" in df.columns:
        max_infant = df["infant deaths"].max()
        max_under5 = df["under-five deaths"].max()
        infant_score = 1 - (df["infant deaths"] / max_infant if max_infant > 0 else 0)
        under5_score = 1 - (df["under-five deaths"] / max_under5 if max_under5 > 0 else 0)
        df["child_health_index"] = (infant_score + under5_score) / 2

    # 4. Women's Empowerment Index
    if "Schooling" in df.columns and "Income composition of resources" in df.columns:
        schooling_norm = df["Schooling"] / df["Schooling"].max()
        income_norm = df["Income composition of resources"]
        df["womens_empowerment"] = (schooling_norm + income_norm) / 2

    # 5. Disease Burden Index
    # mirror notebook's exact check: dataset uses ' HIV/AIDS' (leading space)
    if "HIV/AIDS" in df.columns and "Adult Mortality" in df.columns:
        hiv_norm = df["HIV/AIDS"] / df["HIV/AIDS"].max()
        mortality_norm = df["Adult Mortality"] / df["Adult Mortality"].max()
        df["disease_burden"] = (hiv_norm + mortality_norm) / 2

    # 6. Nutrition Index
    bmi_cols = []
    if " BMI " in df.columns:
        bmi_cols.append(" BMI ")
    if " thinness 1-19 years" in df.columns:
        bmi_cols.append(" thinness 1-19 years")
    if " thinness 5-9 years" in df.columns:
        bmi_cols.append(" thinness 5-9 years")
    if bmi_cols:
        df["nutrition_index"] = df[bmi_cols].mean(axis=1) / 100 if " BMI " in df.columns else df[
            bmi_cols
        ].mean(axis=1)

    # 7. Development Stage
    if "GDP" in df.columns and "Schooling" in df.columns:
        gdp_quartile = pd.qcut(df["GDP"], 4, labels=["Low", "Medium-Low", "Medium-High", "High"])
        schooling_quartile = pd.qcut(
            df["Schooling"], 4, labels=["Low", "Medium-Low", "Medium-High", "High"]
        )
        df["development_stage"] = (
            pd.Categorical(gdp_quartile).codes + pd.Categorical(schooling_quartile).codes
        ) / 2

    # 9. Immunization Coverage
    vaccine_cols = []
    for col in ["Polio", "Diphtheria ", "Hepatitis B"]:
        if col in df.columns:
            vaccine_cols.append(col)
    if vaccine_cols:
        df["immunization_coverage"] = df[vaccine_cols].mean(axis=1)

    # 10. Socioeconomic Health Index
    socio_cols = []
    if "womens_empowerment" in df.columns:
        socio_cols.append("womens_empowerment")
    if "economic_index" in df.columns:
        socio_cols.append("economic_index")
    if "child_health_index" in df.columns:
        socio_cols.append("child_health_index")
    if socio_cols:
        df["socioeconomic_health"] = df[socio_cols].mean(axis=1)

    assert feature_columns is not None
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0

    return df


def retrain_best_model_from_features(
    X: pd.DataFrame, y: pd.Series, model_candidates: dict[str, Any]
) -> tuple[str, Any, StandardScaler, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    local_scaler = StandardScaler()
    X_train_scaled = local_scaler.fit_transform(X_train)
    X_test_scaled = local_scaler.transform(X_test)

    best_name = ""
    best_model = None
    best_mse = float("inf")

    for name, mdl in model_candidates.items():
        if "SGDRegressor" in name or "Linear" in name:
            mdl.fit(X_train_scaled, y_train)
            preds = mdl.predict(X_test_scaled)
        else:
            mdl.fit(X_train, y_train)
            preds = mdl.predict(X_test)

        mse = float(mean_squared_error(y_test, preds))
        if mse < best_mse:
            best_name = name
            best_model = mdl
            best_mse = mse

    assert best_model is not None
    return best_name, best_model, local_scaler, best_mse


def retrain_and_swap(df_raw_combined: pd.DataFrame) -> dict[str, Any]:
    global model, scaler, feature_columns, label_encoders

    df_feat = _engineer_features_like_notebook(df_raw_combined)
    assert feature_columns is not None

    X = df_feat[feature_columns]
    y = df_feat["Life expectancy "]

    candidates: dict[str, Any] = {
        "Gradient Descent LR (SGDRegressor)": SGDRegressor(
            max_iter=5000,
            tol=1e-4,
            learning_rate="adaptive",
            eta0=0.01,
            random_state=42,
        ),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    best_name, best_model, best_scaler, best_mse = retrain_best_model_from_features(
        X=X, y=y, model_candidates=candidates
    )

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(best_scaler, SCALER_PATH)
    joblib.dump(label_encoders or {}, LABEL_ENC_PATH)
    joblib.dump(feature_columns, FEATURE_COLS_PATH)

    # Swap in-memory artifacts so subsequent /predict calls use the new model
    model = best_model
    scaler = best_scaler

    return {
        "status": "success",
        "best_model": best_name,
        "best_mse": float(best_mse),
        "n_samples": int(len(df_raw_combined)),
        "n_features": int(len(feature_columns)),
    }


@app.post("/retrain", summary="Retrain best model from streamed JSON records")
def retrain_endpoint(req: RetrainRequest | None = None):
    try:
        base_df = pd.read_csv(DATA_PATH)

        if req is None or not req.records:
            return retrain_and_swap(base_df)

        # records are in feature-space, so we combine them after feature engineering base_df
        df_base_feat = _engineer_features_like_notebook(base_df)
        assert feature_columns is not None

        rows = []
        targets = []
        for r in req.records:
            row_df = build_feature_row(r)
            rows.append(row_df.iloc[0].to_dict())
            targets.append(float(r.life_expectancy_years))

        df_new_feat = pd.DataFrame(rows)
        for col in feature_columns:
            if col not in df_new_feat.columns:
                df_new_feat[col] = 0.0
        df_new_feat = df_new_feat[feature_columns]

        X = pd.concat([df_base_feat[feature_columns], df_new_feat], ignore_index=True)
        y = pd.concat([df_base_feat["Life expectancy "], pd.Series(targets)], ignore_index=True)

        candidates: dict[str, Any] = {
            "Gradient Descent LR (SGDRegressor)": SGDRegressor(
                max_iter=5000,
                tol=1e-4,
                learning_rate="adaptive",
                eta0=0.01,
                random_state=42,
            ),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            ),
        }
        best_name, best_model, best_scaler, best_mse = retrain_best_model_from_features(
            X=X, y=y, model_candidates=candidates
        )

        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(best_scaler, SCALER_PATH)
        joblib.dump(label_encoders or {}, LABEL_ENC_PATH)
        joblib.dump(feature_columns, FEATURE_COLS_PATH)

        model = best_model
        scaler = best_scaler

        return {
            "status": "success",
            "best_model": best_name,
            "best_mse": float(best_mse),
            "n_samples": int(len(X)),
            "n_features": int(len(feature_columns)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@app.post("/retrain_upload", summary="Retrain best model from uploaded CSV")
def retrain_upload(file: UploadFile = File(...)):
    try:
        df_new_raw = pd.read_csv(file.file)
        base_df = pd.read_csv(DATA_PATH)
        df_combined = pd.concat([base_df, df_new_raw], ignore_index=True)
        return retrain_and_swap(df_combined)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV retraining failed: {str(e)}")


@app.get("/")
def root():
    return {"message": "Life Expectancy Predictor API", "endpoints": ["/predict", "/retrain", "/retrain_upload", "/docs"]}
