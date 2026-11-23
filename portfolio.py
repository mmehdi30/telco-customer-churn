"""Batch churn prediction and portfolio-level analytics."""
from __future__ import annotations

import pandas as pd

from predict import predict_churn


SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
MODEL_INPUT_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges",
]



def prepare_portfolio_data(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare customer data for scoring. Churn labels and extra columns are optional."""
    if uploaded_df is None or uploaded_df.empty:
        raise ValueError("The uploaded CSV is empty.")


    df = uploaded_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Churn is the historical target and is not needed for prediction.
    # Keep optional columns (e.g. customerID, TotalCharges) for reporting/export.
    missing = [c for c in MODEL_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required customer features: " + ", ".join(missing))

    # Normalize whitespace in categorical inputs without changing their meaning.
    for col in MODEL_INPUT_COLUMNS:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string").str.strip()

    # Numeric inputs; invalid values are excluded and reported via the count below.
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")


    # Engineer the feature used by the model from service indicators.
    for col in SERVICE_COLUMNS:
        df[col] = df[col].fillna("No")
    df["NumAdditionalServices"] = df[SERVICE_COLUMNS].eq("Yes").sum(axis=1)



    # Drop rows whose required numeric values cannot be interpreted.
    df = df.dropna(subset=["tenure", "MonthlyCharges", "SeniorCitizen"])
    if df.empty:
        raise ValueError("No valid customer rows remain after numeric validation (tenure, MonthlyCharges, SeniorCitizen).")
    return df



def predict_portfolio(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """Run the existing single-customer predictor for each valid portfolio row."""
    df = prepare_portfolio_data(uploaded_df)
    results = []
    for _, row in df.iterrows():
        customer = {col: row[col] for col in MODEL_INPUT_COLUMNS}
        customer["NumAdditionalServices"] = int(row["NumAdditionalServices"])
        prediction = predict_churn(customer)
        results.append({
            "churn_probability": float(prediction["probability"]),
            "model_prediction": str(prediction["prediction"]),
        })

    result_df = df.reset_index(drop=True).copy()
    predictions_df = pd.DataFrame(results)
    result_df = pd.concat([result_df, predictions_df], axis=1)
    result_df["risk_level"] = pd.cut(
        result_df["churn_probability"],
        bins=[-0.001, 0.15, 0.40, 1.001],
        labels=["Low", "Medium", "High"],
        right=False,
    ).astype(str)
    result_df["expected_monthly_revenue_at_risk"] = (
        result_df["MonthlyCharges"] * result_df["churn_probability"]
    )
    return result_df.sort_values("churn_probability", ascending=False).reset_index(drop=True)



def portfolio_metrics(predictions: pd.DataFrame) -> dict:
    """Calculate portfolio KPIs from batch prediction results."""
    if predictions.empty:
        return {
            "total_customers": 0,
            "high_risk_customers": 0,
            "average_churn_probability": 0.0,
            "monthly_revenue_at_risk": 0.0,
        }
    return {
        "total_customers": int(len(predictions)),
        "high_risk_customers": int((predictions["risk_level"] == "High").sum()),
        "average_churn_probability": float(predictions["churn_probability"].mean()),
        "monthly_revenue_at_risk": float(predictions["expected_monthly_revenue_at_risk"].sum()),
    }
