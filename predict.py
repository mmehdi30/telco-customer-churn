
import joblib
import json
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Load model
model = joblib.load(
    MODEL_DIR / "telco_churn_model.pkl"
)

# Load required features
with open(MODEL_DIR / "model_features.json", "r") as f:
    selected_features = json.load(f)

# Load classification threshold
with open(MODEL_DIR / "model_config.json", "r") as f:
    config = json.load(f)

THRESHOLD = config["threshold"]


def predict_churn(customer_data):

    # Convert input dictionary into DataFrame
    customer_df = pd.DataFrame(
        [customer_data],
        columns=selected_features
    )

    # Predict probability of churn
    yes_index = list(model.classes_).index("Yes")

    churn_probability = model.predict_proba(
        customer_df
    )[0, yes_index]

    # Apply selected threshold
    prediction = (
        "Yes"
        if churn_probability >= THRESHOLD
        else "No"
    )

    return {
        "prediction": prediction,
        "probability": float(churn_probability),
        "threshold": THRESHOLD
    }
