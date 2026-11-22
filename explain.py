
import json
import joblib
import shap
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.special import expit
from scipy.sparse import issparse


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# Load model and SHAP resources
# --------------------------------------------------

def load_explainer_resources():

    model = joblib.load(MODEL_DIR / "telco_churn_model.pkl")

    with open(MODEL_DIR / "model_features.json", "r") as f:
        selected_features = json.load(f)

    background = pd.read_csv(
        MODEL_DIR / "shap_background.csv"
    )

    background = background[selected_features]

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    # Transform the background using the trained preprocessor
    background_transformed = preprocessor.transform(background)

    if issparse(background_transformed):
        background_transformed = background_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    background_transformed = pd.DataFrame(
        background_transformed,
        columns=feature_names
    )

    # Create the SHAP explainer
    explainer = shap.LinearExplainer(
        classifier,
        background_transformed
    )

    return {
        "model": model,
        "selected_features": selected_features,
        "preprocessor": preprocessor,
        "classifier": classifier,
        "feature_names": feature_names,
        "explainer": explainer
    }


# --------------------------------------------------
# Explain an individual customer
# --------------------------------------------------

def explain_customer(customer_data, resources):

    model = resources["model"]
    selected_features = resources["selected_features"]
    preprocessor = resources["preprocessor"]
    classifier = resources["classifier"]
    feature_names = resources["feature_names"]
    explainer = resources["explainer"]

    # Prepare customer data
    customer_df = pd.DataFrame(
        [customer_data],
        columns=selected_features
    )

    # Transform customer data
    customer_transformed = preprocessor.transform(customer_df)

    if issparse(customer_transformed):
        customer_transformed = customer_transformed.toarray()

    customer_transformed = pd.DataFrame(
        customer_transformed,
        columns=feature_names
    )

    # Calculate SHAP values
    explanation = explainer(customer_transformed)

    # Aggregate encoded contributions into original features
    grouped_shap_values = []

    for original_feature in selected_features:

        numerical_name = f"num__{original_feature}"
        categorical_prefix = f"cat__{original_feature}_"

        matching_indices = [
            i for i, name in enumerate(feature_names)
            if name == numerical_name
            or name.startswith(categorical_prefix)
        ]

        if not matching_indices:
            raise ValueError(
                f"No transformed columns found for {original_feature}"
            )

        contribution = explanation.values[
            0, matching_indices
        ].sum()

        grouped_shap_values.append(contribution)

    grouped_shap_values = np.array(grouped_shap_values)

    # Original customer values, not standardized values
    original_values = customer_df.iloc[0][
        selected_features
    ].to_numpy(dtype=object)

    base_value = float(
        np.asarray(explanation.base_values).reshape(-1)[0]
    )

    # Create grouped explanation
    grouped_explanation = shap.Explanation(
        values=grouped_shap_values,
        base_values=base_value,
        data=original_values,
        feature_names=selected_features
    )

    # Calculate churn probability
    yes_index = list(model.classes_).index("Yes")

    churn_probability = model.predict_proba(
        customer_df
    )[0, yes_index]

    # Verify SHAP additivity
    actual_log_odds = classifier.decision_function(
        customer_transformed.to_numpy()
    )[0]

    reconstructed_log_odds = (
        base_value + grouped_shap_values.sum()
    )

    additivity_verified = np.isclose(
        actual_log_odds,
        reconstructed_log_odds,
        atol=1e-5
    )

    return {
        "explanation": grouped_explanation,
        "probability": float(churn_probability),
        "base_value": base_value,
        "additivity_verified": bool(additivity_verified)
    }
