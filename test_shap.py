
import json
import joblib
import shap
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.special import expit
from scipy.sparse import issparse

# --------------------------------------------------
# 1. Load model and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "telco_churn_model.pkl")

with open(MODEL_DIR / "model_features.json") as f:
    selected_features = json.load(f)

# Get the fitted preprocessing and classification steps
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

# --------------------------------------------------
# 2. Load background dataset
# --------------------------------------------------

background = pd.read_csv(MODEL_DIR / "shap_background.csv")
background = background[selected_features]

print("Background shape:", background.shape)

# --------------------------------------------------
# 3. Transform background data
# --------------------------------------------------

background_transformed = preprocessor.transform(background)

if issparse(background_transformed):
    background_transformed = background_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

background_transformed = pd.DataFrame(
    background_transformed,
    columns=feature_names
)

print("Transformed background shape:", background_transformed.shape)

# --------------------------------------------------
# 4. Create Linear SHAP explainer
# --------------------------------------------------

explainer = shap.LinearExplainer(
    classifier,
    background_transformed
)

# --------------------------------------------------
# 5. Select one customer to explain
# --------------------------------------------------

sample_customer = background.iloc[[0]]

sample_transformed = preprocessor.transform(sample_customer)

if issparse(sample_transformed):
    sample_transformed = sample_transformed.toarray()

sample_transformed = pd.DataFrame(
    sample_transformed,
    columns=feature_names
)

# --------------------------------------------------
# 6. Calculate SHAP values
# --------------------------------------------------

explanation = explainer(sample_transformed)


# --------------------------------------------------
# 1. Aggregate SHAP values by original feature
# --------------------------------------------------

grouped_shap_values = []
grouped_feature_names = []

for original_feature in selected_features:

    # Find the transformed columns belonging to this feature
    numerical_name = f"num__{original_feature}"
    categorical_prefix = f"cat__{original_feature}_"

    matching_indices = [
        i for i, name in enumerate(feature_names)
        if name == numerical_name or name.startswith(categorical_prefix)
    ]

    if not matching_indices:
        raise ValueError(
            f"No transformed columns found for {original_feature}"
        )

    # Sum the contributions of all encoded columns
    contribution = explanation.values[0, matching_indices].sum()

    grouped_shap_values.append(contribution)
    grouped_feature_names.append(original_feature)


# --------------------------------------------------
# 2. Build the grouped SHAP explanation
# --------------------------------------------------

grouped_shap_values = np.array(grouped_shap_values)

# Original, unscaled customer values for display
original_values = sample_customer.iloc[0][
    selected_features
].to_numpy(dtype=object)

base_value = float(
    np.asarray(explanation.base_values).reshape(-1)[0]
)

grouped_explanation = shap.Explanation(
    values=grouped_shap_values,
    base_values=base_value,
    data=original_values,
    feature_names=grouped_feature_names
)


# --------------------------------------------------
# 3. Verify additivity
# --------------------------------------------------

yes_index = list(model.classes_).index("Yes")

actual_probability = model.predict_proba(
    sample_customer
)[0, yes_index]

actual_log_odds = classifier.decision_function(
    sample_transformed.to_numpy()
)[0]

reconstructed_log_odds = (
    base_value + grouped_shap_values.sum()
)

reconstructed_probability = expit(
    reconstructed_log_odds
)

print("\n--- Grouped SHAP Verification ---")
print("Actual churn probability:", actual_probability)
print("Reconstructed probability:", reconstructed_probability)
print("Actual log-odds:", actual_log_odds)
print("Reconstructed log-odds:", reconstructed_log_odds)

print(
    "Additivity verified:",
    np.isclose(
        actual_log_odds,
        reconstructed_log_odds,
        atol=1e-5
    )
)


# --------------------------------------------------
# 4. Display grouped contributions
# --------------------------------------------------

results = pd.DataFrame({
    "Feature": grouped_feature_names,
    "Customer Value": original_values,
    "SHAP Value": grouped_shap_values
})

results["Absolute SHAP"] = results["SHAP Value"].abs()

results = results.sort_values(
    "Absolute SHAP",
    ascending=False
)

print("\n--- Grouped SHAP Contributions ---")
print(results.to_string(index=False))


# --------------------------------------------------
# 5. Generate the grouped waterfall plot
# --------------------------------------------------

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))

shap.plots.waterfall(
    grouped_explanation,
    max_display=19,
    show=False
)

plt.title("SHAP Explanation - Customer Churn")
plt.tight_layout()

plt.savefig(
    "shap_waterfall_grouped.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("\nGrouped SHAP experiment completed successfully!")

