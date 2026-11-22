
from explain import load_explainer_resources, explain_customer
import json
import pandas as pd


# Load resources
resources = load_explainer_resources()

# Load one customer from the SHAP background dataset
customer = pd.read_csv(
    "models/shap_background.csv"
).iloc[0].to_dict()

# Generate explanation
result = explain_customer(customer, resources)

print("Churn probability:", result["probability"])
print("Additivity verified:", result["additivity_verified"])

# Display grouped contributions
explanation = result["explanation"]

results = pd.DataFrame({
    "Feature": explanation.feature_names,
    "Customer Value": explanation.data,
    "SHAP Value": explanation.values
})

results["Absolute SHAP"] = results["SHAP Value"].abs()

results = results.sort_values(
    "Absolute SHAP",
    ascending=False
)

print("\nGrouped SHAP Contributions:")
print(results.to_string(index=False))


# Generate waterfall plot
import matplotlib.pyplot as plt
import shap

fig = plt.figure(figsize=(12, 8))

shap.plots.waterfall(
    explanation,
    max_display=19,
    show=False
)

plt.tight_layout()
plt.savefig(
    "test_explain_waterfall.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("\nExplanation test completed successfully!")
