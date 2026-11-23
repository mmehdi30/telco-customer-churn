"""Hypothetical retention campaign financial simulation."""
from __future__ import annotations
import pandas as pd


def simulate_campaign(
    customers: pd.DataFrame,
    cost_per_customer: float,
    success_rate: float,
    months_saved: int,
) -> dict:
    """Estimate campaign outcomes using churn probabilities and stated assumptions.


    success_rate is treated as the share of expected churn that the campaign prevents.
    This is a scenario assumption, not an experimentally measured campaign effect.
    """
    if customers is None or customers.empty:
        raise ValueError("No customers selected for the campaign.")
    if cost_per_customer < 0:
        raise ValueError("Cost per customer cannot be negative.")
    if not 0 <= success_rate <= 1:
        raise ValueError("Success rate must be between 0 and 1.")
    if months_saved < 1:
        raise ValueError("Revenue horizon must be at least one month.")
    required = {"churn_probability", "MonthlyCharges"}
    missing = required - set(customers.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    probability = pd.to_numeric(customers["churn_probability"], errors="coerce").clip(0, 1).fillna(0)
    charges = pd.to_numeric(customers["MonthlyCharges"], errors="coerce").fillna(0).clip(lower=0)
    n = len(customers)
    expected_churn_without_campaign = float(probability.sum())
    expected_churn_prevented = expected_churn_without_campaign * success_rate
    campaign_cost = n * float(cost_per_customer)
    estimated_revenue_saved = float((probability * charges * months_saved * success_rate).sum())
    net_benefit = estimated_revenue_saved - campaign_cost
    if campaign_cost > 0:
        roi = (net_benefit / campaign_cost) * 100
    else:
        roi = None

    return {
        "customers_targeted": int(n),
        "expected_churn_without_campaign": expected_churn_without_campaign,
        "expected_churn_prevented": expected_churn_prevented,
        "campaign_cost": campaign_cost,
        "estimated_revenue_saved": estimated_revenue_saved,
        "net_benefit": net_benefit,
        "roi_percent": roi,
    }
