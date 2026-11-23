"""Rule-based churn retention recommendations, ranked with optional SHAP signals."""

FEATURES_BY_RULE = {
    "contract_incentive": ["Contract"],
    "onboarding_followup": ["tenure"],
    "security_review": ["OnlineSecurity"],
    "support_outreach": ["TechSupport"],
    "backup_review": ["OnlineBackup"],
    "protection_review": ["DeviceProtection"],
    "payment_experience": ["PaymentMethod"],
    "bill_review": ["MonthlyCharges"],
    "retention_contact": [],
}



def _positive_shap_score(features, shap_contributions):
    """Sum positive grouped SHAP contributions for matching original features."""
    if not shap_contributions:
        return None
    score = 0.0
    matched = False
    for feature in features:
        for name, value in shap_contributions.items():
            # Grouped feature names usually look like 'Contract Two year' or 'tenure 51'.
            if str(name) == feature or str(name).startswith(feature + " "):
                matched = True
                score += max(0.0, float(value))
    return score if matched else None



def get_retention_recommendations(customer, churn_probability, shap_contributions=None):
    """Create rule-based recommendations and rank them using positive SHAP influence.



    SHAP is used only as a model-alignment heuristic. It does not measure the
    effectiveness or causal impact of a retention intervention.
    """
    shap_contributions = shap_contributions or {}
    probability = float(churn_probability)
    if probability >= 0.70:
        risk_level, risk_priority = "High", "Urgent"
    elif probability >= 0.40:
        risk_level, risk_priority = "Medium", "Normal"
    else:
        risk_level, risk_priority = "Low", "Routine"



    recommendations = []
    def add(rule_id, title, reason, action, features=None, priority="Normal"):
        features = features or FEATURES_BY_RULE.get(rule_id, [])
        score = _positive_shap_score(features, shap_contributions)
        recommendations.append({
            "rule_id": rule_id,
            "title": title,
            "reason": reason,
            "action": action,
            "priority": priority,
            "shap_score": score,
            "matched_features": features,
        })



    if customer.get("Contract") == "Month-to-month":
        add("contract_incentive", "Contract incentive", "Customer is on a month-to-month contract.", "Offer a suitable longer-term plan incentive.", ["Contract"])
    if customer.get("tenure", 0) <= 6:
        add("onboarding_followup", "New-customer onboarding", "Customer is in the early tenure period.", "Schedule a proactive onboarding check-in.", ["tenure"])
    if customer.get("OnlineSecurity") == "No":
        add("security_review", "Online security review", "Online security service is not enabled.", "Explain available security options and assess interest.", ["OnlineSecurity"])
    if customer.get("TechSupport") == "No":
        add("support_outreach", "Proactive technical support", "Technical support is not enabled.", "Offer a support check or explain support options.", ["TechSupport"])
    if customer.get("OnlineBackup") == "No":
        add("backup_review", "Online backup review", "Online backup is not enabled.", "Review backup needs and available plans.", ["OnlineBackup"])
    if customer.get("DeviceProtection") == "No":
        add("protection_review", "Device protection review", "Device protection is not enabled.", "Review device-protection options.", ["DeviceProtection"])
    if customer.get("PaymentMethod") == "Electronic check":
        add("payment_experience", "Payment experience review", "Customer uses electronic check.", "Offer convenient payment options and explain setup.", ["PaymentMethod"])
    if float(customer.get("MonthlyCharges", 0) or 0) >= 80:
        add("bill_review", "Billing review", "Monthly charges are relatively high.", "Review the bill and check whether a better-fitting plan is available.", ["MonthlyCharges"])
    if probability >= 0.70:
        add("retention_contact", "Priority retention contact", "Model estimates a high churn probability.", "Arrange a timely personal contact to understand concerns and available solutions.", [])
    if not recommendations:
        add("general_satisfaction", "Customer satisfaction check", "No specific retention rule was triggered.", "Maintain regular satisfaction monitoring.", [])



    # When SHAP is available, sort recommendations with a matching positive
    # contribution first; otherwise preserve the rule-generated order.
    if shap_contributions:
        recommendations.sort(
            key=lambda r: (r["shap_score"] is not None, r["shap_score"] or 0),
            reverse=True,
        )
        for index, rec in enumerate(recommendations):
            rec["shap_rank"] = index + 1 if rec["shap_score"] is not None else None



    return {
        "risk_level": risk_level,
        "priority": risk_priority,
        "recommendations": recommendations,
    }
