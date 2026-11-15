
import streamlit as st

from predict import predict_churn
from translations import TRANSLATIONS


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telco Customer Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. LANGUAGE CONFIGURATION
# =========================================================

with st.sidebar:
    st.title("⚙️ Settings")

    language = st.selectbox(
        "Language / Langue",
        ["English", "Français"]
    )

t = TRANSLATIONS[language]


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def translated_selectbox(label_key, options):
    """
    Display translated options while returning
    the original English values expected by the model.

    options: list of (original_value, translation_key)
    """

    values = [value for value, key in options]

    def format_option(value):
        for original_value, translation_key in options:
            if value == original_value:
                return t[translation_key]
        return str(value)

    return st.selectbox(
        t[label_key],
        values,
        format_func=format_option
    )


def get_risk_level(probability):
    """
    Presentation categories.
    These are separate from the model classification threshold.
    """

    if probability < 0.15:
        return t["low_risk"], "success"

    elif probability < 0.40:
        return t["moderate_risk"], "warning"

    else:
        return t["high_risk"], "error"


# =========================================================
# 4. PAGE HEADER
# =========================================================

st.title("📊 " + t["page_title"])
st.caption(t["subtitle"])

st.divider()


# =========================================================
# 5. CUSTOMER INFORMATION
# =========================================================

st.header("👤 " + t["customer_info"])

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = translated_selectbox(
            "gender",
            [
                ("Female", "female"),
                ("Male", "male")
            ]
        )

        senior = st.selectbox(
            t["senior"],
            [0, 1],
            format_func=lambda x: t["yes"] if x == 1 else t["no"]
        )

    with col2:

        partner = translated_selectbox(
            "partner",
            [
                ("Yes", "yes"),
                ("No", "no")
            ]
        )

        dependents = translated_selectbox(
            "dependents",
            [
                ("Yes", "yes"),
                ("No", "no")
            ]
        )

    with col3:

        tenure = st.slider(
            t["tenure"],
            min_value=0,
            max_value=72,
            value=12
        )


# =========================================================
# 6. INTERNET AND ADDITIONAL SERVICES
# =========================================================

st.header("🌐 " + t["services"])

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:

        phone = translated_selectbox(
            "phone",
            [
                ("Yes", "yes"),
                ("No", "no")
            ]
        )

        internet = translated_selectbox(
            "internet",
            [
                ("DSL", "dsl"),
                ("Fiber optic", "fiber"),
                ("No", "no")
            ]
        )

    with col2:

        if phone == "No":

            multiple_lines = "No phone service"
            st.text_input(
                t["multiple_lines"],
                value=t["no_phone"],
                disabled=True
            )

        else:

            multiple_lines = translated_selectbox(
                "multiple_lines",
                [
                    ("Yes", "yes"),
                    ("No", "no")
                ]
            )

    st.divider()

    st.subheader(t["services"])

    if internet == "No":

        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

        st.info(t["no_internet"])

    else:

        service_options = [
            ("Yes", "yes"),
            ("No", "no")
        ]

        col1, col2, col3 = st.columns(3)

        with col1:

            online_security = translated_selectbox(
                "security", service_options
            )

            online_backup = translated_selectbox(
                "backup", service_options
            )

        with col2:

            device_protection = translated_selectbox(
                "protection", service_options
            )

            tech_support = translated_selectbox(
                "support", service_options
            )

        with col3:

            streaming_tv = translated_selectbox(
                "tv", service_options
            )

            streaming_movies = translated_selectbox(
                "movies", service_options
            )


# =========================================================
# 7. ACCOUNT INFORMATION
# =========================================================

st.header("💳 " + t["account"])

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        contract = translated_selectbox(
            "contract",
            [
                ("Month-to-month", "month_to_month"),
                ("One year", "one_year"),
                ("Two year", "two_year")
            ]
        )

    with col2:

        paperless = translated_selectbox(
            "paperless",
            [
                ("Yes", "yes"),
                ("No", "no")
            ]
        )

    with col3:

        payment = translated_selectbox(
            "payment",
            [
                ("Electronic check", "electronic_check"),
                ("Mailed check", "mailed_check"),
                ("Bank transfer (automatic)", "bank_transfer"),
                ("Credit card (automatic)", "credit_card")
            ]
        )

    monthly_charges = st.number_input(
        t["monthly"],
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )


# =========================================================
# 8. FEATURE ENGINEERING
# =========================================================

additional_services = sum([
    online_security == "Yes",
    online_backup == "Yes",
    device_protection == "Yes",
    tech_support == "Yes",
    streaming_tv == "Yes",
    streaming_movies == "Yes"
])

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.metric(
        t["additional"],
        additional_services
    )

with col2:

    st.metric(
        t["tenure"],
        f"{tenure} months"
    )


# =========================================================
# 9. PREDICTION
# =========================================================

st.divider()

if st.button(
    "🔍 " + t["predict"],
    type="primary",
    use_container_width=True
):

    customer = {

        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,

        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,

        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,

        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,

        "MonthlyCharges": monthly_charges,
        "NumAdditionalServices": additional_services
    }

    try:

        result = predict_churn(customer)

        probability = result["probability"]
        prediction = result["prediction"]

        risk_label, risk_status = get_risk_level(probability)

        st.divider()

        st.header("📈 " + t["results"])

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                t["probability"],
                f"{probability:.2%}"
            )

            st.progress(probability)

        with col2:

            st.metric(
                t["threshold"],
                f"{result['threshold']:.1%}"
            )

            if risk_status == "success":
                st.success(risk_label)

            elif risk_status == "warning":
                st.warning(risk_label)

            else:
                st.error(risk_label)

        st.divider()

        if prediction == "Yes":

            st.error(
                "⚠️ " + t["likely_churn"]
            )

        else:

            st.success(
                "✅ " + t["likely_stay"]
            )

        st.caption(t["risk_explanation"])

        st.info("ℹ️ " + t["disclaimer"])

    except Exception as error:

        st.error(f"Prediction error: {error}")


# =========================================================
# 10. FOOTER
# =========================================================

st.divider()

st.caption(
    "Telco Customer Churn Prediction | "
    "Machine Learning Portfolio Project | Merzouki Mehdi"
)
