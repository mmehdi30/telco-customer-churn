## Live Demo

🚀 [Try the Telco Customer Churn Prediction App](https://telco-customer-churn-mmehdi2.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-customer-churn-mmehdi2.streamlit.app/)

# Telco Customer Churn Prediction

An end-to-end Machine Learning project that predicts customer churn using the IBM Telco Customer Churn dataset. The project combines exploratory data analysis, feature engineering, model evaluation, and an interactive Streamlit application.

## Project Overview

Customer churn is a major challenge for telecommunications companies. Identifying customers who are likely to leave can help businesses develop targeted retention strategies.

This project builds a classification model to estimate the probability that a customer will discontinue their service.

**Main objectives:**
- Explore customer behavior and identify patterns associated with churn.
- Perform data preprocessing and feature engineering.
- Train and compare machine learning models.
- Evaluate model performance using appropriate classification metrics.
- Deploy an interactive application for customer churn prediction.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

- **Number of records:** 7,043 customers.
- **Target variable:** `Churn` (Yes / No).
- **Features:** Customer demographics, subscribed services, contract information, and billing details.

The original dataset contains 26.54% churned customers and 73.46% retained customers.

## Project Structure

```text
telco-customer-churn/
│
├── app.py
├── predict.py
├── translations.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── telco_churn_model.pkl
│   ├── model_features.json
│   └── model_config.json
│
└── notebooks/
    └── telco_churn_analysis.ipynb
```

## Methodology

### 1. Exploratory Data Analysis

The notebook investigates:
- Churn distribution.
- Customer tenure and churn behavior.
- Contract types and payment methods.
- Internet services and additional subscriptions.
- Relationships between numerical and categorical features.

Some important observations:
- Month-to-month contracts are associated with substantially higher churn.
- Customers with shorter tenure are more likely to churn.
- Electronic check users exhibit higher churn rates.
- Online Security and Tech Support subscriptions are associated with lower churn rates.

These are descriptive associations, not causal conclusions.

### 2. Data Preprocessing and Feature Engineering

The preprocessing workflow includes:
- Handling missing and inconsistent values.
- Converting numerical columns to appropriate data types.
- Separating numerical and categorical features.
- Scaling numerical variables using `StandardScaler`.
- Encoding categorical variables using `OneHotEncoder`.
- Creating additional customer service features.
- Removing redundant features following feature selection experiments.

The final model uses 19 selected input features.

### 3. Model Training and Selection

Two classification approaches were compared:

- Logistic Regression
- Random Forest

The final model is Logistic Regression.

It was selected because it achieved performance comparable to the tuned Random Forest while offering a simpler and more interpretable model.

### 4. Model Evaluation

The final model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| ROC-AUC | 83.90% |
| Accuracy | 75.94% |
| Precision (Churn = Yes) | 53.51% |
| Recall (Churn = Yes) | 71.39% |
| F1-score (Churn = Yes) | 61.17% |

A classification threshold of **0.325** was selected to improve churn detection, prioritizing recall over precision.

The test set was also consulted during earlier model exploration and feature-importance analysis, so the final test results should be interpreted with this limitation in mind.

## Interactive Application

The project includes a Streamlit application that allows users to:

- Enter customer information.
- Load example customer profiles.
- Estimate churn probability.
- View the predicted churn category.
- Explore descriptive churn indicators.
- Switch between English and French.

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/mmehdi30/telco-customer-churn.git
cd telco-customer-churn
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub

## Limitations

- Predictions are estimates based on historical data, not guarantees of future behavior.
- Model performance depends on the quality and representativeness of the training dataset.
- The classification threshold reflects a particular trade-off between precision and recall.
- The model has not yet been validated on a separate, newly collected customer population.

## Future Improvements

- Add SHAP for individual prediction explanations.
- Develop a REST API using FastAPI.
- Containerize the application using Docker.
- Introduce experiment tracking with MLflow.
- Implement automated testing and model monitoring.

## Author

**Mehdi Merzouki**

Machine Learning and Data Science

[GitHub Repository](https://github.com/mmehdi30/telco-customer-churn)
