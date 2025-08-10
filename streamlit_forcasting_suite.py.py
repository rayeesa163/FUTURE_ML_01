import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For Forecasting
from prophet import Prophet

# For Churn Prediction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# For Chatbot (OpenAI GPT)
import openai

# --- Set your OpenAI API Key here for Chatbot ---
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# -----------------------------------------
# --- Task 1: Sales Forecasting Dashboard ---
# -----------------------------------------
def sales_forecasting_app():
    st.title("Sales Forecasting Dashboard")
    st.write("""
        Upload your retail sales data CSV file with columns 'Date' and 'Sales' to forecast future sales.
    """)

    uploaded_file = st.file_uploader("Upload CSV file for sales data", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Raw Data Preview:")
        st.dataframe(df.head())

        if 'Date' not in df.columns or 'Sales' not in df.columns:
            st.error("CSV must contain 'Date' and 'Sales' columns.")
            return

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df_prophet = df.rename(columns={'Date': 'ds', 'Sales': 'y'})

        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(df_prophet)

        period = st.slider("Select forecast horizon (days):", min_value=7, max_value=365, value=30)

        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)

        st.write("## Forecast Plot")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.write("## Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)


# -----------------------------------------
# --- Task 2: Customer Churn Prediction ---
# -----------------------------------------
def churn_prediction_app():
    st.title("Customer Churn Prediction")

    st.write("""
    Upload your customer churn dataset CSV with a 'Churn' column (1 for churned, 0 for retained).
    """)

    uploaded_file = st.file_uploader("Upload CSV file for churn data", type=["csv"], key="churn")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Raw Data Preview:")
        st.dataframe(df.head())

        if 'Churn' not in df.columns:
            st.error("Dataset must contain a 'Churn' column.")
            return

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        df.fillna(method='ffill', inplace=True)

        X = df.drop(['Churn'], axis=1)
        y = df['Churn']

        test_size = st.slider("Test Set Size (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write(f"### ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
        ax.plot([0,1], [0,1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)

        importances = model.feature_importances_
        features = X.columns
        fig2, ax2 = plt.subplots(figsize=(8,6))
        ax2.barh(features, importances)
        ax2.set_xlabel("Feature Importance")
        ax2.set_title("Feature Importance in XGBoost")
        st.pyplot(fig2)

        st.write("### Churn Probabilities for Test Samples")
        prob_df = X_test.copy()
        prob_df['Churn_Probability'] = y_pred_proba
        st.dataframe(prob_df.head(10))


# -----------------------------------------
# --- Task 3: Customer Support Chatbot ---
# -----------------------------------------
def chatbot_app():
    st.title("Customer Support Chatbot (GPT-4o-mini)")

    if OPENAI_API_KEY == "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" or not OPENAI_API_KEY:
        st.warning("Please set your OpenAI API key in the code to use the chatbot.")
        return

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful customer support assistant."}
        ]

    user_input = st.text_input("Enter your message:")

    if st.button("Send") or user_input:
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.spinner("Waiting for response..."):
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.chat_history,
                    max_tokens=150,
                    temperature=0.7,
                )
                bot_response = response.choices[0].message['content']
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    st.markdown(f"**You:** {chat['content']}")
                elif chat["role"] == "assistant":
                    st.markdown(f"**Bot:** {chat['content']}")
        else:
            st.warning("Please enter a message.")


# ----------------------
# Main app navigation
# ----------------------
def main():
    st.sidebar.title("AI-Powered Suite")
    app_mode = st.sidebar.selectbox(
        "Choose a task",
        ["Sales Forecasting Dashboard", "Customer Churn Prediction", "Customer Support Chatbot"]
    )

    if app_mode == "Sales Forecasting Dashboard":
        sales_forecasting_app()
    elif app_mode == "Customer Churn Prediction":
        churn_prediction_app()
    elif app_mode == "Customer Support Chatbot":
        chatbot_app()

if __name__ == "__main__":
    main()
