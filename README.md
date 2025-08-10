# FUTURE_ML_01
# AI-Powered Sales Forecasting & Customer Churn Prediction Dashboard

## Overview

This is a Streamlit web app that allows users to:

- **Forecast future sales trends** using historical sales data with Prophet (time series forecasting).
- **Predict customer churn** using machine learning (XGBoost classifier) on customer data.

The app provides interactive visualizations and model performance insights to help businesses make data-driven decisions.

## Features

- Upload sales data CSV (columns: `Date`, `Sales`) to generate sales forecasts with confidence intervals.
- Upload customer churn dataset (with `Churn` target column) to train and test a churn prediction model.
- Visualize forecast trends and churn feature importance.
- Easy-to-use tabs for switching between forecasting and churn prediction.
- Built with Python, Streamlit, Prophet, XGBoost, and other popular data science libraries.

Demo vedio : https://www.loom.com/share/53bed10a678340049ab920da5a6b11fd?sid=7efc5d37-5337-4dc1-8e7b-3ad4c8822bd5

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   Install required packages:

   
pip install streamlit prophet xgboost scikit-learn pandas matplotlib




Run the app:
python -m streamlit run notebooks\streamlit_app\sales_forecasting_suite_all.py



Usage
Open the URL shown in your terminal (usually http://localhost:8501).

Use the Sales Forecasting tab to upload sales data CSV and choose forecast horizon.

Use the Churn Prediction tab to upload customer data CSV and view churn prediction results.

Explore the visual outputs and model insights.

Dataset Format
Sales Data CSV
Date	Sales
2023-01-01	1234.56
2023-01-02	1300.45
...	...

Date should be in YYYY-MM-DD format.

Sales is numeric daily/weekly/monthly sales.

Churn Data CSV
CustomerID	Feature1	Feature2	...	Churn
1001	...	...	...	0 or 1

Must contain a Churn column with binary values (0 = retained, 1 = churned).

Future Improvements
Add a customer support chatbot for real-time query handling.

Integrate more advanced forecasting models.

Deploy on cloud platforms for broader access.

License
This project is open source and free to use.

Created by S Rayeesa Tabusum

yaml
Copy
Edit

---

### How to add this README to your GitHub repo:

1. Save above content as `README.md` in your project folder.

2. In your terminal inside the project folder:

```bash
git init               # if you haven't initialized git repo yet
git add README.md
git commit -m "Add README with project overview"
git remote add origin https://github.com/your-username/your-repo-name.git
git push -u origin main

