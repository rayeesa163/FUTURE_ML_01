"""
Task 2 — Prophet Model Comparison: Default vs Tuned
Outputs: forecasts, metrics, and comparison plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- Helper Functions ----------
def find_date_and_sales_cols(df):
    cols = [c.strip() for c in df.columns]
    date_candidates = [c for c in cols if "date" in c.lower() or "day" in c.lower() or "time" in c.lower()]
    sales_candidates = [c for c in cols if "sale" in c.lower() or "revenue" in c.lower() or "amount" in c.lower() or c.lower() == "y"]
    if not date_candidates or not sales_candidates:
        raise ValueError(f"Could not auto-detect date/sales columns. Columns: {cols}")
    return date_candidates[0], sales_candidates[0]

def try_parse_dates(series):
    parsed = pd.to_datetime(series, errors='coerce')
    if parsed.isna().mean() > 0.05:
        parsed2 = pd.to_datetime(series, errors='coerce', dayfirst=True)
        if parsed2.isna().mean() < parsed.isna().mean():
            return parsed2
    return parsed

def load_and_prepare(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    date_col, sales_col = find_date_and_sales_cols(df)
    df = df.rename(columns={date_col: "ds", sales_col: "y"})
    df['ds'] = try_parse_dates(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds','y']).copy()
    df = df.sort_values('ds').reset_index(drop=True)
    if df['ds'].duplicated().any():
        df = df.groupby('ds', as_index=False)['y'].sum()
    df = df.set_index('ds').asfreq('D')
    df['y'] = df['y'].ffill().bfill()
    df = df.reset_index()
    return df

def train_prophet(train_df, holdout_days, tuned=False):
    if tuned:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.5,  # more flexible trend changes
            seasonality_mode='multiplicative'
        )
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    else:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=holdout_days, freq='D')
    forecast = m.predict(future)
    return m, forecast

def evaluate(test_df, forecast_df):
    merged = pd.merge(test_df[['ds','y']], forecast_df[['ds','yhat']], on='ds', how='left').dropna()
    if merged.empty:
        return np.nan, np.nan
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    return mae, rmse

def save_plot_compare(df, fc_default, fc_tuned, assets_dir):
    plt.figure(figsize=(12,5))
    plt.plot(df['ds'], df['y'], label='Historical', color='black', linewidth=1)
    plt.plot(fc_default['ds'], fc_default['yhat'], label='Prophet Default', alpha=0.8)
    plt.plot(fc_tuned['ds'], fc_tuned['yhat'], label='Prophet Tuned', alpha=0.8)
    plt.legend()
    plt.xlabel('Date'); plt.ylabel('Sales'); plt.title('Prophet Model Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'prophet_comparison.png'), dpi=150)
    plt.close()

# ---------- Main ----------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "sales.csv")  # same as Task 1

    df = load_and_prepare(data_path)

    outputs_dir = os.path.join(script_dir, "..", "outputs")
    assets_dir = os.path.join(script_dir, "..", "assets")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # Holdout period
    holdout_days = 90 if len(df) > 100 else max(7, int(len(df) * 0.2))
    train = df.iloc[:-holdout_days].copy()
    test = df.iloc[-holdout_days:].copy()

    print(f"Training on {len(train)} rows, evaluating on {len(test)} rows (holdout_days = {holdout_days})")

    # Prophet Default
    m_default, fc_default = train_prophet(train[['ds','y']], holdout_days, tuned=False)
    fc_default = fc_default[['ds','yhat','yhat_lower','yhat_upper']]
    fc_default.to_csv(os.path.join(outputs_dir, "prophet_default_forecast.csv"), index=False)

    # Prophet Tuned
    m_tuned, fc_tuned = train_prophet(train[['ds','y']], holdout_days, tuned=True)
    fc_tuned = fc_tuned[['ds','yhat','yhat_lower','yhat_upper']]
    fc_tuned.to_csv(os.path.join(outputs_dir, "prophet_tuned_forecast.csv"), index=False)

    # Evaluate
    default_mae, default_rmse = evaluate(test, fc_default)
    tuned_mae, tuned_rmse = evaluate(test, fc_tuned)

    print(f"Prophet Default MAE: {default_mae:.4f}, RMSE: {default_rmse:.4f}")
    print(f"Prophet Tuned   MAE: {tuned_mae:.4f}, RMSE: {tuned_rmse:.4f}")

    # Save metrics
    metrics = pd.DataFrame({
        'model': ['prophet_default', 'prophet_tuned'],
        'mae': [default_mae, tuned_mae],
        'rmse': [default_rmse, tuned_rmse]
    })
    metrics.to_csv(os.path.join(outputs_dir, "prophet_comparison_metrics.csv"), index=False)

    # Plot comparison
    save_plot_compare(df, fc_default, fc_tuned, assets_dir)
    print("Saved plot to assets/prophet_comparison.png")
    print("Outputs written to:", outputs_dir)
    print("Assets written to:", assets_dir)

if __name__ == "__main__":
    main()
