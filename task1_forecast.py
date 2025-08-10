# notebooks/task1_forecast.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional for interactive plotting; script will continue if not installed.
try:
    import plotly.express as px
    from prophet.plot import plot_plotly
    plotly_available = True
except Exception:
    plotly_available = False

# Prophet import with helpful error message if not available
try:
    from prophet import Prophet
except Exception:
    try:
        # older package name
        from fbprophet import Prophet  # type: ignore
    except Exception as e:
        print("ERROR: Prophet is not installed or failed to import.")
        print("Install prophet in your venv (example): pip install prophet")
        raise e

# metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

def find_date_and_sales_cols(df: pd.DataFrame):
    cols = [c.strip() for c in df.columns]
    # find date-like column
    date_candidates = [c for c in cols if "date" in c.lower() or "day" in c.lower() or "time" in c.lower()]
    sales_candidates = [c for c in cols if "sale" in c.lower() or "revenue" in c.lower() or "amount" in c.lower() or c.lower() == "y"]
    if not date_candidates:
        # fallback: show columns to user
        raise ValueError(f"Could not detect a date column automatically. Columns found: {cols}")
    if not sales_candidates:
        raise ValueError(f"Could not detect a sales column automatically. Columns found: {cols}")
    return date_candidates[0], sales_candidates[0]

def try_parse_dates(series: pd.Series):
    # First attempt default parsing
    parsed = pd.to_datetime(series, errors='coerce')
    # If a lot are NaT (more than 5%), try dayfirst parsing (e.g., DD/MM/YYYY)
    nat_frac = parsed.isna().mean()
    if nat_frac > 0.05:
        parsed2 = pd.to_datetime(series, errors='coerce', dayfirst=True)
        nat_frac2 = parsed2.isna().mean()
        if nat_frac2 < nat_frac:
            return parsed2
    return parsed

def main():
    # path to CSV located next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "sales.csv")  # keep sales.csv in notebooks/
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Put your CSV named 'sales.csv' here.")

    # read CSV
    df = pd.read_csv(data_path)
    if df.shape[1] < 2:
        raise ValueError("CSV appears to have < 2 columns. Expected at least date and sales columns.")

    # clean header strings
    df.columns = df.columns.str.strip()

    # detect date and sales columns
    try:
        date_col, sales_col = find_date_and_sales_cols(df)
    except ValueError as e:
        print(e)
        print("Columns found:", list(df.columns))
        sys.exit(1)

    # rename to ds, y for Prophet later
    df = df.rename(columns={date_col: "ds", sales_col: "y"})

    # parse types
    df['ds'] = try_parse_dates(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # drop invalid rows
    before = len(df)
    df = df.dropna(subset=['ds', 'y']).copy()
    after = len(df)
    if after < before:
        print(f"Dropped {before-after} rows with invalid dates or sales values.")

    if df.empty:
        raise ValueError("No valid rows remain after parsing 'ds' and 'y' — check your CSV.")

    # sort
    df = df.sort_values('ds').reset_index(drop=True)

    # aggregate duplicate dates (sum sales)
    if df['ds'].duplicated().any():
        print("Found duplicate dates — aggregating by summing sales per day.")
        df = df.groupby('ds', as_index=False)['y'].sum()

    # ensure ds is datetime index and fill missing dates
    df = df.set_index('ds').sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index is not a DatetimeIndex after conversion — aborting.")

    # resample to daily frequency (fills missing days), using forward fill for any short gaps
    df = df.asfreq('D')  # introduces NaNs where data missing
    # fill small missing values by forward fill then backward fill for leading NaNs
    df['y'] = df['y'].ffill().bfill()

    # reset index for Prophet
    df = df.reset_index()

    print(f"Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    print(f"Total rows after cleaning: {len(df)}")

    # quick static plot
    os.makedirs(os.path.join(script_dir, "..", "assets"), exist_ok=True)
    assets_dir = os.path.join(script_dir, "..", "assets")
    plt.figure(figsize=(10, 4))
    plt.plot(df['ds'], df['y'])
    plt.title("Sales over time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "plot_timeseries.png"), dpi=150)
    plt.close()

    # choose holdout: 90 days or 20% of data if shorter
    if len(df) <= 100:
        holdout_days = max(7, int(len(df) * 0.2))
    else:
        holdout_days = 90
    holdout_days = int(holdout_days)
    print(f"Using holdout_days = {holdout_days}")

    if len(df) <= holdout_days + 10:
        raise ValueError("Not enough data for the chosen holdout. Add more rows or reduce holdout.")

    train = df.iloc[:-holdout_days].copy()
    test = df.iloc[-holdout_days:].copy()

    # train Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    # optional: add holidays if you want and it's appropriate for your country
    # m.add_country_holidays(country_name='US')  # comment out if not needed
    m.fit(train.rename(columns={'ds': 'ds', 'y': 'y'}))

    # forecast
    future = m.make_future_dataframe(periods=holdout_days, freq='D')
    forecast = m.predict(future)

    # save outputs
    outputs_dir = os.path.join(script_dir, "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(os.path.join(outputs_dir, "forecast.csv"), index=False)

    # evaluation: join forecast and test on ds
    eval_df = pd.merge(test[['ds', 'y']], forecast[['ds', 'yhat']], on='ds', how='left').dropna()
    if eval_df.empty:
        print("Warning: no overlap between forecast and test dates for evaluation.")
        mae = float('nan')
        rmse = float('nan')
    else:
        mae = mean_absolute_error(eval_df['y'], eval_df['yhat'])
        rmse = np.sqrt(mean_squared_error(eval_df['y'], eval_df['yhat']))

    print(f"Holdout MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # save static plots (matplotlib)
    fig1 = m.plot(forecast)
    fig1.savefig(os.path.join(assets_dir, "forecast_plot.png"), dpi=150)
    plt.close(fig1)

    fig2 = m.plot_components(forecast)
    fig2.savefig(os.path.join(assets_dir, "components.png"), dpi=150)
    plt.close(fig2)

    # optional interactive plotly output
    if plotly_available:
        try:
            fig3 = plot_plotly(m, forecast)
            interactive_path = os.path.join(assets_dir, "forecast_interactive.html")
            fig3.write_html(interactive_path)
            print(f"Interactive Plotly chart saved to {interactive_path}")
        except Exception as e:
            print("Plotly available but failed to produce interactive plot:", e)

    print("✅ Forecast complete.")
    print(f"Files written to: {outputs_dir} and {assets_dir}")

if __name__ == "__main__":
    main()
# This script is designed to be run in a Jupyter notebook or as a standalone script.
# It processes a CSV file named 'sales.csv' located in the same directory as this script