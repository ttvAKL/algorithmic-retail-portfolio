import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import run_backtest
from models import ManualModel, HeuristicModel  # (DQN stubbed for now)

def run_sensitivity(panel_path, slippages, capitals, period_name, start_date, end_date):
    # Load panel and filter dates if needed
    panel = pd.read_parquet(panel_path)
    # Restrict to specified date range
    mask = (panel.index.get_level_values("date") >= pd.to_datetime(start_date)) & \
           (panel.index.get_level_values("date") <= pd.to_datetime(end_date))
    panel = panel[mask]
    
    if panel.empty:
        print(f"Warning: No data for period {period_name}. Skipping.")
        return pd.DataFrame(columns=[
            "period", "slippage_bps", "model", "capital",
            "cumulative_return", "sharpe_ratio", "max_drawdown", "annual_vol"
        ])
    
    dates = sorted(panel.index.get_level_values("date").unique())
    if not dates:
        print(f"Warning: No trading dates between {start_date} and {end_date} for period {period_name}. Skipping.")
        return pd.DataFrame(columns=[
            "period", "slippage_bps", "model", "capital",
            "cumulative_return", "sharpe_ratio", "max_drawdown", "annual_vol"
        ])
    
    tickers = panel.index.get_level_values("ticker").unique().tolist()
    models = {
        "manual": ManualModel(tickers),
        "heuristic": HeuristicModel(tickers)
    }
    
    records = []
    for slip in slippages:
        for name, model in models.items():
            for cap in capitals:
                print(f"Period={period_name}, Slippage={slip}, Model={name}, Capital={cap}")
                df = run_backtest(panel, model, cap, slippage=slip)
                # Compute performance metrics inline:
                nav = df["nav"].values
                ret = nav[1:] / nav[:-1] - 1
                cum = nav[-1] / nav[0] - 1
                sr  = np.mean(ret) / np.std(ret) * np.sqrt(252)
                dd  = np.min((nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav))
                vol = np.std(ret) * np.sqrt(252)
                
                records.append({
                    "period": period_name,
                    "slippage_bps": int(slip * 10000),
                    "model": name,
                    "capital": cap,
                    "cumulative_return": cum,
                    "sharpe_ratio": sr,
                    "max_drawdown": dd,
                    "annual_vol": vol
                })
    return pd.DataFrame(records)

def main():
    # Common panel_with_signals.parquet
    panel_file = "panel_with_signals.parquet"
    
    # Define sensitivity parameters
    slippages = [0.0002, 0.0005, 0.0010]     # 2, 5, 10 bps
    capitals  = [500, 2500, 5000]            # as before
    
    # 1) Baseline period: 2018–2023
    df1 = run_sensitivity(panel_file, slippages, capitals,
                          period_name="2018–2023",
                          start_date="2018-01-01",
                          end_date="2023-12-31")
    # 2) Pre‑COVID: 2015–2019
    df2 = run_sensitivity(panel_file, slippages, capitals,
                          period_name="2015–2019",
                          start_date="2015-01-01",
                          end_date="2019-12-31")
    
    # Combine and save
    all_df = pd.concat([df1, df2], ignore_index=True)
    all_df.to_csv("sensitivity_summary.csv", index=False)
    print("Saved sensitivity_summary.csv")
    
    # Plot Sharpe vs. Slippage for baseline period
    plt.figure(figsize=(6,4))
    base = all_df[all_df["period"]=="2018–2023"]
    for model in base["model"].unique():
        sub = base[base["model"]==model]
        plt.plot(sub["slippage_bps"], sub["sharpe_ratio"], marker='o', label=model)
    plt.xlabel("Slippage (bps)")
    plt.ylabel("Sharpe Ratio")
    plt.title("Sharpe vs. Slippage (2018–2023)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sharpe_vs_slippage.png")
    print("Saved sharpe_vs_slippage.png")

if __name__ == "__main__":
    main()