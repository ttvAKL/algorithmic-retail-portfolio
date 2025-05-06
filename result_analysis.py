import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_performance(df_nav):
    """
    Given a DataFrame with columns ['date','nav'], compute:
      - cumulative return
      - annualized Sharpe
      - max drawdown
      - annualized volatility
    """
    nav = df_nav["nav"].values
    # Daily returns
    ret = nav[1:] / nav[:-1] - 1
    # Cumulative return
    cum_ret = nav[-1] / nav[0] - 1
    # Annualized volatility
    vol = np.std(ret) * np.sqrt(252)
    # Sharpe ratio (assume zero risk-free for simplicity or subtract RF)
    sharpe = np.mean(ret) / np.std(ret) * np.sqrt(252)
    # Max drawdown
    peak = np.maximum.accumulate(nav)
    drawdown = np.min((nav - peak) / peak)
    return cum_ret, sharpe, drawdown, vol

def main():
    # Load backtest results
    results = pd.read_parquet("backtest_results.parquet")
    
    # Prepare summary container
    rows = []
    for (model, capital), grp in results.groupby(["model","capital"]):
        df = grp.sort_values("date")
        cum, sr, dd, vol = compute_performance(df)
        rows.append({
            "model": model,
            "capital": capital,
            "cumulative_return": cum,
            "sharpe_ratio": sr,
            "max_drawdown": dd,
            "annual_vol": vol
        })
    summary = pd.DataFrame(rows)
    summary.to_csv("performance_summary.csv", index=False)
    print("Saved performance_summary.csv\n", summary)
    
    # Plot equity curves
    plt.figure(figsize=(10,6))
    for (model, capital), grp in results.groupby(["model","capital"]):
        label = f"{model}-{capital}"
        plt.plot(grp["date"], grp["nav"], label=label)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.title("Equity Curves by Model & Capital")
    plt.tight_layout()
    plt.savefig("equity_curves.png")
    print("Saved equity_curves.png")
    
    # Bar chart of Sharpe ratios
    plt.figure(figsize=(6,4))
    pivot = summary.pivot(index="model", columns="capital", values="sharpe_ratio")
    pivot.plot(kind="bar")
    plt.ylabel("Sharpe Ratio")
    plt.title("Sharpe Ratio by Model & Capital")
    plt.tight_layout()
    plt.savefig("sharpe_bars.png")
    print("Saved sharpe_bars.png")

if __name__ == "__main__":
    main()