import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def main():
    path = "master_panel.parquet"
    if not os.path.exists(path):
        print(f"ERROR: Cannot find {path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load the panel
    panel = pd.read_parquet(path)
    print("Loaded master_panel.parquet with shape:", panel.shape)

    # 2. Coverage check
    counts = panel["close"].groupby(level="ticker").count()
    print("\nData points per ticker:")
    print(counts)

    # 3. Missingness check
    missing = panel["close"].isna().groupby(level="ticker").sum()
    print("\nMissing entries per ticker (should be 0):")
    print(missing[missing > 0] if missing.any() else "None")

    # 4. Returns distribution
    returns = panel["return"].dropna()
    print("\nReturn distribution summary:")
    print(returns.describe())

    # 5. Spot-check an equity curve and save to PNG
    ticker = "SPY"
    try:
        spy = panel.xs(ticker).reset_index()
    except KeyError:
        print(f"\nERROR: Ticker {ticker} not found in panel.", file=sys.stderr)
        sys.exit(1)

    plt.figure(figsize=(10, 4))
    plt.plot(spy["date"], spy["close"], label=ticker)
    plt.title(f"{ticker} Adjusted Close (2018–2023)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig("spy_curve.png")
    print("\nSaved equity-curve plot to spy_curve.png")

if __name__ == "__main__":
    main()