import pandas as pd

def compute_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trading signals for each ticker in the multi-index panel.
    Adds:
      - momentum_60: trailing 60-day return
      - vol_30: trailing 30-day volatility of daily returns
    """
    # Compute trailing 60-day momentum
    panel["momentum_60"] = (
        panel.groupby(level="ticker")["close"]
             .pct_change(60)
    )
    
    # Compute trailing 30-day volatility
    panel["vol_30"] = (
        panel.groupby(level="ticker")["return"]
             .rolling(window=30)
             .std()
             .reset_index(level=0, drop=True)
    )
    
    return panel

if __name__ == "__main__":
    # Example usage: load, compute, and save
    panel = pd.read_parquet("master_panel.parquet")
    panel_with_signals = compute_signals(panel)
    panel_with_signals.to_parquet("panel_with_signals.parquet")
    print("Signals computed and saved to panel_with_signals.parquet")