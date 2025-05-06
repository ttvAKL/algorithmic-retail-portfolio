from os import getenv
from dotenv import load_dotenv
from polygon import RESTClient
import pandas as pd
from datetime import date

# 1. Load API key
load_dotenv()
API_KEY = getenv("POLYGON_API_KEY")

# 2. Define your asset universe
EQUITY_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "FB",       # Technology
    "JNJ", "PFE", "MRK", "ABBV", "TMO",         # Healthcare
    "JPM", "BAC", "C", "WFC", "GS",             # Financials
    "TSLA", "HD", "NKE", "MCD", "SBUX"          # Consumer Discretionary
]
ETF_TICKERS = ["SPY", "QQQ", "IWM"]

ALL_TICKERS = EQUITY_TICKERS + ETF_TICKERS

def fetch_daily_bars(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Fetches daily split- and dividend-adjusted bars for a given ticker.
    """
    client = RESTClient(API_KEY)
    bars = client.get_aggs(
        ticker,
        1,
        "day",
        from_=start.isoformat(),
        to=end.isoformat(),
        adjusted=True,
        sort="asc",
        limit=50000
    )
    df = pd.DataFrame([{
        "date":   pd.to_datetime(r.timestamp, unit="ms").date(),
        "open":   r.open,
        "high":   r.high,
        "low":    r.low,
        "close":  r.close,
        "volume": r.volume
    } for r in bars])
    df.set_index("date", inplace=True)
    return df

def build_master_panel(data_dict: dict) -> pd.DataFrame:
    """
    Combines individual ticker DataFrames into a multi-index panel and applies cleaning.
    """
    # Determine full date range index
    all_dates = pd.date_range(
        start=min(df.index.min() for df in data_dict.values()),
        end=max(df.index.max() for df in data_dict.values()),
        freq="D"
    )
    
    # Prepare list to collect cleaned DataFrames
    cleaned_frames = []
    for ticker, df in data_dict.items():
        # Reindex to full date range and forward-fill
        df_reindexed = df.reindex(all_dates).ffill()
        df_reindexed["ticker"] = ticker
        cleaned_frames.append(df_reindexed)
    
    # Concatenate and set multi-index
    panel = pd.concat(cleaned_frames)
    panel.set_index("ticker", append=True, inplace=True)
    panel = panel.swaplevel(0, 1)
    panel.index.set_names(["ticker", "date"], inplace=True)
    
    # Compute returns and clip extremes
    panel["return"] = panel.groupby(level="ticker")["close"].pct_change().clip(-0.20, 0.20)
    
    return panel

if __name__ == "__main__":
    # Define sample period
    start_date = date(2018, 1, 1)
    end_date   = date(2023, 12, 31)

    # Fetch data for all tickers
    all_data = {}
    for ticker in ALL_TICKERS:
        print(f"Fetching data for {ticker}...")
        all_data[ticker] = fetch_daily_bars(ticker, start_date, end_date)

    # Build and save the master panel
    master_panel = build_master_panel(all_data)
    master_panel.to_parquet("master_panel.parquet")
    print("Data ingestion complete. Master panel saved to 'master_panel.parquet'.")