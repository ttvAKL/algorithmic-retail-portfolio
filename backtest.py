import pandas as pd
import numpy as np
from models import ManualModel, HeuristicModel, DQNModel
# Placeholder import for your trained agent:
# from agent import load_trained_agent

def run_backtest(panel, model, capital, slippage):
    """
    Run a backtest for one model and capital base.
    Returns a DataFrame with columns: date, nav, turnover.
    """
    tickers = panel.index.get_level_values('ticker').unique().tolist()
    dates = sorted(panel.index.get_level_values('date').unique())
    
    nav = [capital]
    prev_weights = np.zeros(len(tickers))
    turnovers = [0.0]
    
    for date in dates:
        day_df = panel.xs(date, level='date')
        # Extract signals or price data slice
        signals = day_df[['momentum_60','vol_30']]
        returns = day_df['return'].fillna(0.0).values
        
        # Get allocation weights
        weights = model.get_weights(date, signals)
        weights_arr = weights.values
        
        # Calculate gross and net returns
        gross_return = np.dot(weights_arr, returns)
        turnover = np.sum(np.abs(weights_arr - prev_weights))
        cost = turnover * slippage
        net_return = gross_return - cost
        
        nav.append(nav[-1] * (1 + net_return))
        turnovers.append(turnover)
        prev_weights = weights_arr
    
    # Align dates and nav series
    results = pd.DataFrame({
        'date': [dates[0]] + dates,
        'nav': nav,
        'turnover': turnovers
    })
    return results

def main():
    # Load data with signals
    panel = pd.read_parquet("panel_with_signals.parquet")
    
    # Define models
    tickers = panel.index.get_level_values('ticker').unique().tolist()
    manual = ManualModel(tickers)
    heuristic = HeuristicModel(tickers)
    # Replace with your actual agent loading logic
    # agent = load_trained_agent("path_to_saved_model")
    # dqn = DQNModel(agent, tickers)
    dqn = None  # Stub until agent is available
    
    models = {
        'manual': manual,
        'heuristic': heuristic,
        # 'dqn': dqn
    }
    
    slippage = 0.0005  # 5 bps
    capital_levels = [500, 2500, 5000]
    
    # Collect results
    all_results = []
    for name, model in models.items():
        for capital in capital_levels:
            print(f"Running backtest: {name}, capital={capital}")
            df = run_backtest(panel, model, capital, slippage)
            df['model'] = name
            df['capital'] = capital
            all_results.append(df)
    
    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_parquet("backtest_results.parquet")
    print("Backtest complete. Results saved to backtest_results.parquet")

if __name__ == "__main__":
    main()
