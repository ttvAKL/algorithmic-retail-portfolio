

import numpy as np
import pandas as pd

class ManualModel:
    """
    Manual model: Equal-weight portfolio across all assets.
    """
    def __init__(self, tickers):
        self.tickers = tickers
    
    def get_weights(self, date, signals):
        """
        Returns a pandas Series of equal weights for each ticker.
        `signals` is a DataFrame slice for the given date with tickers as index.
        """
        n = len(self.tickers)
        weights = np.repeat(1.0 / n, n)
        return pd.Series(weights, index=self.tickers)


class HeuristicModel:
    """
    Heuristic momentum model: 80% SPY ETF, 20% equally among top 5 momentum stocks.
    """
    def __init__(self, tickers, etf="SPY", momentum_col="momentum_60"):
        self.tickers = tickers
        self.equity_tickers = [t for t in tickers if t != etf]
        self.etf = etf
        self.momentum_col = momentum_col
    
    def get_weights(self, date, signals):
        """
        `signals` is a DataFrame slice for the given date with tickers as index.
        Selects top 5 momentum stocks and allocates 20% equally among them; 
        the remaining 80% goes to the ETF.
        """
        # Ensure ETF is present
        if self.etf not in signals.index:
            raise KeyError(f"ETF {self.etf} not in signals index")

        # Rank equities by momentum
        eq_signals = signals.loc[self.equity_tickers]
        top5 = eq_signals[self.momentum_col].nlargest(5).index.tolist()
        
        weights = pd.Series(0.0, index=self.tickers)
        weights[self.etf] = 0.80
        equal_weight = 0.20 / len(top5)
        for ticker in top5:
            weights[ticker] = equal_weight
        
        return weights


class DQNModel:
    """
    AI-Assisted model stub: uses a trained DQN agent for allocation decisions.
    """
    def __init__(self, agent, tickers):
        """
        agent: an object with a .predict(state) method returning allocation weights array.
        tickers: list of tickers in the same order the agent expects.
        """
        self.agent = agent
        self.tickers = tickers
    
    def get_weights(self, date, signals):
        """
        signals: DataFrame slice for the given date with tickers as index and
                 input features as columns, ordered according to self.tickers.
        Returns a pandas Series of weights as predicted by the agent.
        """
        # Extract state as numpy array in ticker order
        state = signals.loc[self.tickers].values
        # Predict weights via agent
        weights = self.agent.predict(state)  # assume shape=(n_tickers,)
        return pd.Series(weights, index=self.tickers)