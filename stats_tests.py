import pandas as pd
import numpy as np
from scipy import stats

def paired_tests(results, model_a, model_b, capital):
    # Filter two models at a single capital level
    df_a = results[(results.model==model_a)&(results.capital==capital)].sort_values("date")
    df_b = results[(results.model==model_b)&(results.capital==capital)].sort_values("date")
    # Compute daily returns
    ret_a = df_a.nav.pct_change().dropna()
    ret_b = df_b.nav.pct_change().dropna()
    # Align series
    common_idx = ret_a.index.intersection(ret_b.index)
    ret_a = ret_a.loc[common_idx]
    ret_b = ret_b.loc[common_idx]
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(ret_a, ret_b)
    # Wilcoxon
    w_stat, p_wil = stats.wilcoxon(ret_a, ret_b)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "capital": capital,
        "t_stat": t_stat,
        "p_val": p_val,
        "wilcoxon_stat": w_stat,
        "p_wilcoxon": p_wil
    }

def bootstrap_cum_diff(results, model_a, model_b, capital, n_boot=10000):
    df_a = results[(results.model==model_a)&(results.capital==capital)].sort_values("date")
    df_b = results[(results.model==model_b)&(results.capital==capital)].sort_values("date")
    nav_a = df_a.nav.values
    nav_b = df_b.nav.values
    cum_diff = []
    for _ in range(n_boot):
        idx = np.random.choice(len(nav_a)-1, len(nav_a)-1, replace=True) + 1
        cum_a = nav_a[idx].prod() / nav_a[0] - 1
        cum_b = nav_b[idx].prod() / nav_b[0] - 1
        cum_diff.append(cum_a - cum_b)
    lower, upper = np.percentile(cum_diff, [2.5, 97.5])
    return lower, upper

def main():
    results = pd.read_parquet("backtest_results.parquet")
    capital = 500  # test one level—others are identical in % terms

    tests = []
    tests.append(paired_tests(results, "heuristic", "manual", capital))
    # If AI model is later included:
    # tests.append(paired_tests(results, "dqn", "manual", capital))
    # tests.append(paired_tests(results, "dqn", "heuristic", capital))

    test_df = pd.DataFrame(tests)
    test_df.to_csv("statistical_tests.csv", index=False)
    print("Saved statistical_tests.csv\n", test_df)

    # Bootstrap cumulative diff
    lb, ub = bootstrap_cum_diff(results, "heuristic", "manual", capital)
    print(f"95% CI for cum_return_heuristic – cum_return_manual: [{lb:.3f}, {ub:.3f}]")

if __name__ == "__main__":
    main()