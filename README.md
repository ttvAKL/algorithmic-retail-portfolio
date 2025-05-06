

# Algorithmic Portfolio Strategies for Low-Capital Retail Investors

A reproducible codebase and analysis accompanying the paper **“Algorithmic Portfolio Strategies for Low‑Capital Retail Investors”** by Austin Lee. This repository contains scripts for data ingestion, feature engineering, model implementation, backtesting, results analysis, sensitivity checks, and statistical tests, as well as the LaTeX source of the manuscript.

## Repository Structure

```
.
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── Dockerfile                   # Container specification for reproducibility
├── requirements.txt             # Python dependencies (optional)
├── data_ingestion.py            # Fetch and clean raw price data via Polygon.io
├── features.py                  # Compute momentum & volatility signals
├── models.py                    # Strategy definitions: Manual, Heuristic, DQN stub
├── backtest.py                  # Run backtests and export NAV & turnover series
├── results_analysis.py          # Summarize performance metrics & generate plots
├── sensitivity_analysis.py      # Robustness checks: slippage & historical periods
├── stats_tests.py               # Bootstrap confidence intervals on return differences
├── paper.tex                    # LaTeX manuscript source
└── sample_data/                 # (Optional) Small sample dataset for quick tests
```

**Note:** Large data files (`*.parquet`, `*.csv`, `*.png`) are excluded via `.gitignore`. Use provided scripts to fetch and generate all data.

## Quickstart

### 1. Clone the Repository
```bash
git clone git@github.com:<your-username>/algorithmic-retail-portfolio.git
cd algorithmic-retail-portfolio
```

### 2. Set Up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
If a `requirements.txt` is provided:
```bash
pip install -r requirements.txt
```
Otherwise:
```bash
pip install pandas numpy matplotlib polygon-api-client python-dotenv tensorflow scipy
```

### 4. Configure API Key
Register at [Polygon.io](https://polygon.io/) to get an API key. Create a `.env` file in the project root:
```bash
echo "POLYGON_API_KEY=your_api_key_here" > .env
```

## Usage

### Data Ingestion
Fetch and clean six years of daily adjusted price data:
```bash
python data_ingestion.py
```
Generates `master_panel.parquet`.

### Feature Engineering
Compute trading signals (momentum, volatility):
```bash
python features.py
```
Generates `panel_with_signals.parquet`.

### Backtesting
Run strategy simulations (Manual & Heuristic):
```bash
python backtest.py
```
Produces `backtest_results.parquet`.

### Results Analysis
Generate performance tables and plots:
```bash
python results_analysis.py
```
Outputs:
- `performance_summary.csv`
- `equity_curves.png`
- `sharpe_bars.png`

### Sensitivity Analysis
Test robustness to transaction costs and periods:
```bash
python sensitivity_analysis.py
```
Produces:
- `sensitivity_summary.csv`
- `sharpe_vs_slippage.png`

### Statistical Tests
Compute bootstrap confidence intervals on daily return differences:
```bash
python stats_tests.py
```
Generates `statistical_tests.csv` and prints CI.

### Compile Manuscript
Compile the LaTeX source to generate the PDF:
```bash
pdflatex paper.tex
```
Or upload `paper.tex` to Overleaf.

## Docker (Optional)
Build and run in a containerized environment:
```bash
docker build -t retail-portfolio .
docker run --rm -v $(pwd):/app -w /app retail-portfolio python backtest.py
```

## Contributing
Contributions, issues, and feature requests are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References
- Heaton, J. B., Polson, N. G., & Witte, J. H. (2017). Deep Learning in Finance. *Annual Review of Financial Economics*, 9, 65–92.
- Zhang, Y., & Aggarwal, C. (2024). Machine Learning in Portfolio Management: A Survey. *Journal of Finance and Data Science*, 7(3), 149–169.
- Barber, B. M., & Odean, T. (2001). Boys Will Be Boys: Gender, Overconfidence, and Common Stock Investment. *Quarterly Journal of Economics*, 116(1), 261–292.