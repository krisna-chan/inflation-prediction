# US Inflation Forecasting with Machine Learning

Forecasts US monthly CPI inflation using AR, LASSO, GBDT, Random Forest, and an
ensemble combination. Features 17 macroeconomic predictors from FRED and local
CSV files, with rolling-window evaluation, Clark-West significance tests, and
variable importance analysis.

## Requirements

- Python 3.9+
- pip / venv

## Setup

### 1. Clone & enter

```bash
git clone <repo-url>
cd inflation_prediction
```

### 2. Create virtual environment

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate
```

**Windows (cmd):**

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) FRED API key

The script fetches 10 series live from FRED via `pandas-datareader`. Recent
versions require an API key. Set it as an environment variable (recommended) or
add it to the script:

```bash
export FRED_API_KEY=your_key_here
```

Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html

If no key is set, the script will warn about failed fetches but will still run
using the 7 local CSV files in `Data/`.

## Run

```bash
python us_inflation_forecast.py
```

Output (figures, predictions CSV) is written to `Output/`. Tables and results
are printed to the console.

## Project structure

```
inflation_prediction/
├── Data/                         # Local CSV input files (7 series)
│   ├── CPIAUCSL.csv
│   ├── POILWTIUSDM.csv
│   ├── PPIACO.csv
│   └── ...
├── Output/                       # Generated outputs (figures, CSV)
│   ├── figure_1_inflation_series.png
│   ├── figure_2_variable_importance.png
│   └── predictions_fixed.csv
├── docs/
│   └── METHODOLOGY.md            # Detailed methodology explanation
├── us_inflation_forecast.py      # Main pipeline script
├── requirements.txt              # Python dependencies
└── .gitignore
```

## Methodology

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for a full description of the
data pipeline, model architectures, hyperparameter tuning, evaluation metrics
(RMSE, R²_OOS, Clark-West test), and robustness checks.
