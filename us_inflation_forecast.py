#!/usr/bin/env python3
"""
US Inflation Forecasting with Machine Learning

Implements a fully automated supervised ML pipeline to forecast the US
inflation rate (month-over-month % change in CPI) using a rich set of
macroeconomic predictors. Follows the methodology of:

    "Forecasting China's inflation rate: Evidence from machine learning methods"

including rolling window evaluation, performance metrics (RMSE, R²_OOS,
Clark-West test), and variable importance analysis.

Models:
    - AR (autoregressive benchmark, optimal lag via AIC)
    - GBDT (Gradient Boosted Decision Trees)
    - LASSO (L1-regularized linear regression)
    - RF (Random Forest)
    - Comb (equal-weighted average of GBDT, LASSO, RF)

Author: Auto-generated
"""

import os
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import matplotlib
import os
if not os.environ.get('DISPLAY') and not os.environ.get('MPLBACKEND'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance as sk_perm_importance
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

try:
    from tqdm.autonotebook import tqdm
    HAS_TQDM = True
except ImportError:
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        tqdm = None
        HAS_TQDM = False

# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DATA_DIR = Path('Data')
OUTPUT_DIR = Path('Output')
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. data acquisition
# ---------------------------------------------------------------------------
LOCAL_FILES = {
    'CPIAUCSL': 'CPIAUCSL.csv',
    'POILWTIUSDM': 'POILWTIUSDM.csv',
    'CIS1020000000000I': 'CIS1020000000000I.csv',
    'A191RI1Q225SBEA': 'A191RI1Q225SBEA.csv',
    'PCE': 'PCE.csv',
    'PPIACO': 'PPIACO.csv',
    'FRBATLWGTUMHWGO': 'FRBATLWGTUMHWGO.csv',
}

FRED_SERIES = {
    'INDPRO': 'Industrial Production Index',
    'TCU': 'Capacity Utilization',
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Total Nonfarm Payrolls',
    'SP500': 'S&P 500 Index',
    'GS10': '10-Year Treasury Yield',
    'M2SL': 'M2 Money Supply',
    'BAA': 'Moody\'s Baa Bond Yield',
    'TWEXBPA': 'Trade Weighted Dollar Index',
    'FEDFUNDS': 'Federal Funds Rate',
}


def load_local_data():
    series = {}
    for name, fn in LOCAL_FILES.items():
        fp = DATA_DIR / fn
        if not fp.exists():
            logger.warning('Local file %s not found – skipping', fp)
            continue
        df = pd.read_csv(fp)
        if 'observation_date' not in df.columns:
            logger.warning('File %s missing observation_date – skipping', fn)
            continue
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df.set_index('observation_date', inplace=True)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        s = df.iloc[:, 0].dropna()
        logger.info('Loaded %s: %s – %s (%d obs)',
                     name, s.index[0].strftime('%Y-%m'),
                     s.index[-1].strftime('%Y-%m'), len(s))
        series[name] = s
    return series


def fetch_fred_data():
    try:
        import pandas_datareader.data as web
    except ImportError:
        logger.warning('pandas-datareader not installed – skipping FRED fetch')
        return {}

    series = {}
    for sid, desc in FRED_SERIES.items():
        try:
            raw = web.DataReader(
                sid, 'fred', start='1950-01-01',
                end=datetime.now().strftime('%Y-%m-%d')
            )
            s = raw.squeeze().dropna()
            s = pd.to_numeric(s, errors='coerce').dropna()
            logger.info('Fetched %s (%s): %s – %s (%d obs)',
                         sid, desc,
                         s.index[0].strftime('%Y-%m'),
                         s.index[-1].strftime('%Y-%m'), len(s))
            series[sid] = s
        except Exception as e:
            logger.warning('Failed to fetch %s (%s): %s', sid, desc, e)
    return series


# ---------------------------------------------------------------------------
# 2. data preprocessing
# ---------------------------------------------------------------------------
def _to_end_of_month(series):
    """Shift dates to end-of-month (last calendar day) and deduplicate."""
    idx = pd.DatetimeIndex([
        d + pd.offsets.MonthEnd(0) if d.is_month_end
        else d + pd.offsets.MonthEnd(1)
        for d in series.index
    ])
    series = series.copy()
    series.index = idx
    return series.groupby(level=0).last()


def interpolate_quarterly_to_monthly(series, new_index):
    """Forward-fill interpolation of a quarterly series to monthly frequency.
    Uses ffill instead of cubic spline to avoid leaking future information
    into the training window."""
    return series.reindex(new_index).ffill()


def preprocess_data(local, fred):
    """
    Merge, align, interpolate, and clean the full dataset.
    Returns a single DataFrame with monthly observations.
    """
    all_series = {}

    # --- local ---
    for name, s in local.items():
        all_series[name] = _to_end_of_month(s)

    # --- fred ---
    for name, s in fred.items():
        all_series[name] = _to_end_of_month(s)

    # --- quarterly → monthly interpolation ---
    # Get overall monthly date range
    all_dates = []
    for s in all_series.values():
        all_dates.extend(s.index.tolist())
    full_range = pd.date_range(
        start=min(all_dates), end=max(all_dates), freq='MS'
    ) + pd.offsets.MonthEnd(0)

    for name in list(all_series.keys()):
        s = all_series[name]
        # Check if series is quarterly (gaps > 2 months)
        if len(s) > 3:
            gaps = np.diff(s.index.to_julian_date())
            if np.median(gaps) > 80:   # quarterly ≈ 91 days
                logger.info('Interpolating %s from quarterly → monthly', name)
                interp = interpolate_quarterly_to_monthly(s, full_range)
                all_series[name] = interp

    # --- merge into one DataFrame ---
    df = pd.DataFrame(all_series)
    df.index.name = 'date'

    # --- standardise dates to end-of-month ---
    df.index = pd.DatetimeIndex([
        d + pd.offsets.MonthEnd(0) if d.is_month_end
        else d + pd.offsets.MonthEnd(1)
        for d in df.index
    ])
    df = df[~df.index.duplicated(keep='first')]

    # --- target variable: monthly inflation ---
    # π_t = (CPI_t / CPI_{t-1} - 1) × 100
    if 'CPIAUCSL' in df.columns:
        df['INFLATION'] = df['CPIAUCSL'].pct_change() * 100.0
    else:
        raise ValueError('CPIAUCSL is required for target computation')

    # --- forward fill remaining gaps (no future-leaking splines) ---
    for col in df.columns:
        if col == 'INFLATION':
            continue
        df[col] = df[col].ffill(limit=6)

    # --- drop predictors that are almost all NaN ---
    min_obs = int(len(df) * 0.5)
    before = df.shape[1]
    df = df.dropna(thresh=min_obs, axis=1)
    if df.shape[1] < before:
        logger.info('Dropped %d columns with < 50%% non-NaN', before - df.shape[1])

    # --- determine common period (80% availability) ---
    # We need INFLATION to be available
    required = ['INFLATION']
    predictors = [c for c in df.columns if c not in required]

    # Find earliest date where >= 80% of predictors are non-NaN
    n_predictors = len(predictors)
    if n_predictors == 0:
        raise ValueError('No predictors available')

    n_needed = max(1, int(np.ceil(0.8 * n_predictors)))
    # Count available predictors per row
    avail = df[predictors].notna().sum(axis=1)
    valid = avail >= n_needed
    if not valid.any():
        raise ValueError('No rows meet 80% predictor availability')

    first_valid = valid.idxmax()
    last_valid = df.index[-1]
    logger.info('80%% predictor availability from %s', first_valid.strftime('%Y-%m'))

    # Also require INFLATION non-NaN
    inf_ok = df['INFLATION'].notna()
    first_valid = max(first_valid, df.index[inf_ok.argmax()])

    df = df.loc[first_valid:last_valid].copy()

    # --- forward-fill any remaining early NaN within each series ---
    df[predictors] = df[predictors].ffill()

    # Drop any remaining NaNs
    df.dropna(inplace=True)

    logger.info('Final dataset: %d obs × %d predictors (%s – %s)',
                len(df), len(predictors),
                df.index[0].strftime('%Y-%m'), df.index[-1].strftime('%Y-%m'))
    return df


# ---------------------------------------------------------------------------
# 3. models
# ---------------------------------------------------------------------------
def train_ar(y_train, y_test_prev, max_lag=12, steps=1):
    """
    AR(p) model with optimal lag chosen by AIC.
    Forecasts `steps` ahead from the end of y_train.
    """
    best_aic = np.inf
    best_model = None
    best_lag = 0
    for p in range(1, max_lag + 1):
        if len(y_train) <= p:
            break
        try:
            mod = AutoReg(y_train, lags=p, old_names=False)
            res = mod.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_model = res
                best_lag = p
        except Exception:
            continue
    if best_model is None:
        best_model = AutoReg(y_train, lags=1, old_names=False).fit()
        best_lag = 1

    combined = np.concatenate([y_train, y_test_prev]) if len(y_test_prev) > 0 else y_train
    last_vals = combined[-best_lag:] if len(combined) >= best_lag else combined
    if len(last_vals) < best_lag:
        last_vals = np.pad(last_vals, (best_lag - len(last_vals), 0), mode='edge')
    return best_model.forecast(steps=steps, exog=None)[-1]


def _ts_cv_split(n_samples, n_splits=5):
    """Generates train/test indices for time-series CV."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return tscv.split(range(n_samples))


def train_gbdt(X_train, y_train, X_test):
    """GBDT with CV hyperparameter tuning."""
    param_grid = {
        'n_estimators': [80, 100, 120],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8, 1.0],
    }
    base = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gs = GridSearchCV(
        base, param_grid,
        cv=TimeSeriesSplit(n_splits=3, max_train_size=96),
        scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.predict(X_test.reshape(1, -1))[0], gs.best_estimator_


def train_lasso(X_train, y_train, X_test):
    """LASSO with cross-validated alpha and standardized features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test.reshape(1, -1))
    alphas = [0.001, 0.01, 0.1, 1]
    model = LassoCV(
        alphas=alphas,
        cv=TimeSeriesSplit(n_splits=3, max_train_size=96),
        random_state=RANDOM_STATE,
        max_iter=10000,
    )
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)[0], model


def train_rf(X_train, y_train, X_test):
    """Random Forest with CV hyperparameter tuning."""
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [4, 6, 8],
    }
    base = RandomForestRegressor(random_state=RANDOM_STATE)
    gs = GridSearchCV(
        base, param_grid,
        cv=TimeSeriesSplit(n_splits=3, max_train_size=96),
        scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.predict(X_test.reshape(1, -1))[0], gs.best_estimator_


# ---------------------------------------------------------------------------
# 4. rolling window evaluation
# ---------------------------------------------------------------------------
def compute_clark_west(errors_ar, errors_ml, forecasts_ar, forecasts_ml):
    """
    Clark-West (2007) test statistic for nested model comparison.
    H0: MSE(AR) <= MSE(ML)  (ML does not outperform)
    H1: MSE(AR) >  MSE(ML)  (ML outperforms)
    Returns: (CW_stat, p_value_one_sided)
    """
    errors_ar = np.asarray(errors_ar)
    errors_ml = np.asarray(errors_ml)
    forecasts_ar = np.asarray(forecasts_ar)
    forecasts_ml = np.asarray(forecasts_ml)

    # Adjusted MSE differential
    f_t = errors_ar ** 2 - errors_ml ** 2 + (forecasts_ar - forecasts_ml) ** 2

    # Remove NaN
    f_t = f_t[~np.isnan(f_t)]
    P = len(f_t)
    if P < 2:
        return 0.0, 0.5

    mean_f = np.mean(f_t)
    var_f = np.var(f_t, ddof=1)
    if var_f <= 0:
        return 0.0, 0.5

    cw_stat = np.sqrt(P) * mean_f / np.sqrt(var_f)
    p_value = 1 - scipy_stats.norm.cdf(cw_stat)
    return cw_stat, p_value


def evaluate(
    df,
    window_size=96,
    horizons=(1, 3, 5),
    oos_start=None,
    benchmark='AR',
):
    """
    Rolling-window evaluation for all models.

    Parameters
    ----------
    df : DataFrame with 'INFLATION' column and predictors.
    window_size : int, training window length in months.
    horizons : tuple of int, forecast horizons.
    oos_start : str or None, start date for OOS period.
    benchmark : 'AR' or 'HA' (historical average).

    Returns
    -------
    results : dict of horizon -> {model -> {'forecasts': [...], 'actuals': [...]}}
    models_dict : dict of horizon -> {model -> trained model objects (last window)}
    """
    # Ensure sorted
    df = df.sort_index()

    y = df['INFLATION'].values
    # Drop CPIAUCSL level: it's redundant with 6 lags of inflation (its pct_change)
    predictor_cols = [c for c in df.columns if c not in ('INFLATION', 'CPIAUCSL')]
    X = df[predictor_cols].values
    dates = df.index

    # Determine OOS start
    if oos_start is not None:
        oos_start_idx = max(window_size, df.index.get_indexer([pd.Timestamp(oos_start)], method='pad')[0])
    else:
        oos_start_idx = window_size  # start after first window

    oos_start_date = dates[oos_start_idx]
    logger.info('OOS period starts at %s (index %d)', oos_start_date.strftime('%Y-%m'), oos_start_idx)

    models = ['AR', 'GBDT', 'LASSO', 'RF', 'Comb']
    results = {}
    for h in horizons:
        results[h] = {m: {'forecasts': [], 'actuals': [], 'dates': []} for m in models}

    # For variable importance collection (h=1 only)
    vi_predictions = {}  # predictor -> list of forecasts when zeroed

    # Train/test indicators for each horizon
    step_range = range(oos_start_idx, len(df) - max(horizons))
    total_steps = len(step_range)
    step_iter = tqdm(step_range, desc="Rolling window", total=total_steps) if HAS_TQDM else step_range
    for t in step_iter:
        # Training range: [t - window_size, t]
        train_start = t - window_size
        train_end = t  # inclusive

        for h in horizons:
            test_idx = t + h
            if test_idx >= len(df):
                continue

            y_actual = y[test_idx]

            # --- Feature building ---
            # Use predictors and lags of inflation
            X_train_raw = X[train_start:train_end]
            X_test_raw = X[t]

            # Add lagged inflation as features for ML models
            n_lags = 6
            def _build_ml_features(data_idx):
                features = []
                for i in data_idx:
                    lag_vals = []
                    for lag in range(1, n_lags + 1):
                        if i - lag >= 0:
                            lag_vals.append(y[i - lag])
                        else:
                            lag_vals.append(0.0)
                    row = np.concatenate([lag_vals, X[i]])
                    features.append(row)
                return np.array(features)

            # --- Horizon-correct training ---
            # Feature at index i, label at y[i+h]
            # train_indices range ensures last label = y[t] (known at time t)
            train_indices = list(range(train_start, train_end + 1 - h))
            if len(train_indices) < n_lags + 1:
                continue
            feat_indices = train_indices[n_lags:]
            X_train_ml = _build_ml_features(feat_indices)
            y_train_ml = y[np.array(feat_indices) + h]

            # Test feature for ML (from current time t, predict t+h)
            X_test_ml = _build_ml_features([t])[0]

            # AR: train on y up to t, forecast h steps
            y_train_ar = y[train_start:train_end + 1]
            y_ar_forecast = train_ar(y_train_ar, np.array([]), steps=h)

            # --- Train models ---
            # GBDT
            y_gbdt_forecast, gbdt_model = train_gbdt(X_train_ml, y_train_ml, X_test_ml)

            # LASSO
            y_lasso_forecast, lasso_model = train_lasso(X_train_ml, y_train_ml, X_test_ml)

            # RF
            y_rf_forecast, rf_model = train_rf(X_train_ml, y_train_ml, X_test_ml)

            # Combination
            y_comb_forecast = (y_gbdt_forecast + y_lasso_forecast + y_rf_forecast) / 3.0

            # Store
            forecast_map = {
                'AR': y_ar_forecast,
                'GBDT': y_gbdt_forecast,
                'LASSO': y_lasso_forecast,
                'RF': y_rf_forecast,
                'Comb': y_comb_forecast,
            }

            for model_name, fc in forecast_map.items():
                results[h][model_name]['forecasts'].append(fc)
                results[h][model_name]['actuals'].append(y_actual)
                results[h][model_name]['dates'].append(dates[test_idx])

            # Variable importance (h=1 only, via random substitution from training distribution)
            if h == 1:
                for mod_name, mod_obj in [('GBDT', gbdt_model), ('LASSO', lasso_model), ('RF', rf_model)]:
                    for j, col in enumerate(predictor_cols):
                        X_test_perm = X_test_ml.copy()
                        pred_idx_in_ml = n_lags + j
                        rng = np.random.RandomState(RANDOM_STATE + t + j)
                        X_test_perm[pred_idx_in_ml] = rng.choice(X_train_raw[:, j])
                        perm_fc = mod_obj.predict(X_test_perm.reshape(1, -1))[0]

                        key = (mod_name, col)
                        if key not in vi_predictions:
                            vi_predictions[key] = {'base': [], 'perm': []}
                        vi_predictions[key]['base'].append(forecast_map[mod_name])
                        vi_predictions[key]['perm'].append(perm_fc)

    logger.info('Rolling window evaluation complete')
    return results, vi_predictions, predictor_cols


# ---------------------------------------------------------------------------
# 5. metrics & tables
# ---------------------------------------------------------------------------
def compute_metrics(results, benchmark='AR'):
    """
    Compute RMSE, R²_OOS, and Clark-West test for each model and horizon.
    Returns nested dict: horizon -> model -> {rmse, r2_oos, cw_stat, cw_pval}
    """
    metrics = {}
    for h, model_results in results.items():
        metrics[h] = {}
        ar_errors = None
        if benchmark in model_results:
            ar_errors = np.array(model_results[benchmark]['actuals']) - np.array(model_results[benchmark]['forecasts'])

        for model_name, data in model_results.items():
            actuals = np.array(data['actuals'])
            forecasts = np.array(data['forecasts'])
            errors = actuals - forecasts
            rmse = np.sqrt(np.mean(errors ** 2))

            # R²_OOS relative to benchmark
            if benchmark in model_results and model_name != benchmark:
                ar_mse = np.mean(ar_errors ** 2)
                mse = np.mean(errors ** 2)
                r2_oos = 1.0 - mse / ar_mse if ar_mse > 0 else np.nan
            else:
                r2_oos = np.nan

            # Clark-West test (against AR)
            if model_name != benchmark and ar_errors is not None:
                cw_stat, cw_pval = compute_clark_west(
                    ar_errors, errors,
                    np.array(model_results[benchmark]['forecasts']),
                    forecasts,
                )
            else:
                cw_stat, cw_pval = np.nan, np.nan

            metrics[h][model_name] = {
                'rmse': rmse,
                'r2_oos': r2_oos,
                'cw_stat': cw_stat,
                'cw_pval': cw_pval,
            }
    return metrics


def print_tables(metrics, horizons=(1, 3, 5)):
    """Print formatted tables of RMSE and R²_OOS."""
    models_order = ['AR', 'GBDT', 'LASSO', 'RF', 'Comb']

    print('\n' + '=' * 90)
    print('TABLE 1: Out-of-Sample RMSE')
    print('=' * 90)
    header = f"{'Model':<10}" + ''.join(f"{f'h={h}':<20}" for h in horizons)
    print(header)
    print('-' * 90)
    for m in models_order:
        row = f"{m:<10}"
        for h in horizons:
            if m in metrics.get(h, {}):
                rmse = metrics[h][m]['rmse']
                row += f"{rmse:<20.4f}"
            else:
                row += f"{'N/A':<20}"
        print(row)
    print('-' * 90)

    print('\n' + '=' * 90)
    print('TABLE 2: Out-of-Sample R² (R²_OOS relative to AR)')
    print('=' * 90)
    header = f"{'Model':<10}" + ''.join(f"{f'h={h}':<20}" for h in horizons)
    print(header)
    print('-' * 90)
    for m in models_order:
        if m == 'AR':
            continue
        row = f"{m:<10}"
        for h in horizons:
            if m in metrics.get(h, {}):
                r2 = metrics[h][m]['r2_oos']
                row += f"{r2:<20.4%}"
            else:
                row += f"{'N/A':<20}"
        print(row)
    print('-' * 90)

    print('\n' + '=' * 90)
    print('TABLE 3: Clark-West Test (against AR benchmark)')
    print('=' * 90)
    header = f"{'Model':<10}" + ''.join(f"{f'h={h} (stat/pval)':<30}" for h in horizons)
    print(header)
    print('-' * 90)
    for m in models_order:
        if m == 'AR':
            continue
        row = f"{m:<10}"
        for h in horizons:
            if m in metrics.get(h, {}):
                cw = metrics[h][m]['cw_stat']
                pv = metrics[h][m]['cw_pval']
                sig = '*' if pv < 0.10 else ''
                row += f"{cw:<7.3f}/{pv:<7.3f}{sig:<14}"
            else:
                row += f"{'N/A':<30}"
        print(row)
    print('-' * 90)


# ---------------------------------------------------------------------------
# 6. variable importance
# ---------------------------------------------------------------------------
def compute_variable_importance(vi_predictions, results, predictor_names):
    """
    Compute drop in R²_OOS when each predictor is zeroed.
    For each model and predictor, compute the average drop in R²_OOS
    across the OOS period.
    """
    # Get h=1 AR errors for R²_OOS baseline
    ar_errors = np.array(results[1]['AR']['actuals']) - np.array(results[1]['AR']['forecasts'])
    ar_mse = np.mean(ar_errors ** 2)

    importance = {}
    models_vi = ['GBDT', 'LASSO', 'RF']
    for mod_name in models_vi:
        importance[mod_name] = {}
        for j, pred_name in enumerate(predictor_names):
            key = (mod_name, pred_name)
            if key not in vi_predictions:
                continue
            base_fc = np.array(vi_predictions[key]['base'])
            perm_fc = np.array(vi_predictions[key]['perm'])
            actuals = np.array(results[1][mod_name]['actuals'])

            # Only use overlapping samples
            n = min(len(base_fc), len(perm_fc), len(actuals))
            base_fc = base_fc[:n]
            perm_fc = perm_fc[:n]
            actuals = actuals[:n]

            base_errors = actuals - base_fc
            perm_errors = actuals - perm_fc

            base_mse = np.mean(base_errors ** 2)
            perm_mse = np.mean(perm_errors ** 2)

            # Drop in R²_OOS
            base_r2 = 1.0 - base_mse / ar_mse if ar_mse > 0 else 0
            perm_r2 = 1.0 - perm_mse / ar_mse if ar_mse > 0 else 0
            drop = base_r2 - perm_r2  # positive → important

            importance[mod_name][pred_name] = drop

    return importance


def plot_variable_importance(importance, top_k=5):
    """Bar plot of Top-K most influential predictors for GBDT (Figure 2)."""
    if 'GBDT' not in importance:
        logger.warning('No GBDT importance data to plot')
        return

    gbdt_imp = importance['GBDT']
    sorted_vars = sorted(gbdt_imp.items(), key=lambda x: x[1], reverse=True)
    top_vars = sorted_vars[:top_k]

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [v[0] for v in top_vars]
    scores = [v[1] for v in top_vars]

    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(names)))
    bars = ax.barh(range(len(names)), scores, color=colors, edgecolor='navy', linewidth=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Drop in R²_OOS (importance)')
    ax.set_title('Figure 2: Top-5 Most Influential Predictors — GBDT Model')
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.001 * max(scores),
                bar.get_y() + bar.get_height() / 2,
                f'{score:.4f}',
                va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_2_variable_importance.png', dpi=150)
    logger.info('Saved Figure 2 to Output/figure_2_variable_importance.png')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. figures
# ---------------------------------------------------------------------------
def plot_inflation_series(df):
    """Plot monthly US inflation (Figure 1)."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df.index, df['INFLATION'], color='steelblue', linewidth=0.9, label='MoM CPI Inflation (%)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Percent')
    ax.set_title('Figure 1: US Monthly CPI Inflation (Month-over-Month % Change)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_1_inflation_series.png', dpi=150)
    logger.info('Saved Figure 1 to Output/figure_1_inflation_series.png')
    plt.close(fig)


def save_predictions(results, output_name='predictions_fixed.csv'):
    """Save all forecasts to a CSV file, including dates."""
    records = []
    for h, model_results in results.items():
        for model_name, data in model_results.items():
            dates = data.get('dates', [])
            n = len(data['forecasts'])
            if n == 0:
                continue
            for i in range(n):
                d = dates[i].strftime('%Y-%m-%d') if i < len(dates) else ''
                records.append({
                    'horizon': h,
                    'model': model_name,
                    'date': d,
                    'forecast': data['forecasts'][i],
                    'actual': data['actuals'][i],
                })
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_DIR / output_name, index=False)
    logger.info('Saved predictions to Output/%s (%d rows)', output_name, len(out_df))


# ---------------------------------------------------------------------------
# 8. robustness checks
# ---------------------------------------------------------------------------
def robustness_checks(
    df,
    window_sizes=(60, 120),
    horizons=(3, 5),
    oos_start=None,
):
    """Re-run evaluation with different window sizes (Table 4 equivalent)."""
    logger.info('--- Robustness: window-size sensitivity ---')
    for w in window_sizes:
        logger.info('Window size = %d months', w)
        res, _, _ = evaluate(
            df,
            window_size=w,
            horizons=horizons,
            oos_start=oos_start,
        )
        met = compute_metrics(res, benchmark='AR')
        print(f'\n--- Robustness: window = {w} months ---')
        models_order = ['GBDT', 'LASSO', 'RF', 'Comb']
        header = f"{'Model':<10}" + ''.join(f"{f'h={h} (RMSE / R²)':<28}" for h in horizons)
        print(header)
        print('-' * 90)
        for m in models_order:
            row = f"{m:<10}"
            for h in horizons:
                if m in met.get(h, {}):
                    rmse = met[h][m]['rmse']
                    r2 = met[h][m]['r2_oos']
                    row += f"{rmse:<8.4f} / {r2:<8.4%}       "
                else:
                    row += f"{'N/A':<28}"
            print(row)
        print('-' * 90)


# ---------------------------------------------------------------------------
# 9. evaluation diagnostics (added by evaluator)
# ---------------------------------------------------------------------------

def evaluate_forecasts(y_true, y_pred_ar, y_pred_ml, model_name, h):
    mse_ar = mean_squared_error(y_true, y_pred_ar)
    mse_ml = mean_squared_error(y_true, y_pred_ml)
    rmse_ml = np.sqrt(mse_ml)
    mae_ml = mean_absolute_error(y_true, y_pred_ml)

    r2_oos = 1 - (mse_ml / mse_ar)

    true_diff = np.diff(y_true, prepend=np.nan)
    pred_diff = y_pred_ml - np.roll(y_true, 1)
    valid_idx = ~np.isnan(true_diff) & ~np.isnan(pred_diff)
    directional_acc = np.mean(np.sign(true_diff[valid_idx]) == np.sign(pred_diff[valid_idx])) * 100

    e_ar = y_true - y_pred_ar
    e_ml = y_true - y_pred_ml
    f_t = e_ar**2 - e_ml**2 + (y_pred_ar - y_pred_ml)**2
    cw_stat = np.sqrt(len(f_t)) * np.mean(f_t) / np.std(f_t, ddof=1)
    p_value = 1 - norm.cdf(cw_stat)

    print(f"=== {model_name} (h={h}) ===")
    print(f"RMSE: {rmse_ml:.6f}")
    print(f"MAE:  {mae_ml:.6f}")
    print(f"R\u00b2_OOS (vs AR): {r2_oos:.4f}")
    print(f"Directional Acc: {directional_acc:.1f}%")
    print(f"CW Stat: {cw_stat:.3f} (p={p_value:.4f})")
    print("-" * 40)

    return {
        'model': model_name,
        'horizon': h,
        'rmse': rmse_ml,
        'mae': mae_ml,
        'r2_oos': r2_oos,
        'dir_acc': directional_acc,
        'cw_stat': cw_stat,
        'cw_p': p_value,
    }


def sub_period_analysis(results, horizons=(3, 5)):
    """Output B: Sub-period RMSE for AR and Comb."""
    periods = [
        ('2008\u20132019 (Low Inflation)', '2008-01-01', '2019-12-31'),
        ('2020\u20132021 (COVID Shock)',   '2020-01-01', '2021-12-31'),
        ('2022\u20132026 (Post-COVID)',    '2022-01-01', '2026-12-31'),
    ]
    print('\n' + '=' * 90)
    print('OUTPUT B: SUB-PERIOD ROBUSTNESS (RMSE)')
    print('=' * 90)
    for h in horizons:
        print(f'\n--- Horizon h={h} ---')
        header = f"{'Period':<35} {'AR':<12} {'Comb':<12} {'Winner':<12}"
        print(header)
        print('-' * 70)
        dates = np.array(results[h]['Comb']['dates'])
        ar_fc = np.array(results[h]['AR']['forecasts'])
        ar_act = np.array(results[h]['AR']['actuals'])
        comb_fc = np.array(results[h]['Comb']['forecasts'])
        comb_act = np.array(results[h]['Comb']['actuals'])
        comb_wins_all = True
        for label, start, end in periods:
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            n = mask.sum()
            if n < 2:
                print(f"{label:<35} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue
            rmse_ar = np.sqrt(np.mean((ar_act[mask] - ar_fc[mask])**2))
            rmse_comb = np.sqrt(np.mean((comb_act[mask] - comb_fc[mask])**2))
            winner = 'Comb' if rmse_comb < rmse_ar else 'AR'
            if winner != 'Comb':
                comb_wins_all = False
            print(f"{label:<35} {rmse_ar:<12.4f} {rmse_comb:<12.4f} {winner:<12}")
        if not comb_wins_all:
            flag = "*** BRITTLE REGIME-DEPENDENT MODEL: Comb loses in one or more sub-periods ***"
            print(f"\n  {flag}")
    print('-' * 90)


def rolling_rmse_plot(results, h=3, window=12):
    """Output C: 12-month rolling RMSE for AR vs Comb at horizon h."""
    dates = np.array(results[h]['Comb']['dates'])
    ar_act = np.array(results[h]['AR']['actuals'])
    ar_fc = np.array(results[h]['AR']['forecasts'])
    comb_fc = np.array(results[h]['Comb']['forecasts'])
    comb_act = np.array(results[h]['Comb']['actuals'])

    ar_errors_sq = (ar_act - ar_fc)**2
    comb_errors_sq = (comb_act - comb_fc)**2

    rolling_ar = np.full(len(ar_errors_sq), np.nan)
    rolling_comb = np.full(len(comb_errors_sq), np.nan)

    for i in range(window - 1, len(ar_errors_sq)):
        rolling_ar[i] = np.sqrt(np.mean(ar_errors_sq[i - window + 1:i + 1]))
        rolling_comb[i] = np.sqrt(np.mean(comb_errors_sq[i - window + 1:i + 1]))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(dates, rolling_ar, color='red', linestyle='--', linewidth=0.9, label='AR (12mo rolling RMSE)')
    ax.plot(dates, rolling_comb, color='steelblue', linestyle='-', linewidth=0.9, label='Comb (12mo rolling RMSE)')
    ax.set_ylabel('RMSE')
    ax.set_title(f'Rolling {window}-month RMSE — AR vs Combination (h={h})')
    ax.legend(loc='upper right', framealpha=0.9)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'rolling_rmse_h3.png', dpi=150)
    logger.info('Saved rolling RMSE plot to Output/rolling_rmse_h3.png')
    plt.close(fig)


def output_a_table(metrics, horizons=(1, 3, 5)):
    """Output A: Consolidated diagnostic table."""
    models_order = ['AR', 'GBDT', 'LASSO', 'RF', 'Comb']
    print('\n' + '=' * 90)
    print('OUTPUT A: CONSOLIDATED DIAGNOSTIC TABLE')
    print('=' * 90)
    print(f"{'Model':<10}" + ''.join(f"{f'h={h} RMSE':<14}{f'h={h} R\u00b2':<14}{f'h={h} CW-p':<14}" for h in horizons))
    print('-' * 90)
    for m in models_order:
        row = f"{m:<10}"
        for h in horizons:
            if m in metrics.get(h, {}):
                rmse = metrics[h][m]['rmse']
                r2 = metrics[h][m]['r2_oos']
                cw_p = metrics[h][m]['cw_pval']
                row += f"{rmse:<14.4f}{r2:<14.2%}{cw_p:<14.4f}"
            else:
                row += f"{'N/A':<14}{'N/A':<14}{'N/A':<14}"
        print(row)
    print('=' * 90)


def run_evaluation_diagnostics(results, metrics, df):
    """Run Outputs A, B, C, D."""
    output_a_table(metrics)
    sub_period_analysis(results, horizons=(3, 5))
    rolling_rmse_plot(results, h=3, window=12)

    # Output D: Final Verdict
    print('\n' + '=' * 90)
    print('OUTPUT D: FINAL VERDICT')
    print('=' * 90)
    ml_models = ['GBDT', 'LASSO', 'RF', 'Comb']
    any_significant = False
    for h in [1, 3, 5]:
        for m in ml_models:
            if h in metrics and m in metrics[h]:
                p = metrics[h][m]['cw_pval']
                if not np.isnan(p) and p < 0.10:
                    any_significant = True

    if any_significant:
        print("Statistically significant improvement over AR at the 10% level.")
    else:
        print("No statistical evidence that ML models outperform the AR benchmark.")

    # Basis point saving for h=1 best ML model
    best_bp = 0
    best_ml = None
    for m in ml_models:
        if 1 in metrics and m in metrics[1]:
            rmse_ar = metrics[1]['AR']['rmse']
            rmse_ml = metrics[1][m]['rmse']
            bp = (rmse_ar - rmse_ml) * 100
            if bp > best_bp:
                best_bp = bp
                best_ml = m
    if best_ml:
        print(f"Economic significance: at h=1, {best_ml} reduces RMSE by {best_bp:.1f} basis points "
              f"per forecast relative to AR.")
    print('=' * 90)


# ---------------------------------------------------------------------------
# 10. main pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info('=' * 60)
    logger.info('US Inflation Forecasting Pipeline')
    logger.info('=' * 60)

    # ---- Step 1: Load Data ----
    logger.info('\n--- Step 1: Data Acquisition ---')
    local_data = load_local_data()
    fred_data = fetch_fred_data()

    # ---- Step 2: Preprocess ----
    logger.info('\n--- Step 2: Preprocessing ---')
    df = preprocess_data(local_data, fred_data)

    # ---- Plot Figure 1 ----
    logger.info('\n--- Generating Figure 1 ---')
    plot_inflation_series(df)

    # ---- Step 3 & 4: Rolling Window Evaluation ----
    logger.info('\n--- Step 3/4: Rolling Window Evaluation (window=96) ---')
    oos_start = '2008-01-01'
    results, vi_preds, predictor_names = evaluate(
        df,
        window_size=96,
        horizons=(1, 3, 5),
        oos_start=oos_start,
    )

    # ---- Step 5: Metrics ----
    logger.info('\n--- Step 5: Performance Metrics ---')
    metrics = compute_metrics(results, benchmark='AR')
    print_tables(metrics)

    # Save predictions
    save_predictions(results, output_name='predictions_fixed.csv')

    # ---- Step 6: Variable Importance ----
    logger.info('\n--- Step 6: Variable Importance ---')
    importance = compute_variable_importance(vi_preds, results, predictor_names)

    # Print top-5 for each model
    for model_name in ['GBDT', 'LASSO', 'RF']:
        if model_name not in importance:
            continue
        sorted_vars = sorted(importance[model_name].items(), key=lambda x: x[1], reverse=True)
        print(f'\nTop-5 predictors ({model_name}):')
        for rank, (var, imp) in enumerate(sorted_vars[:5], 1):
            print(f'  {rank}. {var}: {imp:.4f}')

    # Plot Figure 2
    plot_variable_importance(importance, top_k=5)

    # ---- Step 7: Robustness ----
    logger.info('\n--- Step 7: Robustness Checks ---')
    try:
        robustness_checks(df, window_sizes=(60, 120), horizons=(3, 5), oos_start=oos_start)
    except Exception as e:
        logger.warning('Robustness checks failed: %s', e)

    # ---- Summary ----
    print('\n' + '=' * 90)
    print('FINAL SUMMARY')
    print('=' * 90)

    # Find best model for h=1
    best_h1 = None
    best_r2 = -np.inf
    for m in ['GBDT', 'LASSO', 'RF', 'Comb']:
        if m in metrics.get(1, {}):
            r2 = metrics[1][m]['r2_oos']
            if not np.isnan(r2) and r2 > best_r2:
                best_r2 = r2
                best_h1 = m

    actuals_1 = np.array(results[1]['AR']['actuals'])
    oos_start_str = '2008-01'
    oos_end_str = df.index[-min(5, len(df) - 1)].strftime('%Y-%m')
    if len(actuals_1) > 0:
        oos_end_str = results[1]['AR']['actuals'][-1] if False else 'latest'

    # Better OOS period
    # We know the index starts at 2008-01 + window
    # Just use from the results
    date_count = len(results[1]['AR']['forecasts'])
    first_date_str = df.index[df.index.get_indexer([pd.Timestamp(oos_start)], method='pad')[0] - 96 + 1] if False else '2008-01'
    # Simpler: just state what was configured
    print(f'Out-of-sample period: around {oos_start} to {df.index[-6].strftime("%Y-%m")}')
    if best_h1:
        print(f'Best model for h=1: {best_h1}, R²_OOS = {best_r2:.2%}')

    # Top predictor
    if 'GBDT' in importance:
        top_var = max(importance['GBDT'], key=importance['GBDT'].get)
        print(f'Top predictor (GBDT): {top_var}')

    print(f'\nOutput files saved to {OUTPUT_DIR}/')

    # ---- Diagnostics ----
    run_evaluation_diagnostics(results, metrics, df)

    logger.info('Pipeline complete.')


if __name__ == '__main__':
    main()
