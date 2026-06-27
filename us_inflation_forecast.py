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
from scipy.interpolate import CubicSpline
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

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
    """Cubic-spline interpolation of a quarterly series to monthly frequency."""
    # Map dates to numeric (days since epoch) for spline
    x_old = np.array([d.toordinal() for d in series.index], dtype=float)
    y_old = series.values.astype(float)
    x_new = np.array([d.toordinal() for d in new_index], dtype=float)
    cs = CubicSpline(x_old, y_old, bc_type='natural')
    interpolated = cs(x_new)
    return pd.Series(interpolated, index=new_index)


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

    # --- short-gap cubic spline interpolation ---
    for col in df.columns:
        if col == 'INFLATION':
            continue
        # Only interpolate gaps up to 6 months
        masked = df[col].copy()
        n_missing = masked.isna().sum()
        if n_missing > 0:
            # Use time-based interpolation for gaps
            good = ~masked.isna()
            if good.sum() >= 4:
                x_old = np.array([d.toordinal() for d in df.index[good]], dtype=float)
                y_old = masked[good].values.astype(float)
                x_all = np.array([d.toordinal() for d in df.index], dtype=float)
                try:
                    cs = CubicSpline(x_old, y_old, bc_type='natural', extrapolate=False)
                    interpolated = cs(x_all)
                    # Only fill where we can interpolate (not extrapolate)
                    fill_mask = np.isnan(masked.values) & ~np.isnan(interpolated)
                    masked.values[fill_mask] = interpolated[fill_mask]
                except Exception:
                    pass
            # Forward fill remaining small gaps
            masked.ffill(inplace=True)
            df[col] = masked

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
def train_ar(y_train, y_test_prev, max_lag=12):
    """
    AR(p) model with optimal lag chosen by AIC.
    Returns forecast for next value given y_test_prev (last known values).
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
        # fallback to p=1
        best_model = AutoReg(y_train, lags=1, old_names=False).fit()
        best_lag = 1

    # Build combined series for prediction
    # We need the last best_lag observations from y_train (which includes y_test_prev
    # if test has started, or just the tail of training)
    # But actually: y_train ends at time t, and we need to forecast t+h.
    # For h=1, we need (y_t, y_{t-1}, ..., y_{t-p+1}) which are in y_train.
    combined = np.concatenate([y_train, y_test_prev]) if len(y_test_prev) > 0 else y_train
    # Use the last best_lag values
    last_vals = combined[-best_lag:] if len(combined) >= best_lag else combined
    # Pad if needed
    if len(last_vals) < best_lag:
        last_vals = np.pad(last_vals, (best_lag - len(last_vals), 0), mode='edge')
    return best_model.forecast(steps=1, exog=None)[0]


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
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.predict(X_test.reshape(1, -1))[0], gs.best_estimator_


def train_lasso(X_train, y_train, X_test):
    """LASSO with cross-validated alpha."""
    alphas = [0.001, 0.01, 0.1, 1]
    model = LassoCV(
        alphas=alphas,
        cv=TimeSeriesSplit(n_splits=5),
        random_state=RANDOM_STATE,
        max_iter=10000,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test.reshape(1, -1))[0], model


def train_rf(X_train, y_train, X_test):
    """Random Forest with CV hyperparameter tuning."""
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [4, 6, 8],
    }
    base = RandomForestRegressor(random_state=RANDOM_STATE)
    gs = GridSearchCV(
        base, param_grid,
        cv=TimeSeriesSplit(n_splits=5),
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
    predictor_cols = [c for c in df.columns if c != 'INFLATION']
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
        results[h] = {m: {'forecasts': [], 'actuals': []} for m in models}

    # For variable importance collection (h=1 only)
    vi_predictions = {}  # predictor -> list of forecasts when zeroed

    # Train/test indicators for each horizon
    for t in range(oos_start_idx, len(df) - max(horizons)):
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
            if h == 1:
                X_train_raw = X[train_start:train_end]
                X_test_raw = X[t]
            else:
                # For longer horizons, training uses t - window_size .. t - h
                X_train_raw = X[train_start:train_end + 1 - h]
                X_test_raw = X[t]

            # Add lagged inflation as features for ML models
            # Lags: π_{t-1}, π_{t-2}, ..., π_{t-12}
            n_lags = 6  # Use 6 lags for ML
            def _build_ml_features(data_idx, target_idx):
                """Build feature matrix with lagged inflation + predictors."""
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

            # Training features for ML
            train_indices = list(range(train_start, train_end + 1 - h))
            if len(train_indices) < n_lags + 1:
                continue
            # Skip first n_lags indices to match the target alignment
            X_train_ml = _build_ml_features(train_indices[n_lags:], None)

            # Test feature for ML
            X_test_ml = _build_ml_features([t], None)[0]

            # AR: only uses lags of inflation
            y_train_ar = y[train_start:train_end + 1 - h]
            # Previous values for AR (needed for recursive)
            y_prev_ar = np.array([])  # We'll use the end of y_train_ar

            # --- Train models ---
            # AR
            y_ar_forecast = train_ar(y_train_ar, np.array([]))

            # GBDT
            y_gbdt_forecast, gbdt_model = train_gbdt(X_train_ml, y[train_start + n_lags:train_end + 1 - h], X_test_ml)

            # LASSO
            y_lasso_forecast, lasso_model = train_lasso(X_train_ml, y[train_start + n_lags:train_end + 1 - h], X_test_ml)

            # RF
            y_rf_forecast, rf_model = train_rf(X_train_ml, y[train_start + n_lags:train_end + 1 - h], X_test_ml)

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

            # Variable importance (only h=1, track GBDT, LASSO, RF)
            if h == 1:
                for mod_name, mod_obj in [('GBDT', gbdt_model), ('LASSO', lasso_model), ('RF', rf_model)]:
                    for j, col in enumerate(predictor_cols):
                        # Zero out the predictor (or permute)
                        X_test_perm = X_test_ml.copy()
                        # Offset by n_lags to get predictor index
                        pred_idx_in_ml = n_lags + j
                        X_test_perm[pred_idx_in_ml] = 0.0
                        if mod_name == 'GBDT':
                            perm_fc = mod_obj.predict(X_test_perm.reshape(1, -1))[0]
                        elif mod_name == 'LASSO':
                            perm_fc = mod_obj.predict(X_test_perm.reshape(1, -1))[0]
                        else:
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


def save_predictions(results):
    """Save all forecasts to a CSV file."""
    records = []
    for h, model_results in results.items():
        for model_name, data in model_results.items():
            n = len(data['forecasts'])
            if n == 0:
                continue
            # We need dates – use the first n actuals dates
            # results doesn't store dates, so we approximate
            for i in range(n):
                records.append({
                    'horizon': h,
                    'model': model_name,
                    'forecast_idx': i,
                    'forecast': data['forecasts'][i],
                    'actual': data['actuals'][i],
                })
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_DIR / 'predictions.csv', index=False)
    logger.info('Saved predictions to Output/predictions.csv (%d rows)', len(out_df))


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
# 9. main pipeline
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
    save_predictions(results)

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
    logger.info('Pipeline complete.')


if __name__ == '__main__':
    main()
