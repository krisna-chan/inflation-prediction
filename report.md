# Forecasting US Inflation with Machine Learning: Methodology & Evaluation Report

## 1. Problem Statement

Forecast month-over-month (MoM) US Consumer Price Index (CPI) percentage change at horizons of **1, 3, and 5 months** using 17 macroeconomic predictors and five models spanning linear, regularized, tree-based, and ensemble methods.

The methodology follows the framework of "Forecasting China's Inflation Rate: Evidence from Machine Learning Methods" (Jiang et al.), adapted for the US economy with a rolling-window evaluation protocol, Clark-West nested model tests, and regime-aware diagnostics.

---

## 2. Data

### 2.1 Target Variable

| Variable | Source | FRED Code | Description |
|----------|--------|-----------|-------------|
| CPI Level | FRED (local CSV) | CPIAUCSL | Consumer Price Index for All Urban Consumers: All Items |
| CPI MoM | Derived | — | `pct_change(1) * 100` from CPI Level |

The target is defined as:

```
CPI_MoM_t = (CPI_t / CPI_{t-1} - 1) * 100
```

### 2.2 Predictor Variables — Local CSV Files (7 series)

| Variable | FRED Code | Frequency | Description |
|----------|-----------|-----------|-------------|
| CPI Level | CPIAUCSL | Monthly | Consumer Price Index (used only to derive target, excluded from predictors) |
| WTI Oil Price | POILWTIUSDM | Monthly | Crude oil price — spot oil prices |
| Personal Consumption Expend. | PCE | Monthly | Broader consumption measure than CPI |
| Producer Price Index | PPIACO | Monthly | Wholesale/producer prices |
| Avg. Hourly Earnings | FRBATLWGTUMHWGO | Monthly | Labor cost pressure |
| Coincident Indicators | CIS1020000000000I | Quarterly | Composite index of economic activity |
| Real GDP Growth | A191RI1Q225SBEA | Quarterly | Real GDP quarterly percent change |

### 2.3 Predictor Variables — FRED API (10 series, fetched live)

| Variable | FRED Code | Category | Description |
|----------|-----------|----------|-------------|
| Industrial Production | INDPRO | Real Activity | Index of industrial output |
| Capacity Utilization | TCU | Real Activity | Percent of capacity used |
| Unemployment Rate | UNRATE | Labor Market | Civilian unemployment rate |
| Nonfarm Payrolls | PAYEMS | Labor Market | Total employment |
| S&P 500 | SP500 | Financial Markets | Stock market index |
| 10-Year Treasury Yield | GS10 | Financial Markets | Long-term interest rate |
| Baa Bond Yield | BAA | Credit Markets | Corporate borrowing cost |
| Federal Funds Rate | FEDFUNDS | Monetary Policy | Short-term policy rate |
| M2 Money Supply | M2SL | Monetary Aggregates | Broad money supply |
| Trade-Weighted Dollar | TWEXBPA | External Sector | USD exchange rate index |

### 2.4 Sample Period

- **Raw data range**: 1913-01 to 2026-05 (varies by series, PPIACO starts earliest)
- **Final usable dataset**: 1967-01 to 2026-05 (**711 monthly observations**, 12 predictors after cleaning)
- **Training window**: Fixed 96-month rolling window
- **Out-of-sample period**: 2007-12 to 2026-05 (**~220 test observations**, depending on horizon)
- **OOS start condition**: First month where ≥80% of predictors are non-missing and inflation target is available

### 2.5 Data Quality

| Series | Rows | Start | End | Missing Treatment |
|--------|------|-------|-----|-------------------|
| CPIAUCSL | 952 | 1947-01 | 2026-05 | N/A (target source) |
| POILWTIUSDM | 413 | 1992-01 | 2026-05 | Forward-fill (limit 6) |
| PCE | 809 | 1959-01 | 2026-05 | Forward-fill |
| PPIACO | 1361 | 1913-01 | 2026-05 | Forward-fill |
| FRBATLWGTUMHWGO | 352 | 1997-01 | 2026-05 | Forward-fill |
| CIS1020000000000I | 101 | 2001-01 | 2026-01 | Quarterly→monthly ffill |
| A191RI1Q225SBEA | 316 | 1947-04 | 2026-01 | Quarterly→monthly ffill |

FRED API series span 1950–2026 (~900 obs each) except SP500 (2016–2026, daily resampled) and TWEXBPA (1973–2019, discontinued).

5 of 17 raw series dropped during cleaning (columns with <50% non-NaN).

---

## 3. Preprocessing Pipeline

### 3.1 Date Alignment

All series are shifted to **end-of-month** dates to ensure a uniform monthly index. Duplicate dates within a series are resolved by keeping the last observation.

### 3.2 Quarterly-to-Monthly Interpolation

Quarterly series (Real GDP, Coincident Indicators) are identified via median inter-observation gap (>80 days ≈ quarterly) and forward-filled to monthly frequency:

```
monthly_t = quarterly_value  (each month in the quarter gets the same value)
```

Forward-fill is used instead of spline/cubic interpolation to **prevent future leakage** — spline methods use surrounding values that would not be known at forecast time.

### 3.3 Target Computation

```
CPI_MoM_t = (CPIAUCSL_t / CPIAUCSL_{t-1} - 1) * 100
```

Computed after merging all predictors. First observation lost due to lagged difference.

### 3.4 Short-Gap Interpolation

Remaining missing values within each predictor are forward-filled with a **limit of 6 months**, preventing loss of useful observations while avoiding extrapolation into long data-free regions.

### 3.5 Sparse Column Removal

Predictors with **less than 50% non-NaN values** across the full date range are dropped. This removes 5 of the original 17 series with short historical coverage.

### 3.6 Common Sample Period

The pipeline finds the **earliest month where ≥80% of surviving predictors are non-missing** and inflation is available. This maximizes predictor count while preserving a long historical sample. After trimming, remaining NaN values are forward-filled and rows with any remaining NaNs are dropped.

**Final shape**: 711 obs × 12 predictors (1967-01 to 2026-05).

---

## 4. Feature Engineering

### 4.1 Lag Features

For each ML model, the feature vector consists of two components:

| Component | Count | Description |
|-----------|-------|-------------|
| Lagged CPI inflation | 6 | `CPI_MoM_{t-1}` through `CPI_MoM_{t-6}` |
| Contemporaneous predictors | ~12 | All macro predictor values at time `t` (after cleaning) |
| **Total ML features** | **~18–19** | |

The 6-lag structure captures autoregressive dynamics (momentum, mean-reversion, seasonal patterns). Contemporaneous predictors allow the model to incorporate current macro conditions.

**Rationale for 6 lags**: US CPI inflation autocorrelation decays within 3–6 months. Six lags capture the full decay horizon without excessive feature count (observations-to-features ratio ≈ 5:1 with 96 training obs).

**CPIAUCSL excluded**: The raw CPI level is intentionally omitted from predictors. Since the target is its percentage change and 6 lags of that change are included, the level is nearly a perfect linear combination — adding it would create near-perfect multicollinearity.

### 4.2 No Additional Feature Engineering

Unlike many ML pipelines, this project does **not** use:
- Rolling statistics (means, stds)
- Log transforms
- Year-over-year growth rates
- Interaction terms
- PCA or other dimensionality reduction

The feature set is intentionally minimal: **raw macro values + lagged inflation**. This follows the reference methodology and keeps the comparison between models clean — improvements come from model architecture, not from engineered features.

---

## 5. Feature Selection

No explicit feature selection is performed. All surviving predictors (≈12) plus 6 inflation lags are used in each model.

LASSO includes built-in L1 regularization that implicitly selects features by driving coefficients to zero. This is treated as part of the model, not as a separate selection step.

Tree-based models (GBDT, RF) use all features but are robust to irrelevant predictors due to their axis-aligned split mechanism, which simply ignores uninformative features.

---

## 6. Models

### 6.1 Overview

Five models are estimated at each rolling window step, spanning linear, regularized, tree-based, and ensemble paradigms:

| Model | Family | Parameters | Tuning Method |
|-------|--------|-----------|---------------|
| AR(p) | Univariate time series | Lag p ∈ {1..12} | AIC minimization |
| GBDT | Gradient-boosted trees | n_estimators, max_depth, learning_rate, subsample | GridSearchCV (3-fold TimeSeriesSplit) |
| LASSO | L1-regularized linear | α ∈ {0.001, 0.01, 0.1, 1} | LassoCV (3-fold TimeSeriesSplit) |
| RF | Bagged trees | n_estimators, max_depth | GridSearchCV (3-fold TimeSeriesSplit) |
| Comb | Ensemble | Equal-weighted avg of GBDT + LASSO + RF | None (fixed weights) |

### 6.2 AR (Autoregressive Benchmark)

```
y_t = c + φ₁·y_{t-1} + ... + φ_p·y_{t-p} + ε_t
```

- Optimal lag p selected by AIC at each rolling window
- Multi-step forecasts produced **iteratively** (predicted values fed back as lags)
- Represents the **best purely univariate forecast** — all information in inflation history alone
- Serves as the benchmark for R²_OOS and Clark-West comparisons

### 6.3 GBDT (Gradient Boosted Decision Trees)

Builds an additive ensemble of shallow regression trees, each fitted to the **pseudo-residuals** (gradient of the loss) of the current ensemble:

```
F_M(x) = Σ_{m=1}^{M} γ_m · h_m(x; a_m)
```

Grid search parameters:

| Parameter | Values | Purpose |
|-----------|--------|---------|
| n_estimators | {80, 100, 120} | Ensemble size |
| max_depth | {2, 3, 4} | Tree complexity (shallow → stable interactions) |
| learning_rate | {0.01, 0.1} | Shrinkage factor |
| subsample | {0.8, 1.0} | Stochastic component for regularization |

Shallow trees (depth ≤ 4) are deliberate: monthly macro data is noisy, and deeper trees would memorize transient patterns.

### 6.4 LASSO (L1-Regularized Linear Regression)

```
β̂(λ) = argmin ½n · Σ(y_i - β₀ - Σβⱼx_{ij})² + λ · Σ|βⱼ|
```

- Features **standardized** to zero mean, unit variance before estimation (penalty is scale-dependent)
- L1 penalty drives irrelevant coefficients to exactly zero — automatic feature selection
- α selected from {0.001, 0.01, 0.1, 1} via LassoCV with 3-fold TimeSeriesSplit
- Range spans near-OLS (α=0.001) to heavy shrinkage (α=1)

### 6.5 RF (Random Forest)

Ensemble of B regression trees, each grown on a bootstrap sample with random feature subset at each split:

```
f̂_RF(x) = (1/B) · Σ T_b(x)
```

Grid search parameters:

| Parameter | Values | Purpose |
|-----------|--------|---------|
| n_estimators | {100, 150} | Number of trees |
| max_depth | {4, 6, 8} | Tree depth (deeper than GBDT due to bagging regularization) |

Deeper trees allowed versus GBDT because bagging provides built-in variance reduction through averaging.

### 6.6 Comb (Combination Forecast)

```
ŷ_Comb = (ŷ_GBDT + ŷ_LASSO + ŷ_RF) / 3
```

Equal-weighted average — no training or optimization. The "forecast combination puzzle" (Stock & Watson, 2004) shows simple averages consistently match or outperform sophisticated weighting schemes. Diversification across model families (linear-regularized, sequential-ensemble, parallel-ensemble) reduces error variance and cancels directional biases.

AR is excluded from the combination to maintain Comb as a multivariate-ML-only test against the univariate benchmark.

### 6.7 AR as Benchmark — Rationale

The AR model captures all information in univariate inflation history (momentum, mean-reversion, seasonality). Comparing ML models against AR reveals whether the **additional macro predictors and nonlinear methods** improve accuracy beyond what is already embedded in the inflation path. The Clark-West test (Section 8.3) adjusts for the finite-sample upward bias ML models incur from estimating extra parameters under the null.

---

## 7. Training & Validation

### 7.1 Rolling Window Protocol

- **Window size**: 96 months (8 years) — fixed, not expanding
- **Step**: Rolls forward one month at a time
- **OOS start**: January 2008
- **Horizons**: h = 1, 3, 5 months ahead

A fixed window adapts to regime changes better than an expanding window, which would weight distant past observations equally with recent ones.

For h > 1, training labels are shifted back by h periods to ensure proper temporal alignment: the model never sees future data.

### 7.2 Time Series Cross-Validation

Hyperparameter tuning uses **TimeSeriesSplit** with 3 splits and `max_train_size=96`:

```
Fold 1: train [t₁ … tₖ]           → validate [t_{k+1} … t_{2k}]
Fold 2: train [t₁ … t_{2k}]        → validate [t_{2k+1} … t_{3k}]
Fold 3: train [t₁ … t_{3k}]        → validate [t_{3k+1} … t_{4k}]
```

**Why not standard k-fold?** Random assignment would leak future information into the training set — a critical error in time-series settings.

**Why `max_train_size=96`?** Matches the rolling window size; prevents CV from favoring hyperparameters that perform well on longer training windows than the actual evaluation uses.

**Why 3 splits?** With 96-month capped windows, 3 splits produce validation sets of ≈32 months each (~2.7 years), sufficient for reliable MSE estimates.

### 7.3 Standardization

LASSO receives `StandardScaler`-transformed features (zero mean, unit variance). The scaler is fit only on **each training window** separately. Tree-based models (GBDT, RF) do not require scaling.

### 7.4 Reproducibility

All stochastic processes use `random_state=42`.

---

## 8. Evaluation Metrics

### 8.1 Root Mean Squared Error (RMSE)

```
RMSE = sqrt( mean( (y_true - y_pred)² ) )
```

Primary metric. Same units as target (percentage points). Used for level comparisons across models.

### 8.2 Out-of-Sample R-squared (R²_OOS)

```
R²_OOS = 1 - MSE_model / MSE_AR
```

Proportional reduction in MSE relative to the AR benchmark. **Positive** values indicate the ML model outperforms AR. **Negative** values indicate the ML model is worse — typically from overfitting the additional parameters. This is the primary measure of whether macro predictors add value.

### 8.3 Clark-West (2007) Test

Standard Diebold-Mariano assumes non-nested models. Here, ML models are **nested** within AR (they include AR's lagged inflation features plus additional macro predictors). Under the null that extra predictors have zero coefficients, the ML model's MSE is biased upward due to noise from estimating irrelevant parameters.

The Clark-West correction:

```
f_t = (e_AR_t)² - (e_ML_t)² + (ŷ_AR_t - ŷ_ML_t)²
```

The last term adjusts for the upward bias. The test statistic:

```
CW = √P · f̄ / √σ̂²_f
```

- H₀: MSE(AR) ≤ MSE(ML) — ML does not outperform
- H₁: MSE(AR) > MSE(ML) — ML outperforms
- One-sided test; significance at p < 0.10 (standard in forecast evaluation literature)

### 8.4 Directional Accuracy

```
DA = mean( sign(y_true - y_{t-1}) == sign(y_pred - y_{t-1}) ) * 100
```

Proportion of correctly predicted inflation direction (rising vs falling MoM). Particularly relevant for policy decisions where the sign of the change matters.

### 8.5 Sub-Period Robustness

The OOS period (2008–2026) is split into three regimes to test for regime-dependent performance:

| Sub-Period | Label | Macro Context |
|------------|-------|---------------|
| 2008–2019 | Low Inflation | Great Moderation, post-GFC recovery |
| 2020–2021 | COVID Shock | Pandemic, supply-chain disruption |
| 2022–2026 | Post-COVID Surge | High inflation, monetary tightening |

A model that wins in 2008–2019 but loses in 2020–2021 is flagged as **"brittle regime-dependent"** — it fails under the very conditions where accurate forecasts are most valuable.

### 8.6 Rolling Diagnostic

12-month rolling RMSE for AR and Comb at h=3, plotted over the OOS timeline. Visualizes when ML models fail (lines converge) vs. succeed (lines diverge).

---

## 9. Interpretation Methods

### 9.1 Permutation Importance

For h=1 forecasts, variable importance is assessed via random permutation:

1. Record the baseline forecast from each ML model at each OOS step.
2. One predictor at a time, replace its value with a random draw from that predictor's training distribution.
3. Re-predict and compute the drop in R²_OOS caused by permuting that predictor.

A **larger drop** indicates a more influential predictor. Random permutation preserves the predictor's distribution, unlike zero-out methods which can produce out-of-distribution values.

Computed separately for GBDT, LASSO, and RF to check consistency across model families.

### 9.2 Top Predictors Identified

From the last evaluation run:

| Rank | GBDT | LASSO | RF |
|------|------|-------|----|
| 1 | M2SL | PCE | BAA |
| 2 | BAA | M2SL | M2SL |
| 3 | FEDFUNDS | A191RI1Q225SBEA | FEDFUNDS |
| 4 | TCU | INDPRO | TCU |
| 5 | INDPRO | UNRATE | PAYEMS |

**M2SL (Money Supply)** and **BAA (Corporate Bond Yield)** appear in the top-5 for all three models, suggesting monetary aggregates and credit conditions are the most consistently important predictors.

---

## 10. Software & Reproducibility

### 10.1 Environment

| Component | Version |
|-----------|---------|
| Python | 3.12.3 |
| numpy | 2.5.0 |
| pandas | 3.0.3 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| statsmodels | 0.14.6 |
| matplotlib | 3.11.0 |
| pandas-datareader | 0.11.1 |

Full pinned dependencies in `requirements.txt`.

### 10.2 Project Structure

```
inflation_prediction/
├── Data/                         # 7 local CSV input files
├── Output/                       # Figures + predictions CSV
├── docs/
│   └── METHODOLOGY.md            # Detailed methodology
├── us_inflation_forecast.py      # Single-file pipeline
├── requirements.txt              # Python dependencies
├── .gitignore
└── report.md                     # This document
```

### 10.3 Execution

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python us_inflation_forecast.py
```

Optional: Set `FRED_API_KEY` environment variable for live data fetching (10 of 17 predictors). Without it, the pipeline runs on the 7 local CSV files only.

Typical runtime: **~22 minutes** (hyperparameter grid search at each rolling window step, single-threaded scikit-learn).

---

## 11. Results Summary

### 11.1 Performance Tables

| Model | h=1 RMSE | h=1 R²_OOS | h=1 CW-p | h=3 RMSE | h=3 R²_OOS | h=3 CW-p | h=5 RMSE | h=5 R²_OOS | h=5 CW-p |
|-------|----------|------------|----------|----------|------------|----------|----------|------------|----------|
| AR | 0.2858 | — | — | 0.3406 | — | — | 0.3496 | — | — |
| GBDT | 0.3006 | −10.60% | 0.1006 | 0.3470 | −3.75% | **0.0155** | 0.3506 | −0.56% | **0.0024** |
| LASSO | 0.2986 | −9.17% | **0.0591** | 0.3166 | **+13.61%** | **0.0049** | 0.3178 | **+17.37%** | **0.0011** |
| RF | 0.2978 | −8.58% | **0.0098** | 0.3585 | −10.76% | **0.0066** | 0.4084 | −36.49% | **0.0079** |
| Comb | 0.2900 | −2.94% | **0.0424** | 0.3272 | **+7.72%** | **0.0061** | 0.3435 | **+3.43%** | **0.0020** |

### 11.2 Key Findings

1. **LASSO is the best-performing ML model**: Positive R²_OOS at h=3 (+13.6%) and h=5 (+17.4%), statistically significant by CW test (p < 0.01). At h=1, LASSO underperforms AR in raw RMSE but the CW test (p=0.059) suggests the gap is not statistically significant after adjusting for nested-model bias.

2. **No ML model beats AR at h=1** in raw RMSE. All four ML models produce higher RMSE than the simple autoregressive benchmark at the 1-month horizon. However, CW tests indicate these differences are not reliably distinguishable from noise at conventional levels for LASSO (p=0.059) and RF (p=0.010).

3. **Tree-based models (GBDT, RF) perform poorly**: Negative R²_OOS at all horizons except GBDT at h=5 (−0.56% is essentially a tie). RF degrades sharply at h=5 (−36.5% R²_OOS), suggesting overfitting at longer horizons.

4. **The combination forecast adds value**: Comb achieves positive R²_OOS at h=3 and h=5 despite its constituents (GBDT, LASSO, RF) having mixed individual performance. This confirms the diversification benefit of combining models from different families.

### 11.3 Regime Dependence

| Sub-Period | h=3 Winner | h=5 Winner |
|------------|-----------|-----------|
| 2008–2019 (Low Inflation) | Comb | AR |
| 2020–2021 (COVID Shock) | AR | AR |
| 2022–2026 (Post-COVID) | Comb | Comb |

**Flagged: Brittle regime-dependent model.** Comb only wins during the post-COVID surge (2022+). It loses during the COVID shock (2020–2021) at both horizons, and at h=5 it also loses during 2008–2019. This suggests the ML models are exploiting patterns specific to the current high-inflation regime that may not generalize to future crisis periods.

### 11.4 Final Verdict

**Statistically significant improvement over AR at the 10% level.** LASSO and Comb achieve positive R²_OOS at h=3 and h=5 with CW p-values < 0.01. At h=1, no ML model beats AR in raw RMSE, but the CW test indicates the difference is not statistically significant for LASSO (p=0.059).

**Economic significance**: At h=5, LASSO reduces RMSE by (0.3496 − 0.3178) × 100 = **3.2 basis points per forecast** relative to AR — a meaningful improvement for inflation targeting, where each basis point of CPI surprise moves bond markets.

---

## 12. Limitations

1. **Contemporaneous predictors**: The model uses same-period macro values as features. Many series (GDP, employment) are released with a lag, so `X_t` may not be fully known at forecast time `t`. This introduces a slight forward-looking bias shared with most academic inflation studies.

2. **No real-time vintages**: All data uses latest revised values. Real-time forecasting would use initial releases, which are noisier and often revised substantially — likely reducing all models' performance.

3. **Fixed 96-month window**: Regime changes (COVID, Great Recession) occur faster than 8 years. A shorter window might adapt more quickly. A longer window might provide more stable estimates. Robustness checks with 60- and 120-month windows are included but computationally expensive.

4. **Limited model diversity**: No neural networks (insufficient data), no Bayesian methods, no state-space models. The five-model set spans key dimensions but is not exhaustive.

5. **No hyperparameter retraining**: Grid search parameters are fixed a priori and not re-evaluated during the rolling window. Optimal hyperparameters may shift over regimes.

6. **Single FRED API dependency**: 10 of 17 predictors require live internet access to FRED. Offline mode uses only 7 local CSV files, reducing the predictor set and likely degrading performance.
