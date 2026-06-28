# Methodology — US Inflation Forecasting with Machine Learning

## 1. Data Structure

### 1.1 Data Sources

The model draws on **17 macroeconomic predictors** from two sources:

**Local CSV files (7 series):**

| Series | Description | Frequency | Source |
|--------|-------------|-----------|--------|
| CPIAUCSL | CPI for All Urban Consumers | Monthly | FRED |
| POILWTIUSDM | WTI Crude Oil Price | Monthly | FRED |
| PCE | Personal Consumption Expenditures | Monthly | FRED |
| PPIACO | Producer Price Index | Monthly | FRED |
| FRBATLWGTUMHWGO | Average Hourly Earnings (All Employees) | Monthly | FRED |
| CIS1020000000000I | Composite Index of Coincident Indicators | Quarterly | OECD/FRED |
| A191RI1Q225SBEA | Real GDP (% Change) | Quarterly | BEA/FRED |

**FRED API series (10 series, fetched live via `pandas-datareader`):**

| Series | Description | Category |
|--------|-------------|----------|
| INDPRO | Industrial Production Index | Real Activity |
| TCU | Capacity Utilization | Real Activity |
| UNRATE | Unemployment Rate | Labor Market |
| PAYEMS | Total Nonfarm Payrolls | Labor Market |
| SP500 | S&P 500 Index | Financial Markets |
| GS10 | 10-Year Treasury Yield | Financial Markets |
| BAA | Moody's Baa Bond Yield | Credit Markets |
| FEDFUNDS | Federal Funds Rate | Monetary Policy |
| M2SL | M2 Money Supply | Monetary Aggregates |
| TWEXBPA | Trade Weighted Dollar Index | External Sector |

### 1.2 Target Variable

The target is **month-over-month US CPI inflation**, computed as the percentage change in the Consumer Price Index (CPIAUCSL):

```
π_t = (CPI_t / CPI_{t-1} - 1) × 100
```

This is a **stationary transformation** of the CPI level, which is integrated of order 1 (I(1)). A unit-root test (e.g., ADF) on CPI levels would fail to reject non-stationarity; modeling an I(1) series directly would produce spurious regression results. The month-over-month percentage change renders the series stationary, making it suitable for linear models (AR, LASSO) without requiring additional differencing, and for tree-based models (GBDT, RF) which are theoretically scale-invariant but perform better on stationary inputs.

**Why month-over-month instead of year-over-year?** Year-over-year (YoY) inflation, computed as (CPI_t / CPI_{t-12} - 1) × 100, is commonly reported in policy discussions and media. However, YoY inflation is a **moving average of the previous 12 monthly changes**, which induces strong autocorrelation and smooths out turning points — a YoY series will continue rising for months after monthly inflation peaks. For forecasting purposes, the month-over-month rate is preferred because:
- It responds more quickly to regime changes, making forecast improvements detectable sooner in the OOS period.
- Its autocorrelation structure is simpler (decays after 3–6 lags), requiring fewer lagged features.
- The Clark-West test for nested model comparison has better size and power properties when the target is not excessively smoothed.

The first observation of each series is lost after computing the lagged difference, and the preprocessing pipeline drops the raw CPIAUCSL level before training to avoid redundancy with the 6 lagged inflation features (the level is a near-perfect linear combination of lagged changes plus the initial value).

### 1.3 Final Dataset

After preprocessing, the final dataset typically contains **~700 monthly observations** (spanning ~1967 to present) with **13 predictors** (4 of the original 17 are dropped during cleaning). The out-of-sample evaluation period begins in **January 2008**, providing roughly 18+ years of test data.

---

## 2. Data Preprocessing

### 2.1 Date Alignment

All series are shifted to **end-of-month** dates to ensure a uniform time index. Duplicate dates within a series are resolved by keeping the last observation. This step is critical because the raw data arrives at different points within each month (e.g., daily S&P 500 data vs. monthly CPI releases).

### 2.2 Mixed-Frequency Interpolation

The dataset contains both **monthly** and **quarterly** series (Real GDP and the Coincident Indicators Index). For each series, the median gap between consecutive observations is computed. If the median gap exceeds 80 days (indicating quarterly frequency, ~91 days), the series is interpolated to monthly frequency using **forward-fill** (`ffill`). This avoids leaking future information into the training window, which would occur with spline or other interpolation methods that use surrounding values.

### 2.3 Target Computation

CPI inflation is computed after merging all predictors, ensuring that the target aligns with the same monthly frequency. The first observation is lost due to the lagged difference.

### 2.4 Short-Gap Interpolation

Remaining missing values within each predictor series are filled using forward-fill with a **limit of 6 months**. This prevents discarding otherwise useful observations while avoiding extrapolation into long data-free regions and prevents future information leakage that spline-based interpolation would introduce.

### 2.5 Sparse Column Removal

Predictors with **less than 50% non-NaN values** across the full date range are dropped. These are typically series with short historical coverage that would excessively truncate the sample period.

### 2.6 Common Sample Period Determination

The model automatically determines the estimation window by finding the **earliest month where at least 80% of the surviving predictors have non-missing values** and the inflation target is also available. This trade-off maximizes the number of predictors while preserving a long historical sample. After trimming, any remaining isolated NaN values are forward-filled and rows with any remaining missing values are dropped.

---

## 3. Training

### 3.1 Rolling Window Protocol

The model uses a **rolling window** (not expanding window) evaluation scheme:

- **Window size:** 96 months (8 years)
- **Out-of-sample start:** January 2008
- **Forecast horizons:** h = 1, 3, and 5 months ahead
- **Step:** The window rolls forward one month at a time; at each step `t`, the model is trained on the preceding 96 months and evaluates on the observation at `t + h`

This design reflects a realistic forecasting environment where the model has a fixed-length recent history and must generalize forward. A fixed window also adapts more readily to regime changes than an expanding window, which would weight distant past observations equally with recent ones.

For horizons longer than 1 month (h > 1), the training labels are shifted back by `h` periods to ensure proper temporal alignment: the model never sees future data. Specifically, for horizon `h`, the training set ends at `t + 1 - h` rather than `t`, so the last training label corresponds to `t` and the test label to `t + h`.

### 3.2 Feature Engineering

Each model receives a feature vector constructed as:

1. **6 lags of CPI inflation** (π_{t-1}, π_{t-2}, …, π_{t-6})
2. **All macro predictor values** at time `t` (typically 13 predictors after preprocessing)

**Note on CPIAUCSL.** The raw CPI level (CPIAUCSL) is intentionally excluded from the predictor set. Since the target is the percentage change of CPIAUCSL and 6 lags of this change are included, the level is nearly a perfect linear combination of the lagged changes plus the initial value — adding it would introduce near-perfect multicollinearity for linear models and provide no additional information for tree-based models. The predictor set includes only the other 16 macro series (plus 6 lags = ~22 raw features, cleaned to ~19 after preprocessing).

The 6-lag structure captures the autoregressive dynamics of inflation — momentum, mean-reversion, and seasonal patterns — while the contemporaneous predictors allow the model to incorporate external macro conditions. This gives a feature space of **~19 dimensions** (6 lags + 13 predictors after cleaning).

### 3.3 Model Architectures

Five models are estimated at each rolling window step — a univariate benchmark, three machine learning models spanning different inductive biases, and a simple combination.

---

**AR (Autoregressive Benchmark)**

**Algorithm.** The autoregressive model of order `p` expresses the current inflation value as a linear combination of `p` past values plus white noise:

```
y_t = c + φ₁·y_{t-1} + φ₂·y_{t-2} + ... + φ_p·y_{t-p} + ε_t,    ε_t ~ WN(0, σ²)
```

This is equivalent to a linear regression of `y_t` on its own lags. The parameters φ = (φ₁, ..., φ_p) are estimated by ordinary least squares (OLS), minimizing Σ(y_t - c - Σφᵢy_{t-i})². For horizons `h > 1`, forecasts are produced **iteratively**: each step's forecast is fed back as a lagged input for the next step, so that ŷ_{t+h|t} = ĉ + φ̂₁·ŷ_{t+h-1|t} + ... + φ̂_p·ŷ_{t+h-p|t}.

**Lag-order selection.** At each rolling window step, `p` is chosen from {1, 2, ..., 12} by minimizing the Akaike Information Criterion:

```
AIC(p) = 2p - 2·ln(L̂)
```

where L̂ is the maximized likelihood of the model (equivalently, AIC = n·ln(RSS/n) + 2p for OLS). AIC penalizes additional parameters to prevent overfitting while allowing the data to determine the optimal lag length at each window. The lag can change over time, adapting to shifts in the persistence of inflation.

**Why AR as benchmark.** Inflation is known to be highly persistent — a high-inflation month tends to be followed by another. The AR model captures all information in the univariate inflation history (momentum, mean-reversion, seasonal patterns) and thus represents the **best purely univariate forecast**. Comparing ML models against this baseline reveals whether the 17 additional macro predictors and nonlinear methods improve forecasting accuracy beyond what is already embedded in the inflation path.

---

**GBDT (Gradient Boosted Decision Trees)**

**Algorithm — additive expansion in function space.** GBDT builds an ensemble of shallow regression trees sequentially, where each new tree fits the **gradient of the loss function** with respect to the current ensemble's predictions — i.e., each tree is trained on the pseudo-residuals of the preceding model.

Let the ensemble after `M` iterations be an additive expansion:

```
F_M(x) = Σ_{m=1}^{M} γ_m · h_m(x; a_m)
```

where `h_m(x; a_m)` is a regression tree with parameters `a_m` (split variables, split thresholds, leaf values) and `γ_m` is its step-size weight. The algorithm:

1. **Initialize** with a constant: F₀(x) = argmin_γ Σ L(y_i, γ), where L is squared error loss.
2. For each iteration `m = 1, ..., M`:
   - Compute **pseudo-residuals**: For each training observation `i`, evaluate the negative gradient of the loss at the current fit:
     ```
     r_{im} = -[∂L(y_i, F(x_i))/∂F(x_i)]_{F = F_{m-1}}
     ```
     For squared error loss, L(y, F) = ½(y - F)², so `r_{im} = y_i - F_{m-1}(x_i)` — the residuals.
   - Fit a regression tree `h_m(x; a_m)` to the pseudo-residuals: the tree partitions the feature space into `J_m` disjoint regions (leaf nodes) and assigns a constant prediction to each region.
   - **Line search**: Find the optimal step size for each leaf region:
     ```
     γ_{jm} = argmin_γ Σ_{x_i ∈ R_{jm}} L(y_i, F_{m-1}(x_i) + γ)
     ```
     For squared error, this is simply the average of the residuals in that leaf.
   - **Shrinkage update**:
     ```
     F_m(x) = F_{m-1}(x) + ν · Σ_{j=1}^{J_m} γ_{jm} · 1(x ∈ R_{jm})
     ```
     where `ν` (learning rate) is a shrinkage factor ∈ (0, 1] that scales each tree's contribution.

3. Output: F_M(x) as the final prediction function.

**How a decision tree works internally.** A regression tree recursively partitions the feature space. At each node, it searches over all features and all possible split thresholds to find the split that maximizes the **variance reduction** (Friedman's MSE criterion): the weighted average of the MSE in the left and right child nodes, minus the MSE in the parent node. The process stops at `max_depth`, producing leaf nodes whose values are the mean of the target in that region. A depth-3 tree produces at most 2³ = 8 leaf regions, capturing low-order interactions among at most 3 features.

**Stochastic gradient boosting (subsampling).** At each iteration, a fraction `subsample` of the training data is drawn without replacement, and the tree is fitted to this random subset. This introduces randomness that decorrelates the trees, reducing overfitting — the same principle that makes Random Forest effective, applied to boosting.

**Hyperparameter tuning rationale.**

| Parameter | Values Tuned | Purpose |
|-----------|-------------|---------|
| `n_estimators` (M) | {80, 100, 120} | Ensemble size; higher M increases capacity but risks overfitting |
| `max_depth` | {2, 3, 4} | Limits tree complexity; depth ≤ 4 forces focus on simple, stable interactions |
| `learning_rate` (ν) | {0.1, 0.01} | Shrinkage; lower ν requires more trees but generalizes better |
| `subsample` | {0.8, 1.0} | Stochastic component; 0.8 adds regularization via randomness |

Shallow trees (max_depth ∈ {2,3,4}) are a deliberate design choice. Monthly macro data is noisy and has limited signal per observation; deep trees would memorize transient patterns. By constraining depth, GBDT focuses on **stable, low-order interactions** — e.g., the joint effect of oil prices and unemployment on inflation — that generalize out-of-sample.

**Why GBDT for inflation.** Inflation is driven by complex, nonlinear interactions among predictors: the effect of money supply depends on the output gap, the pass-through of oil prices depends on the exchange rate regime, etc. GBDT captures these interactions automatically without requiring the modeler to specify them ex ante. The grid-searched hyperparameters ensure the model finds the right tradeoff between fitting these interactions and generalizing to new regimes.

---

**LASSO (L1-Regularized Linear Regression)**

**Algorithm — penalized least squares with feature selection.** LASSO (Least Absolute Shrinkage and Selection Operator) solves:

```
β̂(λ) = argmin_β ½n · Σ_{i=1}^{n} (y_i - β₀ - Σⱼβⱼx_{ij})² + λ · Σⱼ|βⱼ|
```

The L1 penalty Σ|βⱼ| has two critical properties:
- **Shrinkage**: It continuously shrinks coefficients toward zero as λ increases, reducing variance.
- **Selection**: For sufficiently large λ, coefficients become exactly zero, effectively removing those predictors from the model. This is a key difference from ridge regression (L2 penalty), which shrinks but never zeroes coefficients.

The regularization path is solved by **coordinate descent** (the algorithm used in sklearn's `LassoCV`). Coordinate descent cycles through each coefficient and updates it to the minimizer of the penalized loss while holding all other coefficients fixed. For LASSO, each coordinate update has a closed-form solution via the soft-thresholding operator:

```
βⱼ ← S(ρⱼ, λ) / zⱼ
```

where ρⱼ = Σᵢ x_{ij}(y_i - β₀ - Σ_{k≠j}βₖx_{ik}) is the partial residual, zⱼ = Σᵢ x_{ij}², and S(a, b) = sign(a)(|a| - b)_+ is the soft-threshold function. The path of solutions for all λ values is computed efficiently by warm-starting from the previous λ's solution.

**Standardization.** Because the L1 penalty is scale-dependent (Σ|βⱼ| would penalize large-coefficient predictors more if their scale is small), all predictors are standardized to zero mean and unit variance before estimation:

```
x_{ij}^* = (x_{ij} - x̄ⱼ) / σⱼ
```

This ensures the penalty applies uniformly across all features regardless of their native units (CPI basis points vs. interest rate percentages, etc.).

**Alpha selection.** The regularization strength α (equivalent to λ in the formulation above) is selected from {0.001, 0.01, 0.1, 1} using cross-validated LassoCV with 3-fold TimeSeriesSplit. Each candidate α is evaluated on a held-out validation fold, and the α minimizing the CV MSE is chosen. The grid spans four orders of magnitude to cover the range from near-OLS (α = 0.001) to heavy shrinkage (α = 1).

**Interpretation of α.** When α = 0.001, the penalty is negligible — LASSO approximates OLS, and all ~19 features remain in the model. As α increases, the budget Σ|βⱼ| shrinks, and LASSO must decide which predictors earn a nonzero coefficient. The CV procedure picks the α that optimally balances bias (from shrinkage) and variance (from estimating many noisy coefficients).

**Why LASSO for inflation.** Inflation forecasting is a classic "medium-p, medium-n" problem: 19 features from a 96-month window. Not all 17 macro predictors are likely to be relevant at all times. LASSO's automatic feature selection identifies which predictors matter in each regime — oil prices may dominate in supply-shock periods, while monetary aggregates matter during tightening cycles. The interpretable coefficients also allow us to examine whether the sign and magnitude of each predictor's effect align with economic theory.

---

**RF (Random Forest)**

**Algorithm — bagged decorrelated trees.** Random Forest constructs an ensemble of `B` regression trees, each grown on a different bootstrap sample of the training data, with an additional randomization step at each split.

1. For `b = 1, ..., B`:
   - Draw a bootstrap sample of size `n` (sampling with replacement) from the training data. This sample contains roughly 63.2% of the unique observations; the remaining 36.8% form the **out-of-bag (OOB)** sample.
   - Grow a regression tree `T_b` on the bootstrap sample:
     - At each node, randomly select `m_try` predictors from the full set of `p` features (for regression, the default is `m_try = p/3`).
     - Among these `m_try` features, choose the split that maximizes the variance reduction (MSE decrease).
     - Continue splitting until the tree reaches `max_depth` or a minimum node size is reached.
   - Do not prune the tree.

2. **Ensemble prediction** (for regression): average the predictions of all B trees:
   ```
   f̂_RF(x) = (1/B) · Σ_{b=1}^{B} T_b(x)
   ```

**Why it works — bias-variance decomposition.** A single deep tree has **low bias but high variance**: it fits the training data perfectly but changes substantially if the training data is perturbed slightly. Random Forest reduces variance through:

- **Bagging**: Averaging over B bootstrap trees reduces variance by a factor of ~1/B (if trees were independent), but bootstrap trees are correlated because they share the original data distribution. The variance reduction is approximately ρ·σ² + (1-ρ)·σ²/B, where ρ is the pairwise tree correlation. The `m_try` randomization reduces ρ further.
- **Random feature selection**: By restricting each split to a random subset of features, the trees become decorrelated — different trees focus on different predictors. This reduction in ρ is the key improvement of RF over simple bagging.

The result is a model that maintains the flexibility to capture nonlinear patterns and interactions (low bias) while achieving smooth, stable predictions (low variance through averaging).

**Out-of-bag (OOB) error.** Each tree `T_b` is trained on a bootstrap sample, leaving ~37% of observations unused. These OOB observations can be used as an internal validation set: for each observation `i`, predict using only the trees where `i` was OOB, and compute the MSE. The OOB error is a nearly unbiased estimate of the generalization error and correlates strongly with CV estimates.

**Hyperparameter tuning rationale.**

| Parameter | Values Tuned | Purpose |
|-----------|-------------|---------|
| `n_estimators` (B) | {100, 150} | Ensemble size; more trees reduce variance but yield diminishing returns |
| `max_depth` | {4, 6, 8} | Controls tree complexity; deeper trees capture finer interactions |

Deeper trees (max_depth ∈ {4,6,8}) are allowed compared to GBDT (max_depth ∈ {2,3,4}) because bagging provides **built-in regularization** — averaging many deep trees is less prone to overfitting than boosting deep trees sequentially. The depth cap still prevents trees from memorizing idiosyncratic patterns in noisy monthly data.

**Why RF for inflation.** RF complements GBDT in two ways. First, RF is **robust to outliers**: a single extreme observation can distort early boosting iterations (which weight all points equally), while in RF it influences only the trees that contain it in their bootstrap sample. Second, RF provides **variance-stabilized predictions** through averaging, which is valuable when the signal-to-noise ratio in monthly inflation is low. If GBDT and RF agree on a forecast, confidence is higher; if they diverge, the combination forecast (Comb) smooths the difference.

---

**Comb (Combination Forecast)**

**Algorithm.** The combination forecast is a simple, equal-weighted average:

```
ŷ_Comb = (ŷ_GBDT + ŷ_LASSO + ŷ_RF) / 3
```

No training or optimization is involved — the weights (1/3 each) are fixed a priori.

**Why it works — the forecast combination puzzle.** In the forecast combination literature, the "combination puzzle" (Stock & Watson, 2004; Timmermann, 2006) refers to the finding that simple averages of forecasts **consistently match or outperform** more sophisticated combination schemes (variance-weighted, MSE-weighted, or regression-based combinations). This counterintuitive result has several explanations:

1. **Variance reduction**: Each individual model has a forecast error variance. Averaging across models reduces variance by a factor that depends on the pairwise correlation of the forecast errors. With three models from different families (linear-regularized, sequential-ensemble, parallel-ensemble), error correlations are modest, yielding meaningful variance reduction.

2. **Diversification of biases**: AR under-fits (omits macro predictors). LASSO may over-shrink relevant predictors. GBDT may capture spurious interactions. RF may oversmooth. Averaging cancels these directional biases.

3. **No estimation risk**: Estimating combination weights from time-series data would require a holdout period, reducing the effective sample, and the estimated weights themselves would be noisy. Equal weighting avoids this entirely.

4. **Empirical precedent**: Clemen (1989) surveyed 30 years of combination studies and found that simple averages are rarely outperformed. Stock & Watson (2004) showed this holds specifically for macroeconomic forecasting.

**Rationale for excluding AR from the combination.** AR is intentionally excluded from the Comb ensemble. Including it would dilute the contribution of macro predictors, since the AR forecast uses only lagged inflation. The combination is designed to test whether the **multivariate ML forecasts** improve upon the univariate benchmark when pooled — not to create a "best possible" ad-hoc forecast. AR remains the benchmark, and Comb is compared against AR separately via the Clark-West test.

### 3.4 Cross-Validation Strategy

All hyperparameter tuning uses **TimeSeriesSplit** with 3 splits and `max_train_size=96`, which respects the temporal ordering of the data by training on earlier observations and validating on later ones. The `max_train_size=96` constraint ensures that each training fold uses at most 96 months (matching the rolling window size), preventing models trained on very long early folds from biasing the CV estimate. This contrasts with standard k-fold cross-validation, which would randomly shuffle observations and leak future information into the training set — a critical error in time-series settings.

---

## 4. Decision Making

### 4.1 Model Selection Rationale

The five models are chosen to span the key dimensions of the model-selection space for macroeconomic forecasting: **linear vs. nonlinear**, **univariate vs. multivariate**, **regularized vs. unregularized**, **parametric vs. nonparametric**, and **single model vs. ensemble**. Each model is a representative of its class, not an arbitrary choice.

**Coverage of the model space:**

| Dimension | AR | LASSO | GBDT | RF | Comb |
|-----------|----|-------|------|-----|------|
| Linear / Nonlinear | Linear | Linear | Nonlinear | Nonlinear | Mixed |
| Univariate / Multivariate | Univariate | Multivariate | Multivariate | Multivariate | Multivariate |
| Regularization | Lag-selection (AIC) | L1 penalty | Depth + shrinkage + subsample | Depth + bagging | N/A (fixed weights) |
| Parametric / Nonparametric | Parametric | Parametric | Nonparametric | Nonparametric | Mixed |
| Interpretability | High (coefficients) | High (sparse coefficients) | Low (black-box) | Low (black-box) | N/A |
| Variance reduction | Via lag selection | Via shrinkage | Via sequential averaging | Via bootstrap averaging | Via cross-model averaging |

**Why these five and not others.**

- **AR(\(p\)) is the universal benchmark.** Inflation forecasting has a 50+ year literature; virtually every paper includes an AR benchmark. Without this baseline, it is impossible to determine whether ML models add value or simply rediscover patterns already in the inflation history. The AIC-selected lag ensures the benchmark is itself well-specified — not arbitrarily fixed to \(p=1\) or \(p=12\).

- **LASSO was chosen over ridge, elastic net, or OLS.** Ridge regression (L2 penalty) shrinks coefficients but never zeroes them — all 19 features remain in every model, which contradicts the assumption that only a subset matter. Elastic net adds an L2 term that can improve prediction when predictors are grouped, but LASSO alone is preferred here for maximal sparsity and interpretability. OLS with 19 features from 96 observations would be hopelessly overfit (variance inflation from near-multicollinearity in macro data).

- **GBDT was chosen over AdaBoost or XGBoost/LightGBM.** Standard GBDT (Friedman, 2001) with sklearn's implementation is the canonical gradient boosting formulation. XGBoost and LightGBM add optimizations (regularized loss, histogram-based splitting) that improve speed but introduce additional hyperparameters (L1/L2 on leaf weights, min_child_weight, etc.) that add complexity without clear benefit at this dataset size (~700 obs × 19 features). The grid search over 3 splits × 3 depths × 2 learning rates × 2 subsample rates = 36 configurations is already computationally feasible and covers the relevant tuning range.

- **RF was chosen over bagged trees or Extra Trees.** Standard bagged trees (no `m_try` randomization) would produce highly correlated trees, reducing the variance-reduction benefit of averaging. Extra Trees randomizes both the split threshold and the feature subset, adding further randomness, but at this scale the standard RF with `m_try = p/3` provides sufficient decorrelation. The depth cap {4, 6, 8} is lower than typical RF settings (often full-depth trees) because monthly macro data is far noisier than the image datasets RF was designed for.

- **Comb is not an afterthought — it is a distinct modeling strategy.** Forecast combination is a well-established approach in macroeconomics (Bates & Granger, 1969; Clemen, 1989; Stock & Watson, 2004). Including Comb as its own model (rather than just a footnote) allows it to be formally evaluated via RMSE and the Clark-West test against the AR benchmark, just like the other models.

**What is deliberately excluded.**

- **Neural networks** (MLPs, LSTMs, Transformers). Deep learning methods require substantially more data (both in cross-section and time-series length) than the ~700 monthly observations available here. With only 19 features, a neural network's capacity would be vastly underutilized, and the risk of overfitting to noise in monthly macro data is high. The literature on neural networks for macroeconomic forecasting (e.g., Makridakis et al., 2020, M4/M5 competitions) finds that simpler methods match or outperform deep learning on monthly economic data.

- **Support vector regression / Gaussian processes.** These kernel methods scale poorly with sample size (O(n³) for exact GPs) and do not offer clear advantages over tree-based ensembles for tabular macro data.

- **State-space models / Dynamic factor models (DFMs).** DFMs are a strong alternative that explicitly models the latent factor structure of many macro series. They are excluded here because the focus is on supervised ML methods aligned with the reference methodology paper. A DFM extension would be a natural future addition.

This coverage ensures the final results are robust across modeling paradigms — if LASSO, GBDT, and RF all show similar findings, the conclusions are far more credible than if they came from a single model class.

### 4.2 Feature Design: Lags vs. Contemporaneous Predictors

The feature vector for each ML model consists of two components: (i) 6 lags of CPI inflation and (ii) all contemporaneous (same-period) macro predictor values. This design embeds a specific assumption about the data-generating process.

**Why include lagged inflation as features?** Inflation is strongly autocorrelated — the current month's inflation is the single best predictor of next month's inflation. By including lags, the ML models can learn the same autoregressive dynamics that the AR benchmark captures, augmented with macro predictors. If we omitted the lags, any improvement over AR could be trivially attributed to the missing autoregressive component rather than to the macro predictors. Including lags ensures the comparison is **incremental**: the ML models must justify their macro predictors on top of the same baseline information.

**Why 6 lags specifically?** The choice balances:

1. **Autocorrelation structure of US inflation.** Monthly CPI inflation in the US typically shows statistically significant autocorrelation out to 3–6 months, with a gradual decay. A lag length of 6 captures this full decay horizon. Information beyond 6 months is generally negligible (the partial autocorrelation function cuts off after 2–4 months in most samples).

2. **Seasonal patterns.** Some seasonal effects in CPI have a semi-annual component (e.g., seasonal clothing/apparel adjustments in spring and fall, energy demand cycles). Six lags span a half-year window that can capture these.

3. **Degrees of freedom.** With 13 predictors after cleaning, adding 6 lags yields 19 features from 96 training observations — an observations-to-features ratio of approximately 5:1. This is marginal for OLS but comfortable for LASSO (which selects a subset) and tree-based models (which use axis-aligned splits and are not as susceptible to the curse of dimensionality). More lags (e.g., 12) would yield 25 features and a ratio of ~3.8:1, which would degrade LASSO's variable selection reliability and make tree splits sparser.

4. **Consistency with the reference paper.** The methodology paper on which this pipeline is based ("Forecasting China's inflation rate: Evidence from machine learning methods") uses 6 lags. Keeping the same design allows direct comparability.

**Why contemporaneous (same-period) predictors rather than lagged predictors?** A natural concern is look-ahead bias: using values from time `t` to forecast inflation at time `t+h`. However, many macro series are released with a lag (CPI itself is released mid-month for the previous month), and the pipeline aligns all series to end-of-month dates. The CPI release schedule means that at the time the model makes a forecast, the most recent CPI value (and thus the latest inflation reading) is already available. The same holds for many FRED series — industrial production, unemployment, etc. — which are released within weeks of the reference month. For series with longer publication lags (GDP, some survey measures), the contemporaneous value may not be known in real time. This is a limitation shared with most academic inflation forecasting studies, which routinely use `t`-dated predictors and note the slight forward-looking bias.

A robustness test using only lagged predictors (shifting all predictors back by one month) would be a useful future extension to quantify this bias.

### 4.3 Why 96-Month (8-Year) Window

The 96-month window is a deliberate compromise between two competing demands:

- **Statistical precision.** Parameter estimation in a linear model with 19 features requires a minimum of ~5 observations per feature for reliable estimates; 96 provides ~5:1. For tree-based models, more observations allow deeper splits without overfitting. The window must be long enough to span at least one full business cycle (US business cycles averaged 5.7 years from 1945 to 2009, per NBER) so the model trains on both expansions and contractions.

- **Regime adaptation.** The US economy has undergone several distinct inflation regimes since 1960: the low-inflation 1960s, the Great Inflation (1965–1982), the Great Moderation (1984–2007), the post-GFC lowflation (2008–2020), and the post-COVID surge (2021–2023). An expanding window would weight the high-inflation 1970s equally with the low-inflation 2010s, potentially producing a model that is good for neither regime. A fixed 96-month window naturally drops the most distant observations, allowing the model to specialize to recent conditions.

Robustness checks with 60-month (5-year) and 120-month (10-year) windows confirm whether results depend on this specific window length. The 60-month window tests whether shorter windows (trading statistical precision for faster regime adaptation) change conclusions. The 120-month window tests whether longer windows (trading some regime specificity for more stable estimates) improve or degrade forecasts. If RMSE rankings are consistent across all three window sizes, the findings are robust to this design choice.

### 4.4 Why AR as Benchmark

The AR model (with AIC-selected lag) represents the **best purely univariate forecast** — it captures all information contained in the inflation series itself without external predictors. Comparing ML models against this benchmark reveals whether the additional macro predictors and nonlinear methods add genuine forecasting value beyond what is already embedded in the inflation history.

### 4.5 Why TimeSeriesSplit (Not Standard k-Fold CV)

Standard k-fold cross-validation randomly assigns observations to folds. For time-series data, this creates a critical problem: the training set at fold `k` contains observations from time `t₁, t₂, ..., t_p`, and the validation set contains observations at time `t_{p+1}`. But because the assignment is random, the validation set at one fold may include dates before some training observations in another fold. In effect, the model "sees the future" during training — a gross violation of the temporal ordering that produces artificially optimistic performance estimates (Bergmeir & Benítez, 2012).

TimeSeriesSplit avoids this by constructing training and validation indices that strictly respect time order:

```
Fold 1: train [t₁, t₂, ..., tₖ]            → validate [t_{k+1}, ..., t_{2k}]
Fold 2: train [t₁, t₂, ..., t_{2k}]          → validate [t_{2k+1}, ..., t_{3k}]
Fold 3: train [t₁, t₂, ..., t_{3k}]          → validate [t_{3k+1}, ..., t_{4k}]
```

At each fold, the training set is a prefix of the full time series and the validation set immediately follows.

**Why `max_train_size=96`?** By default, TimeSeriesSplit uses an expanding window — the training set grows with each fold. This means Fold 3 would train on 3× more data than Fold 1. Since the rolling-window evaluation itself uses a fixed 96-month window, the CV procedure must match: `max_train_size=96` caps each training fold at 96 months, making the CV folds resemble the actual evaluation conditions. Without this cap, the CV would favor hyperparameters that perform well on very long training windows, potentially mis-selecting parameters for the 96-month rolling window.

**Why 3 splits instead of 5 or 10?** With `max_train_size=96`, 3 splits produce validation sets of approximately 32 months each (~2.7 years), providing enough observations for reliable MSE estimates. More splits would produce smaller validation sets (e.g., 5 splits → ~19 months per validation set), increasing the variance of the MSE estimate. Fewer splits would waste data. Given the practical constraint that the full dataset has ~700 observations and the window is 96 months, 3 splits is a pragmatic choice.

### 4.6 Why Equal-Weighted Combination

The combination forecast equally averages GBDT, LASSO, and RF. Research in forecast combination (Clemen, 1989; Stock & Watson, 2004) consistently finds that simple averages match or exceed more sophisticated weighting schemes, especially when individual models have comparable performance. Equal weights also avoid the overfitting risk inherent in learning combination weights from a time series.

### 4.7 Why the Clark-West Test (Not Diebold-Mariano)

A standard approach for comparing forecast accuracy is the Diebold-Mariano (DM) test, which tests whether two forecasts have equal expected loss. However, the DM test assumes the models are **non-nested** — neither model's predictor set is a subset of the other's. In this pipeline, the ML models are nested: they include the AR model's 6 lagged inflation values (plus the contemporaneous macro predictors). Under the null hypothesis that the extra predictors have zero coefficients, the ML model's MSE is expected to be **higher** than the AR model's MSE due to the estimation of additional irrelevant parameters (a finite-sample upward bias).

The Clark-West (2007) test directly addresses this problem. The test statistic includes a correction term:

```
f_t = (e_AR_t)² - (e_ML_t)² + (ŷ_AR_t - ŷ_ML_t)²
```

The last term `(ŷ_AR_t - ŷ_ML_t)²` adjusts for the upward bias in the ML model's squared error. Intuitively, when the null is true, the ML model's predictions differ from the AR model's predictions only because of noise in the additional parameter estimates; this noise inflates the ML model's MSE. The correction term subtracts this noise component. Without this adjustment, the DM test would be undersized — it would fail to reject the null even when the ML model genuinely outperforms, because the bias masks the improvement.

**Interpretation.** A significant Clark-West statistic (p < 0.10) is interpreted as evidence that the ML model's additional predictors and/or nonlinear structure add statistically significant predictive power beyond the univariate inflation history. The test is one-sided (H₁: MSE(AR) > MSE(ML)), which is appropriate because we care only about whether the ML model improves upon AR, not whether it performs worse.

---

## 5. Evaluation

### 5.1 Metrics

Three metrics are computed for each model at each horizon:

**RMSE (Root Mean Squared Error):**

```
RMSE = sqrt(mean((y_t - ŷ_t)²))
```

Measures the average magnitude of forecast errors in the same units as inflation (percentage points). Lower is better. RMSE is reported primarily for levels comparison (Table 1), while R²_OOS is the preferred metric for relative comparison because it normalizes by the benchmark's difficulty.

**R²_OOS (Out-of-Sample R-squared):**

```
R²_OOS = 1 - MSE_model / MSE_AR
```

Measures the proportional reduction in mean squared error relative to the AR benchmark. A positive value means the ML model outperforms the simple autoregressive model; a value of 0.10 means a 10% improvement. This is the primary measure of whether macro predictors and ML methods add value. Unlike in-sample R², R²_OOS can be negative — this occurs when the ML model's forecast is worse than the AR benchmark, typically due to overfitting the additional parameters.

**Clark-West (2007) Test:**

The Clark-West test addresses a specific statistical problem: when comparing nested models (the ML models include AR's lagged inflation as a subset of features), the ML model's MSE is biased upward under the null hypothesis because the extra parameters are estimated with noise. The test adjusts for this bias.

The one-sided test setup:
- H₀: MSE(AR) ≤ MSE(ML) — the ML model does not outperform AR
- H₁: MSE(AR) > MSE(ML) — the ML model outperforms AR

For each out-of-sample observation `t`, compute the adjusted loss differential:

```
f_t = (e_AR_t)² - (e_ML_t)² + (ŷ_AR_t - ŷ_ML_t)²
```

where `e_AR_t = y_t - ŷ_AR_t` is the AR forecast error, `e_ML_t = y_t - ŷ_ML_t` is the ML forecast error, and `ŷ_AR_t - ŷ_ML_t` is the difference in point forecasts. The `(ŷ_AR_t - ŷ_ML_t)²` term is the bias correction: under H₀, the ML model's extra parameters introduce noise into ŷ_ML_t, inflating its squared error; the correction term subtracts this noise component.

The test statistic aggregates across all `P` out-of-sample observations:

```
CW_stat = √P × f̄ / √σ̂²_f
```

where `f̄` is the sample mean of `f_t` and `σ̂²_f` is the sample variance (with Bessel correction). Under H₀, `CW_stat` is asymptotically standard normal. The p-value is computed as `P(CW_stat > z) = 1 - Φ(CW_stat)`, where Φ is the standard normal CDF. A significant result (p < 0.10) indicates that the ML model's improvement is statistically meaningful — the additional predictors and/or nonlinear structure genuinely improve forecast accuracy beyond what sampling noise can explain.

**Why p < 0.10 rather than 0.05?** The Clark-West test for nested comparison at typical sample sizes (P ≈ 200–300 OOS observations) has modest power. Using a 10% significance threshold is standard in the forecast evaluation literature (e.g., Stock & Watson, 2004; Clark & West, 2007) to balance Type I and Type II error given the inherent difficulty of detecting improvements in noisy macroeconomic data.

### 5.2 Variable Importance

For the h=1 forecasts, variable importance is assessed using a **random permutation method**:
1. For each out-of-sample prediction, record the baseline forecast from each ML model.
2. One predictor at a time, randomly replace its value with a draw from that predictor's training distribution (using `numpy.random.choice` on the training window values) and re-predict.
3. Compute the drop in R²_OOS caused by permuting that predictor.

A larger drop indicates a more influential predictor. This method is model-agnostic and provides interpretable importance scores even for black-box models like GBDT and RF. Unlike zeroing out, random permutation preserves the predictor's distributional properties and better reflects the impact of losing its predictive signal. It is computed separately for GBDT, LASSO, and RF to check consistency of findings across modeling approaches.

### 5.3 Robustness Checks

The primary evaluation uses a 96-month (8-year) window. To test sensitivity to this choice, the evaluation is repeated with **60-month (5-year)** and **120-month (10-year)** windows for horizons h=3 and h=5. Consistent findings across window sizes indicate that conclusions are not artifacts of a specific window length.

---

## 6. Results Summary

The pipeline produces the following outputs:

| Output | Format | Description |
|--------|--------|-------------|
| Figure 1 | PNG | Time-series plot of US monthly CPI inflation (1947–present) |
| Figure 2 | PNG | Bar chart of top-5 most influential predictors for the GBDT model |
| Table 1 | Console | RMSE for all models at h=1, 3, 5 |
| Table 2 | Console | R²_OOS (vs. AR benchmark) for all ML models |
| Table 3 | Console | Clark-West test statistics and p-values for all ML models |
| predictions_fixed.csv | CSV | All individual forecasts with actual values, by horizon and model |
| Robustness tables | Console | RMSE and R²_OOS for alternative window sizes (60, 120 months) |

Model performance is interpreted through the lens of both statistical significance (Clark-West test) and economic significance (R²_OOS magnitude), with the understanding that even modest R²_OOS improvements can be economically valuable in inflation forecasting.
