"""
Full Analysis Script - HEC Thesis
Computes all stats needed for results tables
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
btc = pd.read_csv('../data/btc_prices.csv', index_col=0, parse_dates=True)
mstr = pd.read_csv('../data/mstr_prices.csv', index_col=0, parse_dates=True)

print("=" * 70)
print("HEC THESIS - FULL ANALYSIS RESULTS")
print("=" * 70)

# =============================================================================
# PARAMETERS (Jan 2026)
# =============================================================================
BTC_HOLDINGS = 713502
TOTAL_CLAIMS = 15.944e9  # converts ($7.414B) + preferred ($8.53B)

# Capital structure (by seniority)
CAPITAL_LAYERS = [
    ('Convertible Debt', 7.414e9),
    ('STRF (Senior Preferred)', 1.37e9),
    ('STRC (Variable Preferred)', 3.38e9),
    ('STRE (Euro Preferred)', 0.80e9),
    ('STRK (Convert Preferred)', 1.54e9),
    ('STRD (Junior Preferred)', 1.44e9),
]

# =============================================================================
# 1. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 70)

btc_price = btc['Close'] if 'Close' in btc.columns else btc.iloc[:, 0]
mstr_price = mstr['Close'] if 'Close' in mstr.columns else mstr.iloc[:, 0]

# NAV premium calc
shares = 244_000_000
nav_per_share = (BTC_HOLDINGS * btc_price) / shares
nav_premium = ((mstr_price - nav_per_share) / nav_per_share * 100).dropna()

def desc_stats(series, name):
    s = series.dropna()
    return {
        'name': name,
        'mean': s.mean(),
        'std': s.std(),
        'min': s.min(),
        'max': s.max(),
    }

stats_list = [
    desc_stats(btc_price, 'BTC Price ($)'),
    desc_stats(mstr_price, 'MSTR Price ($)'),
    desc_stats(nav_premium, 'NAV Premium (%)'),
]

print("\nTable: Descriptive Statistics")
print("-" * 60)
print(f"{'Variable':<20} {'Mean':>12} {'Std Dev':>12} {'Min':>12} {'Max':>12}")
print("-" * 60)
for s in stats_list:
    print(f"{s['name']:<20} {s['mean']:>12,.2f} {s['std']:>12,.2f} {s['min']:>12,.2f} {s['max']:>12,.2f}")

# =============================================================================
# 2. BREAKEVEN ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("2. BREAKEVEN BTC PRICES BY CAPITAL LAYER")
print("=" * 70)

print(f"\nBTC Holdings: {BTC_HOLDINGS:,}")
print(f"Total Claims: ${TOTAL_CLAIMS/1e9:.2f}B")

cumulative = 0
print(f"\n{'Capital Layer':<30} {'Claim ($B)':>12} {'Breakeven BTC':>15}")
print("-" * 60)
for layer, claim in CAPITAL_LAYERS:
    cumulative += claim
    breakeven = cumulative / BTC_HOLDINGS
    print(f"{layer:<30} ${claim/1e9:>10.2f}B ${breakeven:>13,.0f}")

# equity wipeout price
print(f"{'Common Equity (wipeout)':<30} {'--':>12} ${TOTAL_CLAIMS/BTC_HOLDINGS:>13,.0f}")

# =============================================================================
# 3. SENSITIVITY ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("3. CAPITAL STRUCTURE SENSITIVITY")
print("=" * 70)

btc_scenarios = [120000, 95000, 75000, 55000, 35000, 22341]

print(f"\n{'BTC Price':<15} {'NAV ($B)':>12} {'Asset/Debt':>12} {'Equity ($B)':>12}")
print("-" * 55)
for price in btc_scenarios:
    nav = BTC_HOLDINGS * price
    ad_ratio = nav / TOTAL_CLAIMS
    equity = max(0, nav - TOTAL_CLAIMS)
    print(f"${price:>13,} ${nav/1e9:>11.1f} {ad_ratio:>11.2f}x ${equity/1e9:>11.1f}")

# =============================================================================
# 4. NAV PREMIUM PERSISTENCE
# =============================================================================
print("\n" + "=" * 70)
print("4. NAV PREMIUM PERSISTENCE")
print("=" * 70)

# prep data for regression
premium_data = nav_premium.dropna()
premium_lag = premium_data.shift(1)
reg_data = pd.DataFrame({'y': premium_data, 'x': premium_lag}).dropna()

# simple OLS (doing it manually bc why not)
y = reg_data['y'].values
X = np.column_stack([np.ones(len(reg_data)), reg_data['x'].values])

beta = np.linalg.lstsq(X, y, rcond=None)[0]
resid = y - X @ beta
n, k = X.shape
sigma2 = np.sum(resid**2) / (n - k)
var_beta = sigma2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(var_beta))
t_stats = beta / se
r2 = 1 - np.sum(resid**2) / np.sum((y - np.mean(y))**2)

print(f"\nPremium_t = {beta[0]:.3f} + {beta[1]:.3f} * Premium_{{t-1}}")
print(f"\nCoefficient on lagged premium: {beta[1]:.3f}")
print(f"Standard error: {se[1]:.3f}")
print(f"t-statistic: {t_stats[1]:.2f}")
print(f"R-squared: {r2:.3f}")
print(f"Observations: {n}")

if beta[1] > 0.95:
    print("\n-> Premium states are highly persistent")

# =============================================================================
# 5. SCENARIO ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("5. SCENARIO ANALYSIS")
print("=" * 70)

scenarios = [
    ('Base Case', 95000),
    ('Moderate (-30%)', 66500),
    ('Severe (-50%)', 47500),
    ('Prolonged (-70%)', 28500),
]

print(f"\n{'Scenario':<20} {'BTC Price':>12} {'NAV ($B)':>12} {'Asset/Debt':>12} {'Equity ($B)':>12}")
print("-" * 70)
for name, price in scenarios:
    nav = BTC_HOLDINGS * price
    ad_ratio = nav / TOTAL_CLAIMS
    equity = max(0, nav - TOTAL_CLAIMS)
    print(f"{name:<20} ${price:>10,} ${nav/1e9:>11.1f} {ad_ratio:>11.2f}x ${equity/1e9:>11.1f}")

# =============================================================================
# 6. GRANGER CAUSALITY (for appendix)
# =============================================================================
print("\n" + "=" * 70)
print("6. GRANGER CAUSALITY (for appendix)")
print("=" * 70)

btc_ret = btc_price.pct_change().dropna()

def simple_granger(y, x, maxlag=5):
    """
    Manual Granger test - wanted to understand it better so wrote it out
    Basically: does adding x lags improve prediction of y?
    """
    df = pd.DataFrame({'y': y, 'x': x}).dropna()

    for i in range(1, maxlag+1):
        df[f'y_lag{i}'] = df['y'].shift(i)
        df[f'x_lag{i}'] = df['x'].shift(i)

    df = df.dropna()

    # restricted model (just y lags)
    y_vec = df['y'].values
    X_r = df[[f'y_lag{i}' for i in range(1, maxlag+1)]].values
    X_r = np.column_stack([np.ones(len(X_r)), X_r])

    beta_r = np.linalg.lstsq(X_r, y_vec, rcond=None)[0]
    rss_r = np.sum((y_vec - X_r @ beta_r)**2)

    # unrestricted (add x lags)
    X_u = df[[f'y_lag{i}' for i in range(1, maxlag+1)] + [f'x_lag{i}' for i in range(1, maxlag+1)]].values
    X_u = np.column_stack([np.ones(len(X_u)), X_u])

    beta_u = np.linalg.lstsq(X_u, y_vec, rcond=None)[0]
    rss_u = np.sum((y_vec - X_u @ beta_u)**2)

    # F-test
    n = len(y_vec)
    k_r = X_r.shape[1]
    k_u = X_u.shape[1]

    f_stat = ((rss_r - rss_u) / (k_u - k_r)) / (rss_u / (n - k_u))
    p_val = 1 - stats.f.cdf(f_stat, k_u - k_r, n - k_u)

    return f_stat, p_val

premium_aligned = nav_premium.reindex(btc_ret.index).dropna()
btc_aligned = btc_ret.loc[premium_aligned.index]

f1, p1 = simple_granger(btc_aligned.values, premium_aligned.values)
f2, p2 = simple_granger(premium_aligned.values, btc_aligned.values)

print("\nGranger Causality Test Results (5 lags)")
print("-" * 60)
print(f"{'Null Hypothesis':<35} {'F-stat':>10} {'p-value':>10}")
print("-" * 60)
print(f"{'Premium does not GC BTC':<35} {f1:>10.2f} {p1:>10.3f}")
print(f"{'BTC does not GC Premium':<35} {f2:>10.2f} {p2:>10.3f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
