"""
Reflexivity Analysis - MSTR Thesis
Premium persistence, event study, Granger causality
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import os

DATA_DIR = "../data"
FIGURES_DIR = "../figures"


def load_data():
    nav = pd.read_csv(os.path.join(DATA_DIR, "nav_analysis.csv"),
                      index_col=0, parse_dates=True)
    btc = pd.read_csv(os.path.join(DATA_DIR, "btc_prices.csv"),
                      index_col=0, parse_dates=True)

    df = nav[['NAV_Premium']].copy()
    df = df.join(btc[['Return']])
    df.columns = ['Premium', 'BTC_Return']
    df['Premium_Lag1'] = df['Premium'].shift(1)
    df = df.dropna()

    return df


def premium_persistence(df):
    """Test if premium is sticky (beta close to 1 = persistent)"""
    print("\n" + "=" * 50)
    print("NAV PREMIUM PERSISTENCE")
    print("=" * 50)

    X = sm.add_constant(df['Premium_Lag1'])
    y = df['Premium']
    model = sm.OLS(y, X).fit(cov_type='HC3')

    print(f"\nPremium_t = {model.params['const']:.3f} + {model.params['Premium_Lag1']:.3f} * Premium_{{t-1}}")
    print(f"\nCoefficient: {model.params['Premium_Lag1']:.3f}")
    print(f"Std Error:   {model.bse['Premium_Lag1']:.3f}")
    print(f"t-stat:      {model.tvalues['Premium_Lag1']:.2f}")
    print(f"R-squared:   {model.rsquared:.3f}")

    if model.params['Premium_Lag1'] > 0.95:
        print("\n-> Premium states are highly persistent (near unit root)")

    return model


def event_study(df, events_file=None):
    """Compare pre-event premium to unconditional mean"""
    print("\n" + "=" * 50)
    print("EVENT STUDY: CAPITAL RAISES")
    print("=" * 50)

    # load events or use sample dates
    if events_file and os.path.exists(events_file):
        events = pd.read_csv(events_file, parse_dates=['date'])['date']
    else:
        # these are the capital raise dates I compiled from 8-Ks
        events = pd.to_datetime([
            '2024-10-30', '2024-11-11', '2024-11-18', '2024-11-25',
            '2024-12-09', '2024-12-16', '2024-12-23', '2025-01-06',
            '2025-01-13', '2025-01-27', '2025-02-10', '2025-02-24',
            '2025-03-10', '2025-03-24', '2025-04-07', '2025-04-21',
            '2025-05-05', '2025-06-02', '2025-07-07', '2025-08-04',
            '2025-09-08', '2025-10-06', '2025-11-03'
        ])

    uncond_mean = df['Premium'].mean()
    print(f"\nUnconditional mean premium: {uncond_mean*100:.1f}%")
    print(f"Number of events: {len(events)}")

    windows = [5, 10, 20]
    results = []

    for window in windows:
        pre_premiums = []

        for event in events:
            # get premium in window before event
            mask = (df.index >= event - pd.Timedelta(days=window*1.5)) & \
                   (df.index < event)
            if mask.sum() > 0:
                pre_premiums.append(df.loc[mask, 'Premium'].mean())

        if pre_premiums:
            mean_pre = np.mean(pre_premiums)
            std_pre = np.std(pre_premiums) / np.sqrt(len(pre_premiums))
            diff = mean_pre - uncond_mean
            t_stat = diff / std_pre if std_pre > 0 else 0

            results.append({
                'Window': f'{window} days',
                'Pre-Event Mean': mean_pre,
                'Difference': diff,
                't-stat': t_stat
            })

            print(f"\n{window}-day window:")
            print(f"  Pre-event mean: {mean_pre*100:.1f}%")
            print(f"  Difference:     {diff*100:.1f} pp")
            print(f"  t-stat:         {t_stat:.2f}")

    return pd.DataFrame(results)


def granger_test(df, max_lags=5):
    """Granger causality - mostly for the appendix"""
    print("\n" + "=" * 50)
    print("GRANGER CAUSALITY (for appendix)")
    print("=" * 50)

    # does premium predict BTC returns?
    print("\nTest 1: Premium -> BTC Returns")
    data1 = df[['BTC_Return', 'Premium']].dropna()
    try:
        gc1 = grangercausalitytests(data1, maxlag=max_lags, verbose=False)
        for lag in [3, 5]:
            if lag in gc1:
                f_stat = gc1[lag][0]['ssr_ftest'][0]
                p_val = gc1[lag][0]['ssr_ftest'][1]
                print(f"  Lag {lag}: F={f_stat:.2f}, p={p_val:.3f}")
    except Exception as e:
        print(f"  Error: {e}")

    # does BTC return predict premium?
    print("\nTest 2: BTC Returns -> Premium")
    data2 = df[['Premium', 'BTC_Return']].dropna()
    try:
        gc2 = grangercausalitytests(data2, maxlag=max_lags, verbose=False)
        for lag in [3, 5]:
            if lag in gc2:
                f_stat = gc2[lag][0]['ssr_ftest'][0]
                p_val = gc2[lag][0]['ssr_ftest'][1]
                print(f"  Lag {lag}: F={f_stat:.2f}, p={p_val:.3f}")
    except Exception as e:
        print(f"  Error: {e}")


def plot_event_study(df, events):
    """Plot avg premium around events - nice visual for the thesis"""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    days_range = range(-20, 15)
    avg_premium = []

    for day in days_range:
        premiums = []
        for event in events:
            target = event + pd.Timedelta(days=day)
            if target in df.index:
                premiums.append(df.loc[target, 'Premium'])
        avg_premium.append(np.mean(premiums) if premiums else np.nan)

    uncond = df['Premium'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(days_range, [p * 100 for p in avg_premium], 'b-o', linewidth=2, markersize=4)
    plt.axhline(y=uncond * 100, color='r', linestyle='--', label='Unconditional mean')
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    plt.xlabel('Days Relative to Announcement')
    plt.ylabel('Average NAV Premium (%)')
    plt.title('NAV Premium Around Capital Raise Announcements')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(FIGURES_DIR, "event_study.png"), dpi=300, bbox_inches='tight')
    print(f"\nSaved event study plot to {FIGURES_DIR}/event_study.png")


if __name__ == "__main__":
    print("MSTR Reflexivity Analysis")
    print("=" * 50)

    df = load_data()
    print(f"Loaded {len(df)} observations")

    # main analysis
    persistence_model = premium_persistence(df)
    event_results = event_study(df)

    # appendix stuff
    granger_test(df)

    print("\n" + "=" * 50)
    print("Analysis complete!")
