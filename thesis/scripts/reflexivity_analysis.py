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
    """Measure pre/post event premiums using business day offsets"""
    print("\n" + "=" * 50)
    print("EVENT STUDY: CAPITAL RAISES")
    print("=" * 50)

    # load events or use sample dates
    if events_file and os.path.exists(events_file):
        events = pd.read_csv(events_file, parse_dates=['date'])['date']
    else:
        # capital raise dates compiled from 8-Ks
        events = pd.to_datetime([
            '2024-10-30', '2024-11-11', '2024-11-18', '2024-11-25',
            '2024-12-09', '2024-12-16', '2024-12-23', '2025-01-06',
            '2025-01-13', '2025-01-27', '2025-02-10', '2025-02-24',
            '2025-03-10', '2025-03-24', '2025-04-07', '2025-04-21',
            '2025-05-05', '2025-06-02', '2025-07-07', '2025-08-04',
            '2025-09-08', '2025-10-06', '2025-11-03'
        ])

    trading_days = df.index.sort_values()

    def nearest_trading_day(date):
        idx = trading_days.searchsorted(date)
        if idx >= len(trading_days):
            return trading_days[-1]
        if idx > 0 and (trading_days[idx] - date) > (date - trading_days[idx-1]):
            return trading_days[idx-1]
        return trading_days[idx]

    uncond_mean = df['Premium'].mean()
    print(f"\nUnconditional mean premium: {uncond_mean*100:.1f}%")
    print(f"Number of events: {len(events)}")

    # Pre-event and post-event windows
    windows = [
        ('20 days pre-event', -20, -1),
        ('10 days pre-event', -10, -1),
        ('5 days pre-event', -5, -1),
        ('5 days post-event', 1, 5),
        ('10 days post-event', 1, 10),
    ]

    print(f"\n{'Window':<25} {'Avg Premium':>12}")
    print("-" * 40)

    for label, start_offset, end_offset in windows:
        event_premiums = []

        for event in events:
            event_td = nearest_trading_day(event)
            event_idx = trading_days.get_loc(event_td)

            start_idx = max(0, event_idx + start_offset)
            end_idx = min(len(trading_days) - 1, event_idx + end_offset)

            window_days = trading_days[start_idx:end_idx + 1]
            premiums = df.loc[df.index.isin(window_days), 'Premium'].dropna()
            if len(premiums) > 0:
                event_premiums.append(premiums.mean())

        if event_premiums:
            mean_prem = np.mean(event_premiums)
            print(f"{label:<25} {mean_prem*100:>11.1f}%")

    return events


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
    """Plot avg premium around events using business day offsets"""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    trading_days = df.index.sort_values()

    def nearest_trading_day(date):
        idx = trading_days.searchsorted(date)
        if idx >= len(trading_days):
            return trading_days[-1]
        if idx > 0 and (trading_days[idx] - date) > (date - trading_days[idx-1]):
            return trading_days[idx-1]
        return trading_days[idx]

    days_range = range(-20, 15)
    avg_premium = []

    for day in days_range:
        premiums = []
        for event in events:
            event_td = nearest_trading_day(event)
            event_idx = trading_days.get_loc(event_td)
            target_idx = event_idx + day
            if 0 <= target_idx < len(trading_days):
                target_date = trading_days[target_idx]
                if target_date in df.index and not pd.isna(df.loc[target_date, 'Premium']):
                    premiums.append(df.loc[target_date, 'Premium'])
        avg_premium.append(np.mean(premiums) if premiums else np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(days_range, [p * 100 for p in avg_premium], 'b-o', linewidth=2, markersize=4)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    plt.xlabel('Days Relative to Announcement')
    plt.ylabel('Average NAV Premium (%)')
    plt.title('NAV Premium Around Capital Raise Announcements')
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
