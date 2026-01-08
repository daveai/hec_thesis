"""
Capital Structure Analysis - MSTR Thesis
Breakeven and sensitivity analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_DIR = "../data"
FIGURES_DIR = "../figures"


def calc_breakeven(btc_holdings, capital_layers):
    """Breakeven BTC price for each capital layer"""
    print("\n" + "=" * 50)
    print("BREAKEVEN BTC PRICES BY CAPITAL LAYER")
    print("=" * 50)
    print(f"\nBTC Holdings: {btc_holdings:,}")

    results = []
    cumulative = 0

    for layer, claim in capital_layers:
        cumulative += claim
        breakeven = cumulative / btc_holdings
        results.append({
            'Layer': layer,
            'Claim': claim / 1e9,
            'Cumulative': cumulative / 1e9,
            'Breakeven': breakeven
        })
        print(f"{layer}: ${breakeven:,.0f} (claim: ${claim/1e9:.2f}B)")

    return pd.DataFrame(results)


def sensitivity(btc_holdings, total_claims, prices):
    """NAV, asset/debt ratio, equity at different BTC prices"""
    print("\n" + "=" * 50)
    print("CAPITAL STRUCTURE SENSITIVITY")
    print("=" * 50)

    results = []
    for price in prices:
        nav = btc_holdings * price
        ad_ratio = nav / total_claims
        equity = max(0, nav - total_claims)

        results.append({
            'BTC_Price': price,
            'NAV': nav / 1e9,
            'Asset_Debt': ad_ratio,
            'Equity': equity / 1e9
        })
        print(f"${price:>7,}: NAV=${nav/1e9:.1f}B, A/D={ad_ratio:.2f}x, Equity=${equity/1e9:.1f}B")

    return pd.DataFrame(results)


def plot_sensitivity(df):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # asset/debt ratio
    axes[0].plot(df['BTC_Price'] / 1000, df['Asset_Debt'], 'b-', linewidth=2)
    axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Breakeven (1.0x)')
    axes[0].set_xlabel('BTC Price ($000)')
    axes[0].set_ylabel('Asset/Debt Ratio')
    axes[0].set_title('Asset/Debt Ratio vs BTC Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # equity
    axes[1].plot(df['BTC_Price'] / 1000, df['Equity'], 'g-', linewidth=2)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('BTC Price ($000)')
    axes[1].set_ylabel('Equity Value ($B)')
    axes[1].set_title('Equity Value vs BTC Price')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "capital_sensitivity.png"), dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {FIGURES_DIR}/capital_sensitivity.png")


def run_analysis():
    # Current data (Jan 2026)
    btc_holdings = 687410

    # Capital structure by seniority
    capital_layers = [
        ('Convertible Debt', 8.20e9),
        ('STRF (Senior Pref)', 0.75e9),
        ('STRC (Variable Pref)', 2.50e9),
        ('STRK (Convert Pref)', 1.50e9),
        ('STRE (Euro Pref)', 0.72e9),
        ('STRD (Junior Pref)', 1.00e9),
    ]

    total_claims = sum(c for _, c in capital_layers)
    print(f"Total Claims: ${total_claims/1e9:.2f}B")

    breakevens = calc_breakeven(btc_holdings, capital_layers)

    prices = [150000, 120000, 95000, 75000, 55000, 35000, 21300]
    sens = sensitivity(btc_holdings, total_claims, prices)

    plot_sensitivity(sens)

    # save results
    breakevens.to_csv(os.path.join(DATA_DIR, "breakeven_prices.csv"), index=False)
    sens.to_csv(os.path.join(DATA_DIR, "sensitivity_analysis.csv"), index=False)

    return breakevens, sens


if __name__ == "__main__":
    print("MSTR Capital Structure Analysis")
    print("=" * 50)
    run_analysis()
