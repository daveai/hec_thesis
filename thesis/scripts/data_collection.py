"""
Data Collection - MSTR Thesis
Downloads BTC, MSTR, SPY from Yahoo Finance
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os

START_DATE = "2020-08-11"  # MSTR's first BTC buy
END_DATE = datetime.now().strftime("%Y-%m-%d")
DATA_DIR = "../data"


def download_prices():
    """Download price data"""
    tickers = {
        "BTC-USD": "btc_prices.csv",
        "MSTR": "mstr_prices.csv",
    }

    os.makedirs(DATA_DIR, exist_ok=True)

    for ticker, filename in tickers.items():
        print(f"Downloading {ticker}...")

        data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)

        # yfinance sometimes returns multiindex cols, this fixes it
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # calc returns and vol
        data['Return'] = data['Close'].pct_change()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Vol_30d'] = data['Return'].rolling(30).std() * np.sqrt(252)
        data['Vol_90d'] = data['Return'].rolling(90).std() * np.sqrt(252)

        filepath = os.path.join(DATA_DIR, filename)
        data.to_csv(filepath)
        print(f"  Saved to {filepath}")

    print("\nData download complete!")
    return data


def calc_nav_premium():
    """Calculate MSTR NAV premium to BTC holdings"""
    btc = pd.read_csv(os.path.join(DATA_DIR, "btc_prices.csv"), index_col=0, parse_dates=True)
    mstr = pd.read_csv(os.path.join(DATA_DIR, "mstr_prices.csv"), index_col=0, parse_dates=True)

    # BTC holdings over time - manually compiled from 10-Qs
    # TODO: automate this with SEC EDGAR API someday
    holdings = pd.DataFrame({
        'Date': [
            '2020-08-11', '2020-09-14', '2020-12-21',
            '2021-02-24', '2021-06-21', '2021-12-31',
            '2022-12-31', '2023-12-31', '2024-03-31',
            '2024-06-30', '2024-09-30', '2024-12-31',
            '2025-03-31', '2025-06-30', '2025-09-30',
            '2025-12-31'
        ],
        'BTC_Holdings': [
            21454, 38250, 70470,
            90531, 105085, 124391,
            132500, 189150, 214246,
            226500, 252220, 444262,
            506137, 576230, 651140,
            713502
        ]
    })
    holdings['Date'] = pd.to_datetime(holdings['Date'])
    holdings.set_index('Date', inplace=True)

    # merge everything
    combined = btc[['Close']].rename(columns={'Close': 'BTC_Price'})
    combined = combined.join(mstr[['Close']].rename(columns={'Close': 'MSTR_Price'}))
    combined = combined.join(holdings)
    combined['BTC_Holdings'] = combined['BTC_Holdings'].ffill()

    # NAV and premium
    MSTR_SHARES = 244_000_000  # approx, need to update from latest 10-Q
    combined['NAV'] = combined['BTC_Holdings'] * combined['BTC_Price']
    combined['Market_Cap'] = combined['MSTR_Price'] * MSTR_SHARES
    combined['NAV_Premium'] = (combined['Market_Cap'] - combined['NAV']) / combined['NAV']

    combined.to_csv(os.path.join(DATA_DIR, "nav_analysis.csv"))
    print("NAV analysis saved to nav_analysis.csv")

    return combined


def save_sec_data():
    """
    SEC filing data - placeholder
    Would be nice to pull this automatically but for now just hardcoded
    """
    # convertible notes (from SEC filings / prospectus supplements)
    converts = pd.DataFrame({
        'Issue_Date': ['2020-12-11', '2021-02-19', '2024-03-08', '2024-06-15',
                       '2024-09-20', '2024-11-21', '2025-02-15'],
        'Principal': [650e6, 1050e6, 604e6, 800e6, 1010e6, 3000e6, 2000e6],
        'Coupon': [0.0075, 0.0, 0.00875, 0.0225, 0.00625, 0.0, 0.0],
        'Maturity': ['2025-12-15', '2027-02-15', '2031-03-15', '2032-06-15',
                     '2028-09-15', '2029-12-15', '2030-03-15'],
        'Conversion_Price': [398, 1432, 232.72, 204.33, 183.19, 672.40, 433.43],
        'Status': ['Called', 'Called', 'Outstanding', 'Outstanding',
                   'Outstanding', 'Outstanding', 'Outstanding']
    })

    # preferred shares
    prefs = pd.DataFrame({
        'Ticker': ['STRK', 'STRF', 'STRD', 'STRC', 'STRE'],
        'Name': ['Strike', 'Strife', 'Stride', 'Stretch', 'Stream'],
        'Launch': ['2025-02', '2025-03', '2025-06', '2025-07', '2025-11'],
        'Notional': [1.54e9, 1.37e9, 1.44e9, 3.38e9, 0.80e9],
        'Dividend_Rate': [0.08, 0.10, 0.10, 0.11, 0.10],
        'Cumulative': [True, True, False, True, True]
    })

    converts.to_csv(os.path.join(DATA_DIR, "convertible_debt.csv"), index=False)
    prefs.to_csv(os.path.join(DATA_DIR, "preferred_shares.csv"), index=False)

    print("SEC filing data saved.")
    return converts, prefs


if __name__ == "__main__":
    print("=" * 50)
    print("MSTR Thesis Data Collection")
    print("=" * 50)

    download_prices()
    calc_nav_premium()
    save_sec_data()

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)
