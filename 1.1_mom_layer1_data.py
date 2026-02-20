"""
=============================================================
  QUANT STRATEGY: Cross-Sectional Momentum
  Layer 1: Data Pipeline
=============================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
import warnings
warnings.filterwarnings("ignore")

CORE_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","LLY","AVGO","JPM",
    "TSLA","UNH","V","XOM","MA","JNJ","PG","COST","HD","MRK","ABBV","CVX",
    "CRM","BAC","NFLX","AMD","KO","PEP","TMO","ACN","MCD","WMT","CSCO","ABT",
    "ADBE","LIN","DHR","TXN","WFC","PM","NEE","AMGN","ORCL","IBM","RTX","QCOM",
    "INTU","AMAT","CAT","GS","SPGI","BLK","HON","BA","ISRG","BKNG","GILD",
    "AXP","SYK","VRTX","MDT","GE","REGN","DE","PLD","ADI","MS","LRCX","KLAC",
    "MU","CI","CB","SO","DUK","TJX","MMC","ZTS","BSX","PGR","AON","MDLZ",
    "HCA","CME","NOC","USB","ICE","MCO","ELV","ITW","ECL","APD","PSA","O",
    "CL","SHW","MO","NSC","FDX","EMR","ADP","MCK","BMY","ETN","COF","MAR",
    "WM","WELL","TT","PH","HLT","CTAS","ROP","AIG","ALL","GD","TGT","AFL",
    "F","GM","FCX","PCAR","CARR","PAYX","ROST","OKE","PSX","MPC","VLO","SLB",
    "BK","STZ","KMB","YUM","EW","OTIS","FAST","EXC","LHX","KR","IQV","CTSH",
    "BIIB","IDXX","DXCM","NKE","SBUX","DIS","EA","ABNB","UBER","COP","OXY",
    "DVN","EOG","HES","MRO","APA","EQT","WMB","KMI","HAL","NEM","AMT","CCI",
    "EQIX","DLR","SPG","VICI","EQR","AVB","UDR","ESS","MAA","CPT","VTR",
    "WELL","LOW","TGT","EBAY","ETSY","SHOP","SQ","PYPL","COIN","HOOD","SOFI",
    "NET","SNOW","PLTR","PATH","UI","ZS","CRWD","OKTA","DDOG","MDB","GTLB",
    "HCP","BIO","TMO","A","WAT","FMC","ALB","PPG","SHW","RPM","HXL","SON",
    "PKG","IP","WRK","SEE","TRP","ENB","TRP","PPL","AEE","LNT","EVRG","CMS",
]

CORE_TICKERS = list(dict.fromkeys(CORE_TICKERS))


def download_prices(tickers, start="2010-01-01", end="2024-12-31", batch_size=50):
    """
    Download adjusted closing prices in batches.
    Adjusted prices account for splits and dividends — essential
    for momentum. Without adjustment, a stock split looks like
    a massive negative return, destroying the signal.
    """
    all_prices = []
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    print(f"[Layer 1] Downloading {len(tickers)} tickers in {len(batches)} batches...")

    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}/{len(batches)} ({len(batch)} tickers)...", end=" ", flush=True)
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw
            all_prices.append(close)
            print("v")
        except Exception as e:
            print(f"x {e}")
        time.sleep(0.5)

    prices = pd.concat(all_prices, axis=1)
    prices.index = pd.to_datetime(prices.index)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    prices = prices.sort_index()
    print(f"\n[Layer 1] Raw shape: {prices.shape}")
    return prices


def clean_prices(prices, min_history_years=3, max_missing_pct=0.10):
    """
    Clean raw price matrix.
    1. Forward-fill gaps up to 5 days
    2. Drop stocks with >10% missing data
    3. Drop stocks with <3 years of history
    """
    print("\n[Layer 1] Cleaning...")
    n_start = len(prices.columns)

    prices = prices.ffill(limit=5)

    missing = prices.isna().mean()
    prices  = prices.loc[:, missing <= max_missing_pct]
    print(f"  Missing data filter : {len(prices.columns)} stocks (dropped {n_start - len(prices.columns)})")

    min_days = min_history_years * 252
    valid    = prices.notna().sum() >= min_days
    prices   = prices.loc[:, valid]
    print(f"  History filter      : {len(prices.columns)} stocks (need {min_history_years}+ years)")

    prices = prices.dropna(how="all")
    return prices


def to_monthly(prices):
    """
    Resample to month-end prices.
    Momentum is a monthly phenomenon per Jegadeesh & Titman (1993).
    Monthly rebalancing also keeps transaction costs manageable.
    """
    monthly = prices.resample("M").last()
    print(f"\n[Layer 1] Monthly shape: {monthly.shape}")
    return monthly


def data_quality_report(prices_daily, prices_monthly):
    print("\n" + "="*55)
    print("  DATA QUALITY REPORT")
    print("="*55)
    print(f"  Daily  : {prices_daily.shape[0]} days x {prices_daily.shape[1]} stocks")
    print(f"  Monthly: {prices_monthly.shape[0]} months x {prices_monthly.shape[1]} stocks")
    print(f"  Range  : {prices_daily.index[0].date()} to {prices_daily.index[-1].date()}")
    print(f"  Missing: {prices_daily.isna().mean().mean()*100:.1f}% overall\n")
    print("  Stock coverage by year:")
    annual = prices_daily.resample("A").last()
    for yr in annual.index:
        count = annual.loc[yr].notna().sum()
        bar   = "#" * (count // 5)
        print(f"    {yr.year}: {count:3d} stocks  {bar}")
    print("="*55)


if __name__ == "__main__":

    print(f"[Layer 1] Universe: {len(CORE_TICKERS)} tickers\n")

    prices_daily   = download_prices(CORE_TICKERS, "2010-01-01", "2024-12-31")
    prices_daily   = clean_prices(prices_daily)
    prices_monthly = to_monthly(prices_daily)

    data_quality_report(prices_daily, prices_monthly)

    prices_daily.to_csv("prices_daily.csv")
    prices_monthly.to_csv("prices_monthly.csv")
    print("\n[Done] Saved -> prices_daily.csv, prices_monthly.csv")

    print("\nSample (last 3 months, first 5 stocks):")
    print(prices_monthly.iloc[-3:, :5].round(2).to_string())
