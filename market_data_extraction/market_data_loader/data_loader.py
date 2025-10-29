# data_loader.py
# ==========================================
# Module : Data Loader
# Description : Fetches market and macroeconomic data from Alpaca and FRED.
# ==========================================

from datetime import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from alpaca_trade_api.rest import REST

from passwords import ALPACA_API_KEY, ALPACA_API_SECRET
from utils import return_computation


# ==========================================
# 1. Global Parameters
# ==========================================
API_KEY = ALPACA_API_KEY
API_SECRET = ALPACA_API_SECRET
BASE_URL = "https://data.alpaca.markets"
API = REST(API_KEY, API_SECRET, base_url=BASE_URL)

UDL = "AAPL"
TICKERS = [
    UDL, "SPY", "QQQ", "XLK",     # Market and Tech
    "TLT", "VIXY", "GLD", "UUP"   # Macro proxies
]

FRED_CODES = {
    "CPI": "CPIAUCSL",
    "INDPPI": "PPIACO",
    "M1SUPPLY": "M1SL",
    "CCREDIT": "TOTALSL",
    "BMINUSA": "BAA10Y",
    "AAA10Y": "AAA10Y",
    "TB3MS": "TB3MS"
}

START_DATE = "2023-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
TIMEFRAME = "1Day"


# ==========================================
# 2. Alpaca Client
# ==========================================
def alpaca_client(api_key: str, api_secret: str, base_url: str = BASE_URL) -> REST:
    """
    Create an Alpaca REST client instance.
    """
    return REST(api_key, api_secret, base_url=base_url)


# ==========================================
# 3. Market Data Extraction
# ==========================================
def get_market_data(
    api: REST,
    tickers: list[str],
    start: str,
    end: str | None = None,
    timeframe: str = "1Day",
    feed: str = "iex",
    verbose: bool = True
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Fetch historical OHLCV data for a list of tickers from Alpaca API.

    Returns:
        - retn (pd.DataFrame): computed returns for each ticker
        - prices (dict): dictionary of raw price DataFrames per ticker
    """
    prices_close, prices = {}, {}

    for ticker in tickers:
        df = api.get_bars(ticker, timeframe=timeframe, start=start, end=end, feed=feed).df
        prices[ticker] = df
        prices_close[ticker] = df["close"]

    px = pd.DataFrame(prices_close).dropna().sort_index()
    retn = return_computation(px, tickers)

    if verbose:
        print("Market data loaded:")
        print(f"Frequency: {timeframe}")
        print(f"Period: {retn.index.min().date()} → {retn.index.max().date()}")
        print(f"{retn.shape[0]} observations, {retn.shape[1]} tickers")

    return retn, px, prices


# ==========================================
# 4. Macroeconomic Data Extraction (FRED)
# ==========================================
def get_macro_data(
    fred_codes: dict[str, str],
    start: str,
    end: str | None = None,
    resample_rule: str = "D",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch macroeconomic data from FRED via pandas_datareader.
    """
    fred = web.DataReader(list(fred_codes.values()), "fred", start, end)
    fred.columns = fred_codes.keys()

    # Credit spread (BAA - AAA)
    fred["BMINUSA"] = fred["BMINUSA"] - fred["AAA10Y"]

    fred_daily = (
        fred.resample(resample_rule)
        .interpolate(method="time")  # temporal interpolation
        .bfill()                     # security for the first dates
        .ffill()                     # security for the last dates
    )

    if verbose:
        print("FRED data loaded:")
        print(f"Period: {fred_daily.index.min().date()} → {fred_daily.index.max().date()}")
        print(f"{fred_daily.shape[0]} observations, {fred_daily.shape[1]} variables")

    return fred_daily



# ==========================================
# 5. Macro Variable Construction
# ==========================================
def build_macro_variables(fred_data: pd.DataFrame, resample_rule: str = "D", verbose: bool = True) -> pd.DataFrame:
    """
    Construct key macroeconomic variables (inflation, credit, spreads, etc.).
    """
    macro = pd.DataFrame(index=fred_data.index)
    macro["INF"] = np.log(fred_data["CPI"] / fred_data["CPI"].shift(1)) * 100
    macro["DP"] = np.log(fred_data["INDPPI"] / fred_data["INDPPI"].shift(1)) * 100
    macro["DM"] = np.log(fred_data["M1SUPPLY"] / fred_data["M1SUPPLY"].shift(1)) * 100
    macro["DC"] = np.log(fred_data["CCREDIT"] / fred_data["CCREDIT"].shift(1)) * 100
    macro["DS"] = fred_data["BMINUSA"].diff()
    macro["TS"] = fred_data["AAA10Y"] - fred_data["TB3MS"]
    macro["DT"] = macro["TS"].diff()
    if resample_rule == "D": 
        macro["RF"] = ((1 + fred_data["TB3MS"] / 100) ** (1 / 252) - 1) * 100
    elif resample_rule == "W": 
        macro["RF"] = ((1 + fred_data["TB3MS"] / 100) ** (1 / 52) - 1) * 100
    elif resample_rule == "M":
        macro["RF"] = ((1 + fred_data["TB3MS"] / 100) ** (1 / 12) - 1) * 100
    elif resample_rule == "Y":
        macro["RF"] = fred_data["TB3MS"] / 100

    macro = macro.ffill().bfill()


    if verbose:
        print("Macro variables built:")
        print(f"Period: {macro.index.min().date()} → {macro.index.max().date()}")
        print(f"{macro.shape[0]} observations, {macro.shape[1]} variables")

    return macro


# ==========================================
# 6. Main (for standalone testing)
# ==========================================
if __name__ == "__main__":
    api = alpaca_client(API_KEY, API_SECRET)
    returns, prices = get_market_data(api, TICKERS, START_DATE, END_DATE)
    fred_data = get_macro_data(FRED_CODES, START_DATE, END_DATE)
    macro_vars = build_macro_variables(fred_data)
    macro_vars.to_csv("market_data_macro_daily.csv")
    returns.to_csv("market_data_returns.csv")