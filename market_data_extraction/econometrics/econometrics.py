import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def analyze_series(series: pd.Series, lags: int = 20, plot: bool = True) -> pd.DataFrame:
    """
    Perform an in-depth statistical analysis of a time series.
    
    Parameters
    ----------
    series : pd.Series
        The time series to analyze.
    lags : int
        Number of lags for autocorrelation and partial autocorrelation.
    plot : bool
        Whether to plot the distribution and ACF/PACF.

    Returns
    -------
    summary : pd.DataFrame
        DataFrame summarizing descriptive, shape, normality, and stationarity statistics.
    """

    # --- Clean series ---
    series = series.dropna()
    name = series.name if series.name else "Series"

    # --- Descriptive statistics ---
    desc = series.describe()
    mean, std, var = series.mean(), series.std(), series.var()
    skew, kurt = series.skew(), series.kurtosis()

    # --- Normality tests ---
    jb_stat, jb_p = stats.jarque_bera(series)
    shapiro_stat, shapiro_p = stats.shapiro(series.sample(min(5000, len(series))))  # cap to 5k obs for Shapiro

    # --- Stationarity tests ---
    adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag='AIC')
    kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')

    # --- Autocorrelation metrics ---
    acf_vals = acf(series, nlags=lags)
    pacf_vals = pacf(series, nlags=lags)
    lag1_autocorr = series.autocorr(lag=1)
    lag5_autocorr = series.autocorr(lag=5)

    # --- Volatility / rolling stats ---
    rolling_std = series.rolling(window=20).std().mean()
    rolling_mean = series.rolling(window=20).mean().std()

    # --- Create summary table ---
    summary = pd.DataFrame({
        "count": [desc["count"]],
        "mean": [mean],
        "std": [std],
        "var": [var],
        "min": [desc["min"]],
        "max": [desc["max"]],
        "skewness": [skew],
        "kurtosis": [kurt],
        "jarque_bera_p": [jb_p],
        "shapiro_p": [shapiro_p],
        "ADF_stat": [adf_stat],
        "ADF_p": [adf_p],
        "KPSS_stat": [kpss_stat],
        "KPSS_p": [kpss_p],
        "autocorr_lag1": [lag1_autocorr],
        "autocorr_lag5": [lag5_autocorr],
        "rolling_std_mean": [rolling_std],
        "rolling_mean_std": [rolling_mean]
    }, index=[name])

    # --- Optional plots ---
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        series.plot(ax=axes[0, 0], title=f"{name} - Time Series", color='steelblue')
        series.hist(ax=axes[0, 1], bins=30, color='gray', edgecolor='black')
        axes[0, 1].set_title(f"{name} - Distribution")

        plot_acf(series, ax=axes[1, 0], lags=lags, title=f"{name} - ACF")
        plot_pacf(series, ax=axes[1, 1], lags=lags, title=f"{name} - PACF")

        plt.tight_layout()
        plt.show()

    # --- Interpretation hints ---
    print("Normality test (Jarque–Bera): p =", round(jb_p, 4))
    print("Stationarity tests:")
    print(f" - ADF p-value = {adf_p:.4f} (H0: non-stationary)")
    print(f" - KPSS p-value = {kpss_p:.4f} (H0: stationary)")
    print(f"Lag-1 autocorrelation = {lag1_autocorr:.3f}")
    print(f"Lag-5 autocorrelation = {lag5_autocorr:.3f}")

    return summary
