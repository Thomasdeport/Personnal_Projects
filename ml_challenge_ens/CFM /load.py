import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import skew

# ======================
# UTILS
# ======================

def signed_log(x):
    """Signed log transform (robust to zeros and negatives)."""
    x = np.array(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def clip_sequence_outliers(df, group_col='obs_id', cols=None, n_std=6):
    """Caps extreme values within each sequence (fast vectorized version)."""
    if cols is None:
        cols = ['price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux']

    grouped = df.groupby(group_col)[cols]
    means = grouped.transform('mean')
    stds = grouped.transform('std').replace(0, 1e-6)

    df[cols] = np.clip(df[cols], means - n_std * stds, means + n_std * stds)
    return df


def normalize_per_sequence(df, group_col, features):
    """Z-score normalization within each sequence (avoids data leakage)."""
    grouped = df.groupby(group_col)[features]
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, 1e-6)
    df[features] = (df[features] - means) / stds
    return df


# ======================
# FEATURE ENGINEERING
# ======================


def log_transform(x):
    """Signed log transform robust to zeros and negatives."""
    x = np.array(x, dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    return np.sign(x) * np.log1p(np.abs(x))


def add_feature_engineering(
    df: pd.DataFrame,
    test: bool = False,
    use_log=True,
    momentum_windows = [5,20],
    corr_threshold=0.8
):
    """
    Robust market microstructure feature engineering.
    Adds regime-aware and contextual features.
    Focused on stability and avoiding overfitting.
    """

    base_cols = ["obs_id", "order_id", "venue", "action", "side", "trade", "_is_train"]
    features = []
    df = df.copy().fillna(0)
    # === 1. Spread & relative price structure ===
    df["bid_ask_spread"] = df["ask"] - df["bid"]
    df['rel_spread'] = df['bid_ask_spread'] / (df['price'] + 1e-6)
    df["liq_compression"] = np.log1p(df["bid_size"] + df["ask_size"]) / (df["bid_ask_spread"] + 1e-6)

    features += ["bid_ask_spread", "rel_spread", "liq_compression"]

    # === 2. Imbalance & OFI ===
    df['imbalance'] = df['bid_size'] - df['ask_size']
    df['ofi'] = (df['bid_size'] - df.groupby('obs_id')['bid_size'].shift(1).fillna(0)) - (df['ask_size'] - df.groupby('obs_id')['ask_size'].shift(1).fillna(0))
    df["cum_ofi"] = df.groupby("obs_id")["ofi"].cumsum()

    features += ["imbalance", "ofi", "cum_ofi"]

    # === 3. Short-term price & flux dynamics ===
    df["price_change"] = df.groupby("obs_id")["price"].diff().fillna(0)
    df["rolling_volatility"] = df.groupby("obs_id")["price_change"].transform(lambda x: x.rolling(15, min_periods=3).std().fillna(0))
    df["vol_slope"] = df.groupby("obs_id")["rolling_volatility"].diff().fillna(0)
    features += ["price_change", "rolling_volatility", "vol_slope"]

    df["short_vol"] = df.groupby("obs_id")["price_change"].transform(lambda x: x.rolling(5).std())
    df["long_vol"] = df.groupby("obs_id")["price_change"].transform(lambda x: x.rolling(30).std())
    df["vol_ratio"] = (df["short_vol"] / (df["long_vol"] + 1e-6)).fillna(0)
    features += ["vol_ratio"]
    # === 4. Momentum ===
    for window in momentum_windows:
        col = f'momentum_{window}'
        df[col] = df.groupby('obs_id')['price'].diff(window).fillna(0)
        features.append(col)

    # === 5. Flux indicator ===
    df["flux_change"] = df.groupby("obs_id")["flux"].diff().fillna(0)
    features += ["flux_change"]

    # === 6. Contextual flow pressure (optional extra feature) ===
    df['trend_consistency'] = df.groupby('obs_id')['price_change'].transform(lambda x: (x > 0).rolling(10, min_periods=1).mean().fillna(0))
    features += ["trend_consistency"]
    
    # === 7. Market regime feature ===
    # Standardize key drivers
    z_vol = (df["rolling_volatility"] - df["rolling_volatility"].mean()) / (df["rolling_volatility"].std() + 1e-6)
    z_spread = (df["rel_spread"] - df["rel_spread"].mean()) / (df["rel_spread"].std() + 1e-6)
    z_ofi = (df["ofi"] - df["ofi"].mean()) / (df["ofi"].std() + 1e-6)
    
    # Combine with weights
    df["market_stress_index"] = 0.5 * z_vol + 0.3 * z_spread + 0.2 * np.abs(z_ofi)
    
    # Optional normalization for neural stability
    df["market_stress_index"] = np.tanh(df["market_stress_index"])
    features += ["market_stress_index"]
 
    # === 7. Log transform stabilisation for heavy-tailed signals ===
    if use_log:
        always_log = ['flux']
        conditionally_log = [f for f in features if f not in ['flux', 'market_stress_index']]


        for col in always_log:
            if col in df.columns:
                df[col] = log_transform(df[col])
        print("add to log columns : ")
        for col in conditionally_log:
            if col in df.columns:
                data = df[col].fillna(0)
                if abs(skew(data)) > 3 and not test:
                    print (col)
                    df[col] = log_transform(data)

    # === 8. Cleanup ===
    df.drop(columns=["price", "bid", "ask"], inplace=True, errors="ignore")
    df.fillna(0, inplace=True)

    # === 9. Drop highly correlated columns (prevent redundancy) ===
    if corr_threshold and len(features) > 1:
        corr = df[features].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
        if to_drop:
            print(f"Dropped (corr > {corr_threshold}): {to_drop}")
        df.drop(columns=to_drop, inplace=True)
        features = [f for f in features if f not in to_drop]

    final_cols = base_cols + features

    return df[final_cols], final_cols




# ======================
# MAIN PIPELINE
# ======================

def load_cfm_data(
    dummy=False,
    normalize=True,
    verbose=True,
    return_pd=False,
    already_processed=False,
    dict_feature_eng={},
    threshold = 6
):
    """Full CFM pipeline: log flux, intra-sequence clipping, normalization per sequence, with timing."""
    
    t0 = time.time()
    print("Starting CFM pipeline\n")
    
    # === 1. Already preprocessed data ===
    if already_processed:
        t = time.time()
        X_train = pd.read_csv("/kaggle/input/processed-data/X_train.csv")
        y_train = pd.read_csv("/kaggle/input/processed-data/y_train.csv")
        X_test = pd.read_csv("/kaggle/input/processed-data/X_test.csv")

        used_cols = [c for c in X_train.columns if c != "obs_id"]
        X_train_np = X_train[used_cols].to_numpy().reshape(-1, 100, len(used_cols))
        X_test_np = X_test[used_cols].to_numpy().reshape(-1, 100, len(used_cols))
        y_np = y_train["eqt_code_cat"].to_numpy()

        print(f"Loaded preprocessed files in {time.time() - t:.2f} s")
        print(f"Pipeline completed in {time.time() - t0:.2f} s\n")

        if verbose:
            print(f"X_train: {X_train_np.shape}, X_test: {X_test_np.shape}")
            print(f"Number of features: {len(used_cols)}")

        return X_train_np, y_np, X_test_np, used_cols

    # === 2. Raw data loading ===
    t = time.time()
    if dummy:
        X_train = pd.read_csv("/kaggle/input/optimization-cfm/small_X_train.csv", index_col=0)
        y_train = pd.read_csv("/kaggle/input/optimization-cfm/small_y_train.csv", index_col=0)
        X_test = pd.read_csv("/kaggle/input/optimization-cfm/small_X_test.csv", index_col=0)
    else:
        X_train = pd.read_csv("/kaggle/input/training-data-cfm/X_train_N1UvY30.csv")
        y_train = pd.read_csv("/kaggle/input/training-data-cfm/y_train_or6m3Ta.csv")
        X_test = pd.read_csv("/kaggle/input/training-data-cfm/X_test_m4HAPAP.csv")
    print(f"Loaded CSVs in {time.time() - t:.2f} s")

    # === 3. Categorical encoding ===
    t = time.time()
    cat_cols = ["venue", "order_id", "action", "side", "trade"]
    
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = enc.transform(X_test[cat_cols])
    print(f"Categorical encoding: {time.time() - t:.2f} s")

    # === 4. Intra-sequence clipping ===
    t = time.time()
    X_train = clip_sequence_outliers(X_train, n_std=threshold)
    print(f"Intra-sequence clipping: {time.time() - t:.2f} s")

    # === 5. Feature engineering ===
    t = time.time()

    # Fusion of both dataframe to assess that they follow the same preprocessing
    X_train["_is_train"] = 1
    X_test["_is_train"] = 0
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    X_all, num_cols = add_feature_engineering(df=X_all, **dict_feature_eng)

    print(f"Feature engineering: {time.time() - t:.2f} s")

    # === 6. Intra-sequence normalization ===
    t = time.time()
    num_cols = [col for col in X_all.columns if col not in cat_cols + ['_is_train', 'is_existing_order']]
    if normalize:
        X_all = normalize_per_sequence(X_all, "obs_id", num_cols)
    print(f"Intra-sequence normalization: {time.time() - t:.2f} s")
    
    X_train = X_all[X_all["_is_train"] == 1].drop(columns=["_is_train"])
    X_test = X_all[X_all["_is_train"] == 0].drop(columns=["_is_train"])

    # === 7. Optional return as DataFrames ===
    used_cols = [c for c in X_train.columns if c != "obs_id"]
    if return_pd:
        return X_train, X_test, y_train, used_cols

    # === 8. Final reshape ===
    t = time.time()
    X_train_np = X_train[used_cols].to_numpy().reshape(-1, 100, len(used_cols))
    X_test_np = X_test[used_cols].to_numpy().reshape(-1, 100, len(used_cols))
    y_np = y_train["eqt_code_cat"].to_numpy()
    print(f"Final reshape: {time.time() - t:.2f} s")

    print(f"\nPipeline completed in {time.time() - t0:.2f} s\n")
    if verbose:
        print(f"X_train: {X_train_np.shape}, X_test: {X_test_np.shape}")
        print(f"Number of features: {len(used_cols)}")
        print(f"Features: {used_cols}")

    return X_train_np, y_np, X_test_np, used_cols
