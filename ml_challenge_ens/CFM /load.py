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
    x = np.array(x, dtype=np.float32)
    return np.sign(x) * np.log1p(np.abs(x))


def remove_outliers_per_cat(df, group_col, cols, threshold=7):
    """Supprime les séries entières contenant au moins un outlier."""
    stats = df.groupby(group_col)[cols].agg(['mean', 'std'])
    stats.columns = ['_'.join(col).strip() for col in stats.columns]

    df = df.merge(stats, on=group_col, how='left')

    outlier_flags = [
        (np.abs(df[col] - df[f"{col}_mean"]) > threshold * df[f"{col}_std"])
        for col in cols
    ]
    outlier_mask = np.logical_or.reduce(outlier_flags)
    obs_with_outliers = df.loc[outlier_mask, 'obs_id'].unique()

    return df[~df['obs_id'].isin(obs_with_outliers)].drop(
        columns=[f"{col}_mean" for col in cols] + [f"{col}_std" for col in cols]
    )


def normalize_features(X_all, normalize=True):
    if not normalize:
        return X_all

    cat_cols = ["venue", "order_id", "action", "side", "trade", "_is_train", "obs_id", "order_event_count"]
    num_cols = [c for c in X_all.columns if c not in cat_cols]

    train_mask = X_all["_is_train"] == 1
    means = X_all.loc[train_mask, num_cols].mean()
    stds = X_all.loc[train_mask, num_cols].std().replace(0, 1e-6)

    X_all[num_cols] = ((X_all[num_cols] - means) / stds).clip(-10, 10)
    return X_all

# ======================
# FEATURE ENGINEERING
# ======================


import numpy as np
import pandas as pd
import gc
from scipy.stats import skew

EPS = 1e-6

def log_transform(x):
    """Signed log transform (robust to zeros and negatives)."""
    x = np.sign(x) * np.log1p(np.abs(x))
    return x.astype("float32")


import numpy as np
import pandas as pd
import gc
from scipy.stats import skew

EPS = 1e-6

def log_transform(x):
    """Signed log transform (robust to zeros and negatives)."""
    x = np.sign(x) * np.log1p(np.abs(x))
    return x.astype("float32")


def add_feature_engineering(
    df: pd.DataFrame,
    test: bool = False,
    use_log: bool = True,
    corr_threshold: float = None,
):
    """
    === add_feature_engineering (hybrid version) ===
    Combine robust microstructure features with stabilized raw numerical signals.
    - Causal (no leakage)
    - Log-stabilized heavy tails
    - Preserves market level context (log_price, log_flux, etc.)
    """

    df = df.copy().fillna(0)
    df = df.astype({
        "price": "float32", "bid": "float32", "ask": "float32",
        "bid_size": "float32", "ask_size": "float32", "flux": "float32"
    }, errors="ignore")

    g = df.groupby("obs_id", sort=False)
    go = df.groupby(["obs_id", "order_id"], sort=False)
    feats = []
    base_cols = ["obs_id", "order_id", "venue", "action", "side", "trade", "_is_train"]

    # === 1. STRUCTURE ===
    df["mid_price"] = ((df["bid"] + df["ask"]) / 2).astype("float32")
    df["bid_ask_spread"] = (df["ask"] - df["bid"]).astype("float32")
    df["relative_spread"] = (df["bid_ask_spread"] / (df["mid_price"] + EPS)).astype("float32")
    df["spread_change"] = g["bid_ask_spread"].diff().fillna(0).astype("float32")
    feats += ["bid_ask_spread", "relative_spread", "spread_change"]

    # === 2. ORDER FLOW IMBALANCE ===
    d_bid = g["bid"].diff().fillna(0)
    d_ask = g["ask"].diff().fillna(0)
    d_qb = g["bid_size"].diff().fillna(0)
    d_qa = g["ask_size"].diff().fillna(0)
    df["ofi"] = ((d_qb * (d_bid >= 0)) - (d_qa * (d_ask <= 0))).astype("float32")
    feats += ["ofi"]

    del d_bid, d_ask, d_qb, d_qa
    gc.collect()

    # === 3. PRICE DYNAMICS ===
    df["price_change"] = g["price"].diff().fillna(0).astype("float32")
    df["rolling_volatility"] = g["price_change"].transform(
        lambda x: x.rolling(5, min_periods=2).std().fillna(0)
    ).astype("float32")
    feats += ["price_change", "rolling_volatility"]

    # === 4. MOMENTUM ===
    df["momentum_5"] = g["price"].diff(5).fillna(0).astype("float32")
    df["momentum_20"] = g["price"].diff(20).fillna(0).astype("float32")
    feats += ["momentum_5", "momentum_20"]

    # === 5. FLUX ===
    df["flux_change"] = g["flux"].diff().fillna(0).astype("float32")
    feats += ["flux_change"]

    # === 6. ORDER DYNAMICS ===
    cnt_in_order = go.cumcount()
    max_in_order = go.size().rename("order_event_count")
    df = df.join(max_in_order, on=["obs_id", "order_id"])
    df["order_age_norm"] = (
        cnt_in_order / (df["order_event_count"].replace(0, 1) - 1 + EPS)
    ).fillna(0).astype("float32")
    df["order_flux_change"] = go["flux"].diff().fillna(0).astype("float32")
    df["is_new_order"] = (cnt_in_order == 0).astype("int8")
    feats += ["order_event_count", "order_age_norm", "order_flux_change", "is_new_order"]

    # === 7. IMBALANCE / RATIO ===
    df["imbalance"] = (df["bid_size"] - df["ask_size"]).astype("float32")
    df["rel_bid_size"] = (df["bid_size"] / (df["bid_size"] + df["ask_size"] + EPS)).astype("float32")
    feats += ["imbalance", "rel_bid_size"]

    # === 8. LOG STABILIZED RAW LEVELS ===
    # Réintroduit le niveau de prix et de flux (contextuel)
    df["log_bid_size"] = log_transform(df["bid_size"])
    df["log_ask_size"] = log_transform(df["ask_size"])
    df["log_flux"] = log_transform(df["flux"])
    feats += ["log_bid_size", "log_ask_size", "log_flux"]

    # === 9. CONDITIONAL LOG STABILIZATION ===
    if use_log:
        log_candidates = [
            "flux_change", "rolling_volatility", "imbalance"
        ]
        for col in log_candidates:
            if col in df.columns:
                col_data = df[col].replace([np.inf, -np.inf], 0).fillna(0)
                if abs(skew(col_data)) > 3:
                    df[f"log_{col}"] = log_transform(col_data)
                    feats.append(f"log_{col}")

    # === CLEANUP ===
    df[feats] = df[feats].replace([np.inf, -np.inf], 0).fillna(0)
    df.drop(columns=["price", "bid", "ask"], inplace=True, errors="ignore")

    # === OPTIONAL CORRELATION PRUNING ===
    if corr_threshold and len(feats) > 1:
        corr = df[feats].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
        if to_drop:
            print(f"Dropped (corr > {corr_threshold}): {to_drop}")
            df.drop(columns=to_drop, inplace=True)
            feats = [f for f in feats if f not in to_drop]

    # === FINAL OUTPUT ===
    final_cols = base_cols + feats
    final_cols = list(dict.fromkeys(final_cols))  # keep order, drop duplicates
    gc.collect()
    return df[final_cols], final_cols


EPS = 1e-6

def log_transform(x):
    """Signed log transform (robust to zeros and negatives)."""
    x = np.sign(x) * np.log1p(np.abs(x))
    return x.astype("float32")

def add_feature_eng_reduced(
    df: pd.DataFrame,
    test: bool = False,
    use_log: bool = True,
):
    """
    === add_feature_eng_reduced ===
    Version compacte et robuste du feature engineering :
    - Ne conserve que les signaux stables et informatifs
    - Invariante au drift de prix
    - Faible variance, idéale pour modèles séquentiels
    """

    df = df.copy().fillna(0)
    df = df.astype({
        "price": "float32", "bid": "float32", "ask": "float32",
        "bid_size": "float32", "ask_size": "float32", "flux": "float32"
    }, errors="ignore")

    g = df.groupby("obs_id", sort=False)
    feats = []
    base_cols = ["obs_id", "order_id", "venue", "action", "side", "trade", "_is_train"]

    # === 1. LOG-STABILIZED SIZE & FLUX ===
    df["log_bid_size"] = log_transform(df["bid_size"])
    df["log_ask_size"] = log_transform(df["ask_size"])
    df["log_flux"] = log_transform(df["flux"])
    feats += ["log_bid_size", "log_ask_size", "log_flux"]

    # === 2. STRUCTURE ===
    df["relative_spread"] = ((df["ask"] - df["bid"]) / (df["price"] + EPS)).astype("float32")
    #feats.append("relative_spread")

    # === 3. ORDER FLOW IMBALANCE ===
    d_bid = g["bid"].diff().fillna(0)
    d_ask = g["ask"].diff().fillna(0)
    d_qb = g["bid_size"].diff().fillna(0)
    d_qa = g["ask_size"].diff().fillna(0)
    df["ofi"] = ((d_qb * (d_bid >= 0)) - (d_qa * (d_ask <= 0))).astype("float32")
    feats.append("ofi")

     # === 4. MOMENTUM ===
    df["momentum_5"] = g["price"].diff(5).fillna(0).astype("float32")
    feats.append("momentum_5")
    del d_bid, d_ask, d_qb, d_qa
    gc.collect()

    # === 5. PRICE DYNAMICS ===
    df["price_change"] = g["price"].diff().fillna(0).astype("float32")
    df["rolling_volatility"] = g["price_change"].transform(
        lambda x: x.rolling(5, min_periods=2).std().fillna(0)
    ).astype("float32")
    feats.append("rolling_volatility")

    # === CLEANUP ===
    df[feats] = df[feats].replace([np.inf, -np.inf], 0).fillna(0)

    # === FINAL OUTPUT ===
    final_cols = base_cols + feats
    final_cols = list(dict.fromkeys(final_cols))  # garde l'ordre sans doublon
    gc.collect()

    return df[final_cols], final_cols



def add_feature_eng_super_reduced(
    df: pd.DataFrame,
    test: bool = False,
    use_log: bool = True,
):
    """
    === add_feature_eng_reduced ===
    Version compacte et robuste du feature engineering :
    - Ne conserve que les signaux stables et informatifs
    - Invariante au drift de prix
    - Faible variance, idéale pour modèles séquentiels
    """

    df = df.copy().fillna(0)
    df = df.astype({
        "price": "float32", "bid": "float32", "ask": "float32",
        "bid_size": "float32", "ask_size": "float32", "flux": "float32"
    }, errors="ignore")

    g = df.groupby("obs_id", sort=False)
    feats = []
    base_cols = ["obs_id", "order_id", "venue", "action", "side", "trade", "_is_train"]

    # === 1. LOG-STABILIZED SIZE & FLUX ===
    df["log_bid_size"] = log_transform(df["bid_size"])
    df["log_ask_size"] = log_transform(df["ask_size"])
    df["log_flux"] = log_transform(df["flux"])
    feats += ["log_bid_size", "log_ask_size", "log_flux"]

    df["price_change"] = g["price"].diff().fillna(0).astype("float32")
    df["rolling_volatility"] = g["price_change"].transform(
        lambda x: x.rolling(5, min_periods=2).std().fillna(0)
    ).astype("float32")
    feats.append("rolling_volatility")

    
    # === CLEANUP ===
    df[feats] = df[feats].replace([np.inf, -np.inf], 0).fillna(0)

    # === FINAL OUTPUT ===
    final_cols = base_cols + feats
    final_cols = list(dict.fromkeys(final_cols))  # garde l'ordre sans doublon
    gc.collect()

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
    threshold = 6,
    feature_reduced= 2 
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
    if dummy==1:
        X_train = pd.read_csv("/kaggle/input/optimization-cfm/small_X_train.csv", index_col=0)
        y_train = pd.read_csv("/kaggle/input/optimization-cfm/small_y_train.csv", index_col=0)
        X_test = pd.read_csv("/kaggle/input/optimization-cfm/small_X_test.csv", index_col=0)
    elif dummy==2:
        X_train = pd.read_csv("/kaggle/input/small-dataset/small_X_train.csv", index_col=0)
        y_train = pd.read_csv("/kaggle/input/small-dataset/small_y_train.csv", index_col=0)
        X_test = pd.read_csv("/kaggle/input/small-dataset/small_X_test.csv", index_col=0)
    elif dummy==3:
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


    # === 3.5 . Outlier Elimination (1)===
    t = time.time()
    X_train = X_train.merge(y_train, on='obs_id')
    if filter:
        X_train = remove_outliers_per_cat(
            df=X_train,
            group_col='eqt_code_cat',
            cols=['price', 'bid', 'ask'],
            threshold=threshold
        )

    y_train = X_train[["obs_id", 'eqt_code_cat']].drop_duplicates(subset='obs_id').reset_index(drop=True)
    X_train.drop(columns=['eqt_code_cat'], inplace=True)
    print(f"Outlier Elimination: {time.time() - t:.2f} s")

    # === 4. Feature engineering ===
    t = time.time()

    # Fusion of both dataframe to assess that they follow the same preprocessing
    X_train["_is_train"] = 1
    X_test["_is_train"] = 0
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    if feature_reduced ==1 :
        X_all, num_cols = add_feature_eng_reduced(df=X_all)
    elif feature_reduced ==2: 
        X_all, num_cols = add_feature_eng_super_reduced(df=X_all)
    else: 
        X_all, num_cols = add_feature_engineering(df=X_all, **dict_feature_eng)

    print(f"Feature engineering: {time.time() - t:.2f} s")

    # === 5. Normalization (simple global z-score, train-only) ===
    t = time.time()
    
    X_all = normalize_features(X_all, normalize=normalize)
    
    print(f"Global z-score normalization: {time.time() - t:.2f} s")
    
    X_train = X_all[X_all["_is_train"] == 1].drop(columns=["_is_train"])
    X_test = X_all[X_all["_is_train"] == 0].drop(columns=["_is_train"])


    # === 6. Outlier Elimination (2) ===
    t = time.time()
    X_train = X_train.merge(y_train, on='obs_id')
    if filter:
        X_train = remove_outliers_per_cat(
            df=X_train,
            group_col='eqt_code_cat',
            cols=['log_bid_size', 'log_ask_size', 'log_flux'],
            threshold=threshold
        )

    y_train = X_train[["obs_id", 'eqt_code_cat']].drop_duplicates(subset='obs_id').reset_index(drop=True)
    X_train.drop(columns=['eqt_code_cat'], inplace=True)
    print(f"Outlier Elimination: {time.time() - t:.2f} s")

    
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
