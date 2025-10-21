import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder


def add_feature_engineering(
    df: pd.DataFrame,
    test: bool = False,
    list_test=None,
    use_microstructure=True,
    use_imbalance=True,
    use_momentum=True,
    use_liquidity=True,
    use_directional=True,
    use_time=True,
    use_stat_transforms=True,
    momentum_windows=[5, 20, 50],
    corr_threshold=0.85
):
    """
    Feature engineering optimisé pour classification sur gros datasets.
    Supprime les opérations lentes mais conserve des features puissantes.
    """

    if list_test is None:
        list_test = []

    df = df.copy()
    df.fillna(0, inplace=True)
    engineered_features = []

    # === Microstructure ===
    if use_microstructure and set(['bid', 'ask', 'price']).issubset(df.columns):
        df['spread'] = df['ask'] - df['bid']
        df['rel_spread'] = df['spread'] / (df['price'] + 1e-6)
        df['mid_price'] = (df['ask'] + df['bid']) / 2
        df['mid_vs_price'] = df['price'] - df['mid_price']
        engineered_features += ['spread', 'rel_spread', 'mid_price', 'mid_vs_price']

    # === Imbalance ===
    if use_imbalance and set(['bid_size', 'ask_size']).issubset(df.columns):
        df['imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-6)
        df['depth'] = df['bid_size'] + df['ask_size']
        engineered_features += ['imbalance', 'depth']

    # === Momentum simple ===
    if use_momentum and 'price' in df.columns:
        for w in momentum_windows:
            df[f'momentum_{w}'] = df['price'].diff(w).fillna(0)
            df[f'roc_{w}'] = df['price'].pct_change(w).fillna(0)
            engineered_features += [f'momentum_{w}', f'roc_{w}']

        # Delta et signes pour classification
        df['delta'] = df['price'].diff().fillna(0)
        df['delta_sign'] = np.sign(df['delta'])
        df['abs_delta'] = df['delta'].abs()
        engineered_features += ['delta', 'delta_sign', 'abs_delta']
 
    # === Liquidity simple ===
    if use_liquidity:
        if 'flux' in df.columns:
            df['flux_log'] = log_transform(df['flux'])
            df['flux_diff'] = df['flux'].diff().fillna(0)
            engineered_features += ['flux_log', 'flux_diff']

        if 'price' in df.columns and 'flux' in df.columns:
            df['turnover'] = df['price'] * df['flux']
            engineered_features.append('turnover')

    # === Directional flow ===
    if use_directional and 'price' in df.columns:
        df['abs_momentum'] = df['delta'].abs()
        df['momentum_sign'] = np.sign(df['delta'])
        engineered_features += ['abs_momentum', 'momentum_sign']

    # === Time features ===
    if use_time:
        df['event_number'] = df.groupby('obs_id').cumcount()
        df['progress'] = df['event_number'] / (df.groupby('obs_id')['event_number'].transform('max')+1e-6)
        engineered_features += ['event_number', 'progress']

    # === Statistiques simples ===
    if use_stat_transforms:
        for col in engineered_features.copy():
            if col in df.columns:
                df[f'{col}_log'] = log_transform(df[col])
                df[f'{col}_z'] = (df[col] - df[col].mean())/(df[col].std()+1e-6)
                engineered_features += [f'{col}_log', f'{col}_z']
                if test:
                    list_test.append(f'{col}_log')

    # === Nettoyage final ===
    df.fillna(0, inplace=True)

    # Supprimer features trop corrélées (échantillon pour accélérer)
    if corr_threshold and engineered_features:
        corr_matrix = df[engineered_features].sample(min(len(df), 100000)).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        if to_drop:
            print(f"Features supprimées (corr > {corr_threshold}): {to_drop}")
            df.drop(columns=to_drop, inplace=True)
            engineered_features = [f for f in engineered_features if f not in to_drop]

    return df, engineered_features, list_test


def log_transform(X):
    """Applique une transformation log intelligente."""
    X = np.array(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    X = np.nan_to_num(X, nan=0.0)
    return np.sign(X) * np.log(np.abs(X) + 1)


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

def normalize_features(X_train, X_test, features):
    """Normalisation min-max puis standardisation."""
    for feature in features:
        min_val = X_train[feature].min()
        max_val = X_train[feature].max()
        if max_val - min_val > 0:
            X_train[feature] = (X_train[feature] - min_val) / (max_val - min_val)
            X_test[feature] = (X_test[feature] - min_val) / (max_val - min_val)

        mean_val = X_train[feature].mean()
        std_val = X_train[feature].std()
        if std_val > 0:
            X_train[feature] = (X_train[feature] - mean_val) / std_val
            X_test[feature] = (X_test[feature] - mean_val) / std_val

    return X_train, X_test


def load_data(dummy=False, normalize=True, filter=True, threshold = 7, dict_feature_eng = {}, print_shape = True):
    """Charge, traite et renvoie les données."""
    if dummy:
        X_train = pd.read_csv("/kaggle/input/optimization-cfm/small_X_train.csv", index_col=0)
        y_train = pd.read_csv("/kaggle/input/optimization-cfm/small_y_train.csv", index_col=0)
        X_test = pd.read_csv("/kaggle/input/optimization-cfm/small_X_test.csv", index_col=0)
        
    else:
        X_train = pd.read_csv("/kaggle/input/training-data-cfm/X_train_N1UvY30.csv")
        y_train = pd.read_csv("/kaggle/input/training-data-cfm/y_train_or6m3Ta.csv")
        X_test = pd.read_csv("/kaggle/input/training-data-cfm/X_test_m4HAPAP.csv")

    # Encodage
    categorical_columns = ["action", "side", "trade"]
    for col in categorical_columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # Outliers
    X_train = X_train.merge(y_train, on='obs_id')
    if filter:
        X_train = remove_outliers_per_cat(
            df=X_train,
            group_col='eqt_code_cat',
            cols=['price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux'],
            threshold=threshold
        )

    y_train = X_train[["obs_id", 'eqt_code_cat']].drop_duplicates(subset='obs_id').reset_index(drop=True)
    X_train.drop(columns=['eqt_code_cat'], inplace=True)

    # Feature engineering
    X_train, num_cols, list_test = add_feature_engineering(df = X_train, **dict_feature_eng)
    X_test, _, _ = add_feature_engineering(df = X_test, test=True, list_test=list_test, **dict_feature_eng)

    # Normalisation
    if normalize:
        X_train, X_test = normalize_features(X_train, X_test, num_cols)

    # Reshape
    X_train = X_train.to_numpy().reshape(-1, 100, X_train.shape[1])
    X_test = X_test.to_numpy().reshape(-1, 100, X_test.shape[1])

    y_train = y_train['eqt_code_cat'].to_numpy()
    if print_shape: 
        print(f"X_train shape : {X_train.shape}, y_train shape :{y_train.shape}, X_test shape :{X_test.shape}")

    return X_train, y_train, X_test
