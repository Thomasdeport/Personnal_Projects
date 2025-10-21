import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew


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


def add_feature_engineering(
    df: pd.DataFrame,
    test: bool = False,
    list_test=None,
    use_microstructure=True,
    use_imbalance=True,
    use_price_dynamics=True,
    use_momentum=True,
    momentum_windows=[5, 20],
    use_flux=True,
    use_directional=True,
    use_time=True,
    use_log=True,
    corr_threshold=0.7
):
    """
    Ajoute des features de microstructure/finance de marché de manière modulable.
    Supprime les colonnes trop corrélées et affiche celles supprimées.
    """
    if list_test is None:
        list_test = []

    base_cols = ['venue', 'order_id', 'action', 'side', 'trade']
    engineered_features = []
    df = df.copy()
    df.fillna(0, inplace=True)

    # === Microstructure ===
    if use_microstructure and all(col in df.columns for col in ['ask', 'bid', 'price']):
        df['bid_ask_spread'] = df['ask'] - df['bid']
        df['relative_spread'] = df['bid_ask_spread'] / (df['price'] + 1e-6)
        engineered_features += ['bid_ask_spread', 'relative_spread']

    # === Imbalance et flux ===
    if use_imbalance and all(col in df.columns for col in ['bid_size', 'ask_size']):
        df['imbalance'] = df['bid_size'] - df['ask_size']
        df['ofi'] = (
            df['bid_size'] - df.groupby('obs_id')['bid_size'].shift(1).fillna(0)
        ) - (
            df['ask_size'] - df.groupby('obs_id')['ask_size'].shift(1).fillna(0)
        )
        engineered_features += ['imbalance', 'ofi']

    # === Price dynamics ===
    if use_price_dynamics and 'price' in df.columns:
        df['price_change'] = df.groupby('obs_id')['price'].diff().fillna(0)
        df['rolling_volatility'] = df.groupby('obs_id')['price'].transform(
            lambda x: x.rolling(5, min_periods=1).std().fillna(0)
        )
        engineered_features += ['price_change', 'rolling_volatility']

    # === Momentum ===
    if use_momentum and 'price' in df.columns:
        for window in momentum_windows:
            col = f'momentum_{window}'
            df[col] = df.groupby('obs_id')['price'].diff(window).fillna(0)
            engineered_features.append(col)

    # === Flux ===
    if use_flux and 'flux' in df.columns:
        df['flux_change'] = df.groupby('obs_id')['flux'].diff().fillna(0)
        engineered_features.append('flux_change')

    # === Directionnel ===
    if use_directional and 'price_change' in df.columns:
        df['trend_consistency'] = df.groupby('obs_id')['price_change'].transform(
            lambda x: (x > 0).rolling(10, min_periods=1).mean().fillna(0)
        )
        df['abs_price_change'] = df['price_change'].abs()
        #df['trend_strength'] = df.groupby('obs_id')['abs_price_change'].transform(
        #    lambda x: x.rolling(10, min_periods=1).mean().fillna(0)
        #)
        engineered_features += ['trend_consistency', 
                                #'trend_strength'
                               ]

    # === Temps ===
    if use_time:
        df['event_number'] = df.groupby('obs_id').cumcount()
        df['time_proportion'] = df['event_number'] / 100.0
        engineered_features += ['time_proportion']

    # === Log transform ===
    if use_log:
        always_log = ['bid_size', 'ask_size', 'flux']
        conditionally_log = [
            'flux_change', 'price_change', 'imbalance', 'relative_spread',
            'rolling_volatility'
            #, 'trend_strength'
        ]

        for col in always_log:
            if col in df.columns:
                df[col] = log_transform(df[col])

        for col in conditionally_log:
            if col in df.columns:
                data = df[col].fillna(0)
                if abs(skew(data)) > 5 and not test:
                    df[col] = log_transform(data)
                    list_test.append(col)

        if test:
            for col in list_test:
                if col in df.columns:
                    df[col] = log_transform(df[col])

    # Nettoyage final
    df.fillna(0, inplace=True)

    # Retirer colonnes inutiles
    drop_cols = ['price', 'bid', 'ask']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Supprimer features trop corrélées
    if corr_threshold and engineered_features:
        corr_matrix = df[engineered_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        if to_drop:
            print(f"Features supprimées pour corrélation > {corr_threshold}: {to_drop}")
        df.drop(columns=to_drop, inplace=True)
        engineered_features = [f for f in engineered_features if f not in to_drop]

    final_cols = base_cols + engineered_features
    
    print(final_cols)
    return df[[col for col in final_cols if col in df.columns]], engineered_features, list_test


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
