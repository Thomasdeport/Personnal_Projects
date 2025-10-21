import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.utils import to_categorical
import numpy as np



def log_transform(X):
    return np.sign(X) * np.log(np.abs(X) + 1)

def add_feature_engineering(df,graph):
    # Log features
    log_columns  = ["bid_size", "ask_size", "flux"]
    for feature in log_columns:
        df['log_'+feature] = np.sign(df[feature]) * np.log(np.abs(df[feature]) + 1)
    # Bid-ask ratio and imbalance
    df['bid_ask_ratio'] = df['bid_size'] / (df['ask_size'] + 1e-6)
    df['Imbalance'] = df['bid_size'] - df['ask_size']

    # Price and spread features
    df['price_change'] = df.groupby('obs_id')['price'].diff()
    df['cumulative_price_change'] = df.groupby('obs_id')['price_change'].cumsum()
    df['bid_ask_spread'] = df['ask'] - df['bid']
    df['relative_spread'] = df['bid_ask_spread'] / (df['price'] + 1e-6)

    # Price position relative to bid and ask
    df['price_bid_ratio'] = df['price'] / (df['bid'] + 1e-6)
    df['price_ask_ratio'] = df['price'] / (df['ask'] + 1e-6)
    df['log_flux_change'] = df.groupby('obs_id')['log_flux'].diff()
    
    df.fillna(0, inplace=True)
    if graph == True : 
        df= df[['venue','order_id','action','side','trade','price', 'bid', 'ask','log_bid_size', 'log_ask_size',
        'log_flux', 'bid_ask_ratio', 'Imbalance', 'price_change',
        'cumulative_price_change', 'bid_ask_spread', 'relative_spread',
        'price_bid_ratio', 'price_ask_ratio', 'log_flux_change']]
    else : 
        df= df[['venue','action','side','trade','price', 'bid', 'ask','log_bid_size', 'log_ask_size',
        'log_flux', 'bid_ask_ratio', 'Imbalance', 'price_change',
        'cumulative_price_change', 'bid_ask_spread', 'relative_spread',
        'price_bid_ratio', 'price_ask_ratio', 'log_flux_change']]
    return df

def remove_outliers_per_cat(df, group_col, cols, threshold=7):
    stats = df.groupby(group_col)[cols].agg(['mean', 'std'])
    stats.columns = ['_'.join(col).strip() for col in stats.columns]

    # Merge the calculated statistics back into the original DataFrame
    df = df.merge(stats, on=group_col, how='left')

    # Compute a boolean mask to identify rows with outliers
    outlier_flags = [
        (np.abs(df[col] - df[f"{col}_mean"]) > threshold * df[f"{col}_std"])
        for col in cols
    ]
    outlier_mask = np.logical_or.reduce(outlier_flags)  # Combine all masks

    # Identify `obs_id` values containing at least one outlier
    obs_with_outliers = df.loc[outlier_mask, 'obs_id'].unique()

    # Remove all rows with `obs_id` that contain outliers
    return df[~df['obs_id'].isin(obs_with_outliers)].drop(
        columns=[f"{col}_mean" for col in cols] + [f"{col}_std" for col in cols]
    )

       



def load_data(
    dummy=False, shuffle=True, normalize=True, filter=True,graph = False
):
    """features : {0: "venue", 1: "order_id", 2: "action", 3: "side", 4: "price", 5: "bid", 6: "ask", 7: "bid_size", 8: "ask_size", 9: "trade", 10: "flux"}"""
    # LOAD data : 
    if dummy:
        X_train = pd.read_csv("drive/MyDrive/data_folder/small_x_train.csv")
        y_train = pd.read_csv("drive/MyDrive/data_folder/small_y_train.csv")
        X_test = pd.read_csv("drive/MyDrive/data_folder/small_x_test.csv")
    else:
        X_train = pd.read_csv("drive/MyDrive/data_folder/X_train.csv")
        y_train = pd.read_csv("drive/MyDrive/data_folder/y_train.csv")
        X_test = pd.read_csv("drive/MyDrive/data_folder/X_test.csv")

    # LABELS
    categorical_columns = ["action", "side", "trade"]
    for label in categorical_columns:
        label_encoder = LabelEncoder()
        X_train[label] = label_encoder.fit_transform(X_train[label])
        X_test[label] = label_encoder.transform(X_test[label])
    # FILL Nas values 
    nb = X_train.isna().sum().sum()
    if nb != 0:
        print(f"Filling {nb} Nas in train")
    X_train = X_train.fillna(0)

    nb = X_test.isna().sum().sum()
    if nb != 0:
        print("Filling", nb, "NAs in test")
    X_test = X_test.fillna(0)

    # # REMOVE OUTLIERS global
    # features_to_filter = ["price", "bid", "ask", "bid_size", "ask_size", "flux"]
    # to_remove = []
    # for f in features_to_filter:
    #     to_remove.append(X_train[(np.abs(stats.zscore(X_train[f])) >= 8)]["obs_id"])
    # to_remove = pd.concat(to_remove)
    # X_train = X_train[~X_train["obs_id"].isin(to_remove)]
    # y_train = y_train[~y_train["obs_id"].isin(to_remove)]

    # REMOVE OUTLIERS per stock in the traning data
    X_train = X_train.merge(y_train, on = 'obs_id')
    group_col = 'eqt_code_cat'
    cols_to_check = ['price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux']
    if filter:
        remove_outliers_per_cat(df = X_train , group_col = group_col, cols = cols_to_check , threshold=7)

    y_train = X_train[["obs_id", 'eqt_code_cat']].reset_index()
    y_train = y_train.drop_duplicates(subset='obs_id', keep='first')

    X_train.drop(columns=[ 'eqt_code_cat'], inplace=True)

    # ADD some new features : 
    X_train = add_feature_engineering(X_train,graph)
    X_test = add_feature_engineering(X_test,graph)
    
    # NORMALIZE
    if normalize:
        normalize_groups= [
            ["log_bid_size", "log_ask_size"],
        ]
        for group in normalize_groups:
            mini = np.inf
            maxi = -np.inf
            for feature in group:
                mini = min(mini, X_train[feature].min())
                maxi = max(maxi, X_train[feature].max())
            for feature in group:
                X_train[feature] = (X_train[feature] - mini) / (maxi - mini)
                X_test[feature] = (X_test[feature] - mini) / (maxi - mini)

        normalize_groups_standard = []
        for group in normalize_groups_standard:
            mean = 0
            std = 0
            for feature in group:
                mean += X_train[feature].mean()
                std += X_train[feature].std()
            mean /= len(group)
            std /= len(group)
            for feature in group:
                X_train[feature] = (X_train[feature] - mean) / std
                X_test[feature] = (X_test[feature] - mean) / std
    # NUMPY AND RESHAPE
    X_train = X_train.to_numpy()
    X_train = X_train.reshape(-1, 100, X_train.shape[1])
    X_test = X_test.to_numpy()
    X_test = X_test.reshape(-1, 100, X_test.shape[1])
    # Y
    
    y_train = to_categorical(y_train['eqt_code_cat'])
    # SHUFFLE
    if shuffle:
        shuffled_index = np.random.permutation(X_train.shape[0])
    else:
        shuffled_index = np.arange(X_train.shape[0])
    return X_train[shuffled_index, :, :], y_train[shuffled_index], X_test

    
    