# Pré-requis : pip install lightgbm shap scikit-learn pandas numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.inspection import permutation_importance
import shap



# -------------------------
# 1) Agrégation des séquences
# -------------------------
def aggregate_sequences(X_seq, agg_funcs=None):
    """
    X_seq: numpy array shape (n_samples, n_steps, n_features)
    retourne DataFrame shape (n_samples, n_features * n_aggs)
    """
    if agg_funcs is None:
        agg_funcs = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'last': lambda x: x[:, -1],
            'median': np.median,
            'skew': lambda x: pd.DataFrame(x).skew(axis=1).values,
            'pct25': lambda x: np.percentile(x, 25, axis=1),
            'pct75': lambda x: np.percentile(x, 75, axis=1),
            'autocorr1': lambda x: np.array([np.corrcoef(ts[:-1], ts[1:])[0,1] if np.std(ts)>0 else 0.0 for ts in x])
        }

    n_samples, n_steps, n_features = X_seq.shape
    features = []
    names = []
    for f in range(n_features):
        col = X_seq[:, :, f]  # shape (n_samples, n_steps)
        for name, fn in agg_funcs.items():
            try:
                vals = fn(col)  # should return (n_samples,)
            except Exception:
                # fallback per sample
                vals = np.array([fn(col[i]) for i in range(n_samples)])
            features.append(np.asarray(vals).reshape(n_samples, 1))
            names.append(f"feat{f}_{name}")
    X_agg = np.hstack(features)
    return pd.DataFrame(X_agg, columns=names)


# -------------------------
# 2) Entraîner LightGBM / évaluer
# -------------------------

def train_lgb_get_importance(X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = []
    importances = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'metric': 'multi_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbosity': -1,
            'seed': random_state
        }
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
        )
        models.append(model)
        importances.append(model.feature_importance(importance_type='gain'))
    # Calcul de l'importance moyenne des caractéristiques
    imp_mean = np.mean(importances, axis=0)
    feat_imp = pd.DataFrame({'feature': X.columns, 'gain_importance': imp_mean})
    feat_imp = feat_imp.sort_values('gain_importance', ascending=False).reset_index(drop=True)
    return models, feat_imp


# -------------------------
# 3) Permutation importance (sur un modèle entraîné)
# -------------------------
def permutation_importance_on_model(model, X_val, y_val, n_repeats=10, random_state=42):
    # wrapper predict proba -> predict class for sklearn inspector
    # LightGBM model lgb.Booster has predict -> probas
    y_pred = np.argmax(model.predict(X_val), axis=1)
    # For permutation_importance we need a sklearn-style estimator: we create a small wrapper
    class Wrapper:
        def __init__(self, booster):
            self.b = booster
        def predict(self, X):
            return np.argmax(self.b.predict(X), axis=1)
    wrapper = Wrapper(model)
    res = permutation_importance(wrapper, X_val, y_val, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    perm_df = pd.DataFrame({'feature': X_val.columns, 'perm_importance_mean': res.importances_mean})
    perm_df = perm_df.sort_values('perm_importance_mean', ascending=False).reset_index(drop=True)
    return perm_df


# -------------------------
# 4) SHAP (TreeExplainer)
# -------------------------
def shap_importance(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)  # list over classes
    # Aggregate absolute shap across classes and samples
    if isinstance(shap_vals, list):
        # multi-class -> shap_vals is list of arrays (n_samples, n_features)
        abs_mean = np.mean([np.abs(s).mean(axis=0) for s in shap_vals], axis=0)
    else:
        abs_mean = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({'feature': X_sample.columns, 'shap_mean_abs': abs_mean})
    return shap_df.sort_values('shap_mean_abs', ascending=False).reset_index(drop=True)

