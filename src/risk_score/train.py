import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from risk_score.data import _random_oversample


def run_training(
    X_train_enc, X_test_enc, y_train_enc, y_test_enc,
    feature_names_encoded, df_categorized, config: dict,
) -> tuple:
    mode = config["feature_selection"]["mode"]

    if mode == "manual":
        best_subsets = list(config["feature_selection"]["best_subsets"])
    elif mode == "auto":
        original_features = [c for c in df_categorized.columns if c != "y"]
        removal_history, _, _ = _backward_stepwise(
            X_train_enc, X_test_enc, y_train_enc, y_test_enc,
            feature_names_encoded, original_features,
            config["model"]["random_state"],
        )
        best_subsets = [f for f in original_features if f not in removal_history]
    else:
        raise ValueError(
            f"feature_selection.mode must be 'manual' or 'auto', got '{mode}'"
        )

    X_best = df_categorized[best_subsets]
    y_all = df_categorized["y"]

    label_encoder = LabelEncoder()
    y_encoded_all = label_encoder.fit_transform(y_all)

    onehot_best = OneHotEncoder(sparse_output=False)
    onehot_best.fit(X_best)

    X_train_raw2, X_test_raw2, y_train_raw2, y_test_raw2 = train_test_split(
        X_best, y_all,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    X_train_bal, y_train_bal = _random_oversample(
        X_train_raw2, y_train_raw2, random_state=config["data"]["random_state"]
    )

    X_train_enc2 = onehot_best.transform(X_train_bal)
    y_train_enc2 = label_encoder.transform(y_train_bal)
    X_test_enc2 = onehot_best.transform(X_test_raw2)
    y_test_enc2 = label_encoder.transform(y_test_raw2)

    model = LogisticRegression(
        random_state=config["model"]["random_state"],
        max_iter=config["model"]["max_iter"],
    )
    model.fit(X_train_enc2, y_train_enc2)

    y_pred = model.predict(X_test_enc2)
    y_pred_prob = model.predict_proba(X_test_enc2)[:, 1]
    metrics = {
        "auc": float(roc_auc_score(y_test_enc2, y_pred_prob)),
        "confusion_matrix": confusion_matrix(y_test_enc2, y_pred).tolist(),
    }

    return (
        model, best_subsets, onehot_best,
        X_train_enc2, X_test_enc2,
        y_train_enc2, y_test_enc2,
        metrics,
    )


def _get_feature_groups(feature_names_encoded, original_features):
    return {
        feat: [i for i, name in enumerate(feature_names_encoded)
               if name.startswith(f"{feat}_")]
        for feat in original_features
    }


def _calculate_rss(model, X, y):
    return mean_squared_error(y, model.predict_proba(X)[:, 1])


def _calculate_aic(model, X, y):
    k = X.shape[1] + 1
    ll = -log_loss(y, model.predict_proba(X), normalize=False)
    return 2 * k - 2 * ll


def _backward_stepwise(
    X_train, X_test, y_train, y_test,
    feature_names_encoded, original_features, random_state=42,
):
    groups = _get_feature_groups(feature_names_encoded, original_features)
    current = original_features.copy()
    base_model = LogisticRegression(random_state=random_state, max_iter=100)

    removal_history, rss_history, aic_history = [], [], []

    all_idx = [i for f in current for i in groups[f]]
    m = clone(base_model)
    m.fit(X_train[:, all_idx], y_train)
    rss_history.append(_calculate_rss(m, X_test[:, all_idx], y_test))
    aic_history.append(_calculate_aic(m, X_test[:, all_idx], y_test))

    while len(current) > 1:
        rss_values = []
        for feat in current:
            remaining = [f for f in current if f != feat]
            idx = [i for f in remaining for i in groups[f]]
            m = clone(base_model)
            m.fit(X_train[:, idx], y_train)
            rss_values.append(_calculate_rss(m, X_test[:, idx], y_test))

        best_idx = int(np.argmin(rss_values))
        removed = current[best_idx]
        current.remove(removed)
        removal_history.append(removed)

        remaining_idx = [i for f in current for i in groups[f]]
        m_best = clone(base_model)
        m_best.fit(X_train[:, remaining_idx], y_train)
        rss_history.append(_calculate_rss(m_best, X_test[:, remaining_idx], y_test))
        aic_history.append(_calculate_aic(m_best, X_test[:, remaining_idx], y_test))

    return removal_history, rss_history, aic_history
