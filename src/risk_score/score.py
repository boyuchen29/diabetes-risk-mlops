import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_risk_scores(
    model, df_categorized, best_subsets, onehot, y_encoded_all, config: dict
) -> tuple:
    coefficients = model.coef_[0]
    scores = _calculate_scores(coefficients, df_categorized[best_subsets], onehot)

    X_scores = df_categorized[best_subsets].copy()
    for feature in scores:
        X_scores[feature] = X_scores[feature].map(scores[feature])
    X_scores = X_scores.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_scores)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_scaled, y_encoded_all,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    retrained = LogisticRegression(
        random_state=config["model"]["random_state"],
        max_iter=config["model"]["max_iter"],
    )
    retrained.fit(X_train_s, y_train_s)
    beta_i = retrained.coef_[0]
    weights = beta_i / np.sum(beta_i)

    RS_i = np.sum(X_scores.values * weights / 10, axis=1)
    R = X_scores.copy()
    R["RS"] = RS_i
    R["y"] = y_encoded_all

    return scores, weights, R


def _calculate_scores(coefficients, X_categorized, onehot_encoder) -> dict:
    feature_names_enc = onehot_encoder.get_feature_names_out(X_categorized.columns)
    scores = {}

    for feature in X_categorized.columns:
        if isinstance(X_categorized[feature].dtype, pd.CategoricalDtype):
            levels = list(X_categorized[feature].cat.categories)
        else:
            levels = list(X_categorized[feature].unique())

        coeffs = []
        for level in levels:
            name = f"{feature}_{level}"
            if name in feature_names_enc:
                idx = int(np.where(feature_names_enc == name)[0][0])
                coeffs.append(coefficients[idx])

        if not coeffs:
            scores[feature] = {level: 0.0 for level in levels}
            continue

        min_c, max_c = min(coeffs), max(coeffs)
        scores[feature] = {
            level: (float(np.clip(1000.0 * (c - min_c) / (max_c - min_c), 0.0, 1000.0)) if max_c != min_c else 0.0)
            for level, c in zip(levels, coeffs)
        }

    return scores
