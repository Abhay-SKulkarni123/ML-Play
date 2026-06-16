import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


CLASSIFIERS = {
    "logistic_regression":    lambda p: LogisticRegression(max_iter=1000, **p),
    "random_forest":          lambda p: RandomForestClassifier(n_estimators=100, random_state=42, **p),
    "xgboost":                lambda p: XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0, **p),
    "lightgbm":               lambda p: LGBMClassifier(random_state=42, verbose=-1, **p),
    "svm":                    lambda p: SVC(probability=True, **p),
    "knn":                    lambda p: KNeighborsClassifier(**p),
    "naive_bayes":            lambda p: GaussianNB(),
    "decision_tree":          lambda p: DecisionTreeClassifier(random_state=42, **p),
    "gradient_boosting":      lambda p: GradientBoostingClassifier(random_state=42, **p),
}

REGRESSORS = {
    "ridge":                  lambda p: Ridge(**p),
    "lasso":                  lambda p: Lasso(**p),
    "elasticnet":             lambda p: ElasticNet(**p),
    "random_forest":          lambda p: RandomForestRegressor(n_estimators=100, random_state=42, **p),
    "xgboost":                lambda p: XGBRegressor(random_state=42, verbosity=0, **p),
    "lightgbm":               lambda p: LGBMRegressor(random_state=42, verbose=-1, **p),
    "svr":                    lambda p: SVR(**p),
    "knn":                    lambda p: KNeighborsRegressor(**p),
}


def train_model(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    model_name: str,
    model_params: dict,
    test_size: float = 0.2,
) -> dict:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # keep only numeric for now (encoding must be done before this step)
    X = X.select_dtypes(include="number")

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    if task == "classification":
        if model_name not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier: {model_name}")
        model = CLASSIFIERS[model_name](model_params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        if model_name not in REGRESSORS:
            raise ValueError(f"Unknown regressor: {model_name}")
        model = REGRESSORS[model_name](model_params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {}

    if task == "classification":
        metrics["accuracy"]    = round(accuracy_score(y_test, y_pred), 4)
        metrics["f1_weighted"] = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
        metrics["f1_macro"]    = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        try:
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
            else:
                metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob, multi_class="ovr"), 4)
        except Exception:
            metrics["roc_auc"] = None
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")

    else:
        metrics["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        metrics["mae"]  = round(mean_absolute_error(y_test, y_pred), 4)
        metrics["r2"]   = round(r2_score(y_test, y_pred), 4)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_std"]  = round(float(cv_scores.std()), 4)

    # feature importance
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        fi = dict(zip(X.columns, model.feature_importances_))
        feature_importance = {k: round(float(v), 6) for k, v in sorted(fi.items(), key=lambda x: -x[1])}
    elif hasattr(model, "coef_"):
        coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        fi = dict(zip(X.columns, np.abs(coef)))
        feature_importance = {k: round(float(v), 6) for k, v in sorted(fi.items(), key=lambda x: -x[1])}

    return {
        "model": model_name,
        "params": model_params,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": len(X.columns),
        "fitted_model": model,  # kept in memory for SHAP
        "X_test": X_test,
        "y_test": y_test,
    }