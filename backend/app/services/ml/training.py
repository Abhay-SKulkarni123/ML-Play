import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


def _build_classifier(name: str, params: dict, class_weight: str | None = None):
    """
    Build classifier with safe defaults.
    Params from Optuna override defaults — no duplicate keyword args.
    """
    defaults = {
        "random_forest":     {"n_estimators": 100, "random_state": 42},
        "xgboost":           {"random_state": 42, "verbosity": 0, "eval_metric": "logloss"},
        "lightgbm":          {"random_state": 42, "verbose": -1},
        "logistic_regression":{"max_iter": 1000},
        "svm":               {"probability": True},
        "decision_tree":     {"random_state": 42},
        "gradient_boosting": {"random_state": 42},
        "knn":               {},
        "naive_bayes":       {},
        "adaboost":    {"n_estimators": 50, "random_state": 42},
        "catboost":    {"verbose": 0, "random_state": 42},
    }
    # User/Optuna params override defaults — no conflicts
    merged = {**defaults.get(name, {}), **params}

    constructors = {
        "logistic_regression":  lambda p: LogisticRegression(class_weight=class_weight, **p),
        "random_forest":        lambda p: RandomForestClassifier(class_weight=class_weight, **p),
        "xgboost":              lambda p: XGBClassifier(**p),
        "lightgbm":             lambda p: LGBMClassifier(class_weight=class_weight, **p),
        "svm":                  lambda p: SVC(class_weight=class_weight, **p),
        "knn":                  lambda p: KNeighborsClassifier(**p),
        "naive_bayes":          lambda p: GaussianNB(),
        "decision_tree":        lambda p: DecisionTreeClassifier(class_weight=class_weight, **p),
        "gradient_boosting":    lambda p: GradientBoostingClassifier(**p),
        "adaboost":             lambda p: __import__('sklearn.ensemble', fromlist=['AdaBoostClassifier']).AdaBoostClassifier(**p),
        "catboost":             lambda p: __import__('catboost', fromlist=['CatBoostClassifier']).CatBoostClassifier(**p),
    }
    if name not in constructors:
        raise ValueError(f"Unknown classifier: {name}")
    return constructors[name](merged)


def _build_regressor(name: str, params: dict):
    """
    Build regressor with safe defaults.
    Params from Optuna override defaults — no duplicate keyword args.
    """
    defaults = {
        "random_forest": {"n_estimators": 100, "random_state": 42},
        "xgboost":       {"random_state": 42, "verbosity": 0},
        "lightgbm":      {"random_state": 42, "verbose": -1},
        "ridge":         {},
        "lasso":         {},
        "elasticnet":    {},
        "svr":           {},
        "knn":           {},
    }
    merged = {**defaults.get(name, {}), **params}

    constructors = {
        "ridge":         lambda p: Ridge(**p),
        "lasso":         lambda p: Lasso(**p),
        "elasticnet":    lambda p: ElasticNet(**p),
        "random_forest": lambda p: RandomForestRegressor(**p),
        "xgboost":       lambda p: XGBRegressor(**p),
        "lightgbm":      lambda p: LGBMRegressor(**p),
        "svr":           lambda p: SVR(**p),
        "knn":           lambda p: KNeighborsRegressor(**p),
        "adaboost":    lambda p: __import__('sklearn.ensemble', fromlist=['AdaBoostClassifier']).AdaBoostClassifier(**p),
        "catboost":    lambda p: __import__('catboost', fromlist=['CatBoostClassifier']).CatBoostClassifier(**p),

    }
    if name not in constructors:
        raise ValueError(f"Unknown regressor: {name}")
    return constructors[name](merged)


def train_model(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    model_name: str,
    model_params: dict,
    test_size: float = 0.2,
    pipeline_state: dict = None,
) -> dict:
    """
    Correct split-aware training.
    All transformers are fit on X_train only, then applied to X_test.
    This prevents data leakage.
    """
    pipeline_state = pipeline_state or {}

    X_raw = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop ID-like columns that have no predictive value
    id_patterns = ["id", "passengerid", "passenger_id", "unnamed:_0", "index"]
    id_cols = [c for c in X_raw.columns if any(p in c.lower().replace(" ", "_") for p in id_patterns)]
    if id_cols:
        X_raw = X_raw.drop(columns=id_cols)

    # Drop high-cardinality categorical columns (>50 unique values) and text columns
    # These create noise and don't generalize
    cat_cols = X_raw.select_dtypes(exclude="number").columns.tolist()
    noisy_cols = []
    for col in cat_cols:
        unique_count = X_raw[col].nunique()
        if unique_count > 50:
            noisy_cols.append(col)
    if noisy_cols:
        X_raw = X_raw.drop(columns=noisy_cols)

    # Split FIRST — before any fitting
    stratify = y if task == "classification" else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=42, stratify=stratify
    )

    # Apply transformations fit on X_train only
    X_train, X_test, y_train = _apply_pipeline_split_aware(
        X_train_raw, X_test_raw, y_train, pipeline_state, target_col
    )

    # Keep only numeric columns — but if categoricals remain (encoding skipped),
    # auto-encode them so the model always has usable features
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    if cat_cols:
        for col in cat_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col]  = X_test[col].astype(str).map(lambda v: le.classes_.tolist().index(v) if v in le.classes_ else -1)
    X_train = X_train.select_dtypes(include="number")
    X_test  = X_test.select_dtypes(include="number")

    # Align columns — X_test may have different columns after get_dummies
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Build and fit model
    if task == "classification":
        # Detect class imbalance and use balanced weights
        class_counts = y_train.value_counts()
        imbalance_ratio = float(class_counts.min() / class_counts.max())
        class_weight = "balanced" if imbalance_ratio < 0.5 else None
        
        model = _build_classifier(model_name, model_params, class_weight=class_weight)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scoring = "f1_weighted"
    else:
        model = _build_regressor(model_name, model_params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scoring = "r2"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {}
    if task == "classification":
        metrics["accuracy"]    = round(float(accuracy_score(y_test, y_pred)), 4)
        metrics["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        metrics["f1_macro"]    = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
        try:
            y_prob = model.predict_proba(X_test)
            metrics["roc_auc"] = round(float(
                roc_auc_score(y_test, y_prob[:, 1]) if y_prob.shape[1] == 2
                else roc_auc_score(y_test, y_prob, multi_class="ovr")
            ), 4)
        except Exception:
            metrics["roc_auc"] = None
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

        # Imbalance warning
        class_counts = y_test.value_counts()
        imbalance_ratio = float(class_counts.min() / class_counts.max())
        if imbalance_ratio < 0.4:
            metrics["imbalance_warning"] = (
                f"Class imbalance detected (ratio {imbalance_ratio:.2f}). "
                f"Accuracy may be misleading — prioritise F1 and ROC-AUC."
            )
    else:
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        metrics["mae"]  = round(float(mean_absolute_error(y_test, y_pred)), 4)
        metrics["r2"]   = round(float(r2_score(y_test, y_pred)), 4)

    # Cross-validation on training data only
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=cv_scoring)
    metrics["cv_mean"] = round(float(cv_scores.mean()), 4)
    metrics["cv_std"]  = round(float(cv_scores.std()), 4)

    # Feature importance
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        fi = dict(zip(X_train.columns, model.feature_importances_))
        feature_importance = {k: round(float(v), 6) for k, v in sorted(fi.items(), key=lambda x: -x[1])}
    elif hasattr(model, "coef_"):
        coef = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        fi = dict(zip(X_train.columns, np.abs(coef)))
        feature_importance = {k: round(float(v), 6) for k, v in sorted(fi.items(), key=lambda x: -x[1])}

    return {
        "model": model_name,
        "params": model_params,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": len(X_train.columns),
        "note": "All transformers fitted on training set only. No data leakage.",
    }


def _apply_pipeline_split_aware(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    pipeline_state: dict,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Apply each preprocessing step by fitting on X_train, transforming both.
    This is the correct approach — no leakage.
    Returns X_train, X_test, y_train (y_train may be updated by drop_rows).
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    step_order = [
        "missing_values",
        "outliers",
        "feature_engineering",
        "encoding",
        "feature_selection",
        "scaling",
    ]

    for step_name in step_order:
        if step_name not in pipeline_state:
            continue

        technique = pipeline_state[step_name]["technique"]
        params    = pipeline_state[step_name]["params"]

        X_train, X_test, y_train = _apply_step_split_aware(
            X_train, X_test, y_train, step_name, technique, params, target_col
        )

    return X_train, X_test, y_train


def _apply_step_split_aware(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    step_name: str,
    technique: str,
    params: dict,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    from sklearn.impute import KNNImputer
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler,
        MaxAbsScaler, QuantileTransformer, PowerTransformer,
        LabelEncoder, OrdinalEncoder,
    )
    from sklearn.feature_selection import (
        VarianceThreshold, SelectKBest,
        mutual_info_classif, mutual_info_regression,
    )

    if step_name == "missing_values":
        num_cols_train = X_train.select_dtypes(include="number").columns.tolist()
        cat_cols_train = X_train.select_dtypes(exclude="number").columns.tolist()

        if technique == "mean":
            means = X_train[num_cols_train].mean()
            X_train[num_cols_train] = X_train[num_cols_train].fillna(means)
            X_test[num_cols_train]  = X_test[num_cols_train].fillna(means)
            for col in cat_cols_train:
                mode = X_train[col].mode()[0] if not X_train[col].mode().empty else "missing"
                X_train[col] = X_train[col].fillna(mode)
                X_test[col]  = X_test[col].fillna(mode)

        elif technique == "median":
            medians = X_train[num_cols_train].median()
            X_train[num_cols_train] = X_train[num_cols_train].fillna(medians)
            X_test[num_cols_train]  = X_test[num_cols_train].fillna(medians)
            for col in cat_cols_train:
                mode = X_train[col].mode()[0] if not X_train[col].mode().empty else "missing"
                X_train[col] = X_train[col].fillna(mode)
                X_test[col]  = X_test[col].fillna(mode)

        elif technique == "knn":
            imp = KNNImputer(n_neighbors=params.get("n_neighbors", 5))
            X_train[num_cols_train] = imp.fit_transform(X_train[num_cols_train])
            X_test[num_cols_train]  = imp.transform(X_test[num_cols_train])  # transform only

        elif technique == "mice":
            imp = IterativeImputer(max_iter=10, random_state=42)
            X_train[num_cols_train] = imp.fit_transform(X_train[num_cols_train])
            X_test[num_cols_train]  = imp.transform(X_test[num_cols_train])

        elif technique == "constant":
            fill_value = params.get("fill_value", 0)
            X_train = X_train.fillna(fill_value)
            X_test  = X_test.fillna(fill_value)

        elif technique == "drop_rows":
            mask = X_train.notna().all(axis=1)
            X_train = X_train[mask]
            y_train = y_train.loc[X_train.index]
            # Don't drop test rows — impute with median instead
            medians = X_train[num_cols_train].median()
            X_test[num_cols_train] = X_test[num_cols_train].fillna(medians)

        elif technique == "drop_cols":
            threshold = params.get("threshold", 0.5)
            cols_to_drop = [
                c for c in X_train.columns
                if X_train[c].isnull().mean() > threshold
            ]
            X_train = X_train.drop(columns=cols_to_drop)
            X_test  = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])

    elif step_name == "outliers":
        num_cols = X_train.select_dtypes(include="number").columns.tolist()

        if technique == "iqr_cap":
            for col in num_cols:
                Q1 = X_train[col].quantile(0.25)
                Q3 = X_train[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                # Fit bounds on train, apply to both
                X_train[col] = X_train[col].clip(lower=lower, upper=upper)
                X_test[col]  = X_test[col].clip(lower=lower, upper=upper)

        elif technique == "zscore_remove":
            threshold = params.get("threshold", 3.0)
            for col in num_cols:
                mean = X_train[col].mean()
                std  = X_train[col].std()
                # Only remove from train, cap test
                z_train = np.abs((X_train[col] - mean) / std)
                X_train = X_train[z_train <= threshold]
                z_test = np.abs((X_test[col] - mean) / std)
                X_test[col] = X_test[col].clip(
                    lower=mean - threshold * std,
                    upper=mean + threshold * std,
                )
                # Align y_train with updated X_train
                y_train = y_train.loc[X_train.index]

        elif technique == "log_transform":
            for col in num_cols:
                if (X_train[col] > 0).all() and (X_test[col] > 0).all():
                    X_train[col] = np.log1p(X_train[col])
                    X_test[col]  = np.log1p(X_test[col])

    elif step_name == "feature_engineering":
        num_cols = X_train.select_dtypes(include="number").columns.tolist()

        if technique == "log_features":
            for col in num_cols:
                if (X_train[col] > 0).all():
                    X_train[f"log_{col}"] = np.log1p(X_train[col])
                    X_test[f"log_{col}"]  = np.log1p(X_test[col])

        elif technique == "polynomial":
            cols_to_use = num_cols[:5]
            for i, col1 in enumerate(cols_to_use):
                for col2 in cols_to_use[i:]:
                    X_train[f"{col1}_x_{col2}"] = X_train[col1] * X_train[col2]
                    X_test[f"{col1}_x_{col2}"]  = X_test[col1]  * X_test[col2]

        elif technique == "sqrt_features":
            for col in num_cols:
                if (X_train[col] >= 0).all():
                    X_train[f"sqrt_{col}"] = np.sqrt(X_train[col])
                    X_test[f"sqrt_{col}"]  = np.sqrt(X_test[col])

        elif technique == "interaction":
            cols_to_use = num_cols[:6]
            for i in range(len(cols_to_use)):
                for j in range(i + 1, len(cols_to_use)):
                    col1, col2 = cols_to_use[i], cols_to_use[j]
                    X_train[f"{col1}_times_{col2}"] = X_train[col1] * X_train[col2]
                    X_test[f"{col1}_times_{col2}"]  = X_test[col1]  * X_test[col2]

    elif step_name == "encoding":
        cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

        if technique == "onehot":
            # Fit on train, align test to same columns
            X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
            X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=False)
            X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

        elif technique == "label":
            for col in cat_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                # Map test using known classes, unknown → -1
                le_map = {cls: idx for idx, cls in enumerate(le.classes_)}
                X_test[col] = X_test[col].astype(str).map(le_map).fillna(-1).astype(int)

        elif technique == "frequency":
            for col in cat_cols:
                # Frequency from train only
                freq_map = X_train[col].value_counts(normalize=True).to_dict()
                X_train[col] = X_train[col].map(freq_map).fillna(0)
                X_test[col]  = X_test[col].map(freq_map).fillna(0)

        elif technique == "target":
            # Leave-one-out style — compute means from train only
            for col in cat_cols:
                means = y_train.groupby(X_train[col]).mean().to_dict()
                global_mean = float(y_train.mean())
                X_train[col] = X_train[col].map(means).fillna(global_mean)
                X_test[col]  = X_test[col].map(means).fillna(global_mean)

    elif step_name == "feature_selection":
        num_cols = X_train.select_dtypes(include="number").columns.tolist()

        if technique == "variance_threshold":
            threshold = params.get("threshold", 0.01)
            sel = VarianceThreshold(threshold=threshold)
            sel.fit(X_train[num_cols])
            kept = [col for col, s in zip(num_cols, sel.get_support()) if s]
            non_num = [c for c in X_train.columns if c not in num_cols]
            X_train = X_train[kept + non_num]
            X_test  = X_test[[c for c in kept + non_num if c in X_test.columns]]

        elif technique == "correlation":
            threshold = params.get("threshold", 0.95)
            corr = X_train[num_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
            X_train = X_train.drop(columns=to_drop)
            X_test  = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])

        elif technique == "mutual_info":
            k = params.get("k", min(10, len(num_cols)))
            score_fn = mutual_info_classif if len(y_train.unique()) < 20 else mutual_info_regression
            sel = SelectKBest(score_fn, k=k)
            sel.fit(X_train[num_cols], y_train)
            kept = [col for col, s in zip(num_cols, sel.get_support()) if s]
            non_num = [c for c in X_train.columns if c not in num_cols]
            X_train = X_train[kept + non_num]
            X_test  = X_test[[c for c in kept + non_num if c in X_test.columns]]

    elif step_name == "scaling":
        num_cols = X_train.select_dtypes(include="number").columns.tolist()
        scalers = {
            "standard": StandardScaler(),
            "minmax":   MinMaxScaler(),
            "robust":   RobustScaler(),
            "maxabs":   MaxAbsScaler(),
            "quantile": QuantileTransformer(output_distribution="uniform", random_state=42),
            "power":    PowerTransformer(method="yeo-johnson"),
        }
        if technique in scalers:
            scaler = scalers[technique]
            # Fit on train only, transform both
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols]  = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train