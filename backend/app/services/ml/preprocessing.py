import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer,
    LabelEncoder, OrdinalEncoder,
)
from app.services.ml.types import PipelineStepResult


# ─── MISSING VALUES ───────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, technique: str, params: dict) -> PipelineStepResult:
    before_missing  = int(df.isnull().sum().sum())
    before_per_col  = {k: int(v) for k, v in df.isnull().sum().items()}
    df_out          = df.copy()
    warnings        = []

    numeric_cols = df_out.select_dtypes(include="number").columns.tolist()
    cat_cols     = df_out.select_dtypes(exclude="number").columns.tolist()

    def fill_cats(frame):
        for col in cat_cols:
            mode = frame[col].mode()
            frame[col].fillna(mode[0] if not mode.empty else "missing", inplace=True)

    if technique == "mean":
        for col in numeric_cols:
            df_out[col].fillna(df_out[col].mean(), inplace=True)
        fill_cats(df_out)

    elif technique == "median":
        for col in numeric_cols:
            df_out[col].fillna(df_out[col].median(), inplace=True)
        fill_cats(df_out)

    elif technique == "mode":
        for col in df_out.columns:
            mode = df_out[col].mode()
            if not mode.empty:
                df_out[col].fillna(mode[0], inplace=True)

    elif technique == "knn":
        n = params.get("n_neighbors", 5)
        if len(numeric_cols) > 0:
            imp = KNNImputer(n_neighbors=n)
            df_out[numeric_cols] = imp.fit_transform(df_out[numeric_cols])
        fill_cats(df_out)
        warnings.append(
            "KNN imputation fits on the full dataset here for display. "
            "During model training, it fits on X_train only — no leakage."
        )

    elif technique == "mice":
        if len(numeric_cols) > 0:
            imp = IterativeImputer(max_iter=10, random_state=42)
            df_out[numeric_cols] = imp.fit_transform(df_out[numeric_cols])
        fill_cats(df_out)
        warnings.append("MICE is computationally expensive on large datasets.")

    elif technique == "constant":
        fill_value = params.get("fill_value", 0)
        df_out.fillna(fill_value, inplace=True)

    elif technique == "drop_rows":
        rows_before = len(df_out)
        df_out.dropna(inplace=True)
        dropped = rows_before - len(df_out)
        if dropped > 0:
            warnings.append(
                f"Dropped {dropped} rows ({round(dropped/rows_before*100, 1)}% of data). "
                "High drop rates reduce model reliability."
            )

    elif technique == "drop_cols":
        threshold = params.get("threshold", 0.5)
        cols_to_drop = [c for c in df_out.columns if df_out[c].isnull().mean() > threshold]
        df_out.drop(columns=cols_to_drop, inplace=True)
        if cols_to_drop:
            warnings.append(
                f"Dropped {len(cols_to_drop)} column(s): {cols_to_drop}. "
                f"Each had >{int(threshold*100)}% missing values."
            )
        else:
            warnings.append(f"No columns exceeded the {int(threshold*100)}% missing threshold.")

    elif technique == "forward_fill":
        df_out.fillna(method="ffill", inplace=True)
        warnings.append("Forward fill is suitable for time-series data only.")

    else:
        raise ValueError(f"Unknown missing value technique: '{technique}'")

    after_missing = int(df_out.isnull().sum().sum())

    return PipelineStepResult(
        step="missing_values",
        technique=technique,
        params=params,
        stats={
            "missing_before":   before_missing,
            "missing_after":    after_missing,
            "rows_before":      len(df),
            "rows_after":       len(df_out),
            "cols_before":      len(df.columns),
            "cols_after":       len(df_out.columns),
            "per_column_before": before_per_col,
        },
        warnings=warnings,
    )


# ─── OUTLIERS ─────────────────────────────────────────────────────────────────

def handle_outliers(df: pd.DataFrame, technique: str, params: dict) -> PipelineStepResult:
    df_out       = df.copy()
    warnings     = []
    numeric_cols = df_out.select_dtypes(include="number").columns.tolist()
    outlier_counts: dict = {}

    if technique == "iqr_cap":
        for col in numeric_cols:
            Q1  = df_out[col].quantile(0.25)
            Q3  = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_out = int(((df_out[col] < lower) | (df_out[col] > upper)).sum())
            outlier_counts[col] = n_out
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)

    elif technique == "zscore_remove":
        threshold = params.get("threshold", 3.0)
        mask = pd.Series([True] * len(df_out), index=df_out.index)
        for col in numeric_cols:
            std = df_out[col].std()
            if std == 0:
                continue
            z = np.abs((df_out[col] - df_out[col].mean()) / std)
            outlier_counts[col] = int((z > threshold).sum())
            mask = mask & (z <= threshold)
        rows_before = len(df_out)
        df_out = df_out[mask]
        dropped = rows_before - len(df_out)
        if dropped > 0:
            warnings.append(
                f"Removed {dropped} rows with z-score > {threshold}. "
                "During training, removal only applies to the training set."
            )

    elif technique == "log_transform":
        skipped = []
        for col in numeric_cols:
            if (df_out[col] > 0).all():
                df_out[col] = np.log1p(df_out[col])
                outlier_counts[col] = 0
            else:
                skipped.append(col)
        if skipped:
            warnings.append(
                f"Skipped log transform on: {skipped} "
                "(contain zero or negative values)."
            )

    elif technique == "keep":
        for col in numeric_cols:
            Q1  = df_out[col].quantile(0.25)
            Q3  = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outlier_counts[col] = int(((df_out[col] < lower) | (df_out[col] > upper)).sum())
        warnings.append(
            "No outlier treatment applied. "
            "Outliers counted for reference only."
        )

    else:
        raise ValueError(f"Unknown outlier technique: '{technique}'")

    total_outliers = sum(outlier_counts.values())

    return PipelineStepResult(
        step="outliers",
        technique=technique,
        params=params,
        stats={
            "rows_before":          len(df),
            "rows_after":           len(df_out),
            "total_outliers_found": total_outliers,
            "per_column":           outlier_counts,
        },
        warnings=warnings,
    )


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def handle_feature_engineering(
    df: pd.DataFrame, technique: str, params: dict, target_col: str
) -> PipelineStepResult:
    df_out       = df.copy()
    warnings     = []
    cols_before  = len(df_out.columns)
    numeric_cols = [c for c in df_out.select_dtypes(include="number").columns if c != target_col]
    new_features: list[str] = []

    if technique == "polynomial":
        cols_to_use = numeric_cols[:5]
        for i, col1 in enumerate(cols_to_use):
            for col2 in cols_to_use[i:]:
                name = f"{col1}_x_{col2}"
                df_out[name] = df_out[col1] * df_out[col2]
                new_features.append(name)
        if len(numeric_cols) > 5:
            warnings.append(
                "Polynomial features limited to first 5 numeric columns "
                "to avoid feature explosion."
            )

    elif technique == "interaction":
        cols_to_use = numeric_cols[:6]
        for i in range(len(cols_to_use)):
            for j in range(i + 1, len(cols_to_use)):
                col1, col2 = cols_to_use[i], cols_to_use[j]
                name = f"{col1}_times_{col2}"
                df_out[name] = df_out[col1] * df_out[col2]
                new_features.append(name)

    elif technique == "log_features":
        skipped = []
        for col in numeric_cols:
            if (df_out[col] > 0).all():
                name = f"log_{col}"
                df_out[name] = np.log1p(df_out[col])
                new_features.append(name)
            else:
                skipped.append(col)
        if skipped:
            warnings.append(f"Skipped log on: {skipped} (non-positive values).")

    elif technique == "ratio":
        cols_to_use = numeric_cols[:4]
        for i in range(len(cols_to_use)):
            for j in range(len(cols_to_use)):
                if i != j:
                    col1, col2 = cols_to_use[i], cols_to_use[j]
                    if (df_out[col2] != 0).all():
                        name = f"{col1}_div_{col2}"
                        df_out[name] = df_out[col1] / df_out[col2]
                        new_features.append(name)

    elif technique == "binning":
        n_bins = params.get("n_bins", 5)
        for col in numeric_cols[:5]:
            name = f"{col}_binned"
            df_out[name] = pd.cut(
                df_out[col], bins=n_bins, labels=False, duplicates="drop"
            )
            new_features.append(name)

    elif technique == "sqrt_features":
        skipped = []
        for col in numeric_cols:
            if (df_out[col] >= 0).all():
                name = f"sqrt_{col}"
                df_out[name] = np.sqrt(df_out[col])
                new_features.append(name)
            else:
                skipped.append(col)
        if skipped:
            warnings.append(f"Skipped sqrt on: {skipped} (negative values).")

    elif technique == "none":
        warnings.append("No feature engineering applied. Step skipped.")

    else:
        raise ValueError(f"Unknown feature engineering technique: '{technique}'")

    return PipelineStepResult(
        step="feature_engineering",
        technique=technique,
        params=params,
        stats={
            "cols_before":          cols_before,
            "cols_after":           len(df_out.columns),
            "new_features_created": len(new_features),
            "new_feature_names":    new_features[:10],
        },
        warnings=warnings,
    )


# ─── FEATURE SELECTION ────────────────────────────────────────────────────────

def handle_feature_selection(
    df: pd.DataFrame, technique: str, params: dict, target_col: str
) -> PipelineStepResult:
    df_out      = df.copy()
    warnings    = []
    cols_before = len(df_out.columns)
    dropped: list[str] = []

    if target_col not in df_out.columns:
        warnings.append("Target column not found. Feature selection skipped.")
        return PipelineStepResult(
            step="feature_selection",
            technique=technique,
            params=params,
            stats={
                "cols_before":    cols_before,
                "cols_after":     cols_before,
                "dropped_columns": [],
                "n_dropped":      0,
            },
            warnings=warnings,
        )

    X = df_out.drop(columns=[target_col]).select_dtypes(include="number")
    y = df_out[target_col]

    if technique == "variance_threshold":
        from sklearn.feature_selection import VarianceThreshold
        threshold = params.get("threshold", 0.01)
        sel = VarianceThreshold(threshold=threshold)
        sel.fit(X)
        low_var = X.columns[~sel.get_support()].tolist()
        df_out.drop(columns=low_var, inplace=True)
        dropped = low_var
        if not dropped:
            warnings.append(
                f"No features had variance below {threshold}. Nothing dropped."
            )

    elif technique == "correlation":
        threshold = params.get("threshold", 0.95)
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        df_out.drop(columns=to_drop, inplace=True)
        dropped = to_drop
        if not dropped:
            warnings.append(
                f"No feature pairs with correlation > {threshold} found."
            )
        else:
            warnings.append(
                "When highly correlated features exist, keeping one and dropping "
                "the other reduces multicollinearity without losing information."
            )

    elif technique == "mutual_info":
        from sklearn.feature_selection import (
            mutual_info_classif, mutual_info_regression, SelectKBest,
        )
        k = min(params.get("k", 10), len(X.columns))
        score_fn = (
            mutual_info_classif
            if len(y.unique()) < 20
            else mutual_info_regression
        )
        sel = SelectKBest(score_fn, k=k)
        sel.fit(X, y)
        kept    = X.columns[sel.get_support()].tolist()
        dropped = [c for c in X.columns if c not in kept]
        non_numeric = [
            c for c in df_out.columns
            if c not in X.columns and c != target_col
        ]
        keep_cols = kept + [target_col] + non_numeric
        df_out = df_out[[c for c in keep_cols if c in df_out.columns]]
        warnings.append(
            f"Kept top {k} features by mutual information with the target. "
            "MI measures non-linear statistical dependence."
        )

    elif technique == "none":
        warnings.append("No feature selection applied. All features kept.")

    else:
        raise ValueError(f"Unknown feature selection technique: '{technique}'")

    return PipelineStepResult(
        step="feature_selection",
        technique=technique,
        params=params,
        stats={
            "cols_before":     cols_before,
            "cols_after":      len(df_out.columns),
            "dropped_columns": dropped,
            "n_dropped":       len(dropped),
        },
        warnings=warnings,
    )


# ─── ENCODING ─────────────────────────────────────────────────────────────────

def handle_encoding(
    df: pd.DataFrame, technique: str, params: dict, target_col: str
) -> PipelineStepResult:
    df_out      = df.copy()
    warnings    = []
    cat_cols    = [
        c for c in df_out.select_dtypes(exclude="number").columns
        if c != target_col
    ]
    cols_before = len(df_out.columns)

    if not cat_cols:
        warnings.append(
            "No categorical columns found. "
            "Encoding step skipped — dataset may already be numeric."
        )
        return PipelineStepResult(
            step="encoding",
            technique=technique,
            params=params,
            stats={
                "cols_before":     cols_before,
                "cols_after":      cols_before,
                "encoded_columns": [],
                "new_cols_created": 0,
            },
            warnings=warnings,
        )

    if technique == "onehot":
        high_card = [c for c in cat_cols if df_out[c].nunique() > 20]
        if high_card:
            warnings.append(
                f"High cardinality columns: {high_card}. "
                "One-hot encoding will create many columns. "
                "Consider label or frequency encoding for these."
            )
        df_out = pd.get_dummies(df_out, columns=cat_cols, drop_first=False)

    elif technique == "label":
        warnings.append(
            "Label encoding assigns arbitrary integers to categories. "
            "Safe for tree models. Avoid with linear models — use one-hot instead."
        )
        for col in cat_cols:
            le = LabelEncoder()
            df_out[col] = le.fit_transform(df_out[col].astype(str))

    elif technique == "ordinal":
        warnings.append(
            "Ordinal encoding implies order between categories. "
            "Only use when the order is meaningful (e.g. low/medium/high)."
        )
        enc = OrdinalEncoder()
        df_out[cat_cols] = enc.fit_transform(df_out[cat_cols].astype(str))

    elif technique == "frequency":
        warnings.append(
            "Frequency encoding replaces categories with their relative frequency. "
            "Preserves category prevalence information."
        )
        for col in cat_cols:
            freq_map = df_out[col].value_counts(normalize=True).to_dict()
            df_out[col] = df_out[col].map(freq_map)

    elif technique == "target":
        # Display only — training uses train-set means
        warnings.append(
            "⚠ Target encoding shown here uses full dataset means for display. "
            "During model training, only training-set means are used to prevent leakage."
        )
        if target_col not in df_out.columns:
            warnings.append("Target column not found. Target encoding skipped.")
        else:
            global_mean = float(df_out[target_col].mean())
            for col in cat_cols:
                means = df_out.groupby(col)[target_col].mean().to_dict()
                df_out[col] = df_out[col].map(means).fillna(global_mean)

    else:
        raise ValueError(f"Unknown encoding technique: '{technique}'")

    return PipelineStepResult(
        step="encoding",
        technique=technique,
        params=params,
        stats={
            "cols_before":      cols_before,
            "cols_after":       len(df_out.columns),
            "encoded_columns":  cat_cols,
            "new_cols_created": len(df_out.columns) - cols_before,
        },
        warnings=warnings,
    )


# ─── SCALING ──────────────────────────────────────────────────────────────────

def handle_scaling(
    df: pd.DataFrame, technique: str, params: dict, target_col: str
) -> PipelineStepResult:
    df_out       = df.copy()
    warnings     = []
    numeric_cols = [
        c for c in df_out.select_dtypes(include="number").columns
        if c != target_col
    ]

    if technique == "none":
        warnings.append(
            "No scaling applied. "
            "Correct for tree-based models (RF, XGBoost, LightGBM, Decision Tree) "
            "which are scale-invariant. Required for linear models and KNN."
        )
        return PipelineStepResult(
            step="scaling",
            technique="none",
            params={},
            stats={"scaled_columns": [], "n_columns_scaled": 0},
            warnings=warnings,
        )

    scalers = {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
        "robust":   RobustScaler(),
        "maxabs":   MaxAbsScaler(),
        "quantile": QuantileTransformer(output_distribution="uniform", random_state=42),
        "power":    PowerTransformer(method="yeo-johnson"),
    }

    if technique not in scalers:
        raise ValueError(f"Unknown scaler: '{technique}'")

    scaler = scalers[technique]

    before_stats = df_out[numeric_cols].describe().to_dict()
    df_out[numeric_cols] = scaler.fit_transform(df_out[numeric_cols])
    after_stats  = df_out[numeric_cols].describe().to_dict()

    warnings.append(
        "Scaler shown here is fit on the full dataset for display. "
        "During training, it is fit on X_train only and applied to X_test."
    )

    return PipelineStepResult(
        step="scaling",
        technique=technique,
        params=params,
        stats={
            "scaled_columns":   numeric_cols,
            "n_columns_scaled": len(numeric_cols),
            "sample_before": {
                col: {
                    "mean": round(before_stats[col]["mean"], 4),
                    "std":  round(before_stats[col]["std"],  4),
                }
                for col in numeric_cols[:5]
            },
            "sample_after": {
                col: {
                    "mean": round(after_stats[col]["mean"], 4),
                    "std":  round(after_stats[col]["std"],  4),
                }
                for col in numeric_cols[:5]
            },
        },
        warnings=warnings,
    )