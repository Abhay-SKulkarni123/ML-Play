import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer,
    LabelEncoder, OrdinalEncoder
)
from app.services.ml.types import PipelineStepResult


# ─── MISSING VALUE HANDLERS ───────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, technique: str, params: dict) -> PipelineStepResult:
    before_missing = df.isnull().sum().sum()
    before_per_col = df.isnull().sum().to_dict()
    df_out = df.copy()
    warnings = []

    numeric_cols = df_out.select_dtypes(include="number").columns.tolist()
    cat_cols = df_out.select_dtypes(exclude="number").columns.tolist()

    if technique == "mean":
        for col in numeric_cols:
            df_out[col].fillna(df_out[col].mean(), inplace=True)
        for col in cat_cols:
            df_out[col].fillna(df_out[col].mode()[0] if not df_out[col].mode().empty else "missing", inplace=True)

    elif technique == "median":
        for col in numeric_cols:
            df_out[col].fillna(df_out[col].median(), inplace=True)
        for col in cat_cols:
            df_out[col].fillna(df_out[col].mode()[0] if not df_out[col].mode().empty else "missing", inplace=True)

    elif technique == "mode":
        for col in df_out.columns:
            if not df_out[col].mode().empty:
                df_out[col].fillna(df_out[col].mode()[0], inplace=True)

    elif technique == "knn":
        n = params.get("n_neighbors", 5)
        if len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=n)
            df_out[numeric_cols] = imputer.fit_transform(df_out[numeric_cols])
        for col in cat_cols:
            df_out[col].fillna(df_out[col].mode()[0] if not df_out[col].mode().empty else "missing", inplace=True)

    elif technique == "mice":
        if len(numeric_cols) > 0:
            imputer = IterativeImputer(max_iter=10, random_state=42)
            df_out[numeric_cols] = imputer.fit_transform(df_out[numeric_cols])
        for col in cat_cols:
            df_out[col].fillna(df_out[col].mode()[0] if not df_out[col].mode().empty else "missing", inplace=True)
        warnings.append("MICE is slow on large datasets. Results may take a moment.")

    elif technique == "constant":
        fill_value = params.get("fill_value", 0)
        df_out.fillna(fill_value, inplace=True)

    elif technique == "drop_rows":
        df_out.dropna(inplace=True)
        warnings.append(f"Dropped {len(df) - len(df_out)} rows containing missing values.")

    elif technique == "drop_cols":
        threshold = params.get("threshold", 0.5)
        cols_to_drop = [c for c in df_out.columns if df_out[c].isnull().mean() > threshold]
        df_out.drop(columns=cols_to_drop, inplace=True)
        if cols_to_drop:
            warnings.append(f"Dropped columns: {cols_to_drop} (>{threshold*100}% missing)")

    after_missing = df_out.isnull().sum().sum()

    return PipelineStepResult(
        step="missing_values",
        technique=technique,
        params=params,
        stats={
            "missing_before": int(before_missing),
            "missing_after": int(after_missing),
            "rows_before": len(df),
            "rows_after": len(df_out),
            "cols_before": len(df.columns),
            "cols_after": len(df_out.columns),
            "per_column_before": {k: int(v) for k, v in before_per_col.items()},
        },
        warnings=warnings,
    )


# ─── OUTLIER HANDLERS ─────────────────────────────────────────────────────────

def handle_outliers(df: pd.DataFrame, technique: str, params: dict) -> PipelineStepResult:
    df_out = df.copy()
    warnings = []
    numeric_cols = df_out.select_dtypes(include="number").columns.tolist()
    outlier_counts = {}

    if technique == "iqr_cap":
        for col in numeric_cols:
            Q1, Q3 = df_out[col].quantile(0.25), df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            n_outliers = int(((df_out[col] < lower) | (df_out[col] > upper)).sum())
            outlier_counts[col] = n_outliers
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)

    elif technique == "zscore_remove":
        threshold = params.get("threshold", 3.0)
        mask = pd.Series([True] * len(df_out), index=df_out.index)
        for col in numeric_cols:
            z = np.abs((df_out[col] - df_out[col].mean()) / df_out[col].std())
            mask = mask & (z <= threshold)
            outlier_counts[col] = int((z > threshold).sum())
        rows_before = len(df_out)
        df_out = df_out[mask]
        warnings.append(f"Removed {rows_before - len(df_out)} rows with z-score > {threshold}")

    elif technique == "log_transform":
        for col in numeric_cols:
            if (df_out[col] > 0).all():
                df_out[col] = np.log1p(df_out[col])
            else:
                warnings.append(f"Skipped log transform on '{col}' — contains non-positive values.")

    elif technique == "keep":
        warnings.append("No outlier treatment applied. Outliers documented only.")

    total_outliers = sum(outlier_counts.values())

    return PipelineStepResult(
        step="outliers",
        technique=technique,
        params=params,
        stats={
            "rows_before": len(df),
            "rows_after": len(df_out),
            "total_outliers_found": total_outliers,
            "per_column": outlier_counts,
        },
        warnings=warnings,
    )


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def handle_feature_engineering(df: pd.DataFrame, technique: str, params: dict, target_col: str) -> PipelineStepResult:
    df_out = df.copy()
    warnings = []
    cols_before = len(df_out.columns)
    numeric_cols = [c for c in df_out.select_dtypes(include="number").columns if c != target_col]
    new_features = []

    if technique == "polynomial":
        cols_to_use = numeric_cols[:5]
        for i, col1 in enumerate(cols_to_use):
            for col2 in cols_to_use[i:]:
                new_col = f"{col1}_x_{col2}"
                df_out[new_col] = df_out[col1] * df_out[col2]
                new_features.append(new_col)
        if len(numeric_cols) > 5:
            warnings.append("Polynomial features limited to first 5 numeric columns to avoid feature explosion.")

    elif technique == "interaction":
        cols_to_use = numeric_cols[:6]
        for i in range(len(cols_to_use)):
            for j in range(i + 1, len(cols_to_use)):
                col1, col2 = cols_to_use[i], cols_to_use[j]
                new_col = f"{col1}_times_{col2}"
                df_out[new_col] = df_out[col1] * df_out[col2]
                new_features.append(new_col)

    elif technique == "log_features":
        for col in numeric_cols:
            if (df_out[col] > 0).all():
                new_col = f"log_{col}"
                df_out[new_col] = np.log1p(df_out[col])
                new_features.append(new_col)
            else:
                warnings.append(f"Skipped log on '{col}' — contains non-positive values.")

    elif technique == "ratio":
        cols_to_use = numeric_cols[:4]
        for i in range(len(cols_to_use)):
            for j in range(len(cols_to_use)):
                if i != j:
                    col1, col2 = cols_to_use[i], cols_to_use[j]
                    if (df_out[col2] != 0).all():
                        new_col = f"{col1}_div_{col2}"
                        df_out[new_col] = df_out[col1] / df_out[col2]
                        new_features.append(new_col)

    elif technique == "binning":
        n_bins = params.get("n_bins", 5)
        for col in numeric_cols[:5]:
            new_col = f"{col}_binned"
            df_out[new_col] = pd.cut(df_out[col], bins=n_bins, labels=False, duplicates="drop")
            new_features.append(new_col)

    elif technique == "sqrt_features":
        for col in numeric_cols:
            if (df_out[col] >= 0).all():
                new_col = f"sqrt_{col}"
                df_out[new_col] = np.sqrt(df_out[col])
                new_features.append(new_col)

    elif technique == "none":
        warnings.append("No feature engineering applied. Moving to next step.")

    return PipelineStepResult(
        step="feature_engineering",
        technique=technique,
        params=params,
        stats={
            "cols_before": cols_before,
            "cols_after": len(df_out.columns),
            "new_features_created": len(new_features),
            "new_feature_names": new_features[:10],
        },
        warnings=warnings,
    )


# ─── FEATURE SELECTION ────────────────────────────────────────────────────────

def handle_feature_selection(df: pd.DataFrame, technique: str, params: dict, target_col: str) -> PipelineStepResult:
    df_out = df.copy()
    warnings = []
    cols_before = len(df_out.columns)
    dropped = []

    if target_col not in df_out.columns:
        warnings.append("Target column not found. Feature selection skipped.")
        return PipelineStepResult(
            step="feature_selection", technique=technique, params=params,
            stats={"cols_before": cols_before, "cols_after": cols_before, "dropped_columns": []},
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
            warnings.append("No low-variance features found. Nothing dropped.")

    elif technique == "correlation":
        threshold = params.get("threshold", 0.95)
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        df_out.drop(columns=to_drop, inplace=True)
        dropped = to_drop
        if not dropped:
            warnings.append(f"No features with correlation > {threshold} found.")

    elif technique == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest
        k = params.get("k", min(10, len(X.columns)))
        score_fn = mutual_info_classif if len(y.unique()) < 20 else mutual_info_regression
        sel = SelectKBest(score_fn, k=k)
        sel.fit(X, y)
        kept = X.columns[sel.get_support()].tolist()
        dropped = [c for c in X.columns if c not in kept]
        non_numeric = [c for c in df_out.columns if c not in X.columns and c != target_col]
        keep_cols = kept + [target_col] + non_numeric
        df_out = df_out[[c for c in keep_cols if c in df_out.columns]]

    elif technique == "none":
        warnings.append("No feature selection applied. All features kept.")

    return PipelineStepResult(
        step="feature_selection",
        technique=technique,
        params=params,
        stats={
            "cols_before": cols_before,
            "cols_after": len(df_out.columns),
            "dropped_columns": dropped,
            "n_dropped": len(dropped),
        },
        warnings=warnings,
    )


# ─── ENCODING ─────────────────────────────────────────────────────────────────

def handle_encoding(df: pd.DataFrame, technique: str, params: dict, target_col: str) -> PipelineStepResult:
    df_out = df.copy()
    warnings = []
    cat_cols = [c for c in df_out.select_dtypes(exclude="number").columns if c != target_col]
    cols_before = len(df_out.columns)

    if not cat_cols:
        warnings.append("No categorical columns found. Encoding step skipped.")
        return PipelineStepResult(
            step="encoding", technique=technique, params=params,
            stats={"cols_before": cols_before, "cols_after": cols_before, "encoded_columns": []},
            warnings=warnings,
        )

    if technique == "onehot":
        high_card = [c for c in cat_cols if df_out[c].nunique() > 20]
        if high_card:
            warnings.append(f"High cardinality columns {high_card} will create many features. Consider label encoding.")
        df_out = pd.get_dummies(df_out, columns=cat_cols, drop_first=False)

    elif technique == "label":
        for col in cat_cols:
            le = LabelEncoder()
            df_out[col] = le.fit_transform(df_out[col].astype(str))

    elif technique == "ordinal":
        encoder = OrdinalEncoder()
        df_out[cat_cols] = encoder.fit_transform(df_out[cat_cols].astype(str))

    elif technique == "frequency":
        for col in cat_cols:
            freq_map = df_out[col].value_counts(normalize=True).to_dict()
            df_out[col] = df_out[col].map(freq_map)

    elif technique == "target":
        if target_col not in df_out.columns:
            warnings.append("Target column not found. Target encoding skipped.")
        else:
            for col in cat_cols:
                means = df_out.groupby(col)[target_col].mean().to_dict()
                df_out[col] = df_out[col].map(means)
            warnings.append("Target encoding uses full dataset means — use train set only in production.")

    return PipelineStepResult(
        step="encoding",
        technique=technique,
        params=params,
        stats={
            "cols_before": cols_before,
            "cols_after": len(df_out.columns),
            "encoded_columns": cat_cols,
            "new_cols_created": len(df_out.columns) - cols_before,
        },
        warnings=warnings,
    )


# ─── SCALING ──────────────────────────────────────────────────────────────────

def handle_scaling(df: pd.DataFrame, technique: str, params: dict, target_col: str) -> PipelineStepResult:
    df_out = df.copy()
    warnings = []
    numeric_cols = [c for c in df_out.select_dtypes(include="number").columns if c != target_col]

    if technique == "none":
        warnings.append("No scaling applied. Appropriate for tree-based models (RF, XGBoost, LightGBM).")
        return PipelineStepResult(
            step="scaling", technique="none", params={},
            stats={"scaled_columns": [], "note": "No scaling applied"},
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
        raise ValueError(f"Unknown scaler: {technique}")

    scaler = scalers[technique]
    before_stats = df_out[numeric_cols].describe().to_dict()
    df_out[numeric_cols] = scaler.fit_transform(df_out[numeric_cols])
    after_stats = df_out[numeric_cols].describe().to_dict()

    return PipelineStepResult(
        step="scaling",
        technique=technique,
        params=params,
        stats={
            "scaled_columns": numeric_cols,
            "n_columns_scaled": len(numeric_cols),
            "sample_before": {col: {"mean": round(before_stats[col]["mean"], 4), "std": round(before_stats[col]["std"], 4)} for col in numeric_cols[:5]},
            "sample_after":  {col: {"mean": round(after_stats[col]["mean"],  4), "std": round(after_stats[col]["std"],  4)} for col in numeric_cols[:5]},
        },
        warnings=warnings,
    )