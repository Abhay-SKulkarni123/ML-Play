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
        mask = pd.Series([True] * len(df_out))
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
            warnings.append("Target encoding uses full dataset means — ensure this is applied correctly on train set only in production.")

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
        "standard":  StandardScaler(),
        "minmax":    MinMaxScaler(),
        "robust":    RobustScaler(),
        "maxabs":    MaxAbsScaler(),
        "quantile":  QuantileTransformer(output_distribution="uniform", random_state=42),
        "power":     PowerTransformer(method="yeo-johnson"),
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