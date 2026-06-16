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
    before_missing = int(df.isnull().sum().sum())
    before_per_col = {k: int(v) for k, v in df.isnull().sum().items()}
    df_out = df.copy()
    warnings = []

    numeric_cols = df_out.select_dtypes(include="number").columns.tolist()
    cat_cols = df_out.select_dtypes(exclude="number").columns.tolist()

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
        warnings.append("KNN fits on full data for display. During training, fits on X_train only.")

    elif technique == "mice":
        if len(numeric_cols) > 0:
            imp = IterativeImputer(max_iter=10, random_state=42)
            df_out[numeric_cols] = imp.fit_transform(df_out[numeric_cols])
        fill_cats(df_out)
        warnings.append("MICE (Multivariate Imputation by Chained Equations) is computationally expensive on large datasets.")

    elif technique == "random_sample":
        for col in numeric_cols:
            null_mask = df_out[col].isnull()
            if null_mask.any():
                df_out.loc[null_mask, col] = df_out[col].dropna().sample(
                    null_mask.sum(), replace=True, random_state=42
                ).values
        fill_cats(df_out)
        warnings.append("Random sample imputation preserves the original distribution but adds randomness.")

    elif technique == "missing_indicator":
        for col in numeric_cols:
            if df_out[col].isnull().any():
                df_out[f"{col}_was_missing"] = df_out[col].isnull().astype(int)
        for col in numeric_cols:
            df_out[col].fillna(df_out[col].median(), inplace=True)
        fill_cats(df_out)
        warnings.append("Added binary indicator columns for each numeric column that had missing values. Then filled with median.")

    elif technique == "constant":
        fill_value = params.get("fill_value", 0)
        df_out.fillna(fill_value, inplace=True)

    elif technique == "drop_rows":
        rows_before = len(df_out)
        df_out.dropna(inplace=True)
        dropped = rows_before - len(df_out)
        if dropped > 0:
            warnings.append(f"Dropped {dropped} rows ({round(dropped/rows_before*100,1)}% of data).")

    elif technique == "drop_cols":
        threshold = params.get("threshold", 0.5)
        cols_to_drop = [c for c in df_out.columns if df_out[c].isnull().mean() > threshold]
        df_out.drop(columns=cols_to_drop, inplace=True)
        if cols_to_drop:
            warnings.append(f"Dropped {len(cols_to_drop)} column(s): {cols_to_drop}.")
        else:
            warnings.append(f"No columns exceeded the {int(threshold*100)}% missing threshold.")

    else:
        raise ValueError(f"Unknown missing value technique: '{technique}'")

    after_missing = int(df_out.isnull().sum().sum())

    return PipelineStepResult(
        step="missing_values",
        technique=technique,
        params=params,
        stats={
            "missing_before": before_missing,
            "missing_after": after_missing,
            "rows_before": len(df),
            "rows_after": len(df_out),
            "cols_before": len(df.columns),
            "cols_after": len(df_out.columns),
            "per_column_before": before_per_col,
        },
        warnings=warnings,
    )


# ─── OUTLIERS ─────────────────────────────────────────────────────────────────

def handle_outliers(df: pd.DataFrame, technique: str, params: dict) -> PipelineStepResult:
    df_out = df.copy()
    warnings = []
    numeric_cols = df_out.select_dtypes(include="number").columns.tolist()
    outlier_counts: dict = {}

    if technique == "iqr_cap":
        for col in numeric_cols:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
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
        warnings.append(f"Removed {rows_before - len(df_out)} rows with z-score > {threshold}.")

    elif technique == "percentile_cap":
        lower_pct = params.get("lower", 1)
        upper_pct = params.get("upper", 99)
        for col in numeric_cols:
            lower = df_out[col].quantile(lower_pct / 100)
            upper = df_out[col].quantile(upper_pct / 100)
            n_out = int(((df_out[col] < lower) | (df_out[col] > upper)).sum())
            outlier_counts[col] = n_out
            df_out[col] = df_out[col].clip(lower=lower, upper=upper)
        warnings.append(f"Capped values below {lower_pct}th and above {upper_pct}th percentile.")

    elif technique == "log_transform":
        skipped = []
        for col in numeric_cols:
            if (df_out[col] > 0).all():
                df_out[col] = np.log1p(df_out[col])
                outlier_counts[col] = 0
            else:
                skipped.append(col)
        if skipped:
            warnings.append(f"Skipped log transform on: {skipped} (contain zero or negative values).")

    elif technique == "keep":
        for col in numeric_cols:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outlier_counts[col] = int(((df_out[col] < lower) | (df_out[col] > upper)).sum())
        warnings.append("No outlier treatment applied. Outliers counted for reference only.")

    else:
        raise ValueError(f"Unknown outlier technique: '{technique}'")

    return PipelineStepResult(
        step="outliers",
        technique=technique,
        params=params,
        stats={
            "rows_before": len(df),
            "rows_after": len(df_out),
            "total_outliers_found": sum(outlier_counts.values()),
            "per_column": outlier_counts,
        },
        warnings=warnings,
    )


# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

def handle_feature_engineering(
    df: pd.DataFrame, technique: str, params: dict, target_col: str
) -> PipelineStepResult:
    df_out = df.copy()
    warnings = []
    cols_before = len(df_out.columns)
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
            warnings.append("Limited to first 5 numeric columns to avoid feature explosion.")

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

    elif technique == "reciprocal":
        skipped = []
        for col in numeric_cols:
            if (df_out[col] != 0).all():
                name = f"recip_{col}"
                df_out[name] = 1 / df_out[col]
                new_features.append(name)
            else:
                skipped.append(col)
        if skipped:
            warnings.append(f"Skipped reciprocal on: {skipped} (contain zero values).")
        warnings.append("Reciprocal transform (1/x) is useful for rate or speed features.")

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
            df_out[name] = pd.cut(df_out[col], bins=n_bins, labels=False, duplicates="drop")
            new_features.append(name)
        warnings.append(f"Discretised numeric columns into {n_bins} bins. Useful for capturing non-linear relationships.")

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

    elif technique == "date_decompose":
        date_cols = []
        for col in df_out.columns:
            if col == target_col:
                continue
            try:
                parsed = pd.to_datetime(df_out[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df_out[f"{col}_year"]    = parsed.dt.year
                    df_out[f"{col}_month"]   = parsed.dt.month
                    df_out[f"{col}_day"]     = parsed.dt.day
                    df_out[f"{col}_weekday"] = parsed.dt.weekday
                    new_features += [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_weekday"]
                    date_cols.append(col)
            except Exception:
                pass
        if not date_cols:
            warnings.append("No date/time columns detected. Date decomposition skipped.")
        else:
            warnings.append(f"Decomposed date columns: {date_cols} into year, month, day, weekday.")

    elif technique == "none":
        warnings.append("No feature engineering applied. Step skipped.")

    else:
        raise ValueError(f"Unknown feature engineering technique: '{technique}'")

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

def handle_feature_selection(
    df: pd.DataFrame, technique: str, params: dict, target_col: str
) -> PipelineStepResult:
    df_out = df.copy()
    warnings = []
    cols_before = len(df_out.columns)
    dropped: list[str] = []

    if target_col not in df_out.columns:
        warnings.append("Target column not found. Feature selection skipped.")
        return PipelineStepResult(
            step="feature_selection", technique=technique, params=params,
            stats={"cols_before": cols_before, "cols_after": cols_before, "dropped_columns": [], "n_dropped": 0},
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
            warnings.append(f"No features had variance below {threshold}. Nothing dropped.")

    elif technique == "correlation":
        threshold = params.get("threshold", 0.95)
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        df_out.drop(columns=to_drop, inplace=True)
        dropped = to_drop
        if not dropped:
            warnings.append(f"No feature pairs with correlation > {threshold} found.")

    elif technique == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest
        k = min(params.get("k", 10), len(X.columns))
        score_fn = mutual_info_classif if len(y.unique()) < 20 else mutual_info_regression
        sel = SelectKBest(score_fn, k=k)
        sel.fit(X, y)
        kept = X.columns[sel.get_support()].tolist()
        dropped = [c for c in X.columns if c not in kept]
        non_numeric = [c for c in df_out.columns if c not in X.columns and c != target_col]
        df_out = df_out[[c for c in kept + [target_col] + non_numeric if c in df_out.columns]]
        warnings.append(f"Kept top {k} features by mutual information score.")

    elif technique == "none":
        warnings.append("No feature selection applied. All features kept.")

    else:
        raise ValueError(f"Unknown feature selection technique: '{technique}'")

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

def handle_encoding(
    df: pd.DataFrame, technique: str, params: dict, target_col: str
) -> PipelineStepResult:
    df_out = df.copy()
    warnings = []
    cat_cols = [c for c in df_out.select_dtypes(exclude="number").columns if c != target_col]
    cols_before = len(df_out.columns)

    if not cat_cols:
        warnings.append("No categorical columns found. Encoding step skipped.")
        return PipelineStepResult(
            step="encoding", technique=technique, params=params,
            stats={"cols_before": cols_before, "cols_after": cols_before, "encoded_columns": [], "new_cols_created": 0},
            warnings=warnings,
        )

    if technique == "onehot":
        high_card = [c for c in cat_cols if df_out[c].nunique() > 20]
        if high_card:
            warnings.append(f"High cardinality columns {high_card} will create many features. Consider label encoding instead.")
        df_out = pd.get_dummies(df_out, columns=cat_cols, drop_first=False)

    elif technique == "label":
        warnings.append("Label encoding assigns arbitrary integers. Safe for tree models. Avoid with linear models.")
        for col in cat_cols:
            le = LabelEncoder()
            df_out[col] = le.fit_transform(df_out[col].astype(str))

    elif technique == "ordinal":
        warnings.append("Ordinal encoding implies order between categories. Only use when order is meaningful.")
        enc = OrdinalEncoder()
        df_out[cat_cols] = enc.fit_transform(df_out[cat_cols].astype(str))

    elif technique == "frequency":
        warnings.append("Frequency encoding replaces categories with their occurrence rate in the dataset.")
        for col in cat_cols:
            freq_map = df_out[col].value_counts(normalize=True).to_dict()
            df_out[col] = df_out[col].map(freq_map)

    elif technique == "target":
        warnings.append("Target encoding shown here uses full dataset means for display. Training uses train-set means only to prevent leakage.")
        if target_col in df_out.columns:
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
            "cols_before": cols_before,
            "cols_after": len(df_out.columns),
            "encoded_columns": cat_cols,
            "new_cols_created": len(df_out.columns) - cols_before,
        },
        warnings=warnings,
    )


# ─── SCALING ──────────────────────────────────────────────────────────────────

def handle_scaling(
    df: pd.DataFrame, technique: str, params: dict, target_col: str
) -> PipelineStepResult:
    df_out = df.copy()
    warnings = []
    numeric_cols = [c for c in df_out.select_dtypes(include="number").columns if c != target_col]

    if technique == "none":
        warnings.append("No scaling applied. Correct for tree-based models which are scale-invariant.")
        return PipelineStepResult(
            step="scaling", technique="none", params={},
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
    after_stats = df_out[numeric_cols].describe().to_dict()

    warnings.append("Scaler shown here fits on full data for display. During training, fits on X_train only.")

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


# ─── PCA ──────────────────────────────────────────────────────────────────────

def handle_pca(
    df: pd.DataFrame, technique: str, params: dict, target_col: str
) -> PipelineStepResult:
    from sklearn.decomposition import PCA
    df_out = df.copy()
    warnings = []

    numeric_cols = [c for c in df_out.select_dtypes(include="number").columns if c != target_col]
    cols_before = len(df_out.columns)

    # Pre-flight checks
    if technique == "none":
        warnings.append("No dimensionality reduction applied. Step skipped.")
        return PipelineStepResult(
            step="pca", technique="none", params={},
            stats={"cols_before": cols_before, "cols_after": cols_before, "components": 0},
            warnings=warnings,
        )

    null_count = int(df_out[numeric_cols].isnull().sum().sum())
    if null_count > 0:
        warnings.append(
            f"PCA requires no missing values. Found {null_count} nulls in numeric columns. "
            "Complete Step 3 (Missing Value Handling) before running PCA."
        )
        return PipelineStepResult(
            step="pca", technique=technique, params=params,
            stats={"cols_before": cols_before, "cols_after": cols_before, "components": 0,
                   "error": "Missing values present — complete Step 3 first."},
            warnings=warnings,
        )

    if len(numeric_cols) < 2:
        warnings.append("PCA requires at least 2 numeric features.")
        return PipelineStepResult(
            step="pca", technique=technique, params=params,
            stats={"cols_before": cols_before, "cols_after": cols_before, "components": 0},
            warnings=warnings,
        )

    try:
        if technique == "pca_auto":
            pca = PCA(n_components=0.95, random_state=42)
            components = pca.fit_transform(df_out[numeric_cols])
            n_comp = components.shape[1]
            comp_cols = [f"PC{i+1}" for i in range(n_comp)]
            df_pca = pd.DataFrame(components, columns=comp_cols, index=df_out.index)
            non_num = [c for c in df_out.columns if c not in numeric_cols]
            df_out = pd.concat([df_pca, df_out[non_num]], axis=1)
            explained = pca.explained_variance_ratio_
            warnings.append(
                f"Retained {n_comp} components explaining "
                f"{round(float(sum(explained))*100, 1)}% of total variance. "
                "Original feature names are replaced by PC1, PC2, etc."
            )
            return PipelineStepResult(
                step="pca", technique=technique, params=params,
                stats={
                    "cols_before": cols_before,
                    "cols_after": len(df_out.columns),
                    "components": n_comp,
                    "variance_explained": [round(float(v), 4) for v in explained],
                    "total_variance_retained": round(float(sum(explained)) * 100, 2),
                },
                warnings=warnings,
            )

        elif technique == "pca_fixed":
            n = min(params.get("n_components", 5), len(numeric_cols))
            pca = PCA(n_components=n, random_state=42)
            components = pca.fit_transform(df_out[numeric_cols])
            comp_cols = [f"PC{i+1}" for i in range(n)]
            df_pca = pd.DataFrame(components, columns=comp_cols, index=df_out.index)
            non_num = [c for c in df_out.columns if c not in numeric_cols]
            df_out = pd.concat([df_pca, df_out[non_num]], axis=1)
            explained = pca.explained_variance_ratio_
            warnings.append(
                f"{n} components explain {round(float(sum(explained))*100, 1)}% of total variance. "
                "Original feature names are replaced by PC1, PC2, etc."
            )
            return PipelineStepResult(
                step="pca", technique=technique, params=params,
                stats={
                    "cols_before": cols_before,
                    "cols_after": len(df_out.columns),
                    "components": n,
                    "variance_explained": [round(float(v), 4) for v in explained],
                    "total_variance_retained": round(float(sum(explained)) * 100, 2),
                },
                warnings=warnings,
            )

        else:
            raise ValueError(f"Unknown PCA technique: '{technique}'")

    except Exception as e:
        return PipelineStepResult(
            step="pca", technique=technique, params=params,
            stats={"cols_before": cols_before, "cols_after": cols_before, "components": 0,
                   "error": str(e)},
            warnings=[f"PCA failed: {str(e)}. Ensure missing values are handled and encoding is complete."],
        )