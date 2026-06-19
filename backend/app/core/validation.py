"""
Input validation for ML Playground API endpoints.
Provides reusable validators for common input patterns.
"""
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, validator, Field


class PaginationParams(BaseModel):
    """Standard pagination parameters."""
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(20, ge=1, le=100, description="Items per page (max 100)")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class ModelParamsValidator:
    """Validate model hyperparameters."""

    MODEL_CONSTRAINTS = {
        "random_forest": {
            "n_estimators": (10, 1000),
            "max_depth": (1, 100),
            "min_samples_split": (2, 100),
            "min_samples_leaf": (1, 100),
            "max_features": ["sqrt", "log2", "auto", None],
        },
        "xgboost": {
            "n_estimators": (10, 1000),
            "max_depth": (1, 50),
            "learning_rate": (0.001, 1.0),
            "subsample": (0.1, 1.0),
            "colsample_bytree": (0.1, 1.0),
        },
        "lightgbm": {
            "n_estimators": (10, 1000),
            "max_depth": (1, 50),
            "learning_rate": (0.001, 1.0),
            "num_leaves": (2, 512),
        },
        "logistic_regression": {
            "C": (0.001, 1000.0),
            "max_iter": (100, 10000),
            "penalty": ["l1", "l2", "elasticnet", "none"],
        },
        "svm": {
            "C": (0.01, 1000.0),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"],
        },
        "knn": {
            "n_neighbors": (1, 200),
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        "decision_tree": {
            "max_depth": (1, 100),
            "min_samples_split": (2, 100),
            "min_samples_leaf": (1, 100),
            "criterion": ["gini", "entropy", "log_loss"],
        },
        "gradient_boosting": {
            "n_estimators": (10, 1000),
            "max_depth": (1, 50),
            "learning_rate": (0.001, 1.0),
            "subsample": (0.1, 1.0),
        },
    }

    @classmethod
    def validate(cls, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp model parameters to safe ranges."""
        if model_name not in cls.MODEL_CONSTRAINTS:
            return params  # Allow unknown models

        constraints = cls.MODEL_CONSTRAINTS[model_name]
        validated = {}

        for key, value in params.items():
            if key not in constraints:
                continue  # Skip unknown params

            constraint = constraints[key]

            if isinstance(constraint, tuple):
                min_val, max_val = constraint
                if isinstance(value, (int, float)):
                    validated[key] = max(min_val, min(max_val, value))
            elif isinstance(constraint, list):
                if value in constraint:
                    validated[key] = value
                else:
                    validated[key] = constraint[0]  # Default to first option
            else:
                validated[key] = value

        return validated


class PreprocessingParamsValidator:
    """Validate preprocessing step parameters."""

    @staticmethod
    def validate_missing(technique: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate missing values imputation parameters."""
        validated = dict(params)

        if technique == "knn":
            validated["n_neighbors"] = max(1, min(50, int(params.get("n_neighbors", 5))))
        elif technique == "constant":
            validated["fill_value"] = params.get("fill_value", 0)
        elif technique == "drop_cols":
            validated["threshold"] = max(0.0, min(1.0, float(params.get("threshold", 0.5))))

        return validated

    @staticmethod
    def validate_outliers(technique: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate outlier treatment parameters."""
        validated = dict(params)

        if technique == "zscore_remove":
            validated["threshold"] = max(1.0, min(10.0, float(params.get("threshold", 3.0))))
        elif technique == "percentile_cap":
            validated["lower"] = max(0.0, min(50.0, float(params.get("lower", 1.0))))
            validated["upper"] = max(50.0, min(100.0, float(params.get("upper", 99.0))))

        return validated

    @staticmethod
    def validate_features(technique: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature engineering parameters."""
        validated = dict(params)

        if technique == "binning":
            validated["n_bins"] = max(2, min(50, int(params.get("n_bins", 5))))
        elif technique == "polynomial":
            validated["degree"] = max(2, min(5, int(params.get("degree", 2))))

        return validated

    @staticmethod
    def validate_selection(technique: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature selection parameters."""
        validated = dict(params)

        if technique == "variance_threshold":
            validated["threshold"] = max(0.0, min(1.0, float(params.get("threshold", 0.01))))
        elif technique == "correlation":
            validated["threshold"] = max(0.5, min(1.0, float(params.get("threshold", 0.95))))
        elif technique == "mutual_info":
            validated["k"] = max(1, min(100, int(params.get("k", 10))))

        return validated

    @staticmethod
    def validate_pca(technique: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate PCA parameters."""
        validated = dict(params)

        if technique == "pca_fixed":
            validated["n_components"] = max(1, min(100, int(params.get("n_components", 5))))

        return validated


def validate_session_id(session_id: str) -> bool:
    """Validate UUID format for session IDs."""
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(session_id))


def sanitize_string(value: str, max_length: int = 500) -> str:
    """Sanitize string input to prevent injection attacks."""
    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f]', '', value)
    # Truncate to max length
    return sanitized[:max_length]