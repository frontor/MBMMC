from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier


def build_model(model_name: str, random_state: int = 42) -> RandomForestClassifier:
    """Build the publication Random Forest model."""
    if model_name.lower() != "rf":
        raise ValueError(
            "This publication repository retains only the Random Forest model. "
            "Use model_name='rf'."
        )
    return RandomForestClassifier(
        n_estimators=600,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def param_distributions(model_name: str) -> Dict[str, Any]:
    """RandomizedSearchCV parameter distributions for the RF pipeline."""
    if model_name.lower() != "rf":
        raise ValueError(
            "This publication repository retains only the Random Forest model."
        )
    return {
        "clf__n_estimators": [300, 600, 1000, 1500],
        "clf__max_depth": [None, 10, 20, 40, 60],
        "clf__max_features": ["sqrt", "log2", 0.05, 0.1, 0.2],
        "clf__min_samples_leaf": [1, 2, 4, 8],
    }
