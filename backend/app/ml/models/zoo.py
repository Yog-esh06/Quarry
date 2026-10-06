"""Model zoo: baselines, the production GBM and the deep ensemble.

Each algorithm is a scikit-learn Pipeline (preprocessing + estimator) so the
exact transform used in training is the one serialised with the model.

* ``logistic_regression`` - linear baseline; strong when effects are additive.
* ``random_forest``       - non-linear baseline, robust, poorly calibrated.
* ``hist_gradient_boosting`` - default production model: handles mixed types
  and missing values natively, fast on CPU.
* ``mlp_ensemble``        - PyTorch deep ensemble, adds epistemic uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.ml.data.schema import DatasetSchema, coerce_frame
from app.ml.models.calibration import IsotonicCalibrator, PlattCalibrator
from app.ml.models.conformal import SplitConformalClassifier
from app.ml.models.ensemble import MLPEnsembleClassifier, torch_available


def dense_preprocessor(schema: DatasetSchema) -> ColumnTransformer:
    """One-hot + median imputation with missing indicators + standardisation."""
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), schema.numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                schema.categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def tree_preprocessor(schema: DatasetSchema) -> ColumnTransformer:
    """Categoricals to integer codes (unknown/missing -> NaN); numerics untouched."""
    return ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    categories=[schema.get(c).categories for c in schema.categorical],
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                ),
                schema.categorical,
            ),
            ("num", "passthrough", schema.numeric),
        ],
        verbose_feature_names_out=False,
    )


def _forest_preprocessor(schema: DatasetSchema) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), schema.numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                schema.categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    label: str
    role: str  # "baseline" | "production" | "advanced"
    grid: list[dict[str, Any]]
    requires_torch: bool = False


ALGORITHMS: dict[str, AlgorithmSpec] = {
    a.name: a
    for a in (
        AlgorithmSpec("logistic_regression", "Logistic regression", "baseline", [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]),
        AlgorithmSpec("random_forest", "Random forest", "baseline", [{"min_samples_leaf": 5}, {"min_samples_leaf": 25}]),
        AlgorithmSpec(
            "hist_gradient_boosting",
            "Histogram gradient boosting",
            "production",
            [
                {"learning_rate": 0.05, "max_leaf_nodes": 15},
                {"learning_rate": 0.05, "max_leaf_nodes": 31},
                {"learning_rate": 0.1, "max_leaf_nodes": 15},
            ],
        ),
        AlgorithmSpec(
            "mlp_ensemble",
            "MLP deep ensemble (PyTorch)",
            "advanced",
            [{"hidden": 64, "weight_decay": 1e-4}, {"hidden": 32, "weight_decay": 1e-3}],
            requires_torch=True,
        ),
    )
}


def available_algorithms() -> list[str]:
    return [a for a, spec in ALGORITHMS.items() if not spec.requires_torch or torch_available()]


def build_pipeline(algorithm: str, schema: DatasetSchema, params: dict[str, Any] | None = None, seed: int = 0) -> Pipeline:
    params = dict(params or {})
    if algorithm == "logistic_regression":
        est = LogisticRegression(max_iter=2000, **params)
        return Pipeline([("prep", dense_preprocessor(schema)), ("model", est)])
    if algorithm == "random_forest":
        est = RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=seed, **params)
        return Pipeline([("prep", _forest_preprocessor(schema)), ("model", est)])
    if algorithm == "hist_gradient_boosting":
        n_cat = len(schema.categorical)
        mask = [True] * n_cat + [False] * len(schema.numeric)
        est = HistGradientBoostingClassifier(
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            l2_regularization=1.0,
            min_samples_leaf=40,
            categorical_features=mask,
            random_state=seed,
            **params,
        )
        return Pipeline([("prep", tree_preprocessor(schema)), ("model", est)])
    if algorithm == "mlp_ensemble":
        est = MLPEnsembleClassifier(random_state=seed, **params)
        return Pipeline([("prep", dense_preprocessor(schema)), ("model", est)])
    raise ValueError(f"unknown algorithm '{algorithm}'")


def fit_pipeline(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Pipeline:
    if sample_weight is None:
        return pipe.fit(X, y)
    return pipe.fit(X, y, model__sample_weight=sample_weight)


@dataclass
class QuarryModel:
    """A deployable model: pipeline + calibrator + conformal calibrator + decision threshold."""

    algorithm: str
    params: dict[str, Any]
    pipeline: Pipeline
    schema: DatasetSchema
    calibrator: IsotonicCalibrator | PlattCalibrator | None = None
    conformal: SplitConformalClassifier | None = None
    threshold: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return coerce_frame(df, self.schema)

    def raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self._frame(df))[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.raw_proba(df)
        return self.calibrator.transform(p) if self.calibrator is not None else p

    def calibrate_scores(self, raw: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(raw) if self.calibrator is not None else raw

    @property
    def has_uncertainty(self) -> bool:
        return isinstance(self.pipeline.named_steps["model"], MLPEnsembleClassifier)

    def epistemic_uncertainty(self, df: pd.DataFrame) -> np.ndarray | None:
        if not self.has_uncertainty:
            return None
        Z = self.pipeline.named_steps["prep"].transform(self._frame(df))
        return self.pipeline.named_steps["model"].mutual_information(Z)

    def prediction_sets(self, p: np.ndarray, alpha: float = 0.1) -> np.ndarray | None:
        return self.conformal.predict_sets(p, alpha) if self.conformal is not None else None
