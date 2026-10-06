"""Dataset schema: the contract between producers of data and every model.

The schema carries both declared constraints (hard bounds, categories) and
statistics learned from the reference period (soft range, null rate). Data
validation and all encoders read from it, so a column can never be interpreted
two different ways in two places.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

CATEGORICAL_KINDS = {"categorical"}
NUMERIC_KINDS = {"continuous", "count", "binary"}


@dataclass
class FeatureSpec:
    name: str
    kind: str
    description: str = ""
    unit: str = ""
    categories: list[str] = field(default_factory=list)
    hard_min: float | None = None
    hard_max: float | None = None
    ref_low: float | None = None  # reference 0.5th percentile
    ref_high: float | None = None  # reference 99.5th percentile
    ref_null_rate: float = 0.0
    nullable: bool = True

    @property
    def is_categorical(self) -> bool:
        return self.kind in CATEGORICAL_KINDS


@dataclass
class DatasetSchema:
    features: list[FeatureSpec]
    target: str
    synthetic: bool = False

    def __post_init__(self) -> None:
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError("duplicate feature names in schema")
        if self.target in names:
            raise ValueError("target must not be listed as a feature")
        self._by_name = {f.name: f for f in self.features}

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def categorical(self) -> list[str]:
        return [f.name for f in self.features if f.is_categorical]

    @property
    def numeric(self) -> list[str]:
        return [f.name for f in self.features if not f.is_categorical]

    def get(self, name: str) -> FeatureSpec:
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def with_reference_stats(self, df: pd.DataFrame) -> "DatasetSchema":
        """Learn soft ranges, null rates and category sets from reference data."""
        for f in self.features:
            col = df[f.name]
            f.ref_null_rate = float(col.isna().mean())
            if f.is_categorical:
                if not f.categories:
                    f.categories = sorted(map(str, col.dropna().unique()))
            else:
                values = pd.to_numeric(col, errors="coerce").dropna().to_numpy(dtype=float)
                if len(values):
                    f.ref_low = float(np.quantile(values, 0.005))
                    f.ref_high = float(np.quantile(values, 0.995))
        return self

    def to_dict(self) -> dict:
        return {"target": self.target, "synthetic": self.synthetic, "features": [asdict(f) for f in self.features]}

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSchema":
        return cls([FeatureSpec(**f) for f in d["features"]], d["target"], d.get("synthetic", False))

    @classmethod
    def infer(cls, df: pd.DataFrame, target: str, max_categories: int = 30) -> "DatasetSchema":
        """Infer a schema for an uploaded CSV (bring-your-own-data path)."""
        if target not in df.columns:
            raise ValueError(f"target column '{target}' not found")
        feats: list[FeatureSpec] = []
        for name in df.columns:
            if name == target:
                continue
            col = df[name]
            numeric = pd.to_numeric(col, errors="coerce")
            parse_rate = numeric.notna().sum() / max(col.notna().sum(), 1)
            if parse_rate > 0.98:
                is_int = bool(np.all(np.mod(numeric.dropna(), 1) == 0))
                kind = "count" if is_int and numeric.min() >= 0 and numeric.nunique() > 2 else "continuous"
                feats.append(FeatureSpec(name, kind))
            elif col.nunique(dropna=True) <= max_categories:
                feats.append(FeatureSpec(name, "categorical"))
            else:
                raise ValueError(f"column '{name}' is neither numeric nor low-cardinality categorical")
        return cls(feats, target).with_reference_stats(df)


def lending_schema() -> DatasetSchema:
    """Schema of the synthetic lending SCM (see app.ml.simulation.scm)."""
    from app.ml.simulation.scm import TARGET, VARIABLES

    feats = [
        FeatureSpec(
            v.name,
            v.kind,
            description=v.description,
            unit=v.unit,
            categories=list(v.categories),
            hard_min=v.hard_min,
            hard_max=v.hard_max,
        )
        for v in VARIABLES
        if v.name != TARGET
    ]
    return DatasetSchema(feats, TARGET, synthetic=True)


def coerce_frame(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Return the schema's feature columns with canonical dtypes."""
    out = {}
    for f in schema.features:
        col = df[f.name] if f.name in df.columns else pd.Series(np.nan, index=df.index)
        if f.is_categorical:
            out[f.name] = col.astype(object).where(col.notna(), None).map(lambda v: None if v is None else str(v))
        else:
            out[f.name] = pd.to_numeric(col, errors="coerce").astype(float)
    return pd.DataFrame(out, index=df.index)


def encode_frame(df: pd.DataFrame, schema: DatasetSchema, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Numeric matrix for tree models: categoricals become integer codes, missing stays NaN.

    Returns (X, categorical_mask). Unknown categories become NaN, which
    histogram GBMs route as missing values.
    """
    X = np.empty((len(df), len(columns)), dtype=float)
    mask = np.zeros(len(columns), dtype=bool)
    for j, name in enumerate(columns):
        if name in schema and schema.get(name).is_categorical:
            cats = schema.get(name).categories
            lookup = {c: float(i) for i, c in enumerate(cats)}
            X[:, j] = df[name].map(lambda v: lookup.get(v, np.nan) if v is not None else np.nan).to_numpy(dtype=float)
            mask[j] = True
        else:
            X[:, j] = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return X, mask
