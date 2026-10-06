"""Batch data validation against the reference schema.

Validation answers "is this batch even interpretable?" before any drift
statistic is computed. Hard failures (missing columns, impossible values)
reject the batch; statistical anomalies (null-rate spikes, out-of-reference
values, unseen categories) are recorded and later used to classify incidents
as *data-quality* problems, which must not be "fixed" by retraining.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from app.ml.data.schema import DatasetSchema

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    feature: str | None
    status: str
    observed: float | None
    expected: float | None
    message: str


@dataclass
class ValidationReport:
    status: str
    n_rows: int
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def anomalous_features(self) -> set[str]:
        return {c.feature for c in self.checks if c.status != PASS and c.feature}

    def to_dict(self) -> dict:
        return {"status": self.status, "n_rows": self.n_rows, "checks": [asdict(c) for c in self.checks]}


class SchemaError(ValueError):
    """The batch cannot be interpreted with the deployment's schema."""


def _proportion_z(p_ref: float, k: int, n: int) -> float:
    """One-sided z statistic for an increase over a reference proportion."""
    p0 = min(max(p_ref, 1e-3), 1 - 1e-3)
    return (k / max(n, 1) - p0) / np.sqrt(p0 * (1 - p0) / max(n, 1))


def validate_batch(df: pd.DataFrame, schema: DatasetSchema, min_rows: int = 30) -> ValidationReport:
    checks: list[Check] = []
    n = len(df)
    missing_cols = [f for f in schema.feature_names if f not in df.columns]
    if missing_cols:
        raise SchemaError(f"batch is missing required columns: {', '.join(missing_cols)}")
    extra = [c for c in df.columns if c not in schema.feature_names and c != schema.target]
    checks.append(
        Check("unexpected_columns", None, WARN if extra else PASS, float(len(extra)), 0.0,
              f"ignored columns: {', '.join(map(str, extra[:8]))}" if extra else "no unexpected columns")
    )
    checks.append(
        Check("batch_size", None, FAIL if n < min_rows else PASS, float(n), float(min_rows),
              f"{n} rows" + (f" (< {min_rows} minimum)" if n < min_rows else ""))
    )
    if n == 0:
        return ValidationReport(FAIL, 0, checks)

    for f in schema.features:
        col = df[f.name]
        nulls = int(col.isna().sum())
        # Null-rate spike relative to the reference period.
        z = _proportion_z(f.ref_null_rate, nulls, n)
        rate = nulls / n
        delta = rate - f.ref_null_rate
        status = PASS
        if (not f.nullable and nulls) or (z > 6 and delta > 0.05):
            status = FAIL
        elif z > 3.5 and delta > 0.01:
            status = WARN
        checks.append(Check("null_rate", f.name, status, rate, f.ref_null_rate,
                            f"{rate:.1%} null vs {f.ref_null_rate:.1%} in reference"))

        values = col.dropna()
        if f.is_categorical:
            unseen = ~values.astype(str).isin(f.categories)
            k = int(unseen.sum())
            status = FAIL if k / max(len(values), 1) > 0.05 else WARN if k else PASS
            checks.append(Check("unseen_categories", f.name, status, k / max(len(values), 1), 0.0,
                                f"{k} values outside {{{', '.join(f.categories)}}}" if k else "all categories known"))
            continue

        num = pd.to_numeric(values, errors="coerce")
        bad_type = int(num.isna().sum())
        if bad_type:
            checks.append(Check("type", f.name, FAIL, float(bad_type), 0.0, f"{bad_type} non-numeric values"))
        num = num.dropna().to_numpy(dtype=float)
        hard = np.zeros(len(num), dtype=bool)
        if f.hard_min is not None:
            hard |= num < f.hard_min
        if f.hard_max is not None:
            hard |= num > f.hard_max
        k_hard = int(hard.sum())
        checks.append(Check("hard_range", f.name, FAIL if k_hard else PASS, float(k_hard), 0.0,
                            f"{k_hard} values outside [{f.hard_min}, {f.hard_max}]" if k_hard else "within declared bounds"))
        if f.ref_low is not None and f.ref_high is not None and len(num):
            outside = int(((num < f.ref_low) | (num > f.ref_high)).sum())
            # 1% of reference values fall outside [p0.5, p99.5] by construction.
            z_out = _proportion_z(0.01, outside, len(num))
            rate_out = outside / len(num)
            status = FAIL if (z_out > 8 and rate_out > 0.08) else WARN if z_out > 4 else PASS
            checks.append(Check("reference_range", f.name, status, rate_out, 0.01,
                                f"{rate_out:.1%} outside reference range [{f.ref_low:.4g}, {f.ref_high:.4g}]"))

    overall = FAIL if any(c.status == FAIL for c in checks) else WARN if any(c.status == WARN for c in checks) else PASS
    return ValidationReport(overall, n, checks)


def reject_reason(report: ValidationReport) -> str | None:
    """Only structurally uninterpretable batches are rejected.

    Statistical failures (null spikes, range explosions) are *recorded*, not
    rejected: the batch is real traffic the model already scored, so it must be
    monitored and explained rather than silently dropped.
    """
    hard = [c for c in report.checks if c.status == FAIL and c.name in {"batch_size", "type"}]
    return "; ".join(c.message for c in hard) if hard else None


def null_rate_pvalue(k_ref: int, n_ref: int, k_cur: int, n_cur: int) -> float:
    """Two-sided Fisher exact test for a change in missingness."""
    if n_ref == 0 or n_cur == 0:
        return 1.0
    if k_ref == 0 and k_cur == 0:
        return 1.0
    table = [[k_ref, n_ref - k_ref], [k_cur, n_cur - k_cur]]
    return float(stats.fisher_exact(table)[1])
