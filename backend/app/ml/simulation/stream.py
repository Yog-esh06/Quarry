"""Deterministic production streams with scheduled faults (SYNTHETIC DATA).

A scenario is a sequence of daily batches. Batch ``t`` is a pure function of
``(seed, t)``, so any batch can be regenerated for audits and the benchmark is
exactly reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from app.ml.simulation.scm import FAULTS, ActiveFault, sample


@dataclass(frozen=True)
class ScenarioSpec:
    fault: str | None = None  # key of FAULTS, or None for a fault-free control stream
    severity: str = "medium"
    onset: int = 15
    ramp: int = 1  # batches to reach full strength; 1 = abrupt
    n_batches: int = 40
    batch_size: int = 800
    label_delay: int = 3  # batches until ground-truth labels arrive
    seed: int = 0
    start: datetime = field(default_factory=lambda: datetime(2026, 1, 5, tzinfo=timezone.utc))

    def __post_init__(self) -> None:
        if self.fault is not None and self.fault not in FAULTS:
            raise ValueError(f"unknown fault '{self.fault}'")
        if self.fault is not None and self.severity not in FAULTS[self.fault].magnitudes:
            raise ValueError(f"unknown severity '{self.severity}'")
        if self.ramp < 1 or self.batch_size < 10 or self.n_batches < 1:
            raise ValueError("ramp >= 1, batch_size >= 10 and n_batches >= 1 are required")

    @property
    def magnitude(self) -> float:
        return FAULTS[self.fault].magnitudes[self.severity] if self.fault else 0.0

    def strength(self, t: int) -> float:
        if self.fault is None or t < self.onset:
            return 0.0
        return min(1.0, (t - self.onset + 1) / self.ramp)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSpec":
        d = dict(d)
        if isinstance(d.get("start"), str):
            d["start"] = datetime.fromisoformat(d["start"])
        return cls(**d)


@dataclass
class Batch:
    index: int
    window_start: datetime
    features: pd.DataFrame
    labels: np.ndarray
    fault_strength: float


class ScenarioStream:
    def __init__(self, spec: ScenarioSpec):
        self.spec = spec

    def batch(self, t: int) -> Batch:
        spec = self.spec
        rng = np.random.default_rng([spec.seed, 7919, t])
        strength = spec.strength(t)
        fault = ActiveFault(spec.fault, spec.magnitude, strength) if spec.fault and strength > 0 else None
        x, y = sample(spec.batch_size, rng, fault)
        return Batch(t, spec.start + timedelta(days=t), x, y, strength)

    def __iter__(self):
        for t in range(self.spec.n_batches):
            yield self.batch(t)
