"""Structural causal model of a consumer-lending portfolio.

SYNTHETIC DATA. Every variable is produced by an explicit mechanism
``v := f_v(pa(v), noise)``, so the true causal graph and the exact location of
any injected fault are known. That ground truth is what makes root-cause
analysis *measurable*: the benchmark checks whether Quarry blames the mechanism
that was actually changed, not merely the column that moved the most.

Faults are applied *inside* the generating process (a changed mechanism
propagates to its descendants exactly as it would in the real world) or at the
observation layer (a logging/unit bug corrupts what the model sees without
changing the applicant).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "default"

REGIONS = ("north", "south", "east", "west", "central")
REGION_P = np.array([0.22, 0.20, 0.20, 0.18, 0.20])
CHANNELS = ("web", "branch", "partner")
CHANNEL_P = np.array([0.50, 0.30, 0.20])

REGION_UNEMPLOYMENT = {"north": 4.2, "south": 6.0, "east": 5.0, "west": 4.6, "central": 5.4}
REGION_INCOME = {"north": 0.10, "south": -0.12, "east": 0.0, "west": 0.06, "central": -0.04}
CHANNEL_LOAN = {"web": 0.0, "branch": 0.08, "partner": 0.25}
CHANNEL_RISK = {"web": 0.0, "branch": -0.20, "partner": 0.40}


@dataclass(frozen=True)
class Variable:
    name: str
    kind: str  # "categorical" | "continuous" | "count" | "binary"
    parents: tuple[str, ...]
    description: str
    unit: str = ""
    categories: tuple[str, ...] = ()
    hard_min: float | None = None
    hard_max: float | None = None


# Listed in a valid topological order of the true graph.
VARIABLES: tuple[Variable, ...] = (
    Variable("region", "categorical", (), "Applicant home region", categories=REGIONS),
    Variable("channel", "categorical", (), "Acquisition channel", categories=CHANNELS),
    Variable("age", "continuous", (), "Applicant age", "years", hard_min=18, hard_max=100),
    Variable("unemployment_rate", "continuous", ("region",), "Regional unemployment rate", "%", hard_min=0, hard_max=100),
    Variable("income", "continuous", ("age", "region"), "Reported annual income", "USD", hard_min=0),
    Variable("employment_years", "continuous", ("age",), "Years with current employer", "years", hard_min=0, hard_max=80),
    Variable("credit_history_months", "continuous", ("age",), "Length of credit history", "months", hard_min=0, hard_max=1000),
    Variable("loan_amount", "continuous", ("income", "channel"), "Requested principal", "USD", hard_min=0),
    Variable("debt_to_income", "continuous", ("loan_amount", "income"), "Debt service to income ratio", "ratio", hard_min=0),
    Variable("num_delinquencies", "count", ("unemployment_rate", "credit_history_months"), "Delinquencies, last 24 months", hard_min=0),
    Variable("credit_score", "continuous", ("num_delinquencies", "credit_history_months"), "Bureau credit score", "points", hard_min=300, hard_max=850),
    Variable(TARGET, "binary", ("debt_to_income", "credit_score", "unemployment_rate", "employment_years", "channel"), "Defaulted within 12 months"),
)

VARIABLE_BY_NAME = {v.name: v for v in VARIABLES}
FEATURES: tuple[str, ...] = tuple(v.name for v in VARIABLES if v.name != TARGET)
TRUE_EDGES: tuple[tuple[str, str], ...] = tuple((p, v.name) for v in VARIABLES for p in v.parents)


@dataclass(frozen=True)
class FaultDefinition:
    kind: str
    title: str
    category: str  # "covariate" | "mechanism" | "data_quality" | "concept"
    root_cause: str
    description: str
    magnitudes: dict[str, float]
    slice_text: str | None = None  # where the model is expected to fail, if anywhere

    def failure_slice(self, observed: pd.DataFrame) -> np.ndarray | None:
        """Boolean mask of the rows where the fault changes the model's error."""
        if self.kind == "regional_unemployment_shock":
            return (observed["region"] == "south").to_numpy()
        if self.kind == "employment_unit_bug":
            return (observed["channel"] == "partner").to_numpy()
        if self.kind == "bureau_outage":
            return (observed["channel"] == "web").to_numpy()
        if self.kind == "rate_hike_concept_drift":
            return (observed["debt_to_income"] >= DTI_STRESS_THRESHOLD).to_numpy()
        return None


DTI_STRESS_THRESHOLD = 0.28

FAULTS: dict[str, FaultDefinition] = {
    f.kind: f
    for f in (
        FaultDefinition(
            "channel_mix_shift", "Acquisition mix moves to partner channel", "covariate", "channel",
            "Marketing spend moves to a partner network: P(channel) shifts toward 'partner'. "
            "Every mechanism is unchanged; only the population mix moves.",
            {"low": 0.25, "medium": 0.5, "high": 0.8},
        ),
        FaultDefinition(
            "regional_unemployment_shock", "Unemployment shock in the south", "mechanism", "unemployment_rate",
            "A plant closure raises unemployment in region=south. The shift propagates to "
            "delinquencies, credit scores and defaults through the causal graph.",
            {"low": 1.5, "medium": 3.0, "high": 6.0}, slice_text="region = south",
        ),
        FaultDefinition(
            "employment_unit_bug", "Partner feed reports employment in months", "data_quality", "employment_years",
            "A partner integration starts sending employment tenure in months instead of years. "
            "Applicants are unchanged; only the logged feature is wrong.",
            {"low": 0.3, "medium": 0.6, "high": 1.0}, slice_text="channel = partner",
        ),
        FaultDefinition(
            "bureau_outage", "Credit bureau outage on web traffic", "data_quality", "credit_score",
            "The bureau API times out for web applications; credit_score arrives null.",
            {"low": 0.15, "medium": 0.35, "high": 0.7}, slice_text="channel = web",
        ),
        FaultDefinition(
            "rate_hike_concept_drift", "Rate hike stresses high-DTI borrowers", "concept", TARGET,
            "Interest rates rise: the same debt-to-income ratio now carries more default risk. "
            "Inputs do not move at all; only P(default | features) changes.",
            {"low": 4.0, "medium": 8.0, "high": 14.0}, slice_text=f"debt_to_income >= {DTI_STRESS_THRESHOLD}",
        ),
        FaultDefinition(
            "young_applicant_influx", "Influx of young applicants", "covariate", "age",
            "A student-loan promotion brings in young applicants. Age is a root cause of income, "
            "tenure and credit history, so many downstream columns drift together.",
            {"low": 0.15, "medium": 0.3, "high": 0.5},
        ),
    )
}


@dataclass(frozen=True)
class ActiveFault:
    kind: str
    magnitude: float
    strength: float = 1.0  # 0..1 ramp multiplier

    @property
    def effective(self) -> float:
        return self.magnitude * self.strength


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample(n: int, rng: np.random.Generator, fault: ActiveFault | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Draw ``n`` applications. Returns (observed features, default labels)."""
    kind = fault.kind if fault is not None and fault.strength > 0 else None
    eff = fault.effective if kind else 0.0
    strength = fault.strength if kind else 0.0

    channel_p = CHANNEL_P
    if kind == "channel_mix_shift":
        channel_p = (1 - eff) * CHANNEL_P + eff * np.array([0.20, 0.10, 0.70])
    region = rng.choice(np.array(REGIONS, dtype=object), size=n, p=REGION_P)
    channel = rng.choice(np.array(CHANNELS, dtype=object), size=n, p=channel_p)

    age = np.clip(rng.normal(41.0, 12.0, n), 21, 75)
    if kind == "young_applicant_influx":
        young = rng.random(n) < eff
        age = np.where(young, np.clip(rng.normal(24.5, 2.5, n), 21, 75), age)

    unemp = np.array([REGION_UNEMPLOYMENT[r] for r in region]) + rng.normal(0, 0.8, n)
    if kind == "regional_unemployment_shock":
        unemp = unemp + np.where(region == "south", eff, 0.0)
    unemp = np.clip(unemp, 1.0, 30.0)

    a = age - 40.0
    log_income = 10.75 + 0.012 * a - 0.00035 * a**2 + np.array([REGION_INCOME[r] for r in region]) + rng.normal(0, 0.40, n)
    income = np.exp(log_income)

    employment_years = np.clip(0.33 * (age - 21) + rng.normal(0, 3.5, n), 0, None)
    employment_years = np.minimum(employment_years, age - 18)
    credit_history = np.clip(9.0 * (age - 21) + rng.normal(0, 36.0, n), 6, 640)

    loan_ratio = np.exp(rng.normal(np.log(0.28) + np.array([CHANNEL_LOAN[c] for c in channel]), 0.35, n))
    loan_amount = np.clip(income * loan_ratio, 1_000, 400_000)
    dti = loan_amount / income * 0.55 + 0.25 * rng.beta(2.0, 5.0, n)

    lam = np.exp(-1.55 + 0.20 * (unemp - 5.0) - 0.0025 * (credit_history - 180.0))
    delinquencies = rng.poisson(lam).astype(float)
    credit_score = np.clip(
        728.0 - 42.0 * delinquencies + 0.10 * (credit_history - 180.0) + rng.normal(0, 32.0, n), 300, 850
    )

    logit = (
        -2.05
        + 7.0 * (dti - 0.23)
        - 0.016 * (credit_score - 700.0)
        + 0.22 * (unemp - 5.0)
        - 0.045 * (employment_years - 6.0)
        + np.array([CHANNEL_RISK[c] for c in channel])
    )
    if kind == "rate_hike_concept_drift":
        logit = logit + eff * np.maximum(dti - DTI_STRESS_THRESHOLD, 0.0) + 0.15 * strength
    y = (rng.random(n) < _sigmoid(logit)).astype(int)

    observed = pd.DataFrame(
        {
            "region": region,
            "channel": channel,
            "age": np.round(age, 1),
            "unemployment_rate": np.round(unemp, 2),
            "income": np.round(income, 0),
            "employment_years": np.round(employment_years, 1),
            "credit_history_months": np.round(credit_history, 0),
            "loan_amount": np.round(loan_amount, 0),
            "debt_to_income": np.round(dti, 4),
            "num_delinquencies": delinquencies,
            "credit_score": np.round(credit_score, 0),
        }
    )

    # Observation-layer faults: the applicant is unchanged, the logged value is not.
    if kind == "employment_unit_bug":
        hit = (channel == "partner") & (rng.random(n) < eff)
        observed.loc[hit, "employment_years"] = np.round(employment_years[hit] * 12.0, 1)
    elif kind == "bureau_outage":
        hit = (channel == "web") & (rng.random(n) < eff)
        observed.loc[hit, "credit_score"] = np.nan

    return observed, y


def sample_reference(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Fault-free data from the reference period, in time order."""
    return sample(n, np.random.default_rng(seed))
