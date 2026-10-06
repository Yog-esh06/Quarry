"""Deep ensemble of small MLPs (PyTorch, CPU by default).

Why it exists: gradient-boosted trees give a point probability but no notion
of *epistemic* uncertainty. Five independently initialised MLPs disagree
precisely where the training data was thin, so their disagreement (mutual
information between prediction and member) rises when production traffic
moves off the training manifold. Quarry tracks that as a label-free signal and
the benchmark measures whether it adds anything beyond the domain classifier.

PyTorch is optional: ``torch_available()`` gates every use.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

try:  # pragma: no cover - exercised implicitly by the torch-marked tests
    import torch
    from torch import nn

    _TORCH = True
except Exception:  # noqa: BLE001 - any import failure means "no torch"
    _TORCH = False


def torch_available() -> bool:
    return _TORCH


def resolve_device(preference: str | None = None) -> str:
    pref = (preference or os.getenv("QUARRY_DEVICE", "auto")).lower()
    if not _TORCH:
        return "cpu"
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _make_member(d_in: int, hidden: int, dropout: float) -> "nn.Module":
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


class MLPEnsembleClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_members: int = 5,
        hidden: int = 64,
        epochs: int = 20,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        dropout: float = 0.1,
        random_state: int = 0,
        device: str | None = None,
    ):
        self.n_members = n_members
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state
        self.device = device

    def fit(self, X, y, sample_weight=None):
        if not _TORCH:
            raise RuntimeError("PyTorch is not installed; install the 'torch' extra")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        dev = resolve_device(self.device)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        w = np.ones(len(y), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        Xt, yt, wt = (torch.from_numpy(a).to(dev) for a in (X, y, w / w.mean()))
        self.members_ = []
        for m in range(self.n_members):
            gen = torch.Generator().manual_seed(self.random_state * 1000 + m)
            torch.manual_seed(self.random_state * 1000 + m)
            net = _make_member(X.shape[1], self.hidden, self.dropout).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            n = len(yt)
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(n, generator=gen).to(dev)
                for start in range(0, n, self.batch_size):
                    idx = perm[start : start + self.batch_size]
                    opt.zero_grad()
                    loss = (loss_fn(net(Xt[idx]).squeeze(1), yt[idx]) * wt[idx]).mean()
                    loss.backward()
                    opt.step()
            net.eval()
            self.members_.append(net.to("cpu"))
        return self

    def member_proba(self, X) -> np.ndarray:
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            return np.stack([torch.sigmoid(net(X).squeeze(1)).numpy() for net in self.members_])

    def predict_proba(self, X) -> np.ndarray:
        p = self.member_proba(X).mean(axis=0)
        return np.stack([1 - p, p], axis=1)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def mutual_information(self, X) -> np.ndarray:
        """BALD epistemic uncertainty: H[mean p] - mean H[p_m] (nats)."""
        pm = np.clip(self.member_proba(X), 1e-6, 1 - 1e-6)
        pbar = pm.mean(axis=0)

        def h(p):
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))

        return h(pbar) - h(pm).mean(axis=0)

    # Persist weights as numpy arrays so artifacts do not pin a CUDA device.
    def __getstate__(self):
        state = self.__dict__.copy()
        if "members_" in state:
            state["members_"] = [
                {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()} for net in self.members_
            ]
        return state

    def __setstate__(self, state):
        members = state.pop("members_", None)
        self.__dict__.update(state)
        if members is not None:
            if not _TORCH:
                raise RuntimeError("this artifact contains a PyTorch ensemble; install the 'torch' extra")
            self.members_ = []
            for sd in members:
                net = _make_member(self.n_features_in_, self.hidden, self.dropout)
                net.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})
                net.eval()
                self.members_.append(net)
