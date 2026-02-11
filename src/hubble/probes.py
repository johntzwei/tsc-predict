"""Registry of probes that bundle feature extraction config with a classifier."""

from dataclasses import dataclass
from typing import Callable

from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeSpec:
    """Specifies both how to extract features and which classifier to train."""
    name: str
    layer: int          # which hidden layer (-1 = last)
    pool: str           # "mean" or "last"
    classifier_fn: Callable  # (seed) -> sklearn estimator

    @property
    def feature_key(self) -> str:
        """Cache key for the feature extraction config."""
        return f"layer{self.layer}_pool{self.pool}"

    def make_classifier(self, seed: int = 42):
        return self.classifier_fn(seed)


def _logreg(seed):
    return LogisticRegressionCV(
        Cs=10, cv=5, solver="lbfgs", max_iter=1000,
        scoring="accuracy", random_state=seed, n_jobs=-1,
    )


def _logreg_balanced(seed):
    return LogisticRegressionCV(
        Cs=10, cv=5, solver="lbfgs", max_iter=1000,
        scoring="accuracy", class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )


def _logreg_scaled(seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=10, cv=5, solver="lbfgs", max_iter=1000,
            scoring="accuracy", random_state=seed, n_jobs=-1,
        )),
    ])


def _logreg_scaled_balanced(seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=10, cv=5, solver="lbfgs", max_iter=1000,
            scoring="accuracy", class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )),
    ])


PROBES = {
    "logreg_last": ProbeSpec("logreg_last", layer=-1, pool="mean", classifier_fn=_logreg),
    "logreg_balanced_last": ProbeSpec("logreg_balanced_last", layer=-1, pool="mean", classifier_fn=_logreg_balanced),
    "logreg_scaled_last": ProbeSpec("logreg_scaled_last", layer=-1, pool="mean", classifier_fn=_logreg_scaled),
    "logreg_scaled_balanced_last": ProbeSpec("logreg_scaled_balanced_last", layer=-1, pool="mean", classifier_fn=_logreg_scaled_balanced),
}


def get_probe(name: str) -> ProbeSpec:
    """Return a ProbeSpec by name."""
    if name not in PROBES:
        raise ValueError(f"Unknown probe '{name}'. Available: {list_probes()}")
    return PROBES[name]


def list_probes() -> list[str]:
    """Return all registered probe names."""
    return list(PROBES.keys())
