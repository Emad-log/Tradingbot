import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import SklearnTradingModel


def _toy_xy(n: int, p: int = 5, pattern: str = "alternating"):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))
    if pattern == "alternating":
        y = (np.arange(n) % 2 == 0).astype(float)
    elif pattern == "block":
        y = np.zeros(n)
        switch = max(1, n // 3)
        y[:switch] = 1.0
        return X, y
    else:
        frac_pos = 0.5
        positives = int(n * frac_pos)
        y = np.zeros(n)
        y[:positives] = 1.0
        rng.shuffle(y)
    return X, y


def test_skip_calibration_on_single_class_fold():
    X, y = _toy_xy(30, p=4, pattern="block")
    model = SklearnTradingModel(input_dim=X.shape[1])
    model.fit(X, y)
    assert model.pipeline is not None
    assert model._is_fitted is True
    assert getattr(model, "used_calibration", False) is False


def test_use_calibration_when_both_classes():
    X, y = _toy_xy(120, p=6, pattern="alternating")
    model = SklearnTradingModel(input_dim=X.shape[1])
    model.fit(X, y)
    assert model.pipeline is not None
    assert model._is_fitted is True
    assert getattr(model, "used_calibration", False) is True
