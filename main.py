"""Trading bot prototype with quantitative, news-aware decision making."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from html import unescape
from xml.etree import ElementTree as ET

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans, SpectralClustering


logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Central configuration for trading bot parameters."""

    initial_cash: float = 1_000_000.0
    price_cache_max_lookback_days: int = 10
    trade_band: float = 0.05
    min_notional: float = 5_000.0
    risk_max_leverage: float = 3.0
    base_leverage: float = 1.0
    position_limit: float = 0.2
    stop_loss: float = 0.1
    borrowing_rate: float = 0.05
    fee_bps: float = 1.0
    vectorizer_dimension: int = 128
    vectorizer_seed: int = 1234
    model_type: str = "logistic"
    volatility_target: float = 0.15
    gross_leverage_target: float = 1.0
    random_seed: int = 42
    market_tz: str = "America/New_York"
    market_close_hour: int = 16
    offline_mode: bool = False
    use_calibration: bool = True
    min_cv_test_size: int = 10
    cov_lookback: int = 126
    cov_span: int = 20
    cov_shrink: float = 0.3
    risk_use_pca: bool = False
    risk_pca_components: int = 5
    adv_capacity_ratio: float = 0.1
    turnover_target: float = 0.25
    base_turn_penalty: float = 2.0
    cluster_count: int = 4
    laplacian_tau: float = 0.5
    laplacian_k: int = 5
    persist_backtest_artifacts: bool = False


@dataclass
class Position:
    """Represents an open trading position."""

    symbol: str
    quantity: float
    avg_price: float
    leverage: float

    @property
    def direction(self) -> str:
        return "long" if self.quantity >= 0 else "short"

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def notional(self, price: float) -> float:
        return abs(self.market_value(price))


@dataclass
class SentimentProfile:
    """Aggregated sentiment scores for macro and industry news."""

    macro: float = 0.0
    industry: float = 0.0
    macro_confidence: float = 0.0
    industry_confidence: float = 0.0


class NewsVectorizer:
    """Converts unstructured text into dense numeric vectors via hashing tricks."""

    def __init__(self, dimension: int = 128, seed: int = 1234) -> None:
        self.dimension = dimension
        self.seed = seed

    def _hash_idx(self, token: str) -> int:
        digest = hashlib.blake2b(f"{self.seed}|{token}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.dimension

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a deterministic embedding for a piece of text."""

        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dimension)
        vector = np.zeros(self.dimension, dtype=float)
        for token in tokens:
            vector[self._hash_idx(token)] += 1.0
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        vector /= norm
        return vector

    def embed_corpus(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=float)
        return np.vstack([self.embed_text(text) for text in texts])

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())


@dataclass
class RepresentationLearningResult:
    """Container for representation learning artefacts."""

    features: pd.DataFrame
    target: pd.Series
    predictions: pd.Series
    news_embeddings: pd.DataFrame
    training_loss: Optional[float] = None


class AlphaCombiner:
    """Learns a linear combination of daily alpha features via ridge regression."""

    def __init__(self, n_features: int, lam: float = 1.0, decay: float = 0.97) -> None:
        self.n_features = n_features
        self.lam = lam
        self.decay = decay
        self._has_history = False
        self._init_moments()

    def _init_moments(self) -> None:
        self.XtX = self.lam * np.eye(self.n_features)
        self.XtY = np.zeros(self.n_features, dtype=float)

    def ensure_dimension(self, n_features: int) -> None:
        if n_features == self.n_features:
            return
        self.n_features = n_features
        self._has_history = False
        self._init_moments()

    def update(self, features: np.ndarray, future_returns: np.ndarray) -> None:
        if features.size == 0 or future_returns.size == 0:
            return
        self.XtX = self.decay * self.XtX + features.T @ features
        self.XtY = self.decay * self.XtY + features.T @ future_returns
        self._has_history = True

    def beta(self) -> np.ndarray:
        ridge = 1e-6
        XtX_reg = self.XtX + ridge * np.eye(self.n_features)
        return np.linalg.solve(XtX_reg, self.XtY)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros(0, dtype=float)
        return features @ self.beta()

    @property
    def has_history(self) -> bool:
        return self._has_history


class NeuralTradingModel:
    """A lightweight neural network combining price and news vectors."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 200,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0:
            return
        self._ensure_initialized()
        for _ in range(self.epochs):
            z1 = X @ self.W1 + self.b1
            a1 = np.tanh(z1)
            z2 = a1 @ self.W2 + self.b2
            y_pred = np.tanh(z2)
            error = y_pred - y
            loss = float(np.mean(error ** 2))
            self.loss_history.append(loss)

            dz2 = (2.0 / len(X)) * error * (1 - np.tanh(z2) ** 2)
            dW2 = a1.T @ dz2
            db2 = dz2.sum(axis=0, keepdims=True)
            da1 = dz2 @ self.W2.T
            dz1 = da1 * (1 - a1 ** 2)
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0, keepdims=True)

            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W1 is None or self.W2 is None:
            return np.zeros((len(X), 1), dtype=float)
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        return np.tanh(z2)

    def _ensure_initialized(self) -> None:
        if self.W1 is not None and self.W2 is not None:
            return
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.1, size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = rng.normal(0, 0.1, size=(self.hidden_dim, 1))
        self.b2 = np.zeros((1, 1))


class RepresentationLearningEngine:
    """Builds feature matrices and trains machine learning models on mixed inputs."""

    def __init__(
        self,
        vectorizer: Optional[NewsVectorizer] = None,
        model_type: Optional[str] = None,
        config: Optional[TradingConfig] = None,
    ) -> None:
        self.config = config or TradingConfig()
        self.vectorizer = vectorizer or NewsVectorizer(
            dimension=self.config.vectorizer_dimension,
            seed=self.config.vectorizer_seed,
        )
        self.model_type = model_type or self.config.model_type
        self.model: Optional[object] = None

    def train_and_predict(self, prices: pd.DataFrame, news_items: Sequence[Dict[str, str]]) -> RepresentationLearningResult:
        feature_frame, target, news_embeddings = self._build_dataset(prices, news_items)
        if feature_frame.empty or target.empty:
            if feature_frame.empty:
                aligned_embeddings = news_embeddings
            else:
                aligned_embeddings = news_embeddings.reindex(feature_frame.index).fillna(0.0)
            return RepresentationLearningResult(feature_frame, target, pd.Series(dtype=float), aligned_embeddings)

        X = feature_frame.values
        y = target.values
        n_samples = len(y)

        if n_samples < 3:
            self.model = self._build_model(X.shape[1])
            self.model.fit(X, y.reshape(-1, 1))
            preds = self.model.predict(X).reshape(-1)
        else:
            n_splits = max(2, min(5, n_samples // 50))
            n_splits = min(n_splits, n_samples - 1)
            if n_splits < 2:
                n_splits = 2 if n_samples > 2 else 0
            if n_splits >= 2:
                preds = self._oos_predictions(X, y, n_splits)
            else:
                self.model = self._build_model(X.shape[1])
                self.model.fit(X, y.reshape(-1, 1))
                preds = self.model.predict(X).reshape(-1)

        prediction_series = pd.Series(np.nan_to_num(preds, nan=0.0), index=feature_frame.index)
        aligned_embeddings = news_embeddings.reindex(feature_frame.index).fillna(0.0)
        training_loss = getattr(self.model, "training_loss", None)
        used_calibration = getattr(self.model, "used_calibration", None)
        if used_calibration is not None:
            logger.info("ml_calibration_used=%s n=%d", used_calibration, n_samples)
        return RepresentationLearningResult(feature_frame, target, prediction_series, aligned_embeddings, training_loss=training_loss)

    def _build_dataset(
        self,
        prices: pd.DataFrame,
        news_items: Sequence[Dict[str, str]],
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        close = prices["close"].dropna()
        if close.empty:
            return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

        returns = close.pct_change()
        if returns.notna().sum() >= 10:
            lower = returns.quantile(0.01)
            upper = returns.quantile(0.99)
            returns = returns.clip(lower=lower, upper=upper)

        high = prices.get("high", close).reindex(close.index)
        low = prices.get("low", close).reindex(close.index)
        open_price = prices.get("open", close).reindex(close.index)

        hl_range = (high / low - 1.0).replace([np.inf, -np.inf], np.nan)
        overnight = (open_price / close.shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)

        up = returns.clip(lower=0.0)
        down = (-returns.clip(upper=0.0)).abs()
        avg_gain = up.rolling(14).mean()
        avg_loss = down.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi_14 = 100 - (100 / (1 + rs))

        atr_14 = hl_range.rolling(14).mean()
        skew_21 = returns.rolling(21).skew()

        base_features = pd.DataFrame(
            {
                "return": returns,
                "overnight": overnight,
                "momentum_5": close.pct_change(5),
                "momentum_21": close.pct_change(21),
                "volatility_10": returns.rolling(10).std(),
                "volatility_21": returns.rolling(21).std(),
                "range": hl_range,
                "rsi_14": rsi_14,
                "atr_14": atr_14,
                "skew_21": skew_21,
            }
        )

        base_features = base_features.replace([np.inf, -np.inf], np.nan).dropna()
        if base_features.empty:
            return base_features, pd.Series(dtype=float), pd.DataFrame()

        news_embeddings = self._news_embeddings_for_dates(base_features.index, news_items)
        news_embeddings = news_embeddings.sort_index().reindex(base_features.index).fillna(0.0)
        combined = pd.concat([base_features, news_embeddings], axis=1)

        future_returns = close.pct_change().shift(-1)
        aligned_future = future_returns.reindex(combined.index)
        assert aligned_future.index.max() <= close.index.max(), "Target alignment leak"
        valid_mask = aligned_future.notna()
        if not valid_mask.any():
            return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()
        combined = combined.loc[valid_mask]
        aligned_future = aligned_future.loc[valid_mask]
        news_embeddings = news_embeddings.reindex(combined.index).fillna(0.0)
        target = pd.Series(np.where(aligned_future > 0, 1.0, -1.0), index=combined.index)
        return combined, target, news_embeddings

    def _oos_predictions(self, X: np.ndarray, y: np.ndarray, n_splits: int) -> np.ndarray:
        splits = list(self._purged_splits(len(y), n_splits=n_splits, embargo=5))
        if not splits:
            self.model = self._build_model(X.shape[1])
            self.model.fit(X, y.reshape(-1, 1))
            return self.model.predict(X).reshape(-1)
        preds = np.full(len(y), np.nan, dtype=float)
        for train_idx, test_idx in splits:
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            model = self._build_model(X.shape[1])
            model.fit(X[train_idx], y[train_idx].reshape(-1, 1))
            fold_pred = model.predict(X[test_idx]).reshape(-1)
            preds[test_idx] = fold_pred

        self.model = self._build_model(X.shape[1])
        self.model.fit(X, y.reshape(-1, 1))
        return preds

    @staticmethod
    def _purged_splits(length: int, n_splits: int, embargo: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if length <= 1:
            return []
        n_splits = max(1, min(n_splits, length - 1))
        fold = max(1, length // n_splits)
        indices = np.arange(length)
        for i in range(n_splits):
            test_lo = i * fold
            test_hi = (i + 1) * fold if i < n_splits - 1 else length
            test_idx = indices[test_lo:test_hi]
            embargo_lo = max(0, test_lo - embargo)
            embargo_hi = min(length, test_hi + embargo)
            train_mask = np.ones(length, dtype=bool)
            train_mask[embargo_lo:embargo_hi] = False
            train_idx = indices[train_mask]
            yield train_idx, test_idx

    def _news_embeddings_for_dates(
        self,
        dates: Iterable[pd.Timestamp],
        news_items: Sequence[Dict[str, str]],
    ) -> pd.DataFrame:
        normalized_dates = [pd.Timestamp(date).normalize() for date in dates]
        texts = self._generate_daily_texts(normalized_dates, news_items)
        embeddings = self.vectorizer.embed_corpus([texts[d] for d in normalized_dates])
        columns = [f"news_emb_{i}" for i in range(embeddings.shape[1])]
        return pd.DataFrame(embeddings, index=pd.Index(normalized_dates, name="date"), columns=columns)

    def _generate_daily_texts(
        self,
        dates: Iterable[pd.Timestamp],
        news_items: Sequence[Dict[str, str]],
    ) -> Dict[pd.Timestamp, str]:
        date_list = [pd.Timestamp(date).normalize() for date in dates]
        daily_texts: Dict[pd.Timestamp, str] = {}
        if not news_items:
            return {d: "" for d in date_list}

        df = pd.DataFrame(news_items)
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
            market_tz = getattr(self.config, "market_tz", "America/New_York")
            close_hour = getattr(self.config, "market_close_hour", 16)
            assigned_dates: List[Optional[pd.Timestamp]] = []
            next_bday = pd.tseries.offsets.BDay(1)
            for ts in df["published_at"]:
                if pd.isna(ts):
                    assigned_dates.append(None)
                    continue
                local_ts = ts.tz_convert(market_tz)
                naive = local_ts.tz_localize(None)
                trade_date = naive.normalize()
                close_dt = trade_date + pd.Timedelta(hours=close_hour)
                if naive > close_dt:
                    trade_date = (trade_date + next_bday).normalize()
                assigned_dates.append(pd.Timestamp(trade_date))
            df["d"] = assigned_dates
        else:
            df["d"] = max(date_list) if date_list else pd.Timestamp.utcnow().normalize()
        if df["d"].isna().any():
            fallback = max(date_list) if date_list else pd.Timestamp.utcnow().normalize()
            df.loc[df["d"].isna(), "d"] = fallback

        df["age_rank"] = df.groupby("d")["published_at"].rank(ascending=False, method="first")
        df["w"] = 1.0 / (1.0 + df["age_rank"].fillna(0.0))
        grouped = df.groupby("d")

        for d in date_list:
            if d in grouped.groups:
                block = grouped.get_group(d).sort_values("w", ascending=False)
                texts = []
                for _, row in block.iterrows():
                    headline = row.get("headline") or ""
                    body = row.get("body") or ""
                    weight = float(row.get("w", 0.0))
                    repeat = int(max(1, round(3 * weight)))
                    if headline:
                        texts.append((headline + " ") * repeat)
                    if body:
                        texts.append(body)
                daily_texts[d] = " ".join(texts)
            else:
                daily_texts[d] = ""
        return daily_texts

    def _build_model(self, input_dim: int):
        if self.model_type == "neural":
            return NeuralTradingModel(input_dim=input_dim)
        return SklearnTradingModel(
            input_dim=input_dim,
            random_state=self.config.random_seed,
            use_calibration=self.config.use_calibration,
            min_cv_test_size=self.config.min_cv_test_size,
        )


class SklearnTradingModel:
    """Logistic regression model with feature scaling for trading signals."""

    def __init__(
        self,
        input_dim: int,
        penalty: str = "l2",
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 200,
        random_state: int = 0,
        use_calibration: bool = True,
        min_cv_test_size: int = 10,
    ) -> None:
        self.input_dim = input_dim
        self.penalty = penalty
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.training_loss: Optional[float] = None
        self._is_fitted = False
        self.pipeline: Optional[Pipeline] = None
        self.random_state = random_state
        self.use_calibration = use_calibration
        self.min_cv_test_size = min_cv_test_size
        self.used_calibration: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0:
            return
        y_vector = y.reshape(-1)
        y_binary = (y_vector > 0).astype(int)
        if len(np.unique(y_binary)) < 2:
            self._is_fitted = False
            self.training_loss = None
            return

        n_samples = len(y_binary)
        self.used_calibration = False
        if n_samples >= 6 and len(np.unique(y_binary)) > 1:
            cv = min(3, max(2, n_samples // 3))
            base = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                solver="lbfgs",
                random_state=self.random_state,
            )
            method = "isotonic" if n_samples >= 200 else "sigmoid"
            estimator = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", base),
            ])

            def _cv_has_both_classes() -> bool:
                splitter = TimeSeriesSplit(n_splits=cv)
                for train_idx, test_idx in splitter.split(X, y_binary):
                    if test_idx.size < self.min_cv_test_size:
                        logger.info(
                            "calibration_skipped: test fold too small (size=%d, min=%d)",
                            test_idx.size,
                            self.min_cv_test_size,
                        )
                        return False
                    if len(np.unique(y_binary[train_idx])) < 2 or len(np.unique(y_binary[test_idx])) < 2:
                        logger.info(
                            "calibration_skipped: single-class fold encountered (n=%d, cv=%d)",
                            n_samples,
                            cv,
                        )
                        return False
                return True

            if self.use_calibration and _cv_has_both_classes():
                pipeline = CalibratedClassifierCV(
                    estimator=estimator,
                    method=method,
                    cv=TimeSeriesSplit(n_splits=cv),
                )
                self.used_calibration = True
            else:
                if not self.use_calibration:
                    logger.info("calibration_skipped: disabled via configuration")
                pipeline = estimator
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty=self.penalty,
                        C=self.C,
                        class_weight=self.class_weight,
                        max_iter=self.max_iter,
                        solver="lbfgs",
                        random_state=self.random_state,
                    ),
                ),
            ])
        if not self.use_calibration:
            self.used_calibration = False
        self.pipeline = pipeline

        test_size = 0.2 if len(y_binary) > 10 else 0.0
        if test_size > 0.0:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y_binary, test_size=test_size, shuffle=False
            )
        else:
            X_train, X_valid, y_train, y_valid = X, np.empty((0, X.shape[1])), y_binary, np.empty(0)

        self.pipeline.fit(X_train, y_train)
        self._is_fitted = True

        if y_valid.size > 0 and len(np.unique(y_valid)) > 1:
            valid_proba = self.pipeline.predict_proba(X_valid)[:, 1]
            try:
                self.training_loss = float(log_loss(y_valid, valid_proba, labels=[0, 1]))
            except ValueError:
                self.training_loss = None
        else:
            train_proba = self.pipeline.predict_proba(X_train)[:, 1]
            try:
                self.training_loss = float(log_loss(y_train, train_proba, labels=[0, 1]))
            except ValueError:
                self.training_loss = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or X.size == 0 or self.pipeline is None:
            return np.zeros((len(X), 1), dtype=float)
        probabilities = self.pipeline.predict_proba(X)[:, 1]
        scaled_signal = probabilities * 2.0 - 1.0
        return scaled_signal.reshape(-1, 1)
@dataclass
class MacroSnapshot:
    """Collection of macroeconomic indicators for a point in time."""

    gdp_growth: float
    inflation: float
    unemployment: float
    yield_curve_slope: float
    manufacturing_pmi: float
    credit_spread: float
    usd_trend: float
    commodity_trend: float
    financial_conditions_index: float
    housing_starts_growth: float
    consumer_sentiment_index: float
    global_growth_diff: float


@dataclass
class MacroEnvironment:
    """Normalized description of the prevailing macroeconomic regime."""

    growth_score: float
    inflation_score: float
    policy_tightening_probability: float
    risk_appetite: float
    stagflation_risk: float
    commodity_pressure: float
    liquidity_score: float
    housing_cycle_strength: float
    consumer_mood: float
    descriptors: List[str] = field(default_factory=list)


class DataFetcher:
    """Fetches price, news, and macro data from freely accessible sources."""

    STOOQ_URL = "https://stooq.pl/q/d/l/"
    YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    WSJ_MARKETS_FEED = "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
    WORLD_BANK_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    FRED_SERIES_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(
        self,
        price_source: Optional[str] = "stooq",
        news_source: Optional[str] = "rss",
        random_seed: int = 42,
        offline: bool = False,
    ):
        self.price_source = price_source
        self.news_source = news_source
        self.random_seed = random_seed
        self.offline = offline
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0; +https://example.com/contact)",
                "Accept": "application/json, text/xml, application/xml, text/html, */*",
            }
        )
        self.market_tz = "America/New_York"
        self.market_close_hour = 16
        self._price_cache: Dict[Tuple[str, dt.date, dt.date], pd.DataFrame] = {}
        self._raw_stooq_cache: Dict[str, pd.DataFrame] = {}
        self._fred_cache: Dict[str, pd.DataFrame] = {}

    def fetch_price_data(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        if start >= end:
            raise ValueError("start must be before end")

        dates = pd.date_range(start=start, end=end, freq="B")
        if len(dates) == 0:
            raise ValueError("No business days in selected range")

        cache_key = (symbol, start, end)
        if cache_key in self._price_cache:
            return self._price_cache[cache_key].copy()

        if self.offline:
            frame = self._generate_synthetic_prices(symbol, dates)
            self._price_cache[cache_key] = frame.copy()
            return frame

        try:
            if self.price_source == "stooq":
                frame = self._fetch_price_data_from_stooq(symbol, dates)
            else:
                frame = self._fetch_price_data_from_source(symbol, dates)
        except Exception as exc:  # pragma: no cover - network branch
            logger.warning("Falling back to synthetic prices for %s due to %s", symbol, exc)
            frame = self._generate_synthetic_prices(symbol, dates)

        self._price_cache[cache_key] = frame.copy()
        return frame

    def _fetch_price_data_from_stooq(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        suffix = "" if "." in symbol else ".us"
        stooq_symbol = f"{symbol.lower()}{suffix}"
        if stooq_symbol in self._raw_stooq_cache:
            full_history = self._raw_stooq_cache[stooq_symbol]
        else:
            response = self.session.get(self.STOOQ_URL, params={"s": stooq_symbol, "i": "d"}, timeout=10)
            response.raise_for_status()
            frame = pd.read_csv(io.StringIO(response.text))
            if frame.empty or "Data" not in frame:
                raise ValueError("Empty Stooq response")
            frame = frame.rename(
                columns={
                    "Data": "date",
                    "Otwarcie": "open",
                    "Najwyzszy": "high",
                    "Najnizszy": "low",
                    "Zamkniecie": "close",
                    "Wolumen": "volume",
                }
            )
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"])
            numeric_cols = ["open", "high", "low", "close", "volume"]
            frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
            frame = frame.dropna(subset=["close"]).set_index("date").sort_index()
            frame["volume"] = frame["volume"].astype(float)
            frame["adv_20"] = frame["volume"].rolling(20, min_periods=1).mean()
            self._raw_stooq_cache[stooq_symbol] = frame.copy()
            full_history = self._raw_stooq_cache[stooq_symbol]

        frame = full_history.reindex(dates).copy()
        price_cols = ["open", "high", "low", "close"]
        max_gap = 5
        for col in price_cols:
            frame[col] = frame[col].interpolate(limit=max_gap).ffill(limit=max_gap).bfill(limit=max_gap)
        frame["volume"] = frame["volume"].ffill(limit=max_gap).bfill(limit=max_gap).fillna(0.0)
        frame = frame.dropna(subset=["close"])
        frame["volume"] = frame["volume"].astype(float)
        if "adv_20" not in frame.columns:
            frame["adv_20"] = frame["volume"].rolling(20, min_periods=1).mean()
        return frame[price_cols + ["volume", "adv_20"]]

    def _fetch_price_data_from_source(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        if not self.price_source:
            raise RuntimeError("No price source configured")
        response = self.session.get(self.price_source, params={"symbol": symbol}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError("Empty price data")
        frame = pd.DataFrame(data)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").reindex(dates).interpolate().ffill().bfill()
        frame["adv_20"] = frame["volume"].rolling(20, min_periods=1).mean()
        return frame[["open", "high", "low", "close", "volume", "adv_20"]]

    def _generate_synthetic_prices(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        seed = (abs(hash(symbol)) + self.random_seed) % (2 ** 32)
        rng = np.random.default_rng(seed)
        steps = rng.normal(loc=0.0005, scale=0.02, size=len(dates))
        prices = 100 * np.exp(np.cumsum(steps))
        highs = prices * (1 + rng.uniform(0.0, 0.02, size=len(dates)))
        lows = prices * (1 - rng.uniform(0.0, 0.02, size=len(dates)))
        volumes = rng.integers(1_000_000, 5_000_000, size=len(dates))
        frame = pd.DataFrame(
            {
                "open": prices * (1 + rng.normal(0, 0.005, size=len(dates))),
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )
        frame["adv_20"] = frame["volume"].rolling(20, min_periods=1).mean()
        return frame

    def fetch_news(self, symbol: str, as_of: Optional[pd.Timestamp] = None) -> List[Dict[str, str]]:
        if self.offline:
            return self._generate_synthetic_news(symbol, as_of=as_of)
        try:
            macro_news = self._fetch_macro_news()
            industry_news = self._fetch_symbol_news(symbol)
            news_items = macro_news + industry_news
            if as_of is not None:
                tz = getattr(self, "market_tz", "America/New_York")
                close_hour = getattr(self, "market_close_hour", 16)
                as_of_ts = pd.Timestamp(as_of)
                if as_of_ts.tzinfo is None:
                    local_cutoff = as_of_ts.tz_localize(tz)
                else:
                    local_cutoff = as_of_ts.tz_convert(tz)
                local_cutoff = local_cutoff.normalize() + pd.Timedelta(hours=close_hour)
                cutoff_utc = local_cutoff.tz_convert("UTC")

                def _within_cutoff(item: Dict[str, str]) -> bool:
                    ts = item.get("published_at")
                    return ts is not None and pd.notna(ts) and ts <= cutoff_utc

                news_items = [item for item in news_items if _within_cutoff(item)]
            if not news_items:
                raise ValueError("No news items downloaded")
            return news_items
        except Exception as exc:  # pragma: no cover - network branch
            logger.warning("Falling back to synthetic news for %s due to %s", symbol, exc)
            return self._generate_synthetic_news(symbol, as_of=as_of)

    def _fetch_symbol_news(self, symbol: str, limit: int = 12) -> List[Dict[str, str]]:
        if self.news_source != "rss":
            return []
        params = {"s": symbol.upper(), "region": "US", "lang": "en-US"}
        response = self.session.get(self.YAHOO_RSS_URL, params=params, timeout=10)
        response.raise_for_status()
        return self._parse_rss_items(response.text, category="industry", limit=limit)

    def _fetch_macro_news(self, limit: int = 12) -> List[Dict[str, str]]:
        response = self.session.get(self.WSJ_MARKETS_FEED, timeout=10)
        response.raise_for_status()
        return self._parse_rss_items(response.text, category="macro", limit=limit)

    def _parse_rss_items(self, xml_text: str, category: str, limit: int) -> List[Dict[str, str]]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:  # pragma: no cover - feed parsing branch
            raise ValueError(f"Failed to parse RSS feed: {exc}")
        items: List[Dict[str, str]] = []
        seen = set()
        for item in root.findall(".//item"):
            title = unescape((item.findtext("title") or "").strip())
            description = item.findtext("description") or ""
            link = (item.findtext("link") or "").strip()
            pubdate = item.findtext("pubDate") or ""
            try:
                published_at = pd.to_datetime(pubdate, utc=True)
            except Exception:
                published_at = None

            key = (title, link)
            if key in seen:
                continue
            seen.add(key)

            items.append(
                {
                    "category": category,
                    "headline": title,
                    "body": self._strip_html(description),
                    "link": link,
                    "published_at": published_at,
                }
            )
            if len(items) >= limit:
                break
        return items

    @staticmethod
    def _strip_html(raw_text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", raw_text)
        text = re.sub(r"\s+", " ", text)
        return unescape(text.strip())

    def _generate_synthetic_news(
        self, symbol: str, as_of: Optional[pd.Timestamp] = None
    ) -> List[Dict[str, str]]:
        seed = (abs(hash(symbol)) + self.random_seed) % (2 ** 32)
        rng = np.random.default_rng(seed)
        now = pd.Timestamp.now(tz="UTC")
        if as_of is not None:
            tz = getattr(self, "market_tz", "America/New_York")
            close_hour = getattr(self, "market_close_hour", 16)
            as_of_ts = pd.Timestamp(as_of)
            if as_of_ts.tzinfo is None:
                local_cutoff = as_of_ts.tz_localize(tz)
            else:
                local_cutoff = as_of_ts.tz_convert(tz)
            local_cutoff = local_cutoff.normalize() + pd.Timedelta(hours=close_hour)
            cutoff = local_cutoff.tz_convert("UTC")
            if now > cutoff:
                now = cutoff
        macro_headlines = [
            {
                "category": "macro",
                "headline": "Central bank signals cautious approach to interest rates",
                "body": "Policy makers hint at balancing inflation concerns with growth support.",
                "link": "",
                "published_at": now,
            },
            {
                "category": "macro",
                "headline": "Inflation eases as supply chains normalize",
                "body": "Commodity prices decline while consumer demand remains resilient.",
                "link": "",
                "published_at": now - pd.Timedelta(hours=6),
            },
            {
                "category": "macro",
                "headline": "Stronger dollar pressures emerging markets",
                "body": "Investors reassess risk appetite amid currency volatility.",
                "link": "",
                "published_at": now - pd.Timedelta(hours=12),
            },
        ]
        industry_headlines = [
            {
                "category": "industry",
                "headline": f"{symbol} suppliers report improving order flow",
                "body": "Upstream partners note backlog stabilization and better inventory management.",
                "link": "",
                "published_at": now - pd.Timedelta(hours=float(rng.uniform(1, 18))),
            },
            {
                "category": "industry",
                "headline": f"Regulatory scrutiny rises across the {symbol} industry",
                "body": "Analysts warn of higher compliance costs and slower product approvals.",
                "link": "",
                "published_at": now - pd.Timedelta(hours=float(rng.uniform(2, 24))),
            },
            {
                "category": "industry",
                "headline": f"Technological breakthroughs reshape {symbol} competitive landscape",
                "body": "Startups challenge incumbents with faster deployment cycles and AI integration.",
                "link": "",
                "published_at": now - pd.Timedelta(hours=float(rng.uniform(3, 30))),
            },
        ]
        news_items: List[Dict[str, str]] = []
        for pool in (macro_headlines, industry_headlines):
            count = rng.integers(1, len(pool) + 1)
            news_items.extend(rng.choice(pool, size=count, replace=False).tolist())
        return news_items

    def fetch_macro_snapshot(self, as_of: dt.date) -> MacroSnapshot:
        if self.offline:
            return self._generate_synthetic_macro(as_of)
        try:
            return self._fetch_macro_snapshot_from_sources_asof(as_of)
        except Exception as exc:  # pragma: no cover - network branch
            logger.warning("Falling back to synthetic macro snapshot due to %s", exc)
            return self._generate_synthetic_macro(as_of)

    def _fetch_macro_snapshot_from_sources_asof(self, as_of: dt.date) -> MacroSnapshot:
        gdp_growth = self._world_bank_asof("USA", "NY.GDP.MKTP.KD.ZG", as_of) / 100.0
        world_growth = self._world_bank_asof("WLD", "NY.GDP.MKTP.KD.ZG", as_of) / 100.0
        inflation = self._world_bank_asof("USA", "FP.CPI.TOTL.ZG", as_of) / 100.0
        unemployment = self._world_bank_asof("USA", "SL.UEM.TOTL.NE.ZS", as_of) / 100.0

        yield_curve_spread = self._fred_latest_asof("T10Y2Y", as_of) / 100.0
        ip_yoy = self._compute_yoy_growth_asof(self._fetch_fred_series("IPMAN"), as_of)
        manufacturing_pmi = float(np.clip(50 + ip_yoy * 100, 30, 70))

        credit_spread = self._fred_latest_asof("BAMLH0A0HYM2EY", as_of) / 100.0
        usd_trend = self._compute_yoy_growth_asof(self._fetch_fred_series("DTWEXBGS"), as_of)

        commodity_yoy = self._compute_yoy_growth_asof(self._fetch_fred_series("PALLFNFINDEXM"), as_of)
        financial_conditions_index = self._fred_latest_asof("NFCI", as_of)

        housing_yoy = self._compute_yoy_growth_asof(self._fetch_fred_series("HOUST"), as_of)
        consumer_sentiment = self._fred_latest_asof("UMCSENT", as_of)

        return MacroSnapshot(
            gdp_growth=float(gdp_growth),
            inflation=float(inflation),
            unemployment=float(max(unemployment, 0.02)),
            yield_curve_slope=float(yield_curve_spread),
            manufacturing_pmi=float(manufacturing_pmi),
            credit_spread=float(max(credit_spread, 0.005)),
            usd_trend=float(usd_trend),
            commodity_trend=float(commodity_yoy),
            financial_conditions_index=float(financial_conditions_index),
            housing_starts_growth=float(housing_yoy),
            consumer_sentiment_index=float(consumer_sentiment),
            global_growth_diff=float(gdp_growth - world_growth),
        )

    def _world_bank_asof(self, country: str, indicator: str, as_of: dt.date) -> float:
        url = self.WORLD_BANK_URL.format(country=country, indicator=indicator)
        response = self.session.get(url, params={"format": "json", "per_page": 60}, timeout=10)
        response.raise_for_status()
        payload = response.json()
        if len(payload) < 2:
            raise ValueError("Unexpected World Bank response")
        series = payload[1]
        cutoff_year = as_of.year

        def _entry_year(entry: Dict[str, object]) -> int:
            try:
                return int(entry.get("date"))  # type: ignore[arg-type]
            except Exception:
                return -10**9

        for entry in sorted(series, key=_entry_year, reverse=True):
            year = _entry_year(entry)
            if year == -10**9 or year > cutoff_year:
                continue
            value = entry.get("value")
            if value is not None:
                return float(value)
        raise ValueError("World Bank response missing past values")

    def _fred_latest_asof(self, series_id: str, as_of: dt.date) -> float:
        series = self._fetch_fred_series(series_id)
        series = series[series["date"] <= pd.Timestamp(as_of)]
        if series.empty:
            raise ValueError(f"No data for {series_id} as of {as_of}")
        return float(series.iloc[-1]["value"])

    def _compute_yoy_growth_asof(self, series: pd.DataFrame, as_of: dt.date) -> float:
        series = series[series["date"] <= pd.Timestamp(as_of)]
        if len(series) < 13:
            return 0.0
        latest = float(series.iloc[-1]["value"])
        comparison = float(series.iloc[-13]["value"])
        if not np.isfinite(latest) or not np.isfinite(comparison) or comparison == 0:
            return 0.0
        return (latest - comparison) / comparison

    def _fetch_fred_series(self, series_id: str) -> pd.DataFrame:
        if series_id in self._fred_cache:
            return self._fred_cache[series_id].copy()
        response = self.session.get(self.FRED_SERIES_URL, params={"id": series_id}, timeout=10)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        if frame.shape[1] < 2:
            raise ValueError(f"Unexpected FRED data for {series_id}")
        value_col = frame.columns[1]
        frame = frame.rename(columns={"observation_date": "date", value_col: "value"})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["date", "value"]).sort_values("date")
        if frame.empty:
            raise ValueError(f"No data returned for FRED series {series_id}")
        self._fred_cache[series_id] = frame.copy()
        return frame

    def _generate_synthetic_macro(self, as_of: dt.date) -> MacroSnapshot:
        seed = int(as_of.strftime("%Y%m%d"))
        rng = np.random.default_rng(seed)
        base_growth = 0.02 + 0.01 * math.sin(as_of.timetuple().tm_yday / 58.0)
        gdp_growth = base_growth + rng.normal(0, 0.005)
        inflation = 0.025 + 0.015 * math.sin(as_of.timetuple().tm_yday / 73.0 + 1.5) + rng.normal(0, 0.003)
        unemployment = 0.045 + 0.01 * math.cos(as_of.timetuple().tm_yday / 91.0) + rng.normal(0, 0.002)
        yield_curve_slope = 0.01 + 0.015 * math.sin(as_of.timetuple().tm_yday / 45.0) + rng.normal(0, 0.002)
        manufacturing_pmi = 52 + 3 * math.sin(as_of.timetuple().tm_yday / 37.0) + rng.normal(0, 0.5)
        credit_spread = 0.015 + 0.01 * math.cos(as_of.timetuple().tm_yday / 67.0) + rng.normal(0, 0.002)
        usd_trend = rng.normal(0, 0.5)
        commodity_trend = rng.normal(0.0, 0.08) + 0.05 * math.sin(as_of.timetuple().tm_yday / 27.0)
        financial_conditions_index = (
            -0.5
            + 0.4 * math.sin(as_of.timetuple().tm_yday / 31.0)
            + rng.normal(0.0, 0.15)
        )
        housing_starts_growth = 0.03 + 0.01 * math.cos(as_of.timetuple().tm_yday / 83.0) + rng.normal(0, 0.003)
        consumer_sentiment_index = 92 + 5 * math.sin(as_of.timetuple().tm_yday / 41.0) + rng.normal(0, 1.5)
        global_growth_diff = 0.01 + 0.01 * math.sin(as_of.timetuple().tm_yday / 59.0 + 0.5) + rng.normal(0, 0.003)
        return MacroSnapshot(
            gdp_growth=float(gdp_growth),
            inflation=float(inflation),
            unemployment=float(max(unemployment, 0.02)),
            yield_curve_slope=float(yield_curve_slope),
            manufacturing_pmi=float(manufacturing_pmi),
            credit_spread=float(max(credit_spread, 0.005)),
            usd_trend=float(usd_trend),
            commodity_trend=float(commodity_trend),
            financial_conditions_index=float(financial_conditions_index),
            housing_starts_growth=float(housing_starts_growth),
            consumer_sentiment_index=float(consumer_sentiment_index),
            global_growth_diff=float(global_growth_diff),
        )


class NewsAnalyzer:
    """Transforms qualitative news into simple sentiment signals."""

    POSITIVE_WORDS: Iterable[str] = (
        "beat",
        "beats",
        "growth",
        "improving",
        "optimistic",
        "outperform",
        "resilient",
        "strong",
        "support",
        "upgrade",
        "record",
        "surge",
        "tailwind",
        "robust",
        "recover",
        "expansion",
    )
    NEGATIVE_WORDS: Iterable[str] = (
        "concerns",
        "decline",
        "downturn",
        "downgrade",
        "miss",
        "pressure",
        "risk",
        "scrutiny",
        "volatile",
        "probe",
        "settlement",
        "recall",
        "antitrust",
        "shortfall",
        "weak",
        "slowdown",
        "constraint",
        "shortage",
        "penalty",
    )
    POSITIVE_PHRASES: Iterable[str] = (
        "guide up",
        "raises guidance",
        "beats estimates",
        "price target raised",
        "revenue beat",
        "strong demand",
    )
    NEGATIVE_PHRASES: Iterable[str] = (
        "guide down",
        "guidance cut",
        "miss estimates",
        "regulatory probe",
        "antitrust probe",
        "product recall",
        "supply constraint",
        "investigation",
        "settlement charge",
    )

    def __init__(self) -> None:
        self.positive_words = {w.lower() for w in self.POSITIVE_WORDS}
        self.negative_words = {w.lower() for w in self.NEGATIVE_WORDS}
        self.negation_words = {"not", "no", "never", "less"}
        self.positive_phrases = {p.lower() for p in self.POSITIVE_PHRASES}
        self.negative_phrases = {p.lower() for p in self.NEGATIVE_PHRASES}

    def analyze(self, news_items: Iterable[Dict[str, str]]) -> SentimentProfile:
        macro_scores: List[float] = []
        industry_scores: List[float] = []
        for item in news_items:
            score = self._score_text(f"{item.get('headline', '')} {item.get('body', '')}")
            if item.get("category") == "macro":
                macro_scores.append(score)
            elif item.get("category") == "industry":
                industry_scores.append(score)
        macro = float(np.mean(macro_scores)) if macro_scores else 0.0
        industry = float(np.mean(industry_scores)) if industry_scores else 0.0
        macro_conf = min(len(macro_scores) / 5.0, 1.0)
        industry_conf = min(len(industry_scores) / 5.0, 1.0)
        return SentimentProfile(macro=macro, industry=industry, macro_confidence=macro_conf, industry_confidence=industry_conf)

    def _score_text(self, text: str) -> float:
        lowered = text.lower()
        tokens = re.findall(r"[A-Za-z']+", lowered)
        score = 0.0
        hits = 0
        negate_next = False
        for token in tokens:
            if token in self.negation_words:
                negate_next = True
                continue

            token_score = 0.0
            if token in self.positive_words:
                token_score = 1.0
            elif token in self.negative_words:
                token_score = -1.0

            if token_score != 0.0:
                if negate_next:
                    token_score *= -1.0
                score += token_score
                hits += 1
                negate_next = False
            else:
                negate_next = False

        for phrase in self.positive_phrases:
            if phrase in lowered:
                score += 1.0
                hits += 1
        for phrase in self.negative_phrases:
            if phrase in lowered:
                score -= 1.0
                hits += 1

        if hits == 0:
            return 0.0
        return score / hits


class MacroAnalyzer:
    """Normalizes macro data into interpretable regime metrics."""

    def evaluate(self, snapshot: MacroSnapshot) -> MacroEnvironment:
        growth_signal = (
            0.6 * (snapshot.gdp_growth - 0.02) * 100
            + 0.25 * (snapshot.manufacturing_pmi - 50)
            - 0.5 * (snapshot.unemployment - 0.04) * 100
        )
        growth_score = float(np.tanh(growth_signal / 20))

        inflation_signal = (
            0.7 * (snapshot.inflation - 0.02) * 100
            + 0.25 * max(-snapshot.yield_curve_slope, 0.0) * 200
        )
        inflation_score = float(np.tanh(inflation_signal / 20))

        policy_tightening_probability = float(
            np.clip(
                0.5
                + 0.35 * inflation_score
                - 0.15 * growth_score
                + 0.05 * np.tanh(snapshot.usd_trend / 2.0),
                0.0,
                1.0,
            )
        )

        credit_relief = np.tanh((0.02 - snapshot.credit_spread) * 120)
        risk_signal = 0.45 * growth_score - 0.35 * inflation_score + 0.35 * credit_relief - 0.2 * np.tanh(
            snapshot.usd_trend / 2.0
        )
        risk_appetite = float(np.tanh(risk_signal))

        stagflation_raw = (
            0.4 * (inflation_score - growth_score)
            + 0.3 * np.tanh((snapshot.credit_spread - 0.02) * 120)
        )
        stagflation_risk = float(np.clip(0.5 + 0.5 * stagflation_raw, 0.0, 1.0))

        commodity_pressure = float(
            np.tanh(0.6 * snapshot.commodity_trend + 0.4 * (snapshot.inflation - 0.025) * 80)
        )
        liquidity_score = float(
            np.tanh(
                -0.5 * snapshot.credit_spread * 120
                + 0.4 * snapshot.yield_curve_slope * 100
                - 0.3 * snapshot.financial_conditions_index
            )
        )
        housing_cycle_strength = float(
            np.tanh(0.7 * snapshot.housing_starts_growth * 50 - 0.4 * (snapshot.unemployment - 0.04) * 120)
        )
        consumer_mood = float(
            np.tanh(
                0.05 * (snapshot.consumer_sentiment_index - 90)
                - 0.4 * (snapshot.unemployment - 0.045) * 100
            )
        )
        global_growth_bias = np.tanh(snapshot.global_growth_diff * 120)

        descriptors: List[str] = []
        if growth_score > 0.35:
            descriptors.append("expansionary growth")
        elif growth_score < -0.35:
            descriptors.append("growth slowdown")
        if inflation_score > 0.35:
            descriptors.append("elevated inflation pressure")
        elif inflation_score < -0.35:
            descriptors.append("disinflationary trend")
        if risk_appetite > 0.35:
            descriptors.append("risk-on sentiment")
        elif risk_appetite < -0.35:
            descriptors.append("risk-off sentiment")
        if stagflation_risk > 0.6:
            descriptors.append("stagflation watch")
        if commodity_pressure > 0.45:
            descriptors.append("commodity squeeze")
        elif commodity_pressure < -0.45:
            descriptors.append("commodity relief")
        if liquidity_score < -0.4:
            descriptors.append("tight liquidity")
        elif liquidity_score > 0.4:
            descriptors.append("ample liquidity")
        if housing_cycle_strength > 0.35:
            descriptors.append("housing upswing")
        elif housing_cycle_strength < -0.35:
            descriptors.append("housing slowdown")
        if consumer_mood < -0.4:
            descriptors.append("weak consumer sentiment")
        elif consumer_mood > 0.4:
            descriptors.append("confident consumer")
        if global_growth_bias < -0.4:
            descriptors.append("global growth drag")
        elif global_growth_bias > 0.4:
            descriptors.append("global growth tailwind")

        return MacroEnvironment(
            growth_score=growth_score,
            inflation_score=inflation_score,
            policy_tightening_probability=policy_tightening_probability,
            risk_appetite=risk_appetite,
            stagflation_risk=stagflation_risk,
            commodity_pressure=commodity_pressure,
            liquidity_score=liquidity_score,
            housing_cycle_strength=housing_cycle_strength,
            consumer_mood=consumer_mood,
            descriptors=descriptors,
        )


class QuantStrategy:
    """Generates long/short signals using price action, macro, and news sentiment."""

    def __init__(
        self,
        momentum_window: int = 21,
        volatility_window: int = 63,
        value_window: int = 252,
        volatility_target: float = 0.15,
        max_leverage: float = 3.0,
        base_leverage: float = 1.0,
        trade_band: float = 0.05,
        min_notional: float = 5_000.0,
    ):
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.value_window = value_window
        self.base_leverage = base_leverage
        self.max_leverage = max_leverage
        self.volatility_target = volatility_target
        self.trade_band = trade_band
        self.min_notional = min_notional

    def generate_signals(
        self,
        prices: pd.DataFrame,
        sentiment: SentimentProfile,
        macro_environment: MacroEnvironment,
        learned_signal: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        close = prices["close"]
        returns = close.pct_change().fillna(0.0)
        momentum = close.pct_change(self.momentum_window).fillna(0.0)
        volatility = returns.rolling(self.volatility_window).std().bfill().fillna(0.0)
        value_ratio = close / close.rolling(self.value_window).mean()
        value_signal = (value_ratio - 1.0).fillna(0.0)

        macro_bias = sentiment.macro * (0.5 + 0.5 * sentiment.macro_confidence)
        industry_bias = sentiment.industry * (0.5 + 0.5 * sentiment.industry_confidence)
        macro_cycle_bias_value = (
            0.4 * macro_environment.growth_score
            - 0.35 * macro_environment.inflation_score
            + 0.25 * macro_environment.risk_appetite
            - 0.2 * macro_environment.stagflation_risk
            - 0.15 * macro_environment.commodity_pressure
            + 0.2 * macro_environment.liquidity_score
            + 0.15 * macro_environment.housing_cycle_strength
            + 0.2 * macro_environment.consumer_mood
        )
        macro_cycle_bias = pd.Series(macro_cycle_bias_value, index=prices.index)
        ml_signal = pd.Series(0.0, index=prices.index)
        if learned_signal is not None and not learned_signal.empty:
            ml_signal = learned_signal.reindex(prices.index).ffill().fillna(0.0)
            ml_signal = ml_signal.ewm(span=3, adjust=False).mean()

        raw_signal = (
            0.6 * momentum
            - 0.3 * volatility
            + 0.2 * value_signal
            + 0.4 * macro_bias
            + 0.3 * industry_bias
            + 0.35 * macro_cycle_bias
            + 0.5 * ml_signal
        )
        raw_weight = pd.Series(np.tanh(raw_signal.values), index=raw_signal.index)
        annual_vol = (volatility * math.sqrt(252)).clip(lower=1e-6)
        vol_scaler = (self.volatility_target / annual_vol).clip(lower=0.25, upper=3.0)
        target_weight = (raw_weight * vol_scaler).clip(lower=-1.5, upper=1.5)
        news_confidence = max(sentiment.macro_confidence, sentiment.industry_confidence)
        macro_risk_adjustment = (
            0.45 * macro_environment.risk_appetite
            - 0.35 * macro_environment.stagflation_risk
            + 0.25 * macro_environment.liquidity_score
            - 0.2 * max(macro_environment.commodity_pressure, 0.0)
        )
        leverage_signal = self.base_leverage + news_confidence + macro_risk_adjustment
        inflation_penalty = 0.25 * max(macro_environment.inflation_score, 0.0)
        target_leverage = np.clip(leverage_signal - inflation_penalty, 1.0, self.max_leverage)

        return pd.DataFrame(
            {
                "target_weight": target_weight,
                "target_leverage": target_leverage,
                "momentum": momentum,
                "volatility": volatility,
                "value": value_signal,
                "macro_cycle_bias": macro_cycle_bias,
                "ml_signal": ml_signal,
                "volatility_target_multiplier": vol_scaler,
            },
            index=prices.index,
        )


class Portfolio:
    """Tracks positions, cash, and portfolio level risk metrics."""

    def __init__(self, cash: float = 1_000_000.0, fee_bps: float = 1.0):
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self.total_borrow_cost: float = 0.0
        self.total_fees: float = 0.0
        self.fee_bps: float = fee_bps
        self.last_interest_date: Optional[dt.date] = None
        self._equity_hist: List[Tuple[pd.Timestamp, float]] = []

    def update_position(
        self,
        symbol: str,
        target_quantity: float,
        price: float,
        leverage: float,
        adv_value: float = 0.0,
        sigma_daily: Optional[float] = None,
    ) -> None:
        current = self.positions.get(symbol)
        current_qty = current.quantity if current else 0.0
        trade_qty = target_quantity - current_qty
        trade_value = trade_qty * price
        self.cash -= trade_value

        if current:
            same_direction = math.copysign(1.0, current_qty or 1.0) == math.copysign(1.0, target_quantity or 0.0)
            realized_qty = 0.0
            if math.isclose(target_quantity, 0.0, abs_tol=1e-9) or not same_direction:
                realized_qty = current_qty
            elif abs(target_quantity) < abs(current_qty):
                realized_qty = current_qty - target_quantity
            if not math.isclose(realized_qty, 0.0, abs_tol=1e-12):
                self.realized_pnl += realized_qty * (price - current.avg_price)

        slip_bps = estimate_slippage_bps(trade_value, adv_value, sigma_daily=sigma_daily)
        slippage = abs(trade_value) * (slip_bps / 10_000.0)
        if slippage > 0:
            self.cash -= slippage
            self.total_fees += slippage

        fees = abs(trade_value) * (self.fee_bps / 10_000.0)
        if fees > 0:
            self.cash -= fees
            self.total_fees += fees

        if math.isclose(target_quantity, 0.0, abs_tol=1e-9):
            self.positions.pop(symbol, None)
            return

        if current and math.copysign(1.0, current_qty or 1.0) == math.copysign(1.0, target_quantity or 0.0):
            if abs(target_quantity) > abs(current_qty):
                add_qty = abs(target_quantity) - abs(current_qty)
                avg_price = (abs(current_qty) * current.avg_price + add_qty * price) / abs(target_quantity)
            else:
                avg_price = current.avg_price
        else:
            avg_price = price

        self.positions[symbol] = Position(symbol=symbol, quantity=target_quantity, avg_price=avg_price, leverage=leverage)

    def total_equity(self, price_lookup: Dict[str, float]) -> float:
        equity = self.cash
        for symbol, position in self.positions.items():
            price = self._require_price(price_lookup, symbol)
            equity += position.market_value(price)
        return equity

    def total_exposure(self, price_lookup: Dict[str, float]) -> float:
        exposure = 0.0
        for symbol, position in self.positions.items():
            price = self._require_price(price_lookup, symbol)
            exposure += position.notional(price)
        return exposure

    def leverage(self, price_lookup: Dict[str, float]) -> float:
        equity = self.total_equity(price_lookup)
        if math.isclose(equity, 0.0):
            return 0.0
        return self.total_exposure(price_lookup) / abs(equity)

    def apply_leverage_cost(
        self,
        borrowing_rate: float,
        from_date: dt.date,
        to_date: dt.date,
        price_lookup: Dict[str, float],
    ) -> float:
        if self.last_interest_date is None:
            self.last_interest_date = from_date
        days = max((to_date - self.last_interest_date).days, 0)
        if days <= 0:
            return 0.0
        equity = self.total_equity(price_lookup)
        exposure = self.total_exposure(price_lookup)
        debt = max(exposure - equity, 0.0) if equity > 0 else exposure
        interest = debt * borrowing_rate * (days / 365.0)
        if interest > 0:
            self.cash -= interest
            self.total_borrow_cost += interest
        self.last_interest_date = to_date
        return interest

    def snapshot_equity(self, as_of: pd.Timestamp, price_lookup: Dict[str, float]) -> float:
        equity = self.total_equity(price_lookup)
        self._equity_hist.append((pd.Timestamp(as_of), equity))
        if len(self._equity_hist) > 252:
            self._equity_hist = self._equity_hist[-252:]
        return equity

    def realized_vol(self, lookback: int = 21) -> Optional[float]:
        if len(self._equity_hist) < lookback + 1:
            return None
        idx, vals = zip(*self._equity_hist)
        series = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()
        returns = series.pct_change().dropna()
        if returns.empty:
            return None
        window = returns.tail(lookback)
        if window.empty:
            return None
        return float(window.std() * math.sqrt(252))

    def summary(self, price_lookup: Dict[str, float]) -> Dict[str, object]:
        positions_summary = []
        for symbol, position in self.positions.items():
            price = self._require_price(price_lookup, symbol)
            market_value = position.market_value(price)
            pnl = market_value - position.avg_price * position.quantity
            pnl_pct = pnl / (abs(position.avg_price * position.quantity) + 1e-9)
            positions_summary.append(
                {
                    "symbol": symbol,
                    "direction": position.direction,
                    "quantity": position.quantity,
                    "market_value": market_value,
                    "average_price": position.avg_price,
                    "unrealized_pnl": pnl,
                    "unrealized_pnl_pct": pnl_pct,
                    "leverage": position.leverage,
                }
            )
        return {
            "cash": self.cash,
            "equity": self.total_equity(price_lookup),
            "gross_exposure": self.total_exposure(price_lookup),
            "leverage": self.leverage(price_lookup),
            "realized_pnl": self.realized_pnl,
            "borrow_cost": self.total_borrow_cost,
            "fees_paid": self.total_fees,
            "positions": positions_summary,
        }

    @staticmethod
    def _require_price(price_lookup: Dict[str, float], symbol: str) -> float:
        if symbol not in price_lookup:
            raise KeyError(f"Missing price for {symbol}")
        return price_lookup[symbol]


class RiskManager:
    """Ensures leverage, concentration, and stop-loss rules are respected."""

    def __init__(
        self,
        max_leverage: float = 3.0,
        position_limit: float = 0.2,
        stop_loss: float = 0.1,
        borrowing_rate: float = 0.05,
        trade_band: float = 0.05,
        min_notional: float = 5_000.0,
        lot_size: float = 1.0,
        rounding: str = "floor",
        cooldown_days: int = 3,
        trailing_stop: float = 0.12,
    ):
        self.max_leverage = max_leverage
        self.position_limit = position_limit
        self.stop_loss = stop_loss
        self.borrowing_rate = borrowing_rate
        self.trade_band = trade_band
        self.min_notional = min_notional
        self.lot_size = lot_size
        self.rounding = rounding
        self.cooldown_days = cooldown_days
        self.trailing_stop = trailing_stop
        self._cooldowns: Dict[str, pd.Timestamp] = {}
        self._max_favorable_px: Dict[str, float] = {}

    def register_trade(self, symbol: str, position: Position, last_px: float) -> None:
        if math.isclose(position.quantity, 0.0, abs_tol=1e-9):
            self._max_favorable_px.pop(symbol, None)
            return
        current_peak = self._max_favorable_px.get(symbol)
        if position.quantity > 0:
            best = max(current_peak or position.avg_price, last_px)
        else:
            best = min(current_peak or position.avg_price, last_px)
        self._max_favorable_px[symbol] = best

    def can_trade(self, symbol: str, as_of: pd.Timestamp) -> bool:
        cooldown_until = self._cooldowns.get(symbol)
        if cooldown_until is None:
            return True
        return as_of >= cooldown_until

    def evaluate_stop_losses(
        self,
        portfolio: Portfolio,
        price_lookup: Dict[str, float],
        as_of: pd.Timestamp,
    ) -> Dict[str, float]:
        adjustments: Dict[str, float] = {}
        for symbol, position in portfolio.positions.items():
            price = price_lookup.get(symbol)
            if price is None:
                continue
            pnl = (price - position.avg_price) * position.quantity
            basis = abs(position.avg_price * position.quantity) + 1e-9
            if pnl / basis <= -self.stop_loss:
                logger.info("stop_loss_triggered symbol=%s price=%.2f pnl=%.2f", symbol, price, pnl)
                adjustments[symbol] = 0.0
                self._cooldowns[symbol] = as_of + pd.tseries.offsets.BDay(self.cooldown_days)
                self._max_favorable_px.pop(symbol, None)
                continue
            peak = self._max_favorable_px.get(symbol, position.avg_price)
            if position.quantity > 0:
                if price <= peak * (1.0 - self.trailing_stop):
                    adjustments[symbol] = 0.0
                    self._cooldowns[symbol] = as_of + pd.tseries.offsets.BDay(self.cooldown_days)
                    self._max_favorable_px.pop(symbol, None)
            elif position.quantity < 0:
                if price >= peak * (1.0 + self.trailing_stop):
                    adjustments[symbol] = 0.0
                    self._cooldowns[symbol] = as_of + pd.tseries.offsets.BDay(self.cooldown_days)
                    self._max_favorable_px.pop(symbol, None)
        return adjustments

    def _round_qty(self, qty: float) -> float:
        lot = max(self.lot_size, 1.0)
        if self.rounding == "nearest":
            rounded = round(qty / lot) * lot
        else:
            rounded = math.copysign(math.floor(abs(qty) / lot) * lot, qty)
        if abs(rounded) < 1e-9:
            return 0.0
        return rounded

    def size_position(self, portfolio: Portfolio, price_lookup: Dict[str, float], symbol: str, target_weight: float, target_leverage: float) -> float:
        price = price_lookup[symbol]
        equity = portfolio.total_equity(price_lookup)
        if math.isclose(equity, 0.0):
            return 0.0
        max_weight = self.position_limit
        adjusted_weight = float(np.clip(target_weight, -max_weight, max_weight))
        leverage = min(target_leverage, self.max_leverage)
        desired_notional = equity * adjusted_weight * leverage
        current_position = portfolio.positions.get(symbol)
        current_notional_signed = (current_position.quantity * price) if current_position else 0.0
        current_notional = abs(current_notional_signed)
        max_position_notional = equity * self.max_leverage * self.position_limit
        desired_notional = float(np.clip(desired_notional, -max_position_notional, max_position_notional))

        total_exposure = portfolio.total_exposure(price_lookup)
        exposure_excluding_symbol = total_exposure - current_notional
        max_portfolio_exposure = equity * self.max_leverage
        available_capacity = max(max_portfolio_exposure - exposure_excluding_symbol, 0.0)
        desired_abs = min(abs(desired_notional), available_capacity)
        desired_notional = math.copysign(desired_abs, desired_notional)
        if abs(desired_notional - current_notional_signed) < max(self.min_notional, abs(desired_notional) * self.trade_band):
            return current_position.quantity if current_position else 0.0
        if available_capacity <= 0 and abs(desired_notional) > 0:
            logger.info("capacity_exhausted symbol=%s desired=%.2f", symbol, desired_notional)
        quantity = desired_notional / price
        return self._round_qty(quantity)


@dataclass
class TradeDecision:
    symbol: str
    target_weight: float
    target_leverage: float
    executed_quantity: float
    execution_price: float
    leverage_cost: float
    sentiment: SentimentProfile = field(default_factory=SentimentProfile)
    macro_environment: Optional[MacroEnvironment] = None
    indicators: Dict[str, float] = field(default_factory=dict)
    as_of: Optional[pd.Timestamp] = None


@dataclass
class Signal:
    symbol: str
    raw_weight: float
    max_weight: float


@dataclass
class Analysis:
    symbol: str
    price: float
    as_of: pd.Timestamp
    raw_weight: float
    target_leverage: float
    levered_raw: float
    max_weight: float
    adv_20: float
    sentiment: SentimentProfile
    macro_environment: Optional[MacroEnvironment]
    indicators: Dict[str, float]
    price_history: pd.DataFrame


def estimate_slippage_bps(
    trade_value: float,
    adv_value: float,
    sigma_daily: Optional[float] = None,
    half_spread_bps: float = 1.0,
    kappa: float = 15.0,
) -> float:
    if adv_value <= 0 or math.isclose(trade_value, 0.0):
        return half_spread_bps
    participation = min(abs(trade_value) / adv_value, 1.0)
    root_participation = math.sqrt(participation)
    vol_adjustment = 1.0
    if sigma_daily is not None and sigma_daily > 0:
        vol_adjustment = sigma_daily / 0.02
    impact = kappa * vol_adjustment * root_participation
    return half_spread_bps + impact


def _spearman_ic(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra.astype(float)
    rb = rb.astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(ra, rb) / denom)


def ewma_cov(returns: np.ndarray, span: int = 20) -> np.ndarray:
    if returns.size == 0:
        return np.zeros((0, 0), dtype=float)
    alpha = 2.0 / (span + 1.0)
    n_assets = returns.shape[1]
    mean = np.zeros(n_assets, dtype=float)
    cov = np.eye(n_assets) * 1e-6
    for row in returns:
        diff = row - mean
        mean = (1 - alpha) * mean + alpha * row
        cov = (1 - alpha) * cov + alpha * np.outer(diff, diff)
    return cov


def pca_risk(returns: np.ndarray, k: int = 5, shrink: float = 0.2) -> np.ndarray:
    if returns.size == 0:
        return np.zeros((0, 0), dtype=float)
    demeaned = returns - returns.mean(axis=0, keepdims=True)
    if demeaned.shape[0] < 2:
        var = np.var(demeaned, axis=0)
        return np.diag(np.maximum(var, 1e-6))
    U, S, Vt = np.linalg.svd(demeaned, full_matrices=False)
    k = min(k, Vt.shape[0])
    if k == 0:
        spec = np.var(demeaned, axis=0)
        return np.diag(np.maximum(spec, 1e-6))
    loadings = (Vt[:k].T * (S[:k] / math.sqrt(max(demeaned.shape[0] - 1, 1))))
    factor_cov = np.eye(k)
    common = loadings @ factor_cov @ loadings.T
    residual = demeaned - demeaned @ Vt[:k].T @ Vt[:k]
    spec = np.var(residual, axis=0)
    D = np.diag(np.maximum(spec, 1e-6))
    sigma = (1 - shrink) * (common + D) + shrink * np.diag(np.diag(common + D))
    eps = 1e-6 * np.trace(sigma) / sigma.shape[0]
    return sigma + eps * np.eye(sigma.shape[0])


def estimate_covariance_regime(
    data_fetcher: "DataFetcher",
    symbols: Sequence[str],
    as_of_date: dt.date,
    lookback: int = 126,
    span: int = 20,
    shrink: float = 0.3,
    use_pca: bool = False,
    pca_components: int = 5,
    precomputed_returns: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    if precomputed_returns is not None and not precomputed_returns.empty:
        frame = precomputed_returns.tail(lookback)
    else:
        start = (pd.Timestamp(as_of_date) - pd.tseries.offsets.BDay(int(lookback * 1.5))).date()
        returns: List[pd.Series] = []
        for symbol in symbols:
            history = data_fetcher.fetch_price_data(symbol, start, as_of_date)
            series = history["close"].pct_change().dropna().tail(lookback)
            returns.append(series.rename(symbol))
        frame = pd.concat(returns, axis=1).dropna() if returns else pd.DataFrame()
    if frame.shape[0] < 10:
        if frame.empty:
            vols = np.ones(len(symbols)) * 0.02
        else:
            vols = frame.std().fillna(0.02).reindex(symbols).fillna(0.02).values
        return np.diag(vols ** 2)
    values = frame.reindex(columns=symbols).values
    if use_pca:
        return pca_risk(values, k=min(pca_components, len(symbols)), shrink=shrink)
    ewma = ewma_cov(values, span=span)
    diag = np.diag(np.diag(ewma))
    sigma = (1 - shrink) * ewma + shrink * diag
    eps = 1e-6 * np.trace(sigma) / sigma.shape[0]
    return sigma + eps * np.eye(sigma.shape[0])


def solve_markowitz_eq(
    mu: np.ndarray,
    covariance: np.ndarray,
    w_prev: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    risk_aversion: float = 3.0,
    turn_penalty: float = 2.0,
    gross_target: float = 1.0,
    box: float = 0.2,
) -> np.ndarray:
    n = len(mu)
    if w_prev is None:
        w_prev = np.zeros(n, dtype=float)
    H = risk_aversion * covariance + turn_penalty * np.eye(n)
    rhs_w = mu + turn_penalty * w_prev
    try:
        if A is None or A.size == 0:
            weights = np.linalg.solve(H, rhs_w)
        else:
            m = A.shape[1]
            rhs = np.concatenate([rhs_w, b if b is not None else np.zeros(m)])
            KKT = np.block([[H, A], [A.T, np.zeros((m, m))]])
            solution = np.linalg.solve(KKT, rhs)
            weights = solution[:n]
    except np.linalg.LinAlgError:
        jitter = 1e-6 * (np.trace(H) / n if n > 0 else 1.0)
        H_reg = H + jitter * np.eye(n)
        if A is None or A.size == 0:
            weights = np.linalg.lstsq(H_reg, rhs_w, rcond=None)[0]
        else:
            m = A.shape[1]
            rhs = np.concatenate([rhs_w, b if b is not None else np.zeros(m)])
            KKT = np.block([[H_reg, A], [A.T, np.zeros((m, m))]])
            solution = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            weights = solution[:n]
    if not np.all(np.isfinite(weights)):
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    gross = np.sum(np.abs(weights)) + 1e-12
    weights = weights * (gross_target / gross)
    weights = np.clip(weights, -box, box)
    gross = np.sum(np.abs(weights)) + 1e-12
    return weights * (gross_target / gross)


def calibrate_lambda_for_turnover(
    mu: np.ndarray,
    covariance: np.ndarray,
    w_prev: np.ndarray,
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    risk_aversion: float,
    gross_target: float,
    box: float,
    tau_target: float,
    base_penalty: float,
    caps: Optional[np.ndarray] = None,
    post_cap_iters: int = 6,
    max_iter: int = 25,
) -> np.ndarray:
    if tau_target <= 0:
        solution = solve_markowitz_eq(
            mu,
            covariance,
            w_prev=w_prev,
            A=A,
            b=b,
            risk_aversion=risk_aversion,
            turn_penalty=base_penalty,
            gross_target=gross_target,
            box=box,
        )
        if caps is not None and caps.size:
            solution = enforce_box_and_constraints(solution, caps, gross_target, A, b)
        return solution
    lo, hi = 1e-6, 1e3
    candidate = w_prev.copy()
    for _ in range(max_iter):
        lam = math.sqrt(lo * hi)
        candidate = solve_markowitz_eq(
            mu,
            covariance,
            w_prev=w_prev,
            A=A,
            b=b,
            risk_aversion=risk_aversion,
            turn_penalty=lam,
            gross_target=gross_target,
            box=box,
        )
        if not np.all(np.isfinite(candidate)):
            candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
        if caps is not None and caps.size:
            candidate = enforce_box_and_constraints(candidate, caps, gross_target, A, b, iters=post_cap_iters)
        turnover = float(np.sum(np.abs(candidate - w_prev)))
        if turnover > tau_target:
            lo = lam
        else:
            hi = lam
        if abs(turnover - tau_target) < 1e-4:
            break
    if not np.all(np.isfinite(candidate)):
        return w_prev.copy()
    return candidate


def project_to_constraints(
    weights: np.ndarray, A: Optional[np.ndarray], b: Optional[np.ndarray]
) -> np.ndarray:
    if A is None or A.size == 0:
        return weights
    rhs = A.T @ weights - (b if b is not None else np.zeros(A.shape[1]))
    AtA = A.T @ A
    try:
        z = np.linalg.solve(AtA, rhs)
    except np.linalg.LinAlgError:
        z = np.linalg.lstsq(AtA, rhs, rcond=None)[0]
    return weights - A @ z


def enforce_box_and_constraints(
    weights: np.ndarray,
    caps: np.ndarray,
    gross_target: float,
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    iters: int = 8,
) -> np.ndarray:
    if weights.size == 0:
        return weights
    adjusted = weights.copy()
    for _ in range(iters):
        adjusted = np.clip(adjusted, -caps, caps)
        gross = np.sum(np.abs(adjusted)) + 1e-12
        adjusted = adjusted * (gross_target / gross)
        adjusted = project_to_constraints(adjusted, A, b)
    gross = np.sum(np.abs(adjusted)) + 1e-12
    adjusted = adjusted * (gross_target / gross)
    adjusted = np.clip(adjusted, -caps, caps)
    gross = np.sum(np.abs(adjusted)) + 1e-12
    adjusted *= gross_target / gross
    return adjusted


def apply_group_caps(
    weights: np.ndarray,
    groups: Optional[np.ndarray],
    cap: float,
    gross_target: float,
    iters: int = 6,
) -> np.ndarray:
    if groups is None or groups.size == 0 or weights.size == 0:
        return weights
    adjusted = weights.copy()
    for _ in range(iters):
        group_abs = groups.T @ np.abs(adjusted)
        over = group_abs > cap
        if not np.any(over):
            break
        scale = np.ones_like(group_abs)
        scale[over] = cap / (group_abs[over] + 1e-12)
        adjusted = adjusted * (groups @ scale)
        gross = np.sum(np.abs(adjusted)) + 1e-12
        adjusted *= gross_target / gross
    return adjusted


def _adjacency_from_corr(corr: np.ndarray, k: int) -> np.ndarray:
    n = corr.shape[0]
    adjacency = np.zeros_like(corr)
    for i in range(n):
        if corr.shape[1] <= 1:
            continue
        idx = np.argsort(-np.abs(corr[i]))[1 : min(k + 1, corr.shape[1])]
        weights = np.abs(corr[i, idx])
        adjacency[i, idx] = weights
        adjacency[idx, i] = np.maximum(adjacency[idx, i], weights)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def laplacian_smooth(signal: np.ndarray, corr: np.ndarray, k: int = 5, tau: float = 0.5) -> np.ndarray:
    n = len(signal)
    if n == 0 or corr.size == 0:
        return signal
    adjacency = _adjacency_from_corr(corr, k)
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    try:
        smoothed = np.linalg.solve(np.eye(n) + tau * laplacian, signal)
    except np.linalg.LinAlgError:
        return signal
    return smoothed


def build_cluster_exposures(returns: pd.DataFrame, n_clusters: int) -> Optional[np.ndarray]:
    if returns.empty or returns.shape[1] < 2:
        return None
    n_clusters = min(n_clusters, returns.shape[1])
    if n_clusters < 2:
        return None
    corr = returns.corr().fillna(0.0).abs().values
    try:
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0,
        )
        labels = model.fit_predict(corr)
    except Exception:
        try:
            distance = np.sqrt(np.maximum(0.0, 0.5 * (1 - np.clip(corr, -1.0, 1.0))))
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            labels = model.fit_predict(distance)
        except Exception:
            return None
    exposures = np.eye(n_clusters)[labels]
    return exposures


def _condition_exposures(A: np.ndarray, keep_first: bool = True, tol: float = 1e-6) -> np.ndarray:
    if A.size == 0:
        return A
    scale = np.linalg.norm(A, axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    scaled = A / scale
    try:
        _, s, Vt = np.linalg.svd(scaled, full_matrices=False)
    except np.linalg.LinAlgError:
        return A
    if s.size == 0:
        return A
    keep = s / s[0] > tol
    if keep_first and keep.size > 0:
        keep[0] = True
    projection = Vt[keep].T @ Vt[keep]
    return A @ projection


class Backtester:
    """Simple daily rebalancing backtester for sanity checks."""

    def run(self, bot: "TradingBot", symbols: List[str], start: dt.date, end: dt.date) -> Dict[str, object]:
        dates = pd.date_range(start, end, freq="B")
        equity_points: List[Tuple[pd.Timestamp, float]] = []
        weights_history: List[Tuple[pd.Timestamp, Dict[str, float]]] = []
        for current_date in dates:
            analyses: List[Analysis] = []
            if current_date.date() <= start:
                bot.end_of_day(current_date)
                equity = bot.portfolio.total_equity(bot.latest_prices)
                equity_points.append((current_date, equity))
                weights_snapshot = {
                    sym: float(bot.cross_sectional_weights.get(sym, 0.0)) for sym in symbols
                }
                weights_history.append((current_date, weights_snapshot))
                continue
            for symbol in symbols:
                try:
                    analyses.append(bot.analyze(symbol, start, current_date.date()))
                except Exception:
                    logger.exception("analysis_failed symbol=%s as_of=%s", symbol, current_date)
            try:
                bot.execute(analyses)
            except Exception:
                logger.exception("execution_failed as_of=%s", current_date)
            bot.end_of_day(current_date)
            equity = bot.portfolio.total_equity(bot.latest_prices)
            equity_points.append((current_date, equity))
            weights_snapshot = {
                sym: float(bot.cross_sectional_weights.get(sym, 0.0)) for sym in symbols
            }
            weights_history.append((current_date, weights_snapshot))
        if not equity_points:
            return {"equity": pd.Series(dtype=float), "sharpe": float("nan"), "max_drawdown": float("nan")}
        index, values = zip(*equity_points)
        curve = pd.Series(values, index=pd.DatetimeIndex(index))
        returns = curve.pct_change().dropna()
        if returns.empty:
            sharpe = float("nan")
        else:
            sharpe = float(returns.mean() / (returns.std() + 1e-12) * math.sqrt(252))
        drawdown = float((curve / curve.cummax() - 1.0).min()) if not curve.empty else float("nan")
        persist = getattr(getattr(bot, "config", None), "persist_backtest_artifacts", False)
        output_dir: Optional[Path] = None
        if persist:
            output_dir = Path("runs") / pd.Timestamp.utcnow().strftime("backtest_%Y%m%d%H%M%S")
            output_dir.mkdir(parents=True, exist_ok=True)
            curve.to_csv(output_dir / "equity.csv", header=["equity"])
        weights_records = [
            {"date": ts, **snapshot}
            for ts, snapshot in weights_history
        ]
        weights_df = pd.DataFrame.from_records(weights_records).set_index("date") if weights_records else pd.DataFrame()
        if persist and output_dir is not None and not weights_df.empty:
            weights_df.to_csv(output_dir / "weights.csv")
        turnover_series = weights_df.diff().abs().sum(axis=1) if not weights_df.empty else pd.Series(dtype=float)
        avg_turnover = float(turnover_series.mean()) if not turnover_series.empty else float("nan")
        if persist and output_dir is not None:
            summary_payload = {
                "symbols": symbols,
                "start": str(start),
                "end": str(end),
                "sharpe": sharpe,
                "max_drawdown": drawdown,
                "average_turnover": avg_turnover,
            }
            (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))
            logger.info("backtest_artifacts_written path=%s", output_dir)
        return {
            "equity": curve,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "artifact_path": output_dir,
        }


class TradingBot:
    """Coordinates data, strategy, and portfolio management."""

    def __init__(
        self,
        data_fetcher: Optional[DataFetcher] = None,
        news_analyzer: Optional[NewsAnalyzer] = None,
        macro_analyzer: Optional[MacroAnalyzer] = None,
        strategy: Optional[QuantStrategy] = None,
        risk_manager: Optional[RiskManager] = None,
        representation_engine: Optional[RepresentationLearningEngine] = None,
        initial_cash: float = 1_000_000.0,
        config: Optional[TradingConfig] = None,
    ) -> None:
        base_config = config or TradingConfig()
        if not math.isclose(initial_cash, base_config.initial_cash):
            base_config.initial_cash = initial_cash
        self.config = base_config
        self.rng = np.random.default_rng(self.config.random_seed)
        self.data_fetcher = data_fetcher or DataFetcher(
            random_seed=self.config.random_seed,
            offline=self.config.offline_mode,
        )
        self.data_fetcher.market_tz = self.config.market_tz
        self.data_fetcher.market_close_hour = self.config.market_close_hour
        self.news_analyzer = news_analyzer or NewsAnalyzer()
        self.macro_analyzer = macro_analyzer or MacroAnalyzer()
        self.strategy = strategy or QuantStrategy(
            volatility_target=self.config.volatility_target,
            max_leverage=self.config.risk_max_leverage,
            base_leverage=self.config.base_leverage,
            trade_band=self.config.trade_band,
            min_notional=self.config.min_notional,
        )
        self.risk_manager = risk_manager or RiskManager(
            max_leverage=self.strategy.max_leverage,
            position_limit=self.config.position_limit,
            stop_loss=self.config.stop_loss,
            borrowing_rate=self.config.borrowing_rate,
            trade_band=self.config.trade_band,
            min_notional=self.config.min_notional,
        )
        self.representation_engine = representation_engine or RepresentationLearningEngine(
            config=self.config,
        )
        self.portfolio = Portfolio(cash=self.config.initial_cash, fee_bps=self.config.fee_bps)
        self.latest_prices: Dict[str, float] = {}
        self.latest_macro_environment: Optional[MacroEnvironment] = None
        self.latest_representation_result: Optional[RepresentationLearningResult] = None
        self.price_cache_max_lookback_days = self.config.price_cache_max_lookback_days
        self.latest_signals: Dict[str, Signal] = {}
        self.cross_sectional_weights: Dict[str, float] = {}
        self.latest_adv: Dict[str, float] = {}
        self.latest_volatility: Dict[str, float] = {}
        self.alpha_combiner = AlphaCombiner(n_features=6)
        self._prev_alpha_snapshot: Optional[Tuple[pd.Timestamp, List[str], np.ndarray, Dict[str, float], np.ndarray]] = None
        self._ic_history: deque[float] = deque(maxlen=60)
        logger.info("run_config %s", vars(self.config))

    def _build_price_lookup(
        self,
        as_of: pd.Timestamp,
        current_symbol: str,
        current_price: float,
    ) -> Dict[str, float]:
        lookup = dict(self.latest_prices)
        lookup[current_symbol] = float(current_price)

        for sym, position in self.portfolio.positions.items():
            if sym in lookup:
                continue
            try:
                start = (as_of - pd.tseries.offsets.BDay(self.price_cache_max_lookback_days)).date()
                end = as_of.date()
                history = self.data_fetcher.fetch_price_data(sym, start, end)
                last_price = float(history["close"].dropna().iloc[-1])
                lookup[sym] = last_price
            except Exception:
                lookup[sym] = float(position.avg_price)
        self.latest_prices = lookup
        return lookup

    def _update_alpha_combiner_with_realized_returns(self, price_lookup: Dict[str, float]) -> None:
        if not self._prev_alpha_snapshot:
            return
        _, prev_symbols, prev_features, prev_prices, prev_mu_pred = self._prev_alpha_snapshot
        realized_features: List[np.ndarray] = []
        realized_returns: List[float] = []
        realized_predictions: List[float] = []
        for idx, symbol in enumerate(prev_symbols):
            prev_price = prev_prices.get(symbol)
            current_price = price_lookup.get(symbol)
            if prev_price is None or current_price is None or math.isclose(prev_price, 0.0):
                continue
            realized_return = current_price / prev_price - 1.0
            realized_features.append(prev_features[idx])
            realized_returns.append(realized_return)
            if prev_mu_pred.size > idx:
                realized_predictions.append(prev_mu_pred[idx])
        if realized_returns:
            features_matrix = np.vstack(realized_features)
            returns_array = np.array(realized_returns, dtype=float)
            demeaned_returns = returns_array - returns_array.mean()
            demeaned_features = features_matrix - features_matrix.mean(axis=0, keepdims=True)
            stds = demeaned_features.std(axis=0, keepdims=True)
            stds[stds < 1e-8] = 1.0
            normalized_features = np.clip(demeaned_features / stds, -5.0, 5.0)
            self.alpha_combiner.ensure_dimension(normalized_features.shape[1])
            self.alpha_combiner.update(normalized_features, demeaned_returns)
            if realized_predictions:
                preds = np.array(realized_predictions, dtype=float)
                mask = np.isfinite(preds) & np.isfinite(returns_array[: preds.size])
                if np.count_nonzero(mask) >= 3:
                    ic = _spearman_ic(preds[mask], returns_array[: preds.size][mask])
                    if np.isfinite(ic):
                        self._ic_history.append(float(ic))
        self._prev_alpha_snapshot = None

    @staticmethod
    def _build_alpha_matrix(analyses_sorted: List[Analysis]) -> Tuple[np.ndarray, np.ndarray]:
        if not analyses_sorted:
            return np.zeros((0, 0), dtype=float), np.zeros(0, dtype=float)
        feature_rows: List[List[float]] = []
        for analysis in analyses_sorted:
            indicators = analysis.indicators
            feature_rows.append(
                [
                    analysis.levered_raw,
                    indicators.get("momentum", 0.0),
                    -indicators.get("volatility", 0.0),
                    indicators.get("value", 0.0),
                    indicators.get("ml_signal", 0.0),
                    indicators.get("macro_cycle_bias", 0.0),
                ]
            )
        matrix = np.array(feature_rows, dtype=float)
        levered_raw = matrix[:, 0].copy()
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        stds[stds < 1e-8] = 1.0
        normalized = (matrix - means) / stds
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = np.clip(normalized, -5.0, 5.0)
        return normalized, levered_raw

    @staticmethod
    def _build_return_frame(analyses_sorted: List[Analysis], lookback: int) -> pd.DataFrame:
        if not analyses_sorted:
            return pd.DataFrame()
        frames: List[pd.Series] = []
        for analysis in analyses_sorted:
            history = getattr(analysis, "price_history", None)
            if history is None or history.empty or "close" not in history:
                continue
            returns = history["close"].pct_change().dropna()
            frames.append(returns.rename(analysis.symbol))
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, axis=1).dropna()
        if frame.empty:
            return frame
        if lookback > 0:
            frame = frame.tail(lookback)
        symbols = [analysis.symbol for analysis in analyses_sorted]
        frame = frame[[col for col in symbols if col in frame.columns]]
        return frame.reindex(columns=symbols)

    def _market_beta_exposure(
        self,
        returns_frame: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> Optional[np.ndarray]:
        if returns_frame.empty:
            return None
        try:
            start_date = returns_frame.index[0].date()
        except (IndexError, AttributeError):
            return None
        try:
            market_history = self.data_fetcher.fetch_price_data("SPY", start_date, as_of.date())
            market_returns = market_history["close"].pct_change().dropna()
        except Exception:
            return None
        betas: List[float] = []
        for symbol in returns_frame.columns:
            joined = pd.concat([returns_frame[symbol], market_returns], axis=1, join="inner").dropna()
            if joined.shape[0] < 20:
                betas.append(0.0)
                continue
            market = joined.iloc[:, 1].values
            asset = joined.iloc[:, 0].values
            var_market = np.var(market)
            if var_market < 1e-8:
                betas.append(0.0)
                continue
            cov = np.cov(asset, market, bias=True)[0, 1]
            betas.append(float(cov / var_market))
        betas_array = np.array(betas, dtype=float)
        if not np.any(np.isfinite(betas_array)):
            return None
        mean = np.nanmean(betas_array)
        std = np.nanstd(betas_array)
        if std < 1e-8 or np.isnan(std):
            return betas_array - mean
        return (betas_array - mean) / std

    def _constraint_matrix(
        self,
        analyses_sorted: List[Analysis],
        returns_frame: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        n = len(analyses_sorted)
        if n == 0:
            return np.zeros((0, 0)), np.zeros(0), None
        exposures_cols: List[np.ndarray] = []
        adv_values = np.array([max(analysis.adv_20, 0.0) for analysis in analyses_sorted], dtype=float)
        if adv_values.size:
            positive_mask = adv_values > 0
            if np.count_nonzero(positive_mask) >= 2:
                log_adv = np.full_like(adv_values, np.nan, dtype=float)
                log_adv[positive_mask] = np.log(adv_values[positive_mask])
                if np.any(np.isfinite(log_adv)):
                    mean_log = float(np.nanmean(log_adv))
                    std_log = float(np.nanstd(log_adv))
                    if np.isnan(std_log) or std_log < 1e-8:
                        zlogadv = np.zeros_like(adv_values)
                    else:
                        zlogadv = np.nan_to_num(
                            (log_adv - mean_log) / std_log,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                    exposures_cols.append(zlogadv.reshape(-1, 1))
                else:
                    exposures_cols.append(np.zeros((n, 1), dtype=float))
            else:
                exposures_cols.append(np.zeros((n, 1), dtype=float))
        beta_vector = self._market_beta_exposure(returns_frame, as_of)
        if beta_vector is not None:
            exposures_cols.append(beta_vector.reshape(-1, 1))
        cluster_matrix = build_cluster_exposures(returns_frame, self.config.cluster_count)
        if cluster_matrix is not None:
            exposures_cols.append(cluster_matrix)
        if exposures_cols:
            exposures = np.column_stack(exposures_cols)
        else:
            exposures = np.zeros((n, 0), dtype=float)
        ones = np.ones((n, 1), dtype=float)
        if exposures.size:
            A = np.column_stack([ones, exposures])
        else:
            A = ones
        b = np.zeros(A.shape[1], dtype=float)
        A = _condition_exposures(A)
        return A, b, cluster_matrix

    def analyze(self, symbol: str, start: dt.date, end: dt.date) -> Analysis:
        prices = self.data_fetcher.fetch_price_data(symbol, start, end)
        news = self.data_fetcher.fetch_news(symbol, as_of=prices.index[-1])
        if news:
            tz = getattr(self.data_fetcher, "market_tz", "America/New_York")
            close_hour = getattr(self.data_fetcher, "market_close_hour", 16)
            as_of_ts = pd.Timestamp(prices.index[-1])
            if as_of_ts.tzinfo is None:
                local_cutoff = as_of_ts.tz_localize(tz)
            else:
                local_cutoff = as_of_ts.tz_convert(tz)
            local_cutoff = local_cutoff.normalize() + pd.Timedelta(hours=close_hour)
            cutoff_utc = local_cutoff.tz_convert("UTC")
            assert all(
                (item.get("published_at") is None) or (item["published_at"] <= cutoff_utc)
                for item in news
            ), "News leakage: found item after cutoff"
        sentiment = self.news_analyzer.analyze(news)
        macro_snapshot = self.data_fetcher.fetch_macro_snapshot(end)
        macro_environment = self.macro_analyzer.evaluate(macro_snapshot)
        self.latest_macro_environment = macro_environment
        macro_gross = float(
            np.clip(
                1.0
                + 0.5 * macro_environment.risk_appetite
                - 0.4 * macro_environment.stagflation_risk
                + 0.3 * macro_environment.liquidity_score,
                0.6,
                1.4,
            )
        )
        self.config.gross_leverage_target = macro_gross
        representation_result = self.representation_engine.train_and_predict(prices, news)
        self.latest_representation_result = representation_result
        signals = self.strategy.generate_signals(
            prices,
            sentiment,
            macro_environment,
            learned_signal=representation_result.predictions,
        )
        latest_row = signals.iloc[-1]
        price = float(prices["close"].iloc[-1])
        as_of = prices.index[-1]
        ml_loss = float("nan")
        if representation_result.training_loss is not None:
            ml_loss = float(representation_result.training_loss)
        adv_20_series = prices["adv_20"] if "adv_20" in prices.columns else None
        adv_20 = float(adv_20_series.iloc[-1]) if adv_20_series is not None else 0.0
        if math.isnan(adv_20):
            adv_20 = 0.0
        self.latest_adv[symbol] = adv_20
        raw_weight = float(latest_row["target_weight"])
        target_leverage = float(latest_row["target_leverage"])
        indicators = {
            "momentum": float(latest_row["momentum"]),
            "volatility": float(latest_row["volatility"]),
            "value": float(latest_row["value"]),
            "macro_cycle_bias": float(latest_row["macro_cycle_bias"]),
            "ml_signal": float(latest_row["ml_signal"]),
            "volatility_target_multiplier": float(latest_row["volatility_target_multiplier"]),
            "ml_training_loss": ml_loss,
            "raw_target_leverage": target_leverage,
        }
        levered_raw = raw_weight * target_leverage
        self.latest_volatility[symbol] = indicators["volatility"]
        return Analysis(
            symbol=symbol,
            price=price,
            as_of=as_of,
            raw_weight=raw_weight,
            target_leverage=target_leverage,
            levered_raw=levered_raw,
            max_weight=self.config.position_limit,
            adv_20=adv_20,
            sentiment=sentiment,
            macro_environment=macro_environment,
            indicators=indicators,
            price_history=prices.copy(),
        )

    def execute(self, analyses: List[Analysis]) -> Dict[str, TradeDecision]:
        decisions: Dict[str, TradeDecision] = {}
        if not analyses:
            return decisions
        analyses_sorted = sorted(analyses, key=lambda a: (a.as_of, a.symbol))
        as_of = max(a.as_of for a in analyses_sorted)
        price_lookup: Dict[str, float] = dict(self.latest_prices)
        for analysis in analyses_sorted:
            price_lookup = self._build_price_lookup(analysis.as_of, analysis.symbol, analysis.price)
            self.latest_adv[analysis.symbol] = analysis.adv_20

        self._update_alpha_combiner_with_realized_returns(price_lookup)

        stops = self.risk_manager.evaluate_stop_losses(self.portfolio, price_lookup, as_of)
        for stop_symbol, quantity in stops.items():
            stop_price = price_lookup[stop_symbol]
            adv_shares = self.latest_adv.get(stop_symbol, 0.0) or 0.0
            adv_value = stop_price * max(adv_shares, 0.0)
            sigma_daily = self.latest_volatility.get(stop_symbol)
            self.portfolio.update_position(
                stop_symbol,
                quantity,
                stop_price,
                leverage=1.0,
                adv_value=adv_value,
                sigma_daily=sigma_daily,
            )
            self.latest_signals.pop(stop_symbol, None)

        self.latest_signals = {}
        for analysis in analyses_sorted:
            signal = Signal(analysis.symbol, analysis.levered_raw, analysis.max_weight)
            self.latest_signals[analysis.symbol] = signal

        symbols = [analysis.symbol for analysis in analyses_sorted]
        features_matrix, levered_raw = self._build_alpha_matrix(analyses_sorted)
        self.alpha_combiner.ensure_dimension(features_matrix.shape[1] if features_matrix.size else self.alpha_combiner.n_features)
        mu_pred = self.alpha_combiner.predict(features_matrix) if features_matrix.size else np.zeros(len(analyses_sorted))
        mu_pred_map = {symbol: float(mu_pred[idx]) for idx, symbol in enumerate(symbols)} if mu_pred.size else {}
        returns_frame = self._build_return_frame(analyses_sorted, self.config.cov_lookback)
        corr_matrix = returns_frame.corr().fillna(0.0).values if returns_frame.shape[1] >= 2 else np.zeros((0, 0))
        if self.alpha_combiner.has_history and mu_pred.size:
            ic_avg = float(np.mean(self._ic_history)) if self._ic_history else 0.0
            weight_pred = float(np.clip(0.5 + 2.0 * ic_avg, 0.0, 1.0))
            mu_signal = weight_pred * mu_pred + (1.0 - weight_pred) * levered_raw
        else:
            weight_pred = 0.0
            mu_signal = levered_raw.copy()

        if corr_matrix.size and mu_signal.size >= 2:
            mu_smoothed = laplacian_smooth(
                mu_signal,
                corr_matrix,
                k=max(1, min(self.config.laplacian_k, len(mu_signal) - 1)),
                tau=self.config.laplacian_tau,
            )
        else:
            mu_smoothed = mu_signal

        A, b, cluster_matrix = self._constraint_matrix(analyses_sorted, returns_frame, as_of)

        covariance = (
            estimate_covariance_regime(
                self.data_fetcher,
                symbols,
                as_of.date(),
                lookback=self.config.cov_lookback,
                span=self.config.cov_span,
                shrink=self.config.cov_shrink,
                use_pca=self.config.risk_use_pca,
                pca_components=self.config.risk_pca_components,
                precomputed_returns=returns_frame,
            )
            if mu_smoothed.size
            else np.zeros((0, 0))
        )
        if covariance.size and np.linalg.cond(covariance) > 1e8:
            jitter = 1e-4 * np.trace(covariance) / covariance.shape[0]
            covariance = covariance + jitter * np.eye(covariance.shape[0])
        total_equity = max(self.portfolio.total_equity(price_lookup), 1e-9)
        w_prev = np.zeros(len(symbols), dtype=float)
        for idx, analysis in enumerate(analyses_sorted):
            position = self.portfolio.positions.get(analysis.symbol)
            if not position:
                continue
            px = price_lookup.get(analysis.symbol, analysis.price)
            w_prev[idx] = position.market_value(px) / total_equity

        caps = np.full(len(symbols), self.config.position_limit, dtype=float)
        if symbols:
            adv_dollars = np.array([analysis.price * max(analysis.adv_20, 0.0) for analysis in analyses_sorted], dtype=float)
            caps_adv = np.where(
                adv_dollars > 0,
                self.config.adv_capacity_ratio * adv_dollars / total_equity,
                self.config.position_limit,
            )
            caps = np.minimum(
                caps,
                np.nan_to_num(
                    caps_adv,
                    nan=self.config.position_limit,
                    posinf=self.config.position_limit,
                    neginf=self.config.position_limit,
                ),
            )

        try:
            if mu_smoothed.size:
                w_candidate = calibrate_lambda_for_turnover(
                    mu_smoothed,
                    covariance,
                    w_prev,
                    A,
                    b,
                    risk_aversion=3.0,
                    gross_target=self.config.gross_leverage_target,
                    box=self.config.position_limit,
                    tau_target=self.config.turnover_target,
                    base_penalty=self.config.base_turn_penalty,
                    caps=caps,
                )
            else:
                w_candidate = np.zeros(len(symbols))
        except np.linalg.LinAlgError:
            logger.exception("markowitz_solve_failed as_of=%s", as_of.date())
            fallback = mu_smoothed if mu_smoothed.size else levered_raw
            if fallback.size == 0:
                w_candidate = np.zeros(len(symbols))
            else:
                gross = np.sum(np.abs(fallback)) + 1e-12
                scaled = fallback * (self.config.gross_leverage_target / gross)
                w_candidate = np.clip(scaled, -self.config.position_limit, self.config.position_limit)
        except Exception:
            logger.exception("unexpected_optimization_error as_of=%s", as_of.date())
            w_candidate = np.zeros(len(symbols))

        if symbols:
            w_opt = enforce_box_and_constraints(
                w_candidate,
                caps,
                self.config.gross_leverage_target,
                A,
                b,
            )
            if cluster_matrix is not None and cluster_matrix.size:
                cluster_cap = self.config.gross_leverage_target / max(1, cluster_matrix.shape[1])
                w_opt = apply_group_caps(w_opt, cluster_matrix, cluster_cap, self.config.gross_leverage_target)
                w_opt = enforce_box_and_constraints(
                    w_opt,
                    caps,
                    self.config.gross_leverage_target,
                    A,
                    b,
                )
        else:
            w_opt = np.zeros(0)

        if w_opt.size:
            if caps.size and not np.all(np.abs(w_opt) <= caps + 1e-5):
                logger.warning("box_clip_applied as_of=%s", as_of.date())
                w_opt = np.clip(w_opt, -caps, caps)
            gross_check = float(np.sum(np.abs(w_opt)))
            target_gross = self.config.gross_leverage_target
            if not (
                abs(gross_check - target_gross) < 5e-3
                or math.isclose(gross_check, 0.0, abs_tol=1e-9)
            ):
                logger.warning(
                    "gross_rescale_applied as_of=%s gross=%.6f target=%.6f",
                    as_of.date(),
                    gross_check,
                    target_gross,
                )
                if gross_check > 0:
                    w_opt *= target_gross / gross_check

        if mu_smoothed.size:
            try:
                mu_norm = float(np.linalg.norm(mu_smoothed))
                constraint_violation = float(np.linalg.norm(A.T @ w_opt - b)) if A.size else 0.0
                eigvals = np.linalg.eigvalsh(covariance) if covariance.size else np.array([0.0])
                lambda_min = float(eigvals.min()) if eigvals.size else 0.0
                cond_number = float(np.linalg.cond(covariance)) if covariance.size else 0.0
                ex_ante_risk = float(w_opt @ covariance @ w_opt) if covariance.size else 0.0
                turnover = float(np.sum(np.abs(w_opt - w_prev)))
                gross_exposure = float(np.sum(np.abs(w_opt)))
                active_boxes = int(np.sum(np.isclose(np.abs(w_opt), caps, atol=1e-6))) if caps.size else 0
                group_utilization = 0.0
                if cluster_matrix is not None and cluster_matrix.size:
                    cluster_cap = self.config.gross_leverage_target / max(1, cluster_matrix.shape[1])
                    group_abs = cluster_matrix.T @ np.abs(w_opt)
                    group_utilization = float(np.max(group_abs / (cluster_cap + 1e-12)))
                logger.info(
                    "optimizer_diagnostics as_of=%s mu_norm=%.4f constraint_norm=%.4e lambda_min=%.6e cond=%.2f risk=%.6f turnover_l1=%.4f gross=%.4f boxes=%d group_util=%.2f",
                    as_of.date(),
                    mu_norm,
                    constraint_violation,
                    lambda_min,
                    cond_number,
                    ex_ante_risk,
                    turnover,
                    gross_exposure,
                    active_boxes,
                    group_utilization,
                )
            except Exception:
                logger.exception("optimizer_diagnostics_failed as_of=%s", as_of.date())

        self.cross_sectional_weights = {symbol: float(weight) for symbol, weight in zip(symbols, w_opt)}
        if self.cross_sectional_weights:
            logger.info(
                "cross_sectional_allocation as_of=%s %s",
                as_of.date(),
                {k: round(v, 4) for k, v in self.cross_sectional_weights.items()},
            )

        for analysis in analyses_sorted:
            can_trade = self.risk_manager.can_trade(analysis.symbol, as_of)
            allocated_weight = self.cross_sectional_weights.get(analysis.symbol, analysis.levered_raw)
            if not can_trade:
                target_quantity = (
                    self.portfolio.positions.get(analysis.symbol).quantity
                    if analysis.symbol in self.portfolio.positions
                    else 0.0
                )
            else:
                target_quantity = self.risk_manager.size_position(
                    portfolio=self.portfolio,
                    price_lookup=price_lookup,
                    symbol=analysis.symbol,
                    target_weight=allocated_weight,
                    target_leverage=1.0,
                )
            adv_shares = max(analysis.adv_20, 0.0)
            adv_value = analysis.price * adv_shares if adv_shares > 0 else 0.0
            self.portfolio.update_position(
                analysis.symbol,
                target_quantity,
                analysis.price,
                leverage=1.0,
                adv_value=adv_value,
                sigma_daily=analysis.indicators.get("volatility"),
            )
            position = self.portfolio.positions.get(analysis.symbol)
            if position:
                self.risk_manager.register_trade(analysis.symbol, position, analysis.price)
            else:
                self.risk_manager.register_trade(
                    analysis.symbol,
                    Position(analysis.symbol, 0.0, analysis.price, 1.0),
                    analysis.price,
                )
            decision_indicators = dict(analysis.indicators)
            decision_indicators["alpha_combiner_weight"] = weight_pred
            if mu_pred_map:
                decision_indicators["alpha_combiner_signal"] = mu_pred_map.get(analysis.symbol, 0.0)
            decisions[analysis.symbol] = TradeDecision(
                symbol=analysis.symbol,
                target_weight=float(allocated_weight),
                target_leverage=1.0,
                executed_quantity=target_quantity,
                execution_price=analysis.price,
                leverage_cost=0.0,
                sentiment=analysis.sentiment,
                macro_environment=analysis.macro_environment,
                indicators=decision_indicators,
                as_of=analysis.as_of,
            )

        self.latest_prices = price_lookup
        if features_matrix.size:
            prior_prices = {analysis.symbol: analysis.price for analysis in analyses_sorted}
            self._prev_alpha_snapshot = (
                as_of,
                symbols,
                features_matrix.copy(),
                prior_prices,
                mu_pred.copy() if mu_pred.size else np.zeros(len(symbols)),
            )
        port_equity = self.portfolio.snapshot_equity(as_of, price_lookup)
        realized_vol = self.portfolio.realized_vol(lookback=21)
        if realized_vol is not None and realized_vol > 0:
            scale = float(np.clip(self.config.volatility_target / realized_vol, 0.5, 2.0))
            self.risk_manager.max_leverage = float(np.clip(self.config.risk_max_leverage * scale, 1.0, 5.0))
            self.strategy.max_leverage = self.risk_manager.max_leverage

        equity = port_equity
        exposure = self.portfolio.total_exposure(price_lookup)
        portfolio_leverage = self.portfolio.leverage(price_lookup)
        realized_vol_value = realized_vol if realized_vol is not None else float("nan")
        for decision in decisions.values():
            decision.indicators["portfolio_realized_vol"] = realized_vol_value
            logger.info(
                "portfolio_state symbol=%s as_of=%s equity=%.2f exposure=%.2f leverage=%.2f cash=%.2f realized_pnl=%.2f borrow_cost=%.2f fees=%.2f realized_vol=%.4f",
                decision.symbol,
                as_of.date(),
                equity,
                exposure,
                portfolio_leverage,
                self.portfolio.cash,
                self.portfolio.realized_pnl,
                self.portfolio.total_borrow_cost,
                self.portfolio.total_fees,
                realized_vol_value,
            )

        return decisions

    def end_of_day(self, as_of: pd.Timestamp) -> None:
        if not self.latest_prices:
            return
        from_date = (as_of - pd.tseries.offsets.BDay(1)).date()
        self.portfolio.apply_leverage_cost(
            self.risk_manager.borrowing_rate,
            from_date,
            as_of.date(),
            self.latest_prices,
        )

    def trade(self, symbol: str, start: dt.date, end: dt.date) -> TradeDecision:
        analysis = self.analyze(symbol, start, end)
        decisions = self.execute([analysis])
        if symbol in decisions:
            return decisions[symbol]
        current_qty = (
            self.portfolio.positions.get(symbol).quantity if symbol in self.portfolio.positions else 0.0
        )
        fallback_decision = TradeDecision(
            symbol=symbol,
            target_weight=0.0,
            target_leverage=1.0,
            executed_quantity=current_qty,
            execution_price=analysis.price,
            leverage_cost=0.0,
            sentiment=analysis.sentiment,
            macro_environment=analysis.macro_environment,
            indicators=dict(analysis.indicators),
            as_of=analysis.as_of,
        )
        return fallback_decision

    def summary(self) -> Dict[str, object]:
        return self.portfolio.summary(self.latest_prices)


def format_summary(summary: Dict[str, object]) -> str:
    lines = [
        "Portfolio Summary:",
        f"  Cash: ${summary['cash']:.2f}",
        f"  Equity: ${summary['equity']:.2f}",
        f"  Gross Exposure: ${summary['gross_exposure']:.2f}",
        f"  Leverage: {summary['leverage']:.2f}x",
        f"  Realized PnL: ${summary['realized_pnl']:.2f}",
        f"  Borrow Cost Paid: ${summary['borrow_cost']:.2f}",
        f"  Fees Paid: ${summary['fees_paid']:.2f}",
        "  Positions:",
    ]
    positions = summary.get("positions", [])
    if not positions:
        lines.append("    (No open positions)")
    for position in positions:
        lines.append(
            "    {symbol}: {direction} {quantity:.2f} @ ${average_price:.2f} | MV=${market_value:.2f} | PnL=${unrealized_pnl:.2f} ({unrealized_pnl_pct:.2%}) | Leverage={leverage:.2f}".format(
                **position
            )
        )
    return "\n".join(lines)


def _parse_date(date_str: Optional[str]) -> Optional[dt.date]:
    """Parse a string into a date if provided."""

    if not date_str:
        return None
    return pd.to_datetime(date_str).date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading bot entrypoint")
    parser.add_argument("--symbols", default="AAPL,TSLA,XLF", help="Comma-separated list of tickers")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--years", type=int, default=1, help="Years of history if --start not supplied")
    parser.add_argument("--offline", action="store_true", help="Use synthetic offline data sources")
    parser.add_argument("--persist", action="store_true", help="Persist backtest artefacts under ./runs")
    parser.add_argument("--backtest", action="store_true", help="Run a full backtest instead of single evaluation")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for stdout",
    )
    args = parser.parse_args()

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper(), logging.INFO),
            format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        )

    end_date = _parse_date(args.end) or dt.date.today()
    start_date = _parse_date(args.start) or (end_date - dt.timedelta(days=365 * args.years))
    symbols = [token.strip().upper() for token in args.symbols.split(",") if token.strip()]

    config = TradingConfig(offline_mode=args.offline, persist_backtest_artifacts=args.persist)
    bot = TradingBot(config=config)

    if args.backtest:
        logger.info(
            "running_backtest symbols=%s start=%s end=%s offline=%s", symbols, start_date, end_date, args.offline
        )
        results = Backtester().run(bot, symbols, start_date, end_date)
        logger.info(
            "backtest_result sharpe=%.3f max_drawdown=%.2f%% final_equity=%s",
            results.get("sharpe", float("nan")),
            100 * results.get("max_drawdown", float("nan")),
            ("$%.2f" % results["equity"].iloc[-1]) if not results.get("equity", pd.Series(dtype=float)).empty else "n/a",
        )
        artifact_path = results.get("artifact_path")
        if artifact_path:
            logger.info("backtest_artifacts path=%s", artifact_path)
        return

    analyses: List[Analysis] = []
    for symbol in symbols:
        logger.info("evaluating_trades symbol=%s start=%s end=%s", symbol, start_date, end_date)
        analyses.append(bot.analyze(symbol, start_date, end_date))
    decisions_map = bot.execute(analyses)
    bot.end_of_day(pd.Timestamp(end_date))

    for symbol in symbols:
        decision = decisions_map.get(symbol)
        if decision is None:
            continue
        exposure = bot.portfolio.total_exposure(bot.latest_prices)
        leverage = bot.portfolio.leverage(bot.latest_prices)
        logger.info(
            "trade_execution symbol=%s as_of=%s weight=%.4f leverage=%.2f qty=%.2f price=%.2f leverage_cost=%.2f exposure=%.2f",
            decision.symbol,
            decision.as_of.date() if decision.as_of is not None else "",
            decision.target_weight,
            leverage,
            decision.executed_quantity,
            decision.execution_price,
            decision.leverage_cost,
            exposure,
        )
        logger.info(
            "signal_context symbol=%s target_leverage=%.2f momentum=%.4f volatility=%.6f value=%.4f vol_target=%.4f ml_signal=%.4f ml_loss=%.6f",
            decision.symbol,
            decision.indicators.get("raw_target_leverage", decision.target_leverage),
            decision.indicators.get("momentum", float("nan")),
            decision.indicators.get("volatility", float("nan")),
            decision.indicators.get("value", float("nan")),
            decision.indicators.get("volatility_target_multiplier", float("nan")),
            decision.indicators.get("ml_signal", float("nan")),
            decision.indicators.get("ml_training_loss", float("nan")),
        )
        logger.info(
            "sentiment_summary symbol=%s macro=%.4f macro_conf=%.2f industry=%.4f industry_conf=%.2f",
            decision.symbol,
            decision.sentiment.macro,
            decision.sentiment.macro_confidence,
            decision.sentiment.industry,
            decision.sentiment.industry_confidence,
        )
        if decision.macro_environment:
            descriptors = ",".join(decision.macro_environment.descriptors)
            logger.info(
                "macro_snapshot symbol=%s growth=%.4f inflation=%.4f policy_tighten_prob=%.4f risk_appetite=%.4f stagflation_risk=%.4f commodity=%.4f liquidity=%.4f housing=%.4f consumer=%.4f tags=%s",
                decision.symbol,
                decision.macro_environment.growth_score,
                decision.macro_environment.inflation_score,
                decision.macro_environment.policy_tightening_probability,
                decision.macro_environment.risk_appetite,
                decision.macro_environment.stagflation_risk,
                decision.macro_environment.commodity_pressure,
                decision.macro_environment.liquidity_score,
                decision.macro_environment.housing_cycle_strength,
                decision.macro_environment.consumer_mood,
                descriptors,
            )
    print(format_summary(bot.summary()))


if __name__ == "__main__":
    main()
