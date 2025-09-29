"""Trading bot prototype with quantitative, news-aware decision making."""
from __future__ import annotations

import datetime as dt
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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


@dataclass
class MacroEnvironment:
    """Normalized description of the prevailing macroeconomic regime."""

    growth_score: float
    inflation_score: float
    policy_tightening_probability: float
    risk_appetite: float
    stagflation_risk: float
    descriptors: List[str] = field(default_factory=list)


class DataFetcher:
    """Fetches market and news data. Uses synthetic data as an offline fallback."""

    def __init__(self, price_source: Optional[str] = None, news_source: Optional[str] = None):
        self.price_source = price_source
        self.news_source = news_source
        self.session = requests.Session()

    def fetch_price_data(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        if start >= end:
            raise ValueError("start must be before end")

        dates = pd.date_range(start=start, end=end, freq="B")
        if len(dates) == 0:
            raise ValueError("No business days in selected range")

        try:
            return self._fetch_price_data_from_source(symbol, dates)
        except Exception as exc:  # pragma: no cover - network branch
            logging.warning("Falling back to synthetic prices for %s due to %s", symbol, exc)
            return self._generate_synthetic_prices(symbol, dates)

    def _fetch_price_data_from_source(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        if not self.price_source:
            raise RuntimeError("No price source configured")
        response = self.session.get(self.price_source, params={"symbol": symbol}, timeout=5)
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError("Empty price data")
        frame = pd.DataFrame(data)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").reindex(dates).interpolate().ffill().bfill()
        return frame[["open", "high", "low", "close", "volume"]]

    def _generate_synthetic_prices(self, symbol: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        rng = np.random.default_rng(abs(hash(symbol)) % (2 ** 32))
        steps = rng.normal(loc=0.0005, scale=0.02, size=len(dates))
        prices = 100 * np.exp(np.cumsum(steps))
        highs = prices * (1 + rng.uniform(0.0, 0.02, size=len(dates)))
        lows = prices * (1 - rng.uniform(0.0, 0.02, size=len(dates)))
        volumes = rng.integers(1_000_000, 5_000_000, size=len(dates))
        return pd.DataFrame(
            {
                "open": prices * (1 + rng.normal(0, 0.005, size=len(dates))),
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )

    def fetch_news(self, symbol: str) -> List[Dict[str, str]]:
        rng = np.random.default_rng(abs(hash(symbol)) % (2 ** 32))
        macro_headlines = [
            {
                "category": "macro",
                "headline": "Central bank signals cautious approach to interest rates",
                "body": "Policy makers hint at balancing inflation concerns with growth support.",
            },
            {
                "category": "macro",
                "headline": "Inflation eases as supply chains normalize",
                "body": "Commodity prices decline while consumer demand remains resilient.",
            },
            {
                "category": "macro",
                "headline": "Stronger dollar pressures emerging markets",
                "body": "Investors reassess risk appetite amid currency volatility.",
            },
        ]
        industry_headlines = [
            {
                "category": "industry",
                "headline": f"{symbol} suppliers report improving order flow",
                "body": "Upstream partners note backlog stabilization and better inventory management.",
            },
            {
                "category": "industry",
                "headline": f"Regulatory scrutiny rises across the {symbol} industry",
                "body": "Analysts warn of higher compliance costs and slower product approvals.",
            },
            {
                "category": "industry",
                "headline": f"Technological breakthroughs reshape {symbol} competitive landscape",
                "body": "Startups challenge incumbents with faster deployment cycles and AI integration.",
            },
        ]
        news_items = []
        for pool in (macro_headlines, industry_headlines):
            count = rng.integers(1, len(pool) + 1)
            news_items.extend(rng.choice(pool, size=count, replace=False).tolist())
        return news_items

    def fetch_macro_snapshot(self, as_of: dt.date) -> MacroSnapshot:
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
        return MacroSnapshot(
            gdp_growth=float(gdp_growth),
            inflation=float(inflation),
            unemployment=float(max(unemployment, 0.02)),
            yield_curve_slope=float(yield_curve_slope),
            manufacturing_pmi=float(manufacturing_pmi),
            credit_spread=float(max(credit_spread, 0.005)),
            usd_trend=float(usd_trend),
        )


class NewsAnalyzer:
    """Transforms qualitative news into simple sentiment signals."""

    POSITIVE_WORDS: Iterable[str] = (
        "beat",
        "growth",
        "improving",
        "optimistic",
        "resilient",
        "strong",
        "support",
    )
    NEGATIVE_WORDS: Iterable[str] = (
        "concerns",
        "decline",
        "downturn",
        "pressure",
        "risk",
        "scrutiny",
        "volatile",
    )

    def __init__(self) -> None:
        self.positive_words = {w.lower() for w in self.POSITIVE_WORDS}
        self.negative_words = {w.lower() for w in self.NEGATIVE_WORDS}

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
        words = re.findall(r"[A-Za-z']+", text.lower())
        pos_hits = sum(word in self.positive_words for word in words)
        neg_hits = sum(word in self.negative_words for word in words)
        total = pos_hits + neg_hits
        if total == 0:
            return 0.0
        return (pos_hits - neg_hits) / total


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
        risk_signal = 0.5 * growth_score - 0.4 * inflation_score + 0.4 * credit_relief - 0.2 * np.tanh(
            snapshot.usd_trend / 2.0
        )
        risk_appetite = float(np.tanh(risk_signal))

        stagflation_raw = (
            0.4 * (inflation_score - growth_score)
            + 0.3 * np.tanh((snapshot.credit_spread - 0.02) * 120)
        )
        stagflation_risk = float(np.clip(0.5 + 0.5 * stagflation_raw, 0.0, 1.0))

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

        return MacroEnvironment(
            growth_score=growth_score,
            inflation_score=inflation_score,
            policy_tightening_probability=policy_tightening_probability,
            risk_appetite=risk_appetite,
            stagflation_risk=stagflation_risk,
            descriptors=descriptors,
        )


class QuantStrategy:
    """Generates long/short signals using price action, macro, and news sentiment."""

    def __init__(self, momentum_window: int = 21, volatility_window: int = 63, value_window: int = 252):
        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.value_window = value_window
        self.base_leverage = 1.0
        self.max_leverage = 3.0

    def generate_signals(
        self,
        prices: pd.DataFrame,
        sentiment: SentimentProfile,
        macro_environment: MacroEnvironment,
    ) -> pd.DataFrame:
        close = prices["close"]
        returns = close.pct_change().fillna(0.0)
        momentum = close.pct_change(self.momentum_window).fillna(0.0)
        volatility = returns.rolling(self.volatility_window).std().bfill().fillna(0.0)
        value_ratio = close / close.rolling(self.value_window).mean()
        value_signal = (value_ratio - 1.0).fillna(0.0)

        macro_bias = sentiment.macro * (0.5 + 0.5 * sentiment.macro_confidence)
        industry_bias = sentiment.industry * (0.5 + 0.5 * sentiment.industry_confidence)
        macro_cycle_bias = (
            0.45 * macro_environment.growth_score
            - 0.35 * macro_environment.inflation_score
            + 0.25 * macro_environment.risk_appetite
            - 0.2 * macro_environment.stagflation_risk
        )
        raw_signal = (
            0.6 * momentum
            - 0.3 * volatility
            + 0.2 * value_signal
            + 0.4 * macro_bias
            + 0.3 * industry_bias
            + 0.35 * macro_cycle_bias
        )
        target_weight = np.tanh(raw_signal)
        news_confidence = max(sentiment.macro_confidence, sentiment.industry_confidence)
        macro_risk_adjustment = 0.5 * macro_environment.risk_appetite - 0.4 * macro_environment.stagflation_risk
        leverage_signal = self.base_leverage + news_confidence + macro_risk_adjustment
        inflation_penalty = 0.3 * max(macro_environment.inflation_score, 0.0)
        target_leverage = np.clip(leverage_signal - inflation_penalty, 1.0, self.max_leverage)

        return pd.DataFrame(
            {
                "target_weight": target_weight,
                "target_leverage": target_leverage,
                "momentum": momentum,
                "volatility": volatility,
                "value": value_signal,
            },
            index=prices.index,
        )


class Portfolio:
    """Tracks positions, cash, and portfolio level risk metrics."""

    def __init__(self, cash: float = 1_000_000.0):
        self.cash = cash
        self.positions: Dict[str, Position] = {}

    def update_position(self, symbol: str, target_quantity: float, price: float, leverage: float) -> None:
        current = self.positions.get(symbol)
        current_qty = current.quantity if current else 0.0
        trade_qty = target_quantity - current_qty
        trade_value = trade_qty * price
        self.cash -= trade_value

        if math.isclose(target_quantity, 0.0, abs_tol=1e-9):
            if symbol in self.positions:
                del self.positions[symbol]
            return

        if current and math.copysign(1, current_qty or 1.0) == math.copysign(1, target_quantity):
            incremental_notional = abs(trade_qty * price)
            existing_notional = abs(current.avg_price * current_qty)
            total_notional = incremental_notional + existing_notional
            avg_price = total_notional / abs(target_quantity) if abs(target_quantity) > 0 else price
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

    def apply_leverage_cost(self, borrowing_rate: float, days: int, price_lookup: Dict[str, float]) -> float:
        if days <= 0:
            return 0.0
        equity = self.total_equity(price_lookup)
        exposure = self.total_exposure(price_lookup)
        debt = max(exposure - equity, 0.0)
        interest = debt * borrowing_rate * (days / 365.0)
        if interest > 0:
            self.cash -= interest
        return interest

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
            "positions": positions_summary,
        }

    @staticmethod
    def _require_price(price_lookup: Dict[str, float], symbol: str) -> float:
        if symbol not in price_lookup:
            raise KeyError(f"Missing price for {symbol}")
        return price_lookup[symbol]


class RiskManager:
    """Ensures leverage, concentration, and stop-loss rules are respected."""

    def __init__(self, max_leverage: float = 3.0, position_limit: float = 0.2, stop_loss: float = 0.1, borrowing_rate: float = 0.05):
        self.max_leverage = max_leverage
        self.position_limit = position_limit
        self.stop_loss = stop_loss
        self.borrowing_rate = borrowing_rate

    def evaluate_stop_losses(self, portfolio: Portfolio, price_lookup: Dict[str, float]) -> Dict[str, float]:
        adjustments: Dict[str, float] = {}
        for symbol, position in portfolio.positions.items():
            price = price_lookup.get(symbol)
            if price is None:
                continue
            pnl = (price - position.avg_price) * position.quantity
            basis = abs(position.avg_price * position.quantity) + 1e-9
            if pnl / basis <= -self.stop_loss:
                logging.info("Stop loss triggered for %s", symbol)
                adjustments[symbol] = 0.0
        return adjustments

    def size_position(self, portfolio: Portfolio, price_lookup: Dict[str, float], symbol: str, target_weight: float, target_leverage: float) -> float:
        price = price_lookup[symbol]
        equity = portfolio.total_equity(price_lookup)
        if math.isclose(equity, 0.0):
            return 0.0
        max_weight = self.position_limit
        adjusted_weight = float(np.clip(target_weight, -max_weight, max_weight))
        leverage = min(target_leverage, self.max_leverage)
        desired_notional = equity * adjusted_weight * leverage
        max_notional = equity * self.max_leverage * self.position_limit
        desired_notional = float(np.clip(desired_notional, -max_notional, max_notional))
        return desired_notional / price


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


class TradingBot:
    """Coordinates data, strategy, and portfolio management."""

    def __init__(
        self,
        data_fetcher: Optional[DataFetcher] = None,
        news_analyzer: Optional[NewsAnalyzer] = None,
        macro_analyzer: Optional[MacroAnalyzer] = None,
        strategy: Optional[QuantStrategy] = None,
        risk_manager: Optional[RiskManager] = None,
        initial_cash: float = 1_000_000.0,
    ) -> None:
        self.data_fetcher = data_fetcher or DataFetcher()
        self.news_analyzer = news_analyzer or NewsAnalyzer()
        self.macro_analyzer = macro_analyzer or MacroAnalyzer()
        self.strategy = strategy or QuantStrategy()
        self.risk_manager = risk_manager or RiskManager(max_leverage=self.strategy.max_leverage)
        self.portfolio = Portfolio(cash=initial_cash)
        self.latest_prices: Dict[str, float] = {}
        self.latest_macro_environment: Optional[MacroEnvironment] = None

    def trade(self, symbol: str, start: dt.date, end: dt.date) -> TradeDecision:
        prices = self.data_fetcher.fetch_price_data(symbol, start, end)
        news = self.data_fetcher.fetch_news(symbol)
        sentiment = self.news_analyzer.analyze(news)
        macro_snapshot = self.data_fetcher.fetch_macro_snapshot(end)
        macro_environment = self.macro_analyzer.evaluate(macro_snapshot)
        self.latest_macro_environment = macro_environment
        signals = self.strategy.generate_signals(prices, sentiment, macro_environment)
        latest_row = signals.iloc[-1]
        price = float(prices["close"].iloc[-1])
        self.latest_prices[symbol] = price

        for stop_symbol, quantity in self.risk_manager.evaluate_stop_losses(self.portfolio, self.latest_prices).items():
            stop_price = self.latest_prices[stop_symbol]
            self.portfolio.update_position(stop_symbol, quantity, stop_price, leverage=1.0)

        quantity = self.risk_manager.size_position(
            portfolio=self.portfolio,
            price_lookup=self.latest_prices,
            symbol=symbol,
            target_weight=float(latest_row["target_weight"]),
            target_leverage=float(latest_row["target_leverage"]),
        )
        self.portfolio.update_position(symbol, quantity, price, leverage=float(latest_row["target_leverage"]))
        holding_period_days = max((prices.index[-1] - prices.index[0]).days, 1)
        leverage_cost = self.portfolio.apply_leverage_cost(self.risk_manager.borrowing_rate, holding_period_days, self.latest_prices)

        return TradeDecision(
            symbol=symbol,
            target_weight=float(latest_row["target_weight"]),
            target_leverage=float(latest_row["target_leverage"]),
            executed_quantity=quantity,
            execution_price=price,
            leverage_cost=leverage_cost,
            sentiment=sentiment,
            macro_environment=macro_environment,
        )

    def summary(self) -> Dict[str, object]:
        return self.portfolio.summary(self.latest_prices)


def format_summary(summary: Dict[str, object]) -> str:
    lines = [
        "Portfolio Summary:",
        f"  Cash: ${summary['cash']:.2f}",
        f"  Equity: ${summary['equity']:.2f}",
        f"  Gross Exposure: ${summary['gross_exposure']:.2f}",
        f"  Leverage: {summary['leverage']:.2f}x",
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


def main() -> None:
    today = dt.date.today()
    start = today - dt.timedelta(days=365)
    symbols = ["AAPL", "TSLA", "XLF"]
    bot = TradingBot()
    decisions = []
    for symbol in symbols:
        logging.info("Evaluating trades for %s", symbol)
        decisions.append(bot.trade(symbol, start, today))

    for decision in decisions:
        logging.info(
            "Trade decision for %s | Weight=%.2f | Leverage=%.2f | Quantity=%.2f | Price=$%.2f | Leverage cost=$%.2f",
            decision.symbol,
            decision.target_weight,
            decision.target_leverage,
            decision.executed_quantity,
            decision.execution_price,
            decision.leverage_cost,
        )
        logging.info(
            "Sentiment - Macro: %.2f (confidence %.2f), Industry: %.2f (confidence %.2f)",
            decision.sentiment.macro,
            decision.sentiment.macro_confidence,
            decision.sentiment.industry,
            decision.sentiment.industry_confidence,
        )
        descriptors = ", ".join(decision.macro_environment.descriptors) if decision.macro_environment else ""
        logging.info(
            "Macro regime - Growth: %.2f | Inflation: %.2f | Policy Tightening Prob: %.2f | Risk Appetite: %.2f | Stagflation Risk: %.2f %s",
            decision.macro_environment.growth_score if decision.macro_environment else float("nan"),
            decision.macro_environment.inflation_score if decision.macro_environment else float("nan"),
            decision.macro_environment.policy_tightening_probability if decision.macro_environment else float("nan"),
            decision.macro_environment.risk_appetite if decision.macro_environment else float("nan"),
            decision.macro_environment.stagflation_risk if decision.macro_environment else float("nan"),
            f"| Tags: {descriptors}" if descriptors else "",
        )

    summary = bot.summary()
    print(format_summary(summary))


if __name__ == "__main__":
    main()
