#!/usr/bin/env python3
"""Command-line backtester for the TradingBot."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import List

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import Backtester, TradingBot, TradingConfig  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a daily rebalanced backtest.")
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,TSLA,AMZN,GOOGL,XLF,XLK,SPY",
        help="Comma-separated tickers or a path to a text file with one symbol per line.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years to backtest if --start is not provided.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="ISO start date (YYYY-MM-DD). Overrides --years if provided.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="ISO end date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the equity curve as CSV.",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Optional JSON file with TradingConfig overrides.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use synthetic data sources instead of hitting external APIs.",
    )
    return parser.parse_args()


def _load_tickers(arg: str) -> List[str]:
    path = Path(arg)
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [token.strip() for token in arg.split(",") if token.strip()]


def _apply_config_overrides(path: str | None) -> TradingConfig:
    config = TradingConfig()
    if not path:
        return config
    data = json.loads(Path(path).read_text())
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def main() -> None:
    args = _parse_args()
    tickers = _load_tickers(args.tickers)
    end_date = dt.date.today() if args.end is None else dt.date.fromisoformat(args.end)
    if args.start is not None:
        start_date = dt.date.fromisoformat(args.start)
    else:
        start_date = end_date - dt.timedelta(days=365 * args.years)

    config = _apply_config_overrides(args.cfg)
    if args.offline:
        config.offline_mode = True
    bot = TradingBot(config=config)
    backtester = Backtester()
    result = backtester.run(bot, tickers, start_date, end_date)

    equity = result.get("equity")
    sharpe = result.get("sharpe", float("nan"))
    max_drawdown = result.get("max_drawdown", float("nan"))

    print(f"Tickers: {len(tickers)} | Period: {start_date} → {end_date}")
    if equity is not None and not equity.empty:
        print(
            f"Sharpe: {sharpe:.2f} | Max DD: {max_drawdown:.2%} | Final Equity: ${equity.iloc[-1]:,.0f}"
        )
    else:
        print("No equity series produced.")

    if args.out and equity is not None and not equity.empty:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        equity.to_csv(output_path, header=["equity"])
        print(f"Wrote equity curve to {output_path}")


if __name__ == "__main__":
    main()
