import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from main import (
    Backtester,
    TradingBot,
    TradingConfig,
    _parse_date,
    _parse_symbols_arg,
)


def test_parse_symbols_arg_deduplicates_and_caps():
    symbols = _parse_symbols_arg(" aapl MSFT, aapl , tsla ")
    assert symbols == ["AAPL", "MSFT", "TSLA"]


def test_parse_date_invalid_raises():
    try:
        _parse_date("not-a-date")
    except SystemExit as exc:
        assert "Invalid date" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("SystemExit not raised for invalid date")


def test_backtester_uses_configured_artifact_root(tmp_path: Path):
    config = TradingConfig(
        offline_mode=True,
        persist_backtest_artifacts=True,
        backtest_artifacts_root=str(tmp_path),
        random_seed=123,
    )
    bot = TradingBot(config=config)
    start = dt.date(2024, 1, 1)
    end = dt.date(2024, 1, 31)
    results = Backtester().run(bot, ["AAPL"], start, end)
    artifact_path = results.get("artifact_path")
    assert artifact_path is not None
    assert Path(artifact_path).is_dir()
    equity_file = Path(artifact_path) / "equity.csv"
    assert equity_file.exists()
    equity = pd.read_csv(equity_file, index_col=0)
    assert not equity.empty


def test_cli_backtest_smoke(tmp_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "main",
        "--offline",
        "--backtest",
        "--symbols",
        "AAPL",
        "--years",
        "0",
        "--persist",
        "--persist-dir",
        str(tmp_path),
        "--json-out",
        str(tmp_path / "summary.json"),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "running_backtest" in result.stdout or "running_backtest" in result.stderr
    summary_path = tmp_path / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["symbols"] == ["AAPL"]
