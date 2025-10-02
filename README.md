# Tradingbot

A Python-based trading bot project.

## Features

- Modular design for various trading strategies
- Uses popular Python libraries for data analysis and trading
- Easily extendable for custom trading logic

## Installation

```bash
git clone https://github.com/Emad-log/Tradingbot.git
cd Tradingbot
pip install -r requirements.txt
```

## Usage

Run the main script with default settings:

```bash
python -m main
```

### Command-line quick start

The entry point exposes a flexible CLI. Some handy invocations:

```bash
# Offline single-day run with deterministic seed
python -m main --offline --seed 123 --log-level DEBUG

# Backtest three tickers offline for the past two years and persist artefacts
python -m main --offline --backtest --symbols "AAPL,TSLA,XLF" --years 2 --persist

# Backtest fixed dates, save artefacts and JSON summary to a custom directory
python -m main --offline --backtest --symbols "SPY QQQ" --start 2023-01-01 --end 2023-12-31 \
  --persist --persist-dir .out --json-out .out/summary.json

# Disable calibration and use a different initial cash balance
python -m main --offline --no-calibration --initial-cash 250000

# Switch to the neural representation model
python -m main --offline --model-type neural
```

## Requirements

See `requirements.txt` for required Python packages.

## License

Specify your preferred license in a LICENSE file.