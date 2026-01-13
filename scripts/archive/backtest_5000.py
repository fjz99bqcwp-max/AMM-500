#!/usr/bin/env python3
"""
Backtest with 5000 1-minute candles from Hyperliquid
Target: Profit Factor > 1.1, $0.01 per trade profit
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from loguru import logger

# Disable verbose logging
logger.remove()
logger.add(sys.stderr, level="INFO")


def fetch_hyperliquid_candles(
    symbol: str = "BTC", interval: str = "1m", num_candles: int = 5000
) -> pd.DataFrame:
    """Fetch real candles from Hyperliquid API."""
    print(f"Fetching {num_candles} {interval} candles for {symbol} from Hyperliquid...")

    url = "https://api.hyperliquid.xyz/info"

    # Calculate time range
    # 5000 1-minute candles = ~3.5 days
    end_time = int(datetime.now().timestamp() * 1000)

    # Hyperliquid returns up to 5000 candles per request
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": end_time - (num_candles * 60 * 1000),  # num_candles minutes ago
            "endTime": end_time,
        },
    }

    response = requests.post(url, json=payload, timeout=30)
    data = response.json()

    if not data:
        print("ERROR: No data returned from API")
        return pd.DataFrame()

    # Convert to DataFrame
    candles = []
    for c in data:
        candles.append(
            {
                "timestamp": c["t"],
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            }
        )

    df = pd.DataFrame(candles)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Fetched {len(df)} candles from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    return df


def simulate_market_making(candles: pd.DataFrame, config: dict) -> dict:
    """
    Simulate OPTIMIZATION #11 market making strategy.

    Key logic:
    - WIDE spreads (10-12 bps minimum) to avoid adverse selection
    - Quote BEHIND BBO by $5 to avoid toxic flow
    - Aggressive inventory skew to reduce position quickly
    - Skip quoting during high imbalance (toxic flow)
    """
    # Config
    initial_capital = config.get("initial_capital", 1000)
    leverage = config.get("leverage", 10)
    min_spread_bps = config.get("min_spread_bps", 12)  # WIDER: 12 bps minimum (was 3)
    order_size_pct = config.get("order_size_pct", 0.02)
    maker_rebate = 0.00003  # 0.003%
    taker_fee = 0.00035  # 0.035%

    # State
    equity = initial_capital
    position_size = 0.0  # in BTC
    entry_price = 0.0

    # Tracking
    trades = []
    equity_curve = [equity]
    skipped_toxic = 0

    # Simulate orderbook from candles (mid = close, spread = high-low / 10)
    for i in range(1, len(candles)):
        prev = candles.iloc[i - 1]
        curr = candles.iloc[i]

        mid_price = curr["close"]
        candle_range = curr["high"] - curr["low"]

        # Market spread from candle (assume 20% of range is typical spread)
        market_spread = max(candle_range * 0.2, mid_price * 3 / 10000)  # At least 3 bps

        # Our spread: much wider (12 bps minimum)
        our_half_spread = max(mid_price * min_spread_bps / 10000 / 2, market_spread / 2 + 5)

        # TOXIC FLOW DETECTION: Large candle range = directional move
        # Skip quoting when candle range > 0.1% (100 bps)
        toxicity = candle_range / mid_price * 10000  # in bps
        if toxicity > 30:  # >30 bps move = likely toxic
            skipped_toxic += 1
            equity_curve.append(equity)
            continue

        best_bid = mid_price - market_spread / 2
        best_ask = mid_price + market_spread / 2

        # Calculate our quotes - BEHIND BBO by at least $5
        our_bid = min(mid_price - our_half_spread, best_bid - 5)
        our_ask = max(mid_price + our_half_spread, best_ask + 5)

        # Calculate delta (inventory exposure)
        delta = position_size * mid_price / equity if equity > 0 else 0

        # AGGRESSIVE ASYMMETRIC SKEW
        # Full half-spread skew when delta reaches 10%
        delta_factor = min(abs(delta) / 0.10, 1.0)
        skew_amount = our_half_spread * delta_factor

        if delta > 0.005:  # Long - need to SELL
            # Lower ask to sell easier, move bid away to avoid buying more
            our_ask -= skew_amount
            our_bid -= skew_amount * 0.5
        elif delta < -0.005:  # Short - need to BUY
            # Raise bid to buy easier, move ask away to avoid selling more
            our_bid += skew_amount
            our_ask += skew_amount * 0.5

        # Order size
        max_position_value = equity * leverage * order_size_pct
        order_size = max_position_value / mid_price
        min_order_size = 11 / mid_price  # $11 minimum
        order_size = max(order_size, min_order_size)

        # Simulate fills based on price movement
        # CONSERVATIVE: Only fill if price CROSSES our level by at least $2
        # This simulates realistic fills (not just touching our price)
        bid_filled = curr["low"] <= (our_bid - 2) and position_size < equity * leverage / mid_price
        ask_filled = (
            curr["high"] >= (our_ask + 2) and position_size > -equity * leverage / mid_price
        )

        # Fill probability - lower for wider spreads (more conservative)
        fill_prob = min(curr["volume"] / (order_size * 20), 0.7)  # Max 70% fill prob

        # Process fills
        if bid_filled and np.random.random() < fill_prob:
            # Buy fill
            fill_price = our_bid
            fee = order_size * fill_price * maker_rebate  # Rebate (negative fee)

            # Update position
            if position_size >= 0:
                # Adding to long or opening long
                total_value = position_size * entry_price + order_size * fill_price
                position_size += order_size
                entry_price = total_value / position_size if position_size > 0 else fill_price
                realized_pnl = 0
            else:
                # Closing short
                realized_pnl = (entry_price - fill_price) * min(order_size, abs(position_size))
                position_size += order_size
                if position_size > 0:
                    entry_price = fill_price

            equity += realized_pnl + fee
            trades.append(
                {
                    "timestamp": curr["timestamp"],
                    "side": "BUY",
                    "price": fill_price,
                    "size": order_size,
                    "fee": -fee,  # Rebate is negative fee
                    "pnl": realized_pnl,
                    "net": realized_pnl + fee,
                }
            )

        if ask_filled and np.random.random() < fill_prob:
            # Sell fill
            fill_price = our_ask
            fee = order_size * fill_price * maker_rebate  # Rebate (negative fee)

            # Update position
            if position_size <= 0:
                # Adding to short or opening short
                total_value = abs(position_size) * entry_price + order_size * fill_price
                position_size -= order_size
                entry_price = total_value / abs(position_size) if position_size < 0 else fill_price
                realized_pnl = 0
            else:
                # Closing long
                realized_pnl = (fill_price - entry_price) * min(order_size, position_size)
                position_size -= order_size
                if position_size < 0:
                    entry_price = fill_price

            equity += realized_pnl + fee
            trades.append(
                {
                    "timestamp": curr["timestamp"],
                    "side": "SELL",
                    "price": fill_price,
                    "size": order_size,
                    "fee": -fee,  # Rebate is negative fee
                    "pnl": realized_pnl,
                    "net": realized_pnl + fee,
                }
            )

        # Update unrealized PnL
        if position_size != 0:
            unrealized = (mid_price - entry_price) * position_size
        else:
            unrealized = 0

        equity_curve.append(equity + unrealized)

    # Calculate metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    if len(trades_df) > 0:
        total_trades = len(trades_df)
        winning = trades_df[trades_df["net"] > 0]
        losing = trades_df[trades_df["net"] < 0]

        win_rate = len(winning) / total_trades if total_trades > 0 else 0
        avg_win = winning["net"].mean() if len(winning) > 0 else 0
        avg_loss = abs(losing["net"].mean()) if len(losing) > 0 else 0

        gross_profit = winning["pnl"].sum() if len(winning) > 0 else 0
        gross_loss = abs(losing["pnl"].sum()) if len(losing) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        net_pnl = trades_df["net"].sum()
        avg_pnl_per_trade = net_pnl / total_trades

        # Drawdown
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_drawdown = drawdown.max()

        # Sharpe
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(365 * 24 * 60)
            if np.std(returns) > 0
            else 0
        )
    else:
        total_trades = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        net_pnl = 0
        avg_pnl_per_trade = 0
        max_drawdown = 0
        sharpe = 0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "net_pnl": net_pnl,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "final_equity": equity_curve[-1] if equity_curve else initial_capital,
        "trades": trades_df,
        "equity_curve": equity_curve,
    }


def run_backtest():
    """Run backtest with 5000 candles."""
    print("=" * 70)
    print("BACKTEST: 5000 x 1-minute candles from Hyperliquid")
    print("=" * 70)

    # Fetch data
    candles = fetch_hyperliquid_candles("BTC", "1m", 5000)

    if len(candles) < 100:
        print("ERROR: Not enough candles fetched")
        return None

    # Run simulation - OPTIMIZATION #11 settings
    config = {"initial_capital": 1000, "leverage": 10, "min_spread_bps": 12, "order_size_pct": 0.02}

    print(f"\nRunning simulation with config: {config}")
    result = simulate_market_making(candles, config)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Period: {candles['datetime'].iloc[0]} to {candles['datetime'].iloc[-1]}")
    print(f"Candles: {len(candles)}")
    print(f"\n--- Performance ---")
    print(f"Total Trades: {result['total_trades']}")
    print(f"Win Rate: {result['win_rate']:.1%}")
    print(f"Avg Win: ${result['avg_win']:.4f}")
    print(f"Avg Loss: ${result['avg_loss']:.4f}")
    print(f"\n--- KEY METRICS ---")
    print(f"PROFIT FACTOR: {result['profit_factor']:.3f} (Target: >1.1)")
    if result["profit_factor"] >= 1.1:
        print("   ✅ TARGET MET!")
    else:
        print(f"   ❌ Need {1.1 - result['profit_factor']:.3f} improvement")

    print(f"\nAVG PNL PER TRADE: ${result['avg_pnl_per_trade']:.4f} (Target: $0.01)")
    if result["avg_pnl_per_trade"] >= 0.01:
        print("   ✅ TARGET MET!")
    else:
        print(f"   ❌ Need ${0.01 - result['avg_pnl_per_trade']:.4f} improvement")

    print(f"\n--- Risk ---")
    print(f"Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"\n--- PnL ---")
    print(f"Net PnL: ${result['net_pnl']:.2f}")
    print(f"Final Equity: ${result['final_equity']:.2f}")
    print(f"ROI: {(result['final_equity'] / 1000 - 1) * 100:.2f}%")
    print("=" * 70)

    return result


if __name__ == "__main__":
    result = run_backtest()
