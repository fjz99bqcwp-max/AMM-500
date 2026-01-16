#!/usr/bin/env python3
"""
Verify Trading Targets
======================
Validates backtest results meet professional targets:
- Sharpe > 2.5
- ROI > 5%
- Max DD < 0.5%
- Trades/day > 2000
- Maker ratio > 90%

Usage:
    python scripts/verify_targets.py
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.data_fetcher import DataFetcher
from src.core.backtest import BacktestEngine, BacktestConfig


@dataclass
class Targets:
    """Performance targets."""
    min_sharpe: float = 2.5
    min_roi: float = 0.05
    max_dd: float = 0.005
    min_trades_day: int = 2000
    min_maker_ratio: float = 0.90


def check_targets(result: Dict[str, Any], targets: Targets) -> Dict[str, Any]:
    """Check if result meets targets."""
    checks = {
        "sharpe": {
            "value": result.sharpe_ratio,
            "target": f"> {targets.min_sharpe}",
            "passed": result.sharpe_ratio > targets.min_sharpe,
        },
        "roi": {
            "value": f"{result.total_return:.2%}",
            "target": f"> {targets.min_roi:.1%}",
            "passed": result.total_return > targets.min_roi,
        },
        "max_dd": {
            "value": f"{result.max_drawdown:.2%}",
            "target": f"< {targets.max_dd:.2%}",
            "passed": result.max_drawdown < targets.max_dd,
        },
        "trades_day": {
            "value": result.trades_per_day,
            "target": f"> {targets.min_trades_day}",
            "passed": result.trades_per_day > targets.min_trades_day,
        },
        "maker_ratio": {
            "value": f"{result.maker_ratio:.1%}",
            "target": f"> {targets.min_maker_ratio:.0%}",
            "passed": result.maker_ratio > targets.min_maker_ratio,
        },
    }
    
    return checks


def print_report(checks: Dict[str, Any]):
    """Print verification report."""
    print("\n" + "=" * 70)
    print("TARGET VERIFICATION REPORT")
    print("=" * 70)
    print(f"{'Metric':<15} {'Value':>15} {'Target':>15} {'Status':>15}")
    print("-" * 70)
    
    all_passed = True
    for metric, data in checks.items():
        status = "✅ PASS" if data["passed"] else "❌ FAIL"
        if not data["passed"]:
            all_passed = False
        print(f"{metric:<15} {str(data['value']):>15} {data['target']:>15} {status:>15}")
    
    print("-" * 70)
    if all_passed:
        print("🎯 ALL TARGETS MET - Ready for production!")
    else:
        print("⚠️  SOME TARGETS NOT MET - Review configuration")
    print("=" * 70)
    
    return all_passed


async def verify_targets():
    """Run verification against targets."""
    logger.info("Running target verification...")
    
    # Load config and data
    config = load_config()
    fetcher = DataFetcher(config)
    
    data = fetcher.load_cached()
    if data is None:
        logger.info("Fetching data...")
        data = await fetcher.fetch(30)
    
    # Run backtest with current config
    bt_config = BacktestConfig(
        initial_capital=config.trading.collateral,
        leverage=config.trading.leverage,
    )
    
    engine = BacktestEngine(bt_config)
    result = await engine.run(data)
    
    # Check against targets
    targets = Targets()
    checks = check_targets(result, targets)
    all_passed = print_report(checks)
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "leverage": config.trading.leverage,
            "min_spread_bps": config.trading.min_spread_bps,
            "max_spread_bps": config.trading.max_spread_bps,
            "order_levels": config.trading.order_levels,
        },
        "results": {
            "sharpe": result.sharpe_ratio,
            "roi": result.total_return,
            "max_dd": result.max_drawdown,
            "trades_day": result.trades_per_day,
            "maker_ratio": result.maker_ratio,
            "final_equity": result.final_equity,
        },
        "checks": {k: v["passed"] for k, v in checks.items()},
        "all_passed": all_passed,
    }
    
    report_path = Path("logs/verification_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {report_path}")
    
    return all_passed


if __name__ == "__main__":
    passed = asyncio.run(verify_targets())
    exit(0 if passed else 1)
