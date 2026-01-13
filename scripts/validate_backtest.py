#!/usr/bin/env python3
"""Quick backtest validation script."""
import sys
sys.path.insert(0, '.')

from src.backtest import run_backtest, BacktestConfig

print('Testing backtest with synthetic data (7 days)...')
result = run_backtest(synthetic_days=7, use_real_data=False, plot=False, monte_carlo=False)

print('\n=== BACKTEST RESULTS ===')
print(f'Duration: {result.duration_days:.1f} days')
print(f'Total Trades: {result.total_trades}')
print(f'Trades/Day: {result.trades_per_day:.1f}')
print(f'Win Rate: {result.win_rate*100:.1f}%')
print(f'Net PnL: ${result.net_pnl:.2f}')
print(f'ROI: {result.roi_pct:.1f}%')
print(f'Sharpe Ratio: {result.sharpe_ratio:.2f}')
print(f'Max Drawdown: {result.max_drawdown*100:.2f}%')
print('========================')

# Check targets
print('\n=== TARGET VALIDATION ===')
sharpe_pass = result.sharpe_ratio > 2.0
roi_pass = result.roi_pct > 30
dd_pass = result.max_drawdown < 0.05
trades_pass = result.trades_per_day > 500

print(f'Sharpe > 2.0: {"✅" if sharpe_pass else "❌"} ({result.sharpe_ratio:.2f})')
print(f'ROI > 30%: {"✅" if roi_pass else "❌"} ({result.roi_pct:.1f}%)')
print(f'Max DD < 5%: {"✅" if dd_pass else "❌"} ({result.max_drawdown*100:.2f}%)')
print(f'Trades/Day > 500: {"✅" if trades_pass else "❌"} ({result.trades_per_day:.0f})')

all_pass = sharpe_pass and roi_pass and dd_pass and trades_pass
print(f'\nAll Targets Met: {"✅ YES" if all_pass else "❌ NO"}')
