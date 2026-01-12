# Logs Directory

This directory contains trading logs for the AMM-500 US500 trading bot.

## Log Files

- `bot_YYYY-MM-DD.log` - Daily bot operation logs
- `trades_YYYY-MM-DD.log` - Trade-specific logs (fills, orders)
- `monitor.log` - Continuous monitoring system logs
- `paper_trading.log` - Paper trading simulation logs

## Log Retention

Logs are automatically rotated daily and retained for 30 days by default.
Configure with `LOG_RETENTION_DAYS` in your `.env` file.

## Viewing Logs

```bash
# Watch live bot logs
tail -f logs/bot_$(date +%Y-%m-%d).log

# Watch trade logs
tail -f logs/trades_$(date +%Y-%m-%d).log

# Search for errors
grep -i error logs/bot_$(date +%Y-%m-%d).log
```
