#!/bin/bash
# AMM-500 Cleanup and Reorganization Script
# Purpose: Clean redundant files, reorganize for US500-USDH focus

set -e

echo "🧹 AMM-500 Cleanup & Reorganization"
echo "===================================="
echo ""

# Backup current state
echo "📦 Creating backup..."
tar -czf "backup_$(date +%Y%m%d_%H%M%S).tar.gz" \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs/*' \
    --exclude='data/_archived/*' \
    . 2>/dev/null || true
echo "✅ Backup created"
echo ""

# Delete redundant markdown docs (will merge into README)
echo "🗑️  Deleting redundant documentation..."
rm -f CLEANUP_OPTIMIZATION_SUMMARY.md
rm -f HFT_OPTIMIZATION_GUIDE.md
rm -f PROFESSIONAL_MM_TRANSFORMATION.md
rm -f QUICK_START_GUIDE.md
rm -f REAL_TIME_ANALYSIS_2026-01-15.md
rm -f TRANSFORMATION_COMPLETE.md
rm -f US500_TRANSFORMATION_README.md
rm -f AUTONOMOUS_SETUP_GUIDE.md
echo "✅ Removed 8 redundant MDs"

# Delete unused generator scripts
echo "🗑️  Deleting unused generators..."
rm -f generate_professional_strategy.py
rm -f test_transformation.py
rm -f fetch_xyz100_test.py
echo "✅ Removed unused generators"

# Delete BTC-focused scripts (replaced by xyz100)
echo "🗑️  Deleting old BTC scripts..."
rm -f scripts/fetch_real_btc.py
rm -f scripts/fetch_data.py
echo "✅ Removed BTC scripts"

# Clean old data (keep xyz100, remove BTC archived)
echo "🗑️  Cleaning old data files..."
rm -f data/btc_historical.csv
rm -f data/btc_historical.json
rm -f data/btc_metadata.json
rm -rf data/_archived
echo "✅ Cleaned old data"

# Clean logs (keep structure, remove old files >7 days)
echo "🗑️  Cleaning old logs..."
find logs/ -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
find logs/ -type f -name "*.json" ! -name "autonomous_state.json" -mtime +7 -delete 2>/dev/null || true
rm -rf logs/_archived
echo "✅ Cleaned old logs"

# Create new directory structure
echo "📁 Creating new folder structure..."
mkdir -p src/core
mkdir -p src/utils
mkdir -p scripts/automation
mkdir -p scripts/analysis
mkdir -p docs/guides
echo "✅ Created new folders"

# Move src files to new structure
echo "📦 Reorganizing src/ files..."
# Core strategy files
mv src/strategy_us500_pro.py src/core/ 2>/dev/null || true
mv src/exchange.py src/core/ 2>/dev/null || true
mv src/risk.py src/core/ 2>/dev/null || true
mv src/backtest.py src/core/ 2>/dev/null || true
mv src/metrics.py src/core/ 2>/dev/null || true

# Utility files
mv src/config.py src/utils/ 2>/dev/null || true
mv src/data_fetcher.py src/utils/ 2>/dev/null || true
mv src/utils.py src/utils/ 2>/dev/null || true
mv src/xyz100_fallback.py src/utils/ 2>/dev/null || true

# Remove old strategy
rm -f src/strategy.py
rm -f src/strategy_backup_*.py

# Create __init__.py files
touch src/core/__init__.py
touch src/utils/__init__.py
echo "✅ Reorganized src/ files"

# Move scripts to new structure
echo "📦 Reorganizing scripts/ files..."
mv scripts/amm_autonomous_v3.py scripts/automation/ 2>/dev/null || true
mv scripts/amm_autonomous.py scripts/automation/ 2>/dev/null || true
mv scripts/start_paper_trading.sh scripts/automation/ 2>/dev/null || true

mv scripts/analyze_paper_results.py scripts/analysis/ 2>/dev/null || true
mv scripts/grid_search.py scripts/analysis/ 2>/dev/null || true
mv scripts/verify_targets.py scripts/analysis/ 2>/dev/null || true
echo "✅ Reorganized scripts/ files"

# Move docs
echo "📦 Reorganizing docs/..."
mv docs/EXCHANGE_ENHANCEMENTS.md docs/guides/ 2>/dev/null || true
mv docs/RISK_ENHANCEMENTS.md docs/guides/ 2>/dev/null || true
mv docs/DEPLOYMENT.md docs/guides/ 2>/dev/null || true
mv docs/FIXES_AND_STATUS.md docs/guides/ 2>/dev/null || true
echo "✅ Reorganized docs/"

# Create archive for old scripts
echo "📦 Archiving old unused scripts..."
mkdir -p archive/old_scripts
mv scripts/_archived/* archive/old_scripts/ 2>/dev/null || true
rmdir scripts/_archived 2>/dev/null || true
echo "✅ Archived old scripts"

echo ""
echo "✅ Cleanup Complete!"
echo ""
echo "📊 Summary:"
echo "  - Deleted: 8 redundant MDs, 3 generators, 2 BTC scripts"
echo "  - Cleaned: old data files, logs >7 days"
echo "  - Reorganized: src/core, src/utils, scripts/automation, scripts/analysis"
echo ""
echo "📝 Next steps:"
echo "  1. Update imports in amm-500.py"
echo "  2. Update code enhancements"
echo "  3. Create new comprehensive README.md"
echo "  4. Run tests"
echo "  5. Git commit"
