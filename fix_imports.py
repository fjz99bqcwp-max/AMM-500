#!/usr/bin/env python3
"""
Fix imports after reorganization for AMM-500
Updates all imports to match new structure:
- src/core/ - strategy, exchange, risk, backtest, metrics
- src/utils/ - config, data_fetcher, utils, xyz100_fallback
"""

import os
import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    r'from src\.config import': 'from src.utils.config import',
    r'from src\.exchange import': 'from src.core.exchange import',
    r'from src\.risk import': 'from src.core.risk import',
    r'from src\.strategy import': 'from src.core.strategy_us500_pro import',
    r'from src\.backtest import': 'from src.core.backtest import',
    r'from src\.metrics import': 'from src.core.metrics import',
    r'from src\.data_fetcher import': 'from src.utils.data_fetcher import',
    r'from src\.utils import': 'from src.utils.utils import',
    r'from src\.xyz100_fallback import': 'from src.utils.xyz100_fallback import',
    r'from \.config import': 'from src.utils.config import',
    r'from \.exchange import': 'from src.core.exchange import',
    r'from \.risk import': 'from src.core.risk import',
    r'from \.strategy import': 'from src.core.strategy_us500_pro import',
    r'from \.backtest import': 'from src.core.backtest import',
    r'from \.metrics import': 'from src.core.metrics import',
    r'from \.data_fetcher import': 'from src.utils.data_fetcher import',
    r'from \.utils import': 'from src.utils.utils import',
    r'from \.xyz100_fallback import': 'from src.utils.xyz100_fallback import',
}

def fix_imports_in_file(file_path: Path) -> int:
    """Fix imports in a single file. Returns number of changes made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = 0
        
        for old_pattern, new_import in IMPORT_MAPPINGS.items():
            matches = re.findall(old_pattern, content)
            if matches:
                content = re.sub(old_pattern, new_import, content)
                changes += len(matches)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed {changes} imports in {file_path}")
            return changes
        
        return 0
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return 0

def main():
    """Fix all imports in Python files."""
    print("🔧 Fixing imports after reorganization...")
    print("=" * 60)
    
    total_changes = 0
    files_modified = 0
    
    # Directories to process
    directories = [
        Path("src/core"),
        Path("src/utils"),
        Path("scripts/automation"),
        Path("scripts/analysis"),
        Path("tests"),
        Path("."),  # Root level (amm-500.py)
    ]
    
    for directory in directories:
        if not directory.exists():
            continue
            
        # Find all Python files
        if directory == Path("."):
            python_files = [f for f in directory.glob("*.py") if f.is_file()]
        else:
            python_files = list(directory.rglob("*.py"))
        
        for py_file in python_files:
            if "__pycache__" in str(py_file) or ".venv" in str(py_file):
                continue
            
            changes = fix_imports_in_file(py_file)
            if changes > 0:
                total_changes += changes
                files_modified += 1
    
    print("=" * 60)
    print(f"✅ Complete! Fixed {total_changes} imports in {files_modified} files")

if __name__ == "__main__":
    main()
