# HieraticAI Scripts

Helper scripts for maintenance and verification.

## Available Scripts

### `run_tests.sh`
Run the full test suite with pytest (or unittest fallback).

**Usage:**
```bash
./scripts/run_tests.sh
```

---

### `verify_fixes.sh`
Quick verification that all critical fixes are in place.

**Usage:**
```bash
./scripts/verify_fixes.sh
```

**Tests:**
- [PASS] DatasetValidator import
- [PASS] Training script syntax
- [PASS] Dataset validation functionality
- [PASS] Category remapping detection

---

### `cleanup_and_improve.sh`
Cleanup cache files and check for improvements.

**Usage:**
```bash
./scripts/cleanup_and_improve.sh
```

**Actions:**
- Removes Python cache files (`__pycache__/`, `*.pyc`)
- Cleans macOS metadata (`.DS_Store`, `._*`)
- Checks for large files
- Reports git status
- Suggests improvements

---

## Running Scripts

All scripts can be run from any directory in the project:

```bash
# From project root
./scripts/verify_fixes.sh

# From any subdirectory
../scripts/verify_fixes.sh

# Or with full path
/path/to/HieraticAI/scripts/verify_fixes.sh
```

The scripts automatically detect the project root and operate correctly.
