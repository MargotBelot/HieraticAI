# Testing Guide

Comprehensive guide for running and interpreting tests in the HieraticAI pipeline.

## Table of Contents
- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Writing Tests](#writing-tests)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Install Test Dependencies
```bash
pip install pytest pytest-cov pytest-timeout pytest-xdist
```

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_dataset_validation.py
pytest tests/test_training.py
```

### Run with Coverage Report
```bash
pytest --cov=utils --cov=tools --cov-report=html
```

## Test Structure

### Test Files
The test suite is organized into multiple test files:

```
tests/
├── test_dataset_validation.py    # Dataset validation tests
├── test_training.py               # Training pipeline tests
└── __init__.py
```

### Test Categories

#### Unit Tests
Test individual components in isolation:
- `TestDatasetValidator`: Dataset validation logic
- `TestCategoryMapping`: Category ID remapping
- `TestCategoryRemapping`: Remapping algorithms
- `TestHieroglyphTrainingConfig`: Training configuration

#### Integration Tests
Test multiple components working together:
- `TestIntegration`: Complete validation pipeline
- `TestRegressionPrevention`: Prevent known bugs

## Running Tests

### Basic Commands

#### Run all tests
```bash
pytest
```

#### Run with verbose output
```bash
pytest -v
```

#### Run specific test class
```bash
pytest tests/test_dataset_validation.py::TestDatasetValidator
```

#### Run specific test method
```bash
pytest tests/test_dataset_validation.py::TestDatasetValidator::test_file_structure_validation_success
```

#### Run tests matching pattern
```bash
pytest -k "category"  # Runs all tests with "category" in name
```

### Advanced Options

#### Run tests in parallel (faster)
```bash
pytest -n auto  # Uses all CPU cores
pytest -n 4     # Uses 4 workers
```

#### Run with timeout protection
```bash
pytest --timeout=300  # 5 minute timeout per test
```

#### Run and stop on first failure
```bash
pytest -x
```

#### Run only failed tests from last run
```bash
pytest --lf
```

#### Run with detailed output
```bash
pytest -vv --showlocals
```

### Test Markers

Tests can be categorized with markers:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests (useful for CPU-only machines)
pytest -m "not gpu"

# Skip tests that shouldn't run in CI
pytest -m "not skip_ci"
```

## Test Coverage

### Generate Coverage Report

#### Terminal Report
```bash
pytest --cov=utils --cov=tools --cov-report=term-missing
```

Output shows:
- Lines covered/missed per file
- Total coverage percentage
- Missing line numbers

#### HTML Report (recommended)
```bash
pytest --cov=utils --cov=tools --cov-report=html
```

Open `htmlcov/index.html` in browser to view:
- Interactive coverage visualization
- Line-by-line coverage highlighting
- Branch coverage details

#### XML Report (for CI)
```bash
pytest --cov=utils --cov=tools --cov-report=xml
```

Generates `coverage.xml` for tools like Codecov.

### Coverage Thresholds

The pipeline enforces minimum coverage:
- **Current threshold: 70%**
- Tests fail if coverage drops below threshold
- Configure in `pytest.ini`

### Interpreting Coverage

Coverage metrics:
- **Statement coverage**: Lines executed during tests
- **Branch coverage**: Decision points (if/else) tested
- **Function coverage**: Functions called during tests

Good coverage targets:
- Critical code (validators, remapping): **>90%**
- Business logic: **>80%**
- Utilities: **>70%**
- UI code: **>60%**

## CI/CD Pipeline

### GitHub Actions Workflow

The CI pipeline runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

### Pipeline Jobs

#### 1. Lint (Code Quality)
- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Style and syntax checking

#### 2. Test (Multi-Platform)
- Tests on Ubuntu and macOS
- Python versions: 3.8, 3.9, 3.10, 3.11
- Runs full test suite with coverage
- Uploads coverage to Codecov

#### 3. Verify (Pipeline Validation)
- Runs `scripts/verify_fixes.sh`
- Checks file structure
- Validates imports

#### 4. Security (Vulnerability Scanning)
- **Bandit**: Python security scanner
- **Safety**: Dependency vulnerability check
- Generates security reports

#### 5. Build (Package Building)
- Builds distribution packages
- Validates package structure
- Creates artifacts

#### 6. Summary (Status Report)
- Summarizes all job results
- Provides quick status overview

### Viewing CI Results

1. Go to GitHub Actions tab
2. Click on workflow run
3. View job results and logs
4. Download artifacts (coverage, security reports)

### Local CI Simulation

Run locally before pushing:

```bash
# Run linting
black --check utils/ tools/ tests/
isort --check-only utils/ tools/ tests/
flake8 utils/ tools/ tests/

# Run tests
pytest -v --cov=utils --cov=tools -m "not gpu"

# Run verification
bash scripts/verify_fixes.sh
```

## Writing Tests

### Test Structure Template

```python
import unittest
import tempfile
import shutil

class TestYourComponent(unittest.TestCase):
    """Test suite for YourComponent"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.test_dir = tempfile.mkdtemp()
        # Setup code here
    
    def tearDown(self):
        """Clean up after each test"""
        shutil.rmtree(self.test_dir)
    
    def test_basic_functionality(self):
        """Test basic functionality works"""
        # Arrange
        input_data = "test"
        
        # Act
        result = your_function(input_data)
        
        # Assert
        self.assertEqual(result, expected)
    
    def test_edge_case(self):
        """Test edge case handling"""
        with self.assertRaises(ValueError):
            your_function(invalid_input)
```

### Best Practices

#### 1. Test Naming
- Use descriptive names: `test_category_remapping_with_1_based_ids`
- Follow pattern: `test_<what>_<condition>_<expected>`

#### 2. Test Independence
- Each test should run independently
- Don't rely on test execution order
- Clean up resources in `tearDown()`

#### 3. Test Coverage
- Test happy path (normal operation)
- Test edge cases (boundaries, empty input)
- Test error cases (invalid input, exceptions)
- Test integration between components

#### 4. Assertions
```python
# Equality checks
self.assertEqual(actual, expected)
self.assertNotEqual(actual, unexpected)

# Boolean checks
self.assertTrue(condition)
self.assertFalse(condition)

# Membership checks
self.assertIn(item, collection)
self.assertNotIn(item, collection)

# Exception checks
with self.assertRaises(ValueError):
    function_that_raises()

# Numeric checks
self.assertGreater(a, b)
self.assertLess(a, b)
self.assertAlmostEqual(a, b, places=2)
```

#### 5. Mock External Dependencies
```python
from unittest.mock import patch, MagicMock

@patch('module.external_function')
def test_with_mock(self, mock_func):
    mock_func.return_value = "mocked"
    result = your_function()
    self.assertEqual(result, "expected")
```

### Test Markers

Add markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_unit_function():
    """Unit test"""
    pass

@pytest.mark.integration
def test_integration():
    """Integration test"""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Slow test that takes time"""
    pass

@pytest.mark.gpu
def test_requires_gpu():
    """Test that needs GPU"""
    pass
```

## Troubleshooting

### Common Issues

#### Import Errors
```
ModuleNotFoundError: No module named 'utils'
```

**Solution**: Ensure PYTHONPATH includes project root
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/HieraticAI"
pytest
```

Or run from project root:
```bash
cd /path/to/HieraticAI
pytest
```

#### Detectron2 Import Failures
```
ImportError: cannot import name 'DefaultTrainer' from 'detectron2.engine'
```

**Solution**: Install Detectron2
```bash
pip install torch torchvision
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Tests will skip gracefully if Detectron2 is not available.

#### PIL/Pillow Errors
```
ModuleNotFoundError: No module named 'PIL'
```

**Solution**: Install Pillow
```bash
pip install Pillow
```

#### Pytest Not Found
```
pytest: command not found
```

**Solution**: Install pytest
```bash
pip install pytest pytest-cov
```

#### Coverage Below Threshold
```
FAIL Required test coverage of 70% not reached. Total coverage: 65.42%
```

**Solution**: Add more tests or adjust threshold in `pytest.ini`

### Debug Mode

Run tests with debugging:

```bash
# With Python debugger
pytest --pdb  # Drop into debugger on failure

# With detailed output
pytest -vv --showlocals --tb=long

# Capture print statements
pytest -s  # Don't capture stdout/stderr
```

### Performance Issues

If tests are slow:

```bash
# Profile test execution time
pytest --durations=10  # Show 10 slowest tests

# Run tests in parallel
pytest -n auto

# Skip slow tests
pytest -m "not slow"
```

## CI Badge

Add CI status badge to README.md:

```markdown
![CI Status](https://github.com/YourUsername/HieraticAI/workflows/CI%20Pipeline/badge.svg)
[![codecov](https://codecov.io/gh/YourUsername/HieraticAI/branch/main/graph/badge.svg)](https://codecov.io/gh/YourUsername/HieraticAI)
```

## Continuous Improvement

### Monitoring Coverage Trends
- Track coverage over time
- Set goals for improvement
- Review coverage reports in PRs

### Adding New Tests
- Add tests for new features
- Add regression tests for bugs
- Maintain or improve coverage

### Test Maintenance
- Remove obsolete tests
- Update tests when APIs change
- Refactor duplicated test code

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
