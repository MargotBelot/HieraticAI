#!/bin/bash
# Quick Test Runner for HieraticAI Pipeline
# Runs comprehensive unit tests with coverage reporting

set -e

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}HieraticAI Test Suite${NC}"
echo "========================================"

# Detect project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

cd "$PROJECT_ROOT"

echo -e "${BLUE}Project root: $PROJECT_ROOT${NC}"

# Check if pytest is available
if command -v pytest &> /dev/null; then
    echo -e "${GREEN}[OK]${NC} pytest found"
    
    # Run tests with pytest
    echo ""
    echo -e "${BLUE}Running tests with pytest...${NC}"
    pytest tests/ -v --cov=utils --cov=tools --cov-report=term-missing --cov-report=html -m "not gpu" || true
    
    echo ""
    echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
    
elif python3 -m pytest --version &> /dev/null; then
    echo -e "${GREEN}[OK]${NC} pytest found (via python3 -m pytest)"
    
    # Run tests with python3 -m pytest
    echo ""
    echo -e "${BLUE}Running tests with python3 -m pytest...${NC}"
    python3 -m pytest tests/ -v --cov=utils --cov=tools --cov-report=term-missing --cov-report=html -m "not gpu" || true
    
    echo ""
    echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
    
else
    echo -e "${YELLOW}[WARNING]${NC} pytest not found, using unittest runner"
    
    # Run tests with unittest
    echo ""
    echo -e "${BLUE}Running dataset validation tests...${NC}"
    python3 tests/test_dataset_validation.py || true
    
    echo ""
    echo -e "${BLUE}Running training pipeline tests...${NC}"
    python3 tests/test_training.py || true
fi

echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}Test run complete!${NC}"
echo ""
echo "To install pytest for better reporting:"
echo "  pip install pytest pytest-cov"
echo ""
echo "To run specific tests:"
echo "  pytest tests/test_dataset_validation.py"
echo "  pytest tests/test_training.py -k category"
echo ""
echo "For more info, see: docs/TECHNICAL_GUIDE.md"
