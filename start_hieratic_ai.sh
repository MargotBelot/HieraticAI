#!/bin/bash
# HieraticAI Validation Interface Launcher
# Portable script that works for any user

echo "Starting HieraticAI Validation Interface..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ -d "hieratic_env" ]; then
    echo "Activating virtual environment..."
    source "hieratic_env/bin/activate"
else
    echo "Warning: Virtual environment not found at hieratic_env/"
    echo "Continuing without virtual environment activation..."
fi

# Check if streamlit is available
if ! command -v streamlit &> /dev/null; then
    echo "Error: streamlit is not installed"
    echo "Please install it: pip install streamlit"
    exit 1
fi

# Check if the validation script exists
if [ ! -f "tools/validation/prediction_validator.py" ]; then
    echo "Error: Validation script not found at tools/validation/prediction_validator.py"
    exit 1
fi

# Launch Streamlit
echo "Launching Streamlit interface..."
echo "The interface will open in your browser automatically."
echo "Press Ctrl+C to stop the server."
echo ""

# Start Streamlit in background temporarily to get the port
streamlit run tools/validation/prediction_validator.py --server.headless=true --browser.gatherUsageStats=false &
STREAMLIT_PID=$!

# Wait a moment for Streamlit to start
sleep 3

# Open browser explicitly (macOS-compatible)
if command -v open &> /dev/null; then
    # macOS
    open http://localhost:8502
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open http://localhost:8502
elif command -v start &> /dev/null; then
    # Windows Git Bash
    start http://localhost:8502
fi

# Bring Streamlit process to foreground
wait $STREAMLIT_PID
