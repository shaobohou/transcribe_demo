#!/bin/bash
#
# Launch script for transcribe-demo web app
#
# Usage:
#   ./run-webapp.sh              # Run with GPU support
#   ./run-webapp.sh --cpu        # Run with CPU only
#   ./run-webapp.sh --help       # Show help

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
CPU_ONLY=false
PORT=5000
HOST="0.0.0.0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            CPU_ONLY=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu         Run in CPU-only mode (for CI/sandboxes)"
            echo "  --port PORT   Port to run on (default: 5000)"
            echo "  --host HOST   Host to bind to (default: 0.0.0.0)"
            echo "  --help        Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  ANTHROPIC_API_KEY    Required for Realtime backend"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Transcribe Demo Web App${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the right directory
if [[ ! -f "app.py" ]]; then
    echo -e "${RED}Error: app.py not found${NC}"
    echo "Please run this script from the webapp/ directory"
    exit 1
fi

# Install dependencies if needed
echo -e "${YELLOW}Checking dependencies...${NC}"
if $CPU_ONLY; then
    echo -e "${YELLOW}Running in CPU-only mode${NC}"
    cd ..
    uv sync --project ci --group webapp --refresh
    cd webapp
else
    cd ..
    uv sync --group webapp
    cd webapp
fi

echo -e "${GREEN}Dependencies installed${NC}"
echo ""

# Check for Realtime API key
if [[ -z "${ANTHROPIC_API_KEY}" ]]; then
    echo -e "${YELLOW}Warning: ANTHROPIC_API_KEY not set${NC}"
    echo "Realtime backend will not be available"
    echo "Set it with: export ANTHROPIC_API_KEY=your_key_here"
    echo ""
fi

# Start the server
echo -e "${GREEN}Starting web server...${NC}"
echo -e "  Host: ${BLUE}$HOST${NC}"
echo -e "  Port: ${BLUE}$PORT${NC}"
echo -e "  URL:  ${BLUE}http://localhost:$PORT${NC}"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop${NC}"
echo ""

if $CPU_ONLY; then
    cd ..
    uv --project ci run python webapp/app.py
else
    cd ..
    uv run python webapp/app.py
fi
