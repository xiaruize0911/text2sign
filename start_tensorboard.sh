#!/bin/bash
# Simple shell script to start TensorBoard for Text2Sign project

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
LOGDIR="logs"
PORT=6006
HOST="localhost"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --logdir)
            LOGDIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--logdir DIR] [--port PORT] [--host HOST]"
            echo ""
            echo "Options:"
            echo "  --logdir DIR    Log directory (default: logs)"
            echo "  --port PORT     Port number (default: 6006)"
            echo "  --host HOST     Host address (default: localhost)"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Start with defaults"
            echo "  $0 --port 6007              # Use custom port"
            echo "  $0 --logdir experiments     # Use different log directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}🚀 Starting TensorBoard for Text2Sign Diffusion Model${NC}"
echo -e "${BLUE}================================================================${NC}"

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" == "text2sign" ]]; then
    echo -e "${GREEN}✅ Conda environment 'text2sign' is active${NC}"
else
    echo -e "${YELLOW}⚠️  Activating conda environment 'text2sign'...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate text2sign
    if [[ "$CONDA_DEFAULT_ENV" == "text2sign" ]]; then
        echo -e "${GREEN}✅ Successfully activated 'text2sign' environment${NC}"
    else
        echo -e "${RED}❌ Failed to activate 'text2sign' environment${NC}"
        echo -e "${YELLOW}   Please run: conda activate text2sign${NC}"
        exit 1
    fi
fi

# Check if log directory exists
if [[ ! -d "$LOGDIR" ]]; then
    echo -e "${YELLOW}⚠️  Log directory '$LOGDIR' does not exist${NC}"
    echo -e "${BLUE}📁 Creating log directory...${NC}"
    mkdir -p "$LOGDIR"
    echo -e "${GREEN}✅ Created log directory: $LOGDIR${NC}"
else
    # Count log files
    LOG_COUNT=$(find "$LOGDIR" -name "events.out.tfevents.*" 2>/dev/null | wc -l)
    if [[ $LOG_COUNT -gt 0 ]]; then
        echo -e "${GREEN}📊 Found $LOG_COUNT TensorBoard log files in $LOGDIR${NC}"
    else
        echo -e "${YELLOW}📋 No log files found in $LOGDIR yet${NC}"
        echo -e "${BLUE}   Logs will appear when training starts${NC}"
    fi
fi

# Check if TensorBoard is installed
if ! command -v tensorboard &> /dev/null; then
    echo -e "${RED}❌ TensorBoard not found!${NC}"
    echo -e "${YELLOW}   Installing TensorBoard...${NC}"
    pip install tensorboard
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ TensorBoard installed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to install TensorBoard${NC}"
        exit 1
    fi
fi

# Start TensorBoard
URL="http://$HOST:$PORT"
echo -e "${BLUE}🔧 Starting TensorBoard...${NC}"
echo -e "${BLUE}   Command: tensorboard --logdir $LOGDIR --port $PORT --host $HOST${NC}"
echo -e "${GREEN}🌐 TensorBoard will be available at: $URL${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo -e "${GREEN}📖 TensorBoard Features:${NC}"
echo -e "   ${BLUE}•${NC} SCALARS: Training loss, learning rate curves"
echo -e "   ${BLUE}•${NC} IMAGES: Generated video samples during training"
echo -e "   ${BLUE}•${NC} GRAPHS: 3D UNet model architecture visualization"
echo -e "   ${BLUE}•${NC} HISTOGRAMS: Parameter distributions"
echo ""
echo -e "${YELLOW}⌨️  Press Ctrl+C to stop TensorBoard${NC}"
echo ""

# Try to open browser automatically (macOS/Linux)
if command -v open &> /dev/null; then
    # macOS
    (sleep 3 && open "$URL") &
elif command -v xdg-open &> /dev/null; then
    # Linux
    (sleep 3 && xdg-open "$URL") &
fi

# Start TensorBoard
tensorboard \
    --logdir "$LOGDIR" \
    --port "$PORT" \
    --host "$HOST" \
    --reload_interval 1 \
    --samples_per_plugin images=100

echo -e "\n${GREEN}✅ TensorBoard stopped${NC}"
