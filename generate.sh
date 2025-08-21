#!/bin/bash

# Quick inference script
# This script provides an easy way to generate videos from text prompts

# Default values
CHECKPOINT="./checkpoints/best_model.pt"
OUTPUT_DIR="./outputs"
NUM_FRAMES=16
FRAME_SIZE=64
STEPS=50
GUIDANCE_SCALE=7.5

# Default prompts
PROMPTS=("Hello" "How are you?" "Thank you" "Please" "Sorry" "Yes" "No")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --prompts)
            # Read prompts until next flag or end
            PROMPTS=()
            shift
            while [[ $# -gt 0 ]] && [[ $1 != --* ]]; do
                PROMPTS+=("$1")
                shift
            done
            ;;
        --frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --size)
            FRAME_SIZE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --guidance)
            GUIDANCE_SCALE="$2"
            shift 2
            ;;
        --fast)
            # Fast generation settings
            NUM_FRAMES=8
            FRAME_SIZE=32
            STEPS=20
            shift
            ;;
        --quality)
            # High quality settings
            NUM_FRAMES=32
            FRAME_SIZE=128
            STEPS=100
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH    Model checkpoint file (default: ./checkpoints/best_model.pt)"
            echo "  --output_dir DIR     Output directory (default: ./outputs)"
            echo "  --prompts TEXT...    Text prompts to generate (default: common words)"
            echo "  --frames N           Number of frames (default: 16)"
            echo "  --size N             Frame size NxN (default: 64)"
            echo "  --steps N            Inference steps (default: 50)"
            echo "  --guidance N         Guidance scale (default: 7.5)"
            echo "  --fast               Use fast generation settings"
            echo "  --quality            Use high quality settings"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --prompts \"Hello\" \"Goodbye\""
            echo "  $0 --fast --prompts \"Quick test\""
            echo "  $0 --quality --checkpoint ./my_model.pt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file '$CHECKPOINT' does not exist!"
    echo "Please train a model first or provide a valid checkpoint with --checkpoint"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Generating videos with the following settings:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Prompts: ${PROMPTS[*]}"
echo "  Frames: $NUM_FRAMES"
echo "  Frame size: ${FRAME_SIZE}x${FRAME_SIZE}"
echo "  Inference steps: $STEPS"
echo "  Guidance scale: $GUIDANCE_SCALE"
echo ""

# Build the command
CMD="python inference.py"
CMD="$CMD --checkpoint \"$CHECKPOINT\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --num_frames $NUM_FRAMES"
CMD="$CMD --height $FRAME_SIZE"
CMD="$CMD --width $FRAME_SIZE"
CMD="$CMD --steps $STEPS"
CMD="$CMD --guidance_scale $GUIDANCE_SCALE"
CMD="$CMD --prompts"

# Add all prompts
for prompt in "${PROMPTS[@]}"; do
    CMD="$CMD \"$prompt\""
done

# Execute the command
eval $CMD

echo ""
echo "Generation complete! Check the output directory: $OUTPUT_DIR"
