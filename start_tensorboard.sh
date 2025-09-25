#!/bin/zsh
# Start TensorBoard for the Text2Sign project

LOGDIR="logs"
PORT=6006
HOST="localhost"

echo "Starting TensorBoard at http://$HOST:$PORT/ (logdir: $LOGDIR)"
tensorboard --logdir "$LOGDIR" --port "$PORT" --host "$HOST"