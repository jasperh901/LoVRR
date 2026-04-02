#!/bin/bash
# Start Both DeepThinker Service and LoVRR
# Usage: ./start_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "LoVRR + DeepThinker Startup Script"
echo "============================================"
echo ""

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "tmux not found. Starting services in background mode..."
    echo ""
    
    # Start DeepThinker service in background
    echo "[1/2] Starting DeepThinker service in background..."
    nohup bash start_deepthinker_service.sh > deepthinker_service.log 2>&1 &
    DEEPTHINKER_PID=$!
    echo "DeepThinker PID: $DEEPTHINKER_PID"
    echo "Log file: deepthinker_service.log"
    
    # Wait for DeepThinker service to be ready
    echo ""
    echo "Waiting for DeepThinker service to initialize (this may take 1-2 minutes)..."
    for i in {1..120}; do
        if curl -s http://127.0.0.1:8002/health > /dev/null 2>&1; then
            echo "DeepThinker service is ready!"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "Warning: DeepThinker service may not be ready yet."
            echo "Check deepthinker_service.log for details."
        fi
        sleep 2
        echo -n "."
    done
    echo ""
    
    # Start LoVRR
    echo "[2/2] Starting LoVRR..."
    bash start_lovrr.sh
    
else
    echo "Using tmux for multi-terminal management..."
    echo ""
    
    # Kill existing session if any
    tmux kill-session -t lovrr-r1 2>/dev/null || true
    
    # Create new tmux session
    tmux new-session -d -s lovrr-r1 -n deepthinker
    
    # Start DeepThinker service in first window
    tmux send-keys -t lovrr-r1:deepthinker "cd $SCRIPT_DIR && bash start_deepthinker_service.sh" C-m
    
    # Create second window for LoVRR
    tmux new-window -t lovrr-r1 -n lovrr
    
    # Wait a bit then start LoVRR
    tmux send-keys -t lovrr-r1:lovrr "echo 'Waiting for DeepThinker service to start...'; sleep 60; cd $SCRIPT_DIR && bash start_lovrr.sh" C-m
    
    echo "Started tmux session 'lovrr-r1' with two windows:"
    echo "  - deepthinker: DeepThinker service"
    echo "  - lovrr: LoVRR demo"
    echo ""
    echo "To attach to the session, run:"
    echo "  tmux attach -t lovrr-r1"
    echo ""
    echo "To switch between windows in tmux:"
    echo "  Ctrl+B then 0 (for deepthinker)"
    echo "  Ctrl+B then 1 (for lovrr)"
    echo ""
    echo "To stop all services:"
    echo "  tmux kill-session -t lovrr-r1"
    echo ""
    
    # Attach to session
    tmux attach -t lovrr-r1
fi
