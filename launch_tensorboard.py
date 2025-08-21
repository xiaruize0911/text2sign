#!/usr/bin/env python3
"""
TensorBoard launcher script for the text2sign project.
This script helps you easily launch TensorBoard to monitor training progress.
"""

import subprocess
import sys
import argparse
from pathlib import Path
from typing import Optional
import webbrowser
import time
import socket

def find_free_port():
    """Find a free port for TensorBoard."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def launch_tensorboard(log_dir: str, port: Optional[int] = None, auto_open: bool = True):
    """Launch TensorBoard with the specified log directory."""
    
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"❌ Log directory does not exist: {log_dir}")
        print("💡 Make sure you've started training to generate logs.")
        return
    
    if port is None:
        port = find_free_port()
    
    print(f"🚀 Launching TensorBoard...")
    print(f"📊 Log directory: {log_path.absolute()}")
    print(f"🌐 Port: {port}")
    
    try:
        # Launch TensorBoard
        cmd = ["tensorboard", "--logdir", str(log_path), "--port", str(port), "--host", "localhost"]
        
        print(f"🔧 Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        
        # Wait a moment for TensorBoard to start
        time.sleep(3)
        
        url = f"http://localhost:{port}"
        print(f"✅ TensorBoard is running at: {url}")
        
        if auto_open:
            print("🌐 Opening browser...")
            webbrowser.open(url)
        
        print("\n📊 TensorBoard Features Available:")
        print("   🔹 SCALARS: Loss curves, learning rate, gradient norms")
        print("   🔹 GRAPHS: Neural network architecture visualization")
        print("   🔹 IMAGES: Training and validation video samples")
        print("   🔹 TEXT: Associated text prompts")
        print("   🔹 VIDEOS: Generated sign language clips")
        
        print(f"\n⚡ To stop TensorBoard, press Ctrl+C")
        print(f"🔗 Direct link: {url}")
        
        # Keep the script running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping TensorBoard...")
            process.terminate()
            
    except FileNotFoundError:
        print("❌ TensorBoard not found. Please install it:")
        print("   pip install tensorboard")
    except Exception as e:
        print(f"❌ Error launching TensorBoard: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Launch TensorBoard for text2sign project")
    parser.add_argument(
        "--logdir", 
        type=str, 
        default="./checkpoints/tensorboard_logs",
        help="Path to TensorBoard logs directory"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None,
        help="Port for TensorBoard (default: auto-detect free port)"
    )
    parser.add_argument(
        "--no-browser", 
        action="store_true",
        help="Don't automatically open browser"
    )
    
    args = parser.parse_args()
    
    print("🎯 Text2Sign TensorBoard Launcher")
    print("=" * 40)
    
    launch_tensorboard(
        log_dir=args.logdir,
        port=args.port,
        auto_open=not args.no_browser
    )

if __name__ == "__main__":
    main()
