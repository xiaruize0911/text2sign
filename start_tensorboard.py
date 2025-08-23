#!/usr/bin/env python3
"""
TensorBoard launch script for the Text2Sign diffusion model
This script starts TensorBoard to monitor training progress and visualizations
"""

import subprocess
import sys
import os
import webbrowser
import time
import argparse
from pathlib import Path

def find_log_directories():
    """Find all available log directories"""
    current_dir = Path(".")
    log_dirs = []
    
    # Look for standard log directories
    if (current_dir / "logs").exists():
        log_dirs.append("logs")
    
    if (current_dir / "runs").exists():
        log_dirs.append("runs")
    
    # Look for experiment directories
    experiments_dir = current_dir / "experiments"
    if experiments_dir.exists():
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                exp_logs = exp_dir / "logs"
                if exp_logs.exists():
                    log_dirs.append(str(exp_logs))
    
    # Look for model visualization directory
    if (current_dir / "model_visualization").exists():
        log_dirs.append("model_visualization")
    
    return log_dirs

def start_tensorboard(logdir="logs", port=6006, host="localhost", auto_open=True):
    """
    Start TensorBoard with the specified parameters
    
    Args:
        logdir (str): Directory containing TensorBoard logs
        port (int): Port to run TensorBoard on
        host (str): Host to bind TensorBoard to
        auto_open (bool): Whether to automatically open browser
    """
    print("=" * 60)
    print("🚀 Starting TensorBoard for Text2Sign Diffusion Model")
    print("=" * 60)
    
    # Check if logdir exists
    if not os.path.exists(logdir):
        print(f"⚠️  Warning: Log directory '{logdir}' does not exist.")
        print("   TensorBoard will create it when training starts.")
        
        # Create the directory
        os.makedirs(logdir, exist_ok=True)
        print(f"✅ Created log directory: {logdir}")
    
    # Check if there are any log files
    log_files = list(Path(logdir).rglob("events.out.tfevents.*"))
    if log_files:
        print(f"📊 Found {len(log_files)} TensorBoard log files")
    else:
        print("📋 No log files found yet. Logs will appear when training starts.")
    
    # Build TensorBoard command
    cmd = [
        "tensorboard",  # Use tensorboard directly instead of python -m
        "--logdir", logdir,
        "--port", str(port),
        "--host", host,
        "--reload_interval", "1",  # Reload every 1 second
        "--samples_per_plugin", "images=100"  # Show more image samples
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print(f"🌐 TensorBoard will be available at: http://{host}:{port}")
    print("=" * 60)
    
    try:
        # Start TensorBoard
        print("⏳ Starting TensorBoard...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait a moment for TensorBoard to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ TensorBoard started successfully!")
            url = f"http://{host}:{port}"
            print(f"🔗 Access TensorBoard at: {url}")
            
            # Automatically open browser if requested
            if auto_open:
                try:
                    print("🌍 Opening browser...")
                    webbrowser.open(url)
                except Exception as e:
                    print(f"⚠️  Could not open browser automatically: {e}")
                    print(f"   Please manually open: {url}")
            
            print("\n📖 TensorBoard Features:")
            print("   • SCALARS: Training loss, learning rate")
            print("   • IMAGES: Generated video samples")
            print("   • GRAPHS: Model architecture")
            print("   • HISTOGRAMS: Parameter distributions")
            print("\n⌨️  Press Ctrl+C to stop TensorBoard")
            
            # Keep the script running and monitor TensorBoard
            try:
                while True:
                    time.sleep(1)
                    if process.poll() is not None:
                        print("\n❌ TensorBoard process ended unexpectedly")
                        break
            except KeyboardInterrupt:
                print("\n🛑 Stopping TensorBoard...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                    print("✅ TensorBoard stopped successfully")
                except subprocess.TimeoutExpired:
                    print("⚠️  Force killing TensorBoard...")
                    process.kill()
                    process.wait()
                    print("✅ TensorBoard force stopped")
        else:
            # Process ended immediately, probably an error
            stdout, stderr = process.communicate()
            print("❌ Failed to start TensorBoard")
            if stderr:
                print(f"Error: {stderr}")
            if stdout:
                print(f"Output: {stdout}")
            return False
            
    except FileNotFoundError:
        print("❌ TensorBoard not found!")
        print("   Please install TensorBoard: pip install tensorboard")
        return False
    except Exception as e:
        print(f"❌ Failed to start TensorBoard: {e}")
        return False
    
    return True

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Start TensorBoard for Text2Sign diffusion model monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_tensorboard.py                    # Use default settings
  python start_tensorboard.py --port 6007        # Use custom port
  python start_tensorboard.py --logdir runs      # Use custom log directory
  python start_tensorboard.py --list-logs        # List available log directories
        """
    )
    
    parser.add_argument(
        "--logdir", 
        default="logs",
        help="Directory containing TensorBoard logs (default: logs)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=6006,
        help="Port to run TensorBoard on (default: 6006)"
    )
    
    parser.add_argument(
        "--host", 
        default="localhost",
        help="Host to bind TensorBoard to (default: localhost)"
    )
    
    parser.add_argument(
        "--no-browser", 
        action="store_true",
        help="Don't automatically open browser"
    )
    
    parser.add_argument(
        "--list-logs", 
        action="store_true",
        help="List available log directories and exit"
    )
    
    args = parser.parse_args()
    
    # List log directories if requested
    if args.list_logs:
        print("📁 Available log directories:")
        log_dirs = find_log_directories()
        if log_dirs:
            for i, log_dir in enumerate(log_dirs, 1):
                log_path = Path(log_dir)
                log_files = list(log_path.rglob("events.out.tfevents.*"))
                print(f"   {i}. {log_dir} ({len(log_files)} log files)")
        else:
            print("   No log directories found.")
            print("   Run training first: python main.py train")
        return
    
    # Start TensorBoard
    success = start_tensorboard(
        logdir=args.logdir,
        port=args.port,
        host=args.host,
        auto_open=not args.no_browser
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
