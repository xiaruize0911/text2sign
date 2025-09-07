#!/bin/bash

"""
Enhanced TensorBoard Launch Script for Text2Sign Training
This script provides easy access to the comprehensive TensorBoard logging
"""

import sys
import subprocess
import os
from config import Config

def launch_tensorboard():
    """Launch TensorBoard with the configured log directory"""
    
    log_dir = Config.LOG_DIR
    
    print("🚀 Launching TensorBoard for Text2Sign Training")
    print(f"📁 Log Directory: {log_dir}")
    print("="*60)
    
    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"❌ Log directory '{log_dir}' does not exist!")
        print("   Please run training first to generate logs.")
        return
    
    # Print available log categories
    print("📊 Available TensorBoard Categories:")
    categories = Config.TENSORBOARD_LOG_CATEGORIES
    for i, category in enumerate(categories, 1):
        print(f"   {i:2d}. {category}")
    
    print("\n🌟 Key Features:")
    print("   • Comprehensive training metrics and loss tracking")
    print("   • Real-time video sample generation visualization")
    print("   • Detailed noise prediction analysis")
    print("   • Model architecture and parameter monitoring")
    print("   • Learning rate scheduling visualization")
    print("   • System performance and memory tracking")
    print("   • Gradient statistics and histograms")
    
    print("\n" + "="*60)
    print("🎯 TensorBoard will be available at: http://localhost:6006")
    print("   Use Ctrl+C to stop TensorBoard")
    print("="*60)
    
    try:
        # Launch TensorBoard
        cmd = ["tensorboard", "--logdir", log_dir, "--port", "6006"]
        subprocess.run(cmd)
    
    except KeyboardInterrupt:
        print("\n👋 TensorBoard stopped by user")
    
    except FileNotFoundError:
        print("❌ TensorBoard not found!")
        print("   Install with: pip install tensorboard")
    
    except Exception as e:
        print(f"❌ Error launching TensorBoard: {e}")

if __name__ == "__main__":
    launch_tensorboard()
