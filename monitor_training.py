#!/usr/bin/env python3
"""
Real-time training monitor for the text2sign project.
This script monitors training progress and provides live updates.
"""

import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
import numpy as np


class TrainingMonitor:
    """Monitor training progress in real-time."""
    
    def __init__(self, checkpoint_dir: str, refresh_interval: int = 30):
        """
        Initialize the training monitor.
        
        Args:
            checkpoint_dir: Directory containing checkpoints and logs
            refresh_interval: How often to refresh data (seconds)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.refresh_interval = refresh_interval
        self.loss_history = []
        self.lr_history = []
        self.timestamps = []
        self.step_history = []
        self.last_checkpoint = None
        
        # Create plots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Text2Sign Training Monitor', fontsize=16, fontweight='bold')
        
    def check_for_checkpoints(self) -> Optional[Path]:
        """Check for new checkpoints."""
        if not self.checkpoint_dir.exists():
            return None
            
        # Find the most recent checkpoint
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
            
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        
        if self.last_checkpoint != latest_checkpoint:
            self.last_checkpoint = latest_checkpoint
            return latest_checkpoint
            
        return None
    
    def load_checkpoint_data(self, checkpoint_path: Path) -> Dict:
        """Load data from checkpoint."""
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return {
                'epoch': checkpoint.get('epoch', 0),
                'step': checkpoint.get('step', 0),
                'best_loss': checkpoint.get('best_loss', float('inf')),
                'best_train_loss': checkpoint.get('best_train_loss', float('inf')),
                'loss_history': checkpoint.get('loss_history', []),
                'lr_history': checkpoint.get('lr_history', []),
            }
        except Exception as e:
            print(f"⚠️ Error loading checkpoint {checkpoint_path}: {e}")
            return {}
    
    def update_plots(self, frame):
        """Update the plots with new data."""
        # Check for new checkpoint
        new_checkpoint = self.check_for_checkpoints()
        
        if new_checkpoint:
            print(f"📊 Loading data from: {new_checkpoint.name}")
            data = self.load_checkpoint_data(new_checkpoint)
            
            if data:
                self.loss_history = data.get('loss_history', [])
                self.lr_history = data.get('lr_history', [])
                self.step_history = list(range(len(self.loss_history)))
                
                # Clear and update plots
                self.ax1.clear()
                self.ax2.clear()
                
                # Plot loss
                if self.loss_history:
                    self.ax1.plot(self.step_history, self.loss_history, 'b-', alpha=0.7, linewidth=1)
                    
                    # Add moving average
                    if len(self.loss_history) > 50:
                        window = min(50, len(self.loss_history) // 10)
                        moving_avg = np.convolve(self.loss_history, np.ones(window)/window, mode='valid')
                        avg_steps = self.step_history[window-1:]
                        self.ax1.plot(avg_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
                        self.ax1.legend()
                    
                    self.ax1.set_title(f'Training Loss (Best: {data.get("best_train_loss", "N/A"):.4f})')
                    self.ax1.set_xlabel('Step')
                    self.ax1.set_ylabel('Loss')
                    self.ax1.grid(True, alpha=0.3)
                
                # Plot learning rate
                if self.lr_history:
                    self.ax2.semilogy(self.step_history, self.lr_history, 'g-', linewidth=2)
                    self.ax2.set_title('Learning Rate Schedule')
                    self.ax2.set_xlabel('Step')
                    self.ax2.set_ylabel('Learning Rate (log scale)')
                    self.ax2.grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = f"Epoch: {data.get('epoch', 0)}, Step: {data.get('step', 0)}"
                if self.loss_history:
                    recent_loss = np.mean(self.loss_history[-10:]) if len(self.loss_history) >= 10 else self.loss_history[-1]
                    stats_text += f", Recent Loss: {recent_loss:.4f}"
                
                self.fig.suptitle(f'Text2Sign Training Monitor - {stats_text}', fontsize=14)
                
                plt.tight_layout()
        
        return self.ax1, self.ax2
    
    def start_monitoring(self):
        """Start the real-time monitoring."""
        print(f"🔍 Starting training monitor...")
        print(f"📁 Monitoring directory: {self.checkpoint_dir}")
        print(f"🔄 Refresh interval: {self.refresh_interval}s")
        print(f"💡 Close the plot window to stop monitoring")
        
        # Set up animation
        anim = animation.FuncAnimation(
            self.fig, self.update_plots, 
            interval=self.refresh_interval * 1000,  # Convert to milliseconds
            blit=False
        )
        
        plt.show()
        
        print("✅ Monitoring stopped")


def print_training_summary(checkpoint_dir: str):
    """Print a summary of training progress."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find latest checkpoint
    checkpoints = list(checkpoint_path.glob("checkpoint_*.pt"))
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    
    print(f"📊 Training Summary")
    print(f"=" * 50)
    print(f"📁 Checkpoint directory: {checkpoint_path}")
    print(f"📄 Latest checkpoint: {latest_checkpoint.name}")
    print(f"⏰ Last modified: {datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)}")
    
    # Load checkpoint data
    try:
        import torch
        data = torch.load(latest_checkpoint, map_location='cpu')
        
        print(f"\n📈 Progress:")
        print(f"   - Epoch: {data.get('epoch', 0)}")
        print(f"   - Step: {data.get('step', 0)}")
        print(f"   - Best validation loss: {data.get('best_loss', 'N/A')}")
        print(f"   - Best training loss: {data.get('best_train_loss', 'N/A')}")
        
        loss_history = data.get('loss_history', [])
        if loss_history:
            recent_losses = loss_history[-10:] if len(loss_history) >= 10 else loss_history
            print(f"   - Recent avg loss: {np.mean(recent_losses):.4f}")
            print(f"   - Loss trend: {len(loss_history)} data points")
        
        lr_history = data.get('lr_history', [])
        if lr_history:
            print(f"   - Current learning rate: {lr_history[-1]:.2e}")
        
        # Model info
        training_config = data.get('training_config', {})
        if training_config:
            print(f"\n🏗️ Model Configuration:")
            for key, value in training_config.items():
                print(f"   - {key}: {value}")
        
    except Exception as e:
        print(f"⚠️ Error reading checkpoint: {e}")
    
    # List all checkpoints
    print(f"\n💾 Available Checkpoints:")
    for checkpoint in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(checkpoint.stat().st_mtime)
        print(f"   - {checkpoint.name} ({size_mb:.1f} MB, {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    if len(checkpoints) > 5:
        print(f"   ... and {len(checkpoints) - 5} more checkpoints")


def quick_status(checkpoint_dir: str):
    """Print a quick status update."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print("❌ No training found")
        return
    
    # Check if training is active (recent checkpoint)
    checkpoints = list(checkpoint_path.glob("checkpoint_*.pt"))
    if not checkpoints:
        print("❌ No checkpoints found")
        return
    
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    last_modified = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)
    time_since = datetime.now() - last_modified
    
    if time_since < timedelta(minutes=10):
        status = "🟢 ACTIVE"
    elif time_since < timedelta(hours=1):
        status = "🟡 RECENT"
    else:
        status = "🔴 INACTIVE"
    
    print(f"{status} Training last active: {time_since} ago")
    
    # Quick stats
    try:
        import torch
        data = torch.load(latest_checkpoint, map_location='cpu')
        epoch = data.get('epoch', 0)
        step = data.get('step', 0)
        loss_history = data.get('loss_history', [])
        
        if loss_history:
            recent_loss = np.mean(loss_history[-10:]) if len(loss_history) >= 10 else loss_history[-1]
            print(f"📊 Epoch {epoch}, Step {step}, Recent Loss: {recent_loss:.4f}")
        else:
            print(f"📊 Epoch {epoch}, Step {step}")
            
    except Exception:
        print(f"📊 Latest: {latest_checkpoint.name}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Text2Sign Training Monitor")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", 
                       help="Directory containing checkpoints")
    parser.add_argument("--mode", type=str, default="monitor", 
                       choices=["monitor", "summary", "status"],
                       help="Monitoring mode")
    parser.add_argument("--refresh", type=int, default=30, 
                       help="Refresh interval in seconds (for monitor mode)")
    
    args = parser.parse_args()
    
    print("📈 Text2Sign Training Monitor")
    print("=" * 40)
    
    if args.mode == "monitor":
        try:
            monitor = TrainingMonitor(args.checkpoint_dir, args.refresh)
            monitor.start_monitoring()
        except ImportError:
            print("❌ Matplotlib required for live monitoring. Install with: pip install matplotlib")
            print("💡 Try --mode summary for text-only monitoring")
        except KeyboardInterrupt:
            print("\n✅ Monitoring stopped by user")
    elif args.mode == "summary":
        print_training_summary(args.checkpoint_dir)
    elif args.mode == "status":
        quick_status(args.checkpoint_dir)


if __name__ == "__main__":
    main()
