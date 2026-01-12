"""
Exponential Moving Average (EMA) for model weights.

EMA maintains a shadow copy of model weights that are updated with a moving average,
which helps produce more stable and higher-quality samples during inference.
"""

import torch
import torch.nn as nn
from typing import Optional
import copy
import logging

logger = logging.getLogger(__name__)


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Args:
        model: The model whose parameters to track
        decay: EMA decay rate (higher = more smoothing, typical: 0.9999)
        update_every: Update EMA every N steps (to reduce overhead)
        device: Device to store EMA weights on
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_every: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.device = device or next(model.parameters()).device
        self.step_counter = 0
        
        # Create shadow parameters (deep copy of model state)
        self.shadow = {}
        self.backup = {}
        
        self._initialize_shadow()
        logger.info(f"EMA initialized with decay={decay}, update_every={update_every}")
    
    def _initialize_shadow(self):
        """Initialize shadow parameters from model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
    
    @torch.no_grad()
    def update(self):
        """Update EMA parameters (call after each optimizer step)."""
        self.step_counter += 1
        
        # Only update every N steps to reduce overhead
        if self.step_counter % self.update_every != 0:
            return
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA update: shadow = decay * shadow + (1 - decay) * param
                self.shadow[name].mul_(self.decay).add_(
                    param.data.to(self.device), alpha=1.0 - self.decay
                )
    
    def apply_shadow(self):
        """Apply EMA weights to model (for inference/sampling)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights after using EMA for inference."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        """Return EMA state for checkpointing."""
        return {
            'shadow': {k: v.cpu() for k, v in self.shadow.items()},
            'step_counter': self.step_counter,
            'decay': self.decay,
            'update_every': self.update_every,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict.get('decay', self.decay)
        self.update_every = state_dict.get('update_every', self.update_every)
        self.step_counter = state_dict.get('step_counter', 0)
        
        shadow_dict = state_dict.get('shadow', {})
        for name, param in shadow_dict.items():
            if name in self.shadow:
                self.shadow[name] = param.to(self.device)
