import os

file_path = 'methods/trainer.py'

with open(file_path, 'r') as f:
    content = f.read()

old_code = """            # Check for NaN/Inf in loss immediately after forward pass
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at step {self.global_step} AFTER forward pass")
                logger.error(f"  Input video range: [{videos.min().item():.3f}, {videos.max().item():.3f}]")
                logger.error(f"  Predicted noise range: [{predicted_noise.min().item():.3f}, {predicted_noise.max().item():.3f}]")
                logger.error(f"  Actual noise range: [{noise.min().item():.3f}, {noise.max().item():.3f}]")
                logger.error(f"  Has NaN in pred_noise: {torch.isnan(predicted_noise).any().item()}")
                logger.error(f"  Has NaN in noise: {torch.isnan(noise).any().item()}")
                # Skip this batch - no backward was called, so no need to update scaler
                self.optimizer.zero_grad()
                self.accumulation_step = 0
                self.global_step += 1
                continue
            
            # Scale loss by accumulation steps for correct gradients
            scaled_loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
            
            # Clamp loss to prevent extreme values before backward
            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                logger.warning(f"NaN/Inf loss detected at step {self.global_step}, skipping batch")
                self.optimizer.zero_grad()
                self.accumulation_step = 0
                self.global_step += 1
                continue
            
            # Cap loss at reasonable maximum to prevent gradient explosion
            scaled_loss = torch.clamp(scaled_loss, max=100.0)"""

new_code = """            # Check for NaN/Inf in loss immediately after forward pass
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at step {self.global_step} AFTER forward pass")
                logger.error(f"  Input video range: [{videos.min().item():.3f}, {videos.max().item():.3f}]")
                logger.error(f"  Predicted noise range: [{predicted_noise.min().item():.3f}, {predicted_noise.max().item():.3f}]")
                logger.error(f"  Actual noise range: [{noise.min().item():.3f}, {noise.max().item():.3f}]")
                logger.error(f"  Has NaN in pred_noise: {torch.isnan(predicted_noise).any().item()}")
                logger.error(f"  Has NaN in noise: {torch.isnan(noise).any().item()}")
                # Skip this batch - no backward was called, so no need to update scaler
                self.optimizer.zero_grad()
                self.accumulation_step = 0
                self.global_step += 1
                continue
            
            # Check for extreme loss values (potential explosion)
            if loss.item() > 10.0:
                logger.warning(f"Extreme loss detected at step {self.global_step}: {loss.item():.4f}, skipping batch")
                self.optimizer.zero_grad()
                self.accumulation_step = 0
                self.global_step += 1
                continue

            # Scale loss by accumulation steps for correct gradients
            scaled_loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
            
            # Clamp loss to prevent extreme values before backward
            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                logger.warning(f"NaN/Inf loss detected at step {self.global_step}, skipping batch")
                self.optimizer.zero_grad()
                self.accumulation_step = 0
                self.global_step += 1
                continue
            
            # Cap loss at reasonable maximum to prevent gradient explosion
            scaled_loss = torch.clamp(scaled_loss, max=5.0)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w') as f:
        f.write(content)
    print("Successfully updated methods/trainer.py")
else:
    print("Could not find the code block to replace")
