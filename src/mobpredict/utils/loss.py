import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ClassBalanceFocalLoss(nn.Module):
    """
    Class-balanced Focal Loss.
    Alpha weights based on "Effective Number of Samples", can be optionally clamped.
    """
    def __init__(self, gamma=2.0, beta=0.9999, reduction='mean', ignore_index=0,
                 clip_min=None, clip_max=None): # Added clip parameters
        super(ClassBalanceFocalLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta  # Hyperparameter for effective number of samples
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.class_weights = None # CB weights (alpha_t) for Focal Loss
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._clipping_actually_applied = False # Internal flag

    def update_weights(self, class_counts, device_for_weights='cpu'):
        """
        Update class-balancing weights (alpha_t) based on class counts and ENS.
        Args:
            class_counts (Tensor or np.ndarray): Number of samples per class.
            device_for_weights (str or torch.device): Device to place the weights on.
        """
        if isinstance(class_counts, np.ndarray):
            class_counts_tensor = torch.from_numpy(class_counts).float()
        else:
            class_counts_tensor = class_counts.clone().detach().float()
        
        effective_num_beta_N = torch.pow(self.beta, class_counts_tensor)
        weights = (1.0 - self.beta) / (1.0 - effective_num_beta_N + 1e-8) # Add epsilon for stability if 1 - beta^N is close to 0

        weights[class_counts_tensor == 0] = 1.0 # Default weight for classes with no samples

        self._clipping_actually_applied = False # Reset flag
        if self.clip_min is not None or self.clip_max is not None:
            min_val = self.clip_min if self.clip_min is not None else -float('inf')
            max_val = self.clip_max if self.clip_max is not None else float('inf')

            if self.clip_min is not None and self.clip_max is not None and self.clip_min > self.clip_max:
                print(f"Warning: CBFL alpha_clip_min ({self.clip_min}) > alpha_clip_max ({self.clip_max}). Check config.")
            
            original_weights_snapshot = weights.clone()
            weights = torch.clamp(weights, min=min_val, max=max_val)
            if not torch.equal(weights, original_weights_snapshot):
                self._clipping_actually_applied = True
        
        self.class_weights = weights.to(device_for_weights)

        # Logging the weights after update
        mask = class_counts_tensor > 0 # Consider only classes that were observed
        active_weights = self.class_weights[mask] if torch.any(mask) else torch.tensor([])
        clip_status = f"(clipping active: min={self.clip_min}, max={self.clip_max})" if self._clipping_actually_applied else "(clipping not applied or had no effect)"
        
        print(f"ClassBalanceFocalLoss: Alpha weights updated. {clip_status}")
        if active_weights.numel() > 0:
            print(f"  Active alpha stats - Min: {active_weights.min().item():.4f}, Max: {active_weights.max().item():.4f}, Mean: {active_weights.mean().item():.4f}")
            if active_weights.numel() <= 20:
                 print(f"  Some active alpha weights: {active_weights.tolist()}")
        elif self.class_weights.numel() > 0: # If no active classes with count > 0, show all weights
             print(f"  All alpha weights (as no specific active mask applied or all zero counts): Min: {self.class_weights.min().item():.4f}, Max: {self.class_weights.max().item():.4f}, Mean: {self.class_weights.mean().item():.4f}")


    def forward(self, inputs, targets):
        # ... (forward method remains largely the same as previously defined for CBFL) ...
        # It uses self.class_weights as alpha_t.
        current_compute_device = inputs.device

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss) 
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss # Base Focal Loss without alpha

        if self.class_weights is not None:
            if self.class_weights.device != current_compute_device:
                self.class_weights = self.class_weights.to(current_compute_device)

            current_cb_weights = self.class_weights
            if self.class_weights.size(0) != inputs.size(1):
                # print(f"CBFL Warning: Alpha weight dimension ({self.class_weights.size(0)}) mismatch with input classes ({inputs.size(1)}). Adjusting.")
                if self.class_weights.size(0) < inputs.size(1):
                    extended_weights = torch.ones(inputs.size(1), device=current_compute_device)
                    extended_weights[:self.class_weights.size(0)] = self.class_weights
                    current_cb_weights = extended_weights
                else:
                    current_cb_weights = self.class_weights[:inputs.size(1)]
            
            # Gather alpha_t for each sample based on its target class
            # Ensure targets are within bounds for indexing current_cb_weights
            valid_targets_mask = (targets != self.ignore_index) & (targets >= 0) & (targets < current_cb_weights.size(0))
            
            alpha_t = torch.ones_like(targets, dtype=inputs.dtype, device=current_compute_device) # Default alpha = 1
            if torch.any(valid_targets_mask):
                alpha_t[valid_targets_mask] = current_cb_weights[targets[valid_targets_mask]]
            
            loss = alpha_t * loss # Apply class-balancing weights (alpha_t)

        # Apply reduction, considering ignore_index for the final loss
        mask = (targets != self.ignore_index)
        if not torch.any(mask):
            return torch.tensor(0.0, device=current_compute_device, requires_grad=True)

        if self.reduction == 'mean':
            return loss[mask].mean()
        elif self.reduction == 'sum':
            return loss[mask].sum()
        else: 
            # For 'none', we should still mask out the ignored indices in the returned loss tensor
            loss_no_reduction = torch.zeros_like(loss)
            loss_no_reduction[mask] = loss[mask]
            return loss_no_reduction

