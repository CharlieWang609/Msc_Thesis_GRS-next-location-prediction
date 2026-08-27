import numpy as np
import torch


import numpy as np
import torch
import os # Make sure os is imported for os.path.join

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, logdir, patience=7, verbose=False, delta=0):
        """
        Args:
            logdir (str): Directory to save checkpoints.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.log_dir = logdir # Ensure this is used consistently (e.g., self.log_dir, not logdir directly in methods)
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_return_dict = {"val_loss": np.inf} # Initialize with a high val_loss
        self.delta = delta

    def __call__(self, return_dict, model):
        """
        Call this method at the end of each validation epoch.
        Args:
            return_dict (dict): Dictionary containing validation metrics, must include 'val_loss'.
            model (torch.nn.Module): The model to save if validation loss improves.
        """
        score = return_dict["val_loss"] # We are monitoring validation loss

        if self.best_score is None: # First epoch or after a reset
            self.best_score = score
            self.save_checkpoint(return_dict, model)
            return # No change in early_stop status yet

        if score < self.best_score - self.delta: # Improvement
            self.best_score = score
            self.save_checkpoint(return_dict, model)
            self.counter = 0
        else: # No improvement
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True # Signal to stop or decay LR

    def save_checkpoint(self, return_dict, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.best_return_dict['val_loss']:.6f} --> {return_dict['val_loss']:.6f}). Saving model ..."
            )
        # Use os.path.join for constructing paths robustly
        checkpoint_path = os.path.join(self.log_dir, "checkpoint.pt")
        torch.save(model.state_dict(), checkpoint_path)
        self.best_return_dict = return_dict # Save the metrics of the best model

    def reset(self):
        """Resets the EarlyStopping state (counter, best_score, early_stop flag)."""
        if self.verbose:
            print("EarlyStopping state has been reset.")
        self.counter = 0
        self.best_score = None # Will be set again on the next __call__
        self.early_stop = False
        # Optionally, you might want to re-initialize best_return_dict if you
        # want to track the best model strictly *after* the reset.
        # For now, keeping the overall best_return_dict might be fine,
        # or reset it like so:
        # self.best_return_dict = {"val_loss": np.inf}
        # The key is that best_score=None allows a new baseline to be set.