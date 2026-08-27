# mobpredict/train/train.py

import numpy as np
import pandas as pd
import os
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
import time
from transformers import get_linear_schedule_with_warmup # Ensure this is installed
import matplotlib.pyplot as plt

# Assuming these utils are in the correct path and updated for flat config if necessary
from mobpredict.utils.earlystopping import EarlyStopping # Must have .reset() method
from mobpredict.utils.loss import ClassBalanceFocalLoss, WeightedCrossEntropyLoss, ASLSingleLabel # Must accept clip_min/max

# --- Helper Functions ---

def get_performance_dict(return_dict):
    """Converts evaluation results to a standardized performance dictionary."""
    # Use .get for safety in case some keys are missing from return_dict
    total_samples = return_dict.get("total", 0)
    perf = {
        "correct@1": return_dict.get("correct@1", 0),
        "correct@3": return_dict.get("correct@3", 0),
        "correct@5": return_dict.get("correct@5", 0),
        "correct@10": return_dict.get("correct@10", 0),
        "rr": return_dict.get("rr", 0), # This is sum of reciprocal ranks
        "f1_weighted": return_dict.get("f1_weighted", 0) * 100, # Assuming input is 0-1 range
        "f1_macro": return_dict.get("f1_macro", 0) * 100,    # Assuming input is 0-1 range
        "total": total_samples,
        "test_loss": return_dict.get("test_loss") # Optional, from single_test
    }

    if total_samples > 0:
        perf["acc@1"] = perf["correct@1"] / total_samples * 100
        perf["acc@3"] = perf["correct@3"] / total_samples * 100
        perf["acc@5"] = perf["correct@5"] / total_samples * 100
        perf["acc@10"] = perf["correct@10"] / total_samples * 100
        perf["mrr_metric"] = perf["rr"] / total_samples * 100 # Mean Reciprocal Rank as a percentage
    else:
        perf["acc@1"] = perf["acc@3"] = perf["acc@5"] = perf["acc@10"] = perf["mrr_metric"] = 0
    
    # Remove test_loss if it's None (was not computed)
    if perf["test_loss"] is None:
        del perf["test_loss"]
        
    return perf

def send_to_device(inputs, device, config=None): # config is not used here
    """Sends input tensors (x, y, x_dict) to the specified device."""
    x, y, x_dict = inputs
    x = x.to(device)
    for key_in_dict in x_dict: # Corrected variable name
        x_dict[key_in_dict] = x_dict[key_in_dict].to(device)
    y = y.to(device)
    return x, y, x_dict

def calculate_correct_total_prediction(logits, true_y):
    """
    Calculates top-k correct predictions and sum of reciprocal ranks for MRR.
    Returns: batch_metrics_array, true_y_cpu, top1_predictions_cpu
    batch_metrics_array: [correct@1, correct@3, correct@5, correct@10, rr_sum_batch, total_in_batch]
    """
    batch_metrics_values = [0.0] * 6 # Ensure float for accumulation
    top1_preds_cpu = torch.tensor([])

    num_samples_in_batch = true_y.shape[0]
    batch_metrics_values[5] = float(num_samples_in_batch) # Total samples

    if logits.shape[-1] == 0: # No classes to predict from
        # print("Warning: calculate_correct_total_prediction received logits with 0 classes.")
        return np.array(batch_metrics_values, dtype=np.float32), true_y.cpu(), top1_preds_cpu

    for i, k_val in enumerate([1, 3, 5, 10]):
        actual_k = min(k_val, logits.shape[-1]) # Handle k > num_classes
        if actual_k == 0: continue

        prediction_indices = torch.topk(logits, k=actual_k, dim=-1).indices
        
        if k_val == 1: # For F1 score calculation later and top-1 preds
            top1_preds_cpu = torch.squeeze(prediction_indices, dim=-1).cpu()

        # Check if true_y is in the top actual_k predictions
        batch_metrics_values[i] = torch.eq(true_y.unsqueeze(1), prediction_indices).any(dim=1).sum().cpu().item()

    batch_metrics_values[4] = get_mrr_sum_for_batch(logits, true_y) # Sum of reciprocal ranks for the batch
    
    return np.array(batch_metrics_values, dtype=np.float32), true_y.cpu(), top1_preds_cpu

def get_mrr_sum_for_batch(prediction_logits, targets_true):
    """Calculates sum of reciprocal ranks for a batch. Helper for calculate_correct_total_prediction."""
    if prediction_logits.shape[1] == 0: return 0.0 # No predictions
    
    # Sort predictions by score (descending) to get ranks
    indices_sorted_by_score = torch.argsort(prediction_logits, dim=-1, descending=True)
    
    # Find where true targets appear in the sorted list of predictions
    # Create a mask for hits: (targets_true.unsqueeze(-1) == indices_sorted_by_score)
    # nonzero() gives coordinates of hits; the last column is the rank (0-indexed)
    hits_mask = (targets_true.unsqueeze(-1).expand_as(indices_sorted_by_score) == indices_sorted_by_score)
    hit_coordinates = hits_mask.nonzero(as_tuple=False) # Ensure as_tuple=False for older PyTorch versions if needed

    if hit_coordinates.size(0) == 0: return 0.0 # No hits in the batch
    
    # Ranks are 1-indexed, so add 1 to the column index
    ranks_of_hits = (hit_coordinates[:, -1] + 1).float()
    reciprocal_ranks = torch.reciprocal(ranks_of_hits)
    
    return torch.sum(reciprocal_ranks).cpu().item()


def compute_class_distribution(train_loader, num_total_classes, ignore_idx=0):
    """Computes class distribution from train_loader, ignoring ignore_idx."""
    class_counts_arr = np.zeros(num_total_classes, dtype=np.int64)
    for _, y_batch_labels, _ in train_loader: # Assuming loader returns (x, y, x_dict)
        for y_single_label in y_batch_labels:
            class_idx_val = y_single_label.item()
            if class_idx_val != ignore_idx:
                if 0 <= class_idx_val < num_total_classes: # Check bounds
                    class_counts_arr[class_idx_val] += 1
    return class_counts_arr

def analyze_class_distribution(class_counts_arr, config, num_total_classes_configured):
    """Analyzes and prints class distribution statistics. Assumes flat config."""
    active_classes_count = np.sum(class_counts_arr > 0)
    if not getattr(config, "verbose", False): return # Check verbose from flat config
    
    print("=== Class Distribution Analysis ===")
    print(f"Total classes configured (total_loc_num): {num_total_classes_configured}")
    if active_classes_count == 0:
        print("No active classes found in training data (excluding ignore_index).")
        print("=================================="); return

    max_freq = np.max(class_counts_arr)
    active_class_counts = class_counts_arr[class_counts_arr > 0]
    min_freq_active = np.min(active_class_counts)
    mean_freq_active = np.mean(active_class_counts)
    median_freq_active = np.median(active_class_counts)
    
    rare_classes_num = np.sum((class_counts_arr > 0) & (class_counts_arr < 100))
    medium_classes_num = np.sum((class_counts_arr >= 100) & (class_counts_arr < 1000))
    common_classes_num = np.sum(class_counts_arr >= 1000)

    print(f"Active classes (count > 0): {active_classes_count}")
    print(f"Frequency (active) - Max: {max_freq}, Min: {min_freq_active}, Mean: {mean_freq_active:.2f}, Median: {median_freq_active:.0f}")
    if active_classes_count > 0 :
        print(f"Rare (<100): {rare_classes_num} ({100*rare_classes_num/active_classes_count:.2f}%)")
        print(f"Medium (100-999): {medium_classes_num} ({100*medium_classes_num/active_classes_count:.2f}%)")
        print(f"Common (>=1000): {common_classes_num} ({100*common_classes_num/active_classes_count:.2f}%)")
    print("==================================")

def save_class_weights_to_csv(class_counts_arr, weights_arr_to_save, log_dir_path, config, filename_prefix="class_info"):
    """Saves class frequencies and their corresponding weights to a CSV. Assumes flat config."""
    if not getattr(config, "verbose", False): return

    if torch.is_tensor(weights_arr_to_save):
        weights_arr_to_save = weights_arr_to_save.cpu().numpy()
    
    num_classes_total = len(class_counts_arr)
    # Ensure weights_arr_to_save has the same length as class_counts_arr for saving
    if len(weights_arr_to_save) != num_classes_total:
        print(f"Warning: Mismatch in lengths for saving weights. Counts: {num_classes_total}, Weights: {len(weights_arr_to_save)}. Adjusting weights array with 1s.")
        adjusted_weights = np.ones(num_classes_total)
        min_len_to_copy = min(num_classes_total, len(weights_arr_to_save))
        adjusted_weights[:min_len_to_copy] = weights_arr_to_save[:min_len_to_copy]
        weights_to_save_final = adjusted_weights
    else:
        weights_to_save_final = weights_arr_to_save

    data_to_save = {'location_id': np.arange(num_classes_total), 'frequency': class_counts_arr, 'weight': weights_to_save_final}
    weights_df = pd.DataFrame(data_to_save)
    # Use filename_prefix to distinguish files, e.g., "class_info_focal.csv"
    csv_filename = os.path.join(log_dir_path, f"{filename_prefix}.csv")
    try:
        weights_df.to_csv(csv_filename, index=False)
        if getattr(config, "verbose", False): print(f"Class counts and weights saved to {csv_filename}")
    except Exception as e: print(f"Error saving class weights to {csv_filename}: {e}")

def get_optimizer(config, model_parameters): # Takes model.parameters()
    """Gets optimizer based on flat config."""
    opt_name = getattr(config, "optimizer", "AdamW") # Default to AdamW
    lr_val = getattr(config, "lr", 0.001)
    wd_val = getattr(config, "weight_decay", 0.0005)

    if opt_name == "SGD":
        return torch.optim.SGD(model_parameters, lr=lr_val, weight_decay=wd_val, 
                               momentum=getattr(config, "momentum", 0.9), nesterov=True)
    elif opt_name == "Adam":
        return torch.optim.Adam(model_parameters, lr=lr_val, 
                                betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999)), 
                                weight_decay=wd_val)
    elif opt_name == "AdamW":
        return torch.optim.AdamW(model_parameters, lr=lr_val, 
                                 betas=(getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999)), 
                                 weight_decay=wd_val)
    raise ValueError(f"Unsupported optimizer: {opt_name}")

def plot_and_save_loss_curves(epochs_list, train_loss_list, val_loss_list, save_dir_path, run_name_str):
    """Plots training and validation loss curves and saves the plot."""
    if not epochs_list: # Don't plot if no data
        print("No epoch data to plot loss curves.")
        return
    plt.figure(figsize=(12, 7)) # Adjusted size
    plt.plot(epochs_list, train_loss_list, 'o-', label='Training Loss') # Added markers
    plt.plot(epochs_list, val_loss_list, 'o-', label='Validation Loss') # Added markers
    plt.title(f'Loss Curves for {run_name_str}', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs_list) # Ensure all epoch numbers are marked if not too many
    plt.tight_layout() # Adjust layout
    plot_filename = os.path.join(save_dir_path, f"{run_name_str}_loss_curve.png")
    try:
        plt.savefig(plot_filename, dpi=150) # Save with higher DPI
        if getattr(plt.gcf().get_axes(), '__len__', 0) > 0 : # Check if figure has axes (basic check)
             print(f"Loss curve plot saved to {plot_filename}")
    except Exception as e: print(f"Error saving loss curve plot: {e}")
    plt.close() # Close the figure

# --- Main Training and Evaluation Functions ---

def single_train(config, model, train_data_loader, optim_instance, device_to_use, current_epoch, 
                 lr_scheduler_instance, lr_decay_phase_count, global_step_counter, loss_func_instance):
    """Trains the model for one epoch. Assumes flat config."""
    model.train()
    epoch_total_loss = 0.0
    num_batches_in_epoch = len(train_data_loader)
    print_step_val = getattr(config, "print_step", 10) # From flat config
    verbose_flag = getattr(config, "verbose", False)

    # Accumulator for metrics over print_step batches for more stable reporting
    print_interval_loss_sum = 0.0
    print_interval_metrics_sum = np.array([0.0] * 6, dtype=np.float32)
    print_interval_start_time = time.time()

    for batch_idx, batch_inputs in enumerate(train_data_loader):
        global_step_counter += 1
        x_data, y_data, x_context_dict = send_to_device(batch_inputs, device_to_use)
        
        logits_output = model(x_data, x_context_dict, device_to_use)
        if logits_output.shape[-1] == 0:
             if verbose_flag and batch_idx == 0: print(f"Warning: Empty logits in train epoch {current_epoch+1}, batch {batch_idx+1}.")
             continue

        loss_val = loss_func_instance(logits_output, y_data.reshape(-1))
        optim_instance.zero_grad(); loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Standard clip value
        optim_instance.step()
        
        # Assuming get_linear_schedule_with_warmup is the main scheduler tied to optimizer steps
        lr_scheduler_instance.step() 

        epoch_total_loss += loss_val.item()
        print_interval_loss_sum += loss_val.item()
        
        batch_metrics, _, _ = calculate_correct_total_prediction(logits_output, y_data)
        print_interval_metrics_sum += batch_metrics

        if verbose_flag and (batch_idx + 1) % print_step_val == 0:
            current_lr_val = optim_instance.param_groups[0]['lr']
            avg_loss_print_interval = print_interval_loss_sum / print_step_val
            
            total_samples_interval = print_interval_metrics_sum[5]
            acc1_interval = 100 * print_interval_metrics_sum[0] / total_samples_interval if total_samples_interval > 0 else 0
            mrr_interval = 100 * print_interval_metrics_sum[4] / total_samples_interval if total_samples_interval > 0 else 0
            
            avg_epoch_loss_so_far = epoch_total_loss / (batch_idx + 1)

            print(
                f"Epoch {current_epoch + 1} [{100*(batch_idx+1)/num_batches_in_epoch:.1f}%]\t"
                f"Loss: {avg_loss_print_interval:.4f} (Epoch Avg: {avg_epoch_loss_so_far:.4f})\t"
                f"Acc@1: {acc1_interval:.2f}%\tMRR: {mrr_interval:.2f}%\tLR: {current_lr_val:.7f}\t"
                f"Time: {time.time() - print_interval_start_time:.2f}s \r",
                end="", flush=True,
            )
            # Reset for next print interval
            print_interval_loss_sum = 0.0
            print_interval_metrics_sum = np.array([0.0] * 6, dtype=np.float32)
            print_interval_start_time = time.time()

        if getattr(config, "debug", False) and batch_idx > 20 : break # Debug mode short epoch
            
    if verbose_flag: print() # Newline after epoch progress bar
    
    avg_loss_for_epoch = epoch_total_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
    return global_step_counter, avg_loss_for_epoch

def single_validate(config, model, val_data_loader, device_to_use, loss_func_instance):
    """Validates the model for one epoch. Assumes flat config."""
    model.eval()
    epoch_total_val_loss = 0.0
    all_true_labels, all_top1_preds = [], []
    # [c@1, c@3, c@5, c@10, rr_sum, total]
    epoch_metric_sums = np.array([0.0] * 6, dtype=np.float32)
    verbose_flag = getattr(config, "verbose", False)

    with torch.no_grad():
        for batch_idx, batch_inputs in enumerate(val_data_loader):
            x_data, y_data, x_context_dict = send_to_device(batch_inputs, device_to_use)
            logits_output = model(x_data, x_context_dict, device_to_use)

            if logits_output.shape[-1] == 0:
                if verbose_flag and batch_idx == 0: print("Warning: Empty logits in validation.")
                # Still need to count these samples for total if they have labels
                epoch_metric_sums[5] += y_data.shape[0]
                all_true_labels.extend(y_data.cpu().tolist()) # Add true labels even if no preds
                # all_top1_preds will have fewer items if this happens, F1 might be skewed or error
                continue

            loss_val = loss_func_instance(logits_output, y_data.reshape(-1))
            epoch_total_val_loss += loss_val.item()
            
            batch_metrics, true_labels_cpu, top1_preds_cpu = calculate_correct_total_prediction(logits_output, y_data)
            epoch_metric_sums += batch_metrics # Accumulate sums
            all_true_labels.extend(true_labels_cpu.tolist())
            if top1_preds_cpu.numel() > 0 : # Only extend if there are predictions
                 if top1_preds_cpu.ndim == 0: all_top1_preds.append(top1_preds_cpu.item())
                 else: all_top1_preds.extend(top1_preds_cpu.tolist())
    
    avg_val_loss_epoch = epoch_total_val_loss / len(val_data_loader) if len(val_data_loader) > 0 else 0
    
    f1_weighted_val, f1_macro_val = 0.0, 0.0
    # Ensure all_top1_preds has same length as all_true_labels if no logits issue
    # If some batches had empty logits, len(all_top1_preds) < len(all_true_labels)
    # F1 score might be problematic. For now, calculate if lists are non-empty.
    # A better way is to ensure placeholder preds for F1 if logits were empty.
    # Current calculate_correct_total_prediction returns empty top1_preds_cpu if logits empty.
    if all_true_labels and all_top1_preds and len(all_true_labels) == len(all_top1_preds):
        f1_weighted_val = f1_score(all_true_labels, all_top1_preds, average="weighted", zero_division=0)
        f1_macro_val = f1_score(all_true_labels, all_top1_preds, average="macro", zero_division=0)
    elif all_true_labels and verbose_flag:
        print("Warning: F1 score calculation skipped or potentially inaccurate due to empty/mismatched predictions in validation.")

    # Unpack summed metrics
    c1, c3, c5, c10, rr_sum_total, total_samples_epoch = epoch_metric_sums
    acc1_val_epoch = (100 * c1 / total_samples_epoch) if total_samples_epoch > 0 else 0
    mrr_val_epoch = (100 * rr_sum_total / total_samples_epoch) if total_samples_epoch > 0 else 0
    
    if verbose_flag:
        print(
            f"Validation Epoch: Loss={avg_val_loss_epoch:.4f}, Acc@1={acc1_val_epoch:.2f}%, "
            f"F1-W={100*f1_weighted_val:.2f}%, F1-M={100*f1_macro_val:.2f}%, MRR={mrr_val_epoch:.2f}%"
        )

    return {
        "val_loss": avg_val_loss_epoch, "correct@1": c1, "correct@3": c3, "correct@5": c5, "correct@10": c10,
        "f1_weighted": f1_weighted_val, "f1_macro": f1_macro_val, 
        "rr": rr_sum_total, "total": total_samples_epoch, # rr is sum, total is count
    }

def train_net(config, model, train_loader, val_loader, device, log_dir): # Assumes flat config
    """Main training loop for the network. Assumes flat config."""
    performance_dict = {} # To store best performance
    ignore_idx_val = getattr(config, "ignore_index", 0) # Assuming ignore_index is a top-level config or default

    optim_instance = get_optimizer(config, model.parameters())
    
    # Class distribution for weighted losses (total_loc_num from flat config, set in run.py)
    class_counts_arr = compute_class_distribution(train_loader, config.total_loc_num, ignore_idx=ignore_idx_val)
    analyze_class_distribution(class_counts_arr, config, config.total_loc_num)

    # --- Loss Function Setup with Flat Config ---
    loss_type_str = getattr(config, "type", "ce").lower() # 'type' from loss section, now top-level
    use_weights_flag = getattr(config, "use_class_weights", False)
    print(f"Using loss: {loss_type_str}, Config 'use_class_weights': {use_weights_flag}")

    apply_global_clipping = False
    clip_min_global = None
    clip_max_global = None
    enable_clip_flag = getattr(config, "enable_weight_clipping", False) # Explicit flag to enable/disable
    cfg_clip_min = getattr(config, "weight_clip_min", None)
    cfg_clip_max = getattr(config, "weight_clip_max", None)

    if hasattr(config, "enable_weight_clipping") and not enable_clip_flag : # Explicitly False
        if getattr(config, "verbose", False): print("  Global weight clipping explicitly disabled.")
    elif cfg_clip_min is not None or cfg_clip_max is not None: # Values are set
        apply_global_clipping = True
        clip_min_global = cfg_clip_min
        clip_max_global = cfg_clip_max
        if getattr(config, "verbose", False): print(f"  Global weight clipping active: min={clip_min_global}, max={clip_max_global}")
    elif enable_clip_flag: # enable_weight_clipping: true, but no specific min/max values
         apply_global_clipping = True # Will pass None to loss, effectively no clamp unless loss defaults
         if getattr(config, "verbose", False): print("  Global weight clipping enabled by flag, but no min/max values set in config. Effective clip range depends on loss defaults.")
    elif getattr(config, "verbose", False):
        print("  Global weight clipping not configured.")
        
    weights_for_saving = np.ones(config.total_loc_num) # Default for saving if no specific weights used

    if loss_type_str == "focal":
        loss_fn_obj = ClassBalanceFocalLoss(
            gamma=getattr(config, "gamma", 2.0), beta=getattr(config, "beta", 0.9999),
            reduction='mean', ignore_index=ignore_idx_val,
            clip_min=clip_min_global if apply_global_clipping else None,
            clip_max=clip_max_global if apply_global_clipping else None
        )
        if use_weights_flag: # This means use Class-Balancing (ENS alpha)
            loss_fn_obj.update_weights(class_counts_arr, device_for_weights=device)
            if hasattr(loss_fn_obj, 'class_weights') and loss_fn_obj.class_weights is not None:
                weights_for_saving = loss_fn_obj.class_weights.cpu().numpy()
    elif loss_type_str == "weighted_ce":
        loss_fn_obj = WeightedCrossEntropyLoss(
            reduction='mean', ignore_index=ignore_idx_val,
            clip_min=clip_min_global if apply_global_clipping else None,
            clip_max=clip_max_global if apply_global_clipping else None
        )
        loss_fn_obj.update_weights(class_counts_arr, device_for_weights=device) # WCE always updates weights
        if hasattr(loss_fn_obj, 'class_weights') and loss_fn_obj.class_weights is not None:
            weights_for_saving = loss_fn_obj.class_weights.cpu().numpy()
    elif loss_type_str == "asl":
        loss_fn_obj = ASLSingleLabel( # ASL has its own internal 'clip' for probabilities, not for class weights
            gamma_pos=getattr(config, "asl_gamma_pos", 0.0), gamma_neg=getattr(config, "asl_gamma_neg", 2.0),
            clip=getattr(config, "asl_clip", 0.05), 
            reduction='mean', ignore_index=ignore_idx_val
        )
    elif loss_type_str == "ce":
        ce_loss_weights = None
        if use_weights_flag: # Apply simple inverse frequency weights if true
            if np.any(class_counts_arr > 0):
                raw_ce_weights = torch.ones(config.total_loc_num, dtype=torch.float32)
                active_mask = class_counts_arr > 0
                counts_tensor = torch.from_numpy(class_counts_arr[active_mask]).float() if isinstance(class_counts_arr,np.ndarray) else class_counts_arr[active_mask].float()
                raw_ce_weights[active_mask] = 1.0 / counts_tensor
                if apply_global_clipping:
                    min_c = clip_min_global if clip_min_global is not None else -float('inf')
                    max_c = clip_max_global if clip_max_global is not None else float('inf')
                    if getattr(config, "verbose", False): print(f"  Applying global clipping to standard CE weights: min={min_c}, max={max_c}")
                    raw_ce_weights = torch.clamp(raw_ce_weights, min=min_c, max=max_c)
                ce_loss_weights = raw_ce_weights.to(device)
                weights_for_saving = ce_loss_weights.cpu().numpy()
        loss_fn_obj = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_idx_val, weight=ce_loss_weights)
    else:
        raise ValueError(f"Unsupported loss_type in config: {loss_type_str}")

    loss_fn_obj = loss_fn_obj.to(device)
    save_class_weights_to_csv(class_counts_arr, weights_for_saving, log_dir, config, filename_prefix=f"class_info_{loss_type_str}")

    # Schedulers & Early Stopping
    num_train_steps_per_epoch = len(train_loader)
    total_training_steps = num_train_steps_per_epoch * getattr(config, "num_training_epochs", 50) # num_training_epochs from flat config
    num_warmup_steps = num_train_steps_per_epoch * getattr(config, "num_warmup_epochs", 2) # num_warmup_epochs from flat config

    lr_scheduler = get_linear_schedule_with_warmup(optim_instance, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)
    # StepLR for use with early stopping plateaus
    es_lr_scheduler = StepLR(optim_instance, step_size=getattr(config, "lr_step_size", 1), gamma=getattr(config, "lr_gamma", 0.1))
    
    early_stopping_obj = EarlyStopping(log_dir, patience=getattr(config, "patience", 5), 
                                     verbose=getattr(config, "verbose", False), delta=0.001) # patience from flat config
    
    training_start_overall_time = time.time()
    global_step = 0
    lr_decays_done_count = 0
    max_lr_decays_allowed = getattr(config, "max_lr_decays", 2) # max_lr_decays from flat config

    epoch_train_loss_history, epoch_val_loss_history, epochs_plotted = [], [], []

    for epoch_num in range(getattr(config, "max_epoch", 100)): # max_epoch from flat config
        if getattr(config, "verbose", False): 
            print(f"\n--- Epoch {epoch_num + 1}/{getattr(config, 'max_epoch', 100)}, LR: {optim_instance.param_groups[0]['lr']:.7f} ---")

        global_step, avg_train_loss_epoch = single_train(
            config, model, train_loader, optim_instance, device, epoch_num, 
            lr_scheduler, lr_decays_done_count, global_step, loss_fn_obj)
        
        val_metrics_dict = single_validate(config, model, val_loader, device, loss_fn_obj)
        
        epoch_train_loss_history.append(avg_train_loss_epoch)
        epoch_val_loss_history.append(val_metrics_dict["val_loss"])
        epochs_plotted.append(epoch_num + 1)

        early_stopping_obj(val_metrics_dict, model) # Pass full dict and model

        if early_stopping_obj.early_stop:
            if getattr(config, "verbose", False): print("Early stopping patience met.")
            if lr_decays_done_count < max_lr_decays_allowed:
                print(f"  Loading best model and reducing LR. Decay count: {lr_decays_done_count + 1}/{max_lr_decays_allowed}")
                model.load_state_dict(torch.load(os.path.join(log_dir, "checkpoint.pt"), map_location=device))
                es_lr_scheduler.step() # Reduce LR using StepLR
                early_stopping_obj.reset() # Reset early stopping state
                lr_decays_done_count += 1
            else:
                print("  Max LR decays reached. Stopping training.")
                break # Exit epoch loop
        
        if getattr(config, "debug", False) and epoch_num > 0 : # Debug: run only 2 epochs
            print("Debug mode: stopping after 2 epochs."); break

    # After training loop finishes
    if epochs_plotted: # Plot losses if any epochs were run
        plot_and_save_loss_curves(epochs_plotted, epoch_train_loss_history, epoch_val_loss_history, 
                                  log_dir, getattr(config, "run_name", "default_run")) # run_name from flat config

    # Load best model based on early stopping for final performance
    best_model_path = os.path.join(log_dir, "checkpoint.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path} for final performance.")
    else:
        print("Warning: No checkpoint found from early stopping. Using last model state.")

    # Use best_return_dict from early_stopping if available, else last validation
    final_performance_metrics = early_stopping_obj.best_return_dict \
        if early_stopping_obj.best_score is not None else val_metrics_dict
    
    performance_dict = get_performance_dict(final_performance_metrics)

    print(f"Training finished. Total time: {(time.time() - training_start_overall_time) / 60:.2f} min.")
    print(f"Best validation performance (from early stopping):")
    for key, val in performance_dict.items():
        if key not in ["total", "rr"]: # Don't print raw total/rr sums here
             print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")
    
    return model, performance_dict


def single_test(config, model, test_data_loader, device_to_use, loss_fn_obj=None,
                save_results=False, 
                save_dir=None, dataset_name=None):
    """Evaluates model on test set. Assumes flat config."""
    model.eval()
    all_true_labels, all_top1_preds, detailed_predictions_list = [], [], []
    epoch_metric_sums = np.array([0.0] * 6, dtype=np.float32) # c@1, c@3, c@5, c@10, rr_sum, total
    total_test_set_loss = 0.0
    num_batches_processed = 0
    verbose_flag = getattr(config, "verbose", False)

    with torch.no_grad():
        for batch_inputs in test_data_loader:
            x_data, y_data, x_context_dict = send_to_device(batch_inputs, device_to_use)
            
            user_ids_tensor_cpu = x_context_dict.get("user", torch.empty(0)).cpu()
            if user_ids_tensor_cpu.ndim > 1 and user_ids_tensor_cpu.shape[1] > 0:
                user_ids_tensor_cpu = user_ids_tensor_cpu[:, 0]
            user_ids_batch_list = user_ids_tensor_cpu.tolist()

            logits_output = model(x_data, x_context_dict, device_to_use)

            if logits_output.shape[-1] == 0:
                if verbose_flag: print(f"Warning: Empty logits in test for dataset {dataset_name}.")
                epoch_metric_sums[5] += y_data.shape[0]
                all_true_labels.extend(y_data.cpu().tolist())
                if save_results: 
                    for i in range(len(y_data)):
                        sample_detail = {
                            "user_id": user_ids_batch_list[i] if i < len(user_ids_batch_list) else -1,
                            "true_label": y_data[i].item()
                        }
                        for k_pred in range(1, 11): 
                            sample_detail[f"pred_{k_pred}"] = -1
                            sample_detail[f"prob_{k_pred}"] = 0.0
                        detailed_predictions_list.append(sample_detail)
                continue

            if loss_fn_obj is not None:
                loss_val = loss_fn_obj(logits_output, y_data.reshape(-1))
                total_test_set_loss += loss_val.item()
            num_batches_processed +=1

            batch_metrics, true_labels_cpu, top1_preds_cpu = calculate_correct_total_prediction(logits_output, y_data)
            epoch_metric_sums += batch_metrics
            all_true_labels.extend(true_labels_cpu.tolist())
            if top1_preds_cpu.numel() > 0:
                 if top1_preds_cpu.ndim == 0: all_top1_preds.append(top1_preds_cpu.item())
                 else: all_top1_preds.extend(top1_preds_cpu.tolist())

            if save_results: 
                num_classes_available = logits_output.shape[1]
                top_k_to_save = min(10, num_classes_available)
                
                if top_k_to_save > 0: # Ensure there's something to get topk from
                    topk_vals, topk_idx = torch.topk(logits_output, k=top_k_to_save, dim=1)
                    all_probs_batch_tensor = torch.softmax(logits_output, dim=1)
                    topk_probs_selected_tensor = all_probs_batch_tensor.gather(1, topk_idx).cpu().numpy()
                    topk_idx_cpu_numpy = topk_idx.cpu().numpy()
                else: # Handle case where no classes means no predictions to save
                    topk_idx_cpu_numpy = np.array([[] for _ in range(len(y_data))]) # Empty predictions
                    topk_probs_selected_tensor = np.array([[] for _ in range(len(y_data))])


                for i in range(len(y_data)):
                    sample_detail = {
                        "user_id": user_ids_batch_list[i] if i < len(user_ids_batch_list) else -1,
                        "true_label": y_data[i].item()
                    }
                    for k_rank in range(top_k_to_save): 
                        sample_detail[f"pred_{k_rank+1}"] = topk_idx_cpu_numpy[i, k_rank]
                        sample_detail[f"prob_{k_rank+1}"] = topk_probs_selected_tensor[i, k_rank]
                    for k_fill in range(top_k_to_save + 1, 11):
                        sample_detail[f"pred_{k_fill}"] = -1
                        sample_detail[f"prob_{k_fill}"] = 0.0
                    detailed_predictions_list.append(sample_detail)
    
    if save_results and save_dir is not None and dataset_name is not None and detailed_predictions_list: 
        results_df = pd.DataFrame(detailed_predictions_list)
        pred_output_path = os.path.join(save_dir, f"{dataset_name}_predictions.csv")
        try:
            results_df.to_csv(pred_output_path, index=False)
            if verbose_flag: print(f"Test prediction details saved to {pred_output_path}")
        except Exception as e: print(f"Error saving test prediction details: {e}")

    avg_loss_test_epoch = total_test_set_loss / num_batches_processed if loss_fn_obj is not None and num_batches_processed > 0 else None
    
    f1_weighted_test, f1_macro_test = 0.0, 0.0
    if all_true_labels and all_top1_preds and len(all_true_labels) == len(all_top1_preds):
        f1_weighted_test = f1_score(all_true_labels, all_top1_preds, average="weighted", zero_division=0)
        f1_macro_test = f1_score(all_true_labels, all_top1_preds, average="macro", zero_division=0)
    elif all_true_labels and verbose_flag:
         print("Warning: F1 score calculation potentially inaccurate for test set due to empty/mismatched predictions.")

    c1, c3, c5, c10, rr_sum_total, total_samples_epoch = epoch_metric_sums
    acc1_test_epoch = (100 * c1 / total_samples_epoch) if total_samples_epoch > 0 else 0
    mrr_test_epoch = (100 * rr_sum_total / total_samples_epoch) if total_samples_epoch > 0 else 0

    if verbose_flag:
        loss_str_test = f"Loss={avg_loss_test_epoch:.4f}, " if avg_loss_test_epoch is not None else ""
        print(
            f"Test Results ({dataset_name}): {loss_str_test}Acc@1={acc1_test_epoch:.2f}%, "
            f"F1-W={100*f1_weighted_test:.2f}%, F1-M={100*f1_macro_test:.2f}%, MRR={mrr_test_epoch:.2f}%"
        )

    return { 
        "correct@1": c1, "correct@3": c3, "correct@5": c5, "correct@10": c10,
        "f1_weighted": f1_weighted_test, "f1_macro": f1_macro_test, 
        "rr": rr_sum_total, "total": total_samples_epoch, "test_loss": avg_loss_test_epoch
    }