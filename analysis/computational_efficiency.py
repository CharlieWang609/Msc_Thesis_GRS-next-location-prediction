import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from thop import profile

# Import user-defined models
try:
    from mobpredict.networks import MambaEncoder, RNNs, TransEncoder
except ImportError as e:
    raise SystemExit(
        "Could not import all models. Install the project and optional analysis/Mamba dependencies with "
        "`pip install -e '.[analysis,mamba]'`. "
        f"Original import error: {e}"
    ) from e


# Helper class for thop
class ModelWrapperForFlops(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src, time, weekday, length, device):
        context_dict = {"time": time, "weekday": weekday, "len": length}
        return self.model(src, context_dict, device)


def get_base_config():
    """Creates a base configuration object."""
    return SimpleNamespace(
        base_emb_size=256, fc_dropout=0.1, dropout=0.1, if_embed_time=True,
        total_loc_num=2935, num_classes=2935, num_encoder_layers=2, nhead=8,
        dim_feedforward=512, rnn_type='LSTM', hidden_size=256, attention=False,
        num_layers=2, d_state=32, d_conv=4, expand_factor=2
    )

def calculate_rnn_flops(config, seq_len):
    """Manually calculates the approximate FLOPs for the RNNs (LSTM) model."""
    L = seq_len
    D_model = config.base_emb_size
    D_hidden = config.hidden_size
    num_layers = config.num_layers
    num_classes = config.num_classes

    total_lstm_flops = 0
    flops_layer_1 = L * 4 * (2 * D_model * D_hidden + 2 * D_hidden * D_hidden)
    total_lstm_flops += flops_layer_1
    if num_layers > 1:
        flops_per_subsequent_layer = L * 4 * (2 * D_hidden * D_hidden + 2 * D_hidden * D_hidden)
        total_lstm_flops += (num_layers - 1) * flops_per_subsequent_layer
    flops_fc_simple = 2 * D_hidden * num_classes
    total_flops = total_lstm_flops + flops_fc_simple
    return total_flops

def calculate_mamba_flops(config, seq_len):
    """Manually calculates the approximate FLOPs for the MambaEncoder model."""
    L, D, E, N, d_conv, num_layers = seq_len, config.base_emb_size, config.expand_factor, config.d_state, config.d_conv, config.num_layers
    d_inner = int(E * D)
    flops_in_proj = 2 * L * D * d_inner
    flops_conv = 2 * L * d_inner * d_conv
    flops_ssm_disc = L * (d_inner * N + d_inner * N + d_inner)
    flops_ssm_main = 2 * L * d_inner * N
    flops_gating = L * d_inner
    flops_out_proj = 2 * L * d_inner * D
    flops_one_layer = flops_in_proj + flops_conv + flops_ssm_disc + flops_ssm_main + flops_gating + flops_out_proj
    total_flops = num_layers * flops_one_layer
    total_flops += 2 * D * config.num_classes
    return total_flops


def calculate_mhsa_flops(config, seq_len):
    """Manually calculates the approximate FLOPs for the TransEncoder (MHSA) model."""
    n, d, d_ff = seq_len, config.base_emb_size, config.dim_feedforward
    num_layers, num_classes = config.num_encoder_layers, config.num_classes
    flops_embedding = 2 * n * d
    flops_qkv_proj = 3 * (2 * n * d * d)
    flops_scores = 2 * n * n * d
    flops_attn_v = 2 * n * n * d
    flops_out_proj = 2 * n * d * d
    flops_mhsa_total = flops_qkv_proj + flops_scores + flops_attn_v + flops_out_proj
    flops_ffn1 = 2 * n * d * d_ff
    flops_ffn2 = 2 * n * d_ff * d
    flops_ffn_total = flops_ffn1 + flops_ffn2
    flops_one_encoder_layer = flops_mhsa_total + flops_ffn_total
    flops_all_layers = num_layers * flops_one_encoder_layer
    
    flops_fc = 2 * d * num_classes

    total_flops = flops_embedding + flops_all_layers + flops_fc
    return total_flops


def calculate_thop_flops(model, device, config, seq_len):
    """Calculates FLOPs for a given model using thop library."""
    try:
        batch_size = 1
        src = torch.randint(1, config.total_loc_num, (seq_len, batch_size), device=device)
        time_tensor = torch.randint(0, 96, (seq_len, batch_size), device=device)
        weekday = torch.randint(0, 7, (seq_len, batch_size), device=device)
        length = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        wrapped_model = ModelWrapperForFlops(model)
        macs, _ = profile(wrapped_model, inputs=(src, time_tensor, weekday, length, device), verbose=False)
        return macs * 2
    except Exception as e:
        print(f"Warning (thop): Could not calculate FLOPs for {type(model).__name__}. Reason: {e}")
        return 0


def measure_speed(model, device, config, batch_size, seq_len, mode='inference', repetitions=200): 
    """
    Measures the model's speed with a more robust warmup and repetition scheme.
    Returns a dictionary containing relevant metrics.
    """
    warmup_iters = 200
    
    src = torch.randint(1, config.total_loc_num, (seq_len, batch_size), device=device)
    time_tensor = torch.randint(0, 96, (seq_len, batch_size), device=device)
    weekday = torch.randint(0, 7, (seq_len, batch_size), device=device)
    context_dict = {
        "time": time_tensor, "weekday": weekday,
        "len": torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    }
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Perform warmup
    with torch.set_grad_enabled(mode == 'train'):
        for _ in range(warmup_iters):
            if mode == 'train': 
                optimizer.zero_grad()
            output = model(src, context_dict, device)
            if mode == 'train':
                target_shape = (output.size(0),) if len(output.shape) > 1 else ()
                target = torch.randint(0, config.num_classes, target_shape, device=device, dtype=torch.long)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

    # Synchronize before starting the timer for accurate measurement
    if device.type == 'cuda': 
        torch.cuda.synchronize()
    start_time = time.time()
    
    # Main measurement loop
    with torch.set_grad_enabled(mode == 'train'):
        for _ in range(repetitions):
            if mode == 'train': 
                optimizer.zero_grad()
            output = model(src, context_dict, device)
            if mode == 'train':
                target_shape = (output.size(0),) if len(output.shape) > 1 else ()
                target = torch.randint(0, config.num_classes, target_shape, device=device, dtype=torch.long)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

    # Synchronize before ending the timer
    if device.type == 'cuda': 
        torch.cuda.synchronize()
    end_time = time.time()
    
    total_time_seconds = end_time - start_time
    avg_time_ms_per_batch = (total_time_seconds / repetitions) * 1000
    
    if mode == 'inference':
        total_samples = repetitions * batch_size
        pps = total_samples / total_time_seconds
        return {"pps": pps, "ms_per_batch": avg_time_ms_per_batch}
    else:  # train mode
        return {"ms_per_batch": avg_time_ms_per_batch}


def plot_results(df, save_path='benchmark_results.png'):
    """
    Create a combined figure with three subplots showing FLOPs, PPS, and Training time.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    models = df['Model'].unique()
    seq_lengths = sorted(df['SeqLen'].unique())
    
    colors = {'RNN (LSTM)': 'orange', 'MHSA':'blue' , 'Mamba': 'green'}
    markers = {'RNN (LSTM)': 'o', 'MHSA': 'o', 'Mamba': 'o'}
    
    for model in models:
        model_data = df[df['Model'] == model].sort_values('SeqLen')
        color = colors.get(model, 'black')
        marker = markers.get(model, 'o')
        
        # Plot 1: FLOPs vs Sequence Length
        axes[0].plot(model_data['SeqLen'], model_data['FLOPs (M)'], 
                     marker=marker, label=model, color=color, linewidth=2, markersize=8)
        
        # CORRECTED Plot 2: Inference (PPS) vs Sequence Length
        axes[1].plot(model_data['SeqLen'], model_data['Inference (PPS)'], 
                     marker=marker, label=model, color=color, linewidth=2, markersize=8)
        
        # CORRECTED Plot 3: Train (ms/batch) vs Sequence Length
        axes[2].plot(model_data['SeqLen'], model_data['Train (ms/batch)'], 
                     marker=marker, label=model, color=color, linewidth=2, markersize=8)
        
    for ax in axes:
        ax.set_xlabel('Sequence Length (SeqLen)', fontsize=11)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(loc='best')
        # ax.set_xticks(seq_lengths) 
        ax.tick_params(axis='x', rotation=45)

    # Configure subplot 1: FLOPs
    axes[0].set_ylabel('FLOPs (M)', fontsize=11)
    axes[0].set_title('FLOPs vs. Sequence Length', fontsize=12, fontweight='bold')
    
    # CORRECTED Configure subplot 2: Inference PPS
    axes[1].set_ylabel('Inference Throughput (PPS)', fontsize=11)
    axes[1].set_title('Inference Speed vs. Sequence Length', fontsize=12, fontweight='bold')
    
    # CORRECTED Configure subplot 3: Training time
    axes[2].set_ylabel('Training Time (ms/batch)', fontsize=11)
    axes[2].set_title('Training Speed vs. Sequence Length', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nCombined plot saved to {save_path}")


def main():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = get_base_config()
    batch_size = 16
    sequence_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    output_dir = Path("outputs/computational_efficiency")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Model Speed Benchmark ---")
    print(f"Using device: {device.type.upper()}")
    print(f"Testing sequence lengths: {sequence_lengths}")
    print(f"Batch size: {batch_size}")
    
    models_to_test = {
        "RNN (LSTM)": lambda: RNNs(config=base_config),
        "MHSA": lambda: TransEncoder(config=base_config),
        "Mamba": lambda: MambaEncoder(config=base_config)
    }
    
    all_results = []
    
    for seq_len in sequence_lengths:
        print(f"\n{'='*60}")
        print(f"Testing with Sequence Length: {seq_len}")
        print(f"{'='*60}")
        
        for name, model_constructor in models_to_test.items():
            print(f"\n--- Benchmarking {name} (SeqLen={seq_len}) ---")
            
            try:
                # Create fresh model instance for each test
                model = model_constructor().to(device)
                
                # Calculate parameters (only once per model type)
                if seq_len == sequence_lengths[0]:
                    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                else:
                    # This could be optimized to not recalculate, but it's fast enough.
                    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # --- KEY CHANGE: Use manual calculation for all models ---
                if name == "Mamba":
                    flops = calculate_mamba_flops(base_config, seq_len)
                    print(f"   - Using manual calculation for Mamba FLOPs.")
                elif name == "MHSA":
                    flops = calculate_mhsa_flops(base_config, seq_len)
                    print(f"   - Using manual calculation for MHSA FLOPs.")
                elif name == "RNN (LSTM)": # <-- NEWLY ADDED
                    flops = calculate_rnn_flops(base_config, seq_len)
                    print(f"   - Using manual calculation for RNN (LSTM) FLOPs.")
                else:
                    # Fallback for any other models, though none are defined
                    flops = calculate_thop_flops(model, device, base_config, seq_len)
                    print(f"   - Using 'thop' library for FLOPs.")
                
                # Measure speed
                inference_metrics = measure_speed(model, device, base_config, batch_size, seq_len, mode='inference')
                training_metrics = measure_speed(model, device, base_config, batch_size, seq_len, mode='train')
                
                result = {
                    "Model": name,
                    "SeqLen": seq_len,
                    "Parameters (M)": round(total_params / 1e6, 2),
                    "FLOPs (M)": round(flops / 1e6, 2),
                    "Inference (PPS)": round(inference_metrics["pps"], 2),
                    "Inference (ms/batch)": round(inference_metrics["ms_per_batch"], 4),
                    "Train (ms/batch)": round(training_metrics["ms_per_batch"], 4)
                }
                
                all_results.append(result)
                
                print(f"   - Params: {result['Parameters (M)']:.2f}M")
                print(f"   - FLOPs: {result['FLOPs (M)']:.2f}M")
                print(f"   - Inference: {result['Inference (PPS)']:.0f} PPS, {result['Inference (ms/batch)']:.4f} ms/batch")
                print(f"   - Training: {result['Train (ms/batch)']:.4f} ms/batch")
                
                # Clean up to free memory
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ERROR: Failed to benchmark {name} at SeqLen={seq_len}: {e}")
                continue
    
    # ... (The rest of the main function remains the same)
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = output_dir / f"benchmark_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {csv_filename}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    
    # Create and save plots
    plot_filename = output_dir / f"benchmark_plots_{timestamp}.png"
    plot_results(df, plot_filename)
    


if __name__ == "__main__":
    main()
