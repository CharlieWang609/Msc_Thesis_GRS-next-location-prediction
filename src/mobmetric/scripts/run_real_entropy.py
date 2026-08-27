import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import stats 
from mobmetric.entropy import real_entropy 
import pickle       # Added for saving/loading intermediate results
import tempfile     # Added for creating a temporary directory
import shutil       # Added for cleaning up the temporary directory

def setup_seed(seed):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)

def load_processed_data(csv_path):
    """
    Loads mobility data from a CSV file, simplified for entropy calculation.
    """
    print(f"    Loading {csv_path} for entropy calculation...")
    try:
        sp = pd.read_csv(csv_path, index_col="index")
    except FileNotFoundError:
        print(f"    Error: File not found {csv_path}")
        raise
    except Exception as e:
        print(f"    Error reading CSV {csv_path}: {e}")
        raise

    required_cols = ["user_id", "location_id"]
    for col in required_cols:
        if col not in sp.columns:
            raise ValueError(f"    Required column '{col}' for entropy calculation not found in {csv_path}.")
    print(f"    Finished loading {csv_path}. DataFrame shape: {sp.shape}.")
    return sp

def main(input_folder, output_folder, seed, sample_frac):
    """
    Main function to calculate and plot real entropy distributions.
    Saves intermediate results to disk to conserve memory.
    """
    setup_seed(seed)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output directory: {output_folder}")

    # Create a temporary directory to store intermediate results
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for intermediate results: {temp_dir}")

    try:
        # --- File Order Definition (Method 3) ---
        desired_filenames_in_order = [
            "benchmark(0.18).csv", 
            "0.1.csv",
            "0.25.csv",
            "0.5.csv",
            "0.75.csv",
            "0.9.csv",
        ]
        print(f"Using predefined file order: {desired_filenames_in_order}")
        
        csv_files_in_defined_order = [os.path.join(input_folder, fname) for fname in desired_filenames_in_order]
        
        actual_csv_files_to_process = []
        for fpath in csv_files_in_defined_order:
            if os.path.isfile(fpath):
                actual_csv_files_to_process.append(fpath)
            else:
                print(f"Warning: File '{os.path.basename(fpath)}' from predefined order not found in '{input_folder}'. Skipping.")
        
        csv_files = actual_csv_files_to_process
        
        if not csv_files:
            print(f"No CSV files found based on the predefined order in '{input_folder}'. Exiting.")
            return

        print(f"Will process {len(csv_files)} files in the specified order: {[os.path.basename(f) for f in csv_files]}")
        
        # --- Processing Loop: Calculate and Save Intermediate Results ---
        for csv_file_path in csv_files:
            dataset_name = os.path.splitext(os.path.basename(csv_file_path))[0]
            print(f"\nProcessing dataset: {dataset_name}...")

            try:
                sps = load_processed_data(csv_file_path)
                
                if sps.empty:
                    print(f"  Skipping {dataset_name} as loaded data is empty.")
                    continue

                if sample_frac is not None and 0 < sample_frac < 1:
                    print(f"  Sampling {sample_frac * 100:.1f}% of users from {dataset_name}...")
                    all_users = sps['user_id'].unique()
                    if len(all_users) > 0:
                        num_users_to_sample = max(1, int(len(all_users) * sample_frac))
                        sampled_users = np.random.choice(all_users, size=num_users_to_sample, replace=False)
                        sps = sps[sps['user_id'].isin(sampled_users)]
                        print(f"  Proceeding with {len(sampled_users)} sampled users. New data shape: {sps.shape}")

                print(f"  Calculating real entropy for {dataset_name}...")
                real_e_series = real_entropy(sps, print_progress=False, n_jobs=-1)
                
                mean_entropy = np.nan
                if real_e_series is not None and not real_e_series.empty:
                    mean_entropy = real_e_series.mean()
                print(f"    Mean Real Entropy: {mean_entropy:.3f}")
                
                # Create results dictionary for the current dataset
                current_dataset_results = {
                    "dataset": dataset_name,
                    "real_entropy_series": real_e_series,
                    "mean_real_entropy": mean_entropy,
                }

                # Save the results dictionary to a pickle file in the temp directory
                temp_filepath = os.path.join(temp_dir, f"{dataset_name}_results.pkl")
                with open(temp_filepath, 'wb') as f:
                    pickle.dump(current_dataset_results, f)
                print(f"  Intermediate results for {dataset_name} saved.")

            except Exception as e:
                print(f"  An error occurred while processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()

        # --- Aggregation and Plotting ---
        print("\nAll datasets processed. Loading intermediate results for plotting...")
        all_results_data = []
        
        # Get the list of dataset names that were actually processed, in the correct order
        processed_dataset_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
        
        for dataset_name in processed_dataset_names:
            temp_filepath = os.path.join(temp_dir, f"{dataset_name}_results.pkl")
            if os.path.exists(temp_filepath):
                with open(temp_filepath, 'rb') as f:
                    loaded_result = pickle.load(f)
                    all_results_data.append(loaded_result)
            else:
                print(f"Warning: Could not find temporary result file for '{dataset_name}'. It will be skipped in the plot.")

        if not all_results_data:
            print("No valid results were loaded. Cannot generate plot.")
            return

        print("\nGenerating real entropy distribution plot (KDE)...")
        fig_entropy, ax_entropy = plt.subplots(figsize=(10, 7))
        max_entropy_val = 0
        valid_entropy_series_count = 0
        
        for result in all_results_data:
            series = result["real_entropy_series"]
            if series is not None and not series.empty:
                series_finite = series[np.isfinite(series)]
                if not series_finite.empty:
                    current_max = series_finite.max()
                    if pd.notna(current_max) and current_max > max_entropy_val:
                        max_entropy_val = current_max
                    valid_entropy_series_count += 1
        
        if valid_entropy_series_count > 0:
            x_entropy = np.linspace(0, max_entropy_val + 0.2, 200)
            
            for result in all_results_data:
                series = result["real_entropy_series"]
                dataset_name = result["dataset"]
                if series is not None and not series.empty:
                    series_finite = series[np.isfinite(series)]
                    if not series_finite.empty and len(series_finite) > 1:
                        try:
                            kde = stats.gaussian_kde(series_finite)
                            ax_entropy.plot(x_entropy, kde(x_entropy), label=dataset_name)
                        except Exception as e:
                            print(f"Could not plot KDE for entropy of {dataset_name}: {e}")
                    elif len(series_finite) == 1:
                         print(f"Skipping KDE for entropy of {dataset_name} (only 1 finite data point).")

            ax_entropy.set_xlabel('Entropy')
            ax_entropy.set_ylabel('Probability Density (PDF)')
            ax_entropy.set_title('Distribution of Entropy', fontsize=15)
            ax_entropy.legend(title=r'$\mu|_{\gamma}$') # title =[r'$\mu|_{\rho}$' or r'$P_{new}$' or r'$\mu|_{\gamma}$'
            ax_entropy.grid(True, axis='y', linestyle='--', alpha=0.6)
            fig_entropy.tight_layout()
            entropy_plot_path = os.path.join(output_folder, "combined_real_entropy_distribution.png")

            plt.savefig(entropy_plot_path, dpi=300)
            print(f"Real entropy distribution plot saved to {entropy_plot_path}")
        else:
            print("No valid real entropy data to plot.")
        plt.close(fig_entropy)
        
        print("\nScript finished.")

    finally:
        # --- Cleanup: Remove the temporary directory and its contents ---
        if os.path.exists(temp_dir):
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and compare distributions of mobility real entropy for multiple datasets.")
    parser.add_argument("--input_folder", type=str, 
                        default="data/metrics/input",
                        help="Path to the folder containing input CSV files.")
    parser.add_argument("--output_folder", type=str, 
                        default="outputs/metrics",
                        help="Path to the folder where output plots will be saved.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--sample_frac", type=float, default=1,
                        )
    
    args = parser.parse_args()
    
    main(args.input_folder, args.output_folder, args.seed, args.sample_frac)
