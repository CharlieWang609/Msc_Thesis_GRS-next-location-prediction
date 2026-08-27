
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy import stats 
from mobmetric.entropy import real_entropy 
import pickle
import tempfile
import shutil

def setup_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

def load_processed_data(csv_path):
    """Loads a CSV file and verifies it has the required columns for entropy calculation."""
    print(f"    Loading {csv_path}...")
    sp = pd.read_csv(csv_path, index_col="index")
    required_cols = ["user_id", "location_id"]
    for col in required_cols:
        if col not in sp.columns:
            raise ValueError(f"    Required column '{col}' not found in {csv_path}.")
    print(f"    Finished loading {csv_path}. DataFrame shape: {sp.shape}.")
    return sp

def main(input_folder, output_folder, seed, sample_frac, truncate, head_percent):
    """
    Main script execution. Calculates and plots real entropy distributions for multiple datasets,
    saving intermediate results to disk to conserve memory.
    """
    setup_seed(seed)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output directory: {output_folder}")

    # Create a temporary directory that will be automatically cleaned up.
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Define the desired processing order of the input files.
        # desired_filenames_in_order = [
        #     "benchmark(0.64).csv", 
        #     "0.1.csv",
        #     "0.25.csv",
        #     "0.5.csv",
        #     "0.75.csv",
        #     "0.9.csv",
        # ]
        desired_filenames_in_order = [
            "epr.csv", 
            "depr.csv",
            "ipt.csv",
            "dtepr.csv"
        ]
        print(f"Using predefined file order: {desired_filenames_in_order}")
        
        # Construct full file paths and filter for files that actually exist.
        csv_files_in_defined_order = [os.path.join(input_folder, fname) for fname in desired_filenames_in_order]
        csv_files = [fpath for fpath in csv_files_in_defined_order if os.path.isfile(fpath)]
        
        # Warn about any missing files from the defined order.
        found_files_basenames = {os.path.basename(f) for f in csv_files}
        missing_files = [f for f in desired_filenames_in_order if f not in found_files_basenames]
        if missing_files:
            print(f"Warning: The following files from the predefined order were not found and will be skipped: {missing_files}")

        if not csv_files:
            print(f"No valid CSV files found based on the predefined order. Exiting.")
            return

        print(f"Will process {len(csv_files)} files: {[os.path.basename(f) for f in csv_files]}")
        
        # --- Processing Loop: Calculate and Save Intermediate Results ---
        for csv_file_path in csv_files:
            dataset_name = os.path.splitext(os.path.basename(csv_file_path))[0]
            print(f"\nProcessing dataset: {dataset_name}...")

            sps = load_processed_data(csv_file_path)
            
            if sps.empty:
                print(f"  Skipping {dataset_name} as loaded data is empty.")
                continue

            # --- Data Reduction Steps (applied sequentially) ---
            if sample_frac:
                print(f"  Sampling {sample_frac * 100:.1f}% of users...")
                all_users = sps['user_id'].unique()
                if len(all_users) > 0:
                    num_users_to_sample = max(1, int(len(all_users) * sample_frac))
                    sampled_users = np.random.choice(all_users, size=num_users_to_sample, replace=False)
                    sps = sps[sps['user_id'].isin(sampled_users)]

            if head_percent:
                print(f"  Keeping the first {head_percent}% of records for each user...")
                sps = sps.groupby('user_id', group_keys=False).apply(
                    lambda df: df.head(int(np.ceil(len(df) * (head_percent / 100.0))))
                )

            if truncate:
                print(f"  Truncating sequences to a maximum of {truncate} records per user...")
                sps = sps.groupby('user_id', group_keys=False).head(truncate)

            # --- Pre-calculation Diagnostics ---
            print("  Pre-calculation diagnostics:")
            if not sps.empty:
                user_sequence_lengths = sps.groupby('user_id').size()
                print(f"    - Rows to process: {len(sps)}, Users: {sps['user_id'].nunique()}")
                print(f"    - Max/Mean/Median sequence length: {user_sequence_lengths.max()}/{user_sequence_lengths.mean():.1f}/{user_sequence_lengths.median()}")
            else:
                print("    - No data to process after filtering.")
                continue
            
            # --- Real Entropy Calculation ---
            print(f"  Calculating real entropy...")
            real_e_series = real_entropy(sps, print_progress=True, n_jobs=-1)
            
            mean_entropy = real_e_series.mean() if real_e_series is not None else np.nan
            print(f"    Mean Real Entropy: {mean_entropy:.3f}")
            
            # Save results to a temporary pickle file.
            temp_filepath = os.path.join(temp_dir, f"{dataset_name}_results.pkl")
            with open(temp_filepath, 'wb') as f:
                pickle.dump({"dataset": dataset_name, "real_entropy_series": real_e_series}, f)
            print(f"  Intermediate results saved.")

        # --- Aggregation and Plotting ---
        print("\nAll datasets processed. Loading results for plotting...")
        all_results_data = []
        
        # Load results from temp files, respecting the original desired order.
        processed_dataset_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
        for dataset_name in processed_dataset_names:
            temp_filepath = os.path.join(temp_dir, f"{dataset_name}_results.pkl")
            if os.path.exists(temp_filepath):
                with open(temp_filepath, 'rb') as f:
                    all_results_data.append(pickle.load(f))
        
        if not all_results_data:
            print("No results were loaded. Cannot generate plot.")
            return

        # --- Plotting Real Entropy Distributions (KDE) ---
        print("Generating real entropy distribution plot...")
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Find the max entropy value across all datasets for a common x-axis.
        max_entropy_val = 0
        for result in all_results_data:
            series = result["real_entropy_series"]
            if series is not None and not series.empty:
                series_finite = series[np.isfinite(series)]
                if not series_finite.empty:
                    max_entropy_val = max(max_entropy_val, series_finite.max())

        x_entropy = np.linspace(0, max_entropy_val + 0.2, 200)
        
        # Plot the KDE curve for each dataset.
        for result in all_results_data:
            series = result["real_entropy_series"]
            if series is not None and not series.empty:
                series_finite = series[np.isfinite(series)]
                if len(series_finite) > 1: # KDE requires at least 2 points.
                    kde = stats.gaussian_kde(series_finite)
                    ax.plot(x_entropy, kde(x_entropy), label=result["dataset"])

        ax.set_xlabel('Real Entropy')
        ax.set_ylabel('Probability Density (PDF)')
        ax.set_title('Distribution of Entropy', fontsize=15)
        ax.legend(title=r'$\mu|_{\rho}$')# title =[r'$\mu|_{\rho}$' or r'$P_{new}$' or r'$\mu|_{\gamma}$']
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        fig.tight_layout()
        
        plot_path = os.path.join(output_folder, "combined_real_entropy_distribution.png")
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Plot saved to {plot_path}")

    finally:
        # This block ensures the temporary directory is removed, even if errors occur.
        if os.path.exists(temp_dir):
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    print("\nScript finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot real entropy distributions for mobility datasets.")
    parser.add_argument("--input_folder", type=str, default="data/metrics/input", help="Path to the folder containing input CSV files.")
    parser.add_argument("--output_folder", type=str, default="outputs/metrics", help="Path to the folder where output plots will be saved.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--sample_frac", type=float, help="Fraction of USERS to sample (e.g., 0.1 for 10%).")
    parser.add_argument("--truncate", type=int, help="Truncate sequences to this ABSOLUTE number of records per user.")
    parser.add_argument("--head_percent", type=float, help="Keep the first n PERCENT of records for each user (e.g., 10 for 10%%).")
    
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.seed, args.sample_frac, args.truncate, args.head_percent)
