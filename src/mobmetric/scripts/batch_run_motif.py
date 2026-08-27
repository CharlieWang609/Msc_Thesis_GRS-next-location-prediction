import argparse
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime
from shapely import wkt
import matplotlib.pyplot as plt
import random
from scipy import stats # For Kernel Density Estimation

# Assuming mobmetric package is installed and in PYTHONPATH
from mobmetric.entropy import random_entropy
from mobmetric.motifs import mobility_motifs

def setup_seed(seed):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)

def _get_motifs_proportion_for_user_group(df_user_group):
    """
    Calculates the proportion of days that are classified as motifs for a user's data.
    'df_user_group' is the sub-dataframe for a single user from sp_motifs_df.
    It's expected to have a 'class' column for motif classification of user-days.
    """
    if df_user_group.empty or "class" not in df_user_group.columns:
        return 0.0 
    
    motif_days = df_user_group["class"].notna().sum()
    total_days = len(df_user_group["class"])
    
    if total_days == 0:
        return 0.0 
    return motif_days / total_days

def _get_overall_motifs_proportion(df_overall_motifs):
    """
    Calculates the overall proportion of user-days that are classified as motifs in the entire dataset.
    'df_overall_motifs' is the sp_motifs_df for the whole dataset.
    """
    if df_overall_motifs.empty or "class" not in df_overall_motifs.columns:
        return 0.0
    return (len(df_overall_motifs["class"]) - df_overall_motifs["class"].isna().sum()) / len(df_overall_motifs["class"])


def load_processed_data(csv_path, time_format="relative"):
    """
    Loads and preprocesses mobility data from a CSV file.
    """
    print(f"    Loading and preprocessing {csv_path} with time_format='{time_format}'")
    try:
        sp = pd.read_csv(csv_path, index_col="index")
    except FileNotFoundError:
        print(f"    Error: File not found {csv_path}")
        raise
    except Exception as e:
        print(f"    Error reading CSV {csv_path}: {e}")
        raise

    base_required_cols = ["geometry", "user_id", "location_id"]
    if time_format == "absolute":
        specific_required_cols = base_required_cols + ["started_at", "finished_at"]
    elif time_format == "relative":
        specific_required_cols = base_required_cols + ["duration"]
    else:
        raise AttributeError(
            f"    time_format unknown: {time_format}. Supported: 'absolute', 'relative'."
        )

    for col in specific_required_cols:
        if col not in sp.columns:
            raise ValueError(f"    Required column '{col}' for time_format='{time_format}' not found in {csv_path}. Your CSV has columns: {sp.columns.tolist()}")

    try:
        sp["geometry"] = sp["geometry"].apply(wkt.loads)
        sp = gpd.GeoDataFrame(sp, geometry="geometry", crs="EPSG:4326")
    except Exception as e:
        print(f"    Error processing 'geometry' column for {csv_path}: {e}. Ensure it's valid WKT.")
        raise
        
    if "user_id" not in sp.columns and "user_id" == sp.index.name:
        sp.reset_index(inplace=True)

    if time_format == "absolute":
        try:
            if not all(c in sp.columns for c in ["started_at", "finished_at"]):
                 raise ValueError("Columns 'started_at' and 'finished_at' are missing for absolute time format.")
            sp["started_at"] = pd.to_datetime(sp["started_at"], format="mixed", yearfirst=True, utc=True)
            sp["finished_at"] = pd.to_datetime(sp["finished_at"], format="mixed", yearfirst=True, utc=True)
        except Exception as e:
            print(f"    Error converting time columns to datetime for absolute format in {csv_path}: {e}")
            raise
    elif time_format == "relative":
        if "duration" not in sp.columns:
            raise ValueError(f"    'duration' column (in hours) is required for 'relative' time_format in {csv_path}.")

        def _transfer_time_to_absolute(df_user, start_time_dt_utc):
            if df_user.empty:
                return df_user
            
            duration_list = df_user["duration"].to_list()
            if len(duration_list) == 1:
                cumsum_durations_hours = [0.0]
            else:
                adjusted_durations_for_cumsum = [0.0] + duration_list[:-1]
                cumsum_durations_hours = np.cumsum(adjusted_durations_for_cumsum)

            timedelta_arr = np.array([datetime.timedelta(hours=float(i)) for i in cumsum_durations_hours])
            
            df_user_copy = df_user.copy()
            calculated_started_at = [start_time_dt_utc + td for td in timedelta_arr]
            df_user_copy.loc[:, "started_at"] = calculated_started_at
            df_user_copy.loc[:, "finished_at"] = df_user_copy["started_at"] + pd.to_timedelta(df_user_copy["duration"], unit="h")
            return df_user_copy

        default_start_time_utc = datetime.datetime(2023, 1, 1, hour=8, tzinfo=datetime.timezone.utc)
        
        sp = sp.groupby("user_id", group_keys=False).apply(
            _transfer_time_to_absolute, start_time_dt_utc=default_start_time_utc
        )

    print(f"    Finished preprocessing for {csv_path}. DataFrame shape: {sp.shape}. Columns: {sp.columns.tolist()}")
    return sp

def main(input_folder, output_folder, proportion_filter, time_format, seed):
    setup_seed(seed)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output directory: {output_folder}")

    # --- file order ---
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
    

    csv_files = [os.path.join(input_folder, fname) for fname in desired_filenames_in_order]
    actual_csv_files_to_process = []
    for fpath in csv_files:
        if os.path.isfile(fpath):
            actual_csv_files_to_process.append(fpath)
        else:
            print(f"Warning: File '{os.path.basename(fpath)}' from predefined order not found in '{input_folder}'. Skipping.")
    
    csv_files = actual_csv_files_to_process


    if not csv_files:
        print(f"No CSV files found based on the predefined order in '{input_folder}' or none of the specified files exist. Exiting.")
        return

    print(f"Will process {len(csv_files)} files in the specified order: {[os.path.basename(f) for f in csv_files]}")
    all_results_data = [] 

    for csv_file_path in csv_files:
        dataset_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        print(f"\nProcessing dataset: {dataset_name}...")

        current_dataset_results = {
            "dataset": dataset_name,
            "random_entropy_series": None,
            "user_motif_proportion_series": None,
            "mean_random_entropy": np.nan, 
            "overall_motif_proportion": np.nan 
        }

        try:
            sps = load_processed_data(csv_file_path, time_format)
            
            if sps.empty:
                print(f"  Skipping {dataset_name} as loaded data is empty.")
                all_results_data.append(current_dataset_results) 
                continue
            if not all(col in sps.columns for col in ["user_id", "location_id", "started_at", "finished_at"]):
                raise ValueError(f"DataFrame for {dataset_name} is missing required columns after processing. Found: {sps.columns.tolist()}")

            print(f"  Calculating random entropy for {dataset_name}...")
            rand_e_series = random_entropy(sps, print_progress=False)
            current_dataset_results["random_entropy_series"] = rand_e_series
            if not rand_e_series.empty:
                current_dataset_results["mean_random_entropy"] = rand_e_series.mean()
            print(f"    Mean Random Entropy: {current_dataset_results['mean_random_entropy']:.3f}")

            print(f"  Calculating motif proportion for {dataset_name}...")
            sp_motifs_df = mobility_motifs(sps, proportion_filter=proportion_filter)
            
            if not sp_motifs_df.empty:
                user_motif_prop_series = sp_motifs_df.groupby("user_id").apply(_get_motifs_proportion_for_user_group)
                current_dataset_results["user_motif_proportion_series"] = user_motif_prop_series
                current_dataset_results["overall_motif_proportion"] = _get_overall_motifs_proportion(sp_motifs_df)
            else: 
                 current_dataset_results["user_motif_proportion_series"] = pd.Series(dtype=float)

            print(f"    Overall Motif Proportion: {current_dataset_results['overall_motif_proportion']:.3f}")
            if current_dataset_results["user_motif_proportion_series"] is not None and not current_dataset_results["user_motif_proportion_series"].empty:
                 print(f"    Mean of User Motif Proportions: {current_dataset_results['user_motif_proportion_series'].mean():.3f}")

            all_results_data.append(current_dataset_results)

        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results_data.append(current_dataset_results) 

    if not all_results_data:
        print("No data was processed. Cannot generate plots.")
        return

    print("\nGenerating random entropy distribution plot (KDE)...")
    fig_entropy, ax_entropy = plt.subplots(figsize=(10, 7))
    max_entropy_val = 0
    valid_entropy_series_count = 0
    for result in all_results_data:
        series = result["random_entropy_series"]
        if series is not None and not series.empty:
            series = series[np.isfinite(series)]
            if not series.empty:
                if series.max() > max_entropy_val: # Check if series.max() is valid
                    max_entropy_val = series.max() if pd.notna(series.max()) else max_entropy_val
                valid_entropy_series_count +=1

    if valid_entropy_series_count > 0:
        x_entropy = np.linspace(0, max_entropy_val + 0.2, 200) 
        for result in all_results_data:
            series = result["random_entropy_series"]
            dataset_name = result["dataset"]
            if series is not None and not series.empty:
                series = series[np.isfinite(series)] 
                if not series.empty and len(series) > 1: 
                    try:
                        kde = stats.gaussian_kde(series)
                        ax_entropy.plot(x_entropy, kde(x_entropy), label=dataset_name)
                    except Exception as e:
                        print(f"Could not plot KDE for entropy of {dataset_name}: {e}")
                elif len(series) == 1:
                     print(f"Skipping KDE for entropy of {dataset_name} (only 1 data point). Value: {series.iloc[0]}")

        ax_entropy.set_xlabel('Random Entropy')
        ax_entropy.set_ylabel('Probability Density (PDF)')
        ax_entropy.set_title('Distribution of Random Entropy by Dataset', fontsize=15)
        ax_entropy.legend(title="Dataset")
        ax_entropy.grid(True, axis='y', linestyle='--', alpha=0.6)
        fig_entropy.tight_layout()
        entropy_plot_path = os.path.join(output_folder, "combined_random_entropy_distribution.png")
        try:
            plt.savefig(entropy_plot_path, dpi=300)
            print(f"Random entropy distribution plot saved to {entropy_plot_path}")
        except Exception as e:
            print(f"Error saving random entropy distribution plot: {e}")
    else:
        print("No valid random entropy data to plot.")
    plt.close(fig_entropy)

    print("\nGenerating user motif proportion distribution plot (KDE)...")
    fig_motif, ax_motif = plt.subplots(figsize=(10, 7))
    valid_motif_series_count = 0
    for result in all_results_data:
        series = result["user_motif_proportion_series"]
        if series is not None and not series.empty:
            series = series[np.isfinite(series)] 
            if not series.empty: 
                valid_motif_series_count +=1
    
    if valid_motif_series_count > 0:
        x_motif = np.linspace(0, 1, 100) 
        for result in all_results_data:
            series = result["user_motif_proportion_series"]
            dataset_name = result["dataset"]
            if series is not None and not series.empty:
                series = series[np.isfinite(series)] 
                if not series.empty and len(series) > 1: 
                    try:
                        kde = stats.gaussian_kde(series)
                        ax_motif.plot(x_motif, kde(x_motif), label=dataset_name)
                    except Exception as e:
                        print(f"Could not plot KDE for motif proportions of {dataset_name}: {e}")
                elif len(series) == 1: 
                    print(f"Skipping KDE for motif proportions of {dataset_name} (only 1 data point). Value: {series.iloc[0]}")

        ax_motif.set_xlabel('User Motif Proportion')
        ax_motif.set_ylabel('Probability Density (PDF)')
        ax_motif.legend(title=r'model')# title =[r'$\mu|_{\rho}$' or r'$P_{new}$' or r'$\mu|_{\gamma}$']
        ax_motif.set_xlim(-0.05, 1.05) 
        ax_motif.grid(True, axis='y', linestyle='--', alpha=0.6)
        fig_motif.tight_layout()
        motif_plot_path = os.path.join(output_folder, "combined_user_motif_proportion_distribution.png")
        try:
            plt.savefig(motif_plot_path, dpi=300)
            print(f"User motif proportion distribution plot saved to {motif_plot_path}")
        except Exception as e:
            print(f"Error saving user motif proportion distribution plot: {e}")
    else:
        print("No valid user motif proportion data to plot.")
    plt.close(fig_motif)
    
    print("\nScript finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and compare mobility random_entropy and motifs for multiple datasets.")
    parser.add_argument("--input_folder", type=str, 
                        default="data/metrics/input"
                        )
    parser.add_argument("--output_folder", type=str, 
                        default="outputs/metrics"
                       )
    parser.add_argument("--proportion_filter", type=float, 
                        default=0.005,
                        )
    parser.add_argument("--time_format", type=str, 
                        default="relative", choices=["absolute", "relative"],
                        )
    parser.add_argument("--seed", type=int, 
                        default=42,
                        )
    
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.proportion_filter, args.time_format, args.seed)
