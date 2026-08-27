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
from mobmetric.motifs import mobility_motifs

def setup_seed(seed):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)

def _get_motifs_proportion_for_user_group(df_user_group):
    """
    Calculates the proportion of days classified as motifs for a user's data.
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
    Calculates the overall proportion of user-days classified as motifs.
    """
    if df_overall_motifs.empty or "class" not in df_overall_motifs.columns:
        return 0.0
    return (len(df_overall_motifs["class"]) - df_overall_motifs["class"].isna().sum()) / len(df_overall_motifs["class"])

def load_processed_data(csv_path, time_format="relative"):
    """
    Loads and preprocesses mobility data from a CSV file.
    """
    print(f"    Loading and preprocessing {csv_path} with time_format='{time_format}'")

    sp = pd.read_csv(csv_path, index_col="index")

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

    sp["geometry"] = sp["geometry"].apply(wkt.loads)
    sp = gpd.GeoDataFrame(sp, geometry="geometry", crs="EPSG:4326")

    if "user_id" not in sp.columns and "user_id" == sp.index.name:
        sp.reset_index(inplace=True)

    if time_format == "absolute":
        if not all(c in sp.columns for c in ["started_at", "finished_at"]):
            raise ValueError("Columns 'started_at' and 'finished_at' are missing for absolute time format.")
        sp["started_at"] = pd.to_datetime(sp["started_at"], format="mixed", yearfirst=True, utc=True)
        sp["finished_at"] = pd.to_datetime(sp["finished_at"], format="mixed", yearfirst=True, utc=True)
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
            "user_motif_proportion_series": None,
            "overall_motif_proportion": np.nan
        }

        sps = load_processed_data(csv_file_path, time_format)

        if sps.empty:
            print(f"    Skipping {dataset_name} as loaded data is empty.")
            all_results_data.append(current_dataset_results)
            continue
        if not all(col in sps.columns for col in ["user_id", "location_id", "started_at", "finished_at"]):
            raise ValueError(f"DataFrame for {dataset_name} is missing required columns after processing. Found: {sps.columns.tolist()}")

        print(f"    Calculating motif proportion for {dataset_name}...")
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

    if not all_results_data:
        print("No data was processed. Cannot generate plots.")
        return

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
                    kde = stats.gaussian_kde(series)
                    ax_motif.plot(x_motif, kde(x_motif), label=dataset_name)
                elif len(series) == 1:
                    print(f"Skipping KDE for motif proportions of {dataset_name} (only 1 data point). Value: {series.iloc[0]}")

        ax_motif.set_xlabel('User Motif Proportion')
        ax_motif.set_ylabel('Probability Density (PDF)')
        ax_motif.legend(title=r'model')# title =[r'$\mu|_{\rho}$' or r'$P_{new}$' or r'$\mu|_{\gamma}$']
        ax_motif.set_xlim(-0.05, 1.05)
        ax_motif.grid(True, axis='y', linestyle='--', alpha=0.6)
        fig_motif.tight_layout()
        motif_plot_path = os.path.join(output_folder, "combined_user_motif_proportion_distribution.png")
        plt.savefig(motif_plot_path, dpi=300)
        print(f"User motif proportion distribution plot saved to {motif_plot_path}")
    else:
        print("No valid user motif proportion data to plot.")
    plt.close(fig_motif)

    print("\nScript finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and compare mobility motifs for multiple datasets.")
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
