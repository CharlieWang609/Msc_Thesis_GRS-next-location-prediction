import argparse
import numpy as np
import random
import torch
import os
import glob
import pandas as pd
import datetime
import yaml
from easydict import EasyDict as edict
import sys

# Ensure other necessary imports from your project are present
from mobpredict.utils import (
    prepare_nn_dataset_train,
    prepare_nn_dataset_inference,
    get_train_vali_loaders,
    get_inference_loader,
)
from mobpredict.train import init_save_path, get_models, get_test_result, get_trained_nets

def load_config(path):
    """
    Loads config file and flattens its structure.
    Example: loss: {type: "focal"} becomes config.type = "focal" (after edict).
    Key collisions between different sections will result in overwrites.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) # cfg is a nested dict e.g. {'misc': {...}, 'loss': {...}}
    
    flat_config = {}
    for section_name, section_values in cfg.items(): # e.g., section_name='misc', section_values={...}
        if isinstance(section_values, dict):
            for key, value in section_values.items(): # e.g., key='run_name', value='dtepr_lstm'
                if key in flat_config and flat_config[key] != value: 
                    print(f"Warning: Config key '{key}' from section '{section_name}' (value: {value}) "
                          f"overwrites a previously set value ({flat_config[key]}). Using new value.")
                flat_config[key] = value
        else:
            # Handles cases where a top-level YAML key is not a dictionary itself.
            if section_name in flat_config and flat_config[section_name] != section_values:
                 print(f"Warning: Config key '{section_name}' (top-level, value: {section_values}) "
                       f"overwrites a previously set value ({flat_config[section_name]}). Using new value.")
            flat_config[section_name] = section_values # This should not happen with your current config.yml structure
            
    return flat_config # Return a flat dictionary

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_run(config, device, log_dir):
    result_ls = []
    # Using flat config: config.data_save_root, config.train_dataset
    train_path = os.path.join(config.data_save_root, "temp", config.train_dataset + "_trained_train.pk")
    vali_path = os.path.join(config.data_save_root, "temp", config.train_dataset + "_trained_validation.pk")
    
    if not (os.path.exists(train_path) and os.path.exists(vali_path)):
        print(f"Error: Train/Val .pk not found for '{config.train_dataset}'. Expected: {train_path}, {vali_path}")
        return []

    train_loader, val_loader = get_train_vali_loaders(config, train_path=train_path, vali_path=vali_path)
    model = get_models(config, device) # get_models must now expect flat config
    best_model, perf = get_trained_nets(config, model, train_loader, val_loader, device, log_dir)
    result_ls.append(perf)

    test_path = os.path.join(config.data_save_root, "temp", config.train_dataset + "_trained_test.pk")
    if not os.path.exists(test_path):
        print(f"Warning: Test .pk not found for '{config.train_dataset}'. Skipping test.")
        return result_ls 
        
    test_loader = get_inference_loader(config, path=test_path)
    perf = get_test_result(config, best_model, test_loader, device,
                           save_results=True, save_dir=log_dir,
                           dataset_name=f"{config.train_dataset}_test")
    result_ls.append(perf)
    return result_ls

def inference_run(config, device, log_dir):
    # Using flat config: config.data_save_root, config.run_save_root, config.pretrain_dir
    temp_data_dir = os.path.join(config.data_save_root, "temp")
    all_pk_files = glob.glob(os.path.join(temp_data_dir, "*.pk"))
    if not all_pk_files:
        print(f"Warning: No .pk files in temp dir: {temp_data_dir} for inference.")
        return []

    model = get_models(config, device) # Expects flat config
    result_ls = []

    pretrain_model_path = os.path.join(config.run_save_root, config.pretrain_dir, "checkpoint.pt")
    if not os.path.exists(pretrain_model_path):
        print(f"Error: Pretrained model not found: {pretrain_model_path}")
        return []
    model.load_state_dict(torch.load(pretrain_model_path, map_location=device))

    for file_path in all_pk_files:
        print("=" * 50)
        filename_base = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Inferencing on: {filename_base}")
        if filename_base.endswith("_train"):
            print(f"  Skipping {filename_base} (training split).")
            continue

        loader = get_inference_loader(config, file_path)
        perf = get_test_result(config, model, loader, device,
                               save_results=True, save_dir=log_dir, 
                               dataset_name=filename_base)
        perf["dataset"] = filename_base
        result_ls.append(perf)
    return result_ls

def load_data_transform(sp_df):
    # ... (This function does not directly use the main 'config' object, so it's likely fine) ...
    # Ensure this function is correctly defined as in previous versions.
    def _transfer_time_to_absolute(df, start_time):
        duration_list = df["duration"].tolist()
        processed_duration = []
        for item in duration_list:
            try: processed_duration.append(float(item))
            except (ValueError, TypeError): processed_duration.append(0.0)
        duration_arr = processed_duration[:-1]; duration_arr.insert(0, 0.0)
        try: cumsum_duration = np.cumsum(duration_arr)
        except TypeError as e: raise ValueError(f"Error during cumsum of durations: {duration_arr}. Details: {e}")
        df["started_at"] = np.array([datetime.timedelta(hours=float(i)) for i in cumsum_duration]) + start_time
        df["finished_at"] = df["started_at"] + pd.to_timedelta(df["duration"], unit="hours")
        min_day = pd.to_datetime(df["started_at"].min().date())
        df["start_day"] = (df["started_at"] - min_day).dt.days
        df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
        df["weekday"] = df["started_at"].dt.weekday
        df["duration"] = (df["duration"] * 60).round().astype(int)
        return df
    if "user_id" not in sp_df.columns:
        if sp_df.index.name == "user_id": sp_df = sp_df.reset_index()
        else: raise ValueError("'user_id' column required.")
    if sp_df.index.name is not None and sp_df.index.name != "id": sp_df = sp_df.reset_index()
    sp_df.index.name = "id"
    if 'duration' not in sp_df.columns: raise ValueError("'duration' column missing.")
    sp_df['duration'] = pd.to_numeric(sp_df['duration'], errors='coerce').fillna(0)
    sp_df = sp_df.groupby("user_id", as_index=False, group_keys=False).apply(
        _transfer_time_to_absolute, start_time=datetime.datetime(2023, 1, 1, hour=8, tzinfo=None)
    )
    return sp_df


def load_and_preprocess_data_for_config(config_obj): # config_obj is flat here
    dataset_csv_path = os.path.join(config_obj.data_save_root, f"{config_obj.train_dataset}.csv")
    if not os.path.exists(dataset_csv_path):
        print(f"Error: Dataset CSV not found: {dataset_csv_path}"); return None
    try: sp = pd.read_csv(dataset_csv_path, index_col="index")
    except Exception as e: print(f"Error reading CSV {dataset_csv_path}: {e}"); return None
    sp = load_data_transform(sp)
    return sp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training or inference using one or more config files.")
    
    parser.add_argument("config_files",
                        type=str,
                        nargs='+',
                        help="Path(s) to config file(s), for example configs/prediction/config_lstm_ce.yml.")
    
    parser.add_argument("--seed", 
                        type=int, 
                        default=3407, 
                        help="Global random seed.")
    args = parser.parse_args()

    if args.seed is not None: print(f"Setting global seed: {args.seed}"); setup_seed(args.seed)
    else: print("Global seed not specified.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")

    for config_path in args.config_files:
        print(f"\n{'='*25} Processing: {os.path.basename(config_path)} {'='*25}")
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}. Skipping."); continue

        try:
            current_config_dict = load_config(config_path) # Loads as a flat dict
            config = edict(current_config_dict)          # Converts to flat EasyDict

            # Ensure essential keys (now top-level) are present
            required_keys = ['run_save_root', 'run_name', 'data_save_root', 'train_dataset']
            missing_keys = [key for key in required_keys if not hasattr(config, key)]
            if missing_keys:
                print(f"Error: Config {config_path} missing essential keys: {', '.join(missing_keys)}. Skipping."); continue
            
            log_dir = init_save_path(config) # init_save_path must expect flat config
            print(f"Log directory: {log_dir}")

            if config.training: # Assuming 'training' is from 'misc' section, now top-level
                print("Mode: Training")
                sp_df = load_and_preprocess_data_for_config(config)
                if sp_df is None: print(f"Skipping training for {config_path} due to data error."); continue

                max_locations, max_users = prepare_nn_dataset_train(sp_df,
                    train_name=config.train_dataset, save_root=config.data_save_root)
                config.total_loc_num = int(max_locations + 1) # Added to top-level config
                config.total_user_num = int(max_users + 1)   # Added to top-level config

                results_list = train_run(config, device, log_dir)
                if results_list:
                    result_df = pd.DataFrame(results_list)
                    # Assumes networkName is from 'model' section, now top-level config.networkName
                    results_filename = os.path.join(log_dir, f"{config.train_dataset}_{config.networkName}_train_results.csv")
                    result_df.to_csv(results_filename, index=False)
                    print(f"Training results for '{config.run_name}' saved: {results_filename}")
                else: print(f"No training results for: {config.run_name}")

            else: # Inference
                print("Mode: Inference")
                base_sp_df = load_and_preprocess_data_for_config(config) # Uses flat config.train_dataset
                if base_sp_df is None: print(f"Skipping inference for {config_path} (base data error)."); continue
                
                inf_sps_list, inf_filename_list = [], []
                # Assumes inference_data_dir is from 'misc' section, now top-level config.inference_data_dir
                inference_target_dir = os.path.join(config.data_save_root, config.inference_data_dir)
                if not os.path.isdir(inference_target_dir):
                    print(f"Error: Inference dir not found: {inference_target_dir}"); continue

                for inf_file_path in glob.glob(os.path.join(inference_target_dir, "*.csv")):
                    try: inf_sp_df_single = pd.read_csv(inf_file_path, index_col="index")
                    except Exception as e: print(f"Error reading {inf_file_path}: {e}. Skipping."); continue
                    inf_sp_df_single = load_data_transform(inf_sp_df_single)
                    inf_sps_list.append(inf_sp_df_single)
                    inf_filename_list.append(os.path.splitext(os.path.basename(inf_file_path))[0])
                
                if not inf_sps_list: print(f"No valid inference CSVs in {inference_target_dir}. Skipping.")
                else:
                    max_locations, max_users = prepare_nn_dataset_inference(inf_sps_list, base_sp_df,
                        save_root=config.data_save_root, inference_names=inf_filename_list,
                        train_name=config.train_dataset)
                    config.total_loc_num = int(max_locations + 1)
                    config.total_user_num = int(max_users + 1)

                    performance_results = inference_run(config, device, log_dir)
                    if performance_results:
                        pd.DataFrame(performance_results).to_csv(
                            os.path.join(log_dir, f"inference_{config.networkName}_results.csv"), index=False)
                        print(f"Inference results for '{config.run_name}' saved.")
            print(f"--- Finished: {os.path.basename(config_path)} ---")
        except Exception as e:
            print(f"!!! CRITICAL ERROR processing {config_path}: {e} !!!")
            import traceback; traceback.print_exc()
            print(f"--- Skipping to next config ---"); continue 
    print(f"\n{'='*25} All configurations processed. {'='*25}")
