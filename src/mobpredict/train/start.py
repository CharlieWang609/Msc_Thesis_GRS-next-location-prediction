import os

import pickle as pickle
import datetime
import json

from mobpredict.train import train_net, single_test, get_performance_dict
from mobpredict.networks import RNNs, TransEncoder




def get_trained_nets(config, model, train_loader, val_loader, device, log_dir):
    best_model, perf = train_net(config, model, train_loader, val_loader, device, log_dir=log_dir)
    perf["type"] = "vali"
    return best_model, perf


def get_test_result(config, best_model, test_loader, device, save_results=False, save_dir=None, dataset_name=None):

    return_dict = single_test(
        config, 
        best_model, 
        test_loader, 
        device, 
        save_results=save_results, 
        save_dir=save_dir, 
        dataset_name=dataset_name
    )
    performance = get_performance_dict(return_dict)
    performance["type"] = "test"

    return performance



def get_models(config, device): # config is now flat
    # networkName, base_emb_size etc. are now directly attributes of config
    # All __init__ methods of model classes (TransEncoder, RNNs, MambaEncoder,
    # AllEmbedding, FullyConnected) must also expect a flat config object.
    if config.networkName == "mhsa":
        model = TransEncoder(config=config).to(device)
    elif config.networkName == "rnn":
        model = RNNs(config=config).to(device)
    elif config.networkName == "mamba":
        try:
            from mobpredict.networks.mamba import MambaEncoder
        except ImportError as exc:
            raise ImportError(
                "The Mamba model requires the optional 'mamba' dependencies. "
                "Install them with: pip install -e '.[mamba]'"
            ) from exc
        model = MambaEncoder(config=config).to(device)
    else:
        raise ValueError(f"Unknown network name: {config.networkName}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Ensure config.networkName exists from the flattened config (it should if from 'model' section)
    print(f"Model: {getattr(config, 'networkName', 'Unknown')}, Trainable Params: {total_params}")
    return model

def init_save_path(config): # config is now flat
    # run_save_root and run_name are now directly attributes of config (from 'misc' section)
    if not hasattr(config, 'run_save_root') or not hasattr(config, 'run_name'):
        raise AttributeError("Config object must have 'run_save_root' and 'run_name' attributes.")
    
    log_dir = os.path.join(config.run_save_root, config.run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    try:
        with open(os.path.join(log_dir, "conf.json"), "w") as fp:
            # Convert EasyDict to dict for clean JSON dump
            json.dump(dict(config), fp, indent=4, sort_keys=True)
    except Exception as e:
        print(f"Warning: Could not save conf.json in {log_dir}. Error: {e}")
    return log_dir
