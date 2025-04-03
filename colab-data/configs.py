# general hyperparameters
HYPERPARAMETERS = {
    "lr": 1e-3,
    "bs": 32,
    "scheduler_patience": 5,
    "max_epochs": 100,
    "val_freq": 2,  # validation every n epochs
    "patience": 10,  # 10 epochs early stopping
}

PREDICTION_DATA_CONFIG = {
    "inputs": ["speeds"],
    "obs_len": 8,
    "pred_len": 12,
    "output": "speeds",
}

PREDICTION_NETWORK_CONFIG = {
    "hidden_units": [32, 16],
    "dropout": 0.2,
    "batch_norm": True,
    "activation": "prelu",
    "obs_len": PREDICTION_DATA_CONFIG["obs_len"],
    "pred_len": 12,
    "n_features": 3,
}

CLASSIFICATION_DATA_CONFIG = {
    "inputs": ["speeds"],
    "obs_len": 20,
    "pred_len": 0,
    "output": "speeds",
}

CLASSIFICATION_NETWORK_CONFIG = {
    "n_features": 3,
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 1,
    "dropout": 0.1,
    "obs_len": CLASSIFICATION_DATA_CONFIG["obs_len"],
}
