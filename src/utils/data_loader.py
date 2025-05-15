import pandas as pd
import os
import numpy as np

from src.utils.consts import SOC_PATH


def load_partition(client_id, num_clients=632, data_dir=SOC_PATH):
    """Carrega os dados para um cliente específico."""
    files = sorted(os.listdir(data_dir))

    files = [f for f in files if f.endswith('.csv')]

    if client_id >= num_clients or client_id >= len(files):
        raise ValueError(
            f"Client {client_id} exceeds number of available batteries")

    csv_file = os.path.join(data_dir, files[client_id])

    df = pd.read_csv(csv_file).drop(columns=["timestamp"])
    X = df[["v", "c", "t"]].to_numpy()
    y = df["soc"].to_numpy()
    return X, y
