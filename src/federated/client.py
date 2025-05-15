from src.models.battery_model import pretrain_random_forest
from src.utils.data_loader import load_partition

import flwr as fl
import numpy as np
import sys
import os
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Adicionar caminho do projeto
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


class SklearnClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, X, y):
        self.client_id = client_id
        self.model = model
        self.X = X
        self.y = y

        # Inicialização necessária
        n_features = X.shape[1]
        self.model.coef_ = np.zeros(n_features)
        self.model.intercept_ = np.array([0.0])
        self.model.n_iter_ = 1

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X, self.y)
        print(f"\033[94m[CLIENT {self.client_id}] Fit done\033[0m")
        return self.get_parameters(config={}), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        predictions = self.model.predict(self.X)
        loss = mean_squared_error(self.y, predictions)
        print(f"\033[96m[CLIENT {self.client_id}] Eval MSE: {loss:.4f}\033[0m")
        return loss, len(self.X), {}


def main(client_id: int):
    X, y = load_partition(client_id)

    rf_preds, residuals = pretrain_random_forest(X, y)
    # rf_preds, residuals = pretrain_decision_tree(X, y)
    print("Initializing SGD Regressor")
    model = SGDRegressor(max_iter=1, learning_rate="constant",
                         eta0=0.01, warm_start=True)
    client = SklearnClient(client_id, model, X, residuals)
    fl.client.start_numpy_client(
        server_address="localhost:8080", client=client)


if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(client_id)
