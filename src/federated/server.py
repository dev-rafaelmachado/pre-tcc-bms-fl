import flwr as fl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error

from src.utils.consts import MODELS_PATH, PLOTS_PATH, ROUNDS, SOC_PATH

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

loss_history = []


def get_evaluate_fn(csv_path=f"{SOC_PATH}/c002-001.csv"):
    df = pd.read_csv(csv_path).drop(columns=["timestamp"])
    X = df[["v", "c", "t"]].to_numpy()
    y = df["soc"].to_numpy()

    def evaluate(server_round, parameters, config):
        coef, intercept = parameters
        preds = X @ coef + intercept
        loss = mean_squared_error(y, preds)
        print(
            f"\033[92m[SERVER] Global Eval MSE (Round {server_round}): {loss:.4f}\033[0m")
        loss_history.append((server_round, loss))
        return loss, {}

    return evaluate


def save_global_model(parameters):
    coef, intercept = parameters
    model = {"coef": coef, "intercept": intercept}
    with open(f"{MODELS_PATH}/global_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(
        f"\033[92m[SERVER] Global model saved to {MODELS_PATH}/global_model.pkl\033[0m")


def plot_loss_curve():
    rounds, losses = zip(*loss_history)
    plt.plot(rounds, losses, marker='o')
    plt.title("Global MSE vs Rounds (Federated)")
    plt.xlabel("Round")
    plt.ylabel("MSE")
    plt.grid()
    plt.savefig(f"{PLOTS_PATH}/federated_loss_curve.png")
    plt.show()


class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_params = super().aggregate_fit(rnd, results, failures)
        if aggregated_params is not None:
            save_global_model(aggregated_params)
        return aggregated_params


if __name__ == "__main__":
    strategy = CustomFedAvg(

        evaluate_fn=get_evaluate_fn(),
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )

    plot_loss_curve()
