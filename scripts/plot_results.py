import matplotlib.pyplot as plt
import pickle
import os
import sys

from src.utils import MODELS_PATH

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":

    if not os.path.exists(f"{MODELS_PATH}/global_model.pkl"):
        print(
            f"Erro: Arquivo do modelo não encontrado em {MODELS_PATH}/global_model.pkl")
        sys.exit(1)

    with open(f"{MODELS_PATH}/global_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Coeficientes do modelo treinado:")
    print("Coef:", model["coef"])
    print("Intercept:", model["intercept"])

    # Salvar informações em um arquivo de texto
    os.makedirs("outputs/plots", exist_ok=True)
    with open("outputs/plots/model_info.txt", "w") as f:
        f.write(f"Coeficientes: {model['coef']}\n")
        f.write(f"Intercepto: {model['intercept']}\n")

    print(f"Informações do modelo salvas em outputs/plots/model_info.txt")
