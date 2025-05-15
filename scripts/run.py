from src.utils import SOC_PATH, MODELS_PATH, PLOTS_PATH

import subprocess
import time
import sys
import os


if __name__ == "__main__":
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(PLOTS_PATH, exist_ok=True)

    if not os.path.exists(SOC_PATH) or len([f for f in os.listdir(SOC_PATH) if f.endswith('.csv')]) == 0:
        print(
            "\033[91mERRO: A pasta SOC_PATH não existe ou não contém arquivos CSV.\033[0m")
        print(
            f"Por favor, adicione os arquivos de dados na pasta {SOC_PATH} antes de executar.")
        sys.exit(1)

    print("\033[92mIniciando servidor de aprendizagem federada...\033[0m")
    server = subprocess.Popen(["python", "-m", "src.federated.server"])
    time.sleep(2)

    print("\033[92mIniciando clientes...\033[0m")
    clients = []
    for i in range(2):
        clients.append(subprocess.Popen(
            ["python", "-m", "src.federated.client", str(i)]))
        print(f"Cliente {i} iniciado.")

    print("\033[92mAguardando conclusão dos clientes...\033[0m")
    for c in clients:
        c.wait()

    print("\033[92mFinalizando servidor...\033[0m")
    server.terminate()

    print(
        "\033[92mProcesso concluído. Verifique os resultados na pasta 'outputs/'.\033[0m")
