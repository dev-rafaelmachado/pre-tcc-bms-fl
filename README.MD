## Sobre o Projeto

Este projeto utiliza aprendizagem federada para treinar um modelo de regressão que prevê o estado de carga de baterias (SOC) com base em características como voltagem (v), corrente (c) e temperatura (t). 

O processo funciona da seguinte forma:

1. Múltiplos clientes (representando diferentes baterias) treinam modelos localmente
2. Os clientes enviam apenas os parâmetros do modelo (não os dados) para o servidor
3. O servidor agrega os parâmetros e atualiza o modelo global
4. O servidor distribui o modelo global atualizado para os clientes
5. O processo se repete para várias rodadas de treinamento

## Requisitos de Dados

**IMPORTANTE:** A pasta `data/SOC` contendo os arquivos CSV das baterias **não** está incluída no repositório devido ao tamanho grande dos dados. O usuário precisa adicionar esta pasta manualmente:

1. Crie uma pasta chamada `data/SOC` no diretório raiz do projeto
2. Adicione seus arquivos CSV de dados de bateria nesta pasta
3. Cada arquivo CSV deve conter as colunas: `timestamp`, `v` (voltagem), `c` (corrente), `t` (temperatura) e `soc` (estado de carga)

## Requisitos de Software

```
pip install -r requirements.txt
```

O arquivo `requirements.txt` deve conter:

```
flwr==1.4.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

## Como Executar o Projeto

### Método Simples

O projeto inclui um script `run.py` que facilita a execução do sistema. Para iniciar o treinamento federado:

```bash
python -m scripts.run
```

Este comando inicia automaticamente:
1. Um servidor de aprendizagem federada
2. Dois clientes simulando baterias diferentes

### Execução Manual

Alternativamente, você pode executar os componentes separadamente:

1. Primeiro, inicie o servidor:
```bash
python -m src.federated.server
```

2. Em terminais separados, inicie os clientes:
```bash
python -m src.federated.client 0  # Cliente 0
python -m src.federated.client 1  # Cliente 1
# Continue para outros clientes conforme necessário
```

## Componentes do Projeto

### src/federated/server.py
Implementa o servidor de aprendizagem federada que:
- Agrega os parâmetros dos modelos dos clientes
- Avalia o desempenho do modelo global
- Salva o modelo global em `outputs/models/global_model.pkl`
- Gera um gráfico de evolução do erro em `outputs/plots/federated_loss_curve.png`

### src/federated/client.py
Implementa o cliente que:
- Carrega dados de uma bateria específica
- Pré-treina um modelo floresta aleatória para inicialização
- Usa um regressor SGD para aprendizagem incremental
- Comunica os parâmetros do modelo com o servidor

### src/models/battery_model.py
Contém implementações de modelos:
- RandomForestRegressor para pré-treinamento 
- DecisionTreeRegressor como alternativa

### src/utils/data_loader.py
Funções para carregamento e processamento de dados:
- Carrega partições de dados de baterias específicas
- Processa os dados brutos para treinamento

### scripts/run.py
Script para facilitar a execução do sistema completo, iniciando o servidor e múltiplos clientes em paralelo.

### scripts/plot_results.py
Ferramenta para visualizar informações sobre o modelo global treinado.

## Resultados

Após a execução, o sistema gera:
- Um modelo global salvo em `outputs/models/global_model.pkl`
- Um gráfico de evolução do erro em `outputs/plots/federated_loss_curve.png`
- Informações do modelo em `outputs/plots/model_info.txt`

## Personalização

- Ajuste o número de clientes no arquivo `scripts/run.py`
- Modifique os hiperparâmetros dos modelos em `src/models/battery_model.py`
- Altere o número de rodadas de treinamento no arquivo `src/federated/server.py`
- Personalize a função de carregamento de dados em `src/utils/data_loader.py`

## Configuração Inicial (Primeiros Passos)

Para configurar o projeto pela primeira vez:

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd federated-battery-soc
```

2. Crie as pastas necessárias:
```bash
mkdir -p data/SOC
mkdir -p outputs/models
mkdir -p outputs/plots
```

3. Adicione seus arquivos CSV na pasta data/SOC (necessário para execução)

4. Instale as dependências:
```bash
pip install -r requirements.txt
```

5. Execute o projeto:
```bash
python -m scripts.run
```
