from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "heart_failure_clinical_records_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_failure.pkl"

COLUNA_CLASSE = "DEATH_EVENT"
COLUNAS_BINARIAS = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
]
COLUNAS_NUMERICAS = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]


def carregar_dados():
    return pd.read_csv(DATASET_PATH)


def separar_atributos_classes(dados):
    dados_atributos = dados.drop(columns=[COLUNA_CLASSE])
    dados_classes = dados[COLUNA_CLASSE]
    return dados_atributos, dados_classes, COLUNA_CLASSE


def preparar_atributos(dados_atributos, scaler=None, ajustar=False):
    dados_atributos = dados_atributos.copy()
    dados_preparados = dados_atributos[COLUNAS_BINARIAS].astype(int).copy()

    if scaler is None:
        scaler = StandardScaler()

    if ajustar:
        numericas = scaler.fit_transform(dados_atributos[COLUNAS_NUMERICAS])
    else:
        numericas = scaler.transform(dados_atributos[COLUNAS_NUMERICAS])

    numericas = pd.DataFrame(
        numericas,
        columns=COLUNAS_NUMERICAS,
        index=dados_atributos.index,
    )

    dados_preparados = pd.concat([numericas, dados_preparados], axis=1)
    return dados_preparados, scaler


def preparar_nova_instancia(nova_instancia, colunas_treinamento, scaler):
    nova_instancia = nova_instancia.copy()

    for coluna in COLUNAS_NUMERICAS + COLUNAS_BINARIAS:
        if coluna not in nova_instancia.columns:
            nova_instancia[coluna] = 0

    nova_instancia = nova_instancia[COLUNAS_NUMERICAS + COLUNAS_BINARIAS]
    nova_instancia, _ = preparar_atributos(nova_instancia, scaler=scaler)
    return nova_instancia.reindex(columns=colunas_treinamento, fill_value=0)
