from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
RED_PATH = BASE_DIR / "winequality-red.csv"
WHITE_PATH = BASE_DIR / "winequality-white.csv"
COMBINED_PATH = BASE_DIR / "winequality_combined.csv"
MODEL_PATH = BASE_DIR / "models" / "wine_quality.pkl"

COLUNA_CLASSE = "quality"
COLUNAS_NUMERICAS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def carregar_dados():
    if COMBINED_PATH.exists():
        return pd.read_csv(COMBINED_PATH)

    red = pd.read_csv(RED_PATH, sep=";")
    red["tipo_vinho"] = "red"

    white = pd.read_csv(WHITE_PATH, sep=";")
    white["tipo_vinho"] = "white"

    dados = pd.concat([red, white], ignore_index=True)
    dados.to_csv(COMBINED_PATH, index=False)
    return dados


def limpar_categorias(dados):
    dados = dados.copy()
    colunas_categoricas = dados.select_dtypes(include=["object", "string"]).columns

    for coluna in colunas_categoricas:
        dados[coluna] = dados[coluna].astype(str).str.strip()

    return dados


def separar_atributos_classes(dados):
    dados = limpar_categorias(dados)
    dados_atributos = dados.drop(columns=[COLUNA_CLASSE])
    dados_classes = dados[COLUNA_CLASSE]
    return dados_atributos, dados_classes, COLUNA_CLASSE


def preparar_atributos(dados_atributos, scaler=None, ajustar=False):
    dados_atributos = limpar_categorias(dados_atributos)

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

    categoricas = pd.get_dummies(dados_atributos[["tipo_vinho"]], dtype=int)
    dados_preparados = pd.concat([numericas, categoricas], axis=1)
    return dados_preparados, scaler


def preparar_nova_instancia(nova_instancia, colunas_treinamento, scaler):
    nova_instancia = nova_instancia.copy()

    for coluna in COLUNAS_NUMERICAS:
        if coluna not in nova_instancia.columns:
            nova_instancia[coluna] = 0

    if "tipo_vinho" not in nova_instancia.columns:
        nova_instancia["tipo_vinho"] = "red"

    nova_instancia, _ = preparar_atributos(nova_instancia, scaler=scaler)
    return nova_instancia.reindex(columns=colunas_treinamento, fill_value=0)
