from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "black_friday_sales.csv"
MODEL_PATH = BASE_DIR / "models" / "black_friday.pkl"

TARGETS = ["product_category", "payment_method", "age_group"]
IDENTIFICADORES = [
    "User_ID",
    "Product_ID",
    "user_id",
    "product_id",
    "User ID",
    "Product ID",
]

MAPEAMENTO_COLUNAS = {
    "Product_Category": "product_category",
    "Product_Category_1": "product_category",
    "Product Category": "product_category",
    "Category": "product_category",
    "Payment_Method": "payment_method",
    "Payment Method": "payment_method",
    "Age": "age_group",
    "Age_Group": "age_group",
    "Age Group": "age_group",
}


def carregar_dados():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "CSV Black Friday nao encontrado. Execute `python baixar_dados.py` "
            "na raiz de projeto_final com credenciais Kaggle configuradas ou "
            f"coloque o arquivo manualmente em: {DATASET_PATH}"
        )

    dados = pd.read_csv(DATASET_PATH)
    print("Colunas encontradas:")
    print(dados.columns)
    print("Tipos encontrados:")
    print(dados.dtypes)
    print("Quantidade de valores unicos:")
    print(dados.nunique())
    return normalizar_colunas(dados)


def normalizar_colunas(dados):
    dados = dados.copy()
    renomear = {
        coluna: MAPEAMENTO_COLUNAS[coluna]
        for coluna in dados.columns
        if coluna in MAPEAMENTO_COLUNAS
    }
    dados = dados.rename(columns=renomear)
    return dados


def limpar_categorias(dados):
    dados = dados.copy()
    colunas_categoricas = dados.select_dtypes(include=["object", "string"]).columns

    for coluna in colunas_categoricas:
        dados[coluna] = dados[coluna].astype(str).str.strip()

    return dados


def separar_atributos_classes(dados, coluna_alvo):
    dados = limpar_categorias(dados)
    colunas_remover = [
        coluna
        for coluna in TARGETS + IDENTIFICADORES + ["Purchase"]
        if coluna in dados.columns and coluna != coluna_alvo
    ]

    dados_atributos = dados.drop(columns=colunas_remover + [coluna_alvo])
    dados_classes = dados[coluna_alvo]
    return dados_atributos, dados_classes, coluna_alvo


def preparar_atributos(dados_atributos):
    dados_atributos = limpar_categorias(dados_atributos)
    return pd.get_dummies(dados_atributos, dtype=int)


def preparar_nova_instancia(nova_instancia, colunas_treinamento):
    nova_instancia = limpar_categorias(nova_instancia)
    colunas_remover = [
        coluna
        for coluna in TARGETS + IDENTIFICADORES + ["Purchase"]
        if coluna in nova_instancia.columns
    ]
    nova_instancia = nova_instancia.drop(columns=colunas_remover)
    nova_instancia = preparar_atributos(nova_instancia)
    return nova_instancia.reindex(columns=colunas_treinamento, fill_value=0)
