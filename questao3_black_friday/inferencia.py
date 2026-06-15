import argparse
import gzip
import json
import sys
from pickle import load

import pandas as pd

from blackfriday_data import MODEL_PATH, preparar_nova_instancia


DEFAULT_INSTANCE = {
    "Gender": "M",
    "Occupation": 10,
    "City_Category": "A",
    "Stay_In_Current_City_Years": "2",
    "Marital_Status": 0,
    "Product_Category_2": 8.0,
    "Product_Category_3": 14.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia Black Friday.")
    parser.add_argument("--instance", help="Objeto JSON com os atributos da compra.")
    return parser.parse_args()


def carregar_instancia(args):
    if args.instance:
        return json.loads(args.instance)

    return DEFAULT_INSTANCE


args = parse_args()
if not MODEL_PATH.exists():
    print(
        "Modelo Black Friday nao encontrado. Execute `python treinamento.py` "
        "depois de disponibilizar o CSV Kaggle."
    )
    sys.exit(1)

artefato = load(gzip.open(MODEL_PATH, "rb"))
instancia = pd.DataFrame([carregar_instancia(args)])

for target, info in artefato.items():
    modelo = info["modelo"]
    nova_instancia = preparar_nova_instancia(
        instancia,
        info["colunas_treinamento"],
    )
    classe_predita = modelo.predict(nova_instancia)[0]
    probabilidades = modelo.predict_proba(nova_instancia)[0]

    print("=" * 70)
    print("Alvo:", target)
    print("Classe predita:", classe_predita)
    print("Distribuicao de probabilidade:")
    for classe, probabilidade in zip(info["classes"], probabilidades):
        print(f"Classe {classe}: {probabilidade:.4f}")
