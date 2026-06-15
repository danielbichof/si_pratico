import argparse
import gzip
import json
from pickle import load

import pandas as pd

from wine_data import MODEL_PATH, preparar_nova_instancia


DEFAULT_INSTANCE = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "tipo_vinho": "red",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia de qualidade do vinho.")
    parser.add_argument("--instance", help="Objeto JSON com os atributos do vinho.")
    return parser.parse_args()


def carregar_instancia(args):
    if args.instance:
        return json.loads(args.instance)

    return DEFAULT_INSTANCE


args = parse_args()
artefato = load(gzip.open(MODEL_PATH, "rb"))
modelo = artefato["modelo"]

nova_instancia = pd.DataFrame([carregar_instancia(args)])
nova_instancia = preparar_nova_instancia(
    nova_instancia,
    artefato["colunas_treinamento"],
    artefato["scaler"],
)

classe_predita = modelo.predict(nova_instancia)[0]
probabilidades = modelo.predict_proba(nova_instancia)[0]

print("Algoritmo:", artefato["algoritmo"])
print("Classe predita:", classe_predita)
print("Distribuicao de probabilidade:")
for classe, probabilidade in zip(artefato["classes"], probabilidades):
    print(f"Classe {classe}: {probabilidade:.4f}")
