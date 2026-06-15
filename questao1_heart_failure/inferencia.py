import argparse
import gzip
import json
from pickle import load

import numpy as np
import pandas as pd

from heart_data import MODEL_PATH, preparar_nova_instancia


DEFAULT_INSTANCE = {
    "age": 60,
    "anaemia": 0,
    "creatinine_phosphokinase": 250,
    "diabetes": 1,
    "ejection_fraction": 35,
    "high_blood_pressure": 1,
    "platelets": 250000,
    "serum_creatinine": 1.2,
    "serum_sodium": 137,
    "sex": 1,
    "smoking": 0,
    "time": 120,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia de risco em Heart Failure.")
    parser.add_argument("--instance", help="Objeto JSON com os atributos do paciente.")
    return parser.parse_args()


def carregar_instancia(args):
    if args.instance:
        return json.loads(args.instance)

    return DEFAULT_INSTANCE


args = parse_args()
artefato = load(gzip.open(MODEL_PATH, "rb"))

nova_instancia = pd.DataFrame([carregar_instancia(args)])
nova_instancia = preparar_nova_instancia(
    nova_instancia,
    artefato["colunas_treinamento"],
    artefato["scaler"],
)

atributos_sem_grupo = nova_instancia.drop(columns=["grupo"], errors="ignore")
grupo_kmeans = artefato["modelo_kmeans"].predict(atributos_sem_grupo)[0]
nova_instancia["grupo"] = grupo_kmeans

classe_predita = artefato["modelo_classificador"].predict(nova_instancia)[0]
probabilidades = artefato["modelo_classificador"].predict_proba(nova_instancia)[0]
distancias = np.linalg.norm(
    artefato["modelo_kmeans"].cluster_centers_ - atributos_sem_grupo.iloc[0].to_numpy(),
    axis=1,
)

print("Grupo KMeans da instancia:", grupo_kmeans)
print("Classe predita:", classe_predita)
print("Distribuicao de probabilidade:")
for classe, probabilidade in zip(artefato["classes"], probabilidades):
    print(f"Classe {classe}: {probabilidade:.4f}")

print("Distancia euclidiana aos centroides:")
for grupo, distancia in enumerate(distancias):
    print(f"Grupo {grupo}: {distancia:.4f}")
