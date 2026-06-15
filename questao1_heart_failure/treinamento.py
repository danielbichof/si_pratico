import gzip
from pickle import dump
from pprint import pprint

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from heart_data import (
    COLUNAS_BINARIAS,
    COLUNAS_NUMERICAS,
    MODEL_PATH,
    carregar_dados,
    preparar_atributos,
    separar_atributos_classes,
)


dados = carregar_dados()
dados_atributos, dados_classes, coluna_classe = separar_atributos_classes(dados)
dados_preparados, scaler = preparar_atributos(dados_atributos, ajustar=True)

print("Coluna classe:", coluna_classe)
print("Total de instancias:", dados.shape[0])
print("Total de atributos apos preparacao:", dados_preparados.shape[1])
print("Frequencia das classes:")
print(dados_classes.value_counts().sort_index())

melhor_k = None
melhor_score = -1
melhor_kmeans = None

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    grupos = kmeans.fit_predict(dados_preparados)
    score = silhouette_score(dados_preparados, grupos)
    print(f"Silhouette k={k}: {score:.4f}")

    if score > melhor_score:
        melhor_k = k
        melhor_score = score
        melhor_kmeans = kmeans

dados_preparados["grupo"] = melhor_kmeans.predict(dados_preparados)
print("Melhor k:", melhor_k)
print("Melhor silhouette:", f"{melhor_score:.4f}")
print("Frequencia dos grupos:")
print(dados_preparados["grupo"].value_counts().sort_index())

atributos_train, atributos_test, classes_train, classes_test = train_test_split(
    dados_preparados,
    dados_classes,
    test_size=0.3,
    stratify=dados_classes,
    random_state=42,
)

rf_grid = {
    "n_estimators": [80, 120, 180, 240],
    "criterion": ["gini", "entropy", "log_loss"],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_depth": [None, 5, 10, 20, 40],
    "max_features": ["sqrt", "log2"],
}

rf = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_hiperparameters = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=10,
    scoring="f1_macro",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)
rf_hiperparameters.fit(atributos_train, classes_train)

print("Melhores hiperparametros:")
pprint(rf_hiperparameters.best_params_)

modelo_classificador = rf_hiperparameters.best_estimator_
classes_preditas = modelo_classificador.predict(atributos_test)

print("Matriz de confusao:")
print(confusion_matrix(classes_test, classes_preditas))
print("Acuracia:", accuracy_score(classes_test, classes_preditas))
print("Precisao macro:", precision_score(classes_test, classes_preditas, average="macro"))
print("Recall macro:", recall_score(classes_test, classes_preditas, average="macro"))
print("F1 macro:", f1_score(classes_test, classes_preditas, average="macro"))
print("Relatorio de classificacao:")
print(classification_report(classes_test, classes_preditas))

artefato = {
    "modelo_kmeans": melhor_kmeans,
    "scaler": scaler,
    "modelo_classificador": modelo_classificador,
    "coluna_classe": coluna_classe,
    "colunas_treinamento": list(dados_preparados.columns),
    "classes": list(modelo_classificador.classes_),
    "COLUNAS_BINARIAS": COLUNAS_BINARIAS,
    "COLUNAS_NUMERICAS": COLUNAS_NUMERICAS,
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
dump(artefato, gzip.open(MODEL_PATH, "wb"))
print("Modelo salvo em:", MODEL_PATH)
