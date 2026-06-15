import gzip
import sys
from pickle import dump
from pprint import pprint

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from blackfriday_data import (
    MODEL_PATH,
    TARGETS,
    carregar_dados,
    preparar_atributos,
    separar_atributos_classes,
)


def especificidade_por_classe(classes_reais, classes_preditas, classes):
    matriz = confusion_matrix(classes_reais, classes_preditas, labels=classes)
    total = matriz.sum()
    especificidades = {}

    for indice, classe in enumerate(classes):
        tp = matriz[indice, indice]
        fp = matriz[:, indice].sum() - tp
        fn = matriz[indice, :].sum() - tp
        tn = total - tp - fp - fn
        especificidades[classe] = tn / (tn + fp) if (tn + fp) else 0

    return matriz, especificidades


try:
    dados = carregar_dados()
except FileNotFoundError as erro:
    print(erro)
    sys.exit(1)
targets_disponiveis = [target for target in TARGETS if target in dados.columns]

if not targets_disponiveis:
    raise ValueError(
        "Nenhum alvo esperado foi encontrado. Alvos esperados: "
        + ", ".join(TARGETS)
    )

artefato = {}

for target in targets_disponiveis:
    print("=" * 70)
    print("Treinando alvo:", target)
    dados_target = dados.dropna(subset=[target]).copy()
    dados_atributos, dados_classes, coluna_alvo = separar_atributos_classes(
        dados_target,
        target,
    )
    dados_preparados = preparar_atributos(dados_atributos)

    print("Total de instancias:", dados_target.shape[0])
    print("Total de atributos apos preparacao:", dados_preparados.shape[1])
    print("Frequencia das classes:")
    print(dados_classes.value_counts().sort_index())

    menor_classe = dados_classes.value_counts().min()
    if menor_classe < 2:
        print("Alvo ignorado: ha classe com menos de 2 exemplos.")
        continue

    estratificar = dados_classes if dados_classes.value_counts().min() >= 2 else None
    atributos_train, atributos_test, classes_train, classes_test = train_test_split(
        dados_preparados,
        dados_classes,
        test_size=0.3,
        stratify=estratificar,
        random_state=42,
    )

    rf_grid = {
        "n_estimators": [80, 120, 180],
        "criterion": ["gini", "entropy", "log_loss"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_depth": [None, 10, 20, 40],
        "max_features": ["sqrt", "log2"],
    }
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    cv = min(3, classes_train.value_counts().min())
    busca = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_grid,
        n_iter=8,
        scoring="f1_macro",
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    busca.fit(atributos_train, classes_train)

    print("Melhores hiperparametros:")
    pprint(busca.best_params_)

    modelo = busca.best_estimator_
    classes_preditas = modelo.predict(atributos_test)
    classes = list(modelo.classes_)
    matriz, especificidades = especificidade_por_classe(
        classes_test,
        classes_preditas,
        classes,
    )

    print("Matriz de confusao:")
    print(matriz)
    print("Acuracia:", accuracy_score(classes_test, classes_preditas))
    print("Recall macro/sensibilidade:", recall_score(classes_test, classes_preditas, average="macro"))
    print("F1 macro:", f1_score(classes_test, classes_preditas, average="macro"))
    print("Especificidade por classe:")
    for classe, especificidade in especificidades.items():
        print(f"Classe {classe}: {especificidade:.4f}")
    print("Especificidade macro:", np.mean(list(especificidades.values())))
    print("Relatorio de classificacao:")
    print(classification_report(classes_test, classes_preditas))

    artefato[target] = {
        "modelo": modelo,
        "coluna_alvo": coluna_alvo,
        "colunas_treinamento": list(dados_preparados.columns),
        "classes": classes,
    }

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
dump(artefato, gzip.open(MODEL_PATH, "wb"))
print("Modelo salvo em:", MODEL_PATH)
