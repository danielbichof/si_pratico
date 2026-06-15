import gzip
import time
from pickle import dump
from pprint import pprint

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from wine_data import (
    MODEL_PATH,
    carregar_dados,
    preparar_atributos,
    separar_atributos_classes,
)


def aplicar_smote_se_possivel(atributos_train, classes_train):
    menor_classe = classes_train.value_counts().min()
    if menor_classe < 2:
        print("SMOTE nao aplicado: ha classe com menos de 2 exemplos no treino.")
        return atributos_train, classes_train

    k_neighbors = min(5, menor_classe - 1)
    print("Aplicando SMOTE no treino com k_neighbors:", k_neighbors)
    balancer = SMOTE(random_state=42, k_neighbors=k_neighbors)
    return balancer.fit_resample(atributos_train, classes_train)


dados = carregar_dados()
dados_atributos, dados_classes, coluna_classe = separar_atributos_classes(dados)
dados_preparados, scaler = preparar_atributos(dados_atributos, ajustar=True)

print("Coluna classe:", coluna_classe)
print("Total de instancias:", dados.shape[0])
print("Total de atributos apos preparacao:", dados_preparados.shape[1])
print("Frequencia das classes:")
print(dados_classes.value_counts().sort_index())

atributos_train, atributos_test, classes_train, classes_test = train_test_split(
    dados_preparados,
    dados_classes,
    test_size=0.3,
    stratify=dados_classes,
    random_state=42,
)

atributos_train, classes_train = aplicar_smote_se_possivel(
    atributos_train,
    classes_train,
)
print("Frequencia das classes no treino:")
print(classes_train.value_counts().sort_index())

experimentos = [
    (
        "RandomForest",
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {
            "n_estimators": [100, 160, 220],
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_depth": [None, 10, 20, 40],
            "max_features": ["sqrt", "log2"],
        },
    ),
    (
        "GradientBoosting",
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [80, 120, 180],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.8, 1.0],
        },
    ),
    (
        "KNeighbors",
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    ),
]

resultados = []

for nome, estimador, grid in experimentos:
    inicio = time.perf_counter()
    busca = RandomizedSearchCV(
        estimator=estimador,
        param_distributions=grid,
        n_iter=8,
        scoring="f1_macro",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    busca.fit(atributos_train, classes_train)
    duracao = time.perf_counter() - inicio

    modelo = busca.best_estimator_
    classes_preditas = modelo.predict(atributos_test)
    f1_macro = f1_score(classes_test, classes_preditas, average="macro")
    acuracia = accuracy_score(classes_test, classes_preditas)

    resultados.append(
        {
            "algoritmo": nome,
            "busca": busca,
            "modelo": modelo,
            "classes_preditas": classes_preditas,
            "f1_macro": f1_macro,
            "acuracia": acuracia,
            "tempo": duracao,
        }
    )

    print("Algoritmo:", nome)
    print("Melhores hiperparametros:")
    pprint(busca.best_params_)
    print("Tempo de treino:", f"{duracao:.2f}s")
    print("Acuracia:", acuracia)
    print("F1 macro:", f1_macro)

melhor = max(resultados, key=lambda item: item["f1_macro"])
classes_preditas = melhor["classes_preditas"]

print("Melhor algoritmo:", melhor["algoritmo"])
print(
    "Justificativa: escolhido pelo maior f1_macro no teste; a acuracia global foi "
    f"{melhor['acuracia']:.4f}, o f1_macro foi {melhor['f1_macro']:.4f} e o tempo "
    f"de treino foi {melhor['tempo']:.2f}s. O f1_macro foi priorizado porque a "
    "qualidade do vinho possui classes desbalanceadas e a estabilidade por classe "
    "e mais importante que apenas a acuracia global."
)
print("Matriz de confusao:")
print(confusion_matrix(classes_test, classes_preditas))
print("Acuracia:", accuracy_score(classes_test, classes_preditas))
print("Precisao macro:", precision_score(classes_test, classes_preditas, average="macro"))
print("Recall macro:", recall_score(classes_test, classes_preditas, average="macro"))
print("F1 macro:", f1_score(classes_test, classes_preditas, average="macro"))
print("Relatorio de classificacao:")
print(classification_report(classes_test, classes_preditas))

artefato = {
    "modelo": melhor["modelo"],
    "coluna_classe": coluna_classe,
    "colunas_treinamento": list(dados_preparados.columns),
    "scaler": scaler,
    "classes": list(melhor["modelo"].classes_),
    "algoritmo": melhor["algoritmo"],
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
dump(artefato, gzip.open(MODEL_PATH, "wb"))
print("Modelo salvo em:", MODEL_PATH)
