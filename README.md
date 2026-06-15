# Projeto Final - Sistemas Inteligentes Avancados

Este projeto contem tres questoes independentes de classificacao, seguindo o
mesmo estilo do exemplo de risco de default: modulo de dados, treinamento,
avaliacao, salvamento do artefato com `gzip + pickle` e inferencia por JSON.

## Estrutura

| Pasta | Base | Objetivo |
|---|---|---|
| `questao1_heart_failure` | UCI Heart Failure Clinical Records | Classificar `DEATH_EVENT` e informar grupo KMeans |
| `questao2_wine_quality` | UCI Wine Quality | Classificar a nota `quality` do vinho |
| `questao3_black_friday` | Kaggle Black Friday Sales | Classificar `product_category`, `payment_method` e `age_group` quando existirem |

## Preparacao

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Baixe as bases publicas da UCI e tente baixar a base do Kaggle:

```bash
python baixar_dados.py
```

O download do Kaggle depende de credenciais em `~/.kaggle/kaggle.json`. Se as
credenciais nao existirem, coloque manualmente o CSV em:

```text
questao3_black_friday/black_friday_sales.csv
```

## Questao 1 - Heart Failure

A base possui 299 instancias, atributos clinicos e a classe `DEATH_EVENT`.
As colunas numericas sao normalizadas com `StandardScaler`; colunas binarias
sao mantidas como 0/1. O treinamento testa KMeans com `k=2..5`, escolhe o
melhor agrupamento por `silhouette_score`, adiciona o grupo como atributo e
treina um `RandomForestClassifier` com `RandomizedSearchCV`.

Executar:

```bash
cd questao1_heart_failure
python treinamento.py
python inferencia.py
python inferencia.py --instance '{"age": 70, "anaemia": 1, "creatinine_phosphokinase": 300, "diabetes": 0, "ejection_fraction": 30, "high_blood_pressure": 1, "platelets": 220000, "serum_creatinine": 1.8, "serum_sodium": 134, "sex": 1, "smoking": 1, "time": 80}'
```

Artefato salvo:

```text
questao1_heart_failure/models/heart_failure.pkl
```

Resultado observado na validacao local:

| Metrica | Valor |
|---|---:|
| Melhor `k` KMeans | 2 |
| Silhouette | 0.1335 |
| Acuracia | 0.8444 |
| Precisao macro | 0.8384 |
| Recall macro | 0.7948 |
| F1 macro | 0.8107 |

## Questao 2 - Wine Quality

Os arquivos `winequality-red.csv` e `winequality-white.csv` sao combinados em
`winequality_combined.csv`, com a coluna adicional `tipo_vinho`. As variaveis
fisico-quimicas sao normalizadas e `tipo_vinho` e codificada com
`pd.get_dummies`.

O treinamento compara `RandomForestClassifier`, `GradientBoostingClassifier` e
`KNeighborsClassifier`, todos com `predict_proba`. A escolha usa `f1_macro`,
pois as classes de qualidade sao desbalanceadas.

Executar:

```bash
cd questao2_wine_quality
python treinamento.py
python inferencia.py
python inferencia.py --instance '{"fixed acidity": 6.8, "volatile acidity": 0.22, "citric acid": 0.36, "residual sugar": 8.1, "chlorides": 0.047, "free sulfur dioxide": 54, "total sulfur dioxide": 189, "density": 0.995, "pH": 3.13, "sulphates": 0.52, "alcohol": 10.2, "tipo_vinho": "white"}'
```

Artefato salvo:

```text
questao2_wine_quality/models/wine_quality.pkl
```

Resultado observado na validacao local:

| Algoritmo | Acuracia | F1 macro | Tempo |
|---|---:|---:|---:|
| RandomForest | 0.6513 | 0.3997 | 30.85s |
| GradientBoosting | 0.5826 | 0.3568 | 215.08s |
| KNeighbors | 0.5918 | 0.3726 | 1.77s |

O modelo salvo foi `RandomForest`, por apresentar o melhor `f1_macro`.

## Questao 3 - Black Friday

O modulo de dados imprime as colunas, tipos e cardinalidades encontrados no CSV
para facilitar ajustes. Ele tambem normaliza variacoes obvias de nomes para:

- `product_category`
- `payment_method`
- `age_group`

Para cada alvo disponivel, o treinamento remove os outros alvos,
identificadores e `Purchase`, prepara categorias com `pd.get_dummies`, treina um
`RandomForestClassifier` independente e avalia matriz de confusao, acuracia,
recall macro/sensibilidade, F1 macro, especificidade por classe,
especificidade macro e `classification_report`.

Executar:

```bash
cd questao3_black_friday
python treinamento.py
python inferencia.py
python inferencia.py --instance '{"Gender": "F", "Occupation": 4, "City_Category": "B", "Stay_In_Current_City_Years": "1", "Marital_Status": 1, "Product_Category_2": 5, "Product_Category_3": 12}'
```

Artefato salvo:

```text
questao3_black_friday/models/black_friday.pkl
```

## Observacoes de Execucao

- Os modelos ainda nao devem ser versionados obrigatoriamente; eles sao
  recriados pelos scripts de treinamento.
- A base Kaggle indicada no enunciado pode nao conter todos os tres alvos. O
  script treina os alvos encontrados e informa erro apenas se nenhum alvo
  esperado existir.
- A questao 3 nao foi treinada na validacao local porque o CSV Kaggle nao estava
  disponivel e a Kaggle CLI nao estava instalada.
