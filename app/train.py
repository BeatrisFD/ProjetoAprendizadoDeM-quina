import os
import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from merge_csv import merge_creditcard_files
from feature_engineering import process_data


# ==================================================
# CARREGAMENTO DOS DADOS
# ==================================================

csv_path = '../data/creditcard.csv'

if not os.path.exists(csv_path):

    print('Arquivo principal não encontrado.')
    print('Reconstruindo dataset...')

    merge_creditcard_files()

df = pd.read_csv(csv_path)

df.columns = df.columns.str.lower()

df = process_data(df)

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

os.makedirs('../models', exist_ok=True)

# ==================================================
# FUNÇÃO DE AVALIAÇÃO
# ==================================================

def run_experiment(experiment_name, models, X_train, y_train):

    print("\n")
    print("=" * 70)
    print(f"EXPERIMENTO: {experiment_name}")
    print("=" * 70)

    results = []

    for name, model in models.items():

        print("\n" + "=" * 50)
        print(f"Treinando: {name}")
        print("=" * 50)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        print(classification_report(y_test, predictions))

        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        results.append({
            "Model": name,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4)
        })

        model_path = (
            f"../models/"
            f"{experiment_name}_{name}.pkl"
        )

        joblib.dump(model, model_path)

        print(f"Modelo salvo em: {model_path}")

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 50)
    print("COMPARAÇÃO DOS MODELOS")
    print("=" * 50)

    print(results_df)

    best_model = results_df.sort_values(
        by="F1-Score",
        ascending=False
    ).iloc[0]

    print("\nMelhor modelo:")
    print(best_model)

    return results_df


# ==================================================
# EXPERIMENTO 1
# SEM BALANCEAMENTO
# ==================================================

models_no_balance = {

    "DecisionTree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
}

results_exp1 = run_experiment(
    "SEM_BALANCEAMENTO",
    models_no_balance,
    X_train,
    y_train
)


# ==================================================
# EXPERIMENTO 2
# COM SMOTE
# ==================================================

print("\nAplicando SMOTE...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print(y_train.value_counts())
print(y_train_smote.value_counts())

models_smote = {

    "DecisionTree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
}

results_exp2 = run_experiment(
    "SMOTE",
    models_smote,
    X_train_smote,
    y_train_smote
)


# ==================================================
# EXPERIMENTO 3
# CLASS WEIGHT / SCALE POS WEIGHT
# ==================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(
    f"\nscale_pos_weight calculado: "
    f"{scale_pos_weight:.2f}"
)

models_weighted = {

    "DecisionTree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
}

results_exp3 = run_experiment(
    "CLASS_WEIGHT",
    models_weighted,
    X_train,
    y_train
)


# ==================================================
# COMPARAÇÃO FINAL DOS EXPERIMENTOS
# ==================================================

print("\n")
print("=" * 70)
print("COMPARAÇÃO FINAL DOS EXPERIMENTOS")
print("=" * 70)

results_exp1["Experiment"] = "Sem Balanceamento"
results_exp2["Experiment"] = "SMOTE"
results_exp3["Experiment"] = "Class Weight"

final_results = pd.concat(
    [
        results_exp1,
        results_exp2,
        results_exp3
    ],
    ignore_index=True
)

final_results = final_results[
    [
        "Experiment",
        "Model",
        "Precision",
        "Recall",
        "F1-Score"
    ]
]

print(final_results)

best_result = final_results.sort_values(
    by="F1-Score",
    ascending=False
).iloc[0]

print("\n")
print("=" * 70)
print("MELHOR RESULTADO GERAL")
print("=" * 70)
print(best_result)