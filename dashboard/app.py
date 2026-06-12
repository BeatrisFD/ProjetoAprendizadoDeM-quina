import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(BASE_DIR, "app")

sys.path.append(APP_DIR)

from predict import predict_transaction


st.set_page_config(
    page_title="Sistema de Detecção de Fraudes",
    page_icon="💳",
    layout="wide"
)


fraud_example = {
    "time": 406.0,
    "v1": -2.3122265423263,
    "v2": 1.95199201064158,
    "v3": -1.60985073229769,
    "v4": 3.9979055875468,
    "v5": -0.522187864667764,
    "v6": -1.42654531920595,
    "v7": -2.53738730624579,
    "v8": 1.39165724829804,
    "v9": -2.77008927719433,
    "v10": -2.77227214465915,
    "v11": 3.20203320709635,
    "v12": -2.89990738849473,
    "v13": -0.595221881324605,
    "v14": -4.28925378244217,
    "v15": 0.389724120274487,
    "v16": -1.14074717980657,
    "v17": -2.83005567450437,
    "v18": -0.0168224681808257,
    "v19": 0.416955705037907,
    "v20": 0.126910559061474,
    "v21": 0.517232370861764,
    "v22": -0.0350493686052974,
    "v23": -0.465211076182388,
    "v24": 0.320198198514526,
    "v25": 0.0445191674731724,
    "v26": 0.177839798284401,
    "v27": 0.261145002567677,
    "v28": -0.143275874698919,
    "amount": 0.0
}


normal_example = {
    "time": 10000.0,
    "v1": -1.359807,
    "v2": -0.072781,
    "v3": 2.536346,
    "v4": 1.378155,
    "v5": -0.338321,
    "v6": 0.462388,
    "v7": 0.239599,
    "v8": 0.098698,
    "v9": 0.363787,
    "v10": 0.090794,
    "v11": -0.551600,
    "v12": -0.617801,
    "v13": -0.991390,
    "v14": -0.311169,
    "v15": 1.468177,
    "v16": -0.470401,
    "v17": 0.207971,
    "v18": 0.025791,
    "v19": 0.403993,
    "v20": 0.251412,
    "v21": -0.018307,
    "v22": 0.277838,
    "v23": -0.110474,
    "v24": 0.066928,
    "v25": 0.128539,
    "v26": -0.189115,
    "v27": 0.133558,
    "v28": -0.021053,
    "amount": 149.62
}


results = pd.DataFrame([
    ["Sem Balanceamento", "DecisionTree", 0.9079, 0.7753, 0.8364],
    ["Sem Balanceamento", "RandomForest", 0.9221, 0.7978, 0.8554],
    ["Sem Balanceamento", "XGBoost", 0.9375, 0.8427, 0.8876],
    ["SMOTE", "DecisionTree", 0.0999, 0.8367, 0.1785],
    ["SMOTE", "RandomForest", 0.8602, 0.8163, 0.8377],
    ["SMOTE", "XGBoost", 0.4602, 0.8265, 0.5912]
], columns=[
    "Experimento",
    "Modelo",
    "Precision",
    "Recall",
    "F1"
])


st.title("💳 Sistema de Detecção de Fraudes")

tab1, tab2 = st.tabs([
    "Predição",
    "Comparação dos Modelos"
])


with tab1:
    st.header("Análise de Transação")

    model_option = st.selectbox(
        "Modelo",
        ["DecisionTree", "RandomForest", "XGBoost"],
        key="select_model"
    )

    example = st.radio(
        "Exemplo",
        ["Fraude Real", "Transação Normal"],
        key="radio_example"
    )

    if example == "Fraude Real":
        transaction = fraud_example
    else:
        transaction = normal_example

    st.info(
        "Os valores abaixo são exemplos sugeridos, mas podem ser alterados antes da análise."
    )

    edited_transaction = {}

    st.subheader("Dados principais")

    col1, col2 = st.columns(2)

    with col1:
        edited_transaction["time"] = st.number_input(
            "time",
            value=float(transaction["time"]),
            key=f"input_time_{example}"
        )

    with col2:
        edited_transaction["amount"] = st.number_input(
            "amount",
            value=float(transaction["amount"]),
            key=f"input_amount_{example}"
        )

    st.subheader("Variáveis V1 até V28")

    cols = st.columns(4)

    for i in range(1, 29):
        field_name = f"v{i}"

        with cols[(i - 1) % 4]:
            edited_transaction[field_name] = st.number_input(
                field_name,
                value=float(transaction[field_name]),
                format="%.6f",
                key=f"input_{field_name}_{example}"
            )

    if st.button("Analisar Transação", key="btn_analisar_transacao"):
        model_path = os.path.join(
            BASE_DIR,
            "models",
            f"SEM_BALANCEAMENTO_{model_option}.pkl"
        )

        if not os.path.exists(model_path):
            st.error(f"Modelo não encontrado: {model_path}")
        else:
            result = predict_transaction(
                edited_transaction,
                model_path
            )

            prediction = result["fraud"]
            probability = result["risk_score"]

            st.divider()

            if prediction == 1:
                st.error("🚨 FRAUDE DETECTADA")
            else:
                st.success("✅ TRANSAÇÃO LEGÍTIMA")

            st.metric(
                "Probabilidade de Fraude",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

            if probability < 0.30:
                st.success("🟢 Baixo Risco")
            elif probability < 0.70:
                st.warning("🟡 Médio Risco")
            else:
                st.error("🔴 Alto Risco")


with tab2:
    st.header("Comparação dos Modelos")

    st.dataframe(
        results,
        use_container_width=True
    )

    fig1 = px.bar(
        results,
        x="Modelo",
        y="F1",
        color="Experimento",
        barmode="group",
        title="Comparação por F1-Score"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.scatter(
        results,
        x="Recall",
        y="Precision",
        size="F1",
        color="Modelo",
        hover_name="Experimento",
        title="Precision x Recall"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    best = results.loc[
        results["F1"].idxmax()
    ]

    st.success(
        f"""
Melhor Modelo

Modelo: {best['Modelo']}

Experimento: {best['Experimento']}

F1-Score: {best['F1']}
"""
    )