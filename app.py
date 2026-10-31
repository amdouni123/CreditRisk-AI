
import streamlit as st
import pandas as pd
import joblib
import json

# Configuration
st.set_page_config(
    page_title="CreditRisk AI",
    page_icon="🏦",
    layout="wide"
)

# Charger le modèle et les métadonnées
@st.cache_resource
def load_model():
    model = joblib.load("credit_risk_model.pkl")

    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    return model, metadata

model, metadata = load_model()

st.title("🏦 CreditRisk AI")
st.subheader("Intelligent Credit Scoring System")

st.write(
    "Cette application pédagogique estime la probabilité "
    "qu'un client appartienne à la classe des mauvais payeurs."
)

st.info(
    "Prototype académique basé sur le dataset German Credit. "
    "Il ne doit pas être utilisé pour prendre de vraies décisions bancaires."
)

st.divider()

st.header("1. Informations du client")

client_data = {}

with st.form("credit_form"):

    st.subheader("Caractéristiques numériques")

    numeric_cols = st.columns(2)

    for i, col in enumerate(metadata["numeric_features"]):

        with numeric_cols[i % 2]:

            minimum = metadata["numeric_min"][col]
            maximum = metadata["numeric_max"][col]
            default = metadata["numeric_defaults"][col]

            # Élargir légèrement les bornes pour permettre la saisie
            if minimum == maximum:
                minimum -= 1
                maximum += 1

            client_data[col] = st.number_input(
                col.replace("_", " ").title(),
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(default)
            )

    st.subheader("Caractéristiques catégorielles")

    categorical_cols = st.columns(2)

    for i, col in enumerate(metadata["categorical_features"]):

        with categorical_cols[i % 2]:

            options = metadata["categorical_options"][col]
            default = metadata["categorical_defaults"][col]

            default_index = options.index(default)

            client_data[col] = st.selectbox(
                col.replace("_", " ").title(),
                options=options,
                index=default_index
            )

    submitted = st.form_submit_button(
        "Analyser le risque de crédit",
        use_container_width=True
    )

if submitted:

    # Respecter l'ordre exact des variables d'entraînement
    client = pd.DataFrame(
        [[client_data[col] for col in metadata["feature_order"]]],
        columns=metadata["feature_order"]
    )

    # Prédiction
    probability = model.predict_proba(client)[0, 1]
    threshold = metadata["threshold"]
    prediction = int(probability >= threshold)

    st.divider()
    st.header("2. Résultat de l'analyse")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Probabilité estimée de défaut",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Seuil de classification",
            f"{threshold * 100:.0f}%"
        )

    st.progress(float(probability))

    if prediction == 1:
        st.error(
            "⚠️ Risque de défaut élevé selon le modèle"
        )
    else:
        st.success(
            "✅ Risque de défaut faible selon le modèle"
        )

    st.caption(
        "Cette classification dépend du seuil choisi. "
        "Elle constitue une estimation statistique et non une décision bancaire."
    )

st.divider()

st.caption(
    "CreditRisk AI | Projet académique de Machine Learning "
    "et de gestion du risque de crédit."
)
