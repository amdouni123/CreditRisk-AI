from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc

# -----------------------------------------------------------------------------
# CreditRisk AI | Application Streamlit
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CreditRisk AI | Credit Scoring & Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette and small presentation helpers
NAVY = "#14213D"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#D97706"
RED = "#DC2626"
GREEN = "#15803D"

FEATURE_LABELS = {
    "checking_status": "Statut du compte courant",
    "duration": "Durée du crédit (mois)",
    "credit_history": "Historique de crédit",
    "purpose": "Objet du crédit",
    "credit_amount": "Montant du crédit",
    "savings_status": "Épargne",
    "employment": "Ancienneté professionnelle",
    "installment_commitment": "Taux de mensualité (%)",
    "personal_status": "Situation personnelle",
    "other_parties": "Autres parties au contrat",
    "residence_since": "Ancienneté dans le logement",
    "property_magnitude": "Patrimoine",
    "age": "Âge (années)",
    "other_payment_plans": "Autres plans de paiement",
    "housing": "Logement",
    "existing_credits": "Crédits existants",
    "job": "Profession",
    "num_dependents": "Personnes à charge",
    "own_telephone": "Téléphone personnel",
    "foreign_worker": "Travailleur étranger",
}

CATEGORY_LABELS = {
    "checking_status": {"A11": "Découvert inférieur à 0 DM", "A12": "De 0 à 200 DM", "A13": "200 DM ou plus", "A14": "Pas de compte courant"},
    "credit_history": {"A30": "Aucun crédit / tous remboursés", "A31": "Tous les crédits dans cette banque remboursés", "A32": "Crédits existants remboursés à temps", "A33": "Retards de paiement antérieurs", "A34": "Compte / crédits critiques"},
    "purpose": {"A40": "Voiture neuve", "A41": "Voiture d'occasion", "A410": "Autres", "A42": "Mobilier / équipement", "A43": "Radio / télévision", "A44": "Appareils ménagers", "A45": "Réparations", "A46": "Éducation", "A48": "Reconversion", "A49": "Activité professionnelle"},
    "savings_status": {"A61": "Inférieur à 100 DM", "A62": "100 à 500 DM", "A63": "500 à 1 000 DM", "A64": "1 000 DM ou plus", "A65": "Inconnu / sans compte épargne"},
    "employment": {"A71": "Sans emploi", "A72": "Moins d'un an", "A73": "1 à 4 ans", "A74": "4 à 7 ans", "A75": "7 ans ou plus"},
    "personal_status": {"A91": "Homme : divorcé / séparé", "A92": "Femme : divorcée / séparée / mariée", "A93": "Homme : célibataire", "A94": "Homme : marié / veuf"},
    "other_parties": {"A101": "Aucun", "A102": "Caution", "A103": "Codemandeur"},
    "property_magnitude": {"A121": "Immobilier", "A122": "Assurance-vie / épargne", "A123": "Voiture ou autre bien", "A124": "Aucun bien connu"},
    "other_payment_plans": {"A141": "Banque", "A142": "Magasin", "A143": "Aucun"},
    "housing": {"A151": "Location", "A152": "Propriétaire", "A153": "Logement gratuit"},
    "job": {"A171": "Sans emploi / non qualifié", "A172": "Non qualifié résident", "A173": "Employé qualifié / fonctionnaire", "A174": "Cadre / indépendant"},
    "own_telephone": {"A191": "Non", "A192": "Oui"},
    "foreign_worker": {"A201": "Oui", "A202": "Non"},
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Chargement du modèle de crédit…")
def load_model(path_string, modified_time):
    # modified_time participates in the cache key, so replacing the model refreshes it.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        return joblib.load(path_string)


@st.cache_data(show_spinner=False)
def load_csv(path_string, modified_time):
    return pd.read_csv(path_string)


def require_file(filename):
    path = BASE_DIR / filename
    if not path.exists():
        st.error(f"Fichier manquant : `{filename}`. Ajoute-le à la racine du dépôt GitHub.")
        st.stop()
    return path


def display_category(feature, value):
    return CATEGORY_LABELS.get(feature, {}).get(str(value), str(value))


def display_feature(feature):
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def get_default_probability(estimator, frame):
    probabilities = estimator.predict_proba(frame)[0]
    classes = list(getattr(estimator, "classes_", [0, 1]))
    default_index = classes.index(1) if 1 in classes else len(probabilities) - 1
    return float(probabilities[default_index])


def make_client_frame(client_data, metadata):
    return pd.DataFrame([[client_data[c] for c in metadata["feature_order"]]], columns=metadata["feature_order"])


def local_logistic_contributions(estimator, client_frame):
    """Return local log-odds contributions when the saved pipeline exposes LR coefficients."""
    try:
        steps = getattr(estimator, "named_steps", None)
        if not steps:
            return None
        classifier = list(steps.values())[-1]
        if not hasattr(classifier, "coef_"):
            return None
        preprocessor = steps.get("preprocessor")
        if preprocessor is None:
            return None
        transformed = preprocessor.transform(client_frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = preprocessor.get_feature_names_out()
        coefficients = np.asarray(classifier.coef_).reshape(-1)
        values = np.asarray(transformed).reshape(-1)
        if len(feature_names) != len(coefficients) or len(values) != len(coefficients):
            return None
        contributions = values * coefficients
        result = pd.DataFrame({"Variable transformée": feature_names, "Contribution": contributions})
        return result.reindex(result["Contribution"].abs().sort_values(ascending=False).index).head(12)
    except Exception:
        return None


# Load core assets
metadata_path = require_file("metadata.json")
model_path = require_file("credit_risk_model.pkl")
metadata = load_json(metadata_path)
model = load_model(str(model_path), model_path.stat().st_mtime)

comparison_path = BASE_DIR / "model_comparison.csv"
importance_path = BASE_DIR / "feature_importance.csv"
predictions_path = BASE_DIR / "test_predictions.csv"
cv_path = BASE_DIR / "cv_summary.csv"

comparison = load_csv(str(comparison_path), comparison_path.stat().st_mtime) if comparison_path.exists() else pd.DataFrame()
importance = load_csv(str(importance_path), importance_path.stat().st_mtime) if importance_path.exists() else pd.DataFrame()
predictions = load_csv(str(predictions_path), predictions_path.stat().st_mtime) if predictions_path.exists() else pd.DataFrame()
cv_summary = load_csv(str(cv_path), cv_path.stat().st_mtime) if cv_path.exists() else pd.DataFrame()

# Sidebar navigation
with st.sidebar:
    st.markdown("# 🏦 CreditRisk AI")
    st.caption("Intelligent Credit Scoring & Risk Analytics")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Accueil", "Individual Credit Prediction", "Explainable AI", "Model Performance", "Portfolio Risk Analysis"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Modèle principal**")
    st.write(metadata.get("best_model_name", "Logistic Regression"))
    st.caption("Seuil de classification : 50 %")
    st.divider()
    st.caption("Projet académique — German Credit Dataset")


# -----------------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------------
if page == "Accueil":
    st.title("CreditRisk AI")
    st.subheader("Plateforme intelligente de scoring et d'analyse du risque de crédit")
    st.markdown("Évaluez le risque individuel, comprenez les prédictions et explorez les tendances du portefeuille de test.")
    st.warning("Prototype académique : les estimations ne doivent pas être utilisées seules pour accorder ou refuser un crédit réel.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modèles évalués", str(len(comparison)) if not comparison.empty else "—")
    m2.metric("Clients dans le test", f"{len(predictions):,}" if not predictions.empty else "—")
    m3.metric("Variables", str(len(metadata.get("feature_order", []))))
    if not comparison.empty and "ROC-AUC" in comparison.columns:
        best_row = comparison.loc[comparison["ROC-AUC"].idxmax()]
        m4.metric("Meilleur ROC-AUC", f"{best_row['ROC-AUC']:.1%}")
    else:
        m4.metric("Meilleur ROC-AUC", "—")

    st.divider()
    st.markdown("### Les quatre espaces d'analyse")
    cards = st.columns(2)
    modules = [
        ("01", "Individual Credit Prediction", "Saisir les caractéristiques d'un client et estimer sa probabilité de défaut."),
        ("02", "Explainable AI", "Examiner les variables les plus influentes et les facteurs associés à une prédiction."),
        ("03", "Model Performance", "Comparer les modèles à partir des métriques réellement exportées."),
        ("04", "Portfolio Risk Analysis", "Analyser les niveaux de risque et les probabilités sur le jeu de test."),
    ]
    for i, (number, title, description) in enumerate(modules):
        with cards[i % 2]:
            st.markdown(f"#### {number} — {title}")
            st.write(description)


# -----------------------------------------------------------------------------
# INDIVIDUAL PREDICTION
# -----------------------------------------------------------------------------
elif page == "Individual Credit Prediction":
    st.title("Individual Credit Prediction")
    st.write("Renseignez les informations du demandeur. Le modèle estime la probabilité d'appartenir à la classe de défaut (classe 1).")

    client_data = {}
    with st.form("credit_prediction_form"):
        st.markdown("#### Informations numériques")
        numeric_cols = st.columns(2)
        for i, feature in enumerate(metadata["numeric_features"]):
            minimum = float(metadata["numeric_min"][feature])
            maximum = float(metadata["numeric_max"][feature])
            default = float(metadata["numeric_defaults"][feature])
            with numeric_cols[i % 2]:
                client_data[feature] = st.number_input(
                    display_feature(feature), min_value=minimum, max_value=maximum,
                    value=min(max(default, minimum), maximum), step=1.0,
                )

        st.markdown("#### Informations catégorielles")
        categorical_cols = st.columns(2)
        for i, feature in enumerate(metadata["categorical_features"]):
            options = metadata["categorical_options"][feature]
            default = metadata["categorical_defaults"][feature]
            labels = {option: f"{option} — {display_category(feature, option)}" for option in options}
            with categorical_cols[i % 2]:
                client_data[feature] = st.selectbox(
                    display_feature(feature), options=options,
                    index=options.index(default) if default in options else 0,
                    format_func=lambda value, mapping=labels: mapping[value],
                )
        submitted = st.form_submit_button("Analyser le risque", type="primary", use_container_width=True)

    if submitted:
        client_frame = make_client_frame(client_data, metadata)
        probability = get_default_probability(model, client_frame)
        threshold = float(metadata.get("threshold", 0.5))
        prediction = int(probability >= threshold)
        st.divider()
        st.markdown("### Résultat de l'évaluation")
        a, b, c = st.columns(3)
        a.metric("Probabilité estimée de défaut", f"{probability:.1%}")
        b.metric("Seuil appliqué", f"{threshold:.0%}")
        c.metric("Classe prédite", "Défaut (1)" if prediction else "Non-défaut (0)")
        st.progress(min(max(probability, 0.0), 1.0))
        if prediction:
            st.error("Risque estimé élevé — la probabilité atteint ou dépasse le seuil de classification.")
        else:
            st.success("Risque estimé faible — la probabilité est inférieure au seuil de classification.")
        st.caption("Le seuil de 50 % transforme une probabilité en classe. Il ne représente pas une frontière universelle entre bons et mauvais emprunteurs.")

        with st.expander("Voir les données transmises au modèle"):
            readable = pd.DataFrame({"Variable": [display_feature(c) for c in client_frame.columns], "Valeur": [client_data[c] if c in metadata['numeric_features'] else display_category(c, client_data[c]) for c in client_frame.columns]})
            st.dataframe(readable, hide_index=True, use_container_width=True)
        st.session_state["last_client_data"] = client_data
        st.session_state["last_probability"] = probability


# -----------------------------------------------------------------------------
# EXPLAINABLE AI
# -----------------------------------------------------------------------------
elif page == "Explainable AI":
    st.title("Explainable AI")
    st.write("Comprendre quelles variables influencent le modèle et, lorsque les coefficients sont accessibles, examiner les contributions du profil analysé.")

    if importance.empty:
        st.warning("Le fichier feature_importance.csv est absent. Ajoute le fichier exporté depuis Colab pour afficher l'importance globale.")
    else:
        imp = importance.copy()
        imp = imp.sort_values("Importance", ascending=True)
        imp["Nom affiché"] = imp["Variable"].map(display_feature)
        fig = px.bar(imp, x="Importance", y="Nom affiché", orientation="h", error_x="Std" if "Std" in imp.columns else None,
                     title="Importance globale — permutation importance", color="Importance", color_continuous_scale="Blues")
        fig.update_layout(height=650, xaxis_title="Baisse moyenne du score après permutation", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Une importance positive élevée indique que la permutation de la variable dégrade davantage le score du modèle. Ce résultat décrit l'importance prédictive, pas une causalité.")
        st.markdown("#### Top 5 des variables")
        top5 = importance.sort_values("Importance", ascending=False).head(5).copy()
        top5["Variable"] = top5["Variable"].map(display_feature)
        top5 = top5.rename(columns={"Variable": "Variable", "Importance": "Importance moyenne", "Std": "Écart-type"})
        st.dataframe(top5.style.format({c: "{:.4f}" for c in ["Importance moyenne", "Écart-type"] if c in top5.columns}), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### Explication individuelle")
    if "last_client_data" not in st.session_state:
        st.info("Effectue d'abord une prédiction dans le module Individual Credit Prediction. Tu pourras ensuite revenir ici pour examiner ce profil.")
    else:
        client_frame = make_client_frame(st.session_state["last_client_data"], metadata)
        contributions = local_logistic_contributions(model, client_frame)
        st.metric("Probabilité de défaut du dernier profil", f"{st.session_state['last_probability']:.1%}")
        if contributions is not None and not contributions.empty:
            contributions["Sens"] = np.where(contributions["Contribution"] >= 0, "Augmente le log-odds de la classe 1", "Diminue le log-odds de la classe 1")
            fig_local = px.bar(contributions.sort_values("Contribution"), x="Contribution", y="Variable transformée", orientation="h",
                               color="Contribution", color_continuous_scale="RdYlGn_r", title="Contributions locales des variables transformées")
            fig_local.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_local.update_layout(height=520, coloraxis_showscale=False)
            st.plotly_chart(fig_local, use_container_width=True)
            st.caption("Contributions calculées à partir des coefficients de la régression logistique et des variables transformées. Elles sont exprimées sur l'échelle des log-odds, et non en points de pourcentage.")
        else:
            st.info("Les coefficients du classifieur ne sont pas accessibles dans le format du modèle. L'explication globale par permutation reste disponible.")


# -----------------------------------------------------------------------------
# MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Comparaison des algorithmes à partir des résultats sauvegardés pendant l'entraînement.")
    if comparison.empty:
        st.error("Le fichier model_comparison.csv est manquant.")
    else:
        metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"] if c in comparison.columns]
        selected_metric = st.selectbox("Métrique à comparer", metric_cols, index=metric_cols.index("ROC-AUC") if "ROC-AUC" in metric_cols else 0)
        chart = px.bar(comparison.sort_values(selected_metric, ascending=False), x="Model", y=selected_metric, color="Model",
                       text=comparison.sort_values(selected_metric, ascending=False)[selected_metric].map(lambda x: f"{x:.1%}"),
                       title=f"Comparaison des modèles — {selected_metric}", range_y=[0, 1])
        chart.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(comparison.style.format({c: "{:.2%}" for c in metric_cols}), hide_index=True, use_container_width=True)
        best = comparison.loc[comparison["ROC-AUC"].idxmax()] if "ROC-AUC" in comparison.columns else None
        if best is not None:
            st.success(f"Meilleur ROC-AUC dans les résultats exportés : {best['Model']} ({best['ROC-AUC']:.2%}).")

    if not predictions.empty and {"Actual_Class", "Predicted_Class", "Default_Probability"}.issubset(predictions.columns):
        st.divider()
        st.markdown("### Évaluation détaillée du modèle principal")
        st.caption("Les prédictions exportées concernent le modèle principal uniquement. Les matrices de confusion et la courbe ROC ci-dessous ne sont donc pas attribuées aux deux autres modèles.")
        y_true = predictions["Actual_Class"].astype(int)
        y_pred = predictions["Predicted_Class"].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        c1, c2 = st.columns(2)
        with c1:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                               x=["Prédit : 0", "Prédit : 1"], y=["Réel : 0", "Réel : 1"], title="Matrice de confusion — modèle principal")
            fig_cm.update_layout(coloraxis_showscale=False, xaxis_title="Classe prédite", yaxis_title="Classe réelle")
            st.plotly_chart(fig_cm, use_container_width=True)
        with c2:
            fpr, tpr, _ = roc_curve(y_true, predictions["Default_Probability"])
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Modèle principal (AUC = {roc_auc:.3f})", line=dict(color=BLUE, width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Référence aléatoire", line=dict(color="gray", dash="dash")))
            fig_roc.update_layout(title="Courbe ROC — modèle principal", xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_roc, use_container_width=True)

    if not cv_summary.empty:
        st.divider()
        st.markdown("### Validation croisée")
        st.caption("Moyenne et écart-type des métriques enregistrées pendant la validation croisée en 5 parties.")
        st.dataframe(cv_summary.style.format({c: "{:.4f}" for c in ["Mean", "Std"] if c in cv_summary.columns}), hide_index=True, use_container_width=True)


# -----------------------------------------------------------------------------
# PORTFOLIO ANALYSIS
# -----------------------------------------------------------------------------
elif page == "Portfolio Risk Analysis":
    st.title("Portfolio Risk Analysis")
    st.write("Vue agrégée des probabilités et des classes prédites sur le jeu de test exporté depuis Colab.")
    required_columns = {"Actual_Class", "Default_Probability", "Predicted_Class", "Risk_Level"}
    if predictions.empty or not required_columns.issubset(predictions.columns):
        st.error("Le fichier test_predictions.csv est absent ou ne contient pas les colonnes nécessaires.")
    else:
        total = len(predictions)
        high = int((predictions["Predicted_Class"] == 1).sum())
        low = int((predictions["Predicted_Class"] == 0).sum())
        actual_defaults = int((predictions["Actual_Class"] == 1).sum())
        avg_probability = float(predictions["Default_Probability"].mean())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clients testés", f"{total:,}")
        k2.metric("Risque prédit élevé", f"{high:,}", f"{high / total:.1%}" if total else None)
        k3.metric("Risque prédit faible", f"{low:,}", f"{low / total:.1%}" if total else None)
        k4.metric("Probabilité moyenne", f"{avg_probability:.1%}")

        st.divider()
        left, right = st.columns(2)
        with left:
            risk_counts = predictions["Risk_Level"].value_counts().rename_axis("Niveau de risque").reset_index(name="Nombre de clients")
            fig_risk = px.pie(risk_counts, names="Niveau de risque", values="Nombre de clients", hole=0.55,
                              title="Répartition des niveaux de risque", color="Niveau de risque",
                              color_discrete_map={"Low Risk": GREEN, "High Risk": RED})
            st.plotly_chart(fig_risk, use_container_width=True)
        with right:
            fig_hist = px.histogram(predictions, x="Default_Probability", nbins=20,
                                    title="Distribution des probabilités de défaut", labels={"Default_Probability": "Probabilité de défaut"})
            fig_hist.add_vline(x=float(metadata.get("threshold", 0.5)), line_dash="dash", line_color=RED,
                               annotation_text="Seuil 50 %")
            fig_hist.update_layout(yaxis_title="Nombre de clients", xaxis_tickformat=".0%")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()
        st.markdown("### Répartition des classes réelles et prédites")
        class_counts = pd.DataFrame({
            "Classe": ["Classe 0", "Classe 1", "Classe 0", "Classe 1"],
            "Type": ["Réelle", "Réelle", "Prédite", "Prédite"],
            "Nombre": [int((predictions.Actual_Class == 0).sum()), int((predictions.Actual_Class == 1).sum()), low, high],
        })
        fig_classes = px.bar(class_counts, x="Classe", y="Nombre", color="Type", barmode="group", text="Nombre",
                             title="Comparaison des distributions réelles et prédites")
        st.plotly_chart(fig_classes, use_container_width=True)

        if "credit_amount" in predictions.columns:
            st.markdown("### Analyse par montant de crédit")
            fig_amount = px.box(predictions, x="Risk_Level", y="credit_amount", color="Risk_Level",
                                color_discrete_map={"Low Risk": GREEN, "High Risk": RED},
                                title="Montant du crédit selon le niveau de risque")
            st.plotly_chart(fig_amount, use_container_width=True)

        st.markdown("### Données du portefeuille")
        with st.expander("Afficher les observations du jeu de test"):
            st.dataframe(predictions, use_container_width=True, hide_index=True)
            csv_bytes = predictions.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger les prédictions CSV", data=csv_bytes,
                               file_name="creditrisk_portfolio_predictions.csv", mime="text/csv")
        st.caption(f"Dans le jeu de test, {actual_defaults} observations appartiennent réellement à la classe 1. Les niveaux Low/High Risk correspondent aux prédictions du modèle et non à une décision de crédit.")


# Footer
st.divider()
st.caption("CreditRisk AI • Projet académique de Machine Learning • German Credit Dataset • Les résultats dépendent des données et du protocole d'entraînement.")
