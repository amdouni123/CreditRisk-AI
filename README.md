# CreditRisk AI — Intelligent Credit Scoring & Risk Analytics

Application Streamlit académique construite à partir du modèle et des résultats exportés depuis Google Colab.

## Modules

1. **Individual Credit Prediction** — saisie des 20 variables du German Credit Dataset et estimation de la probabilité de classe 1.
2. **Explainable AI** — permutation importance globale et contributions locales lorsque les coefficients du pipeline Logistic Regression sont accessibles.
3. **Model Performance** — comparaison des métriques exportées, matrice de confusion et courbe ROC du modèle principal, validation croisée.
4. **Portfolio Risk Analysis** — distribution des niveaux de risque, probabilités, classes réelles/prédites et montants de crédit.

## Fichiers attendus à la racine

- `app.py`
- `metadata.json`
- `credit_risk_model.pkl`
- `model_comparison.csv`
- `feature_importance.csv`
- `test_predictions.csv`
- `cv_summary.csv`
- `requirements.txt`

`all_models.pkl` peut être conservé dans le dépôt pour archivage ou réutilisation. Le dashboard utilise les métriques du CSV pour comparer les trois algorithmes. Le fichier `test_predictions.csv` contient les prédictions du modèle principal uniquement ; en conséquence, la matrice de confusion et la courbe ROC détaillées sont affichées uniquement pour ce modèle. Les courbes ou matrices des deux autres modèles ne sont pas inventées.

## Environnement compatible avec les artefacts fournis

- Python 3.13
- NumPy 2.1.3
- Scikit-learn 1.6.1
- Joblib 1.4.2

Les fichiers `.pkl` doivent être chargés avec une version de Scikit-learn compatible avec celle qui a servi à les enregistrer (1.6.1). Ne les recharge pas avec une version majeure/minor différente sans réentraînement ou validation.

## Lancement local

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement Streamlit Community Cloud

1. Déposer tous les fichiers listés ci-dessus dans le même dossier à la racine du dépôt GitHub.
2. Vérifier que le fichier principal est `app.py` et que `requirements.txt` est à la racine.
3. Dans les paramètres avancés du déploiement, sélectionner Python 3.13 si cette option est proposée.
4. Déployer l'application en indiquant `app.py` comme fichier principal.
5. Si l'application était déjà déployée, déclencher un redéploiement après le commit.

## Avertissement

Ce projet est un prototype pédagogique basé sur le Statlog German Credit Data. Il ne constitue pas un système validé pour une décision bancaire réelle. Les variables, biais, seuils et performances doivent faire l'objet d'une validation réglementaire et métier avant tout usage opérationnel.
