import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Importation de vos fonctions depuis le dossier src
from src.evaluate_model import generate_shap_summary, generate_shap_individual

def test_shap_generation():
    # 1. Création de données fictives simples
    X_train = pd.DataFrame({"age": [50, 60, 70, 45], "ejection_fraction": [30, 40, 20, 50]})
    y_train = [0, 1, 1, 0]
    patient = pd.DataFrame({"age": [65], "ejection_fraction": [25]})

    # 2. Entraînement d'un modèle basique
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    # 3. Exécution de vos fonctions SHAP
    fig_summary = generate_shap_summary(model, X_train)
    fig_individual = generate_shap_individual(model, X_train, patient)

    # 4. Vérification que le résultat est bien une image (Figure)
    assert isinstance(fig_summary, plt.Figure)
    assert isinstance(fig_individual, plt.Figure)