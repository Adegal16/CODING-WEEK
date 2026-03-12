import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Importation de vos fonctions depuis le dossier src
from src.evaluate_model import generate_shap_summary, generate_shap_individual

def test_shap_generation():
    """
    Test automatisé pour vérifier la génération des graphiques SHAP[cite: 83, 84].
    Doit être exécuté via GitHub Actions[cite: 96].
    """
    # 1. Création de données fictives (avec quelques variables médicales réalistes)
    X_train = pd.DataFrame({
        "age": [50, 60, 70, 45, 55], 
        "ejection_fraction": [30, 40, 20, 50, 35],
        "serum_creatinine": [1.2, 2.5, 1.8, 0.9, 1.5]
    })
    y_train = [0, 1, 1, 0, 1]
    
    patient = pd.DataFrame({
        "age": [65], 
        "ejection_fraction": [25],
        "serum_creatinine": [2.0]
    })

    # 2. Entraînement d'un modèle basé sur les arbres (Requis par TreeExplainer)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    # 3. Exécution de votre nouvelle fonction SHAP Summary
    fig_summary = generate_shap_summary(model, X_train)
    
    # Vérification que le résultat est bien une image (Figure)
    assert isinstance(fig_summary, plt.Figure)
    
    # 4. Exécution de la fonction individuelle (Waterfall)
    fig_individual = generate_shap_individual(model, X_train, patient)
    
    # Vérification que le résultat est bien une image (Figure)
    assert isinstance(fig_individual, plt.Figure)

    # Fermeture explicite des figures pour libérer la mémoire lors des tests
    plt.close('all')