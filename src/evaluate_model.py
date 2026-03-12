import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# Ignorer les avertissements de dépréciation pour une sortie propre dans Streamlit
warnings.filterwarnings('ignore')

def generate_shap_summary(model, X_train):
    """
    Génère un graphique SHAP global (Summary Plot).
    Correction de l'erreur d'indexation multidimensionnelle pour NumPy/SHAP.
    """
    # Conversion en array numpy float64 pour la stabilité numérique exigée 
    X_array = np.asarray(X_train, dtype=np.float64)

    # Utilisation de TreeExplainer pour les modèles basés sur les arbres (RF, XGB, LGBM)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_array, check_additivity=False)

    # CORRECTION DE L'INDEXATION :
    # Si le modèle est un Random Forest (sklearn), shap_values est une liste [classe_0, classe_1]
    if isinstance(shap_values, list):
        sv_to_plot = shap_values[1]
    # Si c'est un tableau 3D (certaines versions de SHAP/XGBoost)
    elif len(shap_values.shape) == 3:
        sv_to_plot = shap_values[:, :, 1]
    else:
        sv_to_plot = shap_values

    plt.figure(figsize=(10, 7))
    # Création du Summary Plot (Importance globale des caractéristiques médicales)
    shap.summary_plot(sv_to_plot, X_train, show=False)
    
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def generate_shap_individual(model, X_train, patient_data):
    """
    Génère un graphique SHAP local (Waterfall Plot).
    Adapté avec TreeExplainer pour renvoyer un objet Explanation compatible.
    """
    # Pour le Waterfall, on appelle directement l'explainer pour obtenir un objet "Explanation" complet
    explainer = shap.TreeExplainer(model)
    
    # On passe patient_data (qui est un DataFrame) pour garder les noms des colonnes
    explanation = explainer(patient_data, check_additivity=False)
    
    plt.figure(figsize=(8, 5))
    
    # Gestion de l'indexation pour la classification binaire
    # L'objet Explanation de TreeExplainer a souvent la forme (n_patients, n_features, n_classes)
    try:
        if len(explanation.shape) == 3:
            # patient 0, toutes les features, classe 1 (risque)
            shap.plots.waterfall(explanation[0, :, 1], show=False)
        else:
            shap.plots.waterfall(explanation[0], show=False)
    except Exception:
        # Solution de repli
        shap.plots.waterfall(explanation[0], show=False)
        
    plt.tight_layout()
    fig = plt.gcf()
    return fig