import shap
import matplotlib.pyplot as plt

def generate_shap_summary(model, X_train):
    """
    Génère un graphique SHAP global (Summary Plot).
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    
    plt.figure(figsize=(10, 6))
    
    # Pour le summary plot, SHAP gère souvent bien les multi-classes,
    # mais pour être sûr de cibler la classe 1 (risque), on peut utiliser [..., 1]
    # Si cela génère une erreur, retirez le [..., 1]
    try:
         shap.summary_plot(shap_values[..., 1], X_train, show=False)
    except Exception:
         shap.summary_plot(shap_values, X_train, show=False)

    fig = plt.gcf()
    return fig

def generate_shap_individual(model, X_train, patient_data):
    """
    Génère un graphique SHAP local (Waterfall Plot).
    """
    explainer = shap.Explainer(model, X_train)
    shap_values_patient = explainer(patient_data)
    
    plt.figure(figsize=(8, 5))
    
    # LA CORRECTION EST ICI : 
    # shap_values_patient[0] = on prend le 1er (et unique) patient
    # [:, 1] = on prend les explications uniquement pour la classe 1 (risque)
    
    # Vérification de la forme (shape) des valeurs SHAP pour s'adapter au modèle
    if len(shap_values_patient.shape) == 3:
        # Si la forme est (nombre_patients, features, classes)
        shap.plots.waterfall(shap_values_patient[0, :, 1], show=False)
    else:
        # Si le modèle renvoie une structure différente (ex: XGBoost gère parfois différemment)
        # On essaie de prendre l'explication du premier patient pour la première classe listée (généralement la classe positive dans les cas binaires si bien configuré)
        try:
             shap.plots.waterfall(shap_values_patient[0, 1], show=False)
        except Exception:
             # Option de repli si la structure est simple (1D array d'objets Explanation)
             shap.plots.waterfall(shap_values_patient[0][:, 1], show=False)
             
    fig = plt.gcf()
    return fig