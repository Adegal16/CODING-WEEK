import pytest
import pandas as pd
import numpy as np
import os
import joblib
from src.data_processing import handle_missing_values, handle_outliers, handle_imbalance, split_data
from src.optimize_memory import optimize_memory

# --- 1. TEST GESTION DES VALEURS MANQUANTES ---
def test_missing_values():
    df_test = pd.DataFrame({'age': [50, None, 60], 'sex': [1, 0, 1]})
    df_cleaned = handle_missing_values(df_test)
    assert df_cleaned.isnull().sum().sum() == 0, "Il reste des valeurs NaN !"

# --- 2. TEST INTÉGRITÉ DES OUTLIERS ---
def test_outliers_integrity():
    df_test = pd.DataFrame({'age': [1, 2, 3, 1000], 'target': [0, 1, 0, 1]})
    lignes_avant = len(df_test)
    df_result = handle_outliers(df_test)
    lignes_apres = len(df_result)
    assert lignes_avant == lignes_apres, "La fonction outliers ne doit pas supprimer de lignes."

# --- 3. TEST ÉQUILIBRAGE (SMOTE) ---
def test_smote_balancing():
    # On crée un dataset très déséquilibré
    df_unbalanced = pd.DataFrame({
        'feature': np.random.rand(10),
        'DEATH_EVENT': [0]*8 + [1]*2
    })
    df_balanced = handle_imbalance(df_unbalanced)
    counts = df_balanced['DEATH_EVENT'].value_counts()
    assert counts[0] == counts[1], "Les classes ne sont pas équilibrées après SMOTE."

# --- 4. TEST SPLIT ET CRÉATION DE FICHIERS ---
def test_data_split():
    df_test = pd.DataFrame(np.random.rand(20, 13), columns=[f'c{i}' for i in range(12)] + ['DEATH_EVENT'])
    X_train, X_test, y_train, y_test = split_data(df_test)
    
    # Vérification structure
    assert len(X_train) > len(X_test)
    assert os.path.exists('data/train.csv')
    assert os.path.exists('data/test.csv')

# --- 5. TEST OPTIMISATION MÉMOIRE ---
def test_memory_reduction():
    df_test = pd.DataFrame({'col': range(1000)}, dtype='int64')
    mem_before = df_test.memory_usage(deep=True).sum()
    df_opt = optimize_memory(df_test)
    mem_after = df_opt.memory_usage(deep=True).sum()
    assert mem_after < mem_before, "L'optimisation n'a pas réduit la taille mémoire."

# --- 6. TEST CHARGEMENT MODÈLE ET PRÉDICTION ---
def test_model_production():
    # On vérifie si le modèle existe (pour éviter de faire échouer GitHub si pas encore push)
    if os.path.exists("models/best_model.pkl"):
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        
        # Test sur un patient fictif (12 caractéristiques)
        fake_patient = np.zeros((1, 12))
        scaled_patient = scaler.transform(fake_patient)
        prediction = model.predict(scaled_patient)
        
        # Vérification binaire
        assert prediction[0] in [0, 1], "La prédiction doit être 0 ou 1."
        
        # Vérification probabilités (si applicable)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(scaled_patient)
            assert 0.0 <= probas.min() <= 1.0