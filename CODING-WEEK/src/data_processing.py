import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

def load_data(filepath):
    """Charge les données à partir d'un fichier CSV et affiche des informations sur le dataset."""
    df = pd.read_csv(filepath)
    print(f' Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes')
    return df

def handle_missing_values(df):
    """Affiche les valeurs manquantes dans le DataFrame et leurs positions."""
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print(' AUCUNE valeur manquante détectée !')
        print(' Aucun traitement nécessaire pour les valeurs manquantes.')
    else:
    # Trouver les indices exacts (ligne, colonne)
        positions = np.argwhere(df.isnull().values)
        print(f' {missing.sum()} valeurs manquantes détectées :')
        print()
        for row, col in positions:
            print(f'   → Ligne {row} | Colonne : {df.columns[col]}')

def handle_outliers(df):
    # Détection simple des outliers
    numeric_cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                'platelets', 'serum_creatinine', 'serum_sodium', 'time']
    for col in numeric_cols :
      mean = df[col].mean()
      std  = df[col].std()

    # Seuil 2*std (modéré)
      n_2 = df[(df[col] < mean - 2*std) | (df[col] > mean + 2*std)].shape[0]

      print(f'{col:35s} →  {n_2:3d}')
    print("conserver (données médicales réelles)")
    
def apply_smote(X, y):
    """SMOTE pour équilibrer les classes."""
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    print(f' Après SMOTE :')
    print(f'   Survécu  (0) : {pd.Series(y_res).value_counts()[0]} patients')
    print(f'   Décédé   (1) : {pd.Series(y_res).value_counts()[1]} patients')
    
    return X_res, y_res

#PROBLEME Rencontré:
""" SMOTE interpole entre 2 patients :
Patient A : age = 60
Patient B : age = 75
Nouveau : age = 67.18591  ← entre 60 et 75 """
""" Pour le modèle ML  → ❌ aucun problème
    Pour la lisibilité → ⚠️ un peu bizarre  """
#SOLUTION :
""" 1. Arrondir à l'entier le plus proche → age = 67"""

def handle_imbalance(df):
    """Détecte le déséquilibre et appelle apply_smote si nécessaire."""
    """exporte le dataset équilibré dans data/heart_failure_balanced.csv"""
    
    class_counts = df['DEATH_EVENT'].value_counts()
    total = len(df)
    pct_0 = class_counts[0] / total * 100
    pct_1 = class_counts[1] / total * 100
    
    print(f' Distribution des classes :')
    print(f'   Survécu  (0) : {class_counts[0]} patients ({pct_0:.1f}%)')
    print(f'   Décédé   (1) : {class_counts[1]} patients ({pct_1:.1f}%)')
    
    if abs(pct_0 - pct_1) < 10:
        print(' Dataset équilibré — aucun traitement nécessaire')
        return df
    
    else:
        print(' Dataset déséquilibré — application de SMOTE...')
        
        X = df.drop(columns=['DEATH_EVENT'])
        y = df['DEATH_EVENT']
        
        # ← appel de apply_smote
        X_res, y_res = apply_smote(X, y)
        
        df_balanced = pd.DataFrame(X_res, columns=X.columns)
        df_balanced['DEATH_EVENT'] = y_res

        # Arrondir les colonnes qui doivent être entières
        int_cols = ['age', 'anaemia', 'diabetes', 'ejection_fraction',
                    'high_blood_pressure', 'serum_sodium', 'sex', 'smoking', 'time','platelets']
        df_balanced[int_cols] = df_balanced[int_cols].round(0).astype(int)
        
        df_balanced.to_csv('data/heart_failure_balanced.csv', index=False)
        print(' Exporté → data/heart_failure_balanced.csv')
        
        return df_balanced
def split_data(df):
    """Sépare le dataset en train (80%) et test (20%) et exporte en CSV."""
    
    # Step 1 — Séparer features et cible
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    
    # Step 2 — Diviser 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Step 3 — Reconstruire les DataFrames complets
    train_df = X_train.copy()
    train_df['DEATH_EVENT'] = y_train
    
    test_df = X_test.copy()
    test_df['DEATH_EVENT'] = y_test
    
    # Step 4 — Exporter en CSV
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    # Step 5 — Afficher les résultats
    print(f' Train : {X_train.shape[0]} patients → data/train.csv')
    print(f' Test  : {X_test.shape[0]} patients → data/test.csv')
    
    return X_train, X_test, y_train, y_test

print('=' * 50)
print('  PIPELINE DATA PROCESSING')
print('=' * 50)
    
# ÉTAPE 1 — Charger le dataset
print('\n ÉTAPE 1 — Chargement des données...')
df = load_data("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")
    
# ÉTAPE 2 — Vérifier les valeurs manquantes
print('\n ÉTAPE 2 — Valeurs manquantes...')
handle_missing_values(df)
    
# ÉTAPE 3 — Détecter les outliers
print('\n ÉTAPE 3 — Détection des outliers...')
handle_outliers(df)
    
# ÉTAPE 4 — Séparer train / test
print('\n  ÉTAPE 4 — Séparation train / test...')
X_train, X_test, y_train, y_test = split_data(df)

# ÉTAPE 5 — Équilibrer les classes (SMOTE sur train seulement)
print('\n  ÉTAPE 5 — Équilibrage des classes (train only)...')
X_train, y_train = apply_smote(X_train, y_train)
    
print()
print('=' * 50)
print('   PIPELINE TERMINÉ AVEC SUCCÈS !')
print('=' * 50)





