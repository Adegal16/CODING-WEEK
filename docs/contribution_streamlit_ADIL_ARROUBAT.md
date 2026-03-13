# Rapport de Contribution — Interface Streamlit
## Projet : CardioAI — Prédiction du Risque d'Insuffisance Cardiaque

> **Coding Week — 09 au 15 Mars 2026**  
> **Centrale Casablanca — Equipe 1**

---

## Table des matières
1. [Introduction](#1-introduction)
2. [Contexte du Projet](#2-contexte-du-projet)
3. [Ma Contribution : Interface Streamlit](#3-ma-contribution--interface-streamlit)
4. [Architecture Technique](#4-architecture-technique)
5. [Défis Techniques et Solutions](#5-défis-techniques-et-solutions)
6. [Résultats et Démonstration](#6-résultats-et-démonstration)
7. [Conclusion](#7-conclusion)

---

## 1. Introduction

Dans le cadre de la Coding Week organisée par Centrale Casablanca, notre équipe a développé **CardioAI**, un outil clinique d'aide à la décision médicale permettant de prédire le risque de décès par insuffisance cardiaque.

Ma contribution au sein de l'équipe a porté sur le **développement intégral de l'interface utilisateur avec Streamlit**, depuis la conception de l'architecture visuelle jusqu'à l'intégration des composants ML développés par mes coéquipiers.

---

## 2. Contexte du Projet

### 2.1 Objectif général

Développer une application clinique permettant aux médecins de :
- Saisir les 12 paramètres cliniques d'un patient
- Obtenir une probabilité de risque de décès calculée par un modèle ML
- Visualiser l'explication SHAP de la prédiction
- Consulter un récapitulatif complet du dossier patient

### 2.2 Dataset utilisé

Le dataset **Heart Failure Clinical Records** (UCI Repository) contient 299 patients avec 12 features médicales et une cible binaire `DEATH_EVENT` (0 = survie, 1 = décès).  
Distribution des classes : **68% survie / 32% décès** (déséquilibré → traité avec SMOTE).

### 2.3 Architecture globale du projet

| Module | Responsable | Description |
|--------|-------------|-------------|
| `data_processing.py` | Collègue A | Chargement, nettoyage, SMOTE, scaling |
| `train_model.py` | Collègue B | Entraînement RF, XGBoost, LightGBM, LR |
| `evaluate_model.py` | Collègue C | SHAP Summary Plot + Waterfall Plot |
| `app.py` | **Moi** | Interface Streamlit complète |
| `tests/` | Equipe | Tests automatisés pytest |

---

## 3. Ma Contribution : Interface Streamlit

### 3.1 Vue d'ensemble

J'ai développé l'intégralité du fichier `app/app.py` qui constitue l'interface utilisateur de CardioAI.  
L'interface est organisée en trois composants principaux :
- La **sidebar** de saisie des données patient
- Les **3 onglets** de résultats
- Le **système de détection automatique** du mode (placeholder vs production)

### 3.2 Fonctionnalités développées

#### A. Système de détection automatique du mode

| Mode | Condition | Prédiction | SHAP |
|------|-----------|------------|------|
| ⚠️ Démonstration | `models/` vide | Aléatoire (placeholder) | Simulé |
| ✅ Production | `best_model.pkl` + `scaler.pkl` présents | Modèle ML réel | Vrai SHAP |

Ce système permet à l'équipe de travailler en **parallèle** : l'interface est fonctionnelle et testable avant même que le modèle soit entraîné.

```python
try:
    model  = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    X_train_scaled = pd.read_csv('models/X_train_scaled.csv')
    MODEL_AVAILABLE = True
except Exception:
    pass  # → mode placeholder automatique
```

#### B. Sidebar de saisie des données patient

La sidebar contient les **12 features exactes** du dataset UCI organisées en deux sections :

- **Informations cliniques** : âge, fraction d'éjection, créatinine sérique, sodium sérique, CPK, plaquettes, période de suivi
- **Facteurs de risque** : anémie, diabète, hypertension, tabagisme, sexe (via toggles et radio buttons)

#### C. Onglet 1 — Résultat & Prédiction

Cet onglet affiche le résultat en **trois colonnes** :
- **Colonne 1** : carte colorée dynamique (Faible / Modéré / Élevé) avec probabilité en %
- **Colonne 2** : gauge chart interactif Plotly avec zones colorées (vert / orange / rouge)
- **Colonne 3** : métriques clés du patient (âge, fraction d'éjection, créatinine)

Une **recommandation clinique** est générée automatiquement selon le niveau de risque.

#### D. Onglet 2 — Explication SHAP

Cet onglet intègre les fonctions SHAP de `evaluate_model.py` et affiche **deux graphiques** :

- **Summary Plot** (gauche) : waterfall des valeurs SHAP moyennes sur tous les patients
- **Waterfall Plot** (droite) : explication individuelle pour le patient analysé

#### E. Onglet 3 — Récapitulatif Patient

Fiche patient structurée en deux colonnes :
- **Données numériques** : toutes les valeurs saisies avec leurs unités médicales
- **Facteurs de risque** : indicateurs visuels (✅/❌) pour chaque facteur binaire

---

## 4. Architecture Technique

### 4.1 Structure du fichier app.py

```
1. Imports et configuration sys.path
2. Import SHAP depuis src/evaluate_model.py (avec gestion d'erreur)
3. Chargement modèle + scaler + X_train_scaled (avec try/except)
4. Configuration Streamlit (page_config, CSS custom)
5. Constante FEATURE_ORDER (ordre des colonnes)
6. predict_real() et predict_placeholder()
7. Sidebar (inputs patient)
8. Main content (bannière + 3 onglets)
9. Footer dynamique
```

### 4.2 Gestion du pipeline de prédiction

La fonction `predict_real()` reproduit exactement le pipeline de `train_model.py` :

```python
# Même ordre de colonnes que train_model.py
features_df     = pd.DataFrame([patient_data])[FEATURE_ORDER]
# Même scaler que train_model.py
features_scaled = scaler.transform(features_df)
# Probabilité de décès (classe 1)
proba           = model.predict_proba(features_scaled)[0][1]
```

### 4.3 Technologies utilisées

| Bibliothèque | Usage |
|---|---|
| `Streamlit` | Framework interface web |
| `Plotly` | Gauge chart interactif |
| `Pandas` | Manipulation des données patient |
| `NumPy` | Conversion arrays pour SHAP |
| `Joblib` | Chargement du modèle et scaler |
| `SHAP` | Graphiques d'explicabilité |
| `Matplotlib` | Rendu des figures SHAP |

---

## 5. Défis Techniques et Solutions

| Défi | Problème | Solution |
|------|----------|----------|
| Page blanche Streamlit | Fichier app.py vide | Vérification avec `type app\app.py` + recréation |
| Modèle absent | App inutilisable avant fin entraînement | Système de mode placeholder automatique |
| Erreur SHAP additivity | `check_additivity` failed avec données scalées | `TreeExplainer` + `check_additivity=False` |
| Graphe interaction SHAP | `feature_perturbation` causait mauvais graphe | Suppression du paramètre + `plot_type='dot'` |
| Ordre colonnes | Prédictions incorrectes | Constante `FEATURE_ORDER` identique à `train_model.py` |
| Erreur numpy indexing | Multi-dimensional indexing no longer supported | Conversion explicite `np.array(..., dtype=np.float64)` |

---

## 6. Résultats et Démonstration

### 6.1 Interface en mode démonstration
- Bannière orange indiquant le mode placeholder
- Prédictions reproductibles (seed basé sur les données patient)
- Graphiques SHAP simulés montrant la structure attendue

### 6.2 Interface en mode production
Une fois `python src/train_model.py` exécuté :
- ✅ Prédictions réelles du meilleur modèle
- ✅ SHAP Summary Plot — features les plus importantes globalement
- ✅ SHAP Waterfall Plot — explication pour le patient spécifique
- ✅ Tableau de comparaison des 4 modèles entraînés

### 6.3 Commandes d'utilisation

```bash
# Installation des dépendances
pip install streamlit plotly pandas numpy scikit-learn joblib shap xgboost lightgbm imbalanced-learn

# Entraînement du modèle
python src/train_model.py

# Lancement de l'interface
python -m streamlit run app/app.py
```

---

## 7. Conclusion

Ma contribution au projet CardioAI a consisté à développer une interface Streamlit professionnelle, robuste et bien intégrée avec les modules ML de l'équipe.

**Principaux apports techniques :**
- Système de détection automatique du mode (placeholder / production)
- Intégration transparente des fonctions SHAP avec gestion complète des erreurs
- Pipeline de prédiction reproduisant exactement le preprocessing de `train_model.py`
- Interface médicale intuitive adaptée aux besoins des cliniciens
- Résolution systématique des problèmes techniques rencontrés

Ce projet m'a permis de consolider mes compétences en développement d'interfaces ML, en intégration de composants d'explicabilité IA, et en travail collaboratif sur un projet technique complexe avec une deadline serrée.

---

*Centrale Casablanca — Coding Week Mars 2026*
