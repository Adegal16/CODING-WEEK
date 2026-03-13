# 🫀 CardioAI — Heart Failure Risk Prediction

> Outil clinique d'aide à la décision médicale basé sur le Machine Learning et l'explicabilité SHAP.  
> **Centrale Casablanca — Coding Week 09-15 Mars 2026 — Equipe 1**

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![ML](https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost%20%7C%20LightGBM-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)

---

## 📋 Table des matières
- [Description](#-description)
- [Dataset](#-dataset)
- [Architecture du projet](#-architecture-du-projet)
- [Installation](#-installation)
- [Entraînement du modèle](#-entraînement-du-modèle)
- [Lancer l'application](#-lancer-lapplication)
- [Modèles et performances](#-modèles-et-performances)
- [Dataset — Questions critiques](#-dataset--questions-critiques)
- [Prompt Engineering](#-prompt-engineering)
- [Equipe et contributions](#-equipe-et-contributions)

---

## 📖 Description

CardioAI est une application web clinique qui prédit le **risque de décès par insuffisance cardiaque** à partir de 12 paramètres médicaux. Elle intègre :

- **4 modèles ML** comparés automatiquement (Random Forest, XGBoost, LightGBM, Logistic Regression)
- **SMOTE** pour gérer le déséquilibre des classes
- **SHAP** pour l'explicabilité des prédictions
- **Interface Streamlit** intuitive pour les médecins
- **CI/CD** automatisé via GitHub Actions

---

## 📊 Dataset

- **Source** : [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart%2Bfailure%2Bclinical%2Brecords)
- **Taille** : 299 patients — 12 features — 1 cible binaire (`DEATH_EVENT`)
- **Distribution** : 68% survie (0) / 32% décès (1) → **déséquilibré**

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Âge du patient | Numérique |
| `anaemia` | Anémie | Binaire (0/1) |
| `creatinine_phosphokinase` | Enzyme CPK (mcg/L) | Numérique |
| `diabetes` | Diabète | Binaire (0/1) |
| `ejection_fraction` | Fraction d'éjection (%) | Numérique |
| `high_blood_pressure` | Hypertension | Binaire (0/1) |
| `platelets` | Plaquettes (kiloplatelets/mL) | Numérique |
| `serum_creatinine` | Créatinine sérique (mg/dL) | Numérique |
| `serum_sodium` | Sodium sérique (mEq/L) | Numérique |
| `sex` | Sexe (0=Femme, 1=Homme) | Binaire (0/1) |
| `smoking` | Tabagisme | Binaire (0/1) |
| `time` | Période de suivi (jours) | Numérique |
| `DEATH_EVENT` | **Cible** — Décès (0/1) | Binaire |

---

## 📁 Architecture du projet

```
heart-failure-prediction/
│
├── data/
│   ├── train.csv                  # Données d'entraînement (80%)
│   └── test.csv                   # Données de test (20%)
│
├── notebooks/
│   └── eda.ipynb                  # Analyse exploratoire des données
│
├── src/
│   ├── data_processing.py         # Chargement, SMOTE, scaling, optimize_memory()
│   ├── train_model.py             # Entraînement des 4 modèles
│   └── evaluate_model.py          # SHAP Summary Plot + Waterfall Plot
│
├── app/
│   └── app.py                     # Interface Streamlit
│
├── models/
│   ├── best_model.pkl             # Meilleur modèle entraîné
│   ├── scaler.pkl                 # StandardScaler
│   ├── X_train_scaled.csv         # Données pour SHAP
│   └── model_comparison.csv       # Comparaison des 4 modèles
│
├── tests/
│   └── test_data_processing.py    # Tests automatisés pytest
│
├── docs/
│   └── rapport_contribution_streamlit.md
│
├── .github/workflows/
│   └── ci.yml                     # CI/CD GitHub Actions
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation

### Prérequis
- Python 3.8+
- pip

### Installer les dépendances

```bash
pip install -r requirements.txt
```

Contenu de `requirements.txt` :
```
streamlit
plotly
pandas
numpy
scikit-learn
xgboost
lightgbm
imbalanced-learn
shap
joblib
matplotlib
pytest
```

---

## 🏋️ Entraînement du modèle

```bash
python src/train_model.py
```

Ce script :
1. Charge `data/train.csv` et `data/test.csv`
2. Applique **SMOTE** pour équilibrer les classes
3. Scale les features avec **StandardScaler**
4. Entraîne **4 modèles** et compare leurs performances
5. Sauvegarde le meilleur modèle dans `models/`

---

## 🚀 Lancer l'application

```bash
streamlit run app/app.py
```

ou sur Windows :

```bash
python -m streamlit run app/app.py
```

L'application s'ouvre sur : `http://localhost:8501`

---

## 📈 Modèles et performances

Quatre modèles sont entraînés et comparés automatiquement :

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Logistic Regression | - | - | - | - | - |
| **Random Forest** | - | - | - | - | - |
| XGBoost | - | - | - | - | - |
| LightGBM | - | - | - | - | - |

> Les métriques sont remplies automatiquement après `python src/train_model.py`

**Critère de sélection** : F1-score (équilibre précision/rappel, crucial en médecine)

**Pourquoi F1 ?**
- En médecine, il faut à la fois **détecter les patients à risque** (recall) et **éviter les fausses alarmes** (precision)
- Le F1-score équilibre ces deux objectifs

---

## ❓ Dataset — Questions critiques

### Le dataset était-il équilibré ?
**Non** — 68% survie / 32% décès.

**Méthode utilisée** : SMOTE (Synthetic Minority Over-sampling Technique)
- Génère des patients synthétiques pour la classe minoritaire
- Appliqué **uniquement sur les données d'entraînement** pour éviter le data leakage
- Impact : amélioration du recall sur la classe décès

### Valeurs manquantes ?
Aucune valeur manquante dans ce dataset. La fonction `handle_missing_values()` dans `data_processing.py` gère les cas futurs (imputation par médiane).

### Outliers ?
Détectés via IQR et boxplots dans `notebooks/eda.ipynb`. Conservés car médicalement plausibles.

### Corrélations ?
Faible corrélation entre features — aucune feature supprimée. Analysé dans `notebooks/eda.ipynb`.

### Quel modèle a le mieux performé ?
→ Voir `models/model_comparison.csv` après entraînement.

### Quelles features influencent le plus les prédictions (SHAP) ?
Les features les plus importantes selon SHAP :
1. `serum_creatinine` — créatinine sérique
2. `ejection_fraction` — fraction d'éjection
3. `time` — période de suivi
4. `age` — âge du patient

---

## 🤖 Prompt Engineering

### Tâche sélectionnée : Développement de l'interface Streamlit

**Prompt utilisé avec Claude AI :**
```
"Je développe une interface Streamlit pour un projet de prédiction 
d'insuffisance cardiaque avec Random Forest et SHAP. 
Je veux une interface avec :
- Une sidebar pour saisir les 12 features du dataset UCI
- Un onglet résultat avec gauge chart Plotly
- Un onglet SHAP avec summary plot et waterfall plot
- Un système placeholder qui fonctionne avant que le modèle soit entraîné
Le code doit être compatible avec evaluate_model.py qui contient 
generate_shap_summary() et generate_shap_individual()"
```

**Résultat** : Code complet et fonctionnel en une itération.

**Efficacité** : Le prompt était précis car il spécifiait le contexte technique (dataset UCI, fonctions existantes), les composants attendus et la contrainte du mode placeholder.

**Amélioration possible** : Spécifier la version de SHAP pour éviter les problèmes de compatibilité (`check_additivity`, numpy indexing).

---

## 👥 Equipe et contributions

| Membre | Contribution |
|--------|-------------|
| **[Nom 1]** | `data_processing.py` — Preprocessing, SMOTE, optimize_memory() |
| **[Nom 2]** | `train_model.py` — Entraînement et comparaison des 4 modèles |
| **[Nom 3]** | `evaluate_model.py` — SHAP explainability |
| **[Votre Nom]** | `app/app.py` — Interface Streamlit complète |
| **Equipe** | `tests/` — Tests automatisés, CI/CD GitHub Actions |

---

## 🔁 Reproductibilité

```bash
# 1. Cloner le repo
git clone https://github.com/VOTRE_USERNAME/heart-failure-prediction.git
cd heart-failure-prediction

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Entraîner le modèle
python src/train_model.py

# 4. Lancer l'application
streamlit run app/app.py
```

---

*Centrale Casablanca — Coding Week Mars 2026 — Equipe 1 🫀*
