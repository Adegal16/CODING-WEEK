# 🫀 CardioAI — Heart Failure Risk Prediction

> Outil clinique d'aide à la décision médicale basé sur le Machine Learning et l'explicabilité SHAP.  
> **Centrale Casablanca — Coding Week 09-15 Mars 2026 — Equipe 1**



---

## Table des matières
- [Description](#-description)
- [Dataset](#-dataset)
- [Architecture du projet](#-architecture-du-projet)
- [Installation](#-installation)
- [Entraînement du modèle](#-entraînement-du-modèle)
- [Lancer l'application](#-lancer-lapplication)
- [Fonctionnalités de l'interface](#-fonctionnalités-de-linterface)
- [Modèles et performances](#-modèles-et-performances)
- [Dataset — Questions critiques](#-dataset--questions-critiques)
- [Prompt Engineering](#-prompt-engineering)
- [Equipe et contributions](#-equipe-et-contributions)

---

##  Description

CardioAI est une application web clinique qui prédit le **risque de décès par insuffisance cardiaque** à partir de 12 paramètres médicaux. Elle intègre :

- **4 modèles ML** comparés automatiquement (Random Forest, XGBoost, LightGBM, Logistic Regression)
- **SMOTE** pour gérer le déséquilibre des classes
- **SHAP** pour l'explicabilité des prédictions (Summary Plot + Waterfall Plot)
- **Interface Streamlit** dark mode intuitive pour les médecins
- **Export PDF** du rapport patient avec graphiques SHAP inclus
- **Mode démonstration automatique** si le modèle n'est pas encore entraîné

---

##  Dataset

- **Source** : [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart%2Bfailure%2Bclinical%2Brecords)
- **Taille** : 299 patients — 12 features — 1 cible binaire (`DEATH_EVENT`)
- **Distribution** : 68% survie (0) / 32% décès (1) → **déséquilibré → traité avec SMOTE**

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

##  Architecture du projet

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
│   ├── evaluate_model.py          # SHAP TreeExplainer — Summary + Waterfall
│   └── __init__.py
│
├── app/
│   └── app.py                     # Interface Streamlit + export PDF (reportlab)
│
├── models/                        # Généré automatiquement par train_model.py
│   ├── best_model.pkl             # Meilleur modèle sélectionné par F1-score
│   ├── scaler.pkl                 # StandardScaler fitted sur train
│   ├── X_train_scaled.csv         # Données scalées float64 pour SHAP
│   └── model_comparison.csv       # Tableau comparatif des 4 modèles
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

##  Installation

### Prérequis
- Python 3.8+
- pip

### Installer les dépendances

```bash
pip install -r requirements.txt
```

**`requirements.txt` :**
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
reportlab
pytest
```



---

##  Entraînement du modèle

```bash
python src/train_model.py
```

| Etape | Description |
|-------|-------------|
| STEP 1 | Chargement de `data/train.csv` et `data/test.csv` |
| STEP 2 | Application de **SMOTE** sur les données d'entraînement uniquement |
| STEP 3 | Scaling avec **StandardScaler** (fit sur train, transform sur test) |
| STEP 4 | Définition des 4 modèles |
| STEP 5 | Entraînement + évaluation (Accuracy, Precision, Recall, F1, ROC-AUC) |
| STEP 6 | Sélection du meilleur modèle par **F1-score** |
| STEP 7 | Sauvegarde dans `models/` avec précision `float64` pour SHAP |

**Fichiers générés dans `models/` :**
```
best_model.pkl          ← meilleur modèle entraîné
scaler.pkl              ← scaler pour l'interface
X_train_scaled.csv      ← données pour SHAP (float64, 10 décimales)
model_comparison.csv    ← comparaison des 4 modèles
```

---

##  Lancer l'application

```bash
streamlit run app/app.py
```

Sur Windows :
```bash
python -m streamlit run app/app.py
```

Accès : **`http://localhost:8501`**

### Modes de fonctionnement automatiques

| Mode | Condition | Prédiction | SHAP |
|------|-----------|------------|------|
|  **Démonstration** | `models/` vide | Aléatoire (placeholder) | Simulé (Plotly) |
|  **Production** | `best_model.pkl` présent | Modèle ML réel | TreeExplainer réel |

Le mode bascule **automatiquement** sans aucune modification du code.

---

##  Fonctionnalités de l'interface

### Sidebar — Saisie patient
- **7 inputs numériques** : âge, fraction d'éjection, créatinine sérique, sodium sérique, CPK, plaquettes, suivi
- **4 toggles** : anémie, diabète, hypertension, tabagisme
- **1 radio** : sexe (Homme / Femme)
- Bouton **"Analyser le Risque"**

### Onglet 1 — Résultat & Prédiction
- Carte colorée dynamique : 🟢 **Faible** (<35%) / 🟡 **Modéré** (35-60%) / 🔴 **Élevé** (>60%)
- **Gauge chart interactif** Plotly avec zones colorées
- Métriques clés (âge, fraction d'éjection, créatinine sérique)
- **Recommandation clinique** automatique

### Onglet 2 — Explication SHAP
- **Summary Plot** : waterfall des valeurs SHAP moyennes — importance globale des features
- **Waterfall Plot** : explication individuelle pour le patient analysé
- En mode démonstration : bar chart simulé Plotly

### Onglet 3 — Récapitulatif Patient
- Fiche patient complète (données numériques + facteurs de risque )
- ** Export PDF** via `reportlab` incluant :
  - Données cliniques du patient
  - Résultat + recommandation clinique
  - Graphiques SHAP (Summary + Waterfall) si disponibles

---

##  Modèles et performances

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| XGBoost | - | - | - | - | - |
| LightGBM | - | - | - | - | - |

> Remplir après `python src/train_model.py` — résultats dans `models/model_comparison.csv`

**Critère de sélection : F1-score**

En médecine, il faut détecter les patients à risque (recall élevé) tout en évitant les fausses alarmes (precision élevée). Le F1-score équilibre ces deux objectifs, ce qui est crucial pour un outil clinique.

---

##  Dataset — Questions critiques

### Le dataset était-il équilibré ?
**Non** — 68% survie / 32% décès. Solution : **SMOTE appliqué uniquement sur les données d'entraînement** pour éviter tout data leakage vers le jeu de test.

### Valeurs manquantes ?
Aucune dans ce dataset. La fonction `handle_missing_values()` dans `data_processing.py` gère les cas futurs par imputation médiane/mode.

### Outliers ?
Détectés via IQR et boxplots dans `notebooks/eda.ipynb`. Conservés car médicalement plausibles (ex : CPK très élevé lors d'une crise cardiaque).

### Pourquoi `float_format='%.10f'` dans `X_train_scaled.csv` ?
Pour éviter les erreurs d'arrondi qui causaient l'erreur SHAP `Additivity check failed`. La précision float64 garantit la cohérence entre les valeurs SHAP et le modèle entraîné.

### Quelles features influencent le plus les prédictions ?
Selon SHAP :
1. `serum_creatinine` — créatinine sérique
2. `ejection_fraction` — fraction d'éjection
3. `time` — période de suivi
4. `age` — âge du patient

---

##  Prompt Engineering

### Tâche : Développement de l'interface Streamlit + intégration SHAP + export PDF

**Prompt utilisé :**
```
"Je développe une interface Streamlit pour un projet de prédiction
d'insuffisance cardiaque avec Random Forest et SHAP.
Je veux :
- Une sidebar pour saisir les 12 features du dataset UCI
- Un onglet résultat avec gauge chart Plotly et recommandation clinique
- Un onglet SHAP avec summary plot et waterfall plot depuis evaluate_model.py
- Un onglet récapitulatif avec export PDF via reportlab incluant les graphiques SHAP
- Un système placeholder automatique si best_model.pkl est absent
- Un thème dark mode professionnel (fond #0f1117, accents bleu #1f6feb)
Compatible avec generate_shap_summary() et generate_shap_individual()"
```

**Problèmes résolus avec l'IA :**

| Problème | Solution |
|----------|----------|
| `Additivity check failed` | `TreeExplainer` + `check_additivity=False` |
| Graphe d'interaction SHAP | Suppression de `feature_perturbation="interventional"` |
| Erreur numpy multi-dim indexing | `np.array(..., dtype=np.float64)` explicite partout |
| `X_train_scaled.csv` arrondi | `float_format='%.10f'` dans `to_csv()` |
| App inutilisable avant entraînement | Système de mode placeholder automatique |

---

##  Equipe et contributions

| Membre | Fichier | Contribution |
|--------|---------|-------------|
| **HAITAM SEKKOURI** | `src/data_processing.py` | Preprocessing, SMOTE, optimize_memory() |
| **ANAS ADEGAL** | `src/train_model.py` | Entraînement 4 modèles, sauvegarde float64 |
| **MOHAMED AMINE SABIRI** | `src/evaluate_model.py` | SHAP TreeExplainer, Summary + Waterfall |
| **ADIL ARROUBAT** | `app/app.py` | Interface Streamlit, dark mode, export PDF |
| **MAROUANE ABSRI** | `tests/` + CI/CD | Tests pytest + GitHub Actions |

---

##  Reproductibilité

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
# → http://localhost:8501
```

---

*🫀 CardioAI • Centrale Casablanca • Coding Week Mars 2026 • Equipe 1*
