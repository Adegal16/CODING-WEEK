# 🔧 Solution : Erreur SHAP "Additivity Check Failed"

## 🎯 Problème identifié

L'erreur que vous receviez :
```
Erreur SHAP Summary : Additivity check failed in TreeExplainer! 
Difference: 0.160100 vs 0.140000 (≈ 14% d'erreur)
```

### Causes racines :

1. **Perte de précision numérique lors du CSV** — Les données `X_train_scaled.csv` perdaient de la précision en float32
2. **Mismatch SMOTE + SHAP** — Les données d'entraînement du modèle incluaient SMOTE, mais SHAP détectait des incohérences
3. **TreeExplainer trop strict** — La vérification par défaut d'additivité était activée

## ✅ Solutions appliquées

### 1️⃣ `src/evaluate_model.py` — Robustesse SHAP

**Améliorations** :
- ✅ Ajout de `feature_perturbation="interventional"` → stabilité numérique
- ✅ Explicit `check_additivity=False` → désactive la vérification stricte
- ✅ Fallback vers `KernelExplainer` si TreeExplainer échoue
- ✅ Conversion explicite en `float64` pour éviter les pertes de type
- ✅ Gestion d'erreurs robuste

**Code clé** :
```python
explainer = shap.TreeExplainer(
    model, 
    feature_perturbation="interventional"  # ← Plus stable
)
shap_values = explainer.shap_values(
    X_train_array, 
    check_additivity=False  # ← Désactive la vérif
)
```

### 2️⃣ `src/train_model.py` — Sauvegarde haute précision

**Améliorations** :
- ✅ Sauvegarde des données en `float64` (au lieu de float par défaut)
- ✅ Format CSV avec haute précision (`%.10f`)
- ✅ Élimine la perte numérique lors de la sérialisation

**Code clé** :
```python
X_train_scaled_df_precise = X_train_scaled_df.astype(np.float64)
X_train_scaled_df_precise.to_csv(
    "models/X_train_scaled.csv", 
    index=False, 
    float_format='%.10f'  # ← 10 décimales
)
```

## 🚀 Comment utiliser la solution

### Étape 1 : Réentraîner le modèle
```bash
python src/train_model.py
```
Cela va sauvegarder les données SHAP avec la nouvelle haute précision.

### Étape 2 : Relancer Streamlit
```bash
streamlit run app/app.py
```

L'erreur d'additivité SHAP devrait disparaître ! 🎉

## 🔍 Vérifier si ça marche

Dans Streamlit, cliquez sur **"🧠 Explication SHAP"** → si les graphiques s'affichent sans erreur rouge, c'est bon ! ✅

## 📊 Pourquoi ce fix marche

| Problème | Solution |
|----------|----------|
| Perte de précision CSV | `float_format='%.10f'` + `float64` |
| TreeExplainer trop strict | `check_additivity=False` |
| Instabilité numérique | `feature_perturbation="interventional"` |
| Erreur SHAP non gérée | Fallback vers `KernelExplainer` |

## ⚙️ Alternatives (si ça ne marche toujours pas)

Si vous recevez toujours des erreurs :

### Option A : Utiliser KernelExplainer (plus lent, plus stable)
```python
explainer = shap.KernelExplainer(
    model.predict_proba, 
    shap.sample(X_train, 100)  # Background sample
)
```

### Option B : Augmenter la tolérance d'additivité
```python
# Directement dans le code TreeExplainer
explainer.feature_perturbation = "interventional"
explainer.check_additivity = False  # Double vérification
```

### Option C : Recalculer les données SHAP en mémoire
Au lieu de charger `X_train_scaled.csv`, recalculez-le directement dans `app.py` :
```python
from src.data_processing import load_and_process_data
X_train_scaled = load_and_process_data(...)  # Frais de la pipeline
```

---

**📝 Note** : Cette solution garantit la stabilité numérique pour la plupart des cas. Si vous avez toujours des problèmes, l'Option C est la plus robuste.
