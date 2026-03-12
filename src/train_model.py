import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# ============================================
#  STEP 1 — Load the train and test datasets
# ============================================
print("STEP 1 — Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")

X_train = train_df.drop(columns=['DEATH_EVENT'])
y_train = train_df['DEATH_EVENT']
X_test  = test_df.drop(columns=['DEATH_EVENT'])
y_test  = test_df['DEATH_EVENT']

print(f"  Train: {X_train.shape[0]} patients, {X_train.shape[1]} features")
print(f"  Test:  {X_test.shape[0]} patients")

# ============================================
#  STEP 2 — Balance training data with SMOTE
# ============================================
print("\nSTEP 2 — Applying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

# ============================================
#  STEP 3 — Scale the features
# ============================================
print("\nSTEP 3 — Scaling features...")
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
X_test_scaled   = scaler.transform(X_test)

# Keep a DataFrame version for SHAP (needs feature names)
feature_names       = X_train.columns.tolist()
X_train_scaled_df   = pd.DataFrame(X_train_scaled, columns=feature_names)
print("  Done — all features now on the same scale")

# ============================================
#  STEP 4 — Define the models to train
# ============================================
print("\nSTEP 4 — Training 4 models...")

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    "LightGBM":            LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}

# ============================================
#  STEP 5 — Train each model and evaluate
# ============================================
print("\nSTEP 5 — Evaluating models...\n")
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred)

    results.append({
        "Model":     name,
        "Accuracy":  round(acc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1":        round(f1,   4),
        "ROC-AUC":   round(auc,  4)
    })

    print(f"  {name}:")
    print(f"    Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
          f"Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n")

# ============================================
#  STEP 6 — Compare all models and pick best
# ============================================
print("=" * 60)
print("  MODEL COMPARISON TABLE")
print("=" * 60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best = results_df.loc[results_df['F1'].idxmax()]
print(f"\n  Best model: {best['Model']} (F1 = {best['F1']})")

# ============================================
#  STEP 7 — Save everything
# ============================================
print("\nSTEP 7 — Saving best model, scaler and SHAP data...")
os.makedirs("models", exist_ok=True)

best_model_name = best['Model']
best_model      = models[best_model_name]

# Save model and scaler
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler,     "models/scaler.pkl")

# Save scaled training data for SHAP (high precision to avoid rounding errors)
X_train_scaled_df_precise = X_train_scaled_df.astype(np.float64)
X_train_scaled_df_precise.to_csv("models/X_train_scaled.csv", index=False, float_format='%.10f')

# Save comparison table
results_df.to_csv("models/model_comparison.csv", index=False)

print(f"  Saved: models/best_model.pkl     ({best_model_name})")
print(f"  Saved: models/scaler.pkl")
print(f"  Saved: models/X_train_scaled.csv  (for SHAP)")
print(f"  Saved: models/model_comparison.csv")

print("\n" + "=" * 60)
print("  MODEL TRAINING COMPLETE!")
print("=" * 60)