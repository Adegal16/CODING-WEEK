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
# These files were created by data_processing.py
# Train = 80% of patients, Test = 20% of patients

print("STEP 1 — Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Separate features (X) from target (y)
# X = all 12 medical measurements
# y = DEATH_EVENT (0 = survived, 1 = died)
X_train = train_df.drop(columns=['DEATH_EVENT'])
y_train = train_df['DEATH_EVENT']
X_test = test_df.drop(columns=['DEATH_EVENT'])
y_test = test_df['DEATH_EVENT']

print(f"  Train: {X_train.shape[0]} patients, {X_train.shape[1]} features")
print(f"  Test: {X_test.shape[0]} patients")

# ============================================
#  STEP 2 — Balance training data with SMOTE
# ============================================
# The dataset is imbalanced (68% survived vs 32% died)
# SMOTE creates synthetic patients for the minority class
# Applied ONLY to training data — test stays real

print("\nSTEP 2 — Applying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"  Before: imbalanced")
print(f"  After: {pd.Series(y_train).value_counts().to_dict()}")

# ============================================
#  STEP 3 — Scale the features
# ============================================
# Different features have very different ranges:
#   - platelets: 25,000 - 850,000
#   - serum_sodium: 113 - 148
# Scaling puts everything on the same scale (mean=0, std=1)
# This helps models like Logistic Regression perform better
#
# fit_transform on train = learn the scale + apply it
# transform on test = apply the SAME scale (no learning from test)

print("\nSTEP 3 — Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  Done — all features now on the same scale")

# ============================================
#  STEP 4 — Define the models to train
# ============================================
# We train 4 different models and compare them
# Each model learns differently:
#   - Logistic Regression: draws a simple boundary line (baseline)
#   - Random Forest: 100 decision trees vote together
#   - XGBoost: trees learn from each other's mistakes
#   - LightGBM: similar to XGBoost but faster

print("\nSTEP 4 — Training 4 models...")

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}

# ============================================
#  STEP 5 — Train each model and evaluate
# ============================================
# For each model we:
#   1. Train it on the training data (fit)
#   2. Make predictions on test data (predict)
#   3. Compare predictions vs reality (metrics)
#
# Metrics explained:
#   - Accuracy: % of correct predictions overall
#   - Precision: when model says "will die", how often is it right?
#   - Recall: of all patients who died, how many did the model catch?
#   - F1: balance between precision and recall
#   - ROC-AUC: how well does the model separate the two classes?

print("\nSTEP 5 — Evaluating models...\n")

results = []

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate all metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    # Store results for comparison
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(auc, 4)
    })
    
    # Print results for this model
    print(f"  {name}:")
    print(f"    Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print()

# ============================================
#  STEP 6 — Compare all models and pick the best
# ============================================
# We choose the best model based on F1-score because:
#   - In medicine, we need BOTH precision and recall
#   - F1 balances catching sick patients (recall)
#     with avoiding false alarms (precision)

print("=" * 60)
print("  MODEL COMPARISON TABLE")
print("=" * 60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find the model with the highest F1 score
best = results_df.loc[results_df['F1'].idxmax()]
print(f"\n  Best model: {best['Model']} (F1 = {best['F1']})")

# ============================================
#  STEP 7 — Save the best model and scaler
# ============================================
# We save two things:
#   - The trained model → so the Streamlit app can make predictions
#   - The scaler → so new patient data gets scaled the same way
# Both are saved as .pkl (pickle) files using joblib

print("\nSTEP 7 — Saving best model and scaler...")
os.makedirs("models", exist_ok=True)

best_model_name = best['Model']
best_model = models[best_model_name]

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Also save the results table for documentation
results_df.to_csv("models/model_comparison.csv", index=False)

print(f"  Saved: models/best_model.pkl ({best_model_name})")
print(f"  Saved: models/scaler.pkl")
print(f"  Saved: models/model_comparison.csv")

print("\n" + "=" * 60)
print("  MODEL TRAINING COMPLETE!")
print("=" * 60)