import pytest
import pandas as pd
import numpy as np
import os
import joblib
from src.data_processing import (
    handle_missing_values,
    handle_outliers,
    handle_imbalance,
    split_data
)
from src.optimize_memory import optimize_memory

DATA_PATH = 'data/heart_failure_clinical_records_dataset.csv'


def test_missing_values():
    df = pd.read_csv(DATA_PATH)
    df_cleaned = handle_missing_values(df.copy())
    assert df_cleaned is not None
    assert df_cleaned.isnull().sum().sum() == 0


def test_outliers_integrity():
    df = pd.read_csv(DATA_PATH)
    df_result = handle_outliers(df.copy())
    assert df_result is not None
    assert 'creatinine_phosphokinase' in df_result.columns


def test_smote_balancing():
    df = pd.read_csv(DATA_PATH)
    # handle_imbalance retourne UN DataFrame pas X_res, y_res
    df_balanced = handle_imbalance(df.copy())
    counts = df_balanced['DEATH_EVENT'].value_counts()
    assert counts[0] == counts[1]


def test_data_split():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df.copy())
    assert len(X_train) > len(X_test)
    assert X_train.shape[1] == 12
    assert os.path.exists('data/train.csv')
    assert os.path.exists('data/test.csv')


def test_memory_reduction():
    df = pd.read_csv(DATA_PATH)
    mem_avant = df.memory_usage(deep=True).sum()
    df_opt    = optimize_memory(df.copy())
    mem_apres = df_opt.memory_usage(deep=True).sum()
    assert mem_apres < mem_avant
    assert all(df_opt[c].dtype == np.float32
               for c in df_opt.select_dtypes('float').columns)
    assert all(df_opt[c].dtype == np.int32
               for c in df_opt.select_dtypes('int').columns)


def test_model_production():
    if os.path.exists("models/best_model.pkl"):
        model    = joblib.load("models/best_model.pkl")
        scaler   = joblib.load("models/scaler.pkl")
        df       = pd.read_csv(DATA_PATH)
        X        = df.drop(columns=['DEATH_EVENT'])
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        assert set(predictions).issubset({0, 1})
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_scaled)[:, 1]
            assert probas.min() >= 0.0 and probas.max() <= 1.0
