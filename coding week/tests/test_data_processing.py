import pytest
import pandas as pd
import numpy as np

from src.data_processing import (
    handle_missing_values,
    handle_outliers,
    handle_imbalance,
    split_data
)

@pytest.fixture
def sample_heart_data():
    data = {
        'age': [75, 55, 65, 50, np.nan],
        'anaemia': [0, 0, 0, 1, 1],
        'creatinine_phosphokinase': [582, 7861, 146, 111, 160],
        'diabetes': [0, 0, 0, 0, 1],
        'ejection_fraction': [20, 38, 20, 20, 20],
        'high_blood_pressure': [1, 0, 0, 0, 0],
        'platelets': [265000, 263358, 162000, 210000, 327000],
        'serum_creatinine': [1.9, 1.1, 1.3, 1.9, 2.7],
        'serum_sodium': [130, 136, 129, 137, 116],
        'sex': [1, 1, 1, 1, 0],
        'smoking': [0, 0, 1, 0, 0],
        'time': [4, 6, 7, 7, 8],
        'DEATH_EVENT': [1, 1, 1, 0, 0]
    }
    return pd.DataFrame(data)

def test_handle_missing_values(sample_heart_data):
    df_clean = handle_missing_values(sample_heart_data)
    assert df_clean.isnull().sum().sum() == 0

def test_handle_outliers(sample_heart_data):
    df_no_outliers = handle_outliers(sample_heart_data, column='creatinine_phosphokinase')
    assert df_no_outliers['creatinine_phosphokinase'].max() < 7000

def test_handle_imbalance():
    # LA CORRECTION EST ICI : 6 features et 2 décès (1)
    df_imbalanced = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6], 
        'DEATH_EVENT': [0, 0, 0, 0, 1, 1]
    })
    X = df_imbalanced[['feature1']]
    y = df_imbalanced['DEATH_EVENT']
    X_bal, y_bal = handle_imbalance(X, y)
    assert y_bal.value_counts()[0] == y_bal.value_counts()[1]

def test_split_data(sample_heart_data):
    df_clean = sample_heart_data.dropna()
    X_train, X_test, y_train, y_test = split_data(df_clean, target_column='DEATH_EVENT', test_size=0.25)
    assert len(X_train) == 3
    assert len(X_test) == 1