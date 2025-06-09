# model_train.py
# Makine öğrenimi ve derin öğrenme ile ilgi olasılığı tahmini yapan fonksiyonlar

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def get_features_targets(df):
    feature_cols = [
        'finansal_performans', 'dijital_aciklik', 'promosyon_duyarlilik', 'yenilik_acikligi',
        'avg_amount', 'total_amount', 'tx_count', 'max_amount', 'std_amount',
        'weekday_ratio', 'tx_category_count'
    ]
    df = df.copy()
    df['ana_harcama_code'] = pd.factorize(df['ana_harcama'])[0]
    feature_cols.append('ana_harcama_code')
    X = df[feature_cols].fillna(0).values
    y = df['past_product_interest'].fillna(0).values
    return X, y, feature_cols

def train_model(df, model_name="RandomForest", max_sample=8000):
    # Modeli sadece örneklem üzerinde eğit, ama proba tüm df için hesaplanacak
    if len(df) > max_sample:
        df_sample = df.sample(n=max_sample, random_state=42)
    else:
        df_sample = df
    X_full, y_full, _ = get_features_targets(df)           # Tüm data (prediction için)
    X, y, _ = get_features_targets(df_sample)               # Sadece eğitim için data
    if len(np.unique(y)) < 2:
        # Sadece tek sınıf var, her müşteri için aynı olasılığı döndür
        return np.ones(len(df))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=60, random_state=42, max_depth=7)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_full)[:, 1]           # DİKKAT: Tüm data üzerinde predict
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(n_estimators=60, random_state=42, max_depth=5, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_full)[:, 1]
    elif model_name == "DeepLearning - MLP":
        model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=12, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_full)[:, 1]
    elif model_name == "DeepLearning - TabNet":
        model = TabNetClassifier(verbose=0)
        model.fit(X_train, y_train, max_epochs=8, patience=3, batch_size=16384)
        proba = model.predict_proba(X_full)[:, 1]
    else:
        raise ValueError("Model seçimi bulunamadı.")
    return proba

