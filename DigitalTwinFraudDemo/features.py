# features.py
import pandas as pd
import numpy as np

def build_twin_profiles(df_trans):
    # Her müşteri için "normal" davranış profili
    profiles = df_trans[df_trans['is_fraud'] == 0].groupby('musteri_id').agg({
        'tutar': ['mean', 'std'],
        'saat': ['mean', 'std'],
        'city': lambda x: x.value_counts().idxmax(),
        'kategori': lambda x: x.value_counts().idxmax()
    })
    profiles.columns = ['tutar_mean', 'tutar_std', 'saat_mean', 'saat_std', 'city_mode', 'kategori_mode']
    return profiles.reset_index()

def score_anomaly(tx_row, twin_row):
    """Bir transaction'ın twin profiline uzaklığını skorla"""
    score = 0
    if twin_row is None:
        return 0
    # Tutar sapması (z-score)
    if twin_row['tutar_std'] > 0:
        score += abs(tx_row['tutar'] - twin_row['tutar_mean']) / twin_row['tutar_std']
    # Saat sapması (z-score)
    if twin_row['saat_std'] > 0:
        score += abs(tx_row['saat'] - twin_row['saat_mean']) / twin_row['saat_std']
    # Kategori farklıysa +1
    if tx_row['kategori'] != twin_row['kategori_mode']:
        score += 1
    # Lokasyon farklıysa +1
    if tx_row['city'] != twin_row['city_mode']:
        score += 1
    return score

def tag_anomaly_scores(df_trans, df_profiles, threshold=3.5):
    # Her transaction için twin ile kıyasla skorla ve flagle
    anomaly_scores = []
    anomaly_flag = []
    for idx, tx in df_trans.iterrows():
        twin = df_profiles[df_profiles['musteri_id'] == tx['musteri_id']]
        if twin.empty:
            score = 0
        else:
            score = score_anomaly(tx, twin.iloc[0])
        anomaly_scores.append(score)
        anomaly_flag.append(int(score > threshold))
    df_trans = df_trans.copy()
    df_trans['anomaly_score'] = anomaly_scores
    df_trans['anomaly_flag'] = anomaly_flag
    return df_trans
