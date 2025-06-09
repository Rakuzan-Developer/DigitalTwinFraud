# data_generator.py
# Dummy müşteri ve transaction verisi üretir, büyük veri için sample ve hızlı üretim mantığı barındırır

import numpy as np
import pandas as pd
from config import SEKTOR_LIST, BIREYSEL_KATEGORILER, KATEGORI_LIST

def generate_customers(n_bireysel=10000, n_kobi=2000, n_ticari=1000, seed=42):
    np.random.seed(seed)
    bireysel_df = pd.DataFrame({
        'musteri_id': [f'BIREYSEL_{i+1}' for i in range(n_bireysel)],
        'grup': 'Bireysel',
        'kategori': np.random.choice(BIREYSEL_KATEGORILER, n_bireysel),
        'finansal_performans': np.random.randint(1, 11, n_bireysel),
        'dijital_aciklik': np.random.uniform(0, 1, n_bireysel),
        'promosyon_duyarlilik': np.random.uniform(0, 1, n_bireysel),
        'yenilik_acikligi': np.random.uniform(0, 1, n_bireysel)
    })
    kobi_df = pd.DataFrame({
        'musteri_id': [f'KOBI_{i+1}' for i in range(n_kobi)],
        'grup': 'KOBİ',
        'sektor': np.random.choice(SEKTOR_LIST, n_kobi),
        'finansal_performans': np.random.randint(1, 11, n_kobi),
        'dijital_aciklik': np.random.uniform(0, 1, n_kobi),
        'promosyon_duyarlilik': np.random.uniform(0, 1, n_kobi),
        'yenilik_acikligi': np.random.uniform(0, 1, n_kobi)
    })
    ticari_df = pd.DataFrame({
        'musteri_id': [f'TICARI_{i+1}' for i in range(n_ticari)],
        'grup': 'Ticari',
        'sektor': np.random.choice(SEKTOR_LIST, n_ticari),
        'finansal_performans': np.random.randint(1, 11, n_ticari),
        'dijital_aciklik': np.random.uniform(0, 1, n_ticari),
        'promosyon_duyarlilik': np.random.uniform(0, 1, n_ticari),
        'yenilik_acikligi': np.random.uniform(0, 1, n_ticari)
    })
    df_musteri = pd.concat([bireysel_df, kobi_df, ticari_df], ignore_index=True)
    df_musteri['segment'] = df_musteri['grup']
    df_musteri['sektor'] = df_musteri['sektor'].fillna('Yok')
    df_musteri['kategori'] = df_musteri['kategori'].fillna('Kurumsal')
    return df_musteri

def generate_transactions(df_musteri, ay_sayisi=6, max_sample=20000):
    # Çok büyük müşteriyle çalışırken demo için ilk N müşteriye işlem üret (memory koruma için)
    if len(df_musteri) > max_sample:
        print(f"UYARI: Transactionlar sadece ilk {max_sample} müşteri için üretiliyor (demo için)!")
        df_musteri = df_musteri.sample(n=max_sample, random_state=42)
    kategori_list = KATEGORI_LIST
    transaction_list = []
    for idx, row in df_musteri.iterrows():
        for ay in range(1, ay_sayisi+1):
            n_trans = np.random.randint(12, 36)
            if row['grup'] == 'Bireysel':
                p_kat = np.array([0.25 if k == row['kategori'] else 0.75/(len(kategori_list)-1) for k in kategori_list])
                p_kat = p_kat / p_kat.sum()
            else:
                p_kat = np.ones(len(kategori_list)) / len(kategori_list)
            for t in range(n_trans):
                kategori = np.random.choice(kategori_list, p=p_kat)
                islem_tutari = np.round(np.random.uniform(100, 20000), 2)
                transaction_list.append({
                    'musteri_id': row['musteri_id'],
                    'ay': ay,
                    'islem_tutari': islem_tutari,
                    'kategori': kategori,
                    'kanal': np.random.choice(['Dijital', 'Şube', 'ATM'], p=[0.7, 0.18, 0.12]),
                    'hafta_ici': np.random.choice([0, 1], p=[0.22, 0.78])
                })
    return pd.DataFrame(transaction_list)
