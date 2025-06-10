# data_generator.py
import numpy as np
import pandas as pd
from config import SEKTOR_LIST, TRANSACTION_CATEGORIES

def generate_customers(n_musteri=1000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        'musteri_id': [f'MUSTERI_{i+1}' for i in range(n_musteri)],
        'sektor': np.random.choice(SEKTOR_LIST, n_musteri),
        'risk_profili': np.random.uniform(0, 1, n_musteri),
        'dijital_aciklik': np.random.uniform(0, 1, n_musteri)
    })
    return df

def generate_transactions(df_musteri, ay_sayisi=6, fraud_rate=0.02, seed=42):
    np.random.seed(seed)
    transaction_list = []
    for _, row in df_musteri.iterrows():
        mean_amt = np.random.uniform(200, 2000)
        std_amt = np.random.uniform(100, 700)
        home_city = np.random.choice(['İstanbul', 'Ankara', 'İzmir', 'Bursa'])
        for ay in range(1, ay_sayisi+1):
            n_trans = np.random.randint(15, 45)
            for _ in range(n_trans):
                is_fraud = np.random.rand() < fraud_rate
                saat = np.random.randint(7, 22) if not is_fraud else np.random.choice([1, 3, 5, 23])
                tutar = np.random.normal(mean_amt, std_amt)
                kategori = np.random.choice(TRANSACTION_CATEGORIES)
                city = home_city if not is_fraud else np.random.choice(['Diyarbakır', 'Adana', 'Antalya'])
                transaction_list.append({
                    'musteri_id': row['musteri_id'],
                    'ay': ay,
                    'tutar': abs(tutar),
                    'kategori': kategori,
                    'city': city,
                    'saat': saat,
                    'is_fraud': int(is_fraud)
                })
    df_trans = pd.DataFrame(transaction_list)
    return df_trans
