# features.py
# Transaction verisini aggregate edip müşteri tablosuna feature olarak ekler

import pandas as pd

def aggregate_transactions(df_musteri, df_trans):
    # Büyük veri için sadece mevcut transactionlara aggregate yapılıyor
    agg_df = df_trans.groupby('musteri_id').agg({
        'islem_tutari': ['mean', 'sum', 'count', 'max', 'std'],
        'kategori': lambda x: x.value_counts().idxmax(),
        'kanal': lambda x: x.value_counts().idxmax(),
        'hafta_ici': 'mean'
    })
    agg_df.columns = [
        'avg_amount', 'total_amount', 'tx_count', 'max_amount', 'std_amount',
        'top_cat', 'top_channel', 'weekday_ratio'
    ]
    agg_df = agg_df.reset_index()
    agg_df['tx_category_count'] = df_trans.groupby('musteri_id')['kategori'].nunique().values

    # Ana harcama kategorisi: bireysellerde kategori, diğerlerinde top_cat
    if 'kategori' in df_musteri.columns:
        agg_df = pd.merge(agg_df, df_musteri[['musteri_id', 'kategori']], on='musteri_id', how='left')
        agg_df['ana_harcama'] = agg_df['kategori'].combine_first(agg_df['top_cat'])
        agg_df.drop('kategori', axis=1, inplace=True)
    else:
        agg_df['ana_harcama'] = agg_df['top_cat']

    # Örnek "ürüne ilgi" etiketi: total_amount > 50k ve tx_category_count > 8
    agg_df['past_product_interest'] = ((agg_df['total_amount'] > 50000) & (agg_df['tx_category_count'] > 8)).astype(int)

    # Ana müşteri verisiyle birleştir
    df_main = pd.merge(df_musteri, agg_df, on='musteri_id', how='left')
    return df_main
