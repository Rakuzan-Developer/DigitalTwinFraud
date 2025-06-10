# viz.py
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def plot_fraud_pie(df_trans):
    fraud_counts = df_trans['is_fraud'].value_counts().rename({0: "Normal", 1: "Fraud"})
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', colors=['#3CB371','#E63946'])
    ax.set_title("Fraud / Normal İşlem Oranı")
    st.pyplot(fig)

def plot_twin_ml_bar(df_trans):
    # Gerçek fraud işlemlerde Twin ve ML flag'lerinin başarı oranı
    real_fraud = df_trans[df_trans['is_fraud'] == 1]
    twin_found = real_fraud['anomaly_flag'].sum()
    ml_found = real_fraud['ml_anomaly_flag'].sum()
    total = len(real_fraud)
    df_plot = pd.DataFrame({
        'Tespit Tipi': ['Twin', 'ML'],
        'Recall (%)': [100 * twin_found/total if total>0 else 0, 100 * ml_found/total if total>0 else 0]
    })
    fig, ax = plt.subplots(figsize=(4,4))
    sns.barplot(x='Tespit Tipi', y='Recall (%)', data=df_plot, ax=ax, palette=['#FFA500','#457B9D'])
    ax.set_ylim(0, 110)
    ax.set_title("Fraud Tespit Başarı Oranları (Recall)")
    st.pyplot(fig)

def plot_precision_table(df_trans):
    # Precision ve recall kutuları
    tp_twin = ((df_trans['is_fraud']==1) & (df_trans['anomaly_flag']==1)).sum()
    fp_twin = ((df_trans['is_fraud']==0) & (df_trans['anomaly_flag']==1)).sum()
    fn_twin = ((df_trans['is_fraud']==1) & (df_trans['anomaly_flag']==0)).sum()
    precision = tp_twin/(tp_twin+fp_twin) if (tp_twin+fp_twin) else 0
    recall = tp_twin/(tp_twin+fn_twin) if (tp_twin+fn_twin) else 0
    st.info(f"Twin Precision: **{precision:.2f}** / Twin Recall: **{recall:.2f}**")
    # ML için de benzer şekilde eklenebilir.

def plot_top_risky(df_trans):
    # Emoji/renkli string ekle
    df = df_trans.sort_values('anomaly_score', ascending=False).head(10).copy()
    df['fraud_flag'] = df['is_fraud'].apply(lambda x: '🟥 FRAUD' if x else '🟩 Normal')
    df['twin_alarm'] = df['anomaly_flag'].apply(lambda x: '🟧 Alarm' if x else '🟩 Yok')
    df['ml_alarm'] = df['ml_anomaly_flag'].apply(lambda x: '🟧 Alarm' if x else '🟩 Yok')
    show = df[['musteri_id','tutar','saat','city','kategori','anomaly_score','fraud_flag','twin_alarm','ml_alarm']]
    st.dataframe(show)
