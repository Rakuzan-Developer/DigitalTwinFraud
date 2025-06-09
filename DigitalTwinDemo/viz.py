# viz.py
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def plot_twin_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    df['twin_tepki'].value_counts().reindex(
        ['alır/başvurur', 'yüksek ilgi', 'orta ilgi', 'ilgisiz kalır', 'negatif tepki']
    ).plot(kind='bar', color=['green', 'limegreen', 'orange', 'skyblue', 'tomato'], ax=ax)
    ax.set_ylabel("Müşteri Sayısı")
    ax.set_xlabel("Tepki")
    ax.set_title("Genel Twin Tepki Dağılımı")
    plt.tight_layout()
    st.pyplot(fig)

def plot_segment_heatmap(df):
    seg_pivot = pd.pivot_table(df, values='musteri_id', index='segment', columns='twin_tepki', aggfunc='count', fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(seg_pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Segment-Tepki Isı Haritası")
    st.pyplot(fig)
    st.dataframe(seg_pivot)

def plot_sektor_heatmap(df):
    if 'sektor' in df.columns:
        sek_pivot = pd.pivot_table(df, values='musteri_id', index='sektor', columns='twin_tepki', aggfunc='count', fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(sek_pivot, annot=True, fmt="d", cmap="PuRd", ax=ax)
        ax.set_title("Sektör-Tepki Isı Haritası")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        st.dataframe(sek_pivot)

def plot_scatter_ilgi(df):
    if len(df) > 3000:
        sample_df = df.sample(n=3000, random_state=42)
    else:
        sample_df = df
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        data=sample_df, x='dijital_aciklik', y='urun_ilgi_olasiligi',
        hue='twin_tepki', alpha=0.6, ax=ax, palette="Set2"
    )
    ax.set_title("Dijital Açıklık vs. Ürün İlgisi")
    plt.tight_layout()
    st.pyplot(fig)

def plot_pie_twin_tepki(df):
    tepki_counts = df['twin_tepki'].value_counts().reindex(
        ['alır/başvurur', 'yüksek ilgi', 'orta ilgi', 'ilgisiz kalır', 'negatif tepki'], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(tepki_counts, labels=tepki_counts.index, autopct='%1.1f%%',
           colors=['green', 'limegreen', 'orange', 'skyblue', 'tomato'], startangle=90)
    ax.set_title("Tüm Müşterilerde Twin Tepki Dağılımı (Pie Chart)")
    st.pyplot(fig)

def plot_segment_ilgi_heatmap(df):
    pivot = pd.pivot_table(df, values='urun_ilgi_olasiligi', index='segment', columns='twin_tepki', aggfunc='mean', fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Segment/Tepki Bazında Ortalama Ürün İlgi Skoru")
    st.pyplot(fig)
