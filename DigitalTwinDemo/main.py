# main.py
import streamlit as st
import pandas as pd
import numpy as np

from config import SEKTOR_LIST, BIREYSEL_KATEGORILER, KATEGORI_LIST, URUN_TIPLERI, URUN_KATEGORILERI, PROMOSYONLAR
from data_generator import generate_customers, generate_transactions
from features import aggregate_transactions
from model_train import train_model
from viz import (
    plot_twin_distribution, plot_segment_heatmap, plot_sektor_heatmap,
    plot_scatter_ilgi, plot_pie_twin_tepki, plot_segment_ilgi_heatmap
)

st.set_page_config(page_title="Dijital Twin & AI Müşteri Simülasyon Demo", layout="wide")
st.title("Kurumsal Ölçekli Dijital Twin & AI Segmentasyon Demo")
st.markdown("""
Bu demo, ürününüz/kampanyanız için müşteri tepkilerini, segment kırılımlarını ve veri analitiğini güçlü şekilde görselleştirir.
""")

@st.cache_data
def load_data(n_bireysel, n_kobi, n_ticari):
    df_musteri = generate_customers(n_bireysel, n_kobi, n_ticari)
    df_trans = generate_transactions(df_musteri, ay_sayisi=6, max_sample=15000)
    df_main = aggregate_transactions(df_musteri, df_trans)
    return df_main

with st.sidebar:
    st.header("Demo Ayarları")
    n_bireysel = st.number_input("Bireysel müşteri sayısı", 1000, 800000, 10000, 1000)
    n_kobi = st.number_input("KOBİ müşteri sayısı", 100, 150000, 2000, 100)
    n_ticari = st.number_input("Ticari müşteri sayısı", 100, 50000, 1000, 100)
    st.caption("Demo için müşteri ve transaction sayısını optimize edin.")

df_main = load_data(n_bireysel, n_kobi, n_ticari)

with st.sidebar:
    st.header("Yeni Ürün/Kampanya Özellikleri")

    urun_tipi = st.selectbox("Ürün Tipi", URUN_TIPLERI, help="Sunulan ürünün tipi")
    urun_kategori = st.selectbox("Ürün Kategorisi", URUN_KATEGORILERI, help="Ürün kategorisi (kredi/ödeme/pos vb.)")
    hedef_segment = st.multiselect("Hedef Müşteri Segmenti", ['Bireysel', 'KOBİ', 'Ticari'], default=['Bireysel', 'KOBİ', 'Ticari'])
    hedef_sektor = st.multiselect("Hedef Sektör", sorted(SEKTOR_LIST), default=sorted(SEKTOR_LIST), help="KOBİ/Ticari için")
    hedef_kategori = st.multiselect("Hedef Kategori (Bireysel için)", BIREYSEL_KATEGORILER, default=BIREYSEL_KATEGORILER)
    vade = st.slider("Vade (Ay)", 1, 60, 12, help="Ürünün vadesi")
    faiz_tipi = st.selectbox("Faiz Tipi", ['Sabit', 'Değişken', 'Yok'], help="Kredi veya benzeri ürünlerde faiz tipi")
    promosyonlar = st.multiselect("Promosyon/Kampanya", PROMOSYONLAR, default=["Cashback"])
    yenilikci_mi = st.radio("Ürün Yenilik Seviyesi", ['Yüksek', 'Orta', 'Düşük'], index=1)
    risk_seviyesi = st.radio("Ürün Risk Seviyesi", ['Yüksek', 'Orta', 'Düşük'], index=1)
    cikis_yili = st.selectbox("Ürünün Çıkış Yılı", [2020, 2021, 2022, 2023, 2024])
    kanal = st.selectbox("Kanal", ["Dijital", "Şube", "Her ikisi"])

    model_secimi = st.radio("Model Seçimi", ["RandomForest", "XGBoost", "DeepLearning - MLP", "DeepLearning - TabNet"])

# Yeni ürün etkisi fonksiyonu
def urun_etki_skoru(row):
    skor = 1
    if row['segment'] not in hedef_segment: skor *= 0.8
    if row['segment'] in ['KOBİ', 'Ticari']:
        if row['sektor'] not in hedef_sektor: skor *= 0.85
    if row['segment'] == 'Bireysel' and row['kategori'] not in hedef_kategori: skor *= 0.85
    if kanal == "Dijital" and row['dijital_aciklik'] < 0.4: skor *= 0.7
    if "Cashback" in promosyonlar and row['promosyon_duyarlilik'] > 0.5: skor *= 1.12
    if "Dijital Kolaylık" in promosyonlar and row['dijital_aciklik'] > 0.7: skor *= 1.10
    if yenilikci_mi == 'Yüksek' and row['yenilik_acikligi'] > 0.7: skor *= 1.08
    if risk_seviyesi == 'Yüksek': skor *= 0.85
    if vade > 36: skor *= 0.92
    if cikis_yili == 2024: skor *= 1.06
    if row['tx_category_count'] > 8: skor *= 1.04
    return skor

def twin_tepki_func(x):
    if x > 0.78: return 'alır/başvurur'
    elif x > 0.55: return 'yüksek ilgi'
    elif x > 0.35: return 'orta ilgi'
    elif x > 0.18: return 'ilgisiz kalır'
    else: return 'negatif tepki'

proba = train_model(df_main, model_secimi)
df_main = df_main.copy()
df_main['urun_skoru'] = df_main.apply(urun_etki_skoru, axis=1)
df_main['urun_ilgi_olasiligi'] = proba * df_main['urun_skoru']
df_main['twin_tepki'] = df_main['urun_ilgi_olasiligi'].apply(twin_tepki_func)

# --- Yeni Grafik ve Analizler ---
st.subheader(f"İlk 100 Müşteride Simülasyon Sonuçları ({model_secimi})")
st.dataframe(df_main[['musteri_id', 'segment', 'sektor', 'kategori', 'avg_amount', 'tx_category_count', 'twin_tepki', 'urun_ilgi_olasiligi']].head(100))

st.subheader("Genel Twin Tepki Dağılımı")
plot_twin_distribution(df_main)

st.subheader("Pie Chart ile Genel Tepki")
plot_pie_twin_tepki(df_main)

st.subheader("Segment Bazlı Twin Tepki Isı Haritası")
plot_segment_heatmap(df_main)

st.subheader("Sektör Bazlı Twin Tepki Isı Haritası")
plot_sektor_heatmap(df_main)

st.subheader("Dijital Açıklık – İlgi Skoru Dağılımı")
plot_scatter_ilgi(df_main)

st.subheader("Segment/Tepki Bazında Ortalama Ürün İlgi Skoru (Heatmap)")
plot_segment_ilgi_heatmap(df_main)

st.subheader("Bireysel Twin Analiz Paneli")
secili_musteri = st.selectbox("Bir müşteri seç", df_main['musteri_id'].sample(n=50, random_state=42).tolist())
st.write(df_main[df_main['musteri_id']==secili_musteri].T)
