# main.py
import streamlit as st
from data_generator import generate_customers, generate_transactions
from features import build_twin_profiles, tag_anomaly_scores
from model_train import fit_isolation_forest
from viz import plot_fraud_pie, plot_twin_ml_bar, plot_precision_table, plot_top_risky

st.set_page_config(page_title="Digital Twin ile Fraud Demo", layout="wide")
st.title("Dijital Twin ile Fraud Detection Demo")
st.markdown("""
Bu interaktif demo, bankacılıkta müşteri dijital twin profilinin fraud (dolandırıcılık) işlemlerin tespitinde nasıl kullanılabileceğini gösterir.
""")

with st.sidebar:
    st.header("Simülasyon Ayarları")
    n_musteri = st.number_input(
        "Müşteri Sayısı", 100, 2000, 1000, 100,
        help="Kaç müşteri için veri üretilecek? (Demo için 1000 önerilir.)"
    )
    st.caption("Toplam müşteri sayısını belirler. Sayı arttıkça işlem süresi uzar.")

    ay_sayisi = st.slider(
        "Kaç Ay İşlem?", 3, 12, 6,
        help="Müşteri başına analiz edilen transaction süresi. (Bankalarda genellikle 6-12 ay kullanılır.)"
    )
    st.caption("Davranış tespitinde daha uzun süre daha doğru sonuç verir.")

    fraud_rate = st.slider(
        "Fraud Oranı (%)", 1, 10, 2,
        help="Sistemde toplam işlemlerin yüzde kaçının fraud (sahte) olacağını simüle eder. %2 genellikle gerçeğe yakındır."
    )
    st.caption("Gerçek bankalarda bu oran çoğunlukla %0.1-%2 arasıdır.")

    threshold = st.slider(
        "Twin Alarm Eşiği", 1.0, 10.0, 3.5, 0.1,
        help="Bir işlemin müşterinin twin profilinden sapma eşiği. Küçültürseniz daha çok alarm üretir."
    )
    st.caption("Düşük eşik daha hassas (çok alarm), yüksek eşik daha seçici sonuç verir.")

st.success("Parametreleri güncelledikçe demo otomatik yenilenecek.")

df_musteri = generate_customers(n_musteri)
df_trans = generate_transactions(df_musteri, ay_sayisi=ay_sayisi, fraud_rate=fraud_rate/100)
profiles = build_twin_profiles(df_trans)
df_trans = tag_anomaly_scores(df_trans, profiles, threshold=threshold)
df_trans = fit_isolation_forest(df_trans)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam İşlem", f"{len(df_trans):,}")
col2.metric("Fraud İşlem", f"{df_trans['is_fraud'].sum():,}")
col3.metric("Twin Alarmı", f"{df_trans['anomaly_flag'].sum():,}")
col4.metric("ML Alarmı", f"{df_trans['ml_anomaly_flag'].sum():,}")

st.subheader("Fraud/Normal İşlem Dağılımı")
st.write("Aşağıdaki pasta grafik, sistemdeki işlemlerin yüzde kaçının fraud (kırmızı) ve yüzde kaçının normal (yeşil) olduğunu gösterir.")
plot_fraud_pie(df_trans)

st.subheader("Fraud Tespit Başarı Oranları (Recall)")
st.write("Gerçek fraud işlemlerin yüzde kaçının twin alarmı (turuncu) veya klasik ML alarmı ile tespit edildiğini karşılaştırır.")
plot_twin_ml_bar(df_trans)

st.subheader("Precision ve Recall (Twin)")
st.write("Twin yaklaşımının doğruluk (precision) ve yakalama (recall) oranı. Precision: Alarm verilenlerin ne kadarı gerçekten fraud? Recall: Gerçek fraudların ne kadarı alarm verdi?")
plot_precision_table(df_trans)

st.subheader("En Riskli İşlemler")
st.write("Aşağıdaki tablo, anomali skoru en yüksek ve dolayısıyla en riskli işlemleri gösterir. Fraud ve alarm sütunları renkli/emoji ile kolay ayrıştırılır.")
plot_top_risky(df_trans)
