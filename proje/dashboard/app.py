import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import glob
import mlflow
from datetime import datetime

# --- CONFIGURATION & PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_PLOTS_DIR = os.path.join(PROJECT_ROOT, "ml_models", "plots")
EDA_PLOTS_DIR = os.path.join(PROJECT_ROOT, "data", "plots")
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")

st.set_page_config(
    page_title="Premium Big Data Dashboard | Adım 7",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #ffffff; }
    .stApp { background-color: transparent; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; color: #00d4ff; }
    .plot-container { background: rgba(255, 255, 255, 0.03); padding: 20px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px; }
    h1, h2, h3 { background: linear-gradient(to right, #00d4ff, #00ffaa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

def load_image(path):
    if os.path.exists(path): return Image.open(path)
    return None

# Sidebar
with st.sidebar:
    st.image("https://www.vectorlogo.zone/logos/databricks/databricks-icon.svg", width=100)
    st.title("Admin Panel")
    st.info("📊 Büyük Veri Dönem Projesi\n\nAdım 7: Görselleştirme ve Dashboard")
    if st.button("🔄 Verileri Yenile"):
        st.cache_data.clear()
        st.rerun()

st.title("💎 Online Retail II: Üst Segment Analitik Dashboard")
st.markdown("##### Gerçek zamanlı MLflow entegrasyonu ve kapsamlı model performansı izleme sistemi")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Model Karşılaştırma", "🎯 Model Detayları & Residuals", "📈 EDA & Veri Dağılımı"])

with tab1:
    st.header("🔍 Modeller Arası Performans Analizi")
    results_csv = os.path.join(ML_PLOTS_DIR, "model_comparison_results.csv")
    if os.path.exists(results_csv):
        df_metrics = pd.read_csv(results_csv)
        df_melted = df_metrics.melt(id_vars="Model", value_vars=["RMSE", "MAE", "R2"], var_name="Metrik", value_name="Değer")
        fig = px.bar(df_melted, x="Model", y="Değer", color="Metrik", barmode="group",
                     title="Model Performans Karşılaştırması (Grouped Bar Chart)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_metrics)
    else:
        st.warning("⚠️ Model sonuçları bulunamadı.")

with tab2:
    st.header("🎯 Derinlemesine Model İncelemesi")
    if os.path.exists(ML_PLOTS_DIR):
        available_models = [f.replace("_importance.png", "") for f in os.listdir(ML_PLOTS_DIR) if f.endswith("_importance.png")]
        if available_models:
            selected_model = st.selectbox("İncelemek istediğiniz modeli seçin:", available_models)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### 🏗️ Feature Importance (Horizontal Bar Chart)")
                img_fi = load_image(os.path.join(ML_PLOTS_DIR, f"{selected_model}_importance.png"))
                if img_fi: st.image(img_fi, use_container_width=True)
            with c2:
                st.markdown(f"### 📉 Residual Analizi (Gerçek vs Tahmin)")
                img_res = load_image(os.path.join(ML_PLOTS_DIR, f"{selected_model}_residual_analysis.png"))
                if img_res: st.image(img_res, use_container_width=True)
        else:
            st.error("Model grafikleri bulunamadı.")

with tab3:
    st.header("📈 Keşifçi Veri Analizi (EDA) Bulguları")
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.markdown("### 🕒 Zaman Serisi Trendleri (Line Chart)")
        img_hourly = load_image(os.path.join(EDA_PLOTS_DIR, "02_hourly_trend.png"))
        if img_hourly: st.image(img_hourly, use_container_width=True)
    with e_col2:
        st.markdown("### 🍕 Olay Tipleri Dağılımı (Pie Chart)")
        img_pie = load_image(os.path.join(EDA_PLOTS_DIR, "04_event_pie_chart.png"))
        if img_pie: st.image(img_pie, use_container_width=True)
    
    st.markdown("### 📊 Veri Dağılımı (Histogram)")
    img_dist = load_image(os.path.join(EDA_PLOTS_DIR, "05_user_activity_dist.png"))
    if img_dist: st.image(img_dist, use_container_width=True)
