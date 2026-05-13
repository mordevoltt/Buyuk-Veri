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
# Local paths based on user environment
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_PLOTS_DIR = os.path.join(PROJECT_ROOT, "ml_models", "plots")
EDA_PLOTS_DIR = os.path.join(PROJECT_ROOT, "data", "plots")
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")

# Streamlit Page Config
st.set_page_config(
    page_title="Premium Big Data Dashboard | Adım 7",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    .stApp {
        background-color: transparent;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px 20px;
        border-radius: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #888;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom-color: #00d4ff !important;
    }
    
    .plot-container {
        background: rgba(255, 255, 255, 0.03);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .plot-container:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    h1, h2, h3 {
        background: linear-gradient(to right, #00d4ff, #00ffaa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #00d4ff;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    return None

def get_latest_runs():
    try:
        # Check if MLflow is reachable locally or in docker
        # Try local first, then docker network name
        tracking_uri = f"file:///{MLRUNS_DIR.replace('\\', '/')}"
        mlflow.set_tracking_uri(tracking_uri)
        experiments = mlflow.search_experiments()
        if not experiments:
            return None
        
        runs = mlflow.search_runs(experiment_names=[ex.name for ex in experiments])
        return runs
    except:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.vectorlogo.zone/logos/databricks/databricks-icon.svg", width=100)
    st.title("Admin Panel")
    st.info("📊 Büyük Veri Dönem Projesi\n\nAdım 7: Görselleştirme ve Dashboard")
    
    st.markdown("---")
    st.write("### 🛠️ Sistem Bilgisi")
    st.write(f"**Tarih:** {datetime.now().strftime('%d.%m.%Y')}")
    st.write(f"**Durum:** 🟢 Aktif")
    
    if st.button("🔄 Verileri Yenile"):
        st.cache_data.clear()
        st.rerun()

# --- MAIN HEADER ---
st.title("💎 Online Retail II: Üst Segment Analitik Dashboard")
st.markdown("##### Gerçek zamanlı MLflow entegrasyonu ve kapsamlı model performansı izleme sistemi")

# --- METRIC TOP ROW ---
results_csv = os.path.join(ML_PLOTS_DIR, "model_comparison_results.csv")
if os.path.exists(results_csv):
    df_metrics = pd.read_csv(results_csv)
    best_model_row = df_metrics.loc[df_metrics['R2'].idxmax()]
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🏆 En İyi Model", best_model_row['Model'])
    with m2:
        st.metric("🎯 Max R²", f"{best_model_row['R2']:.4f}")
    with m3:
        st.metric("📉 Min RMSE", f"{best_model_row['RMSE']:.4f}")
    with m4:
        st.metric("📊 Model Sayısı", len(df_metrics))
else:
    st.warning("⚠️ Model sonuçları bulunamadı. Lütfen `regression_training.py` dosyasını çalıştırın.")

# --- TABS ---
tab1, tab2, tab3 = st.tabs([
    "📊 Model Karşılaştırma", 
    "🎯 Model Detayları & Residuals", 
    "📈 EDA & Veri Dağılımı"
])

# --- TAB 1: MODEL COMPARISON ---
with tab1:
    st.header("🔍 Modeller Arası Performans Analizi")
    st.write("Bu bölümde, eğitilen 5 farklı regresyon modelinin anahtar metrikler üzerinden karşılaştırmasını görebilirsiniz.")
    
    if os.path.exists(results_csv):
        # Interactive Grouped Bar Chart with Plotly
        df_melted = df_metrics.melt(id_vars="Model", value_vars=["RMSE", "MAE", "R2"], 
                                   var_name="Metrik", value_name="Değer")
        
        fig = px.bar(df_melted, x="Model", y="Değer", color="Metrik", barmode="group",
                     title="Model Performans Karşılaştırması (Grouped Bar Chart)",
                     color_discrete_sequence=px.colors.qualitative.Prism,
                     template="plotly_dark")
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=20,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Real Value Metrics (Money scale)
        st.subheader("💰 Gerçek Değer (₺) Bazında Hata Analizi")
        df_real_melted = df_metrics.melt(id_vars="Model", value_vars=["Real_RMSE", "Real_MAE"], 
                                        var_name="Metrik", value_name="Değer (₺)")
        
        fig_real = px.bar(df_real_melted, x="Model", y="Değer (₺)", color="Metrik", barmode="group",
                         title="Gerçek Dünya Para Birimi Hata Payları",
                         color_discrete_sequence=["#00d4ff", "#00ffaa"],
                         template="plotly_dark")
        st.plotly_chart(fig_real, use_container_width=True)
        
        # Comparison Table
        with st.expander("📄 Ham Veri Tablosunu Görüntüle"):
            st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['R2'], color='#1e3d3d')
                                        .highlight_min(axis=0, subset=['RMSE', 'MAE'], color='#3d1e1e'))

# --- TAB 2: MODEL DETAILS ---
with tab2:
    st.header("🎯 Derinlemesine Model İncelemesi")
    
    available_models = [f.replace("_importance.png", "") for f in os.listdir(ML_PLOTS_DIR) if f.endswith("_importance.png")]
    
    if available_models:
        selected_model = st.selectbox("İncelemek istediğiniz modeli seçin:", available_models)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown(f"### 🏗️ Feature Importance: {selected_model}")
            st.write("Modelin karar verirken en çok hangi özelliklere odaklandığını gösterir (Horizontal Bar Chart).")
            img_fi = load_image(os.path.join(ML_PLOTS_DIR, f"{selected_model}_importance.png"))
            if img_fi:
                st.image(img_fi, use_container_width=True)
            else:
                st.info("Bu model için önem grafiği bulunamadı.")
                
        with c2:
            st.markdown(f"### 📉 Residual Analizi: {selected_model}")
            st.write("Artık (residual) dağılımı, tahminlerin ne kadar tutarlı olduğunu ve modelin hatalarını gösterir.")
            img_res = load_image(os.path.join(ML_PLOTS_DIR, f"{selected_model}_residual_analysis.png"))
            if img_res:
                st.image(img_res, use_container_width=True)
            else:
                st.info("Bu model için residual analizi grafiği bulunamadı.")
                
        st.markdown("---")
        st.subheader("💡 Analiz Notu")
        st.info(f"""
        **{selected_model}** modeli için yapılan analizde;
        - **Actual vs Predicted** grafiği, tahminlerin gerçek değerlere yakınlığını gösterir.
        - **Q-Q Plot**, hataların normal dağılıp dağılmadığını test eder.
        - **Residual vs Predicted**, varyansın (heteroscedasticity) sabit olup olmadığını kontrol eder.
        """)
    else:
        st.error("Model grafikleri bulunamadı. Lütfen eğitim sürecini kontrol edin.")

# --- TAB 3: EDA & DATA DISTRIBUTION ---
with tab3:
    st.header("📈 Keşifçi Veri Analizi (EDA) Bulguları")
    
    # Adım 7 Zorunlu Görseller: Zaman serisi, Dağılım, Pie, Histogram
    e_col1, e_col2 = st.columns(2)
    
    with e_col1:
        st.markdown("### 🕒 Zaman Serisi Trendleri")
        st.write("Verideki olay trafiğinin saatlik dağılımı.")
        img_hourly = load_image(os.path.join(EDA_PLOTS_DIR, "02_hourly_trend.png"))
        if img_hourly:
            st.image(img_hourly, use_container_width=True)
            
        st.markdown("### 📅 Günlük Dağılım")
        img_daily = load_image(os.path.join(EDA_PLOTS_DIR, "03_daily_trend.png"))
        if img_daily:
            st.image(img_daily, use_container_width=True)

    with e_col2:
        st.markdown("### 🍕 Olay Tipleri Dağılımı")
        st.write("Farklı işlem tiplerinin veri setindeki ağırlığı (Pie Chart).")
        img_pie = load_image(os.path.join(EDA_PLOTS_DIR, "04_event_pie_chart.png"))
        if img_pie:
            st.image(img_pie, use_container_width=True)
            
        st.markdown("### 👤 Kullanıcı Aktivite Dağılımı")
        st.write("Kullanıcı başına düşen işlem sayısı (Histogram).")
        img_dist = load_image(os.path.join(EDA_PLOTS_DIR, "05_user_activity_dist.png"))
        if img_dist:
            st.image(img_dist, use_container_width=True)

    st.markdown("---")
    st.subheader("📑 Ek EDA Bulguları")
    ae1, ae2 = st.columns(2)
    with ae1:
        img_missing = load_image(os.path.join(EDA_PLOTS_DIR, "01_missing_values.png"))
        if img_missing:
            st.image(img_missing, caption="Eksik Değer Analizi", use_container_width=True)
    with ae2:
        img_stats = load_image(os.path.join(EDA_PLOTS_DIR, "00_general_stats.png"))
        if img_stats:
            st.image(img_stats, caption="Genel Veri İstatistikleri", use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Büyük Veri Analizine Giriş Dönem Projesi | © 2026 | Tüm Hakları Saklıdır."
    "</div>", 
    unsafe_allow_html=True
)
