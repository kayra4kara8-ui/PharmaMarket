# app.py - Profesyonel İlaç Pazarı Dashboard (International Product Analizi Dahil)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Gelişmiş analitik
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import statsmodels.api as sm
from scipy import stats, integrate

# Yardımcı araçlar
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple
import math

# ================================================
# 1. PROFESYONEL KONFİGÜRASYON VE STİL AYARLARI
# ================================================
st.set_page_config(
    page_title="PharmaIntelligence Pro | Enterprise Pharma Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://pharmaintelligence.com/enterprise-support',
        'Report a bug': "https://pharmaintelligence.com/enterprise-bug-report",
        'About': """
        ### PharmaIntelligence Enterprise v5.0
        • International Product Analytics
        • Predictive Modeling
        • Real-time Market Intelligence
        • Advanced Segmentation
        • Automated Reporting
        • Machine Learning Integration
        © 2024 PharmaIntelligence Inc. All Rights Reserved
        """
    }
)

# PROFESYONEL MAVİ TEMA CSS STYLES
PROFESSIONAL_CSS = """
<style>
    /* === ROOT VARIABLES === */
    :root {
        --primary-dark: #0c1a32;
        --secondary-dark: #14274e;
        --accent-blue: #2d7dd2;
        --accent-blue-light: #4a9fe3;
        --accent-blue-dark: #1a5fa0;
        --accent-cyan: #2acaea;
        --accent-teal: #30c9c9;
        --success: #2dd2a3;
        --warning: #f2c94c;
        --danger: #eb5757;
        --info: #2d7dd2;
        
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        
        --bg-primary: #0c1a32;
        --bg-secondary: #14274e;
        --bg-card: #1e3a5f;
        --bg-hover: #2d4a7a;
        --bg-surface: #14274e;
        
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.4);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.5);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.6);
        --shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.7);
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        
        --transition-fast: 200ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* === GLOBAL STYLES === */
    .stApp {
        background: linear-gradient(135deg, var(--primary-dark), var(--secondary-dark));
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text-primary);
        min-height: 100vh;
    }
    
    /* Streamlit component fixes */
    .stDataFrame, .stTable {
        background: var(--bg-card) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--bg-hover) !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Input fields */
    .stSelectbox, .stMultiselect, .stTextInput, .stNumberInput {
        background: var(--bg-card) !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--bg-hover) !important;
    }
    
    /* Slider */
    .stSlider {
        background: var(--bg-card) !important;
        padding: 1rem !important;
        border-radius: var(--radius-sm) !important;
    }
    
    /* === TYPOGRAPHY === */
    .pharma-title {
        font-size: 2.8rem;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan), var(--accent-teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .pharma-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 800px;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.8rem;
        color: var(--text-primary);
        font-weight: 800;
        margin: 2.5rem 0 1.5rem 0;
        padding-left: 1rem;
        border-left: 5px solid var(--accent-blue);
        background: linear-gradient(90deg, rgba(45, 125, 210, 0.1), transparent);
        padding: 1rem;
        border-radius: var(--radius-sm);
    }
    
    .subsection-title {
        font-size: 1.4rem;
        color: var(--text-primary);
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--bg-hover);
    }
    
    /* === CUSTOM METRIC CARDS === */
    .custom-metric-card {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        border: 1px solid var(--bg-hover);
        transition: all var(--transition-normal);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .custom-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    }
    
    .custom-metric-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-xl);
        border-color: var(--accent-blue);
    }
    
    .custom-metric-card.primary {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-blue-dark));
    }
    
    .custom-metric-card.warning {
        background: linear-gradient(135deg, var(--warning), #f2b94c);
    }
    
    .custom-metric-card.danger {
        background: linear-gradient(135deg, var(--danger), #d64545);
    }
    
    .custom-metric-card.success {
        background: linear-gradient(135deg, var(--success), #25b592);
    }
    
    .custom-metric-card.info {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-teal));
    }
    
    .custom-metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0.5rem 0;
        color: var(--text-primary);
        line-height: 1;
    }
    
    .custom-metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .custom-metric-trend {
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
        margin-top: 0.5rem;
    }
    
    .trend-up { color: var(--success); }
    .trend-down { color: var(--danger); }
    .trend-neutral { color: var(--text-muted); }
    
    /* === INSIGHT CARDS === */
    .insight-card {
        background: var(--bg-card);
        padding: 1.2rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        border-left: 5px solid;
        margin: 0.8rem 0;
        transition: all var(--transition-fast);
        position: relative;
        overflow: hidden;
    }
    
    .insight-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.05), transparent);
        opacity: 0;
        transition: opacity var(--transition-normal);
    }
    
    .insight-card:hover::before {
        opacity: 1;
    }
    
    .insight-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
    }
    
    .insight-card.info { border-left-color: var(--accent-blue); }
    .insight-card.success { border-left-color: var(--success); }
    .insight-card.warning { border-left-color: var(--warning); }
    .insight-card.danger { border-left-color: var(--danger); }
    
    .insight-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .insight-title {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .insight-content {
        color: var(--text-secondary);
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* === FILTER SECTION === */
    .filter-section {
        background: var(--bg-card);
        padding: 1.2rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        margin-bottom: 1rem;
        border: 1px solid var(--bg-hover);
    }
    
    .filter-title {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* === FILTER STATUS === */
    .filter-status {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.2), rgba(42, 202, 234, 0.2));
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--success);
        box-shadow: var(--shadow-md);
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    .filter-status-danger {
        background: linear-gradient(135deg, rgba(235, 87, 87, 0.2), rgba(214, 69, 69, 0.2));
        border-left: 5px solid var(--warning);
    }
    
    .filter-status-warning {
        background: linear-gradient(135deg, rgba(242, 201, 76, 0.2), rgba(242, 185, 76, 0.2));
        border-left: 5px solid var(--accent-blue);
    }
    
    /* === SEARCH BOX === */
    .search-box {
        background: var(--bg-card);
        border: 1px solid var(--bg-hover);
        border-radius: var(--radius-sm);
        padding: 0.75rem 1rem;
        color: var(--text-primary);
        font-size: 0.95rem;
        transition: all var(--transition-fast);
        width: 100%;
    }
    
    .search-box:focus {
        outline: none;
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 3px rgba(45, 125, 210, 0.1);
    }
    
    /* === DATA GRID === */
    .data-grid-container {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--bg-hover);
    }
    
    /* === LOADING ANIMATION === */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* === STATUS INDICATORS === */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-online { background: var(--success); }
    .status-warning { background: var(--warning); }
    .status-error { background: var(--danger); }
    
    /* === BADGES === */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success {
        background: rgba(45, 210, 163, 0.2);
        color: var(--success);
        border: 1px solid rgba(45, 210, 163, 0.3);
    }
    
    .badge-warning {
        background: rgba(242, 201, 76, 0.2);
        color: var(--warning);
        border: 1px solid rgba(242, 201, 76, 0.3);
    }
    
    .badge-danger {
        background: rgba(235, 87, 87, 0.2);
        color: var(--danger);
        border: 1px solid rgba(235, 87, 87, 0.3);
    }
    
    .badge-info {
        background: rgba(45, 125, 210, 0.2);
        color: var(--accent-blue);
        border: 1px solid rgba(45, 125, 210, 0.3);
    }
    
    /* === SIDEBAR === */
    .sidebar-title {
        font-size: 1.4rem;
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-blue);
    }
    
    /* === FEATURE CARDS === */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-hover));
        padding: 1.5rem;
        border-radius: var(--radius-md);
        border-left: 4px solid;
        transition: all var(--transition-normal);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg);
    }
    
    .feature-card-blue { border-left-color: var(--accent-blue); }
    .feature-card-cyan { border-left-color: var(--accent-cyan); }
    .feature-card-teal { border-left-color: var(--accent-teal); }
    .feature-card-warning { border-left-color: var(--warning); }
    
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
        opacity: 0.9;
    }
    
    .feature-title {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .feature-description {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* === WELCOME CONTAINER === */
    .welcome-container {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-secondary));
        padding: 3rem;
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow-xl);
        text-align: center;
        margin: 2rem auto;
        max-width: 900px;
        border: 1px solid var(--bg-hover);
    }
    
    .welcome-icon {
        font-size: 5rem;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* === GET STARTED BOX === */
    .get-started-box {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.15), rgba(42, 202, 234, 0.1));
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        border: 1px solid rgba(45, 125, 210, 0.3);
        margin-top: 2rem;
    }
    
    .get-started-title {
        font-weight: 600;
        color: var(--accent-blue);
        margin-bottom: 0.8rem;
        font-size: 1.1rem;
    }
    
    .get-started-steps {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ================================================
# 2. OPTİMİZE VERİ İŞLEME SİSTEMİ
# ================================================

class OptimizeVeriİşleyici:
    """Optimize edilmiş veri işleme sınıfı"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
    def buyuk_veri_yukle(dosya, orneklem=None):
        """Büyük veri setlerini optimize şekilde yükle"""
        try:
            baslangic_zamani = time.time()
            
            if dosya.name.endswith('.csv'):
                if orneklem:
                    df = pd.read_csv(dosya, nrows=orneklem)
                else:
                    with st.spinner("📥 CSV verisi yükleniyor..."):
                        df = pd.read_csv(dosya, low_memory=False)
                        
            elif dosya.name.endswith(('.xlsx', '.xls')):
                if orneklem:
                    parcalar = []
                    parca_boyutu = 50000
                    toplam_parca = (orneklem // parca_boyutu) + 1
                    
                    with st.spinner(f"📥 Büyük veri seti yükleniyor..."):
                        ilerleme_cubugu = st.progress(0)
                        durum_metni = st.empty()
                        
                        for i in range(toplam_parca):
                            parca = pd.read_excel(
                                dosya, 
                                skiprows=i * parca_boyutu,
                                nrows=parca_boyutu,
                                engine='openpyxl'
                            )
                            
                            if parca.empty:
                                break
                            
                            parcalar.append(parca)
                            
                            yuklenen_satir = sum(len(p) for p in parcalar)
                            ilerleme = min(yuklenen_satir / orneklem, 1.0)
                            
                            ilerleme_cubugu.progress(ilerleme)
                            durum_metni.text(f"📊 {yuklenen_satir:,} satır yüklendi...")
                            
                            if yuklenen_satir >= orneklem:
                                break
                        
                        df = pd.concat(parcalar, ignore_index=True)
                        ilerleme_cubugu.progress(1.0)
                        durum_metni.text(f"✅ {len(df):,} satır başarıyla yüklendi")
                        time.sleep(0.5)
                        ilerleme_cubugu.empty()
                        durum_metni.empty()
                else:
                    with st.spinner(f"📥 Tüm veri seti yükleniyor..."):
                        df = pd.read_excel(dosya, engine='openpyxl')
            
            df = OptimizeVeriİşleyici.dataframe_optimize_et(df)
            
            yukleme_suresi = time.time() - baslangic_zamani
            st.success(f"✅ Veri yükleme tamamlandı: {len(df):,} satır, {len(df.columns)} sütun ({yukleme_suresi:.2f}s)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.error(f"Detay: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def dataframe_optimize_et(df):
        """DataFrame'i optimize et"""
        try:
            orijinal_bellek = df.memory_usage(deep=True).sum() / 1024**2
            
            # Sütun isimlerini temizle
            df.columns = OptimizeVeriİşleyici.sutun_isimleri_temizle(df.columns)
            
            # Optimizasyon
            with st.spinner("Veri seti optimize ediliyor..."):
                
                # Kategorik sütunlar
                for sutun in df.select_dtypes(include=['object']).columns:
                    benzersiz_sayi = df[sutun].nunique()
                    toplam_satir = len(df)
                    
                    if benzersiz_sayi < toplam_satir * 0.7:
                        df[sutun] = df[sutun].astype('category')
                
                # Sayısal sütunlar
                for sutun in df.select_dtypes(include=[np.number]).columns:
                    try:
                        sutun_min = df[sutun].min()
                        sutun_max = df[sutun].max()
                        
                        if pd.api.types.is_integer_dtype(df[sutun]):
                            if sutun_min >= 0:
                                if sutun_max <= 255:
                                    df[sutun] = df[sutun].astype(np.uint8)
                                elif sutun_max <= 65535:
                                    df[sutun] = df[sutun].astype(np.uint16)
                                elif sutun_max <= 4294967295:
                                    df[sutun] = df[sutun].astype(np.uint32)
                            else:
                                if sutun_min >= -128 and sutun_max <= 127:
                                    df[sutun] = df[sutun].astype(np.int8)
                                elif sutun_min >= -32768 and sutun_max <= 32767:
                                    df[sutun] = df[sutun].astype(np.int16)
                                elif sutun_min >= -2147483648 and sutun_max <= 2147483647:
                                    df[sutun] = df[sutun].astype(np.int32)
                        else:
                            df[sutun] = df[sutun].astype(np.float32)
                    except:
                        continue
                
                # String temizleme
                for sutun in df.select_dtypes(include=['object']).columns:
                    try:
                        df[sutun] = df[sutun].astype(str).str.strip()
                    except:
                        pass
            
            optimize_bellek = df.memory_usage(deep=True).sum() / 1024**2
            bellek_tasarrufu = orijinal_bellek - optimize_bellek
            
            if bellek_tasarrufu > 0:
                st.success(f"💾 Bellek optimizasyonu başarılı: {orijinal_bellek:.1f}MB → {optimize_bellek:.1f}MB (%{bellek_tasarrufu/orijinal_bellek*100:.1f} tasarruf)")
            
            return df
            
        except Exception as e:
            st.warning(f"Optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def sutun_isimleri_temizle(sutunlar):
        """Sütun isimlerini temizle"""
        temizlenen = []
        for sutun in sutunlar:
            if isinstance(sutun, str):
                # Türkçe karakterleri düzelt
                degisimler = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in degisimler.items():
                    sutun = sutun.replace(tr, en)
                
                # Yeni satır ve boşlukları temizle
                sutun = sutun.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                sutun = ' '.join(sutun.split())
                
                # DÖNÜŞÜMLER
                if "MAT Q3 2022 USD MNF" in sutun:
                    sutun = "Satış_2022"
                elif "MAT Q3 2023 USD MNF" in sutun:
                    sutun = "Satış_2023"
                elif "MAT Q3 2024 USD MNF" in sutun:
                    sutun = "Satış_2024"
                
                elif "MAT Q3 2022 Units" in sutun:
                    sutun = "Birim_2022"
                elif "MAT Q3 2023 Units" in sutun:
                    sutun = "Birim_2023"
                elif "MAT Q3 2024 Units" in sutun:
                    sutun = "Birim_2024"
                
                elif "MAT Q3 2022 Unit Avg Price USD MNF" in sutun:
                    sutun = "Ort_Fiyat_2022"
                elif "MAT Q3 2023 Unit Avg Price USD MNF" in sutun:
                    sutun = "Ort_Fiyat_2023"
                elif "MAT Q3 2024 Unit Avg Price USD MNF" in sutun:
                    sutun = "Ort_Fiyat_2024"
                
                elif "MAT Q3 2022 Standard Units" in sutun:
                    sutun = "Standard_Units_2022"
                elif "MAT Q3 2023 Standard Units" in sutun:
                    sutun = "Standard_Units_2023"
                elif "MAT Q3 2024 Standard Units" in sutun:
                    sutun = "Standard_Units_2024"
                
                elif "MAT Q3 2022 SU Avg Price USD MNF" in sutun:
                    sutun = "SU_Ort_Fiyat_2022"
                elif "MAT Q3 2023 SU Avg Price USD MNF" in sutun:
                    sutun = "SU_Ort_Fiyat_2023"
                elif "MAT Q3 2024 SU Avg Price USD MNF" in sutun:
                    sutun = "SU_Ort_Fiyat_2024"
                
                elif "Source.Name" in sutun:
                    sutun = "Kaynak"
                elif "Country" in sutun:
                    sutun = "Ülke"
                elif "Sector" in sutun:
                    sutun = "Sektör"
                elif "Corporation" in sutun:
                    sutun = "Şirket"
                elif "Manufacturer" in sutun:
                    sutun = "Üretici"
                elif "Molecule List" in sutun:
                    sutun = "Molekül_Listesi"
                elif "Molecule" in sutun:
                    sutun = "Molekül"
                elif "Chemical Salt" in sutun:
                    sutun = "Kimyasal_Tuz"
                elif "International Product" in sutun:
                    sutun = "International_Product"
                elif "Specialty Product" in sutun:
                    sutun = "Özel_Ürün"
                elif "NFC123" in sutun:
                    sutun = "NFC123"
                elif "International Pack" in sutun:
                    sutun = "International_Pack"
                elif "International Strength" in sutun:
                    sutun = "International_Strength"
                elif "International Size" in sutun:
                    sutun = "International_Size"
                elif "International Volume" in sutun:
                    sutun = "International_Volume"
                elif "International Prescription" in sutun:
                    sutun = "International_Prescription"
                elif "Panel" in sutun:
                    sutun = "Panel"
                elif "Region" in sutun:
                    sutun = "Bölge"
                elif "Sub-Region" in sutun:
                    sutun = "Alt_Bölge"
                
                sutun = sutun.strip()
            
            temizlenen.append(str(sutun).strip())
        
        return temizlenen
    
    @staticmethod
    def analiz_verisi_hazirla(df):
        """Analiz için veriyi hazırla"""
        try:
            # Satış sütunlarını bul
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            
            if not satis_sutunlari:
                st.warning("⚠️ Satış sütunları bulunamadı. Veri yapınızı kontrol edin.")
                return df
            
            yillar = []
            for sutun in satis_sutunlari:
                try:
                    yil = sutun.split('_')[-1]
                    if yil.isdigit():
                        yillar.append(int(yil))
                except:
                    continue
            
            yillar = sorted(yillar)
            
            # Büyüme oranlarını hesapla
            for i in range(1, len(yillar)):
                onceki_yil = str(yillar[i-1])
                simdiki_yil = str(yillar[i])
                
                onceki_sutun = f"Satış_{onceki_yil}"
                simdiki_sutun = f"Satış_{simdiki_yil}"
                
                if onceki_sutun in df.columns and simdiki_sutun in df.columns:
                    df[f'Büyüme_{onceki_yil}_{simdiki_yil}'] = np.where(
                        df[onceki_sutun] != 0,
                        ((df[simdiki_sutun] - df[onceki_sutun]) / df[onceki_sutun]) * 100,
                        np.nan
                    )
            
            # CAGR (Compound Annual Growth Rate) hesapla
            if len(yillar) >= 2:
                ilk_yil = str(yillar[0])
                son_yil = str(yillar[-1])
                ilk_sutun = f"Satış_{ilk_yil}"
                son_sutun = f"Satış_{son_yil}"
                
                if ilk_sutun in df.columns and son_sutun in df.columns:
                    df['CAGR'] = np.where(
                        df[ilk_sutun] > 0,
                        ((df[son_sutun] / df[ilk_sutun]) ** (1/len(yillar)) - 1) * 100,
                        np.nan
                    )
            
            # Pazar payı hesapla
            if yillar:
                son_yil = str(yillar[-1])
                son_satis_sutun = f"Satış_{son_yil}"
                
                if son_satis_sutun in df.columns:
                    toplam_satis = df[son_satis_sutun].sum()
                    if toplam_satis > 0:
                        df['Pazar_Payı'] = (df[son_satis_sutun] / toplam_satis) * 100
            
            # Ortalama fiyat sütunları yoksa hesapla
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            if not fiyat_sutunlari:
                for yil in yillar:
                    satis_sutun = f"Satış_{yil}"
                    birim_sutun = f"Birim_{yil}"
                    
                    if satis_sutun in df.columns and birim_sutun in df.columns:
                        df[f'Ort_Fiyat_{yil}'] = np.where(
                            df[birim_sutun] > 0,
                            df[satis_sutun] / df[birim_sutun],
                            np.nan
                        )
                        st.info(f"ℹ️ Ort_Fiyat_{yil} sütunu hesaplandı (Satış/Birim)")
            
            # Fiyat-Hacim oranı
            if yillar:
                son_yil = str(yillar[-1])
                fiyat_sutun = f"Ort_Fiyat_{son_yil}"
                birim_sutun = f"Birim_{son_yil}"
                
                if fiyat_sutun in df.columns and birim_sutun in df.columns:
                    df['Fiyat_Hacim_Oranı'] = df[fiyat_sutun] * df[birim_sutun]
            
            # Performans skoru
            sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(sayisal_sutunlar) >= 3:
                try:
                    olceklendirici = StandardScaler()
                    sayisal_veri = df[sayisal_sutunlar].fillna(0)
                    olcekli_veri = olceklendirici.fit_transform(sayisal_veri)
                    df['Performans_Skoru'] = olcekli_veri.mean(axis=1)
                except Exception as e:
                    st.warning(f"Performans skoru hesaplanamadı: {str(e)}")
            
            # International Product analizi için temel sütunlar
            if 'International_Product' in df.columns:
                df['International_Product'] = df['International_Product'].fillna(0).astype(int)
            
            return df
            
        except Exception as e:
            st.warning(f"Analiz verisi hazırlama hatası: {str(e)}")
            return df

# ================================================
# 3. GELİŞMİŞ FİLTRELEME SİSTEMİ
# ================================================

class GelismisFiltreSistemi:
    """Gelişmiş filtreleme sistemi"""
    
    @staticmethod
    def filtre_sidebar_olustur(df):
        """Filtreleme sidebar'ını oluştur"""
        with st.sidebar.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
            st.markdown('<div class="filter-title">🔍 Arama ve Filtreleme</div>', unsafe_allow_html=True)
            
            arama_terimi = st.text_input(
                "🔎 Genel Arama",
                placeholder="Molekül, Şirket, Ülke...",
                help="Tüm sütunlarda arama yapın",
                key="genel_arama"
            )
            
            filtre_ayar = {}
            mevcut_sutunlar = df.columns.tolist()
            
            if 'Ülke' in mevcut_sutunlar:
                ulkeler = sorted(df['Ülke'].dropna().unique())
                secilen_ulkeler = GelismisFiltreSistemi.arama_yapilabilir_coklu_secim(
                    "🌍 Ülkeler",
                    ulkeler,
                    key="ulkeler_filtresi",
                    tumunu_sec_varsayilan=True
                )
                if secilen_ulkeler and "Tümü" not in secilen_ulkeler:
                    filtre_ayar['Ülke'] = secilen_ulkeler
            
            if 'Şirket' in mevcut_sutunlar:
                sirketler = sorted(df['Şirket'].dropna().unique())
                secilen_sirketler = GelismisFiltreSistemi.arama_yapilabilir_coklu_secim(
                    "🏢 Şirketler",
                    sirketler,
                    key="sirketler_filtresi",
                    tumunu_sec_varsayilan=True
                )
                if secilen_sirketler and "Tümü" not in secilen_sirketler:
                    filtre_ayar['Şirket'] = secilen_sirketler
            
            if 'Molekül' in mevcut_sutunlar:
                molekuller = sorted(df['Molekül'].dropna().unique())
                secilen_molekuller = GelismisFiltreSistemi.arama_yapilabilir_coklu_secim(
                    "🧪 Moleküller",
                    molekuller,
                    key="molekuller_filtresi",
                    tumunu_sec_varsayilan=True
                )
                if secilen_molekuller and "Tümü" not in secilen_molekuller:
                    filtre_ayar['Molekül'] = secilen_molekuller
            
            st.markdown("---")
            st.markdown('<div class="filter-title">📊 Sayısal Filtreler</div>', unsafe_allow_html=True)
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                min_satis = float(df[son_satis_sutun].min())
                max_satis = float(df[son_satis_sutun].max())
                
                satis_araligi = st.slider(
                    f"Satış Filtresi ({son_satis_sutun})",
                    min_value=min_satis,
                    max_value=max_satis,
                    value=(min_satis, max_satis),
                    key="satis_filtresi"
                )
                filtre_ayar['satis_araligi'] = (satis_araligi, son_satis_sutun)
            
            buyume_sutunlari = [sutun for sutun in df.columns if 'Büyüme_' in sutun]
            if buyume_sutunlari:
                son_buyume_sutun = buyume_sutunlari[-1]
                min_buyume = float(df[son_buyume_sutun].min())
                max_buyume = float(df[son_buyume_sutun].max())
                
                buyume_araligi = st.slider(
                    f"Büyüme Filtresi ({son_buyume_sutun})",
                    min_value=min_buyume,
                    max_value=max_buyume,
                    value=(min(min_buyume, -50.0), max(max_buyume, 150.0)),
                    key="buyume_filtresi"
                )
                filtre_ayar['buyume_araligi'] = (buyume_araligi, son_buyume_sutun)
            
            st.markdown("---")
            st.markdown('<div class="filter-title">⚙️ Ek Filtreler</div>', unsafe_allow_html=True)
            
            if 'International_Product' in df.columns:
                intl_filtre = st.selectbox(
                    "International Product",
                    ["Tümü", "Sadece International", "Sadece Yerel"],
                    key="intl_filtre"
                )
                if intl_filtre != "Tümü":
                    filtre_ayar['international_filtre'] = intl_filtre
            
            sadece_pozitif = st.checkbox("📈 Sadece Pozitif Büyüyen Ürünler", value=False)
            if sadece_pozitif and buyume_sutunlari:
                filtre_ayar['pozitif_buyume'] = True
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                filtre_uygula = st.button("✅ Filtre Uygula", use_container_width=True, key="filtre_uygula")
            with col2:
                filtre_temizle = st.button("🗑️ Filtreleri Temizle", use_container_width=True, key="filtre_temizle")
            
            return arama_terimi, filtre_ayar, filtre_uygula, filtre_temizle
    
    @staticmethod
    def arama_yapilabilir_coklu_secim(etiket, secenekler, key, tumunu_sec_varsayilan=False):
        """Arama yapılabilir multiselect"""
        if not secenekler:
            return []
        
        tum_secenekler = ["Tümü"] + secenekler
        
        arama_sorgu = st.text_input(f"{etiket} Ara", key=f"{key}_arama", placeholder="Arama yapın...")
        
        if arama_sorgu:
            filtrelenmis_secenekler = ["Tümü"] + [opt for opt in secenekler if arama_sorgu.lower() in str(opt).lower()]
        else:
            filtrelenmis_secenekler = tum_secenekler
        
        if tumunu_sec_varsayilan:
            varsayilan_secenekler = ["Tümü"]
        else:
            varsayilan_secenekler = filtrelenmis_secenekler[:min(5, len(filtrelenmis_secenekler))]
        
        secilenler = st.multiselect(
            etiket,
            options=filtrelenmis_secenekler,
            default=varsayilan_secenekler,
            key=key,
            help="'Tümü' seçildiğinde diğer tüm seçenekler otomatik seçilir"
        )
        
        if "Tümü" in secilenler and len(secilenler) > 1:
            secilenler = [opt for opt in secilenler if opt != "Tümü"]
        elif "Tümü" in secilenler and len(secilenler) == 1:
            secilenler = secenekler
        
        if secilenler:
            if len(secilenler) == len(secenekler):
                st.caption(f"✅ TÜMÜ seçildi ({len(secenekler)} öğe)")
            else:
                st.caption(f"✅ {len(secilenler)} / {len(secenekler)} seçildi")
        
        return secilenler
    
    @staticmethod
    def filtreleri_uygula(df, arama_terimi, filtre_ayar):
        """Filtreleri uygula"""
        filtrelenmis_df = df.copy()
        
        if arama_terimi:
            arama_maskesi = pd.Series(False, index=filtrelenmis_df.index)
            for sutun in filtrelenmis_df.columns:
                try:
                    arama_maskesi = arama_maskesi | filtrelenmis_df[sutun].astype(str).str.contains(
                        arama_terimi, case=False, na=False
                    )
                except:
                    continue
            filtrelenmis_df = filtrelenmis_df[arama_maskesi]
        
        for sutun, degerler in filtre_ayar.items():
            if sutun in filtrelenmis_df.columns and degerler and sutun not in ['satis_araligi', 'buyume_araligi', 'pozitif_buyume', 'international_filtre']:
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df[sutun].isin(degerler)]
        
        if 'satis_araligi' in filtre_ayar:
            (min_deger, max_deger), sutun_adi = filtre_ayar['satis_araligi']
            if sutun_adi in filtrelenmis_df.columns:
                filtrelenmis_df = filtrelenmis_df[
                    (filtrelenmis_df[sutun_adi] >= min_deger) & 
                    (filtrelenmis_df[sutun_adi] <= max_deger)
                ]
        
        if 'buyume_araligi' in filtre_ayar:
            (min_deger, max_deger), sutun_adi = filtre_ayar['buyume_araligi']
            if sutun_adi in filtrelenmis_df.columns:
                filtrelenmis_df = filtrelenmis_df[
                    (filtrelenmis_df[sutun_adi] >= min_deger) & 
                    (filtrelenmis_df[sutun_adi] <= max_deger)
                ]
        
        if 'international_filtre' in filtre_ayar and 'International_Product' in filtrelenmis_df.columns:
            if filtre_ayar['international_filtre'] == "Sadece International":
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df['International_Product'] == 1]
            elif filtre_ayar['international_filtre'] == "Sadece Yerel":
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df['International_Product'] == 0]
        
        if 'pozitif_buyume' in filtre_ayar and filtre_ayar['pozitif_buyume']:
            buyume_sutunlari = [sutun for sutun in filtrelenmis_df.columns if 'Büyüme_' in sutun]
            if buyume_sutunlari:
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df[buyume_sutunlari[-1]] > 0]
        
        return filtrelenmis_df

# ================================================
# 4. GÖRSELLEŞTİRME MOTORU (HATALARI DÜZELTİLMİŞ)
# ================================================

class ProfesyonelGorsellestirme:
    """Profesyonel görselleştirme motoru"""
    
    @staticmethod
    def dashboard_metrikleri_olustur(df, metrikler):
        """Dashboard metrik kartlarını oluştur"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                toplam_satis = metrikler.get('Toplam_Pazar_Değeri', 0)
                satis_yili = metrikler.get('Son_Satis_Yılı', '')
                st.markdown(f"""
                <div class="custom-metric-card primary">
                    <div class="custom-metric-label">TOPLAM PAZAR DEĞERİ</div>
                    <div class="custom-metric-value">${toplam_satis/1e6:.1f}M</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">{satis_yili}</span>
                        <span>Toplam Pazar</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                ort_buyume = metrikler.get('Ort_Buyume_Oranı', 0)
                buyume_class = "success" if ort_buyume > 0 else "danger"
                st.markdown(f"""
                <div class="custom-metric-card {buyume_class}">
                    <div class="custom-metric-label">ORTALAMA BÜYÜME</div>
                    <div class="custom-metric-value">{ort_buyume:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Yıllık</span>
                        <span>YoY Büyüme</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrikler.get('HHI_Indeksi', 0)
                hhi_durum = "danger" if hhi > 2500 else "warning" if hhi > 1500 else "success"
                hhi_metin = "Monopol" if hhi > 2500 else "Oligopol" if hhi > 1500 else "Rekabetçi"
                st.markdown(f"""
                <div class="custom-metric-card {hhi_durum}">
                    <div class="custom-metric-label">REKABET YOĞUNLUĞU</div>
                    <div class="custom-metric-value">{hhi:.0f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-warning">HHI Index</span>
                        <span>{hhi_metin}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                intl_payi = metrikler.get('International_Product_Payı', 0)
                intl_renk = "success" if intl_payi > 20 else "warning" if intl_payi > 10 else "info"
                st.markdown(f"""
                <div class="custom-metric-card {intl_renk}">
                    <div class="custom-metric-label">INTERNATIONAL PRODUCT</div>
                    <div class="custom-metric-value">{intl_payi:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Global</span>
                        <span>Çoklu Pazar</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                benzersiz_molekul = metrikler.get('Benzersiz_Molekül', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">MOLEKÜL ÇEŞİTLİLİĞİ</div>
                    <div class="custom-metric-value">{benzersiz_molekul:,}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">Benzersiz</span>
                        <span>Farklı Molekül</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                ort_fiyat = metrikler.get('Ort_Fiyat', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">ORTALAMA FİYAT</div>
                    <div class="custom-metric-value">${ort_fiyat:.2f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Birim Başına</span>
                        <span>Ortalama</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                yuksek_buyume = metrikler.get('Yuksek_Buyume_Yüzdesi', 0)
                st.markdown(f"""
                <div class="custom-metric-card success">
                    <div class="custom-metric-label">YÜKSEK BÜYÜME</div>
                    <div class="custom-metric-value">{yuksek_buyume:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">%20+</span>
                        <span>Hızlı Büyüyen</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                ulke_kapsami = metrikler.get('Ülke_Kapsamı', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">COĞRAFİ YAYILIM</div>
                    <div class="custom-metric-value">{ulke_kapsami}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Ülke</span>
                        <span>Global Kapsam</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Metrik kartları oluşturma hatası: {str(e)}")
    
    @staticmethod
    def satis_trend_grafigi(df):
        """Satış trend grafikleri"""
        try:
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if len(satis_sutunlari) >= 2:
                yillik_veri = []
                for sutun in sorted(satis_sutunlari):
                    yil = sutun.split('_')[-1]
                    yillik_veri.append({
                        'Yıl': yil,
                        'Toplam_Satış': df[sutun].sum(),
                        'Ort_Satış': df[sutun].mean(),
                        'Ürün_Sayısı': (df[sutun] > 0).sum()
                    })
                
                yillik_df = pd.DataFrame(yillik_veri)
                
                fig = go.Figure()
                
                # Toplam satış
                fig.add_trace(go.Bar(
                    x=yillik_df['Yıl'],
                    y=yillik_df['Toplam_Satış'],
                    name='Toplam Satış',
                    marker_color='#2d7dd2',
                    text=[f'${x/1e6:.0f}M' for x in yillik_df['Toplam_Satış']],
                    textposition='auto'
                ))
                
                # Ortalama satış (ikinci eksen)
                fig.add_trace(go.Scatter(
                    x=yillik_df['Yıl'],
                    y=yillik_df['Ort_Satış'],
                    name='Ortalama Satış',
                    mode='lines+markers',
                    line=dict(color='#2acaea', width=3),
                    marker=dict(size=10),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title='Satış Trendleri Analizi',
                    xaxis_title='Yıl',
                    yaxis_title='Toplam Satış (USD)',
                    yaxis2=dict(
                        title='Ortalama Satış (USD)',
                        overlaying='y',
                        side='right'
                    ),
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.warning(f"Trend grafiği oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def pazar_payi_analizi(df):
        """Pazar payı analiz grafikleri"""
        try:
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            
            # Şirket bazlı pazar payı
            if 'Şirket' in df.columns:
                sirket_satis = df.groupby('Şirket')[son_satis_sutun].sum().sort_values(ascending=False)
                top_sirketler = sirket_satis.nlargest(15)
                diger_satis = sirket_satis.iloc[15:].sum() if len(sirket_satis) > 15 else 0
                
                pasta_verisi = top_sirketler.copy()
                if diger_satis > 0:
                    pasta_verisi['Diğer'] = diger_satis
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Pazar Payı Dağılımı', 'Top 10 Şirket Satışları'),
                    specs=[[{'type': 'domain'}, {'type': 'bar'}]],
                    column_widths=[0.4, 0.6]
                )
                
                fig.add_trace(
                    go.Pie(
                        labels=pasta_verisi.index,
                        values=pasta_verisi.values,
                        hole=0.4,
                        marker_colors=px.colors.qualitative.Bold,
                        textinfo='percent+label',
                        textposition='outside'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=top_sirketler.values[:10],
                        y=top_sirketler.index[:10],
                        orientation='h',
                        marker_color='#2d7dd2',
                        text=[f'${x/1e6:.1f}M' for x in top_sirketler.values[:10]],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    showlegend=False,
                    title_text="Pazar Konsantrasyonu Analizi",
                    title_x=0.5
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.warning(f"Pazar payı grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def fiyat_hacim_analizi(df):
        """Fiyat-hacim analiz grafikleri"""
        try:
            # Fiyat ve birim sütunlarını bul
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            birim_sutunlari = [sutun for sutun in df.columns if 'Birim_' in sutun]
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            
            # Eğer Ort_Fiyat sütunu yoksa ama Satış ve Birim sütunları varsa, hesapla
            if not fiyat_sutunlari and satis_sutunlari and birim_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                son_birim_sutun = birim_sutunlari[-1]
                
                if son_satis_sutun in df.columns and son_birim_sutun in df.columns:
                    df = df.copy()  # DataFrame'in kopyasını al
                    df['Hesaplanan_Ort_Fiyat'] = np.where(
                        df[son_birim_sutun] != 0,
                        df[son_satis_sutun] / df[son_birim_sutun],
                        np.nan
                    )
                    fiyat_sutunlari = ['Hesaplanan_Ort_Fiyat']
            
            if not fiyat_sutunlari or not birim_sutunlari:
                st.info("Fiyat-hacim analizi için gerekli sütunlar bulunamadı.")
                return None
            
            son_fiyat_sutun = fiyat_sutunlari[-1]
            son_birim_sutun = birim_sutunlari[-1]
            
            # Benzersiz sütun isimleri oluştur
            temp_fiyat_sutun = 'Fiyat_' + str(hash(son_fiyat_sutun))[:6]
            temp_birim_sutun = 'Hacim_' + str(hash(son_birim_sutun))[:6]
            
            # Veri hazırlama - DataFrame'in kopyasını al
            ornek_df = df.copy()
            ornek_df[temp_fiyat_sutun] = ornek_df[son_fiyat_sutun]
            ornek_df[temp_birim_sutun] = ornek_df[son_birim_sutun]
            
            ornek_df = ornek_df[
                (ornek_df[temp_fiyat_sutun] > 0) & 
                (ornek_df[temp_birim_sutun] > 0)
            ].copy()
            
            if len(ornek_df) == 0:
                st.info("Fiyat ve hacim değerleri olan ürün bulunamadı.")
                return None
            
            if len(ornek_df) > 10000:
                ornek_df = ornek_df.sample(10000, random_state=42)
            
            # Hover için isim belirle
            hover_columns = []
            if 'Molekül' in ornek_df.columns:
                hover_columns.append('Molekül')
            elif 'Şirket' in ornek_df.columns:
                hover_columns.append('Şirket')
            
            # Scatter plot
            fig = px.scatter(
                ornek_df,
                x=temp_fiyat_sutun,
                y=temp_birim_sutun,
                size=temp_birim_sutun,
                color=temp_fiyat_sutun,
                hover_name=hover_columns[0] if hover_columns else None,
                title='Fiyat-Hacim İlişkisi',
                labels={
                    temp_fiyat_sutun: 'Fiyat (USD)',
                    temp_birim_sutun: 'Hacim (Birim)'
                },
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Fiyat-hacim grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def fiyat_esneklik_analizi(df):
        """Fiyat esnekliği analizi"""
        try:
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            birim_sutunlari = [sutun for sutun in df.columns if 'Birim_' in sutun]
            
            if not fiyat_sutunlari or not birim_sutunlari:
                return None
            
            son_fiyat_sutun = fiyat_sutunlari[-1]
            son_birim_sutun = birim_sutunlari[-1]
            
            # Korelasyon analizi
            korelasyon_df = df[[son_fiyat_sutun, son_birim_sutun]].dropna()
            
            if len(korelasyon_df) < 10:
                return None
            
            korelasyon = korelasyon_df.corr().iloc[0, 1]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'Fiyat-Hacim Korelasyonu: {korelasyon:.3f}',
                    'Fiyat Dağılımı',
                    'Hacim Dağılımı',
                    'Fiyat Segmentleri'
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=korelasyon_df[son_fiyat_sutun],
                    y=korelasyon_df[son_birim_sutun],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=korelasyon_df[son_birim_sutun],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Ürünler'
                ),
                row=1, col=1
            )
            
            # Fiyat dağılımı
            fig.add_trace(
                go.Histogram(
                    x=df[son_fiyat_sutun],
                    nbinsx=50,
                    marker_color='#2d7dd2',
                    name='Fiyat'
                ),
                row=1, col=2
            )
            
            # Hacim dağılımı
            fig.add_trace(
                go.Histogram(
                    x=df[son_birim_sutun],
                    nbinsx=50,
                    marker_color='#2acaea',
                    name='Hacim'
                ),
                row=2, col=1
            )
            
            # Fiyat segmentleri
            fiyat_verisi = df[son_fiyat_sutun].dropna()
            if len(fiyat_verisi) > 0:
                segmentler = pd.cut(
                    fiyat_verisi,
                    bins=[0, 10, 50, 100, 500, float('inf')],
                    labels=['Ekonomi (<$10)', 'Standart ($10-$50)', 'Premium ($50-$100)', 
                           'Süper Premium ($100-$500)', 'Lüks (>$500)']
                )
                
                segment_sayilari = segmentler.value_counts()
                fig.add_trace(
                    go.Bar(
                        x=segment_sayilari.index,
                        y=segment_sayilari.values,
                        marker_color='#2dd2a3',
                        name='Segment'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Fiyat esnekliği grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def international_product_grafikleri(df, analiz_df):
        """International Product analiz grafikleri"""
        try:
            if analiz_df is None or len(analiz_df) == 0:
                return None
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'International vs Local Dağılımı',
                    'Satış Dağılımı',
                    'Coğrafi Yayılım',
                    'Büyüme Karşılaştırması'
                ),
                specs=[
                    [{'type': 'domain'}, {'type': 'xy'}],
                    [{'type': 'xy'}, {'type': 'xy'}]
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # 1. Pasta grafiği - International vs Local
            if 'International' in analiz_df.columns:
                intl_sayisi = (analiz_df['International'] == True).sum() if 'International' in analiz_df.columns else 0
                local_sayisi = len(analiz_df) - intl_sayisi
            else:
                intl_sayisi = 0
                local_sayisi = len(analiz_df)
            
            fig.add_trace(
                go.Pie(
                    labels=['International', 'Local'],
                    values=[intl_sayisi, local_sayisi],
                    hole=0.4,
                    marker_colors=['#2d7dd2', '#64748b'],
                    textinfo='percent+label'
                ),
                row=1, col=1
            )
            
            # 2. Satış dağılımı
            if 'International' in analiz_df.columns and 'Toplam_Satış' in analiz_df.columns:
                intl_satis = analiz_df[analiz_df['International']]['Toplam_Satış'].sum()
                local_satis = analiz_df[~analiz_df['International']]['Toplam_Satış'].sum()
            else:
                intl_satis = 0
                local_satis = analiz_df['Toplam_Satış'].sum() if 'Toplam_Satış' in analiz_df.columns else 0
            
            fig.add_trace(
                go.Bar(
                    x=['International', 'Local'],
                    y=[intl_satis, local_satis],
                    marker_color=['#2d7dd2', '#64748b'],
                    text=[f'${intl_satis/1e6:.1f}M' if intl_satis > 0 else '$0', 
                          f'${local_satis/1e6:.1f}M' if local_satis > 0 else '$0'],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 3. Coğrafi yayılım
            if 'Ülke_Sayısı' in analiz_df.columns and 'International' in analiz_df.columns:
                intl_df = analiz_df[analiz_df['International']]
                if len(intl_df) > 0:
                    ulke_dagilimi = intl_df['Ülke_Sayısı'].value_counts().sort_index()
                    fig.add_trace(
                        go.Bar(
                            x=ulke_dagilimi.index.astype(str),
                            y=ulke_dagilimi.values,
                            marker_color='#2acaea',
                            name='Ülke Sayısı'
                        ),
                        row=2, col=1
                    )
            
            # 4. Büyüme karşılaştırması
            if 'Ortalama_Büyüme' in analiz_df.columns and 'International' in analiz_df.columns:
                intl_buyume = analiz_df[analiz_df['International']]['Ortalama_Büyüme'].mean()
                local_buyume = analiz_df[~analiz_df['International']]['Ortalama_Büyüme'].mean()
                
                if not pd.isna(intl_buyume) and not pd.isna(local_buyume):
                    fig.add_trace(
                        go.Bar(
                            x=['International', 'Local'],
                            y=[intl_buyume, local_buyume],
                            marker_color=['#2d7dd2', '#64748b'],
                            text=[f'{intl_buyume:.1f}%', f'{local_buyume:.1f}%'],
                            textposition='auto'
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                title_text="International Product Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"International Product grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def rekabet_analizi_grafikleri(df):
        """Rekabet analizi grafikleri"""
        try:
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            
            if 'Şirket' not in df.columns:
                return None
            
            sirket_satis = df.groupby('Şirket')[son_satis_sutun].sum().sort_values(ascending=False)
            top_sirketler = sirket_satis.nlargest(10)
            
            # Treemap için veri hazırlama
            if 'Molekül' in df.columns:
                treemap_data = df.groupby(['Şirket', 'Molekül'])[son_satis_sutun].sum().reset_index()
                
                # Büyüme oranı ekle
                buyume_sutunlari = [sutun for sutun in df.columns if 'Büyüme_' in sutun]
                if buyume_sutunlari:
                    son_buyume_sutun = buyume_sutunlari[-1]
                    sirket_buyume = df.groupby('Şirket')[son_buyume_sutun].mean().reset_index()
                    treemap_data = treemap_data.merge(sirket_buyume, on='Şirket', how='left')
                    color_column = son_buyume_sutun
                else:
                    treemap_data['Ortalama_Büyüme'] = 0
                    color_column = 'Ortalama_Büyüme'
            else:
                treemap_data = pd.DataFrame()
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Top 10 Pazar Liderleri', 'Pazar Hakimiyet Haritası'),
                specs=[[{'type': 'bar'}, {'type': 'treemap'}]],
                column_widths=[0.5, 0.5]
            )
            
            # Bar Chart - Pazar Liderleri
            fig.add_trace(
                go.Bar(
                    x=top_sirketler.values,
                    y=top_sirketler.index,
                    orientation='h',
                    marker_color='#2d7dd2',
                    text=[f'${x/1e6:.1f}M' for x in top_sirketler.values],
                    textposition='auto',
                    name='Pazar Liderleri'
                ),
                row=1, col=1
            )
            
            # Treemap - Pazar Hakimiyet Haritası
            if not treemap_data.empty and 'Molekül' in treemap_data.columns:
                treemap_fig = px.treemap(
                    treemap_data,
                    path=['Şirket', 'Molekül'],
                    values=son_satis_sutun,
                    color=color_column,
                    color_continuous_scale='Viridis',
                    title='Şirket-Molekül Hiyerarşisi',
                    hover_data=[son_satis_sutun, color_column]
                )
                
                fig.add_trace(
                    treemap_fig.data[0],
                    row=1, col=2
                )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                title_text="Rekabet Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Rekabet analizi grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def lorenz_egrisi_olustur(sirket_satis):
        """Lorenz Eğrisi - Pazar Tekelleşme Analizi"""
        try:
            # Eğer sirket_satis Series değilse dönüştür
            if isinstance(sirket_satis, pd.DataFrame):
                st.warning("Şirket satışları DataFrame olarak geldi, Series'e dönüştürülüyor")
                sirket_satis = sirket_satis.iloc[:, 0]  # İlk sütunu al
            
            sorted_sales = np.sort(sirket_satis.values)
            cum_sales = np.cumsum(sorted_sales)
            
            # Eğer toplam sıfırsa hata vermeyi önle
            if cum_sales[-1] == 0:
                st.info("Lorenz eğrisi için sıfır olmayan satış verisi gerekiyor")
                return None
            
            cum_percentage_sales = cum_sales / cum_sales[-1]
            
            perfect_line = np.linspace(0, 1, len(cum_percentage_sales))
            
            # Gini katsayısını hesapla - scipy.integrate.trapz kullan
            gini_coefficient = 1 - 2 * integrate.trapz(cum_percentage_sales, perfect_line)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=np.linspace(0, 1, len(cum_percentage_sales)),
                y=cum_percentage_sales,
                mode='lines',
                line=dict(color='#2acaea', width=3),
                name=f'Lorenz Eğrisi (Gini: {gini_coefficient:.3f})',
                fill='tozeroy',
                fillcolor='rgba(42, 202, 234, 0.3)'
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='#f8fafc', width=2, dash='dash'),
                name='Tam Eşitlik'
            ))
            
            fig.update_layout(
                title='Lorenz Eğrisi - Pazar Konsantrasyonu',
                xaxis_title='Şirketlerin Kümülatif Oranı',
                yaxis_title='Satışların Kümülatif Oranı',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Lorenz eğrisi oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def dunya_haritasi_olustur(df):
        """Coğrafi Dağılım Dünya Haritası"""
        try:
            if 'Ülke' not in df.columns:
                return None
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            
            ulke_satis = df.groupby('Ülke')[son_satis_sutun].sum().reset_index()
            ulke_satis.columns = ['Country', 'Total_Sales']
            
            country_mapping = {
                'USA': 'United States',
                'US': 'United States',
                'U.S.A': 'United States',
                'United States of America': 'United States',
                'UK': 'United Kingdom',
                'U.K': 'United Kingdom',
                'United Kingdom of Great Britain': 'United Kingdom',
                'UAE': 'United Arab Emirates',
                'U.A.E': 'United Arab Emirates',
                'S. Korea': 'South Korea',
                'South Korea': 'Korea, Republic of',
                'North Korea': 'Korea, Democratic People\'s Republic of',
                'Russia': 'Russian Federation',
                'Russian Federation': 'Russian Federation',
                'Iran': 'Iran, Islamic Republic of',
                'Vietnam': 'Viet Nam',
                'Syria': 'Syrian Arab Republic',
                'Laos': 'Lao People\'s Democratic Republic',
                'Bolivia': 'Bolivia, Plurinational State of',
                'Venezuela': 'Venezuela, Bolivarian Republic of',
                'Tanzania': 'Tanzania, United Republic of',
                'Moldova': 'Moldova, Republic of',
                'Macedonia': 'North Macedonia'
            }
            
            ulke_satis['Country'] = ulke_satis['Country'].replace(country_mapping)
            
            fig = px.choropleth(
                ulke_satis,
                locations='Country',
                locationmode='country names',
                color='Total_Sales',
                hover_name='Country',
                hover_data={'Total_Sales': ':.2f'},
                color_continuous_scale='Viridis',
                title='Global İlaç Pazarı Dağılımı - Coğrafi Yayılım',
                projection='natural earth'
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                geo=dict(
                    bgcolor='rgba(0,0,0,0)',
                    lakecolor='#1e3a5f',
                    landcolor='#2d4a7a',
                    subunitcolor='#64748b'
                ),
                coloraxis_colorbar=dict(
                    title="Toplam Satış (USD)",
                    tickprefix="$",
                    ticksuffix=""
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Dünya haritası oluşturma hatası: {str(e)}")
            return None

# ================================================
# 5. ANA UYGULAMA
# ================================================

def main():
    """Ana uygulama fonksiyonu"""
    
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="pharma-title">💊 PHARMAINTELLIGENCE PRO</h1>
        <p class="pharma-subtitle">
        Enterprise-level pharmaceutical market analytics platform with International Product analysis, 
        advanced filtering, predictive insights, and strategic recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Session state başlatma
    if 'veri' not in st.session_state:
        st.session_state.veri = None
    if 'filtrelenmis_veri' not in st.session_state:
        st.session_state.filtrelenmis_veri = None
    if 'metrikler' not in st.session_state:
        st.session_state.metrikler = None
    if 'icgoruler' not in st.session_state:
        st.session_state.icgoruler = []
    if 'aktif_filtreler' not in st.session_state:
        st.session_state.aktif_filtreler = {}
    if 'international_analiz' not in st.session_state:
        st.session_state.international_analiz = None
    
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
        
        with st.expander("📁 VERİ YÜKLEME", expanded=True):
            yuklenen_dosya = st.file_uploader(
                "Excel/CSV Dosyası Yükleyin",
                type=['xlsx', 'xls', 'csv'],
                help="1M+ satır desteklenir. Büyük dosyalar için dikkatli olun."
            )
            
            if yuklenen_dosya:
                st.info("⚠️ Tüm veri seti yüklenecektir")
                st.info(f"Dosya: {yuklenen_dosya.name}")
                
                if st.button("🚀 Tüm Veriyi Yükle & Analiz Et", type="primary", use_container_width=True):
                    with st.spinner("Tüm veri seti işleniyor..."):
                        isleyici = OptimizeVeriİşleyici()
                        
                        veri = isleyici.buyuk_veri_yukle(yuklenen_dosya, orneklem=None)
                        
                        if veri is not None and len(veri) > 0:
                            veri = isleyici.analiz_verisi_hazirla(veri)
                            
                            st.session_state.veri = veri
                            st.session_state.filtrelenmis_veri = veri.copy()
                            
                            # Analitik sınıfını import et
                            from sklearn.preprocessing import StandardScaler
                            from sklearn.cluster import KMeans, DBSCAN
                            
                            # Basit metrikler hesapla
                            satis_sutunlari = [sutun for sutun in veri.columns if 'Satış_' in sutun]
                            buyume_sutunlari = [sutun for sutun in veri.columns if 'Büyüme_' in sutun]
                            
                            metrikler = {}
                            if satis_sutunlari:
                                son_satis_sutun = satis_sutunlari[-1]
                                metrikler['Toplam_Pazar_Değeri'] = veri[son_satis_sutun].sum()
                                metrikler['Son_Satis_Yılı'] = son_satis_sutun.split('_')[-1]
                            
                            if buyume_sutunlari:
                                son_buyume_sutun = buyume_sutunlari[-1]
                                metrikler['Ort_Buyume_Oranı'] = veri[son_buyume_sutun].mean()
                            
                            metrikler['Toplam_Satır'] = len(veri)
                            metrikler['Toplam_Sütun'] = len(veri.columns)
                            
                            if 'International_Product' in veri.columns:
                                intl_df = veri[veri['International_Product'] == 1]
                                metrikler['International_Product_Sayısı'] = len(intl_df)
                                if satis_sutunlari:
                                    metrikler['International_Product_Satış'] = intl_df[son_satis_sutun].sum()
                                    metrikler['International_Product_Payı'] = (metrikler['International_Product_Satış'] / metrikler['Toplam_Pazar_Değeri'] * 100) if metrikler['Toplam_Pazar_Değeri'] > 0 else 0
                            
                            st.session_state.metrikler = metrikler
                            st.session_state.icgoruler = []
                            
                            st.success(f"✅ {len(veri):,} satır TÜM VERİ başarıyla yüklendi!")
                            st.rerun()
        
        if st.session_state.veri is not None:
            veri = st.session_state.veri
            
            # Basit filtreleme
            with st.expander("🔍 TEMEL FİLTRELEME", expanded=True):
                st.markdown('<div class="filter-title">Arama ve Filtreleme</div>', unsafe_allow_html=True)
                
                arama_terimi = st.text_input(
                    "Genel Arama",
                    placeholder="Molekül, Şirket, Ülke...",
                    key="genel_arama_simple"
                )
                
                # Ülke filtreleme
                if 'Ülke' in veri.columns:
                    ulkeler = sorted(veri['Ülke'].dropna().unique())
                    secilen_ulkeler = st.multiselect(
                        "Ülkeler",
                        options=ulkeler,
                        default=ulkeler[:min(5, len(ulkeler))],
                        key="ulkeler_simple"
                    )
                
                # Şirket filtreleme
                if 'Şirket' in veri.columns:
                    sirketler = sorted(veri['Şirket'].dropna().unique())
                    secilen_sirketler = st.multiselect(
                        "Şirketler",
                        options=sirketler,
                        default=sirketler[:min(5, len(sirketler))],
                        key="sirketler_simple"
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    filtre_uygula = st.button("✅ Filtre Uygula", use_container_width=True, key="filtre_uygula_simple")
                with col2:
                    filtre_temizle = st.button("🗑️ Filtreleri Temizle", use_container_width=True, key="filtre_temizle_simple")
                
                if filtre_uygula:
                    filtrelenmis_veri = veri.copy()
                    
                    # Arama terimi uygula
                    if arama_terimi:
                        mask = pd.Series(False, index=filtrelenmis_veri.index)
                        for sutun in filtrelenmis_veri.columns:
                            try:
                                mask = mask | filtrelenmis_veri[sutun].astype(str).str.contains(arama_terimi, case=False, na=False)
                            except:
                                continue
                        filtrelenmis_veri = filtrelenmis_veri[mask]
                    
                    # Ülke filtreleme
                    if 'Ülke' in veri.columns and secilen_ulkeler:
                        filtrelenmis_veri = filtrelenmis_veri[filtrelenmis_veri['Ülke'].isin(secilen_ulkeler)]
                    
                    # Şirket filtreleme
                    if 'Şirket' in veri.columns and secilen_sirketler:
                        filtrelenmis_veri = filtrelenmis_veri[filtrelenmis_veri['Şirket'].isin(secilen_sirketler)]
                    
                    st.session_state.filtrelenmis_veri = filtrelenmis_veri
                    st.success(f"✅ Filtre uygulandı: {len(filtrelenmis_veri):,} satır")
                    st.rerun()
                
                if filtre_temizle:
                    st.session_state.filtrelenmis_veri = st.session_state.veri.copy()
                    st.success("✅ Filtreler temizlendi")
                    st.rerun()
    
    if st.session_state.veri is None:
        hosgeldiniz_ekrani_goster()
        return
    
    veri = st.session_state.filtrelenmis_veri
    metrikler = st.session_state.metrikler
    icgoruler = st.session_state.icgoruler
    
    # Tablar
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ",
        "💰 FİYAT ANALİZİ",
        "🏆 REKABET ANALİZİ",
        "🌍 INTERNATIONAL PRODUCT",
        "🔮 STRATEJİK ANALİZ",
        "📑 RAPORLAMA"
    ])
    
    with tab1:
        genel_bakis_tab_goster(veri, metrikler, icgoruler)
    
    with tab2:
        pazar_analizi_tab_goster(veri)
    
    with tab3:
        fiyat_analizi_tab_goster(veri)
    
    with tab4:
        rekabet_analizi_tab_goster(veri, metrikler)
    
    with tab5:
        international_product_tab_goster(veri, metrikler)
    
    with tab6:
        stratejik_analiz_tab_goster(veri, icgoruler)
    
    with tab7:
        raporlama_tab_goster(veri, metrikler, icgoruler)

# ================================================
# TAB FONKSİYONLARI
# ================================================

def hosgeldiniz_ekrani_goster():
    """Hoşgeldiniz ekranını göster"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">💊</div>
            <h2 style="color: #f1f5f9; margin-bottom: 1rem;">PharmaIntelligence Pro'ya Hoşgeldiniz</h2>
            <p style="color: #cbd5e1; margin-bottom: 2rem; line-height: 1.6;">
            İlaç pazarı verilerinizi yükleyin ve güçlü analitik özelliklerin kilidini açın.
            <br>International Product analizi ile çoklu pazar stratejilerinizi optimize edin.
            </p>
            
            <div class="feature-grid">
                <div class="feature-card feature-card-blue">
                    <div class="feature-icon">🌍</div>
                    <div class="feature-title">International Product</div>
                    <div class="feature-description">Çoklu pazar ürün analizi ve strateji geliştirme</div>
                </div>
                <div class="feature-card feature-card-cyan">
                    <div class="feature-icon">📈</div>
                    <div class="feature-title">Pazar Analizi</div>
                    <div class="feature-description">Derin pazar içgörüleri ve trend analizi</div>
                </div>
                <div class="feature-card feature-card-teal">
                    <div class="feature-icon">💰</div>
                    <div class="feature-title">Fiyat Zekası</div>
                    <div class="feature-description">Rekabetçi fiyatlandırma ve optimizasyon analizi</div>
                </div>
                <div class="feature-card feature-card-warning">
                    <div class="feature-icon">🏆</div>
                    <div class="feature-title">Rekabet Analizi</div>
                    <div class="feature-description">Rakiplerinizi analiz edin ve fırsatları belirleyin</div>
                </div>
            </div>
            
            <div class="get-started-box">
                <div class="get-started-title">🎯 Başlamak İçin</div>
                <div class="get-started-steps">
                1. Sol taraftaki panelden veri dosyanızı yükleyin<br>
                2. "Tüm Veriyi Yükle & Analiz Et" butonuna tıklayın<br>
                3. Analiz sonuçlarını görmek için tabları kullanın
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def genel_bakis_tab_goster(df, metrikler, icgoruler):
    """Genel Bakış tab'ını göster"""
    st.markdown('<h2 class="section-title">Genel Bakış ve Performans Göstergeleri</h2>', unsafe_allow_html=True)
    
    gorsellestirme = ProfesyonelGorsellestirme()
    gorsellestirme.dashboard_metrikleri_olustur(df, metrikler)
    
    st.markdown('<h3 class="subsection-title">🔍 Veri Önizleme</h3>', unsafe_allow_html=True)
    
    onizleme_col1, onizleme_col2 = st.columns([1, 3])
    
    with onizleme_col1:
        satir_sayisi = st.slider("Gösterilecek Satır Sayısı", 10, 1000, 100, 10, key="satir_onizleme")
        
        mevcut_sutunlar = df.columns.tolist()
        varsayilan_sutunlar = []
        
        oncelikli_sutunlar = ['Molekül', 'Şirket', 'Ülke', 'Satış_2024', 'Büyüme_2023_2024']
        for sutun in oncelikli_sutunlar:
            if sutun in mevcut_sutunlar:
                varsayilan_sutunlar.append(sutun)
            if len(varsayilan_sutunlar) >= 5:
                break
        
        if len(varsayilan_sutunlar) < 5:
            varsayilan_sutunlar.extend([sutun for sutun in mevcut_sutunlar[:5] if sutun not in varsayilan_sutunlar])
        
        gosterilecek_sutunlar = st.multiselect(
            "Gösterilecek Sütunlar",
            options=mevcut_sutunlar,
            default=varsayilan_sutunlar[:min(5, len(varsayilan_sutunlar))],
            key="sutun_onizleme"
        )
    
    with onizleme_col2:
        if gosterilecek_sutunlar:
            st.dataframe(
                df[gosterilecek_sutunlar].head(satir_sayisi),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(
                df.head(satir_sayisi),
                use_container_width=True,
                height=400
            )

def pazar_analizi_tab_goster(df):
    """Pazar Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Pazar Analizi ve Trendler</h2>', unsafe_allow_html=True)
    
    gorsellestirme = ProfesyonelGorsellestirme()
    
    st.markdown('<h3 class="subsection-title">📈 Satış Trendleri</h3>', unsafe_allow_html=True)
    trend_grafik = gorsellestirme.satis_trend_grafigi(df)
    if trend_grafik:
        st.plotly_chart(trend_grafik, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Satış trend analizi için yeterli yıllık veri bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">🏆 Pazar Payı Analizi</h3>', unsafe_allow_html=True)
    pazar_payi_grafik = gorsellestirme.pazar_payi_analizi(df)
    if pazar_payi_grafik:
        st.plotly_chart(pazar_payi_grafik, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Pazar payı analizi için gerekli veri bulunamadı.")

def fiyat_analizi_tab_goster(df):
    """Fiyat Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Fiyat Analizi ve Optimizasyon</h2>', unsafe_allow_html=True)
    
    gorsellestirme = ProfesyonelGorsellestirme()
    
    st.markdown('<h3 class="subsection-title">💰 Fiyat-Hacim İlişkisi</h3>', unsafe_allow_html=True)
    fiyat_hacim_grafik = gorsellestirme.fiyat_hacim_analizi(df)
    if fiyat_hacim_grafik:
        st.plotly_chart(fiyat_hacim_grafik, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Fiyat-hacim analizi için yeterli veri bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">📉 Fiyat Esnekliği Analizi</h3>', unsafe_allow_html=True)
    esneklik_grafik = gorsellestirme.fiyat_esneklik_analizi(df)
    if esneklik_grafik:
        st.plotly_chart(esneklik_grafik, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Fiyat esnekliği analizi için yeterli veri bulunamadı.")

def rekabet_analizi_tab_goster(df, metrikler):
    """Rekabet Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Rekabet Analizi ve Pazar Yapısı</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">📊 Rekabet Yoğunluğu Metrikleri</h3>', unsafe_allow_html=True)
    
    rekabet_sutunlar = st.columns(4)
    
    with rekabet_sutunlar[0]:
        hhi = metrikler.get('HHI_Indeksi', 0)
        if hhi > 2500:
            hhi_durum = "Monopolistik"
        elif hhi > 1800:
            hhi_durum = "Oligopol"
        else:
            hhi_durum = "Rekabetçi"
        st.metric("HHI İndeksi", f"{hhi:.0f}", hhi_durum)
    
    with rekabet_sutunlar[1]:
        top3_payi = metrikler.get('Top_3_Pay', 0)
        if top3_payi > 50:
            konsantrasyon = "Yüksek"
        elif top3_payi > 30:
            konsantrasyon = "Orta"
        else:
            konsantrasyon = "Düşük"
        st.metric("Top 3 Payı", f"{top3_payi:.1f}%", konsantrasyon)
    
    with rekabet_sutunlar[2]:
        top5_payi = metrikler.get('Top_5_Pay', 0)
        st.metric("Top 5 Payı", f"{top5_payi:.1f}%")
    
    with rekabet_sutunlar[3]:
        top10_molekul = metrikler.get('Top_10_Molekul_Payı', 0)
        st.metric("Top 10 Molekül Payı", f"{top10_molekul:.1f}%")
    
    st.markdown('<h3 class="subsection-title">📈 Rekabet Analizi Grafikleri</h3>', unsafe_allow_html=True)
    
    gorsellestirme = ProfesyonelGorsellestirme()
    rekabet_grafik = gorsellestirme.rekabet_analizi_grafikleri(df)
    
    if rekabet_grafik:
        st.plotly_chart(rekabet_grafik, use_container_width=True, config={'displayModeBar': True})
    
    # Lorenz Eğrisi
    if 'Şirket' in df.columns:
        satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
        if satis_sutunlari:
            son_satis_sutun = satis_sutunlari[-1]
            sirket_satis = df.groupby('Şirket')[son_satis_sutun].sum()
            lorenz_grafik = gorsellestirme.lorenz_egrisi_olustur(sirket_satis)
            if lorenz_grafik:
                st.plotly_chart(lorenz_grafik, use_container_width=True, config={'displayModeBar': True})

def international_product_tab_goster(df, metrikler):
    """International Product Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">🌍 International Product Analizi</h2>', unsafe_allow_html=True)
    
    if 'International_Product' not in df.columns:
        st.warning("International Product sütunu bulunamadı.")
        return
    
    gorsellestirme = ProfesyonelGorsellestirme()
    
    st.markdown('<h3 class="subsection-title">📊 International Product Genel Bakış</h3>', unsafe_allow_html=True)
    
    intl_sutunlar = st.columns(4)
    
    with intl_sutunlar[0]:
        intl_sayisi = (df['International_Product'] == 1).sum()
        toplam_urun = len(df)
        intl_yuzde = (intl_sayisi / toplam_urun * 100) if toplam_urun > 0 else 0
        st.metric("International Product Sayısı", f"{intl_sayisi}", f"%{intl_yuzde:.1f}")
    
    with intl_sutunlar[1]:
        satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
        if satis_sutunlari:
            son_satis_sutun = satis_sutunlari[-1]
            intl_satis = df[df['International_Product'] == 1][son_satis_sutun].sum()
            toplam_satis = df[son_satis_sutun].sum()
            intl_payi = (intl_satis / toplam_satis * 100) if toplam_satis > 0 else 0
            st.metric("Pazar Payı", f"%{intl_payi:.1f}")
    
    with intl_sutunlar[2]:
        if 'Ülke' in df.columns:
            intl_ulke_sayisi = df[df['International_Product'] == 1]['Ülke'].nunique()
            toplam_ulke_sayisi = df['Ülke'].nunique()
            st.metric("Ülke Sayısı", f"{intl_ulke_sayisi}/{toplam_ulke_sayisi}")
    
    with intl_sutunlar[3]:
        if 'Molekül' in df.columns:
            intl_molekul_sayisi = df[df['International_Product'] == 1]['Molekül'].nunique()
            toplam_molekul_sayisi = df['Molekül'].nunique()
            st.metric("Molekül Sayısı", f"{intl_molekul_sayisi}/{toplam_molekul_sayisi}")
    
    # Coğrafi Dağılım Dünya Haritası
    st.markdown('<h3 class="subsection-title">🗺️ Coğrafi Dağılım - Dünya Haritası</h3>', unsafe_allow_html=True)
    
    dunya_haritasi = gorsellestirme.dunya_haritasi_olustur(df)
    if dunya_haritasi:
        st.plotly_chart(dunya_haritasi, use_container_width=True, config={'displayModeBar': True})
    
    # International Product detay tablosu
    st.markdown('<h3 class="subsection-title">📋 International Product Detayları</h3>', unsafe_allow_html=True)
    
    intl_df = df[df['International_Product'] == 1].copy()
    
    if len(intl_df) > 0:
        # Hangi sütunları göstereceğimizi belirleyelim
        gosterilecek_sutunlar = []
        
        for sutun in ['Molekül', 'Şirket', 'Ülke', 'Satış_2024', 'Ort_Fiyat_2024', 'Büyüme_2023_2024']:
            if sutun in intl_df.columns:
                gosterilecek_sutunlar.append(sutun)
        
        if gosterilecek_sutunlar:
            # Veriyi formatla
            goster_df = intl_df[gosterilecek_sutunlar].copy()
            
            # Satış sütununu formatla
            for sutun in goster_df.columns:
                if 'Satış' in sutun:
                    goster_df[sutun] = goster_df[sutun].apply(lambda x: f"${x/1e6:.2f}M" if pd.notnull(x) else "N/A")
                elif 'Fiyat' in sutun:
                    goster_df[sutun] = goster_df[sutun].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                elif 'Büyüme' in sutun:
                    goster_df[sutun] = goster_df[sutun].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
            
            st.dataframe(
                goster_df.sort_values('Satış_2024' if 'Satış_2024' in goster_df.columns else gosterilecek_sutunlar[0], ascending=False),
                use_container_width=True,
                height=400
            )
        else:
            st.info("Gösterilecek sütun bulunamadı.")
    else:
        st.info("International Product bulunamadı.")

def stratejik_analiz_tab_goster(df, icgoruler):
    """Stratejik Analiz tab'ını göster"""
    st.markdown('<h2 class="section-title">Stratejik Analiz ve Öngörüler</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">🎯 Temel Analiz</h3>', unsafe_allow_html=True)
    
    # Basit analizler
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Satış_2024' in df.columns:
            top_urun = df.nlargest(1, 'Satış_2024')
            if len(top_urun) > 0:
                urun_adi = top_urun.iloc[0]['Molekül'] if 'Molekül' in top_urun.columns else "Ürün"
                satis_degeri = top_urun.iloc[0]['Satış_2024']
                st.metric("🏆 En Çok Satan Ürün", urun_adi, f"${satis_degeri/1e6:.1f}M")
    
    with col2:
        if 'Büyüme_2023_2024' in df.columns:
            top_buyume = df.nlargest(1, 'Büyüme_2023_2024')
            if len(top_buyume) > 0:
                urun_adi = top_buyume.iloc[0]['Molekül'] if 'Molekül' in top_buyume.columns else "Ürün"
                buyume_degeri = top_buyume.iloc[0]['Büyüme_2023_2024']
                st.metric("🚀 En Hızlı Büyüyen", urun_adi, f"%{buyume_degeri:.1f}")
    
    with col3:
        if 'Şirket' in df.columns and 'Satış_2024' in df.columns:
            top_sirket = df.groupby('Şirket')['Satış_2024'].sum().nlargest(1)
            if len(top_sirket) > 0:
                sirket_adi = top_sirket.index[0]
                satis_degeri = top_sirket.iloc[0]
                st.metric("🏢 Pazar Lideri", sirket_adi, f"${satis_degeri/1e6:.1f}M")
    
    st.markdown('<h3 class="subsection-title">📈 Öneriler</h3>', unsafe_allow_html=True)
    
    # Basit öneriler
    oneriler = []
    
    # Büyüme fırsatı önerisi
    if 'Büyüme_2023_2024' in df.columns:
        yuksek_buyume_df = df[df['Büyüme_2023_2024'] > 20]
        if len(yuksek_buyume_df) > 0:
            oneriler.append(f"**{len(yuksek_buyume_df)} ürün %20'den fazla büyüme gösteriyor.** Bu ürünlere odaklanın.")
    
    # International Product önerisi
    if 'International_Product' in df.columns:
        intl_orani = (df['International_Product'] == 1).mean() * 100
        if intl_orani < 30:
            oneriler.append(f"**International Product oranı %{intl_orani:.1f}.** Global pazara açılma fırsatları değerlendirilebilir.")
    
    # Fiyat segmenti önerisi
    if 'Ort_Fiyat_2024' in df.columns:
        ortalama_fiyat = df['Ort_Fiyat_2024'].mean()
        yuksek_fiyatli_df = df[df['Ort_Fiyat_2024'] > ortalama_fiyat * 2]
        if len(yuksek_fiyatli_df) > 0:
            oneriler.append(f"**{len(yuksek_fiyatli_df)} premium ürün bulunuyor.** Premium segmentte fiyatlandırma stratejileri gözden geçirilebilir.")
    
    # Önerileri göster
    for i, oneri in enumerate(oneriler[:3]):  # İlk 3 öneriyi göster
        st.markdown(f"""
        <div class="insight-card info">
            <div class="insight-icon">💡</div>
            <div class="insight-content">{oneri}</div>
        </div>
        """, unsafe_allow_html=True)

def raporlama_tab_goster(df, metrikler, icgoruler):
    """Raporlama tab'ını göster"""
    st.markdown('<h2 class="section-title">Raporlama ve İndirme</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">📊 Rapor Türleri</h3>', unsafe_allow_html=True)
    
    rapor_turu = st.radio(
        "Rapor Türü Seçin",
        ['Excel Detaylı Rapor', 'CSV Ham Veri', 'International Product Raporu'],
        horizontal=True,
        key="rapor_turu"
    )
    
    st.markdown('<h3 class="subsection-title">🛠️ Rapor Oluşturma</h3>', unsafe_allow_html=True)
    
    rapor_sutunlar = st.columns(3)
    
    with rapor_sutunlar[0]:
        if st.button("📈 Excel Raporu Oluştur", use_container_width=True, key="excel_raporu"):
            with st.spinner("Excel raporu oluşturuluyor..."):
                zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                excel_veri = df.to_csv(index=False)
                
                st.download_button(
                    label="⬇️ Excel İndir",
                    data=excel_veri,
                    file_name=f"pharma_rapor_{zaman_damgasi}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="indir_excel"
                )
    
    with rapor_sutunlar[1]:
        if st.button("🔄 Analizi Sıfırla", use_container_width=True, key="analiz_sifirla"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with rapor_sutunlar[2]:
        if 'International_Product' in df.columns:
            if st.button("💾 International Product CSV", use_container_width=True, key="intl_csv"):
                intl_df = df[df['International_Product'] == 1]
                zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_veri = intl_df.to_csv(index=False)
                
                st.download_button(
                    label="⬇️ CSV İndir",
                    data=csv_veri,
                    file_name=f"international_productlar_{zaman_damgasi}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="indir_intl_csv"
                )
    
    st.markdown('<h3 class="subsection-title">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)
    
    istatistik_sutunlar = st.columns(4)
    
    with istatistik_sutunlar[0]:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with istatistik_sutunlar[1]:
        st.metric("Toplam Sütun", len(df.columns))
    
    with istatistik_sutunlar[2]:
        bellek_kullanimi = df.memory_usage(deep=True).sum()/1024**2
        st.metric("Bellek Kullanımı", f"{bellek_kullanimi:.1f} MB")
    
    with istatistik_sutunlar[3]:
        intl_sayisi = (df['International_Product'] == 1).sum() if 'International_Product' in df.columns else 0
        st.metric("International Product", intl_sayisi)

# ================================================
# 6. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        st.error("Detaylı hata bilgisi:")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Sayfayı Yenile", use_container_width=True):
            # Session state'i sıfırla
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
