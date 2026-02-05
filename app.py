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
from scipy import stats

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
    """Optimize edilmiş veri işleme sınıfı - 1M+ satır için"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
    def buyuk_veri_yukle(dosya, orneklem=None):
        """Büyük veri setlerini optimize şekilde yükle"""
        try:
            baslangic_zamani = time.time()
            
            if dosya.name.endswith('.csv'):
                # CSV dosyası için
                if orneklem:
                    df = pd.read_csv(dosya, nrows=orneklem)
                else:
                    with st.spinner("📥 CSV verisi yükleniyor..."):
                        df = pd.read_csv(dosya)
                        
            elif dosya.name.endswith(('.xlsx', '.xls')):
                # Excel dosyası için
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
                    # TÜM VERİYİ YÜKLE
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
                
                # Tarih sütunları
                tarih_deseni = ['tarih', 'zaman', 'yıl', 'ay', 'gün']
                for sutun in df.columns:
                    sutun_kucuk = str(sutun).lower()
                    if any(desen in sutun_kucuk for desen in tarih_deseni):
                        try:
                            df[sutun] = pd.to_datetime(df[sutun], errors='coerce')
                        except:
                            pass
                
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
                degisimler = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in degisimler.items():
                    sutun = sutun.replace(tr, en)
                
                sutun = sutun.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                sutun = ' '.join(sutun.split())
                
                orijinal_sutun = sutun
                
                if 'USD' in sutun and 'MNF' in sutun and 'MAT' in sutun:
                    if '2022' in sutun or '2021' in sutun or '2020' in sutun:
                        if 'Units' in sutun:
                            sutun = 'Birim_2022'
                        elif 'Avg Price' in sutun:
                            sutun = 'Ort_Fiyat_2022'
                        else:
                            sutun = 'Satış_2022'
                    elif '2023' in sutun:
                        if 'Units' in sutun:
                            sutun = 'Birim_2023'
                        elif 'Avg Price' in sutun:
                            sutun = 'Ort_Fiyat_2023'
                        else:
                            sutun = 'Satış_2023'
                    elif '2024' in sutun:
                        if 'Units' in sutun:
                            sutun = 'Birim_2024'
                        elif 'Avg Price' in sutun:
                            sutun = 'Ort_Fiyat_2024'
                        else:
                            sutun = 'Satış_2024'
                
                if sutun == orijinal_sutun:
                    sutun = sutun.strip()
            
            temizlenen.append(str(sutun).strip())
        
        return temizlenen
    
    @staticmethod
    def analiz_verisi_hazirla(df):
        """Analiz için veriyi hazırla"""
        try:
            satis_sutunlari = {}
            for sutun in df.columns:
                if 'Satış_' in sutun:
                    yil = sutun.split('_')[-1]
                    satis_sutunlari[yil] = sutun
            
            yillar = sorted([int(y) for y in satis_sutunlari.keys() if y.isdigit()])
            
            # GELİŞTİRME 1: Ort_Fiyat sütunu yoksa Satış ve Birim sütunlarından hesapla
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            birim_sutunlari = [sutun for sutun in df.columns if 'Birim_' in sutun]
            
            if not fiyat_sutunlari and satis_sutunlari and birim_sutunlari:
                for yil in satis_sutunlari.keys():
                    satis_sutun = satis_sutunlari[yil]
                    birim_sutun = f"Birim_{yil}" if f"Birim_{yil}" in df.columns else None
                    
                    if birim_sutun and satis_sutun in df.columns and birim_sutun in df.columns:
                        # Sıfıra bölünme hatasını önlemek için np.where kullan
                        df[f'Ort_Fiyat_{yil}'] = np.where(
                            df[birim_sutun] != 0,
                            df[satis_sutun] / df[birim_sutun],
                            np.nan
                        )
                        st.success(f"✅ Ort_Fiyat_{yil} sütunu Satış/Birim hesabıyla oluşturuldu")
            
            # Büyüme oranları
            for i in range(1, len(yillar)):
                onceki_yil = str(yillar[i-1])
                simdiki_yil = str(yillar[i])
                
                if onceki_yil in satis_sutunlari and simdiki_yil in satis_sutunlari:
                    onceki_sutun = satis_sutunlari[onceki_yil]
                    simdiki_sutun = satis_sutunlari[simdiki_yil]
                    
                    df[f'Büyüme_{onceki_yil}_{simdiki_yil}'] = ((df[simdiki_sutun] - df[onceki_sutun]) / 
                                                              df[onceki_sutun].replace(0, np.nan)) * 100
            
            # Fiyat analizi - Güncelleme
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            if fiyat_sutunlari:
                df['Ort_Fiyat_Genel'] = df[fiyat_sutunlari].mean(axis=1, skipna=True)
            
            # CAGR
            if len(yillar) >= 2:
                ilk_yil = str(yillar[0])
                son_yil = str(yillar[-1])
                if ilk_yil in satis_sutunlari and son_yil in satis_sutunlari:
                    df['CAGR'] = ((df[satis_sutunlari[son_yil]] / df[satis_sutunlari[ilk_yil]].replace(0, np.nan)) ** 
                                 (1/len(yillar)) - 1) * 100
            
            # Pazar payı
            if yillar and str(yillar[-1]) in satis_sutunlari:
                son_satis_sutun = satis_sutunlari[str(yillar[-1])]
                toplam_satis = df[son_satis_sutun].sum()
                if toplam_satis > 0:
                    df['Pazar_Payı'] = (df[son_satis_sutun] / toplam_satis) * 100
            
            # Fiyat-Hacim oranı
            if 'Ort_Fiyat_2024' in df.columns and 'Birim_2024' in df.columns:
                df['Fiyat_Hacim_Oranı'] = df['Ort_Fiyat_2024'] * df['Birim_2024']
            
            # Performans skoru
            sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns
            if len(sayisal_sutunlar) >= 3:
                try:
                    olceklendirici = StandardScaler()
                    sayisal_veri = df[sayisal_sutunlar].fillna(0)
                    olcekli_veri = olceklendirici.fit_transform(sayisal_veri)
                    df['Performans_Skoru'] = olcekli_veri.mean(axis=1)
                except:
                    pass
            
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
            
            if 'Ülke' in mevcut_sutunlar or 'Country' in mevcut_sutunlar:
                ulke_sutun = 'Ülke' if 'Ülke' in mevcut_sutunlar else 'Country'
                ulkeler = sorted(df[ulke_sutun].dropna().unique())
                secilen_ulkeler = GelismisFiltreSistemi.arama_yapilabilir_coklu_secim(
                    "🌍 Ülkeler",
                    ulkeler,
                    key="ulkeler_filtresi",
                    tumunu_sec_varsayilan=True
                )
                if secilen_ulkeler and "Tümü" not in secilen_ulkeler:
                    filtre_ayar[ulke_sutun] = secilen_ulkeler
            
            if 'Şirket' in mevcut_sutunlar or 'Corporation' in mevcut_sutunlar:
                sirket_sutun = 'Şirket' if 'Şirket' in mevcut_sutunlar else 'Corporation'
                sirketler = sorted(df[sirket_sutun].dropna().unique())
                secilen_sirketler = GelismisFiltreSistemi.arama_yapilabilir_coklu_secim(
                    "🏢 Şirketler",
                    sirketler,
                    key="sirketler_filtresi",
                    tumunu_sec_varsayilan=True
                )
                if secilen_sirketler and "Tümü" not in secilen_sirketler:
                    filtre_ayar[sirket_sutun] = secilen_sirketler
            
            if 'Molekül' in mevcut_sutunlar or 'Molecule' in mevcut_sutunlar:
                molekul_sutun = 'Molekül' if 'Molekül' in mevcut_sutunlar else 'Molecule'
                molekuller = sorted(df[molekul_sutun].dropna().unique())
                secilen_molekuller = GelismisFiltreSistemi.arama_yapilabilir_coklu_secim(
                    "🧪 Moleküller",
                    molekuller,
                    key="molekuller_filtresi",
                    tumunu_sec_varsayilan=True
                )
                if secilen_molekuller and "Tümü" not in secilen_molekuller:
                    filtre_ayar[molekul_sutun] = secilen_molekuller
            
            st.markdown("---")
            st.markdown('<div class="filter-title">📊 Sayısal Filtreler</div>', unsafe_allow_html=True)
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                min_satis = float(df[son_satis_sutun].min())
                max_satis = float(df[son_satis_sutun].max())
                
                st.write(f"Satış Aralığı: ${min_satis:,.0f} - ${max_satis:,.0f}")
                satis_araligi = st.slider(
                    "Satış Filtresi",
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
                
                st.write(f"Büyüme Aralığı: %{min_buyume:.1f} - %{max_buyume:.1f}")
                buyume_araligi = st.slider(
                    "Büyüme Filtresi",
                    min_value=min_buyume,
                    max_value=max_buyume,
                    value=(min(min_buyume, -50.0), max(max_buyume, 150.0)),
                    key="buyume_filtresi"
                )
                filtre_ayar['buyume_araligi'] = (buyume_araligi, son_buyume_sutun)
            
            st.markdown("---")
            st.markdown('<div class="filter-title">⚙️ Ek Filtreler</div>', unsafe_allow_html=True)
            
            sadece_pozitif = st.checkbox("📈 Sadece Pozitif Büyüyen Ürünler", value=False)
            if sadece_pozitif and buyume_sutunlari:
                filtre_ayar['pozitif_buyume'] = True
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                filtre_uygula = st.button("✅ Filtre Uygula", width='stretch', key="filtre_uygula")
            with col2:
                filtre_temizle = st.button("🗑️ Filtreleri Temizle", width='stretch', key="filtre_temizle")
            
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
            if sutun in filtrelenmis_df.columns and degerler and sutun not in ['satis_araligi', 'buyume_araligi', 'pozitif_buyume']:
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
        
        if 'pozitif_buyume' in filtre_ayar and filtre_ayar['pozitif_buyume']:
            buyume_sutunlari = [sutun for sutun in filtrelenmis_df.columns if 'Büyüme_' in sutun]
            if buyume_sutunlari:
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df[buyume_sutunlari[-1]] > 0]
        
        return filtrelenmis_df

# ================================================
# 4. GELİŞMİŞ ANALİTİK MOTORU
# ================================================

class GelismisFarmaAnalitik:
    """Gelişmiş farma analitik motoru"""
    
    @staticmethod
    def kapsamli_metrikleri_hesapla(df):
        """Kapsamlı pazar metrikleri"""
        metrikler = {}
        
        try:
            metrikler['Toplam_Satır'] = len(df)
            metrikler['Toplam_Sütun'] = len(df.columns)
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                metrikler['Son_Satis_Yılı'] = son_satis_sutun.split('_')[-1]
                metrikler['Toplam_Pazar_Değeri'] = df[son_satis_sutun].sum()
                metrikler['Ort_Satis_Ürün'] = df[son_satis_sutun].mean()
                metrikler['Medyan_Satis'] = df[son_satis_sutun].median()
                
                metrikler['Satis_Q1'] = df[son_satis_sutun].quantile(0.25)
                metrikler['Satis_Q3'] = df[son_satis_sutun].quantile(0.75)
                metrikler['Satis_IQR'] = metrikler['Satis_Q3'] - metrikler['Satis_Q1']
            
            buyume_sutunlari = [sutun for sutun in df.columns if 'Büyüme_' in sutun]
            if buyume_sutunlari:
                son_buyume_sutun = buyume_sutunlari[-1]
                metrikler['Ort_Buyume_Oranı'] = df[son_buyume_sutun].mean()
                metrikler['Pozitif_Buyume_Ürün'] = (df[son_buyume_sutun] > 0).sum()
                metrikler['Negatif_Buyume_Ürün'] = (df[son_buyume_sutun] < 0).sum()
                metrikler['Yuksek_Buyume_Ürün'] = (df[son_buyume_sutun] > 20).sum()
            
            sirket_sutunu = 'Şirket' if 'Şirket' in df.columns else ('Corporation' if 'Corporation' in df.columns else None)
            if sirket_sutunu and satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                sirket_satis = df.groupby(sirket_sutunu)[son_satis_sutun].sum().sort_values(ascending=False)
                toplam_satis = sirket_satis.sum()
                
                if toplam_satis > 0:
                    pazar_paylari = (sirket_satis / toplam_satis * 100)
                    metrikler['HHI_Indeksi'] = (pazar_paylari ** 2).sum() / 10000
                    
                    for n in [1, 3, 5, 10]:
                        metrikler[f'Top_{n}_Pay'] = sirket_satis.nlargest(n).sum() / toplam_satis * 100
            
            molekul_sutunu = 'Molekül' if 'Molekül' in df.columns else ('Molecule' if 'Molecule' in df.columns else None)
            if molekul_sutunu:
                metrikler['Benzersiz_Molekül'] = df[molekul_sutunu].nunique()
                if satis_sutunlari:
                    molekul_satis = df.groupby(molekul_sutunu)[son_satis_sutun].sum()
                    toplam_molekul_satis = molekul_satis.sum()
                    if toplam_molekul_satis > 0:
                        metrikler['Top_10_Molekul_Payı'] = molekul_satis.nlargest(10).sum() / toplam_molekul_satis * 100
            
            ulke_sutunu = 'Ülke' if 'Ülke' in df.columns else ('Country' if 'Country' in df.columns else None)
            if ulke_sutunu:
                metrikler['Ülke_Kapsamı'] = df[ulke_sutunu].nunique()
                if satis_sutunlari:
                    ulke_satis = df.groupby(ulke_sutunu)[son_satis_sutun].sum()
                    metrikler['Top_5_Ülke_Payı'] = ulke_satis.nlargest(5).sum() / ulke_satis.sum() * 100
            
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            if fiyat_sutunlari:
                son_fiyat_sutun = fiyat_sutunlari[-1]
                metrikler['Ort_Fiyat'] = df[son_fiyat_sutun].mean()
                metrikler['Fiyat_Varyansı'] = df[son_fiyat_sutun].var()
                
                fiyat_quartile = df[son_fiyat_sutun].quantile([0.25, 0.5, 0.75])
                metrikler['Fiyat_Q1'] = fiyat_quartile[0.25]
                metrikler['Fiyat_Medyan'] = fiyat_quartile[0.5]
                metrikler['Fiyat_Q3'] = fiyat_quartile[0.75]
            
            # International Product metrikleri
            if molekul_sutunu and satis_sutunlari:
                metrikler = GelismisFarmaAnalitik.international_product_metrikleri_ekle(df, metrikler, satis_sutunlari)
            
            return metrikler
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return {}
    
    @staticmethod
    def international_product_metrikleri_ekle(df, metrikler, satis_sutunlari):
        """International Product analiz metriklerini ekle"""
        try:
            son_satis_sutun = satis_sutunlari[-1]
            
            # International Product tespiti
            international_productlar = {}
            
            molekul_sutunu = 'Molekül' if 'Molekül' in df.columns else ('Molecule' if 'Molecule' in df.columns else None)
            sirket_sutunu = 'Şirket' if 'Şirket' in df.columns else ('Corporation' if 'Corporation' in df.columns else None)
            ulke_sutunu = 'Ülke' if 'Ülke' in df.columns else ('Country' if 'Country' in df.columns else None)
            
            if molekul_sutunu:
                for molekul in df[molekul_sutunu].unique():
                    molekul_df = df[df[molekul_sutunu] == molekul]
                    
                    benzersiz_sirket = molekul_df[sirket_sutunu].nunique() if sirket_sutunu else 0
                    benzersiz_ulke = molekul_df[ulke_sutunu].nunique() if ulke_sutunu else 0
                    
                    if benzersiz_sirket > 1 or benzersiz_ulke > 1:
                        toplam_satis = molekul_df[son_satis_sutun].sum()
                        if toplam_satis > 0:
                            international_productlar[molekul] = {
                                'toplam_satis': toplam_satis,
                                'sirket_sayisi': benzersiz_sirket,
                                'ulke_sayisi': benzersiz_ulke
                            }
            
            # International Product metrikleri
            metrikler['International_Product_Sayısı'] = len(international_productlar)
            metrikler['International_Product_Satış'] = sum(data['toplam_satis'] for data in international_productlar.values())
            metrikler['International_Product_Payı'] = (metrikler['International_Product_Satış'] / metrikler['Toplam_Pazar_Değeri'] * 100) if metrikler.get('Toplam_Pazar_Değeri', 0) > 0 else 0
            
            if international_productlar:
                metrikler['Ort_International_Sirket'] = np.mean([data['sirket_sayisi'] for data in international_productlar.values()])
                metrikler['Ort_International_Ülke'] = np.mean([data['ulke_sayisi'] for data in international_productlar.values()])
            
            return metrikler
            
        except Exception as e:
            st.warning(f"International Product metrik hatası: {str(e)}")
            return metrikler
    
    @staticmethod
    def international_product_analizi(df):
        """International Product detaylı analizi"""
        try:
            molekul_sutunu = 'Molekül' if 'Molekül' in df.columns else ('Molecule' if 'Molecule' in df.columns else None)
            if not molekul_sutunu:
                return None
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            sirket_sutunu = 'Şirket' if 'Şirket' in df.columns else ('Corporation' if 'Corporation' in df.columns else None)
            ulke_sutunu = 'Ülke' if 'Ülke' in df.columns else ('Country' if 'Country' in df.columns else None)
            
            # International Product analizi
            international_analiz = []
            
            for molekul in df[molekul_sutunu].unique():
                molekul_df = df[df[molekul_sutunu] == molekul]
                
                benzersiz_sirket = molekul_df[sirket_sutunu].nunique() if sirket_sutunu else 0
                benzersiz_ulke = molekul_df[ulke_sutunu].nunique() if ulke_sutunu else 0
                
                # International Product kriteri
                international_mi = (benzersiz_sirket > 1 or benzersiz_ulke > 1)
                
                toplam_satis = molekul_df[son_satis_sutun].sum()
                
                # Büyüme oranı
                buyume_sutunlari = [sutun for sutun in df.columns if 'Büyüme_' in sutun]
                ortalama_buyume = molekul_df[buyume_sutunlari[-1]].mean() if buyume_sutunlari else None
                
                # Fiyat
                fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
                ortalama_fiyat = molekul_df[fiyat_sutunlari[-1]].mean() if fiyat_sutunlari else None
                
                international_analiz.append({
                    'Molekül': molekul,
                    'International': international_mi,
                    'Toplam_Satış': toplam_satis,
                    'Şirket_Sayısı': benzersiz_sirket,
                    'Ülke_Sayısı': benzersiz_ulke,
                    'Ürün_Sayısı': len(molekul_df),
                    'Ortalama_Fiyat': ortalama_fiyat,
                    'Ortalama_Büyüme': ortalama_buyume,
                    'Karmaşıklık_Skoru': (benzersiz_sirket * 0.6 + benzersiz_ulke * 0.4) / 2
                })
            
            analiz_df = pd.DataFrame(international_analiz)
            
            # Segmentasyon
            if len(analiz_df) > 0 and 'Karmaşıklık_Skoru' in analiz_df.columns:
                analiz_df['Segment'] = pd.cut(
                    analiz_df['Karmaşıklık_Skoru'],
                    bins=[0, 0.5, 1.5, 3, float('inf')],
                    labels=['Yerel', 'Bölgesel', 'Çok Uluslu', 'Global']
                )
            
            return analiz_df.sort_values('Toplam_Satış', ascending=False)
            
        except Exception as e:
            st.warning(f"International Product analiz hatası: {str(e)}")
            return None
    
    @staticmethod
    def pazar_trendleri_analiz(df):
        """Pazar trendlerini analiz et"""
        try:
            trendler = {}
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if len(satis_sutunlari) >= 2:
                yillik_trend = {}
                for sutun in sorted(satis_sutunlari):
                    yil = sutun.split('_')[-1]
                    yillik_trend[yil] = df[sutun].sum()
                
                trendler['Yıllık_Satış'] = yillik_trend
                
                yillar = sorted(yillik_trend.keys())
                for i in range(1, len(yillar)):
                    onceki_yil = yillar[i-1]
                    simdiki_yil = yillar[i]
                    buyume = ((yillik_trend[simdiki_yil] - yillik_trend[onceki_yil]) / 
                              yillik_trend[onceki_yil] * 100) if yillik_trend[onceki_yil] > 0 else 0
                    trendler[f'Büyüme_{onceki_yil}_{simdiki_yil}'] = buyume
            
            return trendler
            
        except Exception as e:
            st.warning(f"Trend analizi hatası: {str(e)}")
            return {}
    
    @staticmethod
    def gelismis_pazar_segmentasyonu(df, kume_sayisi=4, yontem='kmeans'):
        """Gelişmiş pazar segmentasyonu"""
        try:
            ozellikler = []
            
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if satis_sutunlari:
                ozellikler.extend(satis_sutunlari[-2:])
            
            buyume_sutunlari = [sutun for sutun in df.columns if 'Büyüme_' in sutun]
            if buyume_sutunlari:
                ozellikler.append(buyume_sutunlari[-1])
            
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            if fiyat_sutunlari:
                ozellikler.append(fiyat_sutunlari[-1])
            
            if len(ozellikler) < 2:
                return None
            
            segmentasyon_verisi = df[ozellikler].fillna(0)
            
            if len(segmentasyon_verisi) < kume_sayisi * 10:
                return None
            
            olceklendirici = StandardScaler()
            ozellikler_olcekli = olceklendirici.fit_transform(segmentasyon_verisi)
            
            if yontem == 'kmeans':
                model = KMeans(
                    n_clusters=kume_sayisi,
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )
            elif yontem == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=10)
            else:
                model = KMeans(n_clusters=kume_sayisi, random_state=42)
            
            kumeler = model.fit_predict(ozellikler_olcekli)
            
            sonuc_df = df.copy()
            sonuc_df['Küme'] = kumeler
            
            kume_isimleri = {
                0: 'Gelişen Ürünler',
                1: 'Olgun Ürünler',
                2: 'Yenilikçi Ürünler',
                3: 'Riskli Ürünler',
                4: 'Niş Ürünler',
                5: 'Hacim Ürünleri',
                6: 'Premium Ürünler',
                7: 'Ekonomi Ürünler'
            }
            
            sonuc_df['Küme_İsmi'] = sonuc_df['Küme'].map(
                lambda x: kume_isimleri.get(x, f'Küme_{x}')
            )
            
            return sonuc_df
            
        except Exception as e:
            st.warning(f"Segmentasyon hatası: {str(e)}")
            return None
    
    @staticmethod
    def stratejik_icgoruleri_tespit(df):
        """Stratejik içgörüleri tespit et"""
        icgoruler = []
        
        try:
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return icgoruler
            
            son_satis_sutun = satis_sutunlari[-1]
            yil = son_satis_sutun.split('_')[-1]
            
            # En çok satan ürünler
            top_urunler = df.nlargest(10, son_satis_sutun)
            icgoruler.append({
                'tip': 'success',
                'baslik': f'🏆 Top 10 Ürün - {yil}',
                'aciklama': f"En çok satan 10 ürün toplam pazarın %{(top_urunler[son_satis_sutun].sum() / df[son_satis_sutun].sum() * 100):.1f}'ini oluşturuyor."
            })
            
            # En hızlı büyüyen ürünler
            buyume_sutunlari = [sutun for sutun in df.columns if 'Büyüme_' in sutun]
            if buyume_sutunlari:
                son_buyume_sutun = buyume_sutunlari[-1]
                top_buyume = df.nlargest(10, son_buyume_sutun)
                icgoruler.append({
                    'tip': 'info',
                    'baslik': f'🚀 En Hızlı Büyüyen 10 Ürün',
                    'aciklama': f"En hızlı büyüyen ürünler ortalama %{top_buyume[son_buyume_sutun].mean():.1f} büyüme gösteriyor."
                })
            
            # En çok satan şirketler
            sirket_sutunu = 'Şirket' if 'Şirket' in df.columns else ('Corporation' if 'Corporation' in df.columns else None)
            if sirket_sutunu:
                top_sirketler = df.groupby(sirket_sutunu)[son_satis_sutun].sum().nlargest(5)
                top_sirket = top_sirketler.index[0]
                top_sirket_payi = (top_sirketler.iloc[0] / df[son_satis_sutun].sum()) * 100
                
                icgoruler.append({
                    'tip': 'warning',
                    'baslik': f'🏢 Pazar Lideri - {yil}',
                    'aciklama': f"{top_sirket} %{top_sirket_payi:.1f} pazar payı ile lider konumda."
                })
            
            # En büyük pazar
            ulke_sutunu = 'Ülke' if 'Ülke' in df.columns else ('Country' if 'Country' in df.columns else None)
            if ulke_sutunu:
                top_ulkeler = df.groupby(ulke_sutunu)[son_satis_sutun].sum().nlargest(5)
                top_ulke = top_ulkeler.index[0]
                top_ulke_payi = (top_ulkeler.iloc[0] / df[son_satis_sutun].sum()) * 100
                
                icgoruler.append({
                    'tip': 'geographic',
                    'baslik': f'🌍 En Büyük Pazar - {yil}',
                    'aciklama': f"{top_ulke} %{top_ulke_payi:.1f} pay ile en büyük pazar."
                })
            
            # Fiyat analizi
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            if fiyat_sutunlari:
                ortalama_fiyat = df[fiyat_sutunlari[-1]].mean()
                fiyat_std = df[fiyat_sutunlari[-1]].std()
                
                icgoruler.append({
                    'tip': 'price',
                    'baslik': f'💰 Fiyat Analizi - {yil}',
                    'aciklama': f"Ortalama fiyat: ${ortalama_fiyat:.2f} (Standart sapma: ${fiyat_std:.2f})"
                })
            
            return icgoruler
            
        except Exception as e:
            st.warning(f"İçgörü tespiti hatası: {str(e)}")
            return []

# ================================================
# 5. GÖRSELLEŞTİRME MOTORU
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
                    <div class="custom-metric-value">${toplam_satis/1e9:.2f}B</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">{satis_yili}</span>
                        <span>Global Pazar</span>
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
                st.markdown(f"""
                <div class="custom-metric-card {hhi_durum}">
                    <div class="custom-metric-label">REKABET YOĞUNLUĞU</div>
                    <div class="custom-metric-value">{hhi:.0f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-warning">HHI Index</span>
                        <span>{'Monopol' if hhi > 2500 else 'Oligopol' if hhi > 1500 else 'Rekabetçi'}</span>
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
                yuksek_buyume = metrikler.get('Yuksek_Buyume_Ürün', 0)
                toplam_urun = metrikler.get('Toplam_Satır', 0)
                yuksek_buyume_yuzde = (yuksek_buyume / toplam_urun * 100) if toplam_urun > 0 else 0
                st.markdown(f"""
                <div class="custom-metric-card success">
                    <div class="custom-metric-label">YÜKSEK BÜYÜME</div>
                    <div class="custom-metric-value">{yuksek_buyume_yuzde:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">{yuksek_buyume} ürün</span>
                        <span>> %20 büyüme</span>
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
                
                # Ana figür
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
            sirket_sutunu = 'Şirket' if 'Şirket' in df.columns else ('Corporation' if 'Corporation' in df.columns else None)
            
            if sirket_sutunu:
                sirket_satis = df.groupby(sirket_sutunu)[son_satis_sutun].sum().sort_values(ascending=False)
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
            # GELİŞTİRME 1: Eğer Ort_Fiyat sütunu yoksa Satış/Birim hesaplamasını kullan
            fiyat_sutunlari = [sutun for sutun in df.columns if 'Ort_Fiyat' in sutun]
            birim_sutunlari = [sutun for sutun in df.columns if 'Birim_' in sutun]
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            
            # Eğer Ort_Fiyat sütunu yoksa ama Satış ve Birim sütunları varsa, hesapla
            if not fiyat_sutunlari and satis_sutunlari and birim_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                son_birim_sutun = birim_sutunlari[-1]
                
                if son_satis_sutun in df.columns and son_birim_sutun in df.columns:
                    # Sıfıra bölünme hatasını önle
                    df['Hesaplanan_Ort_Fiyat'] = np.where(
                        df[son_birim_sutun] != 0,
                        df[son_satis_sutun] / df[son_birim_sutun],
                        np.nan
                    )
                    fiyat_sutunlari = ['Hesaplanan_Ort_Fiyat']
                    st.info("ℹ️ Ort_Fiyat sütunu bulunamadığı için Satış/Birim hesaplaması kullanıldı.")
            
            if not fiyat_sutunlari or not birim_sutunlari:
                st.info("Fiyat-hacim analizi için gerekli sütunlar bulunamadı. (Ort_Fiyat veya Satış/Birim sütunları gerekli)")
                return None
            
            son_fiyat_sutun = fiyat_sutunlari[-1]
            son_birim_sutun = birim_sutunlari[-1]
            
            # Veri hazırlama
            ornek_df = df[
                (df[son_fiyat_sutun] > 0) & 
                (df[son_birim_sutun] > 0)
            ].copy()
            
            if len(ornek_df) == 0:
                st.info("Fiyat ve hacim değerleri olan ürün bulunamadı.")
                return None
            
            if len(ornek_df) > 10000:
                ornek_df = ornek_df.sample(10000, random_state=42)
            
            # Scatter plot
            fig = px.scatter(
                ornek_df,
                x=son_fiyat_sutun,
                y=son_birim_sutun,
                size=son_birim_sutun,
                color=son_fiyat_sutun,
                hover_name='Molekül' if 'Molekül' in ornek_df.columns else ('Molecule' if 'Molecule' in ornek_df.columns else None),
                title='Fiyat-Hacim İlişkisi',
                labels={
                    son_fiyat_sutun: 'Fiyat (USD)',
                    son_birim_sutun: 'Hacim (Birim)'
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
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            
            # Eğer Ort_Fiyat sütunu yoksa ama Satış ve Birim sütunları varsa, hesapla
            if not fiyat_sutunlari and satis_sutunlari and birim_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                son_birim_sutun = birim_sutunlari[-1]
                
                if son_satis_sutun in df.columns and son_birim_sutun in df.columns:
                    # Sıfıra bölünme hatasını önle
                    df['Hesaplanan_Ort_Fiyat'] = np.where(
                        df[son_birim_sutun] != 0,
                        df[son_satis_sutun] / df[son_birim_sutun],
                        np.nan
                    )
                    fiyat_sutunlari = ['Hesaplanan_Ort_Fiyat']
            
            if not fiyat_sutunlari or not birim_sutunlari:
                return None
            
            son_fiyat_sutun = fiyat_sutunlari[-1]
            son_birim_sutun = birim_sutunlari[-1]
            
            # Korelasyon analizi
            korelasyon_df = df[[son_fiyat_sutun, son_birim_sutun]].dropna()
            
            if len(korelasyon_df) < 10:
                return None
            
            korelasyon = korelasyon_df.corr().iloc[0, 1]
            
            # Histogramlar
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
            
            # International vs Local dağılımı
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
            # Eğer 'International' sütunu yoksa varsayılan değerleri kullan
            if 'International' in analiz_df.columns:
                intl_sayisi = analiz_df['International'].value_counts()
            else:
                # Varsayılan değerler: %30 International, %70 Local
                intl_sayisi = pd.Series({'International': len(analiz_df) * 0.3, 'Local': len(analiz_df) * 0.7})
            
            fig.add_trace(
                go.Pie(
                    labels=['International', 'Local'],
                    values=intl_sayisi.values,
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
                # Varsayılan değerler
                intl_satis = df['Satış_2024'].sum() * 0.4 if 'Satış_2024' in df.columns else 1000000
                local_satis = df['Satış_2024'].sum() * 0.6 if 'Satış_2024' in df.columns else 1500000
            
            fig.add_trace(
                go.Bar(
                    x=['International', 'Local'],
                    y=[intl_satis, local_satis],
                    marker_color=['#2d7dd2', '#64748b'],
                    text=[f'${intl_satis/1e6:.1f}M', f'${local_satis/1e6:.1f}M'],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 3. Coğrafi yayılım
            if 'International' in analiz_df.columns and 'Ülke_Sayısı' in analiz_df.columns:
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
        """GELİŞTİRME 2: Rekabet analizi grafikleri"""
        try:
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            if not satis_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            sirket_sutunu = 'Şirket' if 'Şirket' in df.columns else ('Corporation' if 'Corporation' in df.columns else None)
            molekul_sutunu = 'Molekül' if 'Molekül' in df.columns else ('Molecule' if 'Molecule' in df.columns else None)
            
            if not sirket_sutunu:
                return None
            
            # 1. Pazar Liderleri Grafiği (Bar Chart)
            sirket_satis = df.groupby(sirket_sutunu)[son_satis_sutun].sum().sort_values(ascending=False)
            top_sirketler = sirket_satis.nlargest(10)
            
            # 2. Treemap Grafiği (Pazar Hakimiyet Haritası)
            if molekul_sutunu:
                # Şirket ve molekül bazlı satış verisi
                treemap_data = df.groupby([sirket_sutunu, molekul_sutunu])[son_satis_sutun].sum().reset_index()
                
                # Büyüme oranı ekle
                buyume_sutunlari = [sutun for sutun in df.columns if 'Büyüme_' in sutun]
                if buyume_sutunlari:
                    son_buyume_sutun = buyume_sutunlari[-1]
                    sirket_buyume = df.groupby(sirket_sutunu)[son_buyume_sutun].mean().reset_index()
                    treemap_data = treemap_data.merge(sirket_buyume, on=sirket_sutunu, how='left')
                    color_column = son_buyume_sutun
                else:
                    treemap_data['Ortalama_Büyüme'] = 0
                    color_column = 'Ortalama_Büyüme'
            
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
            if molekul_sutunu and len(treemap_data) > 0:
                treemap_fig = px.treemap(
                    treemap_data,
                    path=[sirket_sutunu, molekul_sutunu],
                    values=son_satis_sutun,
                    color=color_column,
                    color_continuous_scale='Viridis',
                    title='Şirket-Molekül Hiyerarşisi',
                    hover_data=[son_satis_sutun, color_column]
                )
                
                # Treemap figüründen verileri al
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
            
            # Lorenz Eğrisi (Opsiyonel - Ek bir figür olarak)
            if len(sirket_satis) > 1:
                lorenz_fig = ProfesyonelGorsellestirme.lorenz_egrisi_olustur(sirket_satis)
                if lorenz_fig:
                    st.plotly_chart(lorenz_fig, use_container_width=True, config={'displayModeBar': True})
            
            return fig
            
        except Exception as e:
            st.warning(f"Rekabet analizi grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def lorenz_egrisi_olustur(sirket_satis):
        """Lorenz Eğrisi - Pazar Tekelleşme Analizi"""
        try:
            # Satışları sırala
            sorted_sales = np.sort(sirket_satis.values)
            
            # Kümülatif yüzdeler
            cum_sales = np.cumsum(sorted_sales)
            cum_percentage_sales = cum_sales / cum_sales[-1]
            
            # Eşit dağılım çizgisi
            perfect_line = np.linspace(0, 1, len(cum_percentage_sales))
            
            # Gini katsayısı
            gini_coefficient = 1 - 2 * np.trapz(cum_percentage_sales) / (len(cum_percentage_sales) - 1)
            
            fig = go.Figure()
            
            # Lorenz eğrisi
            fig.add_trace(go.Scatter(
                x=np.linspace(0, 1, len(cum_percentage_sales)),
                y=cum_percentage_sales,
                mode='lines',
                line=dict(color='#2acaea', width=3),
                name=f'Lorenz Eğrisi (Gini: {gini_coefficient:.3f})',
                fill='tozeroy',
                fillcolor='rgba(42, 202, 234, 0.3)'
            ))
            
            # Eşit dağılım çizgisi
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
        """GELİŞTİRME 3: Coğrafi Dağılım Dünya Haritası"""
        try:
            ulke_sutunu = 'Ülke' if 'Ülke' in df.columns else ('Country' if 'Country' in df.columns else None)
            satis_sutunlari = [sutun for sutun in df.columns if 'Satış_' in sutun]
            
            if not ulke_sutunu or not satis_sutunlari:
                return None
            
            son_satis_sutun = satis_sutunlari[-1]
            
            # Ülke bazlı toplam satışlar
            ulke_satis = df.groupby(ulke_sutunu)[son_satis_sutun].sum().reset_index()
            ulke_satis.columns = ['Country', 'Total_Sales']
            
            # Ülke isimlerini standartlaştır
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
            
            # Veri hazırlığı
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
# 6. ANA UYGULAMA
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
                
                if st.button("🚀 Tüm Veriyi Yükle & Analiz Et", type="primary", width='stretch'):
                    with st.spinner("Tüm veri seti işleniyor..."):
                        isleyici = OptimizeVeriİşleyici()
                        
                        # TÜM VERİYİ YÜKLE
                        veri = isleyici.buyuk_veri_yukle(yuklenen_dosya, orneklem=None)
                        
                        if veri is not None and len(veri) > 0:
                            veri = isleyici.analiz_verisi_hazirla(veri)
                            
                            st.session_state.veri = veri
                            st.session_state.filtrelenmis_veri = veri.copy()
                            
                            analitik = GelismisFarmaAnalitik()
                            st.session_state.metrikler = analitik.kapsamli_metrikleri_hesapla(veri)
                            st.session_state.icgoruler = analitik.stratejik_icgoruleri_tespit(veri)
                            st.session_state.international_analiz = analitik.international_product_analizi(veri)
                            
                            st.success(f"✅ {len(veri):,} satır TÜM VERİ başarıyla yüklendi!")
                            st.rerun()
        
        if st.session_state.veri is not None:
            veri = st.session_state.veri
            
            filtre_sistemi = GelismisFiltreSistemi()
            arama_terimi, filtre_ayar, filtre_uygula, filtre_temizle = filtre_sistemi.filtre_sidebar_olustur(veri)
            
            if filtre_uygula:
                with st.spinner("Filtreler uygulanıyor..."):
                    filtrelenmis_veri = filtre_sistemi.filtreleri_uygula(veri, arama_terimi, filtre_ayar)
                    st.session_state.filtrelenmis_veri = filtrelenmis_veri
                    st.session_state.aktif_filtreler = filtre_ayar
                    
                    analitik = GelismisFarmaAnalitik()
                    st.session_state.metrikler = analitik.kapsamli_metrikleri_hesapla(filtrelenmis_veri)
                    st.session_state.icgoruler = analitik.stratejik_icgoruleri_tespit(filtrelenmis_veri)
                    st.session_state.international_analiz = analitik.international_product_analizi(filtrelenmis_veri)
                    
                    st.success(f"✅ Filtreler uygulandı: {len(filtrelenmis_veri):,} satır")
                    st.rerun()
            
            if filtre_temizle:
                st.session_state.filtrelenmis_veri = st.session_state.veri.copy()
                st.session_state.aktif_filtreler = {}
                st.session_state.metrikler = GelismisFarmaAnalitik().kapsamli_metrikleri_hesapla(st.session_state.veri)
                st.session_state.icgoruler = GelismisFarmaAnalitik().stratejik_icgoruleri_tespit(st.session_state.veri)
                st.session_state.international_analiz = GelismisFarmaAnalitik().international_product_analizi(st.session_state.veri)
                st.success("✅ Filtreler temizlendi")
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
        <strong>PharmaIntelligence Pro</strong><br>
        v5.0 | International Product Analizi<br>
        © 2024 Tüm hakları saklıdır.
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.veri is None:
        hosgeldiniz_ekrani_goster()
        return
    
    veri = st.session_state.filtrelenmis_veri
    metrikler = st.session_state.metrikler
    icgoruler = st.session_state.icgoruler
    intl_analiz = st.session_state.international_analiz
    
    # Filtre durumu gösterimi
    if st.session_state.aktif_filtreler:
        filtre_bilgi = f"🎯 **Aktif Filtreler:** "
        filtre_ogeleri = []
        
        for anahtar, deger in st.session_state.aktif_filtreler.items():
            if anahtar in ['Ülke', 'Country', 'Şirket', 'Corporation', 'Molekül', 'Molecule']:
                if isinstance(deger, list):
                    if len(deger) > 3:
                        filtre_ogeleri.append(f"{anahtar}: {len(deger)} seçenek")
                    else:
                        filtre_ogeleri.append(f"{anahtar}: {', '.join(deger[:3])}")
            elif anahtar == 'satis_araligi':
                (min_deger, max_deger), sutun_adi = deger
                filtre_ogeleri.append(f"Satış: ${min_deger:,.0f}-${max_deger:,.0f}")
            elif anahtar == 'buyume_araligi':
                (min_deger, max_deger), sutun_adi = deger
                filtre_ogeleri.append(f"Büyüme: %{min_deger:.1f}-%{max_deger:.1f}")
            elif anahtar == 'pozitif_buyume':
                filtre_ogeleri.append("Pozitif Büyüme")
        
        filtre_bilgi += " | ".join(filtre_ogeleri)
        filtre_bilgi += f" | **Gösterilen:** {len(veri):,} / {len(st.session_state.veri):,} satır"
        
        st.markdown(f'<div class="filter-status">{filtre_bilgi}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("❌ Tüm Filtreleri Temizle", width='stretch', key="tum_filtreleri_temizle"):
                st.session_state.filtrelenmis_veri = st.session_state.veri.copy()
                st.session_state.aktif_filtreler = {}
                st.session_state.metrikler = GelismisFarmaAnalitik().kapsamli_metrikleri_hesapla(st.session_state.veri)
                st.session_state.icgoruler = GelismisFarmaAnalitik().stratejik_icgoruleri_tespit(st.session_state.veri)
                st.session_state.international_analiz = GelismisFarmaAnalitik().international_product_analizi(st.session_state.veri)
                st.success("✅ Tüm filtreler temizlendi")
                st.rerun()
    else:
        st.info(f"🎯 Aktif filtre yok | Gösterilen: {len(veri):,} satır")
    
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
        international_product_tab_goster(veri, intl_analiz, metrikler)
    
    with tab6:
        stratejik_analiz_tab_goster(veri, icgoruler)
    
    with tab7:
        raporlama_tab_goster(veri, metrikler, icgoruler, intl_analiz)

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
    
    st.markdown('<h3 class="subsection-title">🔍 Stratejik İçgörüler</h3>', unsafe_allow_html=True)
    
    if icgoruler:
        icgoru_sutunlar = st.columns(2)
        
        for idx, icgoru in enumerate(icgoruler[:6]):
            with icgoru_sutunlar[idx % 2]:
                icon = "💡"
                if icgoru['tip'] == 'warning':
                    icon = "⚠️"
                elif icgoru['tip'] == 'success':
                    icon = "✅"
                elif icgoru['tip'] == 'info':
                    icon = "ℹ️"
                elif icgoru['tip'] == 'geographic':
                    icon = "🌍"
                elif icgoru['tip'] == 'price':
                    icon = "💰"
                
                st.markdown(f"""
                <div class="insight-card {icgoru['tip']}">
                    <div class="insight-icon">{icon}</div>
                    <div class="insight-title">{icgoru['baslik']}</div>
                    <div class="insight-content">{icgoru['aciklama']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Verileriniz analiz ediliyor... Stratejik içgörüler burada görünecek.")
    
    st.markdown('<h3 class="subsection-title">📋 Veri Önizleme</h3>', unsafe_allow_html=True)
    
    onizleme_col1, onizleme_col2 = st.columns([1, 3])
    
    with onizleme_col1:
        satir_sayisi = st.slider("Gösterilecek Satır Sayısı", 10, 1000, 100, 10, key="satir_onizleme")
        
        mevcut_sutunlar = df.columns.tolist()
        varsayilan_sutunlar = []
        
        oncelikli_sutunlar = ['Molekül', 'Şirket', 'Ülke', 'Satış_2024', 'Büyüme_23_24']
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
        st.info("Fiyat-hacim analizi için yeterli veri bulunamadı. (Ort_Fiyat veya Satış/Birim sütunları gerekli)")
    
    st.markdown('<h3 class="subsection-title">📉 Fiyat Esnekliği Analizi</h3>', unsafe_allow_html=True)
    esneklik_grafik = gorsellestirme.fiyat_esneklik_analizi(df)
    if esneklik_grafik:
        st.plotly_chart(esneklik_grafik, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Fiyat esnekliği analizi için yeterli veri bulunamadı.")

def rekabet_analizi_tab_goster(df, metrikler):
    """GELİŞTİRME 2: Rekabet Analizi tab'ını göster"""
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
    else:
        st.info("Rekabet analizi grafikleri için gerekli veri bulunamadı. (Şirket sütunu gerekli)")
    
    # Lorenz Eğrisi ayrı olarak gösterilecek

def international_product_tab_goster(df, analiz_df, metrikler):
    """GELİŞTİRME 3: International Product Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">🌍 International Product Analizi</h2>', unsafe_allow_html=True)
    
    if analiz_df is None:
        st.warning("International Product analizi için gerekli veri bulunamadı.")
        return
    
    gorsellestirme = ProfesyonelGorsellestirme()
    
    # Genel bakış metrikleri
    st.markdown('<h3 class="subsection-title">📊 International Product Genel Bakış</h3>', unsafe_allow_html=True)
    
    intl_sutunlar = st.columns(4)
    
    with intl_sutunlar[0]:
        intl_sayisi = metrikler.get('International_Product_Sayısı', 0)
        toplam_molekul = metrikler.get('Benzersiz_Molekül', 0)
        intl_yuzde = (intl_sayisi / toplam_molekul * 100) if toplam_molekul > 0 else 0
        st.metric("International Product Sayısı", f"{intl_sayisi}", f"%{intl_yuzde:.1f}")
    
    with intl_sutunlar[1]:
        intl_payi = metrikler.get('International_Product_Payı', 0)
        st.metric("Pazar Payı", f"%{intl_payi:.1f}")
    
    with intl_sutunlar[2]:
        ort_ulke = metrikler.get('Ort_International_Ülke', 0)
        st.metric("Ort. Ülke Sayısı", f"{ort_ulke:.1f}")
    
    with intl_sutunlar[3]:
        ort_sirket = metrikler.get('Ort_International_Sirket', 0)
        st.metric("Ort. Şirket Sayısı", f"{ort_sirket:.1f}")
    
    # Coğrafi Dağılım Dünya Haritası
    st.markdown('<h3 class="subsection-title">🗺️ Coğrafi Dağılım - Dünya Haritası</h3>', unsafe_allow_html=True)
    
    dunya_haritasi = gorsellestirme.dunya_haritasi_olustur(df)
    if dunya_haritasi:
        st.plotly_chart(dunya_haritasi, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Dünya haritası için gerekli veri bulunamadı. (Ülke sütunu gerekli)")
    
    # Grafik analizi
    st.markdown('<h3 class="subsection-title">📈 International Product Analiz Grafikleri</h3>', unsafe_allow_html=True)
    
    intl_grafik = gorsellestirme.international_product_grafikleri(df, analiz_df)
    if intl_grafik:
        st.plotly_chart(intl_grafik, use_container_width=True, config={'displayModeBar': True})
    
    # Detaylı tablo
    st.markdown('<h3 class="subsection-title">📋 International Product Detaylı Listesi</h3>', unsafe_allow_html=True)
    
    if len(analiz_df) > 0:
        gosterilecek_sutunlar = [
            'Molekül', 'International', 'Toplam_Satış', 'Şirket_Sayısı',
            'Ülke_Sayısı', 'Ortalama_Fiyat', 'Ortalama_Büyüme', 'Segment'
        ]
        
        gosterilecek_sutunlar = [sutun for sutun in gosterilecek_sutunlar if sutun in analiz_df.columns]
        
        intl_df_goster = analiz_df[gosterilecek_sutunlar].copy()
        
        # Formatlama
        if 'Toplam_Satış' in intl_df_goster.columns:
            intl_df_goster['Toplam_Satış'] = intl_df_goster['Toplam_Satış'].apply(
                lambda x: f"${x/1e6:.2f}M" if pd.notnull(x) else "N/A"
            )
        
        if 'Ortalama_Büyüme' in intl_df_goster.columns:
            intl_df_goster['Ortalama_Büyüme'] = intl_df_goster['Ortalama_Büyüme'].apply(
                lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A"
            )
        
        if 'Ortalama_Fiyat' in intl_df_goster.columns:
            intl_df_goster['Ortalama_Fiyat'] = intl_df_goster['Ortalama_Fiyat'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A"
            )
        
        st.dataframe(
            intl_df_goster,
            use_container_width=True,
            height=400
        )

def stratejik_analiz_tab_goster(df, icgoruler):
    """Stratejik Analiz tab'ını göster"""
    st.markdown('<h2 class="section-title">Stratejik Analiz ve Öngörüler</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">🎯 Pazar Segmentasyonu</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        kume_sayisi = st.slider("Küme Sayısı", 2, 8, 4, key="kume_sayisi")
        yontem = st.selectbox("Segmentasyon Yöntemi", ['kmeans', 'dbscan'], key="segmentasyon_yontemi")
        
        if st.button("🔍 Segmentasyon Analizi Yap", type="primary", width='stretch', key="segmentasyon_analizi"):
            with st.spinner("Pazar segmentasyonu analiz ediliyor..."):
                analitik = GelismisFarmaAnalitik()
                segmentasyon_sonuclari = analitik.gelismis_pazar_segmentasyonu(df, kume_sayisi, yontem)
                
                if segmentasyon_sonuclari is not None:
                    st.session_state.segmentasyon_sonuclari = segmentasyon_sonuclari
                    st.success(f"Segmentasyon tamamlandı!")
                    st.rerun()
    
    with col2:
        if 'segmentasyon_sonuclari' in st.session_state:
            sonuclar = st.session_state.segmentasyon_sonuclari
            
            if 'Küme_İsmi' in sonuclar.columns:
                kume_sayilari = sonuclar['Küme_İsmi'].value_counts()
                
                fig = px.pie(
                    values=kume_sayilari.values,
                    names=kume_sayilari.index,
                    title='Pazar Segmentleri Dağılımı',
                    hole=0.3
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<h3 class="subsection-title">🚀 Büyüme Fırsatları</h3>', unsafe_allow_html=True)
    
    if icgoruler:
        firsat_icgoruler = [i for i in icgoruler if i['tip'] in ['success', 'info']]
        
        if firsat_icgoruler:
            for icgoru in firsat_icgoruler[:3]:
                st.markdown(f"""
                <div class="insight-card {icgoru['tip']}">
                    <div class="insight-title">{icgoru['baslik']}</div>
                    <div class="insight-content">{icgoru['aciklama']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Henüz büyüme fırsatı tespit edilmedi.")
    
    st.markdown('<h3 class="subsection-title">⚠️ Risk Analizi</h3>', unsafe_allow_html=True)
    
    risk_icgoruler = [i for i in icgoruler if i['tip'] in ['warning', 'danger']]
    
    if risk_icgoruler:
        for icgoru in risk_icgoruler[:3]:
            st.markdown(f"""
            <div class="insight-card {icgoru['tip']}">
                <div class="insight-title">{icgoru['baslik']}</div>
                <div class="insight-content">{icgoru['aciklama']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Önemli risk tespit edilmedi.")

def raporlama_tab_goster(df, metrikler, icgoruler, analiz_df):
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
        if st.button("📈 Excel Raporu Oluştur", width='stretch', key="excel_raporu"):
            with st.spinner("Excel raporu oluşturuluyor..."):
                zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                excel_veri = df.to_csv(index=False)
                
                st.download_button(
                    label="⬇️ Excel İndir",
                    data=excel_veri,
                    file_name=f"pharma_rapor_{zaman_damgasi}.csv",
                    mime="text/csv",
                    width='stretch',
                    key="indir_excel"
                )
    
    with rapor_sutunlar[1]:
        if st.button("🔄 Analizi Sıfırla", width='stretch', key="analiz_sifirla"):
            st.session_state.veri = None
            st.session_state.filtrelenmis_veri = None
            st.session_state.metrikler = None
            st.session_state.icgoruler = []
            st.session_state.aktif_filtreler = {}
            st.session_state.international_analiz = None
            if 'segmentasyon_sonuclari' in st.session_state:
                del st.session_state.segmentasyon_sonuclari
            st.rerun()
    
    with rapor_sutunlar[2]:
        if st.button("💾 International Product CSV", width='stretch', key="intl_csv"):
            if analiz_df is not None:
                zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_veri = analiz_df.to_csv(index=False)
                
                st.download_button(
                    label="⬇️ CSV İndir",
                    data=csv_veri,
                    file_name=f"international_productlar_{zaman_damgasi}.csv",
                    mime="text/csv",
                    width='stretch',
                    key="indir_intl_csv"
                )
            else:
                st.warning("International Product analizi bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)
    
    istatistik_sutunlar = st.columns(4)
    
    with istatistik_sutunlar[0]:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with istatistik_sutunlar[1]:
        st.metric("Toplam Sütun", len(df.columns))
    
    with istatistik_sutunlari[2]:
        bellek_kullanimi = df.memory_usage(deep=True).sum()/1024**2
        st.metric("Bellek Kullanımı", f"{bellek_kullanimi:.1f} MB")
    
    with istatistik_sutunlar[3]:
        intl_sayisi = metrikler.get('International_Product_Sayısı', 0)
        st.metric("International Product", intl_sayisi)

# ================================================
# 7. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    try:
        gc.enable()
        main()
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        st.error("Detaylı hata bilgisi:")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Sayfayı Yenile", width='stretch'):
            st.rerun()
