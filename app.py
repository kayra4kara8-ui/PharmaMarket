# app.py - Profesyonel İlaç Pazarı Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Advanced analytics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import statsmodels.api as sm
from scipy import stats

# Utilities
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
    page_title="PharmaIntelligence Pro | İlaç Pazarı Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://pharmaintelligence.com/support',
        'Report a bug': "https://pharmaintelligence.com/bug",
        'About': "### PharmaIntelligence Pro v3.2\nInternational Product Analizi Eklendi"
    }
)

# PROFESYONEL MAVİ TEMA CSS STYLES
PROFESSIONAL_CSS = """
<style>
    /* === ROOT VARIABLES === */
    :root {
        --primary-dark: #0c1a32;
        --secondary-dark: #1a2d50;
        --accent-blue: #2563eb;
        --accent-blue-light: #3b82f6;
        --accent-blue-dark: #1d4ed8;
        --accent-cyan: #06b6d4;
        --accent-cyan-light: #22d3ee;
        --accent-green: #10b981;
        --accent-yellow: #f59e0b;
        --accent-red: #ef4444;
        
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        
        --bg-primary: #0c1a32;
        --bg-secondary: #1a2d50;
        --bg-card: #1e3a8a;
        --bg-card-light: #2563eb;
        --bg-hover: #1e40af;
        --bg-surface: #1e293b;
        
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #2563eb;
        
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
        background: linear-gradient(135deg, var(--accent-blue-light), var(--accent-cyan-light));
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
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.15), transparent);
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
    
    .custom-metric-card.premium {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-blue-dark));
    }
    
    .custom-metric-card.warning {
        background: linear-gradient(135deg, var(--accent-yellow), #f97316);
    }
    
    .custom-metric-card.danger {
        background: linear-gradient(135deg, var(--accent-red), #dc2626);
    }
    
    .custom-metric-card.success {
        background: linear-gradient(135deg, var(--accent-green), #059669);
    }
    
    .custom-metric-card.info {
        background: linear-gradient(135deg, var(--accent-blue-light), var(--accent-cyan));
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
    
    .trend-up { color: var(--accent-green); }
    .trend-down { color: var(--accent-red); }
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
    .insight-card.success { border-left-color: var(--accent-green); }
    .insight-card.warning { border-left-color: var(--accent-yellow); }
    .insight-card.danger { border-left-color: var(--accent-red); }
    
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
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.2), rgba(6, 182, 212, 0.2));
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--accent-blue);
        box-shadow: var(--shadow-md);
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    .filter-status-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.2));
        border-left: 5px solid var(--accent-yellow);
    }
    
    .filter-status-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(249, 115, 22, 0.2));
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
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
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
    
    .status-online { background: var(--accent-green); }
    .status-warning { background: var(--accent-yellow); }
    .status-error { background: var(--accent-red); }
    
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
        background: rgba(16, 185, 129, 0.2);
        color: var(--accent-green);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: var(--accent-yellow);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-info {
        background: rgba(37, 99, 235, 0.2);
        color: var(--accent-blue);
        border: 1px solid rgba(37, 99, 235, 0.3);
    }
    
    .badge-cyan {
        background: rgba(6, 182, 212, 0.2);
        color: var(--accent-cyan);
        border: 1px solid rgba(6, 182, 212, 0.3);
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
    .feature-card-green { border-left-color: var(--accent-green); }
    .feature-card-yellow { border-left-color: var(--accent-yellow); }
    
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
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(6, 182, 212, 0.1));
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        border: 1px solid rgba(37, 99, 235, 0.3);
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

class OptimizedDataProcessor:
    """Optimize edilmiş veri işleme sınıfı"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
    def load_large_dataset(file, sample_size=None):
        """Büyük veri setlerini optimize şekilde yükle"""
        try:
            start_time = time.time()
            
            if file.name.endswith('.csv'):
                if sample_size:
                    df = pd.read_csv(file, nrows=sample_size)
                else:
                    with st.spinner("📥 CSV verisi yükleniyor..."):
                        df = pd.read_csv(file)
                        
            elif file.name.endswith(('.xlsx', '.xls')):
                if sample_size:
                    chunks = []
                    chunk_size = 50000
                    total_chunks = (sample_size // chunk_size) + 1
                    
                    with st.spinner(f"📥 Büyük veri seti yükleniyor..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(total_chunks):
                            chunk = pd.read_excel(
                                file, 
                                skiprows=i * chunk_size,
                                nrows=chunk_size,
                                engine='openpyxl'
                            )
                            
                            if chunk.empty:
                                break
                            
                            chunks.append(chunk)
                            
                            loaded_rows = sum(len(c) for c in chunks)
                            progress = min(loaded_rows / sample_size, 1.0)
                            
                            progress_bar.progress(progress)
                            status_text.text(f"📊 {loaded_rows:,} satır yüklendi...")
                            
                            if loaded_rows >= sample_size:
                                break
                        
                        df = pd.concat(chunks, ignore_index=True)
                        progress_bar.progress(1.0)
                        status_text.text(f"✅ {len(df):,} satır başarıyla yüklendi")
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                else:
                    with st.spinner(f"📥 Tüm veri seti yükleniyor..."):
                        df = pd.read_excel(file, engine='openpyxl')
            
            df = OptimizedDataProcessor.optimize_dataframe(df)
            
            load_time = time.time() - start_time
            st.success(f"✅ Veri yükleme tamamlandı: {len(df):,} satır, {len(df.columns)} sütun ({load_time:.2f}s)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.error(f"Detay: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def optimize_dataframe(df):
        """DataFrame'i optimize et"""
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            df.columns = OptimizedDataProcessor.clean_column_names(df.columns)
            
            with st.spinner("Veri seti optimize ediliyor..."):
                for col in df.select_dtypes(include=['object']).columns:
                    num_unique = df[col].nunique()
                    total_rows = len(df)
                    
                    if num_unique < total_rows * 0.7:
                        df[col] = df[col].astype('category')
                
                for col in df.select_dtypes(include=[np.number]).columns:
                    try:
                        col_min = df[col].min()
                        col_max = df[col].max()
                        
                        if pd.api.types.is_integer_dtype(df[col]):
                            if col_min >= 0:
                                if col_max <= 255:
                                    df[col] = df[col].astype(np.uint8)
                                elif col_max <= 65535:
                                    df[col] = df[col].astype(np.uint16)
                                elif col_max <= 4294967295:
                                    df[col] = df[col].astype(np.uint32)
                                else:
                                    df[col] = df[col].astype(np.uint64)
                            else:
                                if col_min >= -128 and col_max <= 127:
                                    df[col] = df[col].astype(np.int8)
                                elif col_min >= -32768 and col_max <= 32767:
                                    df[col] = df[col].astype(np.int16)
                                elif col_min >= -2147483648 and col_max <= 2147483647:
                                    df[col] = df[col].astype(np.int32)
                                else:
                                    df[col] = df[col].astype(np.int64)
                        else:
                            df[col] = df[col].astype(np.float32)
                    except:
                        continue
                
                date_patterns = ['date', 'time', 'year', 'month', 'day', 'tarih']
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(pattern in col_lower for pattern in date_patterns):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
                
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        df[col] = df[col].astype(str).str.strip()
                    except:
                        pass
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_saved = original_memory - optimized_memory
            
            if memory_saved > 0:
                st.success(f"💾 Bellek optimizasyonu başarılı: {original_memory:.1f}MB → {optimized_memory:.1f}MB (%{memory_saved/original_memory*100:.1f} tasarruf)")
            
            return df
            
        except Exception as e:
            st.warning(f"Optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def clean_column_names(columns):
        """Sütun isimlerini temizle"""
        cleaned = []
        for col in columns:
            if isinstance(col, str):
                replacements = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in replacements.items():
                    col = col.replace(tr, en)
                
                col = col.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                col = ' '.join(col.split())
                
                original_col = col
                
                if 'USD' in col and 'MNF' in col and 'MAT' in col:
                    if '2022' in col or '2021' in col or '2020' in col:
                        if 'Units' in col:
                            col = 'Units_2022'
                        elif 'Avg Price' in col:
                            col = 'Ort_Fiyat_2022'
                        else:
                            col = 'Satış_2022'
                    elif '2023' in col:
                        if 'Units' in col:
                            col = 'Units_2023'
                        elif 'Avg Price' in col:
                            col = 'Ort_Fiyat_2023'
                        else:
                            col = 'Satış_2023'
                    elif '2024' in col:
                        if 'Units' in col:
                            col = 'Units_2024'
                        elif 'Avg Price' in col:
                            col = 'Ort_Fiyat_2024'
                        else:
                            col = 'Satış_2024'
                
                if col == original_col:
                    col = col.strip()
            
            cleaned.append(str(col).strip())
        
        return cleaned
    
    @staticmethod
    def prepare_analytics_data(df):
        """Analiz için veriyi hazırla"""
        try:
            # Satış sütunlarını bul
            satış_kelimeleri = ['satış', 'sales', 'cıro', 'hasılat']
            satış_sütunları = []
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kelime in col_lower for kelime in satış_kelimeleri):
                    satış_sütunları.append(col)
            
            if satış_sütunları:
                df['Satış_2024'] = df[satış_sütunları[-1]] if satış_sütunları else None
                
            # Fiyat sütunlarını bul
            fiyat_kelimeleri = ['fiyat', 'price', 'birim fiyat', 'unit price']
            fiyat_sütunları = []
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kelime in col_lower for kelime in fiyat_kelimeleri):
                    fiyat_sütunları.append(col)
            
            if fiyat_sütunları:
                df['Ort_Fiyat_2024'] = df[fiyat_sütunları[-1]] if fiyat_sütunları else None
            
            # Hacim/Adet sütunlarını bul
            hacim_kelimeleri = ['units', 'adet', 'hacim', 'volume', 'quantity']
            hacim_sütunları = []
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kelime in col_lower for kelime in hacim_kelimeleri):
                    hacim_sütunları.append(col)
            
            if hacim_sütunları:
                df['Units_2024'] = df[hacim_sütunları[-1]] if hacim_sütunları else None
            
            # Molekül sütununu bul
            molekül_kelimeleri = ['molecule', 'molekül', 'active', 'aktif', 'ingredient']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kelime in col_lower for kelime in molekül_kelimeleri):
                    df['Molekül'] = df[col]
                    break
            
            # Şirket sütununu bul
            şirket_kelimeleri = ['corporation', 'company', 'firma', 'şirket', 'manufacturer']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kelime in col_lower for kelime in şirket_kelimeleri):
                    df['Şirket'] = df[col]
                    break
            
            # Ülke sütununu bul
            ülke_kelimeleri = ['country', 'ülke', 'market', 'pazar']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kelime in col_lower for kelime in ülke_kelimeleri):
                    df['Ülke'] = df[col]
                    break
            
            # Büyüme oranlarını hesapla
            if 'Satış_2024' in df.columns and 'Satış_2023' in df.columns:
                df['Büyüme_23_24'] = ((df['Satış_2024'] - df['Satış_2023']) / 
                                      df['Satış_2023'].replace(0, np.nan)) * 100
            
            # Pazar payı hesapla
            if 'Satış_2024' in df.columns:
                toplam_satış = df['Satış_2024'].sum()
                if toplam_satış > 0:
                    df['Pazar_Payı'] = (df['Satış_2024'] / toplam_satış) * 100
            
            # Fiyat-Hacim Oranı
            if 'Ort_Fiyat_2024' in df.columns and 'Units_2024' in df.columns:
                df['Fiyat_Hacim_Oranı'] = df['Ort_Fiyat_2024'] * df['Units_2024']
            
            return df
            
        except Exception as e:
            st.warning(f"Analiz verisi hazırlama hatası: {str(e)}")
            return df

# ================================================
# 3. GELİŞMİŞ FİLTRELEME SİSTEMİ
# ================================================

class AdvancedFilterSystem:
    """Gelişmiş filtreleme sistemi"""
    
    @staticmethod
    def create_filter_sidebar(df):
        """Filtreleme sidebar'ını oluştur"""
        with st.sidebar.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
            st.markdown('<div class="filter-title">🔍 Arama ve Filtreleme</div>', unsafe_allow_html=True)
            
            search_term = st.text_input(
                "🔎 Global Arama",
                placeholder="Molekül, Şirket, Ülke...",
                help="Tüm sütunlarda arama yapın",
                key="global_search"
            )
            
            filter_config = {}
            available_columns = df.columns.tolist()
            
            if 'Ülke' in available_columns:
                ülkeler = sorted(df['Ülke'].dropna().unique())
                selected_ülkeler = AdvancedFilterSystem.create_searchable_multiselect_with_all(
                    "🌍 Ülkeler",
                    ülkeler,
                    key="ülkeler_filter",
                    select_all_by_default=True
                )
                if selected_ülkeler and "Tümü" not in selected_ülkeler:
                    filter_config['Ülke'] = selected_ülkeler
            
            if 'Şirket' in available_columns:
                şirketler = sorted(df['Şirket'].dropna().unique())
                selected_şirketler = AdvancedFilterSystem.create_searchable_multiselect_with_all(
                    "🏢 Şirketler",
                    şirketler,
                    key="şirketler_filter",
                    select_all_by_default=True
                )
                if selected_şirketler and "Tümü" not in selected_şirketler:
                    filter_config['Şirket'] = selected_şirketler
            
            if 'Molekül' in available_columns:
                moleküller = sorted(df['Molekül'].dropna().unique())
                selected_moleküller = AdvancedFilterSystem.create_searchable_multiselect_with_all(
                    "🧪 Moleküller",
                    moleküller,
                    key="moleküller_filter",
                    select_all_by_default=True
                )
                if selected_moleküller and "Tümü" not in selected_moleküller:
                    filter_config['Molekül'] = selected_moleküller
            
            st.markdown("---")
            st.markdown('<div class="filter-title">📊 Sayısal Filtreler</div>', unsafe_allow_html=True)
            
            satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
            if satış_sütunları:
                son_satış_sütunu = satış_sütunları[-1]
                min_satış = float(df[son_satış_sütunu].min())
                max_satış = float(df[son_satış_sütunu].max())
                
                col_slider1, col_slider2 = st.columns(2)
                with col_slider1:
                    min_değer = st.number_input(
                        "Min Satış ($)",
                        min_value=min_satış,
                        max_value=max_satış,
                        value=min_satış,
                        step=1000.0,
                        key="satış_min"
                    )
                with col_slider2:
                    max_değer = st.number_input(
                        "Max Satış ($)",
                        min_value=min_satış,
                        max_value=max_satış,
                        value=max_satış,
                        step=1000.0,
                        key="satış_max"
                    )
                
                if min_değer <= max_değer:
                    filter_config['satış_aralığı'] = ((min_değer, max_değer), son_satış_sütunu)
                else:
                    st.warning("Min değer Max değerden küçük olmalıdır")
            
            büyüme_sütunları = [col for col in df.columns if 'Büyüme' in col or 'Growth' in col]
            if büyüme_sütunları:
                son_büyüme_sütunu = büyüme_sütunları[-1]
                min_büyüme = float(df[son_büyüme_sütunu].min())
                max_büyüme = float(df[son_büyüme_sütunu].max())
                
                col_büyüme1, col_büyüme2 = st.columns(2)
                with col_büyüme1:
                    min_büyüme_değer = st.number_input(
                        "Min Büyüme (%)",
                        min_value=min_büyüme,
                        max_value=max_büyüme,
                        value=min(min_büyüme, -50.0),
                        step=5.0,
                        key="büyüme_min"
                    )
                with col_büyüme2:
                    max_büyüme_değer = st.number_input(
                        "Max Büyüme (%)",
                        min_value=min_büyüme,
                        max_value=max_büyüme,
                        value=max(max_büyüme, 150.0),
                        step=5.0,
                        key="büyüme_max"
                    )
                
                if min_büyüme_değer <= max_büyüme_değer:
                    filter_config['büyüme_aralığı'] = ((min_büyüme_değer, max_büyüme_değer), son_büyüme_sütunu)
            
            st.markdown("---")
            st.markdown('<div class="filter-title">⚙️ Ek Filtreler</div>', unsafe_allow_html=True)
            
            sadece_pozitif_büyüme = st.checkbox("📈 Sadece Pozitif Büyüyen Ürünler", value=False)
            if sadece_pozitif_büyüme and büyüme_sütunları:
                filter_config['pozitif_büyüme'] = True
            
            if satış_sütunları:
                satış_eşiği = st.number_input(
                    "Satış Eşiği ($)",
                    min_value=0.0,
                    max_value=float(df[satış_sütunları[-1]].max()),
                    value=0.0,
                    step=1000.0,
                    key="satış_eşiği"
                )
                if satış_eşiği > 0:
                    filter_config['satış_eşiği'] = (satış_eşiği, satış_sütunları[-1])
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                filtre_uygula = st.button("✅ Filtre Uygula", width='stretch', key="filtre_uygula")
            with col2:
                filtre_temizle = st.button("🗑️ Filtreleri Temizle", width='stretch', key="filtre_temizle")
            with col3:
                filtre_kaydet = st.button("💾 Filtreyi Kaydet", width='stretch', key="filtre_kaydet")
            
            if 'kayıtlı_filtreler' not in st.session_state:
                st.session_state.kayıtlı_filtreler = {}
            
            if filtre_kaydet and filter_config:
                filtre_adı = st.text_input("Filtre Adı", placeholder="Örn: Yüksek Büyüyen Ürünler")
                if filtre_adı:
                    st.session_state.kayıtlı_filtreler[filtre_adı] = filter_config
                    st.success(f"✅ '{filtre_adı}' filtresi kaydedildi!")
            
            if st.session_state.kayıtlı_filtreler:
                st.markdown('<div class="filter-title">💾 Kayıtlı Filtreler</div>', unsafe_allow_html=True)
                kayıtlı_filtre = st.selectbox(
                    "Kayıtlı Filtreler",
                    options=[""] + list(st.session_state.kayıtlı_filtreler.keys()),
                    key="kayıtlı_filtreler_select"
                )
                
                if kayıtlı_filtre:
                    if st.button("📂 Bu Filtreyi Yükle", width='stretch'):
                        st.session_state.mevcut_filtreler = st.session_state.kayıtlı_filtreler[kayıtlı_filtre]
                        st.success(f"✅ '{kayıtlı_filtre}' filtresi yüklendi!")
                        st.rerun()
            
            return search_term, filter_config, filtre_uygula, filtre_temizle
    
    @staticmethod
    def create_searchable_multiselect_with_all(label, options, key, select_all_by_default=False):
        """Arama yapılabilir multiselect - Tümü seçeneği dahil"""
        if not options:
            return []
        
        tümü_seçenekler = ["Tümü"] + options
        
        arama_sorgusu = st.text_input(f"{label} Ara", key=f"{key}_arama", placeholder="Arama yapın...")
        
        if arama_sorgusu:
            filtrelenmiş_seçenekler = ["Tümü"] + [opt for opt in options if arama_sorgusu.lower() in str(opt).lower()]
        else:
            filtrelenmiş_seçenekler = tümü_seçenekler
        
        if select_all_by_default:
            varsayılan_seçenekler = ["Tümü"]
        else:
            varsayılan_seçenekler = filtrelenmiş_seçenekler[:min(5, len(filtrelenmiş_seçenekler))]
        
        seçilenler = st.multiselect(
            label,
            options=filtrelenmiş_seçenekler,
            default=varsayılan_seçenekler,
            key=key,
            help="'Tümü' seçildiğinde diğer tüm seçenekler otomatik seçilir"
        )
        
        if "Tümü" in seçilenler and len(seçilenler) > 1:
            seçilenler = [opt for opt in seçilenler if opt != "Tümü"]
        elif "Tümü" in seçilenler and len(seçilenler) == 1:
            seçilenler = options
        
        if seçilenler:
            if len(seçilenler) == len(options):
                st.caption(f"✅ TÜMÜ seçildi ({len(options)} öğe)")
            else:
                st.caption(f"✅ {len(seçilenler)} / {len(options)} seçildi")
        
        return seçilenler
    
    @staticmethod
    def apply_filters(df, search_term, filter_config):
        """Filtreleri uygula"""
        filtrelenmiş_df = df.copy()
        
        if search_term:
            arama_maske = pd.Series(False, index=filtrelenmiş_df.index)
            for col in filtrelenmiş_df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(filtrelenmiş_df[col]):
                        arama_maske = arama_maske | filtrelenmiş_df[col].astype(str).str.contains(
                            search_term, case=False, na=False
                        )
                    else:
                        arama_maske = arama_maske | filtrelenmiş_df[col].astype(str).str.contains(
                            search_term, case=False, na=False
                        )
                except:
                    continue
            filtrelenmiş_df = filtrelenmiş_df[arama_maske]
            if len(filtrelenmiş_df) == 0:
                st.warning("Arama sonucu bulunamadı!")
        
        for sütun, değerler in filter_config.items():
            if sütun in filtrelenmiş_df.columns and değerler and sütun not in ['satış_aralığı', 'büyüme_aralığı', 'pozitif_büyüme', 'satış_eşiği']:
                filtrelenmiş_df = filtrelenmiş_df[filtrelenmiş_df[sütun].isin(değerler)]
        
        if 'satış_aralığı' in filter_config:
            (min_değer, max_değer), sütun_adı = filter_config['satış_aralığı']
            if sütun_adı in filtrelenmiş_df.columns:
                filtrelenmiş_df = filtrelenmiş_df[
                    (filtrelenmiş_df[sütun_adı] >= min_değer) & 
                    (filtrelenmiş_df[sütun_adı] <= max_değer)
                ]
        
        if 'büyüme_aralığı' in filter_config:
            (min_değer, max_değer), sütun_adı = filter_config['büyüme_aralığı']
            if sütun_adı in filtrelenmiş_df.columns:
                filtrelenmiş_df = filtrelenmiş_df[
                    (filtrelenmiş_df[sütun_adı] >= min_değer) & 
                    (filtrelenmiş_df[sütun_adı] <= max_değer)
                ]
        
        if 'pozitif_büyüme' in filter_config and filter_config['pozitif_büyüme']:
            büyüme_sütunları = [col for col in filtrelenmiş_df.columns if 'Büyüme' in col or 'Growth' in col]
            if büyüme_sütunları:
                filtrelenmiş_df = filtrelenmiş_df[filtrelenmiş_df[büyüme_sütunları[-1]] > 0]
        
        if 'satış_eşiği' in filter_config:
            eşik, sütun_adı = filter_config['satış_eşiği']
            if sütun_adı in filtrelenmiş_df.columns:
                filtrelenmiş_df = filtrelenmiş_df[filtrelenmiş_df[sütun_adı] >= eşik]
        
        return filtrelenmiş_df
    
    @staticmethod
    def show_filter_status(current_filters, filtered_df, original_df):
        """Filtre durumunu göster"""
        if current_filters:
            filtre_bilgisi = f"🎯 **Aktif Filtreler:** "
            filtre_maddeleri = []
            
            for key, value in current_filters.items():
                if key in ['Ülke', 'Şirket', 'Molekül']:
                    if isinstance(value, list):
                        if len(value) > 3:
                            filtre_maddeleri.append(f"{key}: {len(value)} seçenek")
                        else:
                            filtre_maddeleri.append(f"{key}: {', '.join(value[:3])}")
                elif key == 'satış_aralığı':
                    (min_val, max_val), sütun_adı = value
                    filtre_maddeleri.append(f"Satış: ${min_val:,.0f}-${max_val:,.0f}")
                elif key == 'büyüme_aralığı':
                    (min_val, max_val), sütun_adı = value
                    filtre_maddeleri.append(f"Büyüme: %{min_val:.1f}-%{max_val:.1f}")
                elif key == 'pozitif_büyüme':
                    filtre_maddeleri.append("Pozitif Büyüme")
                elif key == 'satış_eşiği':
                    eşik, sütun_adı = value
                    filtre_maddeleri.append(f"Satış > ${eşik:,.0f}")
            
            filtre_bilgisi += " | ".join(filtre_maddeleri)
            filtre_bilgisi += f" | **Gösterilen:** {len(filtered_df):,} / {len(original_df):,} satır"
            
            st.markdown(f'<div class="filter-status">{filtre_bilgisi}</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("❌ Tüm Filtreleri Temizle", width='stretch', key="tüm_filtreleri_temizle"):
                    st.session_state.filtered_df = st.session_state.df.copy()
                    st.session_state.mevcut_filtreler = {}
                    st.session_state.metrics = AdvancedPharmaAnalytics().calculate_comprehensive_metrics(st.session_state.df)
                    st.session_state.insights = AdvancedPharmaAnalytics().detect_strategic_insights(st.session_state.df)
                    st.success("✅ Tüm filtreler temizlendi")
                    st.rerun()

# ================================================
# 4. GELİŞMİŞ ANALİTİK MOTORU
# ================================================

class AdvancedPharmaAnalytics:
    """Gelişmiş farma analitik motoru"""
    
    @staticmethod
    def calculate_comprehensive_metrics(df):
        """Kapsamlı pazar metrikleri"""
        metrics = {}
        
        try:
            metrics['Toplam_Satır'] = len(df)
            metrics['Toplam_Sütun'] = len(df.columns)
            
            satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
            if satış_sütunları:
                son_satış_sütunu = satış_sütunları[-1]
                metrics['Son_Satış_Yılı'] = son_satış_sütunu.split('_')[-1] if '_' in son_satış_sütunu else '2024'
                metrics['Toplam_Pazar_Değeri'] = df[son_satış_sütunu].sum()
                metrics['Ort_Satış_Per_Ürün'] = df[son_satış_sütunu].mean()
                metrics['Medyan_Satış'] = df[son_satış_sütunu].median()
                metrics['Satış_Std_Sapma'] = df[son_satış_sütunu].std()
                
                metrics['Satış_Q1'] = df[son_satış_sütunu].quantile(0.25)
                metrics['Satış_Q3'] = df[son_satış_sütunu].quantile(0.75)
                metrics['Satış_IQR'] = metrics['Satış_Q3'] - metrics['Satış_Q1']
            
            büyüme_sütunları = [col for col in df.columns if 'Büyüme' in col or 'Growth' in col]
            if büyüme_sütunları:
                son_büyüme_sütunu = büyüme_sütunları[-1]
                metrics['Ort_Büyüme_Oranı'] = df[son_büyüme_sütunu].mean()
                metrics['Büyüme_Std_Sapma'] = df[son_büyüme_sütunu].std()
                metrics['Pozitif_Büyüme_Ürünleri'] = (df[son_büyüme_sütunu] > 0).sum()
                metrics['Negatif_Büyüme_Ürünleri'] = (df[son_büyüme_sütunu] < 0).sum()
                metrics['Yüksek_Büyüme_Ürünleri'] = (df[son_büyüme_sütunu] > 20).sum()
            
            if 'Şirket' in df.columns and satış_sütunları:
                son_satış_sütunu = satış_sütunları[-1]
                şirket_satışları = df.groupby('Şirket')[son_satış_sütunu].sum().sort_values(ascending=False)
                toplam_satış = şirket_satışları.sum()
                
                if toplam_satış > 0:
                    pazar_payları = (şirket_satışları / toplam_satış * 100)
                    metrics['HHI_Endeksi'] = (pazar_payları ** 2).sum() / 10000
                    
                    top_n = [1, 3, 5, 10]
                    for n in top_n:
                        metrics[f'Top_{n}_Pay'] = şirket_satışları.nlargest(n).sum() / toplam_satış * 100
                    
                    metrics['CR4_Oranı'] = metrics['Top_4_Pay'] if 'Top_4_Pay' in metrics else 0
            
            if 'Molekül' in df.columns:
                metrics['Benzersiz_Moleküller'] = df['Molekül'].nunique()
                if satış_sütunları:
                    molekül_satışları = df.groupby('Molekül')[son_satış_sütunu].sum()
                    toplam_molekül_satış = molekül_satışları.sum()
                    if toplam_molekül_satış > 0:
                        metrics['Top_10_Molekül_Payı'] = molekül_satışları.nlargest(10).sum() / toplam_molekül_satış * 100
            
            if 'Ülke' in df.columns:
                metrics['Ülke_Kapsamı'] = df['Ülke'].nunique()
                if satış_sütunları:
                    ülke_satışları = df.groupby('Ülke')[son_satış_sütunu].sum()
                    metrics['Top_5_Ülke_Payı'] = ülke_satışları.nlargest(5).sum() / ülke_satışları.sum() * 100
            
            fiyat_sütunları = [col for col in df.columns if 'Fiyat' in col or 'Price' in col]
            if fiyat_sütunları:
                son_fiyat_sütunu = fiyat_sütunları[-1]
                metrics['Ort_Fiyat'] = df[son_fiyat_sütunu].mean()
                metrics['Fiyat_Varyansı'] = df[son_fiyat_sütunu].var()
                metrics['Fiyat_CV'] = (df[son_fiyat_sütunu].std() / df[son_fiyat_sütunu].mean()) * 100 if df[son_fiyat_sütunu].mean() > 0 else 0
                
                fiyat_çeyreklikleri = df[son_fiyat_sütunu].quantile([0.25, 0.5, 0.75])
                metrics['Fiyat_Q1'] = fiyat_çeyreklikleri[0.25]
                metrics['Fiyat_Medyan'] = fiyat_çeyreklikleri[0.5]
                metrics['Fiyat_Q3'] = fiyat_çeyreklikleri[0.75]
            
            metrics['Eksik_Değerler'] = df.isnull().sum().sum()
            metrics['Eksik_Yüzde'] = (metrics['Eksik_Değerler'] / (len(df) * len(df.columns))) * 100
            
            # International Product analizi
            if 'Molekül' in df.columns and satış_sütunları:
                metrics = AdvancedPharmaAnalytics.add_international_product_metrics(df, metrics, satış_sütunları)
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return {}
    
    @staticmethod
    def add_international_product_metrics(df, metrics, satış_sütunları):
        """International Product analiz metriklerini ekle"""
        try:
            son_satış_sütunu = satış_sütunları[-1]
            
            international_ürünler = {}
            
            for molekül in df['Molekül'].unique():
                molekül_df = df[df['Molekül'] == molekül]
                
                benzersiz_şirketler = molekül_df['Şirket'].nunique() if 'Şirket' in df.columns else 0
                benzersiz_ülkeler = molekül_df['Ülke'].nunique() if 'Ülke' in df.columns else 0
                
                if benzersiz_şirketler > 1 or benzersiz_ülkeler > 1:
                    toplam_satış = molekül_df[son_satış_sütunu].sum()
                    if toplam_satış > 0:
                        international_ürünler[molekül] = {
                            'toplam_satış': toplam_satış,
                            'şirket_sayısı': benzersiz_şirketler,
                            'ülke_sayısı': benzersiz_ülkeler,
                            'ürün_sayısı': len(molekül_df),
                            'ort_büyüme': molekül_df['Büyüme_23_24'].mean() if 'Büyüme_23_24' in df.columns else None
                        }
            
            metrics['International_Ürün_Sayısı'] = len(international_ürünler)
            metrics['International_Ürün_Satışları'] = sum(data['toplam_satış'] for data in international_ürünler.values())
            metrics['International_Ürün_Payı'] = (metrics['International_Ürün_Satışları'] / metrics['Toplam_Pazar_Değeri'] * 100) if metrics.get('Toplam_Pazar_Değeri', 0) > 0 else 0
            
            if international_ürünler:
                metrics['Ort_International_Şirketler'] = np.mean([data['şirket_sayısı'] for data in international_ürünler.values()])
                metrics['Ort_International_Ülkeler'] = np.mean([data['ülke_sayısı'] for data in international_ürünler.values()])
            
            top_international = sorted(international_ürünler.items(), 
                                     key=lambda x: x[1]['toplam_satış'], 
                                     reverse=True)[:10]
            
            metrics['Top_10_International_Satış'] = sum(data['toplam_satış'] for _, data in top_international)
            metrics['Top_10_International_Pay'] = (metrics['Top_10_International_Satış'] / metrics['International_Ürün_Satışları'] * 100) if metrics.get('International_Ürün_Satışları', 0) > 0 else 0
            
            if 'Büyüme_23_24' in df.columns:
                international_büyüme = []
                yerel_büyüme = []
                
                for molekül in df['Molekül'].unique():
                    molekül_df = df[df['Molekül'] == molekül]
                    ort_büyüme = molekül_df['Büyüme_23_24'].mean()
                    
                    if molekül in international_ürünler:
                        international_büyüme.append(ort_büyüme)
                    else:
                        yerel_büyüme.append(ort_büyüme)
                
                if international_büyüme and yerel_büyüme:
                    metrics['International_Ort_Büyüme'] = np.mean(international_büyüme)
                    metrics['Yerel_Ort_Büyüme'] = np.mean(yerel_büyüme)
                    metrics['International_Büyüme_Premium'] = metrics['International_Ort_Büyüme'] - metrics['Yerel_Ort_Büyüme']
            
            return metrics
            
        except Exception as e:
            st.warning(f"International Product metrik hatası: {str(e)}")
            return metrics
    
    @staticmethod
    def analyze_international_products(df):
        """International Product detaylı analizi"""
        try:
            if 'Molekül' not in df.columns:
                return None
            
            satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
            if not satış_sütunları:
                return None
            
            son_satış_sütunu = satış_sütunları[-1]
            
            international_analiz = []
            
            for molekül in df['Molekül'].unique():
                molekül_df = df[df['Molekül'] == molekül]
                
                benzersiz_şirketler = molekül_df['Şirket'].nunique() if 'Şirket' in df.columns else 0
                benzersiz_ülkeler = molekül_df['Ülke'].nunique() if 'Ülke' in df.columns else 0
                
                is_international = (benzersiz_şirketler > 1 or benzersiz_ülkeler > 1)
                
                toplam_satış = molekül_df[son_satış_sütunu].sum()
                ort_fiyat = molekül_df['Ort_Fiyat_2024'].mean() if 'Ort_Fiyat_2024' in molekül_df.columns else None
                ort_büyüme = molekül_df['Büyüme_23_24'].mean() if 'Büyüme_23_24' in molekül_df.columns else None
                
                if 'Şirket' in df.columns:
                    top_şirket = molekül_df.groupby('Şirket')[son_satış_sütunu].sum().idxmax() if not molekül_df['Şirket'].empty else None
                    şirket_pazar_payı = (molekül_df[molekül_df['Şirket'] == top_şirket][son_satış_sütunu].sum() / toplam_satış * 100) if toplam_satış > 0 and top_şirket else 0
                else:
                    top_şirket = None
                    şirket_pazar_payı = 0
                
                if 'Ülke' in df.columns:
                    top_ülke = molekül_df.groupby('Ülke')[son_satış_sütunu].sum().idxmax() if not molekül_df['Ülke'].empty else None
                    ülke_pazar_payı = (molekül_df[molekül_df['Ülke'] == top_ülke][son_satış_sütunu].sum() / toplam_satış * 100) if toplam_satış > 0 and top_ülke else 0
                else:
                    top_ülke = None
                    ülke_pazar_payı = 0
                
                karmaşıklık_puanı = (benzersiz_şirketler * 0.6 + benzersiz_ülkeler * 0.4) / 2
                
                international_analiz.append({
                    'Molekül': molekül,
                    'international_mı': is_international,
                    'toplam_satış': toplam_satış,
                    'şirket_sayısı': benzersiz_şirketler,
                    'ülke_sayısı': benzersiz_ülkeler,
                    'ürün_sayısı': len(molekül_df),
                    'ort_fiyat': ort_fiyat,
                    'ort_büyüme': ort_büyüme,
                    'top_şirket': top_şirket,
                    'şirket_pazar_payı': şirket_pazar_payı,
                    'top_ülke': top_ülke,
                    'ülke_pazar_payı': ülke_pazar_payı,
                    'karmaşıklık_puanı': karmaşıklık_puanı,
                    'satış_konsantrasyonu': max(şirket_pazar_payı, ülke_pazar_payı)
                })
            
            analiz_df = pd.DataFrame(international_analiz)
            
            if len(analiz_df) > 0 and 'karmaşıklık_puanı' in analiz_df.columns:
                analiz_df['international_segment'] = pd.cut(
                    analiz_df['karmaşıklık_puanı'],
                    bins=[0, 0.5, 1.5, 3, float('inf')],
                    labels=['Yerel', 'Bölgesel', 'Çok-Ulusal', 'Global']
                )
            
            return analiz_df.sort_values('toplam_satış', ascending=False)
            
        except Exception as e:
            st.warning(f"International Product analiz hatası: {str(e)}")
            return None
    
    @staticmethod
    def get_international_product_insights(df):
        """International Product içgörüleri"""
        içgörüler = []
        
        try:
            analiz_df = AdvancedPharmaAnalytics.analyze_international_products(df)
            
            if analiz_df is None or len(analiz_df) == 0:
                return içgörüler
            
            international_sayısı = analiz_df['international_mı'].sum()
            toplam_molekül = len(analiz_df)
            international_yüzde = (international_sayısı / toplam_molekül * 100) if toplam_molekül > 0 else 0
            
            içgörüler.append({
                'type': 'info',
                'title': f'🌍 International Ürün Dağılımı',
                'description': f"Toplam {toplam_molekül} molekülden {international_sayısı} tanesi (%{international_yüzde:.1f}) International Ürün.",
                'data': analiz_df[analiz_df['international_mı']]
            })
            
            international_df = analiz_df[analiz_df['international_mı']]
            if len(international_df) > 0:
                toplam_international_satış = international_df['toplam_satış'].sum()
                toplam_satış = df['Satış_2024'].sum() if 'Satış_2024' in df.columns else 0
                
                if toplam_satış > 0:
                    international_satış_payı = (toplam_international_satış / toplam_satış * 100)
                    
                    içgörüler.append({
                        'type': 'success',
                        'title': f'💰 International Ürün Pazar Payı',
                        'description': f"International Ürünler toplam pazarın %{international_satış_payı:.1f}'ini oluşturuyor.",
                        'data': None
                    })
            
            top_international = analiz_df[analiz_df['international_mı']].nlargest(5, 'toplam_satış')
            if len(top_international) > 0:
                top_molekül = top_international.iloc[0]['Molekül']
                top_satış = top_international.iloc[0]['toplam_satış']
                
                içgörüler.append({
                    'type': 'warning',
                    'title': f'🏆 En Büyük International Ürün',
                    'description': f"{top_molekül} ${top_satış/1e6:.1f}M satış ile en büyük International Ürün.",
                    'data': top_international
                })
            
            if 'ort_büyüme' in analiz_df.columns:
                international_büyüme = analiz_df[analiz_df['international_mı']]['ort_büyüme'].mean()
                yerel_büyüme = analiz_df[~analiz_df['international_mı']]['ort_büyüme'].mean()
                
                if not pd.isna(international_büyüme) and not pd.isna(yerel_büyüme):
                    büyüme_farkı = international_büyüme - yerel_büyüme
                    
                    if büyüme_farkı > 0:
                        içgörüler.append({
                            'type': 'success',
                            'title': f'📈 International Ürün Büyüme Avantajı',
                            'description': f"International Ürünler yerel ürünlerden %{büyüme_farkı:.1f} daha hızlı büyüyor.",
                            'data': None
                        })
                    else:
                        içgörüler.append({
                            'type': 'warning',
                            'title': f'⚠️ International Ürün Büyüme Riski',
                            'description': f"International Ürünler yerel ürünlerden %{abs(büyüme_farkı):.1f} daha yavaş büyüyor.",
                            'data': None
                        })
            
            if 'ülke_sayısı' in analiz_df.columns:
                ort_ülkeler = analiz_df[analiz_df['international_mı']]['ülke_sayısı'].mean()
                if not pd.isna(ort_ülkeler):
                    içgörüler.append({
                        'type': 'geographic',
                        'title': f'🗺️ Ortalama Coğrafi Yayılım',
                        'description': f"International Ürünler ortalama {ort_ülkeler:.1f} ülkede satılıyor.",
                        'data': None
                    })
            
            return içgörüler
            
        except Exception as e:
            st.warning(f"International Ürün içgörü hatası: {str(e)}")
            return []
    
    @staticmethod
    def analyze_market_trends(df):
        """Pazar trendlerini analiz et"""
        try:
            trends = {}
            
            satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
            if len(satış_sütunları) >= 2:
                yıllık_trend = {}
                for col in sorted(satış_sütunları):
                    yıl = col.split('_')[-1] if '_' in col else col
                    yıllık_trend[yıl] = df[col].sum()
                
                trends['Yıllık_Satışlar'] = yıllık_trend
                
                yıllar = sorted(yıllık_trend.keys())
                for i in range(1, len(yıllar)):
                    önceki_yıl = yıllar[i-1]
                    mevcut_yıl = yıllar[i]
                    büyüme = ((yıllık_trend[mevcut_yıl] - yıllık_trend[önceki_yıl]) / 
                              yıllık_trend[önceki_yıl] * 100) if yıllık_trend[önceki_yıl] > 0 else 0
                    trends[f'Büyüme_{önceki_yıl}_{mevcut_yıl}'] = büyüme
            
            return trends
            
        except Exception as e:
            st.warning(f"Trend analizi hatası: {str(e)}")
            return {}
    
    @staticmethod
    def perform_advanced_segmentation(df, n_clusters=4, method='kmeans'):
        """Gelişmiş pazar segmentasyonu"""
        try:
            özellikler = []
            
            satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
            if satış_sütunları:
                özellikler.extend(satış_sütunları[-2:])
            
            büyüme_sütunları = [col for col in df.columns if 'Büyüme' in col or 'Growth' in col]
            if büyüme_sütunları:
                özellikler.append(büyüme_sütunları[-1])
            
            fiyat_sütunları = [col for col in df.columns if 'Fiyat' in col or 'Price' in col]
            if fiyat_sütunları:
                özellikler.append(fiyat_sütunları[-1])
            
            if len(özellikler) < 2:
                st.warning("Segmentasyon için yeterli özellik bulunamadı")
                return None
            
            segmentasyon_verisi = df[özellikler].fillna(0)
            
            if len(segmentasyon_verisi) < n_clusters * 10:
                st.warning("Segmentasyon için yeterli veri noktası yok")
                return None
            
            scaler = StandardScaler()
            özellikler_scaled = scaler.fit_transform(segmentasyon_verisi)
            
            if method == 'kmeans':
                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10,
                    max_iter=300,
                    tol=1e-4
                )
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=10)
            else:
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            clusters = model.fit_predict(özellikler_scaled)
            
            if hasattr(model, 'inertia_'):
                inertia = model.inertia_
            else:
                inertia = None
            
            if len(np.unique(clusters)) > 1:
                try:
                    silhouette = silhouette_score(özellikler_scaled, clusters)
                    calinski = calinski_harabasz_score(özellikler_scaled, clusters)
                except:
                    silhouette = None
                    calinski = None
            else:
                silhouette = None
                calinski = None
            
            result_df = df.copy()
            result_df['Segment'] = clusters
            
            segment_isimleri = {
                0: 'Gelişen Ürünler',
                1: 'Olgun Ürünler',
                2: 'Yenilikçi Ürünler',
                3: 'Riskli Ürünler',
                4: 'Niş Ürünler',
                5: 'Volume Ürünleri',
                6: 'Premium Ürünler',
                7: 'Ekonomi Ürünler'
            }
            
            result_df['Segment_İsmi'] = result_df['Segment'].map(
                lambda x: segment_isimleri.get(x, f'Segment_{x}')
            )
            
            return {
                'data': result_df,
                'metrics': {
                    'inertia': inertia,
                    'silhouette_score': silhouette,
                    'calinski_score': calinski,
                    'n_clusters': len(np.unique(clusters))
                },
                'features_used': özellikler
            }
            
        except Exception as e:
            st.warning(f"Segmentasyon hatası: {str(e)}")
            return None
    
    @staticmethod
    def detect_strategic_insights(df):
        """Stratejik içgörüleri tespit et"""
        içgörüler = []
        
        try:
            satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
            if not satış_sütunları:
                return içgörüler
            
            son_satış_sütunu = satış_sütunları[-1]
            yıl = son_satış_sütunu.split('_')[-1] if '_' in son_satış_sütunu else '2024'
            
            büyüme_sütunları = [col for col in df.columns if 'Büyüme' in col or 'Growth' in col]
            son_büyüme_sütunu = büyüme_sütunları[-1] if büyüme_sütunları else None
            
            # 1. En çok satan ürünler
            top_ürünler = df.nlargest(10, son_satış_sütunu)
            içgörüler.append({
                'type': 'success',
                'title': f'🏆 Top 10 Ürün - {yıl}',
                'description': f"En çok satan 10 ürün toplam pazarın %{(top_ürünler[son_satış_sütunu].sum() / df[son_satış_sütunu].sum() * 100):.1f}'ini oluşturuyor.",
                'data': top_ürünler
            })
            
            # 2. En hızlı büyüyen ürünler
            if son_büyüme_sütunu:
                top_büyüme = df.nlargest(10, son_büyüme_sütunu)
                içgörüler.append({
                    'type': 'info',
                    'title': f'🚀 En Hızlı Büyüyen 10 Ürün',
                    'description': f"En hızlı büyüyen ürünler ortalama %{top_büyüme[son_büyüme_sütunu].mean():.1f} büyüme gösteriyor.",
                    'data': top_büyüme
                })
            
            # 3. En çok satan şirketler
            if 'Şirket' in df.columns:
                top_şirketler = df.groupby('Şirket')[son_satış_sütunu].sum().nlargest(5)
                top_şirket = top_şirketler.index[0]
                top_şirket_payı = (top_şirketler.iloc[0] / df[son_satış_sütunu].sum()) * 100
                
                içgörüler.append({
                    'type': 'warning',
                    'title': f'🏢 Pazar Lideri - {yıl}',
                    'description': f"{top_şirket} %{top_şirket_payı:.1f} pazar payı ile lider konumda.",
                    'data': None
                })
            
            # 4. Coğrafi dağılım
            if 'Ülke' in df.columns:
                top_ülkeler = df.groupby('Ülke')[son_satış_sütunu].sum().nlargest(5)
                top_ülke = top_ülkeler.index[0]
                top_ülke_payı = (top_ülkeler.iloc[0] / df[son_satış_sütunu].sum()) * 100
                
                içgörüler.append({
                    'type': 'geographic',
                    'title': f'🌍 En Büyük Pazar - {yıl}',
                    'description': f"{top_ülke} %{top_ülke_payı:.1f} pay ile en büyük pazar.",
                    'data': None
                })
            
            # 5. Fiyat analizi
            fiyat_sütunları = [col for col in df.columns if 'Fiyat' in col or 'Price' in col]
            if fiyat_sütunları:
                ort_fiyat = df[fiyat_sütunları[-1]].mean()
                fiyat_std = df[fiyat_sütunları[-1]].std()
                
                içgörüler.append({
                    'type': 'price',
                    'title': f'💰 Fiyat Analizi - {yıl}',
                    'description': f"Ortalama fiyat: ${ort_fiyat:.2f} (Standart sapma: ${fiyat_std:.2f})",
                    'data': None
                })
            
            # 6. International Product içgörüleri
            international_içgörüler = AdvancedPharmaAnalytics.get_international_product_insights(df)
            içgörüler.extend(international_içgörüler)
            
            return içgörüler
            
        except Exception as e:
            st.warning(f"İçgörü tespiti hatası: {str(e)}")
            return []

# ================================================
# 5. GÖRSELLEŞTİRME MOTORU
# ================================================

class ProfessionalVisualization:
    """Profesyonel görselleştirme motoru"""
    
    @staticmethod
    def create_dashboard_metrics(df, metrics):
        """Dashboard metrik kartlarını oluştur"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                toplam_satış = metrics.get('Toplam_Pazar_Değeri', 0)
                satış_yılı = metrics.get('Son_Satış_Yılı', '')
                st.markdown(f"""
                <div class="custom-metric-card premium">
                    <div class="custom-metric-label">TOPLAM PAZAR DEĞERİ</div>
                    <div class="custom-metric-value">${toplam_satış/1e9:.2f}B</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">{satış_yılı}</span>
                        <span>Global Pazar</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                ort_büyüme = metrics.get('Ort_Büyüme_Oranı', 0)
                büyüme_class = "success" if ort_büyüme > 0 else "danger"
                st.markdown(f"""
                <div class="custom-metric-card {büyüme_class}">
                    <div class="custom-metric-label">ORTALAMA BÜYÜME</div>
                    <div class="custom-metric-value">{ort_büyüme:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">YoY</span>
                        <span>Yıllık Büyüme</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrics.get('HHI_Endeksi', 0)
                hhi_durum = "danger" if hhi > 2500 else "warning" if hhi > 1500 else "success"
                st.markdown(f"""
                <div class="custom-metric-card {hhi_durum}">
                    <div class="custom-metric-label">REKABET YOĞUNLUĞU</div>
                    <div class="custom-metric-value">{hhi:.0f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-warning">HHI Endeksi</span>
                        <span>{'Monopol' if hhi > 2500 else 'Oligopol' if hhi > 1500 else 'Rekabetçi'}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                international_pay = metrics.get('International_Ürün_Payı', 0)
                international_renk = "success" if international_pay > 20 else "warning" if international_pay > 10 else "info"
                st.markdown(f"""
                <div class="custom-metric-card {international_renk}">
                    <div class="custom-metric-label">INTERNATIONAL ÜRÜN PAYI</div>
                    <div class="custom-metric-value">{international_pay:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-cyan">Global Yayılım</span>
                        <span>Multi-Market</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                benzersiz_moleküller = metrics.get('Benzersiz_Moleküller', 0)
                st.markdown(f"""
                <div class="custom-metric-card info">
                    <div class="custom-metric-label">MOLEKÜL ÇEŞİTLİLİĞİ</div>
                    <div class="custom-metric-value">{benzersiz_moleküller:,}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Unique</span>
                        <span>Farklı Molekül</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                ort_fiyat = metrics.get('Ort_Fiyat', 0)
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
                yüksek_büyüme = metrics.get('Yüksek_Büyüme_Ürünleri', 0)
                toplam_ürünler = metrics.get('Toplam_Satır', 0)
                yüksek_büyüme_yüzde = (yüksek_büyüme / toplam_ürünler * 100) if toplam_ürünler > 0 else 0
                st.markdown(f"""
                <div class="custom-metric-card success">
                    <div class="custom-metric-label">YÜKSEK BÜYÜME</div>
                    <div class="custom-metric-value">{yüksek_büyüme_yüzde:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">{yüksek_büyüme} ürün</span>
                        <span>> %20 büyüme</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                ülke_kapsamı = metrics.get('Ülke_Kapsamı', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">COĞRAFİ YAYILIM</div>
                    <div class="custom-metric-value">{ülke_kapsamı}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-cyan">Ülke</span>
                        <span>Global Kapsam</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Metrik kartları oluşturma hatası: {str(e)}")
    
    @staticmethod
    def create_international_product_analysis(df, analysis_df):
        """International Product analiz grafikleri"""
        try:
            if analysis_df is None or len(analysis_df) == 0:
                return None
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('International vs Yerel Dağılımı', 'International Ürün Pazar Payı',
                               'Coğrafi Yayılım Analizi', 'Büyüme Performansı Karşılaştırması'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # International vs Yerel dağılımı
            international_sayıları = analysis_df['international_mı'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=['International', 'Yerel'],
                    values=international_sayıları.values,
                    hole=0.4,
                    marker_colors=['#2563eb', '#64748b'],
                    textinfo='percent+label',
                    textposition='outside'
                ),
                row=1, col=1
            )
            
            # Pazar payı karşılaştırması
            international_satış = analysis_df[analysis_df['international_mı']]['toplam_satış'].sum()
            yerel_satış = analysis_df[~analysis_df['international_mı']]['toplam_satış'].sum()
            
            fig.add_trace(
                go.Bar(
                    x=['International', 'Yerel'],
                    y=[international_satış, yerel_satış],
                    marker_color=['#2563eb', '#64748b'],
                    text=[f'${international_satış/1e6:.1f}M', f'${yerel_satış/1e6:.1f}M'],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Coğrafi yayılım (International Ürünler için)
            international_df = analysis_df[analysis_df['international_mı']]
            if len(international_df) > 0:
                ülke_dağılımı = international_df['ülke_sayısı'].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(
                        x=ülke_dağılımı.index.astype(str),
                        y=ülke_dağılımı.values,
                        marker_color='#06b6d4',
                        name='Ülke Sayısı'
                    ),
                    row=2, col=1
                )
            
            # Büyüme karşılaştırması
            if 'ort_büyüme' in analysis_df.columns:
                international_büyüme = analysis_df[analysis_df['international_mı']]['ort_büyüme'].mean()
                yerel_büyüme = analysis_df[~analysis_df['international_mı']]['ort_büyüme'].mean()
                
                if not pd.isna(international_büyüme) and not pd.isna(yerel_büyüme):
                    fig.add_trace(
                        go.Bar(
                            x=['International', 'Yerel'],
                            y=[international_büyüme, yerel_büyüme],
                            marker_color=['#2563eb', '#64748b'],
                            text=[f'{international_büyüme:.1f}%', f'{yerel_büyüme:.1f}%'],
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
                title_text="International Ürün Analizi",
                title_x=0.5,
                title_font=dict(size=20)
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"International Ürün grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_sales_trend_chart(df):
        """Satış trend grafikleri"""
        try:
            satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
            if len(satış_sütunları) >= 2:
                yıllık_veri = []
                for col in sorted(satış_sütunları):
                    yıl = col.split('_')[-1] if '_' in col else col
                    yıllık_veri.append({
                        'Yıl': yıl,
                        'Toplam_Satış': df[col].sum(),
                        'Ort_Satış': df[col].mean(),
                        'Ürün_Sayısı': (df[col] > 0).sum()
                    })
                
                yıllık_df = pd.DataFrame(yıllık_veri)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Yıllık Toplam Satış', 'Ortalama Satış Trendi', 
                                   'Ürün Sayısı Trendi', 'Büyüme Oranları'),
                    specs=[[{"type": "bar"}, {"type": "scatter"}],
                           [{"type": "bar"}, {"type": "bar"}]],
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )
                
                fig.add_trace(
                    go.Bar(
                        x=yıllık_df['Yıl'],
                        y=yıllık_df['Toplam_Satış'],
                        name='Toplam Satış',
                        marker_color='#2563eb',
                        text=[f'${x/1e6:.0f}M' for x in yıllık_df['Toplam_Satış']],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=yıllık_df['Yıl'],
                        y=yıllık_df['Ort_Satış'],
                        mode='lines+markers',
                        name='Ortalama Satış',
                        line=dict(color='#06b6d4', width=3),
                        marker=dict(size=10)
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(
                        x=yıllık_df['Yıl'],
                        y=yıllık_df['Ürün_Sayısı'],
                        name='Ürün Sayısı',
                        marker_color='#10b981',
                        text=yıllık_df['Ürün_Sayısı'],
                        textposition='auto'
                    ),
                    row=2, col=1
                )
                
                if len(yıllık_df) > 1:
                    büyüme_oranları = []
                    for i in range(1, len(yıllık_df)):
                        büyüme = ((yıllık_df['Toplam_Satış'].iloc[i] - yıllık_df['Toplam_Satış'].iloc[i-1]) / 
                                  yıllık_df['Toplam_Satış'].iloc[i-1] * 100) if yıllık_df['Toplam_Satış'].iloc[i-1] > 0 else 0
                        büyüme_oranları.append(büyüme)
                    
                    fig.add_trace(
                        go.Bar(
                            x=yıllık_df['Yıl'].iloc[1:],
                            y=büyüme_oranları,
                            name='Büyüme (%)',
                            marker_color=['#ef4444' if g < 0 else '#10b981' for g in büyüme_oranları],
                            text=[f'{g:.1f}%' for g in büyüme_oranları],
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
                    title_text="Satış Trendleri Analizi",
                    title_x=0.5,
                    title_font=dict(size=20)
                )
                
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False, gridcolor='rgba(255,255,255,0.1)')
                
                return fig
            
            return None
            
        except Exception as e:
            st.warning(f"Trend grafiği oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_market_share_analysis(df):
        """Pazar payı analiz grafikleri"""
        try:
            satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
            if not satış_sütunları:
                return None
            
            son_satış_sütunu = satış_sütunları[-1]
            
            if 'Şirket' in df.columns:
                şirket_satışları = df.groupby('Şirket')[son_satış_sütunu].sum().sort_values(ascending=False)
                top_şirketler = şirket_satışları.nlargest(15)
                diğer_satışlar = şirket_satışları.iloc[15:].sum() if len(şirket_satışları) > 15 else 0
                
                pie_veri = top_şirketler.copy()
                if diğer_satışlar > 0:
                    pie_veri['Diğer'] = diğer_satışlar
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Pazar Payı Dağılımı', 'Top 10 Şirket Satışları'),
                    specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                    column_widths=[0.4, 0.6]
                )
                
                fig.add_trace(
                    go.Pie(
                        labels=pie_veri.index,
                        values=pie_veri.values,
                        hole=0.4,
                        marker_colors=px.colors.qualitative.Bold,
                        textinfo='percent+label',
                        textposition='outside',
                        insidetextorientation='radial'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=top_şirketler.values[:10],
                        y=top_şirketler.index[:10],
                        orientation='h',
                        marker_color='#2563eb',
                        text=[f'${x/1e6:.1f}M' for x in top_şirketler.values[:10]],
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
    def create_geographic_distribution(df):
        """Coğrafi dağılım grafikleri"""
        try:
            satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
            if not satış_sütunları:
                return None
            
            son_satış_sütunu = satış_sütunları[-1]
            
            if 'Ülke' in df.columns:
                ülke_satışları = df.groupby('Ülke')[son_satış_sütunu].sum().reset_index()
                ülke_satışları = ülke_satışları.sort_values(son_satış_sütunu, ascending=False)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Top 15 Ülke', 'Coğrafi Satış Dağılımı'),
                    specs=[[{'type': 'bar'}, {'type': 'choropleth'}],
                           [{'type': 'treemap'}, {'type': 'scatter'}]],
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )
                
                # Top 15 Ülke
                top_ülkeler = ülke_satışları.head(15)
                fig.add_trace(
                    go.Bar(
                        x=top_ülkeler[son_satış_sütunu],
                        y=top_ülkeler['Ülke'],
                        orientation='h',
                        marker_color='#06b6d4',
                        text=[f'${x/1e6:.1f}M' for x in top_ülkeler[son_satış_sütunu]],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
                
                # Harita
                try:
                    fig.add_trace(
                        go.Choropleth(
                            locations=ülke_satışları['Ülke'],
                            locationmode='country names',
                            z=ülke_satışları[son_satış_sütunu],
                            colorscale='Blues',
                            colorbar_title="Satış (USD)",
                            hoverinfo='location+z'
                        ),
                        row=1, col=2
                    )
                except:
                    fig.add_trace(
                        go.Scatter(x=[0], y=[0], mode='text', text=['Harita yüklenemedi']),
                        row=1, col=2
                    )
                
                # Treemap
                fig.add_trace(
                    go.Treemap(
                        labels=ülke_satışları['Ülke'].head(20),
                        parents=[''] * min(20, len(ülke_satışları)),
                        values=ülke_satışları[son_satış_sütunu].head(20),
                        textinfo="label+value",
                        marker_colorscale='Viridis'
                    ),
                    row=2, col=1
                )
                
                # Büyüme-Satış dağılımı
                büyüme_sütunları = [col for col in df.columns if 'Büyüme' in col or 'Growth' in col]
                if büyüme_sütunları:
                    ülke_büyüme = df.groupby('Ülke')[büyüme_sütunları[-1]].mean().reset_index()
                    ülke_birleşik = pd.merge(ülke_satışları, ülke_büyüme, on='Ülke')
                    
                    fig.add_trace(
                        go.Scatter(
                            x=ülke_birleşik[son_satış_sütunu],
                            y=ülke_birleşik[büyüme_sütunları[-1]],
                            mode='markers',
                            marker=dict(
                                size=ülke_birleşik[son_satış_sütunu] / ülke_birleşik[son_satış_sütunu].max() * 50,
                                color=ülke_birleşik[büyüme_sütunları[-1]],
                                colorscale='RdYlGn',
                                showscale=True,
                                colorbar=dict(title="Büyüme %")
                            ),
                            text=ülke_birleşik['Ülke'],
                            hoverinfo='text+x+y'
                        ),
                        row=2, col=2
                    )
                
                fig.update_layout(
                    height=800,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    showlegend=False,
                    title_text="Coğrafi Analiz",
                    title_x=0.5
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.warning(f"Coğrafi analiz grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_price_volume_analysis(df):
        """Fiyat-hacim analiz grafikleri"""
        try:
            fiyat_sütunları = [col for col in df.columns if 'Fiyat' in col or 'Price' in col]
            hacim_sütunları = [col for col in df.columns if 'Units' in col or 'Adet' in col or 'Hacim' in col]
            
            if not fiyat_sütunları or not hacim_sütunları:
                return None
            
            son_fiyat_sütunu = fiyat_sütunları[-1]
            son_hacim_sütunu = hacim_sütunları[-1]
            
            örnek_df = df[
                (df[son_fiyat_sütunu] > 0) & 
                (df[son_hacim_sütunu] > 0)
            ].copy()
            
            if len(örnek_df) > 10000:
                örnek_df = örnek_df.sample(10000, random_state=42)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Fiyat-Hacim İlişkisi', 'Fiyat Dağılımı',
                               'Hacim Dağılımı', 'Fiyat-Hacim Kategorileri'),
                specs=[[{"type": "scatter"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "box"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # Fiyat-Hacim ilişkisi
            fig.add_trace(
                go.Scatter(
                    x=örnek_df[son_fiyat_sütunu],
                    y=örnek_df[son_hacim_sütunu],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=örnek_df[son_hacim_sütunu],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Hacim")
                    ),
                    text=örnek_df['Molekül'] if 'Molekül' in örnek_df.columns else None,
                    hoverinfo='text+x+y'
                ),
                row=1, col=1
            )
            
            # Fiyat dağılımı
            fig.add_trace(
                go.Histogram(
                    x=df[son_fiyat_sütunu],
                    nbinsx=50,
                    marker_color='#2563eb',
                    name='Fiyat Dağılımı'
                ),
                row=1, col=2
            )
            
            # Hacim dağılımı
            fig.add_trace(
                go.Histogram(
                    x=df[son_hacim_sütunu],
                    nbinsx=50,
                    marker_color='#10b981',
                    name='Hacim Dağılımı'
                ),
                row=2, col=1
            )
            
            # Şirket bazlı fiyat karşılaştırması
            if 'Şirket' in df.columns:
                top_şirketler = df['Şirket'].value_counts().nlargest(5).index
                şirket_veri = df[df['Şirket'].isin(top_şirketler)]
                
                fig.add_trace(
                    go.Box(
                        x=şirket_veri['Şirket'],
                        y=şirket_veri[son_fiyat_sütunu],
                        marker_color='#06b6d4',
                        name='Şirket Bazlı Fiyat'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                title_text="Fiyat-Hacim Analizi",
                title_x=0.5,
                title_font=dict(size=20)
            )
            
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False, gridcolor='rgba(255,255,255,0.1)')
            
            return fig
            
        except Exception as e:
            st.warning(f"Fiyat-hacim grafiği hatası: {str(e)}")
            return None

# ================================================
# 6. RAPORLAMA SİSTEMİ
# ================================================

class ProfessionalReporting:
    """Profesyonel raporlama sistemi"""
    
    @staticmethod
    def generate_excel_report(df, metrics, insights, analysis_df=None, file_name="pharma_report"):
        """Excel raporu oluştur"""
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='HAM_VERI', index=False)
                
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['METRİK', 'DEĞER'])
                metrics_df.to_excel(writer, sheet_name='ÖZET_METRİKLER', index=False)
                
                satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
                if satış_sütunları and 'Şirket' in df.columns:
                    son_satış_sütunu = satış_sütunları[-1]
                    pazar_payı = df.groupby('Şirket')[son_satış_sütunu].sum().sort_values(ascending=False)
                    pazar_payı_df = pazar_payı.reset_index()
                    pazar_payı_df.columns = ['ŞİRKET', 'SATIŞ']
                    pazar_payı_df['PAY (%)'] = (pazar_payı_df['SATIŞ'] / pazar_payı_df['SATIŞ'].sum()) * 100
                    pazar_payı_df['KÜMÜLATİF_PAY'] = pazar_payı_df['PAY (%)'].cumsum()
                    pazar_payı_df.to_excel(writer, sheet_name='PAZAR_PAYI', index=False)
                
                if 'Ülke' in df.columns:
                    if satış_sütunları:
                        son_satış_sütunu = satış_sütunları[-1]
                        ülke_analizi = df.groupby('Ülke').agg({
                            son_satış_sütunu: ['sum', 'mean', 'count']
                        }).round(2)
                        ülke_analizi.columns = ['_'.join(col).strip() for col in ülke_analizi.columns.values]
                        ülke_analizi.to_excel(writer, sheet_name='ÜLKE_ANALİZİ')
                
                if 'Molekül' in df.columns:
                    if satış_sütunları:
                        son_satış_sütunu = satış_sütunları[-1]
                        molekül_analizi = df.groupby('Molekül').agg({
                            son_satış_sütunu: ['sum', 'mean', 'count']
                        }).round(2)
                        molekül_analizi.columns = ['_'.join(col).strip() for col in molekül_analizi.columns.values]
                        molekül_analizi.nlargest(50, (son_satış_sütunu, 'sum')).to_excel(
                            writer, sheet_name='MOLEKÜL_ANALİZİ'
                        )
                
                if analysis_df is not None:
                    analysis_df.to_excel(writer, sheet_name='INTERNATIONAL_ANALİZİ', index=False)
                
                if insights:
                    insights_veri = []
                    for insight in insights:
                        insights_veri.append({
                            'TİP': insight['type'],
                            'BAŞLIK': insight['title'],
                            'AÇIKLAMA': insight['description']
                        })
                    
                    insights_df = pd.DataFrame(insights_veri)
                    insights_df.to_excel(writer, sheet_name='STRATEJİK_İÇGÖRÜLER', index=False)
                
                writer.save()
            
            output.seek(0)
            return output
            
        except Exception as e:
            st.error(f"Excel rapor oluşturma hatası: {str(e)}")
            return None

# ================================================
# 7. ANA UYGULAMA
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
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'mevcut_filtreler' not in st.session_state:
        st.session_state.mevcut_filtreler = {}
    if 'kayıtlı_filtreler' not in st.session_state:
        st.session_state.kayıtlı_filtreler = {}
    if 'international_analiz' not in st.session_state:
        st.session_state.international_analiz = None
    
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
        
        with st.expander("📁 VERİ YÜKLEME", expanded=True):
            yüklenen_dosya = st.file_uploader(
                "Excel/CSV Dosyası Yükleyin",
                type=['xlsx', 'xls', 'csv'],
                help="1M+ satır desteklenir. Büyük dosyalar için dikkatli olun."
            )
            
            if yüklenen_dosya:
                st.info("⚠️ Tüm veri seti yüklenecektir")
                st.info(f"Dosya: {yüklenen_dosya.name}")
                
                if st.button("🚀 Tüm Veriyi Yükle & Analiz Et", type="primary", width='stretch'):
                    with st.spinner("Tüm veri seti işleniyor..."):
                        processor = OptimizedDataProcessor()
                        
                        df = processor.load_large_dataset(yüklenen_dosya, sample_size=None)
                        
                        if df is not None and len(df) > 0:
                            df = processor.optimize_dataframe(df)
                            df = processor.prepare_analytics_data(df)
                            
                            st.session_state.df = df
                            st.session_state.filtered_df = df.copy()
                            
                            analytics = AdvancedPharmaAnalytics()
                            st.session_state.metrics = analytics.calculate_comprehensive_metrics(df)
                            st.session_state.insights = analytics.detect_strategic_insights(df)
                            st.session_state.international_analiz = analytics.analyze_international_products(df)
                            
                            st.success(f"✅ {len(df):,} satır TÜM VERİ başarıyla yüklendi!")
                            st.rerun()
        
        if st.session_state.df is not None:
            st.markdown("---")
            df = st.session_state.df
            
            filter_system = AdvancedFilterSystem()
            arama_terimi, filtre_config, filtre_uygula, filtre_temizle = filter_system.create_filter_sidebar(df)
            
            if filtre_uygula:
                with st.spinner("Filtreler uygulanıyor..."):
                    filtered_df = filter_system.apply_filters(df, arama_terimi, filtre_config)
                    st.session_state.filtered_df = filtered_df
                    st.session_state.mevcut_filtreler = filtre_config
                    
                    analytics = AdvancedPharmaAnalytics()
                    st.session_state.metrics = analytics.calculate_comprehensive_metrics(filtered_df)
                    st.session_state.insights = analytics.detect_strategic_insights(filtered_df)
                    st.session_state.international_analiz = analytics.analyze_international_products(filtered_df)
                    
                    st.success(f"✅ Filtreler uygulandı: {len(filtered_df):,} satır")
                    st.rerun()
            
            if filtre_temizle:
                st.session_state.filtered_df = st.session_state.df.copy()
                st.session_state.mevcut_filtreler = {}
                st.session_state.metrics = AdvancedPharmaAnalytics().calculate_comprehensive_metrics(st.session_state.df)
                st.session_state.insights = AdvancedPharmaAnalytics().detect_strategic_insights(st.session_state.df)
                st.session_state.international_analiz = AdvancedPharmaAnalytics().analyze_international_products(st.session_state.df)
                st.success("✅ Filtreler temizlendi")
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
        <strong>PharmaIntelligence Pro</strong><br>
        v3.2 | International Product Analytics<br>
        © 2024 Tüm hakları saklıdır.
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.df is None:
        show_welcome_screen()
        return
    
    df = st.session_state.filtered_df
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    international_analiz = st.session_state.international_analiz
    
    if st.session_state.mevcut_filtreler:
        AdvancedFilterSystem.show_filter_status(
            st.session_state.mevcut_filtreler,
            df,
            st.session_state.df
        )
    else:
        st.info(f"🎯 Aktif filtre yok | Gösterilen: {len(df):,} satır")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ",
        "💰 FİYAT ANALİZİ",
        "🏆 REKABET ANALİZİ",
        "🌍 INTERNATIONAL ÜRÜN",
        "🔮 STRATEJİK ANALİZ",
        "📑 RAPORLAMA"
    ])
    
    with tab1:
        show_overview_tab(df, metrics, insights)
    
    with tab2:
        show_market_analysis_tab(df)
    
    with tab3:
        show_price_analysis_tab(df)
    
    with tab4:
        show_competition_analysis_tab(df, metrics)
    
    with tab5:
        show_international_product_tab(df, international_analiz, metrics)
    
    with tab6:
        show_strategic_analysis_tab(df, insights)
    
    with tab7:
        show_reporting_tab(df, metrics, insights, international_analiz)

# ================================================
# TAB FONKSİYONLARI
# ================================================

def show_welcome_screen():
    """Hoşgeldiniz ekranını göster"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">💊</div>
            <h2 style="color: #f8fafc; margin-bottom: 1rem;">PharmaIntelligence Pro'ya Hoşgeldiniz</h2>
            <p style="color: #cbd5e1; margin-bottom: 2rem; line-height: 1.6;">
            İlaç pazarı verilerinizi yükleyin ve güçlü analitik özelliklerin kilidini açın.
            <br>International Ürün analizi ile çoklu pazar stratejilerinizi optimize edin.
            </p>
            
            <div class="feature-grid">
                <div class="feature-card feature-card-blue">
                    <div class="feature-icon">🌍</div>
                    <div class="feature-title">International Ürün</div>
                    <div class="feature-description">Çoklu pazar ürün analizi ve strateji geliştirme</div>
                </div>
                <div class="feature-card feature-card-cyan">
                    <div class="feature-icon">📈</div>
                    <div class="feature-title">Pazar Analizi</div>
                    <div class="feature-description">Derin pazar içgörüleri ve trend analizi</div>
                </div>
                <div class="feature-card feature-card-green">
                    <div class="feature-icon">💰</div>
                    <div class="feature-title">Fiyat Zekası</div>
                    <div class="feature-description">Rekabetçi fiyatlandırma ve optimizasyon analizi</div>
                </div>
                <div class="feature-card feature-card-yellow">
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

def show_overview_tab(df, metrics, insights):
    """Genel Bakış tab'ını göster"""
    st.markdown('<h2 class="section-title">Genel Bakış ve Performans Göstergeleri</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    viz.create_dashboard_metrics(df, metrics)
    
    st.markdown('<h3 class="subsection-title">🔍 Stratejik İçgörüler</h3>', unsafe_allow_html=True)
    
    if insights:
        insight_cols = st.columns(2)
        
        for idx, insight in enumerate(insights[:6]):
            with insight_cols[idx % 2]:
                icon = "💡"
                if insight['type'] == 'warning':
                    icon = "⚠️"
                elif insight['type'] == 'success':
                    icon = "✅"
                elif insight['type'] == 'info':
                    icon = "ℹ️"
                elif insight['type'] == 'geographic':
                    icon = "🌍"
                elif insight['type'] == 'price':
                    icon = "💰"
                
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <div class="insight-icon">{icon}</div>
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-content">{insight['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if insight.get('data') is not None and not insight['data'].empty:
                    with st.expander("📋 Detaylı Liste"):
                        display_columns = []
                        for col in ['Molekül', 'Şirket', 'Ülke', 'Satış_2024', 'Büyüme_23_24']:
                            if col in insight['data'].columns:
                                display_columns.append(col)
                        
                        if display_columns:
                            st.dataframe(
                                insight['data'][display_columns].head(10),
                                use_container_width=True
                            )
    else:
        st.info("Verileriniz analiz ediliyor... Stratejik içgörüler burada görünecek.")
    
    st.markdown('<h3 class="subsection-title">📋 Veri Önizleme</h3>', unsafe_allow_html=True)
    
    preview_col1, preview_col2 = st.columns([1, 3])
    
    with preview_col1:
        rows_to_show = st.slider("Gösterilecek Satır Sayısı", 10, 1000, 100, 10, key="rows_preview")
        
        available_columns = df.columns.tolist()
        default_columns = []
        
        priority_columns = ['Molekül', 'Şirket', 'Ülke', 'Satış_2024', 'Büyüme_23_24']
        for col in priority_columns:
            if col in available_columns:
                default_columns.append(col)
            if len(default_columns) >= 5:
                break
        
        if len(default_columns) < 5:
            default_columns.extend([col for col in available_columns[:5] if col not in default_columns])
        
        show_columns = st.multiselect(
            "Gösterilecek Sütunlar",
            options=available_columns,
            default=default_columns[:min(5, len(default_columns))],
            key="columns_preview"
        )
    
    with preview_col2:
        if show_columns:
            st.dataframe(
                df[show_columns].head(rows_to_show),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(
                df.head(rows_to_show),
                use_container_width=True,
                height=400
            )
    
    st.markdown('<h3 class="subsection-title">📊 Veri Kalitesi Analizi</h3>', unsafe_allow_html=True)
    
    quality_cols = st.columns(4)
    
    with quality_cols[0]:
        eksik_yüzde = metrics.get('Eksik_Yüzde', 0)
        durum_rengi = "normal"
        if eksik_yüzde < 5:
            durum_rengi = "normal"
        elif eksik_yüzde < 20:
            durum_rengi = "off"
        else:
            durum_rengi = "inverse"
        st.metric("Eksik Veri Oranı", f"{eksik_yüzde:.1f}%", delta=None, delta_color=durum_rengi)
    
    with quality_cols[1]:
        kopya_satırlar = df.duplicated().sum()
        kopya_yüzde = (kopya_satırlar / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Kopya Satırlar", f"{kopya_yüzde:.1f}%")
    
    with quality_cols[2]:
        sayısal_sütunlar = len(df.select_dtypes(include=[np.number]).columns)
        toplam_sütunlar = len(df.columns)
        st.metric("Sayısal Sütunlar", f"{sayısal_sütunlar}/{toplam_sütunlar}")
    
    with quality_cols[3]:
        tarih_sütunları = len([col for col in df.columns if 'date' in col.lower() or 'tarih' in col.lower()])
        st.metric("Tarih Sütunları", tarih_sütunları)

def show_market_analysis_tab(df):
    """Pazar Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Pazar Analizi ve Trendler</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    
    st.markdown('<h3 class="subsection-title">📈 Satış Trendleri</h3>', unsafe_allow_html=True)
    trend_fig = viz.create_sales_trend_chart(df)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Satış trend analizi için yeterli yıllık veri bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">🌍 Coğrafi Dağılım</h3>', unsafe_allow_html=True)
    geo_fig = viz.create_geographic_distribution(df)
    if geo_fig:
        st.plotly_chart(geo_fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Coğrafi analiz için yeterli veri bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">🧪 Molekül Bazlı Analiz</h3>', unsafe_allow_html=True)
    
    if 'Molekül' in df.columns:
        satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
        if satış_sütunları:
            son_satış_sütunu = satış_sütunları[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_moleküller = df.groupby('Molekül')[son_satış_sütunu].sum().nlargest(15)
                fig = px.bar(
                    top_moleküller,
                    orientation='h',
                    title=f'Top 15 Molekül - Satış Performansı',
                    color=top_moleküller.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    xaxis_title='Satış (USD)',
                    yaxis_title='Molekül'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                büyüme_sütunları = [col for col in df.columns if 'Büyüme' in col or 'Growth' in col]
                if büyüme_sütunları:
                    son_büyüme_sütunu = büyüme_sütunları[-1]
                    molekül_büyüme = df.groupby('Molekül')[son_büyüme_sütunu].mean().nlargest(15)
                    fig = px.bar(
                        molekül_büyüme,
                        orientation='h',
                        title='Top 15 Molekül - Büyüme Oranları',
                        color=molekül_büyüme.values,
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#f8fafc',
                        xaxis_title='Büyüme Oranı (%)',
                        yaxis_title='Molekül'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Molekül analizi için gerekli sütun bulunamadı.")

def show_price_analysis_tab(df):
    """Fiyat Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Fiyat Analizi ve Optimizasyon</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    
    st.markdown('<h3 class="subsection-title">💰 Fiyat-Hacim İlişkisi</h3>', unsafe_allow_html=True)
    price_fig = viz.create_price_volume_analysis(df)
    if price_fig:
        st.plotly_chart(price_fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Fiyat-hacim analizi için yeterli veri bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">🎯 Fiyat Segmentasyonu</h3>', unsafe_allow_html=True)
    
    fiyat_sütunları = [col for col in df.columns if 'Fiyat' in col or 'Price' in col]
    if fiyat_sütunları:
        son_fiyat_sütunu = fiyat_sütunları[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fiyat_verisi = df[son_fiyat_sütunu].dropna()
            if len(fiyat_verisi) > 0:
                fiyat_segmentleri = pd.cut(
                    fiyat_verisi,
                    bins=[0, 10, 50, 100, 500, float('inf')],
                    labels=['Ekonomi (<$10)', 'Standart ($10-$50)', 'Premium ($50-$100)', 
                           'Süper Premium ($100-$500)', 'Lüks (>$500)']
                )
                
                segment_sayıları = fiyat_segmentleri.value_counts()
                fig = px.pie(
                    values=segment_sayıları.values,
                    names=segment_sayıları.index,
                    title='Fiyat Segmentleri Dağılımı',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            büyüme_sütunları = [col for col in df.columns if 'Büyüme' in col or 'Growth' in col]
            if büyüme_sütunları and len(fiyat_verisi) > 0:
                son_büyüme_sütunu = büyüme_sütunları[-1]
                df_temp = df.copy()
                df_temp['Fiyat_Segmenti'] = pd.cut(
                    df_temp[son_fiyat_sütunu],
                    bins=[0, 10, 50, 100, 500, float('inf')],
                    labels=['Ekonomi', 'Standart', 'Premium', 'Süper Premium', 'Lüks']
                )
                
                segment_büyüme = df_temp.groupby('Fiyat_Segmenti')[son_büyüme_sütunu].mean().dropna()
                
                if len(segment_büyüme) > 0:
                    fig = px.bar(
                        segment_büyüme,
                        orientation='v',
                        title='Fiyat Segmenti Bazlı Büyüme',
                        color=segment_büyüme.values,
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#f8fafc',
                        xaxis_title='Fiyat Segmenti',
                        yaxis_title='Ortalama Büyüme (%)',
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<h3 class="subsection-title">📉 Fiyat Esnekliği Analizi</h3>', unsafe_allow_html=True)
    
    fiyat_sütunları = [col for col in df.columns if 'Fiyat' in col or 'Price' in col]
    hacim_sütunları = [col for col in df.columns if 'Units' in col or 'Adet' in col or 'Hacim' in col]
    
    if fiyat_sütunları and hacim_sütunları:
        son_fiyat_sütunu = fiyat_sütunları[-1]
        son_hacim_sütunu = hacim_sütunları[-1]
        
        korelasyon_df = df[[son_fiyat_sütunu, son_hacim_sütunu]].dropna()
        if len(korelasyon_df) > 10:
            korelasyon = korelasyon_df.corr().iloc[0, 1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fiyat-Hacim Korelasyonu", f"{korelasyon:.3f}")
            
            with col2:
                if korelasyon < -0.3:
                    esneklik_durumu = "Yüksek Esneklik"
                elif korelasyon > 0.3:
                    esneklik_durumu = "Düşük Esneklik"
                else:
                    esneklik_durumu = "Nötr"
                st.metric("Esneklik Durumu", esneklik_durumu)
            
            with col3:
                if korelasyon < -0.3:
                    öneri = "Fiyat Artışı Riskli"
                elif korelasyon > 0.3:
                    öneri = "Fiyat Artışı Mümkün"
                else:
                    öneri = "Limitli Fiyat Artışı"
                st.metric("Öneri", öneri)

def show_competition_analysis_tab(df, metrics):
    """Rekabet Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Rekabet Analizi ve Pazar Yapısı</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    
    st.markdown('<h3 class="subsection-title">🏆 Pazar Payı Analizi</h3>', unsafe_allow_html=True)
    share_fig = viz.create_market_share_analysis(df)
    if share_fig:
        st.plotly_chart(share_fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Pazar payı analizi için gerekli veri bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">📊 Rekabet Yoğunluğu Metrikleri</h3>', unsafe_allow_html=True)
    
    comp_cols = st.columns(4)
    
    with comp_cols[0]:
        hhi = metrics.get('HHI_Endeksi', 0)
        if hhi > 2500:
            hhi_durumu = "Monopolistik"
        elif hhi > 1800:
            hhi_durumu = "Oligopol"
        else:
            hhi_durumu = "Rekabetçi"
        st.metric("HHI Endeksi", f"{hhi:.0f}", hhi_durumu)
    
    with comp_cols[1]:
        top3_pay = metrics.get('Top_3_Pay', 0)
        if top3_pay > 50:
            konsantrasyon = "Yüksek"
        elif top3_pay > 30:
            konsantrasyon = "Orta"
        else:
            konsantrasyon = "Düşük"
        st.metric("Top 3 Payı", f"{top3_pay:.1f}%", konsantrasyon)
    
    with comp_cols[2]:
        cr4 = metrics.get('CR4_Oranı', 0)
        st.metric("CR4 Oranı", f"{cr4:.1f}%")
    
    with comp_cols[3]:
        top10_molekül = metrics.get('Top_10_Molekül_Payı', 0)
        st.metric("Top 10 Molekül Payı", f"{top10_molekül:.1f}%")
    
    st.markdown('<h3 class="subsection-title">📈 Şirket Performans Analizi</h3>', unsafe_allow_html=True)
    
    if 'Şirket' in df.columns:
        satış_sütunları = [col for col in df.columns if 'Satış' in col or 'Sales' in col]
        if satış_sütunları:
            son_satış_sütunu = satış_sütunları[-1]
            
            şirket_metrikleri = df.groupby('Şirket').agg({
                son_satış_sütunu: ['sum', 'mean', 'count']
            }).round(2)
            
            şirket_metrikleri.columns = ['_'.join(col).strip() for col in şirket_metrikleri.columns.values]
            şirket_metrikleri = şirket_metrikleri.sort_values(f'{son_satış_sütunu}_sum', ascending=False)
            
            top_şirketler = şirket_metrikleri.head(20)
            
            if len(top_şirketler) > 0:
                try:
                    fig = px.imshow(
                        top_şirketler.T,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Viridis',
                        title='Top 20 Şirket Performans Matrisi'
                    )
                    fig.update_layout(
                        height=600,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#f8fafc'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Heatmap oluşturulamadı. Verileri tablo olarak gösteriliyor.")
                
                with st.expander("📋 Detaylı Şirket Performans Tablosu"):
                    st.dataframe(
                        şirket_metrikleri.head(50),
                        use_container_width=True,
                        height=400
                    )

def show_international_product_tab(df, analysis_df, metrics):
    """International Ürün Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">🌍 International Ürün Analizi</h2>', unsafe_allow_html=True)
    
    if analysis_df is None:
        st.warning("International Ürün analizi için gerekli veri bulunamadı.")
        return
    
    viz = ProfessionalVisualization()
    
    st.markdown('<h3 class="subsection-title">📊 International Ürün Genel Bakış</h3>', unsafe_allow_html=True)
    
    intl_cols = st.columns(4)
    
    with intl_cols[0]:
        intl_sayı = metrics.get('International_Ürün_Sayısı', 0)
        toplam_molekül = metrics.get('Benzersiz_Moleküller', 0)
        intl_yüzde = (intl_sayı / toplam_molekül * 100) if toplam_molekül > 0 else 0
        st.metric("International Ürün Sayısı", f"{intl_sayı}", f"%{intl_yüzde:.1f}")
    
    with intl_cols[1]:
        intl_pay = metrics.get('International_Ürün_Payı', 0)
        st.metric("Pazar Payı", f"%{intl_pay:.1f}")
    
    with intl_cols[2]:
        ort_ülkeler = metrics.get('Ort_International_Ülkeler', 0)
        st.metric("Ort. Ülke Sayısı", f"{ort_ülkeler:.1f}")
    
    with intl_cols[3]:
        intl_büyüme = metrics.get('International_Ort_Büyüme', 0)
        yerel_büyüme = metrics.get('Yerel_Ort_Büyüme', 0)
        büyüme_farkı = intl_büyüme - yerel_büyüme if intl_büyüme and yerel_büyüme else 0
        st.metric("Büyüme Farkı", f"%{büyüme_farkı:.1f}")
    
    st.markdown('<h3 class="subsection-title">📈 International Ürün Analiz Grafikleri</h3>', unsafe_allow_html=True)
    
    intl_fig = viz.create_international_product_analysis(df, analysis_df)
    if intl_fig:
        st.plotly_chart(intl_fig, use_container_width=True, config={'displayModeBar': True})
    
    st.markdown('<h3 class="subsection-title">📋 International Ürün Detaylı Listesi</h3>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Tüm International Ürünler", "Top Performanslılar", "Segment Bazlı"])
    
    with tab1:
        if len(analysis_df) > 0:
            display_columns = [
                'Molekül', 'international_mı', 'toplam_satış', 'şirket_sayısı',
                'ülke_sayısı', 'ort_fiyat', 'ort_büyüme', 'international_segment'
            ]
            
            display_columns = [col for col in display_columns if col in analysis_df.columns]
            
            intl_df_display = analysis_df[display_columns].copy()
            
            def güvenli_format(value, format_type):
                try:
                    if pd.isna(value) or value is None:
                        return "N/A"
                    
                    if format_type == 'currency':
                        return f"${float(value)/1e6:,.2f}M"
                    elif format_type == 'percentage':
                        return f"{float(value):.1f}%"
                    elif format_type == 'price':
                        return f"${float(value):,.2f}"
                    else:
                        return str(value)
                except:
                    return "N/A"
            
            if 'toplam_satış' in intl_df_display.columns:
                intl_df_display['toplam_satış'] = intl_df_display['toplam_satış'].apply(
                    lambda x: güvenli_format(x, 'currency')
                )
            
            if 'ort_büyüme' in intl_df_display.columns:
                intl_df_display['ort_büyüme'] = intl_df_display['ort_büyüme'].apply(
                    lambda x: güvenli_format(x, 'percentage')
                )
            
            if 'ort_fiyat' in intl_df_display.columns:
                intl_df_display['ort_fiyat'] = intl_df_display['ort_fiyat'].apply(
                    lambda x: güvenli_format(x, 'price')
                )
            
            st.dataframe(
                intl_df_display,
                use_container_width=True,
                height=400
            )
    
    with tab2:
        if len(analysis_df) > 0:
            top_intl = analysis_df[analysis_df['international_mı']].nlargest(20, 'toplam_satış')
            
            if len(top_intl) > 0:
                top_display_columns = [
                    'Molekül', 'toplam_satış', 'şirket_sayısı', 'ülke_sayısı',
                    'ort_büyüme', 'top_şirket', 'top_ülke'
                ]
                
                top_display_columns = [col for col in top_display_columns if col in top_intl.columns]
                
                top_intl_display = top_intl[top_display_columns].copy()
                
                if 'toplam_satış' in top_intl_display.columns:
                    top_intl_display['toplam_satış'] = top_intl_display['toplam_satış'].apply(
                        lambda x: güvenli_format(x, 'currency') if not pd.isna(x) and x is not None else "N/A"
                    )
                
                if 'ort_büyüme' in top_intl_display.columns:
                    top_intl_display['ort_büyüme'] = top_intl_display['ort_büyüme'].apply(
                        lambda x: güvenli_format(x, 'percentage') if not pd.isna(x) and x is not None else "N/A"
                    )
                
                st.dataframe(
                    top_intl_display,
                    use_container_width=True,
                    height=400
                )
    
    with tab3:
        if 'international_segment' in analysis_df.columns:
            segment_analiz = analysis_df.groupby('international_segment').agg({
                'Molekül': 'count',
                'toplam_satış': 'sum',
                'ort_büyüme': 'mean',
                'şirket_sayısı': 'mean',
                'ülke_sayısı': 'mean'
            }).round(2)
            
            segment_analiz.columns = ['Molekül Sayısı', 'Toplam Satış', 'Ort Büyüme %', 'Ort Şirket', 'Ort Ülke']
            
            segment_analiz_display = segment_analiz.copy()
            
            def güvenli_format_segment(value, format_type):
                try:
                    if pd.isna(value) or value is None:
                        return "N/A"
                    
                    if format_type == 'currency':
                        return f"${float(value)/1e6:,.2f}M"
                    elif format_type == 'percentage':
                        return f"{float(value):.1f}%"
                    elif format_type == 'number':
                        return f"{float(value):.1f}"
                    else:
                        return str(value)
                except:
                    return "N/A"
            
            if 'Toplam Satış' in segment_analiz_display.columns:
                segment_analiz_display['Toplam Satış'] = segment_analiz_display['Toplam Satış'].apply(
                    lambda x: güvenli_format_segment(x, 'currency')
                )
            
            if 'Ort Büyüme %' in segment_analiz_display.columns:
                segment_analiz_display['Ort Büyüme %'] = segment_analiz_display['Ort Büyüme %'].apply(
                    lambda x: güvenli_format_segment(x, 'percentage')
                )
            
            if 'Ort Şirket' in segment_analiz_display.columns:
                segment_analiz_display['Ort Şirket'] = segment_analiz_display['Ort Şirket'].apply(
                    lambda x: güvenli_format_segment(x, 'number')
                )
            
            if 'Ort Ülke' in segment_analiz_display.columns:
                segment_analiz_display['Ort Ülke'] = segment_analiz_display['Ort Ülke'].apply(
                    lambda x: güvenli_format_segment(x, 'number')
                )
            
            st.dataframe(
                segment_analiz_display,
                use_container_width=True
            )
    
    st.markdown('<h3 class="subsection-title">💡 International Ürün İçgörüleri</h3>', unsafe_allow_html=True)
    
    içgörüler = AdvancedPharmaAnalytics.get_international_product_insights(df)
    
    if içgörüler:
        for insight in içgörüler:
            icon = "🌍"
            if insight['type'] == 'warning':
                icon = "⚠️"
            elif insight['type'] == 'success':
                icon = "✅"
            elif insight['type'] == 'info':
                icon = "ℹ️"
            
            st.markdown(f"""
            <div class="insight-card {insight['type']}">
                <div class="insight-icon">{icon}</div>
                <div class="insight-title">{insight['title']}</div>
                <div class="insight-content">{insight['description']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if insight.get('data') is not None and not insight['data'].empty:
                with st.expander("📋 Detaylı Liste"):
                    display_columns = []
                    for col in ['Molekül', 'toplam_satış', 'şirket_sayısı', 'ülke_sayısı', 'ort_büyüme']:
                        if col in insight['data'].columns:
                            display_columns.append(col)
                    
                    if display_columns:
                        data_display = insight['data'][display_columns].copy()
                        
                        def güvenli_format_insight(value, format_type):
                            try:
                                if pd.isna(value) or value is None:
                                    return "N/A"
                                
                                if format_type == 'currency':
                                    return f"${float(value)/1e6:,.2f}M"
                                elif format_type == 'percentage':
                                    return f"{float(value):.1f}%"
                                else:
                                    return str(value)
                            except:
                                return "N/A"
                        
                        if 'toplam_satış' in data_display.columns:
                            data_display['toplam_satış'] = data_display['toplam_satış'].apply(
                                lambda x: güvenli_format_insight(x, 'currency')
                            )
                        if 'ort_büyüme' in data_display.columns:
                            data_display['ort_büyüme'] = data_display['ort_büyüme'].apply(
                                lambda x: güvenli_format_insight(x, 'percentage')
                            )
                        
                        st.dataframe(
                            data_display,
                            use_container_width=True
                        )
    
    st.markdown('<h3 class="subsection-title">🎯 Strateji Önerileri</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-card info">
            <div class="insight-title">🚀 International Ürün Büyüme Stratejisi</div>
            <div class="insight-content">
            1. Yüksek büyüme gösteren International Ürünleri belirleyin<br>
            2. Bu ürünlerin diğer ülkelere yayılma potansiyelini değerlendirin<br>
            3. Yerel pazarlarda lider olan ürünleri International Ürüne dönüştürün
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card success">
            <div class="insight-title">💰 International Ürün Fiyatlandırma</div>
            <div class="insight-content">
            1. Ülke bazında fiyatlandırma stratejileri geliştirin<br>
            2. Premium segmentteki International Ürünlerin fiyatını optimize edin<br>
            3. Fiyat esnekliği düşük ürünlere odaklanın
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_strategic_analysis_tab(df, insights):
    """Stratejik Analiz tab'ını göster"""
    st.markdown('<h2 class="section-title">Stratejik Analiz ve Öngörüler</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">🎯 Pazar Segmentasyonu</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_clusters = st.slider("Segment Sayısı", 2, 8, 4, key="n_clusters")
        method = st.selectbox("Segmentasyon Metodu", ['kmeans', 'dbscan'], key="seg_method")
        
        if st.button("🔍 Segmentasyon Analizi Yap", type="primary", width='stretch', key="run_segmentation"):
            with st.spinner("Pazar segmentasyonu analiz ediliyor..."):
                analytics = AdvancedPharmaAnalytics()
                segmentation_results = analytics.perform_advanced_segmentation(df, n_clusters, method)
                
                if segmentation_results:
                    st.session_state.segmentation_results = segmentation_results
                    st.success(f"{segmentation_results['metrics']['n_clusters']} segment tespit edildi!")
                    st.rerun()
    
    with col2:
        if 'segmentation_results' in st.session_state:
            results = st.session_state.segmentation_results
            
            if 'Segment_İsmi' in results['data'].columns:
                segment_counts = results['data']['Segment_İsmi'].value_counts()
                
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
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
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if results['metrics']['inertia']:
                        st.metric("Inertia", f"{results['metrics']['inertia']:,.0f}")
                with col_b:
                    if results['metrics']['silhouette_score']:
                        st.metric("Silhouette Skoru", f"{results['metrics']['silhouette_score']:.3f}")
                with col_c:
                    if results['metrics']['calinski_score']:
                        st.metric("Calinski Skoru", f"{results['metrics']['calinski_score']:,.0f}")
    
    st.markdown('<h3 class="subsection-title">🚀 Büyüme Fırsatları</h3>', unsafe_allow_html=True)
    
    if insights:
        fırsat_içgörüleri = [i for i in insights if i['type'] in ['success', 'info']]
        
        if fırsat_içgörüleri:
            for insight in fırsat_içgörüleri[:3]:
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-content">{insight['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if insight.get('data') is not None and not insight['data'].empty:
                    with st.expander("🚀 Bu Fırsattaki Ürünler"):
                        display_columns = []
                        for col in ['Molekül', 'Şirket', 'Ülke', 'Satış_2024', 'Büyüme_23_24']:
                            if col in insight['data'].columns:
                                display_columns.append(col)
                        
                        if display_columns:
                            st.dataframe(
                                insight['data'][display_columns],
                                use_container_width=True
                            )
        else:
            st.info("Henüz büyüme fırsatı tespit edilmedi.")
    
    st.markdown('<h3 class="subsection-title">⚠️ Risk Analizi</h3>', unsafe_allow_html=True)
    
    risk_içgörüleri = [i for i in insights if i['type'] in ['warning', 'danger']]
    
    if risk_içgörüleri:
        for insight in risk_içgörüleri[:3]:
            st.markdown(f"""
            <div class="insight-card {insight['type']}">
                <div class="insight-title">{insight['title']}</div>
                <div class="insight-content">{insight['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Önemli risk tespit edilmedi.")

def show_reporting_tab(df, metrics, insights, analysis_df):
    """Raporlama tab'ını göster"""
    st.markdown('<h2 class="section-title">Raporlama ve İndirme</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">📊 Rapor Türleri</h3>', unsafe_allow_html=True)
    
    report_type = st.radio(
        "Rapor Türü Seçin",
        ['Excel Detaylı Rapor', 'PDF Özet Rapor', 'JSON Veri Paketi', 'CSV Ham Veri'],
        horizontal=True,
        key="report_type"
    )
    
    st.markdown('<h3 class="subsection-title">🛠️ Rapor Oluşturma</h3>', unsafe_allow_html=True)
    
    report_cols = st.columns(3)
    
    with report_cols[0]:
        if st.button("📈 Excel Raporu Oluştur", width='stretch', key="excel_report"):
            with st.spinner("Excel raporu oluşturuluyor..."):
                reporting = ProfessionalReporting()
                excel_report = reporting.generate_excel_report(df, metrics, insights, analysis_df)
                
                if excel_report:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="⬇️ Excel İndir",
                        data=excel_report,
                        file_name=f"pharma_report_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch',
                        key="download_excel"
                    )
                else:
                    st.error("Excel raporu oluşturulamadı.")
    
    with report_cols[1]:
        if st.button("🔄 Analizi Sıfırla", width='stretch', key="reset_analysis"):
            st.session_state.df = None
            st.session_state.filtered_df = None
            st.session_state.metrics = None
            st.session_state.insights = []
            st.session_state.mevcut_filtreler = {}
            if 'segmentation_results' in st.session_state:
                del st.session_state.segmentation_results
            if 'international_analiz' in st.session_state:
                del st.session_state.international_analiz
            st.rerun()
    
    with report_cols[2]:
        if st.button("💾 International CSV", width='stretch', key="intl_csv"):
            if analysis_df is not None:
                csv = analysis_df.to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="⬇️ CSV İndir",
                    data=csv,
                    file_name=f"international_ürünler_{timestamp}.csv",
                    mime="text/csv",
                    width='stretch',
                    key="download_intl_csv"
                )
            else:
                st.warning("International Ürün analizi bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)
    
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with stat_cols[1]:
        st.metric("Toplam Sütun", len(df.columns))
    
    with stat_cols[2]:
        bellek_kullanımı = df.memory_usage(deep=True).sum()/1024**2
        st.metric("Bellek Kullanımı", f"{bellek_kullanımı:.1f} MB")
    
    with stat_cols[3]:
        intl_sayı = metrics.get('International_Ürün_Sayısı', 0)
        st.metric("International Ürün", intl_sayı)

# ================================================
# 8. UYGULAMA BAŞLATMA
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
