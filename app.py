# app.py - Profesyonel İlaç Pazarı Dashboard (INTERNATIONAL PRODUCT ANALİZİ EKLENDİ)
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
        'About': "### PharmaIntelligence Pro v3.2\nInternational Product Analytics Eklendi"
    }
)

# PROFESYONEL DARK THEME CSS STYLES
PROFESSIONAL_CSS = """
<style>
    /* === ROOT VARIABLES === */
    :root {
        --primary-dark: #0f172a;
        --secondary-dark: #1e293b;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-green: #10b981;
        --accent-yellow: #f59e0b;
        --accent-red: #ef4444;
        --accent-cyan: #06b6d4;
        
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #334155;
        --bg-hover: #475569;
        --bg-surface: #1e293b;
        
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
        
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
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple), var(--accent-cyan));
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
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), transparent);
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
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    }
    
    .custom-metric-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-xl);
        border-color: var(--accent-blue);
    }
    
    .custom-metric-card.premium {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
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
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--accent-green);
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
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
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
        background: rgba(59, 130, 246, 0.2);
        color: var(--accent-blue);
        border: 1px solid rgba(59, 130, 246, 0.3);
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
    .feature-card-purple { border-left-color: var(--accent-purple); }
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
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* === GET STARTED BOX === */
    .get-started-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1));
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        border: 1px solid rgba(59, 130, 246, 0.3);
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
# 2. OPTİMİZE VERİ İŞLEME SİSTEMİ - TÜM VERİ İÇİN DÜZENLENDİ
# ================================================

class OptimizedDataProcessor:
    """Optimize edilmiş veri işleme sınıfı - TÜM VERİ için"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
    def load_large_dataset(file):
        """TÜM veri setlerini optimize şekilde yükle - örneklem yok"""
        try:
            start_time = time.time()
            st.info(f"📥 Veri yükleniyor: {file.name}")
            
            # Excel dosyası mı kontrol et
            if file.name.endswith(('.xlsx', '.xls')):
                # Excel dosyası için
                with st.spinner("Excel dosyası işleniyor..."):
                    # Tüm veriyi yükle
                    df = pd.read_excel(file, engine='openpyxl')
                    
                    # Bellek optimizasyonu
                    original_memory = df.memory_usage(deep=True).sum() / 1024**2
                    df = OptimizedDataProcessor.optimize_dataframe(df)
                    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
                    
                    load_time = time.time() - start_time
                    
                    # Bilgilendirme mesajı
                    st.success(f"""
                    ✅ Veri yükleme tamamlandı:
                    - **{len(df):,}** satır
                    - **{len(df.columns)}** sütun
                    - Bellek: {original_memory:.1f}MB → {optimized_memory:.1f}MB
                    - Süre: {load_time:.2f}s
                    """)
                    
            elif file.name.endswith('.csv'):
                # CSV dosyası için
                with st.spinner("CSV dosyası işleniyor..."):
                    # Tüm veriyi yükle
                    df = pd.read_csv(file)
                    
                    # Bellek optimizasyonu
                    original_memory = df.memory_usage(deep=True).sum() / 1024**2
                    df = OptimizedDataProcessor.optimize_dataframe(df)
                    optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
                    
                    load_time = time.time() - start_time
                    
                    # Bilgilendirme mesajı
                    st.success(f"""
                    ✅ Veri yükleme tamamlandı:
                    - **{len(df):,}** satır
                    - **{len(df.columns)}** sütun
                    - Bellek: {original_memory:.1f}MB → {optimized_memory:.1f}MB
                    - Süre: {load_time:.2f}s
                    """)
            else:
                st.error("Desteklenmeyen dosya formatı. Lütfen Excel (.xlsx, .xls) veya CSV (.csv) dosyası yükleyin.")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.error(f"Detay: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def optimize_dataframe(df):
        """DataFrame'i optimize et"""
        try:
            # Sütun isimlerini temizle
            df.columns = OptimizedDataProcessor.clean_column_names(df.columns)
            
            # String sütunları için kategoriye çevir
            for col in df.select_dtypes(include=['object']).columns:
                num_unique = df[col].nunique()
                total_rows = len(df)
                if num_unique / total_rows < 0.5:  # Eğer unique değerler toplamın %50'sinden azsa
                    df[col] = df[col].astype('category')
            
            # Sayısal sütunları optimize et
            for col in df.select_dtypes(include=[np.number]).columns:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if pd.api.types.is_integer_dtype(df[col]):
                    if col_min >= 0:
                        if col_max < 255:
                            df[col] = df[col].astype(np.uint8)
                        elif col_max < 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif col_max < 4294967295:
                            df[col] = df[col].astype(np.uint32)
                        else:
                            df[col] = df[col].astype(np.uint64)
                    else:
                        if col_min > -128 and col_max < 127:
                            df[col] = df[col].astype(np.int8)
                        elif col_min > -32768 and col_max < 32767:
                            df[col] = df[col].astype(np.int16)
                        elif col_min > -2147483648 and col_max < 2147483647:
                            df[col] = df[col].astype(np.int32)
                        else:
                            df[col] = df[col].astype(np.int64)
                else:
                    # Float değerleri
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
            
            # Tarih sütunlarını bul ve çevir
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
            
            # String sütunları temizle
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()
            
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
                # Türkçe karakterleri İngilizce'ye çevir
                replacements = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in replacements.items():
                    col = col.replace(tr, en)
                
                # Özel karakterleri kaldır
                col = col.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                col = ' '.join(col.split())  # Fazla boşlukları kaldır
                
                # Özel isimlendirmeleri düzelt
                if 'USD' in col and 'MNF' in col and 'MAT' in col:
                    if '2022' in col:
                        if 'Units' in col:
                            col = 'Units_2022'
                        elif 'Avg Price' in col:
                            col = 'Avg_Price_2022'
                        else:
                            col = 'Sales_2022'
                    elif '2023' in col:
                        if 'Units' in col:
                            col = 'Units_2023'
                        elif 'Avg Price' in col:
                            col = 'Avg_Price_2023'
                        else:
                            col = 'Sales_2023'
                    elif '2024' in col:
                        if 'Units' in col:
                            col = 'Units_2024'
                        elif 'Avg Price' in col:
                            col = 'Avg_Price_2024'
                        else:
                            col = 'Sales_2024'
                
                col = col.strip()
            
            cleaned.append(str(col).strip())
        
        return cleaned
    
    @staticmethod
    def prepare_analytics_data(df):
        """Analiz için veriyi hazırla"""
        try:
            # Satış sütunlarını bul
            sales_cols = {}
            for col in df.columns:
                if 'Sales_' in col:
                    year = col.split('_')[-1]
                    sales_cols[year] = col
            
            # Büyüme oranlarını hesapla
            years = sorted([int(y) for y in sales_cols.keys() if y.isdigit()])
            
            for i in range(1, len(years)):
                prev_year = str(years[i-1])
                curr_year = str(years[i])
                
                if prev_year in sales_cols and curr_year in sales_cols:
                    prev_col = sales_cols[prev_year]
                    curr_col = sales_cols[curr_year]
                    
                    # Büyüme oranını hesapla
                    df[f'Growth_{prev_year}_{curr_year}'] = ((df[curr_col] - df[prev_col]) / 
                                                             df[prev_col].replace(0, np.nan)) * 100
            
            # Fiyat analizi
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if price_cols:
                df['Avg_Price_Overall'] = df[price_cols].mean(axis=1, skipna=True)
            
            # CAGR hesapla
            if len(years) >= 2:
                first_year = str(years[0])
                last_year = str(years[-1])
                if first_year in sales_cols and last_year in sales_cols:
                    df['CAGR'] = ((df[sales_cols[last_year]] / df[sales_cols[first_year]].replace(0, np.nan)) ** 
                                 (1/len(years)) - 1) * 100
            
            # Pazar payı hesapla
            if years and str(years[-1]) in sales_cols:
                last_sales_col = sales_cols[str(years[-1])]
                total_sales = df[last_sales_col].sum()
                if total_sales > 0:
                    df['Market_Share'] = (df[last_sales_col] / total_sales) * 100
            
            # Fiyat-hacim oranı
            if 'Avg_Price_2024' in df.columns and 'Units_2024' in df.columns:
                df['Price_Volume_Ratio'] = df['Avg_Price_2024'] * df['Units_2024']
            
            return df
            
        except Exception as e:
            st.warning(f"Analiz verisi hazırlama hatası: {str(e)}")
            return df

# ================================================
# 3. GELİŞMİŞ FİLTRELEME SİSTEMİ
# ================================================

class AdvancedFilterSystem:
    """Gelişmiş filtreleme sistemi - Tümü seçeneği dahil"""
    
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
            
            # Ülke filtresi
            if 'Country' in available_columns:
                countries = sorted(df['Country'].dropna().unique())
                selected_countries = AdvancedFilterSystem.create_searchable_multiselect_with_all(
                    "🌍 Ülkeler",
                    countries,
                    key="countries_filter"
                )
                if selected_countries and "Tümü" not in selected_countries:
                    filter_config['Country'] = selected_countries
            
            # Şirket filtresi
            if 'Corporation' in available_columns:
                companies = sorted(df['Corporation'].dropna().unique())
                selected_companies = AdvancedFilterSystem.create_searchable_multiselect_with_all(
                    "🏢 Şirketler",
                    companies,
                    key="companies_filter"
                )
                if selected_companies and "Tümü" not in selected_companies:
                    filter_config['Corporation'] = selected_companies
            
            # Molekül filtresi
            if 'Molecule' in available_columns:
                molecules = sorted(df['Molecule'].dropna().unique())
                selected_molecules = AdvancedFilterSystem.create_searchable_multiselect_with_all(
                    "🧪 Moleküller",
                    molecules,
                    key="molecules_filter"
                )
                if selected_molecules and "Tümü" not in selected_molecules:
                    filter_config['Molecule'] = selected_molecules
            
            st.markdown("---")
            st.markdown('<div class="filter-title">📊 Numerik Filtreler</div>', unsafe_allow_html=True)
            
            # Satış filtresi
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                latest_sales_col = sales_cols[-1]
                min_sales = float(df[latest_sales_col].min())
                max_sales = float(df[latest_sales_col].max())
                
                sales_range = st.slider(
                    f"Satış Aralığı ({latest_sales_col})",
                    min_value=float(min_sales),
                    max_value=float(max_sales),
                    value=(float(min_sales), float(max_sales)),
                    step=(max_sales - min_sales) / 1000,
                    help="Satış aralığını seçin"
                )
                filter_config['sales_range'] = (sales_range, latest_sales_col)
            
            # Büyüme filtresi
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                latest_growth_col = growth_cols[-1]
                min_growth = float(df[latest_growth_col].min())
                max_growth = float(df[latest_growth_col].max())
                
                growth_range = st.slider(
                    f"Büyüme Oranı (%)",
                    min_value=float(min_growth),
                    max_value=float(max_growth),
                    value=(float(min(min_growth, -50.0)), float(max(max_growth, 150.0))),
                    step=5.0,
                    help="Büyüme oranı aralığını seçin"
                )
                filter_config['growth_range'] = (growth_range, latest_growth_col)
            
            st.markdown("---")
            st.markdown('<div class="filter-title">⚙️ Ek Filtreler</div>', unsafe_allow_html=True)
            
            # Ek filtreler
            if growth_cols:
                only_positive_growth = st.checkbox("📈 Sadece Pozitif Büyüyen Ürünler", value=False)
                if only_positive_growth:
                    filter_config['positive_growth'] = True
            
            if sales_cols:
                high_sales = st.checkbox("💰 Yüksek Satışlı Ürünler (Top 20%)", value=False)
                if high_sales:
                    latest_sales_col = sales_cols[-1]
                    threshold = df[latest_sales_col].quantile(0.8)  # Top 20%
                    filter_config['high_sales'] = (threshold, latest_sales_col)
            
            # Filtre butonları
            col1, col2 = st.columns(2)
            with col1:
                apply_filter = st.button("✅ Filtre Uygula", type="primary", use_container_width=True)
            with col2:
                clear_filter = st.button("🗑️ Temizle", use_container_width=True)
            
            # Filtre kaydetme
            with st.expander("💾 Filtre Yönetimi", expanded=False):
                if st.button("Filtreyi Kaydet", use_container_width=True):
                    filter_name = st.text_input("Filtre Adı", placeholder="Örn: Yüksek Büyüyen Ürünler")
                    if filter_name and filter_config:
                        if 'saved_filters' not in st.session_state:
                            st.session_state.saved_filters = {}
                        st.session_state.saved_filters[filter_name] = filter_config
                        st.success(f"✅ '{filter_name}' filtresi kaydedildi!")
                
                if 'saved_filters' in st.session_state and st.session_state.saved_filters:
                    saved_filter = st.selectbox(
                        "Kayıtlı Filtreler",
                        options=[""] + list(st.session_state.saved_filters.keys()),
                        key="saved_filters_select"
                    )
                    
                    if saved_filter and st.button("Seçili Filtreyi Yükle", use_container_width=True):
                        st.session_state.current_filters = st.session_state.saved_filters[saved_filter]
                        st.success(f"✅ '{saved_filter}' filtresi yüklendi!")
                        st.rerun()
            
            return search_term, filter_config, apply_filter, clear_filter
    
    @staticmethod
    def create_searchable_multiselect_with_all(label, options, key):
        """Arama yapılabilir multiselect - Tümü seçeneği dahil"""
        if not options:
            return []
        
        all_options = ["Tümü"] + options
        
        # Arama kutusu
        search_query = st.text_input(f"{label} Ara", key=f"{key}_search", placeholder="Arama yapın...")
        
        if search_query:
            filtered_options = ["Tümü"] + [opt for opt in options if search_query.lower() in str(opt).lower()]
        else:
            filtered_options = all_options
        
        # Multi-select
        selected = st.multiselect(
            label,
            options=filtered_options,
            default=["Tümü"],
            key=key,
            help="'Tümü' seçildiğinde diğer tüm seçenekler otomatik seçilir"
        )
        
        # "Tümü" mantığı
        if "Tümü" in selected and len(selected) > 1:
            # "Tümü" ve başka seçenekler seçilmişse, "Tümü"yü kaldır
            selected = [opt for opt in selected if opt != "Tümü"]
        elif "Tümü" in selected and len(selected) == 1:
            # Sadece "Tümü" seçilmişse, tümünü seç
            selected = options
        
        # Seçim bilgisi
        if selected:
            if len(selected) == len(options):
                st.caption(f"✅ TÜMÜ seçildi ({len(options)} öğe)")
            else:
                st.caption(f"✅ {len(selected)} / {len(options)} seçildi")
        
        return selected
    
    @staticmethod
    def apply_filters(df, search_term, filter_config):
        """Filtreleri uygula"""
        filtered_df = df.copy()
        
        # Global arama
        if search_term:
            search_mask = pd.Series(False, index=filtered_df.index)
            for col in filtered_df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(filtered_df[col]):
                        search_mask = search_mask | filtered_df[col].astype(str).str.contains(
                            search_term, case=False, na=False
                        )
                    else:
                        search_mask = search_mask | filtered_df[col].astype(str).str.contains(
                            search_term, case=False, na=False
                        )
                except:
                    continue
            filtered_df = filtered_df[search_mask]
            if len(filtered_df) == 0:
                st.warning("Arama sonucu bulunamadı!")
        
        # Kategorik filtreler
        for column, values in filter_config.items():
            if column in ['Country', 'Corporation', 'Molecule'] and column in filtered_df.columns:
                if values:
                    filtered_df = filtered_df[filtered_df[column].isin(values)]
        
        # Satış aralığı filtresi
        if 'sales_range' in filter_config:
            (min_val, max_val), col_name = filter_config['sales_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        # Büyüme aralığı filtresi
        if 'growth_range' in filter_config:
            (min_val, max_val), col_name = filter_config['growth_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        # Pozitif büyüme filtresi
        if 'positive_growth' in filter_config:
            growth_cols = [col for col in filtered_df.columns if 'Growth_' in col]
            if growth_cols:
                filtered_df = filtered_df[filtered_df[growth_cols[-1]] > 0]
        
        # Yüksek satış filtresi
        if 'high_sales' in filter_config:
            threshold, col_name = filter_config['high_sales']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col_name] >= threshold]
        
        return filtered_df
    
    @staticmethod
    def show_filter_status(current_filters, filtered_df, original_df):
        """Filtre durumunu göster"""
        if current_filters:
            filter_info = f"🎯 **Aktif Filtreler:** "
            filter_items = []
            
            for key, value in current_filters.items():
                if key in ['Country', 'Corporation', 'Molecule']:
                    if isinstance(value, list):
                        if len(value) > 3:
                            filter_items.append(f"{key}: {len(value)} seçenek")
                        else:
                            filter_items.append(f"{key}: {', '.join(value[:3])}")
                elif key == 'sales_range':
                    (min_val, max_val), col_name = value
                    filter_items.append(f"Satış: ${min_val:,.0f}-${max_val:,.0f}")
                elif key == 'growth_range':
                    (min_val, max_val), col_name = value
                    filter_items.append(f"Büyüme: %{min_val:.1f}-%{max_val:.1f}")
                elif key == 'positive_growth':
                    filter_items.append("Pozitif Büyüme")
                elif key == 'high_sales':
                    threshold, col_name = value
                    filter_items.append(f"Yüksek Satış (Top 20%)")
            
            filter_info += " | ".join(filter_items)
            filter_info += f" | **Gösterilen:** {len(filtered_df):,} / {len(original_df):,} satır"
            
            st.markdown(f'<div class="filter-status">{filter_info}</div>', unsafe_allow_html=True)
            
            if st.button("❌ Tüm Filtreleri Temizle", key="clear_all_filters"):
                st.session_state.filtered_df = st.session_state.df.copy()
                st.session_state.current_filters = {}
                st.rerun()

# ================================================
# 4. GELİŞMİŞ ANALİTİK MOTORU (INTERNATIONAL PRODUCT EKLENDİ)
# ================================================

class AdvancedPharmaAnalytics:
    """Gelişmiş farma analitik motoru"""
    
    @staticmethod
    def calculate_comprehensive_metrics(df):
        """Kapsamlı pazar metrikleri"""
        metrics = {}
        
        try:
            # Temel metrikler
            metrics['Total_Rows'] = len(df)
            metrics['Total_Columns'] = len(df.columns)
            
            # Satış metrikleri
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                latest_sales_col = sales_cols[-1]
                metrics['Latest_Sales_Year'] = latest_sales_col.split('_')[-1]
                metrics['Total_Market_Value'] = df[latest_sales_col].sum()
                metrics['Avg_Sales_Per_Product'] = df[latest_sales_col].mean()
                metrics['Median_Sales'] = df[latest_sales_col].median()
                metrics['Sales_Std_Dev'] = df[latest_sales_col].std()
                
                # Çeyreklikler
                metrics['Sales_Q1'] = df[latest_sales_col].quantile(0.25)
                metrics['Sales_Q3'] = df[latest_sales_col].quantile(0.75)
                metrics['Sales_IQR'] = metrics['Sales_Q3'] - metrics['Sales_Q1']
            
            # Büyüme metrikleri
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                latest_growth_col = growth_cols[-1]
                metrics['Avg_Growth_Rate'] = df[latest_growth_col].mean()
                metrics['Growth_Std_Dev'] = df[latest_growth_col].std()
                metrics['Positive_Growth_Products'] = (df[latest_growth_col] > 0).sum()
                metrics['Negative_Growth_Products'] = (df[latest_growth_col] < 0).sum()
                metrics['High_Growth_Products'] = (df[latest_growth_col] > 20).sum()
            
            # Şirket bazlı metrikler
            if 'Corporation' in df.columns and sales_cols:
                latest_sales_col = sales_cols[-1]
                corp_sales = df.groupby('Corporation')[latest_sales_col].sum().sort_values(ascending=False)
                total_sales = corp_sales.sum()
                
                if total_sales > 0:
                    market_shares = (corp_sales / total_sales * 100)
                    metrics['HHI_Index'] = (market_shares ** 2).sum() / 10000
                    
                    # Top şirket payları
                    top_n = [1, 3, 5, 10]
                    for n in top_n:
                        metrics[f'Top_{n}_Share'] = corp_sales.nlargest(n).sum() / total_sales * 100
                    
                    metrics['CR4_Ratio'] = metrics.get('Top_4_Share', 0)
            
            # Molekül metrikleri
            if 'Molecule' in df.columns:
                metrics['Unique_Molecules'] = df['Molecule'].nunique()
                if sales_cols:
                    mol_sales = df.groupby('Molecule')[latest_sales_col].sum()
                    total_mol_sales = mol_sales.sum()
                    if total_mol_sales > 0:
                        metrics['Top_10_Molecule_Share'] = mol_sales.nlargest(10).sum() / total_mol_sales * 100
            
            # Ülke metrikleri
            if 'Country' in df.columns:
                metrics['Country_Coverage'] = df['Country'].nunique()
                if sales_cols:
                    country_sales = df.groupby('Country')[latest_sales_col].sum()
                    metrics['Top_5_Country_Share'] = country_sales.nlargest(5).sum() / country_sales.sum() * 100
            
            # Fiyat metrikleri
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if price_cols:
                latest_price_col = price_cols[-1]
                metrics['Avg_Price'] = df[latest_price_col].mean()
                metrics['Price_Variance'] = df[latest_price_col].var()
                metrics['Price_CV'] = (df[latest_price_col].std() / df[latest_price_col].mean()) * 100 if df[latest_price_col].mean() > 0 else 0
                
                price_quartiles = df[latest_price_col].quantile([0.25, 0.5, 0.75])
                metrics['Price_Q1'] = price_quartiles[0.25]
                metrics['Price_Median'] = price_quartiles[0.5]
                metrics['Price_Q3'] = price_quartiles[0.75]
            
            # International Product metrikleri
            if 'Molecule' in df.columns and sales_cols:
                metrics = AdvancedPharmaAnalytics.add_international_product_metrics(df, metrics, sales_cols)
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return {}
    
    @staticmethod
    def add_international_product_metrics(df, metrics, sales_cols):
        """International Product analiz metriklerini ekle"""
        try:
            latest_sales_col = sales_cols[-1]
            
            # International Product analizi
            intl_analysis = AdvancedPharmaAnalytics.analyze_international_products(df)
            
            if intl_analysis is not None and len(intl_analysis) > 0:
                # International Product sayısı
                intl_count = intl_analysis['is_international'].sum()
                metrics['International_Product_Count'] = intl_count
                
                # International Product satışları
                intl_sales = intl_analysis[intl_analysis['is_international']]['total_sales'].sum()
                metrics['International_Product_Sales'] = intl_sales
                
                # Pazar payı
                total_sales = metrics.get('Total_Market_Value', 0)
                if total_sales > 0:
                    metrics['International_Product_Share'] = (intl_sales / total_sales) * 100
                
                # Ortalama özellikler
                intl_df = intl_analysis[intl_analysis['is_international']]
                if len(intl_df) > 0:
                    metrics['Avg_International_Corporations'] = intl_df['corporation_count'].mean()
                    metrics['Avg_International_Countries'] = intl_df['country_count'].mean()
                    
                    # Büyüme karşılaştırması
                    if 'avg_growth' in intl_df.columns:
                        intl_growth = intl_df['avg_growth'].mean()
                        local_growth = intl_analysis[~intl_analysis['is_international']]['avg_growth'].mean()
                        
                        if not pd.isna(intl_growth) and not pd.isna(local_growth):
                            metrics['International_Avg_Growth'] = intl_growth
                            metrics['Local_Avg_Growth'] = local_growth
                            metrics['International_Growth_Premium'] = intl_growth - local_growth
                
                # Top International Products
                top_intl = intl_df.nlargest(10, 'total_sales')
                if len(top_intl) > 0:
                    metrics['Top_10_International_Sales'] = top_intl['total_sales'].sum()
                    if intl_sales > 0:
                        metrics['Top_10_International_Share'] = (metrics['Top_10_International_Sales'] / intl_sales) * 100
            
            return metrics
            
        except Exception as e:
            st.warning(f"International Product metrik hatası: {str(e)}")
            return metrics
    
    @staticmethod
    def analyze_international_products(df):
        """International Product detaylı analizi"""
        try:
            if 'Molecule' not in df.columns:
                return None
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if not sales_cols:
                return None
            
            latest_sales_col = sales_cols[-1]
            
            # International Product analizi
            international_analysis = []
            
            for molecule in df['Molecule'].unique():
                molecule_df = df[df['Molecule'] == molecule]
                
                # Benzersiz şirket ve ülke sayıları
                unique_corporations = molecule_df['Corporation'].nunique() if 'Corporation' in molecule_df.columns else 0
                unique_countries = molecule_df['Country'].nunique() if 'Country' in molecule_df.columns else 0
                
                # International Product kriteri
                is_international = (unique_corporations > 1 or unique_countries > 1)
                
                # Temel metrikler
                total_sales = molecule_df[latest_sales_col].sum()
                
                # International Product için detaylı analiz
                if is_international:
                    analysis_entry = {
                        'Molecule': molecule,
                        'is_international': is_international,
                        'total_sales': total_sales,
                        'corporation_count': unique_corporations,
                        'country_count': unique_countries,
                        'product_count': len(molecule_df),
                    }
                    
                    # Ek metrikler
                    price_cols = [col for col in molecule_df.columns if 'Avg_Price' in col]
                    if price_cols:
                        analysis_entry['avg_price'] = molecule_df[price_cols[-1]].mean()
                    
                    growth_cols = [col for col in molecule_df.columns if 'Growth_' in col]
                    if growth_cols:
                        analysis_entry['avg_growth'] = molecule_df[growth_cols[-1]].mean()
                    
                    # Lider şirket ve ülke
                    if 'Corporation' in molecule_df.columns:
                        corp_sales = molecule_df.groupby('Corporation')[latest_sales_col].sum()
                        if len(corp_sales) > 0:
                            analysis_entry['top_corporation'] = corp_sales.idxmax()
                            analysis_entry['corp_market_share'] = (corp_sales.max() / total_sales * 100) if total_sales > 0 else 0
                    
                    if 'Country' in molecule_df.columns:
                        country_sales = molecule_df.groupby('Country')[latest_sales_col].sum()
                        if len(country_sales) > 0:
                            analysis_entry['top_country'] = country_sales.idxmax()
                            analysis_entry['country_market_share'] = (country_sales.max() / total_sales * 100) if total_sales > 0 else 0
                    
                    # Karmaşıklık skoru
                    complexity_score = (unique_corporations * 0.6 + unique_countries * 0.4) / 2
                    analysis_entry['complexity_score'] = complexity_score
                    
                    # Satış konsantrasyonu
                    corp_share = analysis_entry.get('corp_market_share', 0)
                    country_share = analysis_entry.get('country_market_share', 0)
                    analysis_entry['sales_concentration'] = max(corp_share, country_share)
                    
                    international_analysis.append(analysis_entry)
                else:
                    # Local ürünler için basit kayıt
                    international_analysis.append({
                        'Molecule': molecule,
                        'is_international': False,
                        'total_sales': total_sales,
                        'corporation_count': unique_corporations,
                        'country_count': unique_countries,
                        'product_count': len(molecule_df),
                        'avg_price': molecule_df[price_cols[-1]].mean() if price_cols else None,
                        'avg_growth': molecule_df[growth_cols[-1]].mean() if growth_cols else None,
                    })
            
            analysis_df = pd.DataFrame(international_analysis)
            
            # Segmentasyon
            if len(analysis_df) > 0 and 'complexity_score' in analysis_df.columns:
                analysis_df['international_segment'] = 'Local'
                
                # Sadece International Product'ları segmentlere ayır
                intl_mask = analysis_df['is_international']
                if intl_mask.any():
                    intl_df = analysis_df[intl_mask]
                    
                    # Karmaşıklık skoruna göre segmentasyon
                    conditions = [
                        intl_df['complexity_score'] <= 1.5,
                        intl_df['complexity_score'] <= 2.5,
                        intl_df['complexity_score'] > 2.5
                    ]
                    choices = ['Regional', 'Multi-National', 'Global']
                    
                    analysis_df.loc[intl_mask, 'international_segment'] = np.select(conditions, choices, default='Local')
            
            return analysis_df.sort_values('total_sales', ascending=False)
            
        except Exception as e:
            st.warning(f"International Product analiz hatası: {str(e)}")
            return None
    
    @staticmethod
    def get_international_product_insights(df):
        """International Product içgörüleri"""
        insights = []
        
        try:
            analysis_df = AdvancedPharmaAnalytics.analyze_international_products(df)
            
            if analysis_df is None or len(analysis_df) == 0:
                return insights
            
            # International Product sayısı
            intl_count = analysis_df['is_international'].sum()
            total_molecules = len(analysis_df)
            
            if total_molecules > 0:
                intl_percentage = (intl_count / total_molecules * 100)
                
                insights.append({
                    'type': 'info',
                    'title': f'🌍 International Product Dağılımı',
                    'description': f"Toplam {total_molecules} molekülden {intl_count} tanesi (%{intl_percentage:.1f}) International Product.",
                    'data': analysis_df[analysis_df['is_international']]
                })
            
            # Sales konsantrasyonu
            intl_df = analysis_df[analysis_df['is_international']]
            if len(intl_df) > 0:
                total_intl_sales = intl_df['total_sales'].sum()
                sales_cols = [col for col in df.columns if 'Sales_' in col]
                
                if sales_cols:
                    total_sales = df[sales_cols[-1]].sum()
                    
                    if total_sales > 0:
                        intl_sales_share = (total_intl_sales / total_sales * 100)
                        
                        insights.append({
                            'type': 'success',
                            'title': f'💰 International Product Pazar Payı',
                            'description': f"International Product'lar toplam pazarın %{intl_sales_share:.1f}'ini oluşturuyor.",
                            'data': None
                        })
            
            # Top International Products
            top_intl = analysis_df[analysis_df['is_international']].nlargest(5, 'total_sales')
            if len(top_intl) > 0:
                top_molecule = top_intl.iloc[0]['Molecule']
                top_sales = top_intl.iloc[0]['total_sales']
                
                insights.append({
                    'type': 'warning',
                    'title': f'🏆 En Büyük International Product',
                    'description': f"{top_molecule} ${top_sales/1e6:.1f}M satış ile en büyük International Product.",
                    'data': top_intl
                })
            
            # Growth karşılaştırması
            if 'avg_growth' in analysis_df.columns:
                intl_growth = analysis_df[analysis_df['is_international']]['avg_growth'].mean()
                local_growth = analysis_df[~analysis_df['is_international']]['avg_growth'].mean()
                
                if not pd.isna(intl_growth) and not pd.isna(local_growth):
                    growth_diff = intl_growth - local_growth
                    
                    if growth_diff > 0:
                        insights.append({
                            'type': 'success',
                            'title': f'📈 International Product Büyüme Avantajı',
                            'description': f"International Product'lar yerel ürünlerden %{growth_diff:.1f} daha hızlı büyüyor.",
                            'data': None
                        })
                    else:
                        insights.append({
                            'type': 'warning',
                            'title': f'⚠️ International Product Büyüme Riski',
                            'description': f"International Product'lar yerel ürünlerden %{abs(growth_diff):.1f} daha yavaş büyüyor.",
                            'data': None
                        })
            
            # Coğrafi yayılım
            if 'country_count' in analysis_df.columns:
                avg_countries = analysis_df[analysis_df['is_international']]['country_count'].mean()
                if not pd.isna(avg_countries):
                    insights.append({
                        'type': 'geographic',
                        'title': f'🗺️ Ortalama Coğrafi Yayılım',
                        'description': f"International Product'lar ortalama {avg_countries:.1f} ülkede satılıyor.",
                        'data': None
                    })
            
            return insights
            
        except Exception as e:
            st.warning(f"International Product içgörü hatası: {str(e)}")
            return []
    
    @staticmethod
    def analyze_market_trends(df):
        """Pazar trendlerini analiz et"""
        try:
            trends = {}
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if len(sales_cols) >= 2:
                yearly_trend = {}
                for col in sorted(sales_cols):
                    year = col.split('_')[-1]
                    yearly_trend[year] = df[col].sum()
                
                trends['Yearly_Sales'] = yearly_trend
                
                years = sorted(yearly_trend.keys())
                for i in range(1, len(years)):
                    prev_year = years[i-1]
                    curr_year = years[i]
                    growth = ((yearly_trend[curr_year] - yearly_trend[prev_year]) / 
                              yearly_trend[prev_year] * 100) if yearly_trend[prev_year] > 0 else 0
                    trends[f'Growth_{prev_year}_{curr_year}'] = growth
            
            return trends
            
        except Exception as e:
            st.warning(f"Trend analizi hatası: {str(e)}")
            return {}
    
    @staticmethod
    def detect_strategic_insights(df):
        """Stratejik içgörüleri tespit et"""
        insights = []
        
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if not sales_cols:
                return insights
            
            latest_sales_col = sales_cols[-1]
            year = latest_sales_col.split('_')[-1]
            
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            latest_growth_col = growth_cols[-1] if growth_cols else None
            
            # 1. En çok satan ürünler
            top_products = df.nlargest(10, latest_sales_col)
            if not top_products.empty:
                total_sales = df[latest_sales_col].sum()
                if total_sales > 0:
                    top_products_share = (top_products[latest_sales_col].sum() / total_sales * 100)
                    
                    insights.append({
                        'type': 'success',
                        'title': f'🏆 Top 10 Ürün - {year}',
                        'description': f"En çok satan 10 ürün toplam pazarın %{top_products_share:.1f}'ini oluşturuyor.",
                        'data': top_products
                    })
            
            # 2. En hızlı büyüyen ürünler
            if latest_growth_col:
                top_growth = df.nlargest(10, latest_growth_col)
                if not top_growth.empty:
                    insights.append({
                        'type': 'info',
                        'title': f'🚀 En Hızlı Büyüyen 10 Ürün',
                        'description': f"En hızlı büyüyen ürünler ortalama %{top_growth[latest_growth_col].mean():.1f} büyüme gösteriyor.",
                        'data': top_growth
                    })
            
            # 3. En çok satan şirketler
            if 'Corporation' in df.columns:
                top_companies = df.groupby('Corporation')[latest_sales_col].sum().nlargest(5)
                if not top_companies.empty:
                    top_company = top_companies.index[0]
                    top_company_share = (top_companies.iloc[0] / df[latest_sales_col].sum()) * 100
                    
                    insights.append({
                        'type': 'warning',
                        'title': f'🏢 Pazar Lideri - {year}',
                        'description': f"{top_company} %{top_company_share:.1f} pazar payı ile lider konumda.",
                        'data': None
                    })
            
            # 4. Coğrafi dağılım
            if 'Country' in df.columns:
                top_countries = df.groupby('Country')[latest_sales_col].sum().nlargest(5)
                if not top_countries.empty:
                    top_country = top_countries.index[0]
                    top_country_share = (top_countries.iloc[0] / df[latest_sales_col].sum()) * 100
                    
                    insights.append({
                        'type': 'geographic',
                        'title': f'🌍 En Büyük Pazar - {year}',
                        'description': f"{top_country} %{top_country_share:.1f} pay ile en büyük pazar.",
                        'data': None
                    })
            
            # 5. Fiyat analizi
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if price_cols:
                avg_price = df[price_cols[-1]].mean()
                price_std = df[price_cols[-1]].std()
                
                insights.append({
                    'type': 'price',
                    'title': f'💰 Fiyat Analizi - {year}',
                    'description': f"Ortalama fiyat: ${avg_price:.2f} (Standart sapma: ${price_std:.2f})",
                    'data': None
                })
            
            # 6. International Product içgörüleri
            intl_insights = AdvancedPharmaAnalytics.get_international_product_insights(df)
            insights.extend(intl_insights)
            
            return insights
            
        except Exception as e:
            st.warning(f"İçgörü tespiti hatası: {str(e)}")
            return []

# ================================================
# 5. GÖRSELLEŞTİRME MOTORU (INTERNATIONAL PRODUCT EKLENDİ)
# ================================================

class ProfessionalVisualization:
    """Profesyonel görselleştirme motoru"""
    
    @staticmethod
    def create_dashboard_metrics(df, metrics):
        """Dashboard metrik kartlarını oluştur"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sales = metrics.get('Total_Market_Value', 0)
                sales_year = metrics.get('Latest_Sales_Year', '')
                st.markdown(f"""
                <div class="custom-metric-card premium">
                    <div class="custom-metric-label">TOPLAM PAZAR DEĞERİ</div>
                    <div class="custom-metric-value">${total_sales/1e9:.2f}B</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">{sales_year}</span>
                        <span>Global Pazar</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_growth = metrics.get('Avg_Growth_Rate', 0)
                growth_class = "success" if avg_growth > 0 else "danger"
                st.markdown(f"""
                <div class="custom-metric-card {growth_class}">
                    <div class="custom-metric-label">ORTALAMA BÜYÜME</div>
                    <div class="custom-metric-value">{avg_growth:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">YoY</span>
                        <span>Yıllık Büyüme</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrics.get('HHI_Index', 0)
                hhi_status = "danger" if hhi > 2500 else "warning" if hhi > 1500 else "success"
                st.markdown(f"""
                <div class="custom-metric-card {hhi_status}">
                    <div class="custom-metric-label">REKABET YOĞUNLUĞU</div>
                    <div class="custom-metric-value">{hhi:.0f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-warning">HHI Index</span>
                        <span>{'Monopol' if hhi > 2500 else 'Oligopol' if hhi > 1500 else 'Rekabetçi'}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                intl_share = metrics.get('International_Product_Share', 0)
                intl_color = "success" if intl_share > 20 else "warning" if intl_share > 10 else "info"
                st.markdown(f"""
                <div class="custom-metric-card {intl_color}">
                    <div class="custom-metric-label">INTERNATIONAL PRODUCT PAYI</div>
                    <div class="custom-metric-value">{intl_share:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Global Yayılım</span>
                        <span>Multi-Market</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                unique_molecules = metrics.get('Unique_Molecules', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">MOLEKÜL ÇEŞİTLİLİĞİ</div>
                    <div class="custom-metric-value">{unique_molecules:,}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">Unique</span>
                        <span>Farklı Molekül</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                avg_price = metrics.get('Avg_Price', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">ORTALAMA FİYAT</div>
                    <div class="custom-metric-value">${avg_price:.2f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Per Unit</span>
                        <span>Ortalama</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                high_growth = metrics.get('High_Growth_Products', 0)
                total_products = metrics.get('Total_Rows', 0)
                high_growth_pct = (high_growth / total_products * 100) if total_products > 0 else 0
                st.markdown(f"""
                <div class="custom-metric-card success">
                    <div class="custom-metric-label">YÜKSEK BÜYÜME</div>
                    <div class="custom-metric-value">{high_growth_pct:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">{high_growth} ürün</span>
                        <span>> %20 büyüme</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                country_coverage = metrics.get('Country_Coverage', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">COĞRAFİ YAYILIM</div>
                    <div class="custom-metric-value">{country_coverage}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Ülke</span>
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
            
            # Sadece International Product'ları filtrele
            intl_df = analysis_df[analysis_df['is_international']]
            
            if len(intl_df) == 0:
                return None
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('International vs Local Dağılımı', 'International Product Pazar Payı',
                               'Coğrafi Yayılım Analizi', 'Büyüme Performansı Karşılaştırması'),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # 1. International vs Local dağılımı
            intl_counts = analysis_df['is_international'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=['International', 'Local'],
                    values=intl_counts.values,
                    hole=0.4,
                    marker_colors=['#3b82f6', '#64748b'],
                    textinfo='percent+label',
                    textposition='outside'
                ),
                row=1, col=1
            )
            
            # 2. Pazar payı karşılaştırması
            intl_sales = intl_df['total_sales'].sum()
            local_sales = analysis_df[~analysis_df['is_international']]['total_sales'].sum()
            total_sales = intl_sales + local_sales
            
            if total_sales > 0:
                intl_share = (intl_sales / total_sales) * 100
                local_share = (local_sales / total_sales) * 100
                
                fig.add_trace(
                    go.Bar(
                        x=['International', 'Local'],
                        y=[intl_share, local_share],
                        marker_color=['#3b82f6', '#64748b'],
                        text=[f'{intl_share:.1f}%', f'{local_share:.1f}%'],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
            
            # 3. Coğrafi yayılım
            if 'country_count' in intl_df.columns:
                country_dist = intl_df['country_count'].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(
                        x=country_dist.index.astype(str),
                        y=country_dist.values,
                        marker_color='#10b981',
                        name='Ülke Sayısı'
                    ),
                    row=2, col=1
                )
            
            # 4. Büyüme karşılaştırması
            if 'avg_growth' in analysis_df.columns:
                intl_growth = intl_df['avg_growth'].mean()
                local_growth = analysis_df[~analysis_df['is_international']]['avg_growth'].mean()
                
                if not pd.isna(intl_growth) and not pd.isna(local_growth):
                    fig.add_trace(
                        go.Bar(
                            x=['International', 'Local'],
                            y=[intl_growth, local_growth],
                            marker_color=['#3b82f6', '#64748b'],
                            text=[f'{intl_growth:.1f}%', f'{local_growth:.1f}%'],
                            textposition='auto'
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                showlegend=False,
                title_text="International Product Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"International Product grafiği hatası: {str(e)}")
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
                # Ham veri
                df.to_excel(writer, sheet_name='HAM_VERI', index=False)
                
                # Özet metrikler
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['METRİK', 'DEĞER'])
                metrics_df.to_excel(writer, sheet_name='OZET_METRIKLER', index=False)
                
                # Pazar payı analizi
                sales_cols = [col for col in df.columns if 'Sales_' in col]
                if sales_cols and 'Corporation' in df.columns:
                    latest_sales_col = sales_cols[-1]
                    market_share = df.groupby('Corporation')[latest_sales_col].sum().sort_values(ascending=False)
                    market_share_df = market_share.reset_index()
                    market_share_df.columns = ['ŞİRKET', 'SATIŞ']
                    market_share_df['PAY (%)'] = (market_share_df['SATIŞ'] / market_share_df['SATIŞ'].sum()) * 100
                    market_share_df['KÜMÜLATİF_PAY'] = market_share_df['PAY (%)'].cumsum()
                    market_share_df.to_excel(writer, sheet_name='PAZAR_PAYI', index=False)
                
                # International Product analizi
                if analysis_df is not None:
                    analysis_df.to_excel(writer, sheet_name='INTERNATIONAL_ANALIZI', index=False)
                
                # İçgörüler
                if insights:
                    insights_data = []
                    for insight in insights:
                        insights_data.append({
                            'TİP': insight['type'],
                            'BAŞLIK': insight['title'],
                            'AÇIKLAMA': insight['description']
                        })
                    
                    insights_df = pd.DataFrame(insights_data)
                    insights_df.to_excel(writer, sheet_name='STRATEJIK_ICGORULER', index=False)
                
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
    
    # Session state başlatma
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'current_filters' not in st.session_state:
        st.session_state.current_filters = {}
    if 'international_analysis' not in st.session_state:
        st.session_state.international_analysis = None
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
        
        with st.expander("📁 VERİ YÜKLEME", expanded=True):
            uploaded_file = st.file_uploader(
                "Excel/CSV Dosyası Yükleyin",
                type=['xlsx', 'xls', 'csv'],
                help="TÜM VERİ yüklenecektir (örneklem yok). Büyük dosyalar için yükleme süresi uzun olabilir."
            )
            
            if uploaded_file:
                if st.button("🚀 TÜM VERİYİ YÜKLE & ANALİZ ET", type="primary", use_container_width=True):
                    with st.spinner("Tüm veri yükleniyor ve analiz ediliyor..."):
                        processor = OptimizedDataProcessor()
                        
                        # TÜM veriyi yükle
                        df = processor.load_large_dataset(uploaded_file)
                        
                        if df is not None and len(df) > 0:
                            # Veriyi hazırla
                            df = processor.prepare_analytics_data(df)
                            
                            # Session state güncelle
                            st.session_state.df = df
                            st.session_state.filtered_df = df.copy()
                            
                            # Analiz yap
                            analytics = AdvancedPharmaAnalytics()
                            st.session_state.metrics = analytics.calculate_comprehensive_metrics(df)
                            st.session_state.insights = analytics.detect_strategic_insights(df)
                            st.session_state.international_analysis = analytics.analyze_international_products(df)
                            
                            st.success(f"✅ {len(df):,} satır veri başarıyla yüklendi!")
                            st.rerun()
        
        # Filtreleme (veri yüklendiyse)
        if st.session_state.df is not None:
            st.markdown("---")
            df = st.session_state.df
            
            filter_system = AdvancedFilterSystem()
            search_term, filter_config, apply_filter, clear_filter = filter_system.create_filter_sidebar(df)
            
            if apply_filter:
                with st.spinner("Filtreler uygulanıyor..."):
                    filtered_df = filter_system.apply_filters(df, search_term, filter_config)
                    st.session_state.filtered_df = filtered_df
                    st.session_state.current_filters = filter_config
                    
                    # Analizleri güncelle
                    analytics = AdvancedPharmaAnalytics()
                    st.session_state.metrics = analytics.calculate_comprehensive_metrics(filtered_df)
                    st.session_state.insights = analytics.detect_strategic_insights(filtered_df)
                    st.session_state.international_analysis = analytics.analyze_international_products(filtered_df)
                    
                    st.success(f"✅ Filtreler uygulandı: {len(filtered_df):,} satır")
                    st.rerun()
            
            if clear_filter:
                st.session_state.filtered_df = st.session_state.df.copy()
                st.session_state.current_filters = {}
                st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
        <strong>PharmaIntelligence Pro</strong><br>
        v3.2 | International Product Analytics<br>
        © 2024 Tüm hakları saklıdır.
        </div>
        """, unsafe_allow_html=True)
    
    # Ana içerik
    if st.session_state.df is None:
        show_welcome_screen()
        return
    
    # Değişkenleri al
    df = st.session_state.filtered_df
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    intl_analysis = st.session_state.international_analysis
    
    # Filtre durumunu göster
    if st.session_state.current_filters:
        AdvancedFilterSystem.show_filter_status(
            st.session_state.current_filters,
            df,
            st.session_state.df
        )
    else:
        st.info(f"🎯 Aktif filtre yok | Gösterilen: {len(df):,} satır")
    
    # Tab'lar
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
        show_overview_tab(df, metrics, insights)
    
    with tab2:
        show_market_analysis_tab(df)
    
    with tab3:
        show_price_analysis_tab(df)
    
    with tab4:
        show_competition_analysis_tab(df, metrics)
    
    with tab5:
        show_international_product_tab(df, intl_analysis, metrics)
    
    with tab6:
        show_strategic_analysis_tab(df, insights)
    
    with tab7:
        show_reporting_tab(df, metrics, insights, intl_analysis)

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
            <h2 style="color: #f1f5f9; margin-bottom: 1rem;">PharmaIntelligence Pro'ya Hoşgeldiniz</h2>
            <p style="color: #cbd5e1; margin-bottom: 2rem; line-height: 1.6;">
            İlaç pazarı verilerinizi yükleyin ve güçlü analitik özelliklerin kilidini açın.
            <br><strong>TÜM VERİ</strong> yüklenir (örneklem yok) - 500K+ satır desteklenir.
            </p>
            
            <div class="get-started-box">
                <div class="get-started-title">🎯 Başlamak İçin</div>
                <div class="get-started-steps">
                1. Sol taraftaki panelden veri dosyanızı yükleyin<br>
                2. "TÜM VERİYİ YÜKLE & ANALİZ ET" butonuna tıklayın<br>
                3. Analiz sonuçlarını tablarda inceleyin
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_overview_tab(df, metrics, insights):
    """Genel Bakış tab'ını göster"""
    st.markdown('<h2 class="section-title">Genel Bakış ve Performans Göstergeleri</h2>', unsafe_allow_html=True)
    
    # Metrik kartları
    viz = ProfessionalVisualization()
    viz.create_dashboard_metrics(df, metrics)
    
    # İçgörüler
    st.markdown('<h3 class="subsection-title">🔍 Stratejik İçgörüler</h3>', unsafe_allow_html=True)
    
    if insights:
        for insight in insights[:6]:  # İlk 6 içgörüyü göster
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
    else:
        st.info("Verileriniz analiz ediliyor... Stratejik içgörüler burada görünecek.")
    
    # Veri önizleme
    st.markdown('<h3 class="subsection-title">📋 Veri Önizleme</h3>', unsafe_allow_html=True)
    
    preview_col1, preview_col2 = st.columns([1, 3])
    
    with preview_col1:
        rows_to_show = st.slider("Gösterilecek Satır Sayısı", 10, 1000, 100, 10)
        
        # Öncelikli sütunlar
        priority_columns = ['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_23_24']
        available_columns = df.columns.tolist()
        
        # Mevcut sütunları kontrol et
        default_columns = []
        for col in priority_columns:
            if col in available_columns:
                default_columns.append(col)
        
        show_columns = st.multiselect(
            "Gösterilecek Sütunlar",
            options=available_columns,
            default=default_columns[:min(5, len(default_columns))]
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

def show_market_analysis_tab(df):
    """Pazar Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Pazar Analizi ve Trendler</h2>', unsafe_allow_html=True)
    
    # Satış trendleri
    st.markdown('<h3 class="subsection-title">📈 Satış Trendleri</h3>', unsafe_allow_html=True)
    
    sales_cols = [col for col in df.columns if 'Sales_' in col]
    if len(sales_cols) >= 2:
        # Yıllık satışları topla
        yearly_sales = {}
        for col in sorted(sales_cols):
            year = col.split('_')[-1]
            yearly_sales[year] = df[col].sum()
        
        # Çubuk grafik
        fig = px.bar(
            x=list(yearly_sales.keys()),
            y=list(yearly_sales.values()),
            title='Yıllık Toplam Satış Trendi',
            labels={'x': 'Yıl', 'y': 'Satış (USD)'},
            color=list(yearly_sales.values()),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Satış trend analizi için yeterli yıllık veri bulunamadı.")
    
    # Coğrafi dağılım
    st.markdown('<h3 class="subsection-title">🌍 Coğrafi Dağılım</h3>', unsafe_allow_html=True)
    
    if 'Country' in df.columns and sales_cols:
        latest_sales_col = sales_cols[-1]
        country_sales = df.groupby('Country')[latest_sales_col].sum().reset_index()
        country_sales = country_sales.sort_values(latest_sales_col, ascending=False)
        
        # Top 10 ülke
        top_countries = country_sales.head(10)
        
        fig = px.bar(
            top_countries,
            x='Country',
            y=latest_sales_col,
            title='Top 10 Ülke - Satış Performansı',
            color=latest_sales_col,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Coğrafi analiz için gerekli veri bulunamadı.")

def show_price_analysis_tab(df):
    """Fiyat Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Fiyat Analizi ve Optimizasyon</h2>', unsafe_allow_html=True)
    
    # Fiyat dağılımı
    price_cols = [col for col in df.columns if 'Avg_Price' in col]
    if price_cols:
        latest_price_col = price_cols[-1]
        
        st.markdown('<h3 class="subsection-title">💰 Fiyat Dağılımı</h3>', unsafe_allow_html=True)
        
        # Histogram
        fig = px.histogram(
            df,
            x=latest_price_col,
            nbins=50,
            title='Fiyat Dağılımı',
            labels={latest_price_col: 'Fiyat (USD)', 'count': 'Ürün Sayısı'}
        )
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Fiyat istatistikleri
        st.markdown('<h3 class="subsection-title">📊 Fiyat İstatistikleri</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_price = df[latest_price_col].mean()
            st.metric("Ortalama Fiyat", f"${avg_price:.2f}")
        
        with col2:
            median_price = df[latest_price_col].median()
            st.metric("Medyan Fiyat", f"${median_price:.2f}")
        
        with col3:
            min_price = df[latest_price_col].min()
            st.metric("Minimum Fiyat", f"${min_price:.2f}")
        
        with col4:
            max_price = df[latest_price_col].max()
            st.metric("Maximum Fiyat", f"${max_price:.2f}")
    else:
        st.info("Fiyat analizi için gerekli veri bulunamadı.")

def show_competition_analysis_tab(df, metrics):
    """Rekabet Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Rekabet Analizi ve Pazar Yapısı</h2>', unsafe_allow_html=True)
    
    # Pazar payı analizi
    if 'Corporation' in df.columns:
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        if sales_cols:
            latest_sales_col = sales_cols[-1]
            
            st.markdown('<h3 class="subsection-title">🏆 Pazar Payı Dağılımı</h3>', unsafe_allow_html=True)
            
            # Şirket bazlı satışlar
            company_sales = df.groupby('Corporation')[latest_sales_col].sum().sort_values(ascending=False)
            top_companies = company_sales.head(15)
            
            # Pasta grafik
            fig = px.pie(
                values=top_companies.values,
                names=top_companies.index,
                title='Top 15 Şirket - Pazar Payı',
                hole=0.4
            )
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Rekabet metrikleri
    st.markdown('<h3 class="subsection-title">📊 Rekabet Yoğunluğu</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hhi = metrics.get('HHI_Index', 0)
        if hhi > 2500:
            status = "Monopolistik"
        elif hhi > 1800:
            status = "Oligopol"
        else:
            status = "Rekabetçi"
        st.metric("HHI İndeksi", f"{hhi:.0f}", status)
    
    with col2:
        top3_share = metrics.get('Top_3_Share', 0)
        st.metric("Top 3 Payı", f"{top3_share:.1f}%")
    
    with col3:
        unique_corps = df['Corporation'].nunique() if 'Corporation' in df.columns else 0
        st.metric("Şirket Sayısı", unique_corps)

def show_international_product_tab(df, analysis_df, metrics):
    """International Product Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">🌍 International Product Analizi</h2>', unsafe_allow_html=True)
    
    if analysis_df is None:
        st.warning("International Product analizi için gerekli veri bulunamadı.")
        return
    
    # Genel metrikler
    st.markdown('<h3 class="subsection-title">📊 International Product Genel Bakış</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        intl_count = metrics.get('International_Product_Count', 0)
        total_molecules = metrics.get('Unique_Molecules', 0)
        if total_molecules > 0:
            intl_percentage = (intl_count / total_molecules * 100)
            st.metric("International Product", f"{intl_count}", f"%{intl_percentage:.1f}")
        else:
            st.metric("International Product", intl_count)
    
    with col2:
        intl_share = metrics.get('International_Product_Share', 0)
        st.metric("Pazar Payı", f"%{intl_share:.1f}")
    
    with col3:
        avg_countries = metrics.get('Avg_International_Countries', 0)
        st.metric("Ort. Ülke", f"{avg_countries:.1f}")
    
    with col4:
        intl_growth = metrics.get('International_Avg_Growth', 0)
        if intl_growth is not None:
            st.metric("Ort. Büyüme", f"%{intl_growth:.1f}")
    
    # Grafikler
    viz = ProfessionalVisualization()
    intl_fig = viz.create_international_product_analysis(df, analysis_df)
    
    if intl_fig:
        st.markdown('<h3 class="subsection-title">📈 International Product Analiz Grafikleri</h3>', unsafe_allow_html=True)
        st.plotly_chart(intl_fig, use_container_width=True)
    
    # Detaylı tablo - DÜZELTİLMİŞ VERSİYON
    st.markdown('<h3 class="subsection-title">📋 International Product Detaylı Listesi</h3>', unsafe_allow_html=True)
    
    # Sadece International Product'ları göster
    intl_df = analysis_df[analysis_df['is_international']].copy()
    
    if len(intl_df) > 0:
        # Formatlama fonksiyonu - DÜZELTİLDİ
        def format_value(x, format_type='currency'):
            if pd.isna(x):
                return "N/A"
            try:
                if format_type == 'currency':
                    return f"${float(x)/1e6:.2f}M"
                elif format_type == 'percent':
                    return f"{float(x):.1f}%"
                elif format_type == 'number':
                    return f"{float(x):.1f}"
                elif format_type == 'price':
                    return f"${float(x):.2f}"
                else:
                    return str(x)
            except:
                return str(x)
        
        # Gösterilecek sütunları seç
        display_columns = []
        possible_columns = [
            'Molecule', 'total_sales', 'corporation_count', 'country_count',
            'avg_price', 'avg_growth', 'international_segment'
        ]
        
        for col in possible_columns:
            if col in intl_df.columns:
                display_columns.append(col)
        
        # DataFrame'i kopyala ve formatla
        display_df = intl_df[display_columns].copy()
        
        # Her sütunu uygun şekilde formatla
        for col in display_df.columns:
            if col == 'total_sales':
                display_df[col] = display_df[col].apply(lambda x: format_value(x, 'currency'))
            elif col == 'avg_growth':
                display_df[col] = display_df[col].apply(lambda x: format_value(x, 'percent'))
            elif col == 'avg_price':
                display_df[col] = display_df[col].apply(lambda x: format_value(x, 'price'))
            elif col in ['corporation_count', 'country_count']:
                display_df[col] = display_df[col].apply(lambda x: format_value(x, 'number'))
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
    else:
        st.info("International Product bulunamadı.")
    
    # İçgörüler
    insights = AdvancedPharmaAnalytics.get_international_product_insights(df)
    
    if insights:
        st.markdown('<h3 class="subsection-title">💡 International Product İçgörüleri</h3>', unsafe_allow_html=True)
        
        for insight in insights:
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

def show_strategic_analysis_tab(df, insights):
    """Stratejik Analiz tab'ını göster"""
    st.markdown('<h2 class="section-title">Stratejik Analiz ve Öngörüler</h2>', unsafe_allow_html=True)
    
    # Büyüme fırsatları
    st.markdown('<h3 class="subsection-title">🚀 Büyüme Fırsatları</h3>', unsafe_allow_html=True)
    
    growth_insights = [i for i in insights if i['type'] in ['success', 'info']]
    
    if growth_insights:
        for insight in growth_insights[:3]:
            st.markdown(f"""
            <div class="insight-card {insight['type']}">
                <div class="insight-title">{insight['title']}</div>
                <div class="insight-content">{insight['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Henüz büyüme fırsatı tespit edilmedi.")
    
    # Risk analizi
    st.markdown('<h3 class="subsection-title">⚠️ Risk Analizi</h3>', unsafe_allow_html=True)
    
    risk_insights = [i for i in insights if i['type'] in ['warning', 'danger']]
    
    if risk_insights:
        for insight in risk_insights[:3]:
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
    
    # Rapor oluşturma
    st.markdown('<h3 class="subsection-title">📊 Rapor Oluşturma</h3>', unsafe_allow_html=True)
    
    if st.button("📈 Excel Raporu Oluştur", use_container_width=True):
        with st.spinner("Excel raporu oluşturuluyor..."):
            reporting = ProfessionalReporting()
            excel_report = reporting.generate_excel_report(df, metrics, insights, analysis_df)
            
            if excel_report:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="⬇️ Excel Raporunu İndir",
                    data=excel_report,
                    file_name=f"pharma_report_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.error("Excel raporu oluşturulamadı.")
    
    # International Product CSV
    if analysis_df is not None:
        st.markdown('<h3 class="subsection-title">🌍 International Product Verisi</h3>', unsafe_allow_html=True)
        
        if st.button("📊 International Product CSV", use_container_width=True):
            csv = analysis_df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="⬇️ CSV İndir",
                data=csv,
                file_name=f"international_products_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # İstatistikler
    st.markdown('<h3 class="subsection-title">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with col2:
        st.metric("Toplam Sütun", len(df.columns))
    
    with col3:
        mem_usage = df.memory_usage(deep=True).sum()/1024**2
        st.metric("Bellek Kullanımı", f"{mem_usage:.1f} MB")

# ================================================
# 8. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        if st.button("🔄 Sayfayı Yenile", use_container_width=True):
            st.rerun()
