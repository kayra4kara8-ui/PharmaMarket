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
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ================================================
# 2. OPTİMİZE VERİ İŞLEME SİSTEMİ
# ================================================

class OptimizedDataProcessor:
    """Optimize edilmiş veri işleme sınıfı - 500K+ satır için"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
    def load_large_dataset(file, sample_size=None, chunk_size=50000):
        """Büyük veri setlerini optimize şekilde yükle"""
        try:
            start_time = time.time()
            
            if file.name.endswith('.csv'):
                total_rows = sum(1 for line in file) - 1
                file.seek(0)
                
                if total_rows > 100000 and sample_size:
                    df = pd.read_csv(file, nrows=sample_size)
                else:
                    df = pd.read_csv(file)
                    
            elif file.name.endswith(('.xlsx', '.xls')):
                file_size = file.size / (1024 ** 2)
                
                if file_size > 50 or (sample_size and sample_size < 100000):
                    chunks = []
                    total_chunks = (sample_size // chunk_size) + 1 if sample_size else 10
                    
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
                            if sample_size:
                                progress = min(loaded_rows / sample_size, 1.0)
                            else:
                                progress = min(i / total_chunks, 0.95)
                            
                            progress_bar.progress(progress)
                            status_text.text(f"📊 {loaded_rows:,} satır yüklendi...")
                            
                            if sample_size and loaded_rows >= sample_size:
                                break
                        
                        df = pd.concat(chunks, ignore_index=True)
                        progress_bar.progress(1.0)
                        status_text.text(f"✅ {len(df):,} satır başarıyla yüklendi")
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                else:
                    df = pd.read_excel(file, engine='openpyxl')
            
            if sample_size and len(df) > sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
            
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
            
            for col in df.select_dtypes(include=['object']).columns:
                num_unique = df[col].nunique()
                if num_unique / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            
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
                        if col_min > -128 and col_max < 127:
                            df[col] = df[col].astype(np.int8)
                        elif col_min > -32768 and col_max < 32767:
                            df[col] = df[col].astype(np.int16)
                        elif col_min > -2147483648 and col_max < 2147483647:
                            df[col] = df[col].astype(np.int32)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
            
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
            
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_saved = original_memory - optimized_memory
            
            if memory_saved > 0:
                st.info(f"💾 Bellek optimizasyonu: {original_memory:.1f}MB → {optimized_memory:.1f}MB (%{memory_saved/original_memory*100:.1f} tasarruf)")
            
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
                
                if col == original_col:
                    col = col.strip()
            
            cleaned.append(str(col).strip())
        
        return cleaned
    
    @staticmethod
    def prepare_analytics_data(df):
        """Analiz için veriyi hazırla"""
        try:
            sales_cols = {}
            for col in df.columns:
                if 'Sales_' in col:
                    year = col.split('_')[-1]
                    sales_cols[year] = col
            
            years = sorted([int(y) for y in sales_cols.keys() if y.isdigit()])
            
            for i in range(1, len(years)):
                prev_year = str(years[i-1])
                curr_year = str(years[i])
                
                if prev_year in sales_cols and curr_year in sales_cols:
                    prev_col = sales_cols[prev_year]
                    curr_col = sales_cols[curr_year]
                    
                    df[f'Growth_{prev_year}_{curr_year}'] = ((df[curr_col] - df[prev_col]) / 
                                                             df[prev_col].replace(0, np.nan)) * 100
            
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if price_cols:
                df['Avg_Price_Overall'] = df[price_cols].mean(axis=1, skipna=True)
            
            if len(years) >= 2:
                first_year = str(years[0])
                last_year = str(years[-1])
                if first_year in sales_cols and last_year in sales_cols:
                    df['CAGR'] = ((df[sales_cols[last_year]] / df[sales_cols[first_year]].replace(0, np.nan)) ** 
                                 (1/len(years)) - 1) * 100
            
            if years and str(years[-1]) in sales_cols:
                last_sales_col = sales_cols[str(years[-1])]
                total_sales = df[last_sales_col].sum()
                if total_sales > 0:
                    df['Market_Share'] = (df[last_sales_col] / total_sales) * 100
            
            if 'Avg_Price_2024' in df.columns and 'Units_2024' in df.columns:
                df['Price_Volume_Ratio'] = df['Avg_Price_2024'] * df['Units_2024']
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 3:
                try:
                    scaler = StandardScaler()
                    numeric_data = df[numeric_cols].fillna(0)
                    scaled_data = scaler.fit_transform(numeric_data)
                    df['Performance_Score'] = scaled_data.mean(axis=1)
                except:
                    pass
            
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
            
            if 'Country' in available_columns:
                countries = sorted(df['Country'].dropna().unique())
                selected_countries = AdvancedFilterSystem.create_searchable_multiselect_with_all(
                    "🌍 Ülkeler",
                    countries,
                    key="countries_filter",
                    select_all_by_default=True
                )
                if selected_countries and "Tümü" not in selected_countries:
                    filter_config['Country'] = selected_countries
            
            if 'Corporation' in available_columns:
                companies = sorted(df['Corporation'].dropna().unique())
                selected_companies = AdvancedFilterSystem.create_searchable_multiselect_with_all(
                    "🏢 Şirketler",
                    companies,
                    key="companies_filter",
                    select_all_by_default=True
                )
                if selected_companies and "Tümü" not in selected_companies:
                    filter_config['Corporation'] = selected_companies
            
            if 'Molecule' in available_columns:
                molecules = sorted(df['Molecule'].dropna().unique())
                selected_molecules = AdvancedFilterSystem.create_searchable_multiselect_with_all(
                    "🧪 Moleküller",
                    molecules,
                    key="molecules_filter",
                    select_all_by_default=True
                )
                if selected_molecules and "Tümü" not in selected_molecules:
                    filter_config['Molecule'] = selected_molecules
            
            st.markdown("---")
            st.markdown('<div class="filter-title">📊 Numerik Filtreler</div>', unsafe_allow_html=True)
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                latest_sales_col = sales_cols[-1]
                min_sales = float(df[latest_sales_col].min())
                max_sales = float(df[latest_sales_col].max())
                
                col_slider1, col_slider2 = st.columns(2)
                with col_slider1:
                    min_value = st.number_input(
                        "Min Satış ($)",
                        min_value=min_sales,
                        max_value=max_sales,
                        value=min_sales,
                        step=1000.0,
                        key="sales_min"
                    )
                with col_slider2:
                    max_value = st.number_input(
                        "Max Satış ($)",
                        min_value=min_sales,
                        max_value=max_sales,
                        value=max_sales,
                        step=1000.0,
                        key="sales_max"
                    )
                
                if min_value <= max_value:
                    filter_config['sales_range'] = ((min_value, max_value), latest_sales_col)
                else:
                    st.warning("Min değer Max değerden küçük olmalıdır")
            
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                latest_growth_col = growth_cols[-1]
                min_growth = float(df[latest_growth_col].min())
                max_growth = float(df[latest_growth_col].max())
                
                col_growth1, col_growth2 = st.columns(2)
                with col_growth1:
                    min_growth_val = st.number_input(
                        "Min Büyüme (%)",
                        min_value=min_growth,
                        max_value=max_growth,
                        value=min(min_growth, -50.0),
                        step=5.0,
                        key="growth_min"
                    )
                with col_growth2:
                    max_growth_val = st.number_input(
                        "Max Büyüme (%)",
                        min_value=min_growth,
                        max_value=max_growth,
                        value=max(max_growth, 150.0),
                        step=5.0,
                        key="growth_max"
                    )
                
                if min_growth_val <= max_growth_val:
                    filter_config['growth_range'] = ((min_growth_val, max_growth_val), latest_growth_col)
            
            st.markdown("---")
            st.markdown('<div class="filter-title">⚙️ Ek Filtreler</div>', unsafe_allow_html=True)
            
            only_positive_growth = st.checkbox("📈 Sadece Pozitif Büyüyen Ürünler", value=False)
            if only_positive_growth and growth_cols:
                filter_config['positive_growth'] = True
            
            if sales_cols:
                sales_threshold = st.number_input(
                    "Satış Eşiği ($)",
                    min_value=0.0,
                    max_value=float(df[sales_cols[-1]].max()),
                    value=0.0,
                    step=1000.0,
                    key="sales_threshold"
                )
                if sales_threshold > 0:
                    filter_config['sales_threshold'] = (sales_threshold, sales_cols[-1])
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                apply_filter = st.button("✅ Filtre Uygula", width='stretch', key="apply_filter")
            with col2:
                clear_filter = st.button("🗑️ Filtreleri Temizle", width='stretch', key="clear_filter")
            with col3:
                save_filter = st.button("💾 Filtreyi Kaydet", width='stretch', key="save_filter")
            
            if 'saved_filters' not in st.session_state:
                st.session_state.saved_filters = {}
            
            if save_filter and filter_config:
                filter_name = st.text_input("Filtre Adı", placeholder="Örn: Yüksek Büyüyen Ürünler")
                if filter_name:
                    st.session_state.saved_filters[filter_name] = filter_config
                    st.success(f"✅ '{filter_name}' filtresi kaydedildi!")
            
            if st.session_state.saved_filters:
                st.markdown('<div class="filter-title">💾 Kayıtlı Filtreler</div>', unsafe_allow_html=True)
                saved_filter = st.selectbox(
                    "Kayıtlı Filtreler",
                    options=[""] + list(st.session_state.saved_filters.keys()),
                    key="saved_filters_select"
                )
                
                if saved_filter:
                    if st.button("📂 Bu Filtreyi Yükle", width='stretch'):
                        st.session_state.current_filters = st.session_state.saved_filters[saved_filter]
                        st.success(f"✅ '{saved_filter}' filtresi yüklendi!")
                        st.rerun()
            
            return search_term, filter_config, apply_filter, clear_filter
    
    @staticmethod
    def create_searchable_multiselect_with_all(label, options, key, select_all_by_default=False):
        """Arama yapılabilir multiselect - Tümü seçeneği dahil"""
        if not options:
            return []
        
        all_options = ["Tümü"] + options
        
        search_query = st.text_input(f"{label} Ara", key=f"{key}_search", placeholder="Arama yapın...")
        
        if search_query:
            filtered_options = ["Tümü"] + [opt for opt in options if search_query.lower() in str(opt).lower()]
        else:
            filtered_options = all_options
        
        if select_all_by_default:
            default_options = ["Tümü"]
        else:
            default_options = filtered_options[:min(5, len(filtered_options))]
        
        selected = st.multiselect(
            label,
            options=filtered_options,
            default=default_options,
            key=key,
            help="'Tümü' seçildiğinde diğer tüm seçenekler otomatik seçilir"
        )
        
        if "Tümü" in selected and len(selected) > 1:
            selected = [opt for opt in selected if opt != "Tümü"]
        elif "Tümü" in selected and len(selected) == 1:
            selected = options
        
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
        
        for column, values in filter_config.items():
            if column in filtered_df.columns and values and column not in ['sales_range', 'growth_range', 'positive_growth', 'sales_threshold']:
                filtered_df = filtered_df[filtered_df[column].isin(values)]
        
        if 'sales_range' in filter_config:
            (min_val, max_val), col_name = filter_config['sales_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        if 'growth_range' in filter_config:
            (min_val, max_val), col_name = filter_config['growth_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        if 'positive_growth' in filter_config and filter_config['positive_growth']:
            growth_cols = [col for col in filtered_df.columns if 'Growth_' in col]
            if growth_cols:
                filtered_df = filtered_df[filtered_df[growth_cols[-1]] > 0]
        
        if 'sales_threshold' in filter_config:
            threshold, col_name = filter_config['sales_threshold']
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
                elif key == 'sales_threshold':
                    threshold, col_name = value
                    filter_items.append(f"Satış > ${threshold:,.0f}")
            
            filter_info += " | ".join(filter_items)
            filter_info += f" | **Gösterilen:** {len(filtered_df):,} / {len(original_df):,} satır"
            
            st.markdown(f'<div class="filter-status">{filter_info}</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("❌ Tüm Filtreleri Temizle", width='stretch', key="clear_all_filters"):
                    st.session_state.filtered_df = st.session_state.df.copy()
                    st.session_state.current_filters = {}
                    st.session_state.metrics = AdvancedPharmaAnalytics().calculate_comprehensive_metrics(st.session_state.df)
                    st.session_state.insights = AdvancedPharmaAnalytics().detect_strategic_insights(st.session_state.df)
                    st.success("✅ Tüm filtreler temizlendi")
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
            metrics['Total_Rows'] = len(df)
            metrics['Total_Columns'] = len(df.columns)
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                latest_sales_col = sales_cols[-1]
                metrics['Latest_Sales_Year'] = latest_sales_col.split('_')[-1]
                metrics['Total_Market_Value'] = df[latest_sales_col].sum()
                metrics['Avg_Sales_Per_Product'] = df[latest_sales_col].mean()
                metrics['Median_Sales'] = df[latest_sales_col].median()
                metrics['Sales_Std_Dev'] = df[latest_sales_col].std()
                
                metrics['Sales_Q1'] = df[latest_sales_col].quantile(0.25)
                metrics['Sales_Q3'] = df[latest_sales_col].quantile(0.75)
                metrics['Sales_IQR'] = metrics['Sales_Q3'] - metrics['Sales_Q1']
            
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                latest_growth_col = growth_cols[-1]
                metrics['Avg_Growth_Rate'] = df[latest_growth_col].mean()
                metrics['Growth_Std_Dev'] = df[latest_growth_col].std()
                metrics['Positive_Growth_Products'] = (df[latest_growth_col] > 0).sum()
                metrics['Negative_Growth_Products'] = (df[latest_growth_col] < 0).sum()
                metrics['High_Growth_Products'] = (df[latest_growth_col] > 20).sum()
            
            if 'Corporation' in df.columns and sales_cols:
                latest_sales_col = sales_cols[-1]
                corp_sales = df.groupby('Corporation')[latest_sales_col].sum().sort_values(ascending=False)
                total_sales = corp_sales.sum()
                
                if total_sales > 0:
                    market_shares = (corp_sales / total_sales * 100)
                    metrics['HHI_Index'] = (market_shares ** 2).sum() / 10000
                    
                    top_n = [1, 3, 5, 10]
                    for n in top_n:
                        metrics[f'Top_{n}_Share'] = corp_sales.nlargest(n).sum() / total_sales * 100
                    
                    metrics['CR4_Ratio'] = metrics['Top_4_Share'] if 'Top_4_Share' in metrics else 0
            
            if 'Molecule' in df.columns:
                metrics['Unique_Molecules'] = df['Molecule'].nunique()
                if sales_cols:
                    mol_sales = df.groupby('Molecule')[latest_sales_col].sum()
                    total_mol_sales = mol_sales.sum()
                    if total_mol_sales > 0:
                        metrics['Top_10_Molecule_Share'] = mol_sales.nlargest(10).sum() / total_mol_sales * 100
            
            if 'Country' in df.columns:
                metrics['Country_Coverage'] = df['Country'].nunique()
                if sales_cols:
                    country_sales = df.groupby('Country')[latest_sales_col].sum()
                    metrics['Top_5_Country_Share'] = country_sales.nlargest(5).sum() / country_sales.sum() * 100
            
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
            
            metrics['Missing_Values'] = df.isnull().sum().sum()
            metrics['Missing_Percentage'] = (metrics['Missing_Values'] / (len(df) * len(df.columns))) * 100
            
            # INTERNATIONAL PRODUCT ANALİZİ EKLENDİ
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
            year = latest_sales_col.split('_')[-1]
            
            # International Product'ları tespit et
            international_products = {}
            
            # Molekül bazında International Product tespiti
            for molecule in df['Molecule'].unique():
                molecule_df = df[df['Molecule'] == molecule]
                
                # Eğer aynı molekül birden fazla şirkette veya ülkede varsa International Product
                unique_corporations = molecule_df['Corporation'].nunique() if 'Corporation' in df.columns else 0
                unique_countries = molecule_df['Country'].nunique() if 'Country' in df.columns else 0
                
                if unique_corporations > 1 or unique_countries > 1:
                    total_sales = molecule_df[latest_sales_col].sum()
                    if total_sales > 0:
                        international_products[molecule] = {
                            'total_sales': total_sales,
                            'corporation_count': unique_corporations,
                            'country_count': unique_countries,
                            'product_count': len(molecule_df),
                            'avg_growth': molecule_df['Growth_23_24'].mean() if 'Growth_23_24' in df.columns else None
                        }
            
            # International Product metrikleri
            metrics['International_Product_Count'] = len(international_products)
            metrics['International_Product_Sales'] = sum(data['total_sales'] for data in international_products.values())
            metrics['International_Product_Share'] = (metrics['International_Product_Sales'] / metrics['Total_Market_Value'] * 100) if metrics['Total_Market_Value'] > 0 else 0
            
            # Ortalama International Product özellikleri
            if international_products:
                metrics['Avg_International_Corporations'] = np.mean([data['corporation_count'] for data in international_products.values()])
                metrics['Avg_International_Countries'] = np.mean([data['country_count'] for data in international_products.values()])
            
            # Top International Products
            top_international = sorted(international_products.items(), 
                                     key=lambda x: x[1]['total_sales'], 
                                     reverse=True)[:10]
            
            metrics['Top_10_International_Sales'] = sum(data['total_sales'] for _, data in top_international)
            metrics['Top_10_International_Share'] = (metrics['Top_10_International_Sales'] / metrics['International_Product_Sales'] * 100) if metrics['International_Product_Sales'] > 0 else 0
            
            # Growth karşılaştırması
            if 'Growth_23_24' in df.columns:
                international_growth = []
                local_growth = []
                
                for molecule in df['Molecule'].unique():
                    molecule_df = df[df['Molecule'] == molecule]
                    avg_growth = molecule_df['Growth_23_24'].mean()
                    
                    if molecule in international_products:
                        international_growth.append(avg_growth)
                    else:
                        local_growth.append(avg_growth)
                
                if international_growth and local_growth:
                    metrics['International_Avg_Growth'] = np.mean(international_growth)
                    metrics['Local_Avg_Growth'] = np.mean(local_growth)
                    metrics['International_Growth_Premium'] = metrics['International_Avg_Growth'] - metrics['Local_Avg_Growth']
            
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
                
                unique_corporations = molecule_df['Corporation'].nunique() if 'Corporation' in df.columns else 0
                unique_countries = molecule_df['Country'].nunique() if 'Country' in df.columns else 0
                
                # International Product kriteri
                is_international = (unique_corporations > 1 or unique_countries > 1)
                
                total_sales = molecule_df[latest_sales_col].sum()
                avg_price = molecule_df['Avg_Price_2024'].mean() if 'Avg_Price_2024' in molecule_df.columns else None
                avg_growth = molecule_df['Growth_23_24'].mean() if 'Growth_23_24' in molecule_df.columns else None
                
                # Corporation distribution
                if 'Corporation' in df.columns:
                    top_corp = molecule_df.groupby('Corporation')[latest_sales_col].sum().idxmax()
                    corp_market_share = (molecule_df[molecule_df['Corporation'] == top_corp][latest_sales_col].sum() / total_sales * 100) if total_sales > 0 else 0
                else:
                    top_corp = None
                    corp_market_share = 0
                
                # Country distribution
                if 'Country' in df.columns:
                    top_country = molecule_df.groupby('Country')[latest_sales_col].sum().idxmax()
                    country_market_share = (molecule_df[molecule_df['Country'] == top_country][latest_sales_col].sum() / total_sales * 100) if total_sales > 0 else 0
                else:
                    top_country = None
                    country_market_share = 0
                
                # Complexity score (ne kadar yaygın)
                complexity_score = (unique_corporations * 0.6 + unique_countries * 0.4) / 2
                
                international_analysis.append({
                    'Molecule': molecule,
                    'is_international': is_international,
                    'total_sales': total_sales,
                    'corporation_count': unique_corporations,
                    'country_count': unique_countries,
                    'product_count': len(molecule_df),
                    'avg_price': avg_price,
                    'avg_growth': avg_growth,
                    'top_corporation': top_corp,
                    'corp_market_share': corp_market_share,
                    'top_country': top_country,
                    'country_market_share': country_market_share,
                    'complexity_score': complexity_score,
                    'sales_concentration': max(corp_market_share, country_market_share)
                })
            
            analysis_df = pd.DataFrame(international_analysis)
            
            # Segmentasyon
            if len(analysis_df) > 0:
                analysis_df['international_segment'] = pd.cut(
                    analysis_df['complexity_score'],
                    bins=[0, 0.5, 1.5, 3, float('inf')],
                    labels=['Local', 'Regional', 'Multi-National', 'Global']
                )
            
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
            intl_percentage = (intl_count / total_molecules * 100) if total_molecules > 0 else 0
            
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
                total_sales = df['Sales_2024'].sum() if 'Sales_2024' in df.columns else 0
                
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
    def perform_advanced_segmentation(df, n_clusters=4, method='kmeans'):
        """Gelişmiş pazar segmentasyonu"""
        try:
            features = []
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                features.extend(sales_cols[-2:])
            
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                features.append(growth_cols[-1])
            
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if price_cols:
                features.append(price_cols[-1])
            
            if len(features) < 2:
                st.warning("Segmentasyon için yeterli özellik bulunamadı")
                return None
            
            segmentation_data = df[features].fillna(0)
            
            if len(segmentation_data) < n_clusters * 10:
                st.warning("Segmentasyon için yeterli veri noktası yok")
                return None
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(segmentation_data)
            
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
            
            clusters = model.fit_predict(features_scaled)
            
            if hasattr(model, 'inertia_'):
                inertia = model.inertia_
            else:
                inertia = None
            
            if len(np.unique(clusters)) > 1:
                try:
                    silhouette = silhouette_score(features_scaled, clusters)
                    calinski = calinski_harabasz_score(features_scaled, clusters)
                except:
                    silhouette = None
                    calinski = None
            else:
                silhouette = None
                calinski = None
            
            result_df = df.copy()
            result_df['Segment'] = clusters
            
            segment_names = {
                0: 'Gelişen Ürünler',
                1: 'Olgun Ürünler',
                2: 'Yenilikçi Ürünler',
                3: 'Riskli Ürünler',
                4: 'Niş Ürünler',
                5: 'Volume Ürünleri',
                6: 'Premium Ürünler',
                7: 'Economy Ürünler'
            }
            
            result_df['Segment_Name'] = result_df['Segment'].map(
                lambda x: segment_names.get(x, f'Segment_{x}')
            )
            
            return {
                'data': result_df,
                'metrics': {
                    'inertia': inertia,
                    'silhouette_score': silhouette,
                    'calinski_score': calinski,
                    'n_clusters': len(np.unique(clusters))
                },
                'features_used': features
            }
            
        except Exception as e:
            st.warning(f"Segmentasyon hatası: {str(e)}")
            return None
    
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
            insights.append({
                'type': 'success',
                'title': f'🏆 Top 10 Ürün - {year}',
                'description': f"En çok satan 10 ürün toplam pazarın %{(top_products[latest_sales_col].sum() / df[latest_sales_col].sum() * 100):.1f}'ini oluşturuyor.",
                'data': top_products
            })
            
            # 2. En hızlı büyüyen ürünler
            if latest_growth_col:
                top_growth = df.nlargest(10, latest_growth_col)
                insights.append({
                    'type': 'info',
                    'title': f'🚀 En Hızlı Büyüyen 10 Ürün',
                    'description': f"En hızlı büyüyen ürünler ortalama %{top_growth[latest_growth_col].mean():.1f} büyüme gösteriyor.",
                    'data': top_growth
                })
            
            # 3. En çok satan şirketler
            if 'Corporation' in df.columns:
                top_companies = df.groupby('Corporation')[latest_sales_col].sum().nlargest(5)
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
            
            # 6. International Product içgörüleri (YENİ EKLENDİ)
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
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('International vs Local Dağılımı', 'International Product Pazar Payı',
                               'Coğrafi Yayılım Analizi', 'Büyüme Performansı Karşılaştırması'),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # International vs Local dağılımı
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
            
            # Pazar payı karşılaştırması
            intl_sales = analysis_df[analysis_df['is_international']]['total_sales'].sum()
            local_sales = analysis_df[~analysis_df['is_international']]['total_sales'].sum()
            
            fig.add_trace(
                go.Bar(
                    x=['International', 'Local'],
                    y=[intl_sales, local_sales],
                    marker_color=['#3b82f6', '#64748b'],
                    text=[f'${intl_sales/1e6:.1f}M', f'${local_sales/1e6:.1f}M'],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Coğrafi yayılım (International Product'lar için)
            intl_df = analysis_df[analysis_df['is_international']]
            if len(intl_df) > 0:
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
            
            # Büyüme karşılaştırması
            if 'avg_growth' in analysis_df.columns:
                intl_growth = analysis_df[analysis_df['is_international']]['avg_growth'].mean()
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
    
    @staticmethod
    def create_sales_trend_chart(df):
        """Satış trend grafikleri"""
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if len(sales_cols) >= 2:
                yearly_data = []
                for col in sorted(sales_cols):
                    year = col.split('_')[-1]
                    yearly_data.append({
                        'Year': year,
                        'Total_Sales': df[col].sum(),
                        'Avg_Sales': df[col].mean(),
                        'Product_Count': (df[col] > 0).sum()
                    })
                
                yearly_df = pd.DataFrame(yearly_data)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Yıllık Toplam Satış', 'Ortalama Satış Trendi', 
                                   'Ürün Sayısı Trendi', 'Büyüme Oranları'),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )
                
                fig.add_trace(
                    go.Bar(
                        x=yearly_df['Year'],
                        y=yearly_df['Total_Sales'],
                        name='Toplam Satış',
                        marker_color='#3b82f6',
                        text=[f'${x/1e6:.0f}M' for x in yearly_df['Total_Sales']],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=yearly_df['Year'],
                        y=yearly_df['Avg_Sales'],
                        mode='lines+markers',
                        name='Ortalama Satış',
                        line=dict(color='#8b5cf6', width=3),
                        marker=dict(size=10)
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(
                        x=yearly_df['Year'],
                        y=yearly_df['Product_Count'],
                        name='Ürün Sayısı',
                        marker_color='#10b981',
                        text=yearly_df['Product_Count'],
                        textposition='auto'
                    ),
                    row=2, col=1
                )
                
                if len(yearly_df) > 1:
                    growth_rates = []
                    for i in range(1, len(yearly_df)):
                        growth = ((yearly_df['Total_Sales'].iloc[i] - yearly_df['Total_Sales'].iloc[i-1]) / 
                                  yearly_df['Total_Sales'].iloc[i-1] * 100) if yearly_df['Total_Sales'].iloc[i-1] > 0 else 0
                        growth_rates.append(growth)
                    
                    fig.add_trace(
                        go.Bar(
                            x=yearly_df['Year'].iloc[1:],
                            y=growth_rates,
                            name='Büyüme (%)',
                            marker_color=['#ef4444' if g < 0 else '#10b981' for g in growth_rates],
                            text=[f'{g:.1f}%' for g in growth_rates],
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
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if not sales_cols:
                return None
            
            latest_sales_col = sales_cols[-1]
            
            if 'Corporation' in df.columns:
                company_sales = df.groupby('Corporation')[latest_sales_col].sum().sort_values(ascending=False)
                top_companies = company_sales.nlargest(15)
                others_sales = company_sales.iloc[15:].sum() if len(company_sales) > 15 else 0
                
                pie_data = top_companies.copy()
                if others_sales > 0:
                    pie_data['Diğer'] = others_sales
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Pazar Payı Dağılımı', 'Top 10 Şirket Satışları'),
                    specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                    column_widths=[0.4, 0.6]
                )
                
                fig.add_trace(
                    go.Pie(
                        labels=pie_data.index,
                        values=pie_data.values,
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
                        x=top_companies.values[:10],
                        y=top_companies.index[:10],
                        orientation='h',
                        marker_color='#3b82f6',
                        text=[f'${x/1e6:.1f}M' for x in top_companies.values[:10]],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9',
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
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if not sales_cols:
                return None
            
            latest_sales_col = sales_cols[-1]
            
            if 'Country' in df.columns:
                country_sales = df.groupby('Country')[latest_sales_col].sum().reset_index()
                country_sales = country_sales.sort_values(latest_sales_col, ascending=False)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Coğrafi Satış Dağılımı', 'Top 15 Ülke'),
                    specs=[[{'type': 'choropleth'}, {'type': 'bar'}],
                           [{'type': 'treemap'}, {'type': 'scatter'}]],
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )
                
                try:
                    fig.add_trace(
                        go.Choropleth(
                            locations=country_sales['Country'],
                            locationmode='country names',
                            z=country_sales[latest_sales_col],
                            colorscale='Blues',
                            colorbar_title="Satış (USD)",
                            hoverinfo='location+z'
                        ),
                        row=1, col=1
                    )
                except:
                    fig.add_trace(
                        go.Scatter(x=[0], y=[0], mode='text', text=['Harita yüklenemedi']),
                        row=1, col=1
                    )
                
                top_countries = country_sales.head(15)
                fig.add_trace(
                    go.Bar(
                        x=top_countries[latest_sales_col],
                        y=top_countries['Country'],
                        orientation='h',
                        marker_color='#8b5cf6',
                        text=[f'${x/1e6:.1f}M' for x in top_countries[latest_sales_col]],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Treemap(
                        labels=country_sales['Country'].head(20),
                        parents=[''] * min(20, len(country_sales)),
                        values=country_sales[latest_sales_col].head(20),
                        textinfo="label+value",
                        marker_colorscale='Viridis'
                    ),
                    row=2, col=1
                )
                
                if 'Growth_' in ''.join(df.columns):
                    growth_cols = [col for col in df.columns if 'Growth_' in col]
                    if growth_cols:
                        country_growth = df.groupby('Country')[growth_cols[-1]].mean().reset_index()
                        country_combined = pd.merge(country_sales, country_growth, on='Country')
                        
                        fig.add_trace(
                            go.Scatter(
                                x=country_combined[latest_sales_col],
                                y=country_combined[growth_cols[-1]],
                                mode='markers',
                                marker=dict(
                                    size=country_combined[latest_sales_col] / country_combined[latest_sales_col].max() * 50,
                                    color=country_combined[growth_cols[-1]],
                                    colorscale='RdYlGn',
                                    showscale=True,
                                    colorbar=dict(title="Büyüme %")
                                ),
                                text=country_combined['Country'],
                                hoverinfo='text+x+y'
                            ),
                            row=2, col=2
                        )
                
                fig.update_layout(
                    height=800,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9',
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
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            units_cols = [col for col in df.columns if 'Units_' in col]
            
            if not price_cols or not units_cols:
                return None
            
            latest_price_col = price_cols[-1]
            latest_units_col = units_cols[-1]
            
            sample_df = df[
                (df[latest_price_col] > 0) & 
                (df[latest_units_col] > 0)
            ].copy()
            
            if len(sample_df) > 10000:
                sample_df = sample_df.sample(10000, random_state=42)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Fiyat-Hacim İlişkisi', 'Fiyat Dağılımı',
                               'Hacim Dağılımı', 'Fiyat-Hacim Kategorileri'),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sample_df[latest_price_col],
                    y=sample_df[latest_units_col],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=sample_df[latest_units_col],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Hacim")
                    ),
                    text=sample_df['Molecule'] if 'Molecule' in sample_df.columns else None,
                    hoverinfo='text+x+y'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(
                    x=df[latest_price_col],
                    nbinsx=50,
                    marker_color='#3b82f6',
                    name='Fiyat Dağılımı'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Histogram(
                    x=df[latest_units_col],
                    nbinsx=50,
                    marker_color='#10b981',
                    name='Hacim Dağılımı'
                ),
                row=2, col=1
            )
            
            if 'Corporation' in df.columns:
                top_companies = df['Corporation'].value_counts().nlargest(5).index
                company_data = df[df['Corporation'].isin(top_companies)]
                
                fig.add_trace(
                    go.Box(
                        x=company_data['Corporation'],
                        y=company_data[latest_price_col],
                        marker_color='#8b5cf6',
                        name='Şirket Bazlı Fiyat'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                showlegend=False
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
                metrics_df.to_excel(writer, sheet_name='OZET_METRIKLER', index=False)
                
                sales_cols = [col for col in df.columns if 'Sales_' in col]
                if sales_cols and 'Corporation' in df.columns:
                    latest_sales_col = sales_cols[-1]
                    market_share = df.groupby('Corporation')[latest_sales_col].sum().sort_values(ascending=False)
                    market_share_df = market_share.reset_index()
                    market_share_df.columns = ['ŞİRKET', 'SATIŞ']
                    market_share_df['PAY (%)'] = (market_share_df['SATIŞ'] / market_share_df['SATIŞ'].sum()) * 100
                    market_share_df['KÜMÜLATİF_PAY'] = market_share_df['PAY (%)'].cumsum()
                    market_share_df.to_excel(writer, sheet_name='PAZAR_PAYI', index=False)
                
                if 'Country' in df.columns:
                    if sales_cols:
                        latest_sales_col = sales_cols[-1]
                        country_analysis = df.groupby('Country').agg({
                            latest_sales_col: ['sum', 'mean', 'count']
                        }).round(2)
                        country_analysis.columns = ['_'.join(col).strip() for col in country_analysis.columns.values]
                        country_analysis.to_excel(writer, sheet_name='ULKE_ANALIZI')
                
                if 'Molecule' in df.columns:
                    if sales_cols:
                        latest_sales_col = sales_cols[-1]
                        molecule_analysis = df.groupby('Molecule').agg({
                            latest_sales_col: ['sum', 'mean', 'count']
                        }).round(2)
                        molecule_analysis.columns = ['_'.join(col).strip() for col in molecule_analysis.columns.values]
                        molecule_analysis.nlargest(50, (latest_sales_col, 'sum')).to_excel(
                            writer, sheet_name='MOLEKUL_ANALIZI'
                        )
                
                if analysis_df is not None:
                    analysis_df.to_excel(writer, sheet_name='INTERNATIONAL_ANALIZI', index=False)
                
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
    if 'saved_filters' not in st.session_state:
        st.session_state.saved_filters = {}
    if 'international_analysis' not in st.session_state:
        st.session_state.international_analysis = None
    
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
        
        with st.expander("📁 VERİ YÜKLEME", expanded=True):
            uploaded_file = st.file_uploader(
                "Excel/CSV Dosyası Yükleyin",
                type=['xlsx', 'xls', 'csv'],
                help="500K+ satır desteklenir. Büyük dosyalar için örneklem önerilir."
            )
            
            if uploaded_file:
                col1, col2 = st.columns(2)
                with col1:
                    use_sample = st.checkbox("Örneklem Kullan", value=True)
                with col2:
                    sample_size = st.number_input(
                        "Örneklem Büyüklüğü", 
                        min_value=1000,
                        max_value=500000,
                        value=50000,
                        step=10000,
                        disabled=not use_sample
                    )
                
                if st.button("🚀 Yükle & Analiz Et", type="primary", width='stretch'):
                    with st.spinner("Veri işleniyor..."):
                        processor = OptimizedDataProcessor()
                        
                        if use_sample and sample_size:
                            df = processor.load_large_dataset(uploaded_file, sample_size=sample_size)
                        else:
                            df = processor.load_large_dataset(uploaded_file)
                        
                        if df is not None and len(df) > 0:
                            df = processor.optimize_dataframe(df)
                            df = processor.prepare_analytics_data(df)
                            
                            st.session_state.df = df
                            st.session_state.filtered_df = df.copy()
                            
                            analytics = AdvancedPharmaAnalytics()
                            st.session_state.metrics = analytics.calculate_comprehensive_metrics(df)
                            st.session_state.insights = analytics.detect_strategic_insights(df)
                            st.session_state.international_analysis = analytics.analyze_international_products(df)
                            
                            st.success(f"✅ {len(df):,} satır veri başarıyla yüklendi!")
                            st.rerun()
        
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
                    
                    analytics = AdvancedPharmaAnalytics()
                    st.session_state.metrics = analytics.calculate_comprehensive_metrics(filtered_df)
                    st.session_state.insights = analytics.detect_strategic_insights(filtered_df)
                    st.session_state.international_analysis = analytics.analyze_international_products(filtered_df)
                    
                    st.success(f"✅ Filtreler uygulandı: {len(filtered_df):,} satır")
                    st.rerun()
            
            if clear_filter:
                st.session_state.filtered_df = st.session_state.df.copy()
                st.session_state.current_filters = {}
                st.session_state.metrics = AdvancedPharmaAnalytics().calculate_comprehensive_metrics(st.session_state.df)
                st.session_state.insights = AdvancedPharmaAnalytics().detect_strategic_insights(st.session_state.df)
                st.session_state.international_analysis = AdvancedPharmaAnalytics().analyze_international_products(st.session_state.df)
                st.success("✅ Filtreler temizlendi")
                st.rerun()
        
        if st.session_state.df is not None:
            with st.expander("⚙️ ANALİZ AYARLARI", expanded=False):
                analysis_mode = st.selectbox(
                    "Analiz Modu",
                    ['Temel Analiz', 'Gelişmiş Analiz', 'Derin Öğrenme'],
                    help="Analiz derinliğini seçin"
                )
        
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
    intl_analysis = st.session_state.international_analysis
    
    if st.session_state.current_filters:
        AdvancedFilterSystem.show_filter_status(
            st.session_state.current_filters,
            df,
            st.session_state.df
        )
    else:
        st.info(f"🎯 Aktif filtre yok | Gösterilen: {len(df):,} satır")
    
    # YENİ TAB EKLENDİ: INTERNATIONAL PRODUCT ANALİZİ
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ",
        "💰 FİYAT ANALİZİ",
        "🏆 REKABET ANALİZİ",
        "🌍 INTERNATIONAL PRODUCT",  # YENİ TAB
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
    
    with tab5:  # YENİ TAB
        show_international_product_tab(df, intl_analysis, metrics)
    
    with tab6:
        show_strategic_analysis_tab(df, insights)
    
    with tab7:
        show_reporting_tab(df, metrics, insights, intl_analysis)

# ================================================
# TAB FONKSİYONLARI (INTERNATIONAL PRODUCT TAB EKLENDİ)
# ================================================

def show_welcome_screen():
    """Hoşgeldiniz ekranını göster"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem; background: #334155; 
                 border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin: 2rem 0;">
            <div style="font-size: 5rem; margin-bottom: 1rem; background: linear-gradient(135deg, #3b82f6, #8b5cf6); 
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;">💊</div>
            <h2 style="color: #f1f5f9; margin-bottom: 1rem;">PharmaIntelligence Pro'ya Hoşgeldiniz</h2>
            <p style="color: #cbd5e1; margin-bottom: 2rem; line-height: 1.6;">
            İlaç pazarı verilerinizi yükleyin ve güçlü analitik özelliklerin kilidini açın.
            <br>International Product analizi ile çoklu pazar stratejilerinizi optimize edin.
            </p>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 2rem 0;">
                <div style="text-align: left; padding: 1.5rem; background: #475569; border-radius: 12px; border-left: 4px solid #3b82f6;">
                    <div style="font-size: 2rem; color: #3b82f6; margin-bottom: 0.5rem;">🌍</div>
                    <div style="font-weight: 700; color: #f1f5f9; font-size: 1.1rem;">International Product</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.5rem;">Çoklu pazar ürün analizi</div>
                </div>
                <div style="text-align: left; padding: 1.5rem; background: #475569; border-radius: 12px; border-left: 4px solid #8b5cf6;">
                    <div style="font-size: 2rem; color: #8b5cf6; margin-bottom: 0.5rem;">📈</div>
                    <div style="font-weight: 700; color: #f1f5f9; font-size: 1.1rem;">Pazar Analizi</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.5rem;">Derin pazar içgörüleri ve trend analizi</div>
                </div>
                <div style="text-align: left; padding: 1.5rem; background: #475569; border-radius: 12px; border-left: 4px solid #10b981;">
                    <div style="font-size: 2rem; color: #10b981; margin-bottom: 0.5rem;">💰</div>
                    <div style="font-weight: 700; color: #f1f5f9; font-size: 1.1rem;">Fiyat Zekası</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.5rem;">Rekabetçi fiyatlandırma analizi</div>
                </div>
                <div style="text-align: left; padding: 1.5rem; background: #475569; border-radius: 12px; border-left: 4px solid #f59e0b;">
                    <div style="font-size: 2rem; color: #f59e0b; margin-bottom: 0.5rem;">🏆</div>
                    <div style="font-weight: 700; color: #f1f5f9; font-size: 1.1rem;">Rekabet Analizi</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.5rem;">Rakiplerinizi analiz edin ve fırsatları belirleyin</div>
                </div>
            </div>
            
            <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(59, 130, 246, 0.1); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3);">
                <div style="font-weight: 600; color: #3b82f6; margin-bottom: 0.5rem;">🎯 Başlamak İçin</div>
                <div style="color: #cbd5e1; font-size: 0.95rem;">
                1. Sol taraftaki panelden veri dosyanızı yükleyin<br>
                2. İstediğiniz örneklem boyutunu seçin<br>
                3. "Yükle & Analiz Et" butonuna tıklayın
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
                        for col in ['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_23_24']:
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
        
        priority_columns = ['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_23_24']
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
        missing_pct = metrics.get('Missing_Percentage', 0)
        status_color = "normal"
        if missing_pct < 5:
            status_color = "normal"
        elif missing_pct < 20:
            status_color = "off"
        else:
            status_color = "inverse"
        st.metric("Eksik Veri Oranı", f"{missing_pct:.1f}%", delta=None, delta_color=status_color)
    
    with quality_cols[1]:
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Kopya Satırlar", f"{duplicate_pct:.1f}%")
    
    with quality_cols[2]:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        total_cols = len(df.columns)
        st.metric("Sayısal Sütunlar", f"{numeric_cols}/{total_cols}")
    
    with quality_cols[3]:
        date_cols = len([col for col in df.columns if 'date' in col.lower()])
        st.metric("Tarih Sütunları", date_cols)

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
    
    if 'Molecule' in df.columns:
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        if sales_cols:
            latest_sales_col = sales_cols[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_molecules = df.groupby('Molecule')[latest_sales_col].sum().nlargest(15)
                fig = px.bar(
                    top_molecules,
                    orientation='h',
                    title=f'Top 15 Molekül - Satış Performansı',
                    color=top_molecules.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9',
                    xaxis_title='Satış (USD)',
                    yaxis_title='Molekül'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                growth_cols = [col for col in df.columns if 'Growth_' in col]
                if growth_cols:
                    latest_growth_col = growth_cols[-1]
                    molecule_growth = df.groupby('Molecule')[latest_growth_col].mean().nlargest(15)
                    fig = px.bar(
                        molecule_growth,
                        orientation='h',
                        title='Top 15 Molekül - Büyüme Oranları',
                        color=molecule_growth.values,
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#f1f5f9',
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
    
    price_cols = [col for col in df.columns if 'Avg_Price' in col]
    if price_cols:
        latest_price_col = price_cols[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            price_data = df[latest_price_col].dropna()
            if len(price_data) > 0:
                price_segments = pd.cut(
                    price_data,
                    bins=[0, 10, 50, 100, 500, float('inf')],
                    labels=['Economy (<$10)', 'Standard ($10-$50)', 'Premium ($50-$100)', 
                           'Super Premium ($100-$500)', 'Luxury (>$500)']
                )
                
                segment_counts = price_segments.value_counts()
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title='Fiyat Segmentleri Dağılımı',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols and len(price_data) > 0:
                latest_growth_col = growth_cols[-1]
                df_temp = df.copy()
                df_temp['Price_Segment'] = pd.cut(
                    df_temp[latest_price_col],
                    bins=[0, 10, 50, 100, 500, float('inf')],
                    labels=['Economy', 'Standard', 'Premium', 'Super Premium', 'Luxury']
                )
                
                segment_growth = df_temp.groupby('Price_Segment')[latest_growth_col].mean().dropna()
                
                if len(segment_growth) > 0:
                    fig = px.bar(
                        segment_growth,
                        orientation='v',
                        title='Fiyat Segmenti Bazlı Büyüme',
                        color=segment_growth.values,
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#f1f5f9',
                        xaxis_title='Fiyat Segmenti',
                        yaxis_title='Ortalama Büyüme (%)',
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<h3 class="subsection-title">📉 Fiyat Esnekliği Analizi</h3>', unsafe_allow_html=True)
    
    price_cols = [col for col in df.columns if 'Avg_Price' in col]
    units_cols = [col for col in df.columns if 'Units_' in col]
    
    if price_cols and units_cols:
        latest_price_col = price_cols[-1]
        latest_units_col = units_cols[-1]
        
        correlation_df = df[[latest_price_col, latest_units_col]].dropna()
        if len(correlation_df) > 10:
            correlation = correlation_df.corr().iloc[0, 1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fiyat-Hacim Korelasyonu", f"{correlation:.3f}")
            
            with col2:
                if correlation < -0.3:
                    elasticity_status = "Yüksek Esneklik"
                elif correlation > 0.3:
                    elasticity_status = "Düşük Esneklik"
                else:
                    elasticity_status = "Nötr"
                st.metric("Esneklik Durumu", elasticity_status)
            
            with col3:
                if correlation < -0.3:
                    recommendation = "Fiyat Artışı Riskli"
                elif correlation > 0.3:
                    recommendation = "Fiyat Artışı Mümkün"
                else:
                    recommendation = "Limitli Fiyat Artışı"
                st.metric("Öneri", recommendation)

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
        hhi = metrics.get('HHI_Index', 0)
        if hhi > 2500:
            hhi_status = "Monopolistik"
        elif hhi > 1800:
            hhi_status = "Oligopol"
        else:
            hhi_status = "Rekabetçi"
        st.metric("HHI İndeksi", f"{hhi:.0f}", hhi_status)
    
    with comp_cols[1]:
        top3_share = metrics.get('Top_3_Share', 0)
        if top3_share > 50:
            concentration = "Yüksek"
        elif top3_share > 30:
            concentration = "Orta"
        else:
            concentration = "Düşük"
        st.metric("Top 3 Payı", f"{top3_share:.1f}%", concentration)
    
    with comp_cols[2]:
        cr4 = metrics.get('CR4_Ratio', 0)
        st.metric("CR4 Oranı", f"{cr4:.1f}%")
    
    with comp_cols[3]:
        top10_molecule = metrics.get('Top_10_Molecule_Share', 0)
        st.metric("Top 10 Molekül Payı", f"{top10_molecule:.1f}%")
    
    st.markdown('<h3 class="subsection-title">📈 Şirket Performans Analizi</h3>', unsafe_allow_html=True)
    
    if 'Corporation' in df.columns:
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        if sales_cols:
            latest_sales_col = sales_cols[-1]
            
            company_metrics = df.groupby('Corporation').agg({
                latest_sales_col: ['sum', 'mean', 'count']
            }).round(2)
            
            company_metrics.columns = ['_'.join(col).strip() for col in company_metrics.columns.values]
            company_metrics = company_metrics.sort_values(f'{latest_sales_col}_sum', ascending=False)
            
            top_companies = company_metrics.head(20)
            
            if len(top_companies) > 0:
                try:
                    fig = px.imshow(
                        top_companies.T,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Viridis',
                        title='Top 20 Şirket Performans Matrisi'
                    )
                    fig.update_layout(
                        height=600,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#f1f5f9'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Heatmap oluşturulamadı. Verileri tablo olarak gösteriliyor.")
                
                with st.expander("📋 Detaylı Şirket Performans Tablosu"):
                    st.dataframe(
                        company_metrics.head(50),
                        use_container_width=True,
                        height=400
                    )

def show_international_product_tab(df, analysis_df, metrics):
    """YENİ TAB: International Product Analizi"""
    st.markdown('<h2 class="section-title">🌍 International Product Analizi</h2>', unsafe_allow_html=True)
    
    if analysis_df is None:
        st.warning("International Product analizi için gerekli veri bulunamadı.")
        return
    
    viz = ProfessionalVisualization()
    
    # Genel bakış metrikleri
    st.markdown('<h3 class="subsection-title">📊 International Product Genel Bakış</h3>', unsafe_allow_html=True)
    
    intl_cols = st.columns(4)
    
    with intl_cols[0]:
        intl_count = metrics.get('International_Product_Count', 0)
        total_molecules = metrics.get('Unique_Molecules', 0)
        intl_percentage = (intl_count / total_molecules * 100) if total_molecules > 0 else 0
        st.metric("International Product Sayısı", f"{intl_count}", f"%{intl_percentage:.1f}")
    
    with intl_cols[1]:
        intl_share = metrics.get('International_Product_Share', 0)
        st.metric("Pazar Payı", f"%{intl_share:.1f}")
    
    with intl_cols[2]:
        avg_countries = metrics.get('Avg_International_Countries', 0)
        st.metric("Ort. Ülke Sayısı", f"{avg_countries:.1f}")
    
    with intl_cols[3]:
        intl_growth = metrics.get('International_Avg_Growth', 0)
        local_growth = metrics.get('Local_Avg_Growth', 0)
        growth_diff = intl_growth - local_growth if intl_growth and local_growth else 0
        st.metric("Büyüme Farkı", f"%{growth_diff:.1f}")
    
    # Grafik analizi
    st.markdown('<h3 class="subsection-title">📈 International Product Analiz Grafikleri</h3>', unsafe_allow_html=True)
    
    intl_fig = viz.create_international_product_analysis(df, analysis_df)
    if intl_fig:
        st.plotly_chart(intl_fig, use_container_width=True, config={'displayModeBar': True})
    
    # Detaylı tablo
    st.markdown('<h3 class="subsection-title">📋 International Product Detaylı Listesi</h3>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Tüm International Product'lar", "Top Performanslılar", "Segment Bazlı"])
    
    with tab1:
        # International Product'ların detaylı listesi
        if len(analysis_df) > 0:
            display_columns = [
                'Molecule', 'is_international', 'total_sales', 'corporation_count',
                'country_count', 'avg_price', 'avg_growth', 'international_segment'
            ]
            
            display_columns = [col for col in display_columns if col in analysis_df.columns]
            
            intl_df_display = analysis_df[display_columns].copy()
            intl_df_display['total_sales'] = intl_df_display['total_sales'].apply(lambda x: f"${x/1e6:.2f}M")
            intl_df_display['avg_growth'] = intl_df_display['avg_growth'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "N/A")
            intl_df_display['avg_price'] = intl_df_display['avg_price'].apply(lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A")
            
            st.dataframe(
                intl_df_display,
                use_container_width=True,
                height=400
            )
    
    with tab2:
        # Top International Products
        if len(analysis_df) > 0:
            top_intl = analysis_df[analysis_df['is_international']].nlargest(20, 'total_sales')
            
            if len(top_intl) > 0:
                top_display_columns = [
                    'Molecule', 'total_sales', 'corporation_count', 'country_count',
                    'avg_growth', 'top_corporation', 'top_country'
                ]
                
                top_display_columns = [col for col in top_display_columns if col in top_intl.columns]
                
                top_intl_display = top_intl[top_display_columns].copy()
                top_intl_display['total_sales'] = top_intl_display['total_sales'].apply(lambda x: f"${x/1e6:.2f}M")
                top_intl_display['avg_growth'] = top_intl_display['avg_growth'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "N/A")
                
                st.dataframe(
                    top_intl_display,
                    use_container_width=True,
                    height=400
                )
    
    with tab3:
        # Segment bazlı analiz
        if 'international_segment' in analysis_df.columns:
            segment_analysis = analysis_df.groupby('international_segment').agg({
                'Molecule': 'count',
                'total_sales': 'sum',
                'avg_growth': 'mean',
                'corporation_count': 'mean',
                'country_count': 'mean'
            }).round(2)
            
            segment_analysis.columns = ['Molecule Count', 'Total Sales', 'Avg Growth %', 'Avg Corps', 'Avg Countries']
            segment_analysis['Total Sales'] = segment_analysis['Total Sales'].apply(lambda x: f"${x/1e6:.2f}M")
            segment_analysis['Avg Growth %'] = segment_analysis['Avg Growth %'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "N/A")
            
            st.dataframe(
                segment_analysis,
                use_container_width=True
            )
    
    # International Product içgörüleri
    st.markdown('<h3 class="subsection-title">💡 International Product İçgörüleri</h3>', unsafe_allow_html=True)
    
    insights = AdvancedPharmaAnalytics.get_international_product_insights(df)
    
    if insights:
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
            
            if insight.get('data') is not None and not insight['data'].empty:
                with st.expander("📋 Detaylı Liste"):
                    display_columns = []
                    for col in ['Molecule', 'total_sales', 'corporation_count', 'country_count', 'avg_growth']:
                        if col in insight['data'].columns:
                            display_columns.append(col)
                    
                    if display_columns:
                        st.dataframe(
                            insight['data'][display_columns],
                            use_container_width=True
                        )
    
    # International Product strateji önerileri
    st.markdown('<h3 class="subsection-title">🎯 Strateji Önerileri</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-card info">
            <div class="insight-title">🚀 International Product Büyüme Stratejisi</div>
            <div class="insight-content">
            1. Yüksek büyüme gösteren International Product'ları belirleyin<br>
            2. Bu ürünlerin diğer ülkelere yayılma potansiyelini değerlendirin<br>
            3. Yerel pazarlarda lider olan ürünleri International Product'a dönüştürün
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card success">
            <div class="insight-title">💰 International Product Fiyatlandırma</div>
            <div class="insight-content">
            1. Ülke bazında fiyatlandırma stratejileri geliştirin<br>
            2. Premium segmentteki International Product'ların fiyatını optimize edin<br>
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
            
            if 'Segment_Name' in results['data'].columns:
                segment_counts = results['data']['Segment_Name'].value_counts()
                
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
                    font_color='#f1f5f9'
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
        opportunity_insights = [i for i in insights if i['type'] in ['success', 'info']]
        
        if opportunity_insights:
            for insight in opportunity_insights[:3]:
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-content">{insight['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if insight.get('data') is not None and not insight['data'].empty:
                    with st.expander("🚀 Bu Fırsattaki Ürünler"):
                        display_columns = []
                        for col in ['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_23_24']:
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
            st.session_state.current_filters = {}
            if 'segmentation_results' in st.session_state:
                del st.session_state.segmentation_results
            if 'international_analysis' in st.session_state:
                del st.session_state.international_analysis
            st.rerun()
    
    with report_cols[2]:
        if st.button("💾 International Product CSV", width='stretch', key="intl_csv"):
            if analysis_df is not None:
                csv = analysis_df.to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="⬇️ CSV İndir",
                    data=csv,
                    file_name=f"international_products_{timestamp}.csv",
                    mime="text/csv",
                    width='stretch',
                    key="download_intl_csv"
                )
            else:
                st.warning("International Product analizi bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)
    
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with stat_cols[1]:
        st.metric("Toplam Sütun", len(df.columns))
    
    with stat_cols[2]:
        mem_usage = df.memory_usage(deep=True).sum()/1024**2
        st.metric("Bellek Kullanımı", f"{mem_usage:.1f} MB")
    
    with stat_cols[3]:
        intl_count = metrics.get('International_Product_Count', 0)
        st.metric("International Product", intl_count)

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
