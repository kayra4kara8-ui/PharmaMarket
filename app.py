# app.py - Profesyonel İlaç Pazarı Dashboard (DÜZELTİLMİŞ)
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
        'About': "### PharmaIntelligence Pro v3.1\nEnterprise Pharmaceutical Analytics Platform"
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
                # CSV için tahmini satır sayısı
                total_rows = sum(1 for line in file) - 1
                file.seek(0)
                
                if total_rows > 100000 and sample_size:
                    # Büyük CSV'ler için sample
                    df = pd.read_csv(file, nrows=sample_size)
                else:
                    df = pd.read_csv(file)
                    
            elif file.name.endswith(('.xlsx', '.xls')):
                # Excel için optimize yükleme
                file_size = file.size / (1024 ** 2)  # MB cinsinden
                
                if file_size > 50 or (sample_size and sample_size < 100000):
                    # Chunk'larla okuma
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
                            
                            # Progress update
                            loaded_rows = sum(len(c) for c in chunks)
                            if sample_size:
                                progress = min(loaded_rows / sample_size, 1.0)
                            else:
                                progress = min(i / total_chunks, 0.95)
                            
                            progress_bar.progress(progress)
                            status_text.text(f"📊 {loaded_rows:,} satır yüklendi...")
                            
                            # Sample boyutuna ulaşıldıysa dur
                            if sample_size and loaded_rows >= sample_size:
                                break
                        
                        df = pd.concat(chunks, ignore_index=True)
                        progress_bar.progress(1.0)
                        status_text.text(f"✅ {len(df):,} satır başarıyla yüklendi")
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                else:
                    # Küçük dosyalar için direkt okuma
                    df = pd.read_excel(file, engine='openpyxl')
            
            # Örneklem boyutu kontrolü
            if sample_size and len(df) > sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            # Veri optimizasyonu
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
            
            # Sütun isimlerini standardize et (Türkçe karakterleri koru)
            df.columns = OptimizedDataProcessor.clean_column_names(df.columns)
            
            # Kategorik sütunları optimize et
            for col in df.select_dtypes(include=['object']).columns:
                num_unique = df[col].nunique()
                if num_unique / len(df) < 0.5:  # %50'den az unique değer
                    df[col] = df[col].astype('category')
            
            # Numerik sütunları optimize et
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
                    # Float için optimize
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
            
            # Tarih sütunlarını optimize et
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
            
            # Boşlukları temizle
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
                # Türkçe karakterleri normalize et
                replacements = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in replacements.items():
                    col = col.replace(tr, en)
                
                # Özel karakterleri temizle
                col = col.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                col = ' '.join(col.split())  # Multiple spaces'i temizle
                
                # Standart sütun isimlerini eşleştir
                original_col = col  # Orijinal ismi sakla
                
                # Satış sütunlarını tespit et
                if 'USD' in col and 'MNF' in col and 'MAT' in col:
                    # Yılı tespit et
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
                
                # Eğer mapping yapılamadıysa orijinal ismi koru (temizlenmiş haliyle)
                if col == original_col:
                    # Sadece temizlik yap
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
            
            # Growth oranlarını hesapla (hangi sütunlar varsa)
            years = sorted([int(y) for y in sales_cols.keys() if y.isdigit()])
            
            for i in range(1, len(years)):
                prev_year = str(years[i-1])
                curr_year = str(years[i])
                
                if prev_year in sales_cols and curr_year in sales_cols:
                    prev_col = sales_cols[prev_year]
                    curr_col = sales_cols[curr_year]
                    
                    df[f'Growth_{prev_year}_{curr_year}'] = ((df[curr_col] - df[prev_col]) / 
                                                             df[prev_col].replace(0, np.nan)) * 100
            
            # Ortalama fiyat sütunlarını birleştir
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if price_cols:
                df['Avg_Price_Overall'] = df[price_cols].mean(axis=1, skipna=True)
            
            # CAGR hesapla (en az 2 yıl varsa)
            if len(years) >= 2:
                first_year = str(years[0])
                last_year = str(years[-1])
                if first_year in sales_cols and last_year in sales_cols:
                    df['CAGR'] = ((df[sales_cols[last_year]] / df[sales_cols[first_year]].replace(0, np.nan)) ** 
                                 (1/len(years)) - 1) * 100
            
            # Market Share hesapla (en son yıl için)
            if years and str(years[-1]) in sales_cols:
                last_sales_col = sales_cols[str(years[-1])]
                total_sales = df[last_sales_col].sum()
                if total_sales > 0:
                    df['Market_Share'] = (df[last_sales_col] / total_sales) * 100
            
            # Price-Volume Ratio (varsa)
            if 'Avg_Price_2024' in df.columns and 'Units_2024' in df.columns:
                df['Price_Volume_Ratio'] = df['Avg_Price_2024'] * df['Units_2024']
            
            # Performance Score (standartlaştırılmış)
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
    """Gelişmiş filtreleme sistemi"""
    
    @staticmethod
    def create_filter_sidebar(df):
        """Filtreleme sidebar'ını oluştur"""
        with st.sidebar.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
            st.markdown('<div class="filter-title">🔍 Arama ve Filtreleme</div>', unsafe_allow_html=True)
            
            # Global Arama
            search_term = st.text_input(
                "🔎 Global Arama",
                placeholder="Molekül, Şirket, Ülke...",
                help="Tüm sütunlarda arama yapın",
                key="global_search"
            )
            
            # Kategori bazlı filtreler
            filter_config = {}
            
            # Mevcut sütunları kontrol et
            available_columns = df.columns.tolist()
            
            if 'Country' in available_columns:
                countries = sorted(df['Country'].dropna().unique())
                selected_countries = AdvancedFilterSystem.create_searchable_multiselect(
                    "🌍 Ülkeler",
                    countries,
                    key="countries_filter"
                )
                if selected_countries:
                    filter_config['Country'] = selected_countries
            
            if 'Corporation' in available_columns:
                companies = sorted(df['Corporation'].dropna().unique())
                selected_companies = AdvancedFilterSystem.create_searchable_multiselect(
                    "🏢 Şirketler",
                    companies,
                    key="companies_filter"
                )
                if selected_companies:
                    filter_config['Corporation'] = selected_companies
            
            if 'Molecule' in available_columns:
                molecules = sorted(df['Molecule'].dropna().unique())
                selected_molecules = AdvancedFilterSystem.create_searchable_multiselect(
                    "🧪 Moleküller",
                    molecules,
                    key="molecules_filter"
                )
                if selected_molecules:
                    filter_config['Molecule'] = selected_molecules
            
            # Numerik filtreler
            st.markdown("---")
            st.markdown('<div class="filter-title">📊 Numerik Filtreler</div>', unsafe_allow_html=True)
            
            # Satış sütunlarını bul
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                latest_sales_col = sales_cols[-1]  # En son yıl
                min_sales = float(df[latest_sales_col].min())
                max_sales = float(df[latest_sales_col].max())
                
                sales_range = st.slider(
                    f"Satış Aralığı ({latest_sales_col.split('_')[-1]})",
                    min_value=min_sales,
                    max_value=max_sales,
                    value=(min_sales, max_sales),
                    format="$%.0f",
                    key="sales_range"
                )
                filter_config['sales_range'] = (sales_range, latest_sales_col)
            
            # Büyüme sütunlarını bul
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                latest_growth_col = growth_cols[-1]
                min_growth = float(df[latest_growth_col].min())
                max_growth = float(df[latest_growth_col].max())
                
                growth_range = st.slider(
                    "Büyüme Oranı (%)",
                    min_value=min_growth,
                    max_value=max_growth,
                    value=(min(min_growth, -50.0), max(max_growth, 150.0)),
                    format="%.1f%%",
                    key="growth_range"
                )
                filter_config['growth_range'] = (growth_range, latest_growth_col)
            
            # Filtreleme butonları
            col1, col2 = st.columns(2)
            with col1:
                apply_filter = st.button("✅ Filtre Uygula", use_container_width=True, key="apply_filter")
            with col2:
                clear_filter = st.button("🗑️ Filtreleri Temizle", use_container_width=True, key="clear_filter")
            
            return search_term, filter_config, apply_filter, clear_filter
    
    @staticmethod
    def create_searchable_multiselect(label, options, key):
        """Arama yapılabilir multiselect"""
        if not options:
            return []
        
        # Arama kutusu
        search_query = st.text_input(f"{label} Ara", key=f"{key}_search", placeholder="Arama yapın...")
        
        # Filtrelenmiş seçenekler
        if search_query:
            filtered_options = [opt for opt in options if search_query.lower() in str(opt).lower()]
        else:
            filtered_options = options
        
        # Başlangıç seçimleri
        if len(filtered_options) <= 20:
            default_options = filtered_options[:min(5, len(filtered_options))]
        else:
            default_options = []
        
        # Multiselect
        selected = st.multiselect(
            label,
            options=filtered_options,
            default=default_options,
            key=key
        )
        
        # İstatistikler
        if selected:
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
                    search_mask = search_mask | filtered_df[col].astype(str).str.contains(
                        search_term, case=False, na=False
                    )
                except:
                    continue
            filtered_df = filtered_df[search_mask]
        
        # Kategori filtreleri
        for column, values in filter_config.items():
            if column in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[column].isin(values)]
        
        # Numerik filtreler
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
        
        return filtered_df

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
            # Temel metrikler
            metrics['Total_Rows'] = len(df)
            metrics['Total_Columns'] = len(df.columns)
            
            # Satış sütunlarını bul
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
            
            # Pazar konsantrasyonu
            if 'Corporation' in df.columns and sales_cols:
                latest_sales_col = sales_cols[-1]
                corp_sales = df.groupby('Corporation')[latest_sales_col].sum().sort_values(ascending=False)
                total_sales = corp_sales.sum()
                
                if total_sales > 0:
                    # HHI Index
                    market_shares = (corp_sales / total_sales * 100)
                    metrics['HHI_Index'] = (market_shares ** 2).sum() / 10000
                    
                    # Top şirket payları
                    top_n = [1, 3, 5, 10]
                    for n in top_n:
                        metrics[f'Top_{n}_Share'] = corp_sales.nlargest(n).sum() / total_sales * 100
                    
                    # CR4 Ratio
                    metrics['CR4_Ratio'] = metrics['Top_4_Share'] if 'Top_4_Share' in metrics else 0
            
            # Molekül çeşitliliği
            if 'Molecule' in df.columns:
                metrics['Unique_Molecules'] = df['Molecule'].nunique()
                if sales_cols:
                    mol_sales = df.groupby('Molecule')[latest_sales_col].sum()
                    total_mol_sales = mol_sales.sum()
                    if total_mol_sales > 0:
                        metrics['Top_10_Molecule_Share'] = mol_sales.nlargest(10).sum() / total_mol_sales * 100
            
            # Coğrafi dağılım
            if 'Country' in df.columns:
                metrics['Country_Coverage'] = df['Country'].nunique()
                if sales_cols:
                    country_sales = df.groupby('Country')[latest_sales_col].sum()
                    metrics['Top_5_Country_Share'] = country_sales.nlargest(5).sum() / country_sales.sum() * 100
            
            # Fiyat analizleri
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if price_cols:
                latest_price_col = price_cols[-1]
                metrics['Avg_Price'] = df[latest_price_col].mean()
                metrics['Price_Variance'] = df[latest_price_col].var()
                metrics['Price_CV'] = (df[latest_price_col].std() / df[latest_price_col].mean()) * 100 if df[latest_price_col].mean() > 0 else 0
                
                # Price segments
                price_quartiles = df[latest_price_col].quantile([0.25, 0.5, 0.75])
                metrics['Price_Q1'] = price_quartiles[0.25]
                metrics['Price_Median'] = price_quartiles[0.5]
                metrics['Price_Q3'] = price_quartiles[0.75]
            
            # Veri kalitesi
            metrics['Missing_Values'] = df.isnull().sum().sum()
            metrics['Missing_Percentage'] = (metrics['Missing_Values'] / (len(df) * len(df.columns))) * 100
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return {}
    
    @staticmethod
    def analyze_market_trends(df):
        """Pazar trendlerini analiz et"""
        try:
            trends = {}
            
            # Yıllık trendler
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if len(sales_cols) >= 2:
                yearly_trend = {}
                for col in sorted(sales_cols):
                    year = col.split('_')[-1]
                    yearly_trend[year] = df[col].sum()
                
                trends['Yearly_Sales'] = yearly_trend
                
                # Büyüme trendleri
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
            # Özellik seçimi
            features = []
            
            # Satış özellikleri
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                features.extend(sales_cols[-2:])  # Son 2 yıl
            
            # Büyüme özellikleri
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                features.append(growth_cols[-1])
            
            # Fiyat özellikleri
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if price_cols:
                features.append(price_cols[-1])
            
            if len(features) < 2:
                st.warning("Segmentasyon için yeterli özellik bulunamadı")
                return None
            
            # Veriyi hazırla
            segmentation_data = df[features].fillna(0)
            
            if len(segmentation_data) < n_clusters * 10:
                st.warning("Segmentasyon için yeterli veri noktası yok")
                return None
            
            # Standardizasyon
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(segmentation_data)
            
            # Segmentasyon algoritması
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
            
            # Segmentasyon kalitesi
            if hasattr(model, 'inertia_'):
                inertia = model.inertia_
            else:
                inertia = None
            
            # Silhouette skoru
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
            
            # Sonuçları birleştir
            result_df = df.copy()
            result_df['Segment'] = clusters
            
            # Segment isimlendirme
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
            # Satış sütunlarını bul
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if not sales_cols:
                return insights
            
            latest_sales_col = sales_cols[-1]
            year = latest_sales_col.split('_')[-1]
            
            # Büyüme sütunlarını bul
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
            
            # 2. En hızlı büyüyen ürünler (varsa)
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
            
            return insights
            
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
            # Ana metrikler
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
                top3_share = metrics.get('Top_3_Share', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">TOP 3 PAYI</div>
                    <div class="custom-metric-value">{top3_share:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Pazar Kons.</span>
                        <span>Lider Şirketler</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # İkinci satır metrikler
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
    def create_sales_trend_chart(df):
        """Satış trend grafikleri"""
        try:
            # Yıllık satış trendi
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
                
                # Toplam satış
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
                
                # Ortalama satış
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
                
                # Ürün sayısı
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
                
                # Büyüme oranları
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
                # Şirket bazlı pazar payı
                company_sales = df.groupby('Corporation')[latest_sales_col].sum().sort_values(ascending=False)
                top_companies = company_sales.nlargest(15)
                others_sales = company_sales.iloc[15:].sum() if len(company_sales) > 15 else 0
                
                # Pie chart için veri hazırla
                pie_data = top_companies.copy()
                if others_sales > 0:
                    pie_data['Diğer'] = others_sales
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Pazar Payı Dağılımı', 'Top 10 Şirket Satışları'),
                    specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                    column_widths=[0.4, 0.6]
                )
                
                # Pie chart
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
                
                # Bar chart
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
                # Ülke bazlı satışlar
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
                
                # Choropleth map
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
                    # Harita oluşturulamazsa boş bir scatter ekle
                    fig.add_trace(
                        go.Scatter(x=[0], y=[0], mode='text', text=['Harita yüklenemedi']),
                        row=1, col=1
                    )
                
                # Top 15 ülke bar chart
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
                
                # Treemap
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
                
                # Scatter plot (satış vs büyüme)
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
            # Fiyat ve hacim sütunlarını bul
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            units_cols = [col for col in df.columns if 'Units_' in col]
            
            if not price_cols or not units_cols:
                return None
            
            latest_price_col = price_cols[-1]
            latest_units_col = units_cols[-1]
            
            # Sample veri (büyük datasetler için)
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
            
            # Scatter plot
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
            
            # Fiyat dağılımı
            fig.add_trace(
                go.Histogram(
                    x=df[latest_price_col],
                    nbinsx=50,
                    marker_color='#3b82f6',
                    name='Fiyat Dağılımı'
                ),
                row=1, col=2
            )
            
            # Hacim dağılımı
            fig.add_trace(
                go.Histogram(
                    x=df[latest_units_col],
                    nbinsx=50,
                    marker_color='#10b981',
                    name='Hacim Dağılımı'
                ),
                row=2, col=1
            )
            
            # Kategori bazlı box plot
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
    def generate_excel_report(df, metrics, insights, file_name="pharma_report"):
        """Excel raporu oluştur"""
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # 1. Ham veri
                df.to_excel(writer, sheet_name='HAM_VERI', index=False)
                
                # 2. Özet metrikler
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['METRİK', 'DEĞER'])
                metrics_df.to_excel(writer, sheet_name='OZET_METRIKLER', index=False)
                
                # 3. Pazar payı analizi
                sales_cols = [col for col in df.columns if 'Sales_' in col]
                if sales_cols and 'Corporation' in df.columns:
                    latest_sales_col = sales_cols[-1]
                    market_share = df.groupby('Corporation')[latest_sales_col].sum().sort_values(ascending=False)
                    market_share_df = market_share.reset_index()
                    market_share_df.columns = ['ŞİRKET', 'SATIŞ']
                    market_share_df['PAY (%)'] = (market_share_df['SATIŞ'] / market_share_df['SATIŞ'].sum()) * 100
                    market_share_df['KÜMÜLATİF_PAY'] = market_share_df['PAY (%)'].cumsum()
                    market_share_df.to_excel(writer, sheet_name='PAZAR_PAYI', index=False)
                
                # 4. Ülke analizi
                if 'Country' in df.columns:
                    if sales_cols:
                        latest_sales_col = sales_cols[-1]
                        country_analysis = df.groupby('Country').agg({
                            latest_sales_col: ['sum', 'mean', 'count']
                        }).round(2)
                        country_analysis.columns = ['_'.join(col).strip() for col in country_analysis.columns.values]
                        country_analysis.to_excel(writer, sheet_name='ULKE_ANALIZI')
                
                # 5. Molekül analizi
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
                
                # 6. İçgörüler
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
    
    @staticmethod
    def generate_pdf_summary(df, metrics, insights):
        """PDF özet raporu oluştur"""
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            latest_sales_col = sales_cols[-1] if sales_cols else 'Satışlar'
            sales_year = latest_sales_col.split('_')[-1] if sales_cols else ''
            
            summary = f"""
            PHARMAINTELLIGENCE PRO - PAZAR ANALİZ RAPORU
            Oluşturulma Tarihi: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            
            GENEL BAKIŞ
            ----------
            Toplam Ürün Sayısı: {metrics.get('Total_Rows', 0):,}
            Toplam Pazar Değeri ({sales_year}): ${metrics.get('Total_Market_Value', 0)/1e9:.2f}B
            Ortalama Büyüme Oranı: {metrics.get('Avg_Growth_Rate', 0):.1f}%
            Pazar Konsantrasyonu (HHI): {metrics.get('HHI_Index', 0):.0f}
            
            ANA PERFORMANS GÖSTERGELERİ
            ---------------------------
            • Top 3 Şirket Pazar Payı: {metrics.get('Top_3_Share', 0):.1f}%
            • Molekül Çeşitliliği: {metrics.get('Unique_Molecules', 0):,}
            • Coğrafi Kapsam: {metrics.get('Country_Coverage', 0)} ülke
            • Ortalama Fiyat: ${metrics.get('Avg_Price', 0):.2f}
            
            STRATEJİK İÇGÖRÜLER
            -------------------
            """
            
            for i, insight in enumerate(insights[:5], 1):
                summary += f"\n{i}. {insight['title']}: {insight['description']}\n"
            
            summary += f"""
            
            ÖNERİLER
            --------
            1. Yüksek büyüme potansiyeli olan segmentlere odaklanın
            2. Pazar konsantrasyonu {'yüksek' if metrics.get('HHI_Index', 0) > 1800 else 'orta' if metrics.get('HHI_Index', 0) > 1000 else 'düşük'}
            3. Coğrafi genişleme için {metrics.get('Country_Coverage', 0)} ülkeyi değerlendirin
            
            NOT: Bu özet rapor PharmaIntelligence Pro tarafından otomatik oluşturulmuştur.
            """
            
            return summary
            
        except Exception as e:
            st.error(f"PDF özet oluşturma hatası: {str(e)}")
            return "Rapor oluşturulamadı."

# ================================================
# 7. ANA UYGULAMA
# ================================================

def main():
    """Ana uygulama fonksiyonu"""
    
    # Header
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="pharma-title">💊 PHARMAINTELLIGENCE PRO</h1>
        <p class="pharma-subtitle">
        Enterprise-level pharmaceutical market analytics platform with advanced filtering, 
        predictive insights, and strategic recommendations for data-driven decision making.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Session State yönetimi
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
    
    # ================================================
    # SIDEBAR - KONTROL PANELİ
    # ================================================
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
        
        # Veri Yükleme
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
                
                if st.button("🚀 Yükle & Analiz Et", type="primary", use_container_width=True):
                    with st.spinner("Veri işleniyor..."):
                        processor = OptimizedDataProcessor()
                        
                        if use_sample and sample_size:
                            df = processor.load_large_dataset(uploaded_file, sample_size=sample_size)
                        else:
                            df = processor.load_large_dataset(uploaded_file)
                        
                        if df is not None and len(df) > 0:
                            # Veriyi optimize et ve hazırla
                            df = processor.optimize_dataframe(df)
                            df = processor.prepare_analytics_data(df)
                            
                            # Session state'e kaydet
                            st.session_state.df = df
                            st.session_state.filtered_df = df.copy()
                            
                            # Metrikleri ve içgörüleri hesapla
                            analytics = AdvancedPharmaAnalytics()
                            st.session_state.metrics = analytics.calculate_comprehensive_metrics(df)
                            st.session_state.insights = analytics.detect_strategic_insights(df)
                            
                            st.success(f"✅ {len(df):,} satır veri başarıyla yüklendi!")
                            st.rerun()
        
        # Filtreleme Sistemi
        if st.session_state.df is not None:
            st.markdown("---")
            with st.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
                df = st.session_state.df
                
                # Filtreleme sistemi
                filter_system = AdvancedFilterSystem()
                search_term, filter_config, apply_filter, clear_filter = filter_system.create_filter_sidebar(df)
                
                if apply_filter:
                    with st.spinner("Filtreler uygulanıyor..."):
                        filtered_df = filter_system.apply_filters(df, search_term, filter_config)
                        st.session_state.filtered_df = filtered_df
                        st.session_state.current_filters = filter_config
                        
                        # Filtrelenmiş veri için metrikleri güncelle
                        analytics = AdvancedPharmaAnalytics()
                        st.session_state.metrics = analytics.calculate_comprehensive_metrics(filtered_df)
                        st.session_state.insights = analytics.detect_strategic_insights(filtered_df)
                        
                        st.success(f"✅ Filtreler uygulandı: {len(filtered_df):,} satır")
                        st.rerun()
                
                if clear_filter:
                    st.session_state.filtered_df = st.session_state.df.copy()
                    st.session_state.current_filters = {}
                    st.session_state.metrics = AdvancedPharmaAnalytics().calculate_comprehensive_metrics(st.session_state.df)
                    st.session_state.insights = AdvancedPharmaAnalytics().detect_strategic_insights(st.session_state.df)
                    st.success("✅ Filtreler temizlendi")
                    st.rerun()
        
        # Analiz Ayarları
        if st.session_state.df is not None:
            with st.expander("⚙️ ANALİZ AYARLARI", expanded=False):
                analysis_mode = st.selectbox(
                    "Analiz Modu",
                    ['Temel Analiz', 'Gelişmiş Analiz', 'Derin Öğrenme'],
                    help="Analiz derinliğini seçin"
                )
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
        <strong>PharmaIntelligence Pro</strong><br>
        v3.1 | Enterprise Edition<br>
        © 2024 Tüm hakları saklıdır.
        </div>
        """, unsafe_allow_html=True)
    
    # ================================================
    # ANA İÇERİK
    # ================================================
    
    if st.session_state.df is None:
        # Hoşgeldiniz ekranı
        show_welcome_screen()
        return
    
    # Veri yüklendi, analiz ekranını göster
    df = st.session_state.filtered_df
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    
    # Filtre durumu göstergesi
    if len(st.session_state.current_filters) > 0:
        filter_info = f"🎯 Aktif Filtreler: {len(st.session_state.current_filters)} | "
        filter_info += f"Gösterilen: {len(df):,} / {len(st.session_state.df):,} satır"
        st.info(filter_info)
    
    # Ana Tablar
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ",
        "💰 FİYAT ANALİZİ",
        "🏆 REKABET ANALİZİ",
        "🔮 STRATEJİK ANALİZ",
        "📑 RAPORLAMA"
    ])
    
    # TAB 1: GENEL BAKIŞ
    with tab1:
        show_overview_tab(df, metrics, insights)
    
    # TAB 2: PAZAR ANALİZİ
    with tab2:
        show_market_analysis_tab(df)
    
    # TAB 3: FİYAT ANALİZİ
    with tab3:
        show_price_analysis_tab(df)
    
    # TAB 4: REKABET ANALİZİ
    with tab4:
        show_competition_analysis_tab(df, metrics)
    
    # TAB 5: STRATEJİK ANALİZ
    with tab5:
        show_strategic_analysis_tab(df, insights)
    
    # TAB 6: RAPORLAMA
    with tab6:
        show_reporting_tab(df, metrics, insights)

# ================================================
# TAB FONKSİYONLARI
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
            <br>500,000+ satır veri desteği ile enterprise-level analiz yapın.
            </p>
            
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 2rem 0;">
                <div style="text-align: left; padding: 1.5rem; background: #475569; border-radius: 12px; border-left: 4px solid #3b82f6;">
                    <div style="font-size: 2rem; color: #3b82f6; margin-bottom: 0.5rem;">📈</div>
                    <div style="font-weight: 700; color: #f1f5f9; font-size: 1.1rem;">Pazar Analizi</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.5rem;">Derin pazar içgörüleri ve trend analizi</div>
                </div>
                <div style="text-align: left; padding: 1.5rem; background: #475569; border-radius: 12px; border-left: 4px solid #8b5cf6;">
                    <div style="font-size: 2rem; color: #8b5cf6; margin-bottom: 0.5rem;">💰</div>
                    <div style="font-weight: 700; color: #f1f5f9; font-size: 1.1rem;">Fiyat Zekası</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.5rem;">Rekabetçi fiyatlandırma analizi</div>
                </div>
                <div style="text-align: left; padding: 1.5rem; background: #475569; border-radius: 12px; border-left: 4px solid #10b981;">
                    <div style="font-size: 2rem; color: #10b981; margin-bottom: 0.5rem;">🚀</div>
                    <div style="font-weight: 700; color: #f1f5f9; font-size: 1.1rem;">Büyüme Tahmini</div>
                    <div style="font-size: 0.9rem; color: #cbd5e1; margin-top: 0.5rem;">Öngörülebilir analitik ve trend tahmini</div>
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
    
    # Dashboard metrikleri
    viz = ProfessionalVisualization()
    viz.create_dashboard_metrics(df, metrics)
    
    # Stratejik içgörüler
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
                        # Hangi sütunların mevcut olduğunu kontrol et
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
    
    # Hızlı Veri Önizleme
    st.markdown('<h3 class="subsection-title">📋 Veri Önizleme</h3>', unsafe_allow_html=True)
    
    preview_col1, preview_col2 = st.columns([1, 3])
    
    with preview_col1:
        rows_to_show = st.slider("Gösterilecek Satır Sayısı", 10, 1000, 100, 10, key="rows_preview")
        
        # Mevcut sütunlardan default seçenekleri oluştur
        available_columns = df.columns.tolist()
        default_columns = []
        
        # Öncelikli sütunları kontrol et
        priority_columns = ['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_23_24']
        for col in priority_columns:
            if col in available_columns:
                default_columns.append(col)
            if len(default_columns) >= 5:
                break
        
        # Eğer yeterli sütun yoksa, ilk 5 sütunu al
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
    
    # Veri Kalitesi Göstergeleri
    st.markdown('<h3 class="subsection-title">📊 Veri Kalitesi Analizi</h3>', unsafe_allow_html=True)
    
    quality_cols = st.columns(4)
    
    with quality_cols[0]:
        missing_pct = metrics.get('Missing_Percentage', 0)
        status_color = "normal"  # Streamlit metric için delta_color
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
    
    # Satış trendleri
    st.markdown('<h3 class="subsection-title">📈 Satış Trendleri</h3>', unsafe_allow_html=True)
    trend_fig = viz.create_sales_trend_chart(df)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Satış trend analizi için yeterli yıllık veri bulunamadı.")
    
    # Coğrafi dağılım
    st.markdown('<h3 class="subsection-title">🌍 Coğrafi Dağılım</h3>', unsafe_allow_html=True)
    geo_fig = viz.create_geographic_distribution(df)
    if geo_fig:
        st.plotly_chart(geo_fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Coğrafi analiz için yeterli veri bulunamadı.")
    
    # Molekül analizi
    st.markdown('<h3 class="subsection-title">🧪 Molekül Bazlı Analiz</h3>', unsafe_allow_html=True)
    
    if 'Molecule' in df.columns:
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        if sales_cols:
            latest_sales_col = sales_cols[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top moleküller
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
                # Molekül büyüme dağılımı
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
    
    # Fiyat-Hacim analizi
    st.markdown('<h3 class="subsection-title">💰 Fiyat-Hacim İlişkisi</h3>', unsafe_allow_html=True)
    price_fig = viz.create_price_volume_analysis(df)
    if price_fig:
        st.plotly_chart(price_fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Fiyat-hacim analizi için yeterli veri bulunamadı.")
    
    # Fiyat segmentasyonu
    st.markdown('<h3 class="subsection-title">🎯 Fiyat Segmentasyonu</h3>', unsafe_allow_html=True)
    
    price_cols = [col for col in df.columns if 'Avg_Price' in col]
    if price_cols:
        latest_price_col = price_cols[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fiyat segmentleri
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
            # Segment bazlı büyüme
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
    
    # Fiyat esnekliği analizi
    st.markdown('<h3 class="subsection-title">📉 Fiyat Esnekliği Analizi</h3>', unsafe_allow_html=True)
    
    price_cols = [col for col in df.columns if 'Avg_Price' in col]
    units_cols = [col for col in df.columns if 'Units_' in col]
    
    if price_cols and units_cols:
        latest_price_col = price_cols[-1]
        latest_units_col = units_cols[-1]
        
        # Korelasyon analizi
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
    
    # Pazar payı analizi
    st.markdown('<h3 class="subsection-title">🏆 Pazar Payı Analizi</h3>', unsafe_allow_html=True)
    share_fig = viz.create_market_share_analysis(df)
    if share_fig:
        st.plotly_chart(share_fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Pazar payı analizi için gerekli veri bulunamadı.")
    
    # Rekabet metrikleri
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
    
    # Şirket performans karşılaştırması
    st.markdown('<h3 class="subsection-title">📈 Şirket Performans Analizi</h3>', unsafe_allow_html=True)
    
    if 'Corporation' in df.columns:
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        if sales_cols:
            latest_sales_col = sales_cols[-1]
            
            # Performans matrisi
            company_metrics = df.groupby('Corporation').agg({
                latest_sales_col: ['sum', 'mean', 'count']
            }).round(2)
            
            company_metrics.columns = ['_'.join(col).strip() for col in company_metrics.columns.values]
            company_metrics = company_metrics.sort_values(f'{latest_sales_col}_sum', ascending=False)
            
            # Top 20 şirket
            top_companies = company_metrics.head(20)
            
            if len(top_companies) > 0:
                # Heatmap
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
                
                # Detaylı tablo
                with st.expander("📋 Detaylı Şirket Performans Tablosu"):
                    st.dataframe(
                        company_metrics.head(50),
                        use_container_width=True,
                        height=400
                    )

def show_strategic_analysis_tab(df, insights):
    """Stratejik Analiz tab'ını göster"""
    st.markdown('<h2 class="section-title">Stratejik Analiz ve Öngörüler</h2>', unsafe_allow_html=True)
    
    # Segmentasyon analizi
    st.markdown('<h3 class="subsection-title">🎯 Pazar Segmentasyonu</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_clusters = st.slider("Segment Sayısı", 2, 8, 4, key="n_clusters")
        method = st.selectbox("Segmentasyon Metodu", ['kmeans', 'dbscan'], key="seg_method")
        
        if st.button("🔍 Segmentasyon Analizi Yap", type="primary", use_container_width=True, key="run_segmentation"):
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
            
            # Segment dağılımı
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
                
                # Segmentasyon kalitesi
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
    
    # Büyüme fırsatları
    st.markdown('<h3 class="subsection-title">🚀 Büyüme Fırsatları</h3>', unsafe_allow_html=True)
    
    if insights:
        # Fırsatları filtrele
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
                        # Hangi sütunların mevcut olduğunu kontrol et
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

def show_reporting_tab(df, metrics, insights):
    """Raporlama tab'ını göster"""
    st.markdown('<h2 class="section-title">Raporlama ve İndirme</h2>', unsafe_allow_html=True)
    
    # Rapor seçenekleri
    st.markdown('<h3 class="subsection-title">📊 Rapor Türleri</h3>', unsafe_allow_html=True)
    
    report_type = st.radio(
        "Rapor Türü Seçin",
        ['Excel Detaylı Rapor', 'PDF Özet Rapor', 'JSON Veri Paketi', 'CSV Ham Veri'],
        horizontal=True,
        key="report_type"
    )
    
    # Rapor oluşturma
    st.markdown('<h3 class="subsection-title">🛠️ Rapor Oluşturma</h3>', unsafe_allow_html=True)
    
    report_cols = st.columns(3)
    
    with report_cols[0]:
        if st.button("📈 Excel Raporu Oluştur", use_container_width=True, key="excel_report"):
            with st.spinner("Excel raporu oluşturuluyor..."):
                reporting = ProfessionalReporting()
                excel_report = reporting.generate_excel_report(df, metrics, insights)
                
                if excel_report:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="⬇️ Excel İndir",
                        data=excel_report,
                        file_name=f"pharma_report_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="download_excel"
                    )
                else:
                    st.error("Excel raporu oluşturulamadı.")
    
    with report_cols[1]:
        if st.button("📄 PDF Özet Oluştur", use_container_width=True, key="pdf_report"):
            with st.spinner("PDF özeti oluşturuluyor..."):
                reporting = ProfessionalReporting()
                pdf_summary = reporting.generate_pdf_summary(df, metrics, insights)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="⬇️ PDF İndir",
                    data=pdf_summary.encode('utf-8'),
                    file_name=f"pharma_summary_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_pdf"
                )
    
    with report_cols[2]:
        if st.button("🔄 Analizi Sıfırla", use_container_width=True, key="reset_analysis"):
            st.session_state.df = None
            st.session_state.filtered_df = None
            st.session_state.metrics = None
            st.session_state.insights = []
            st.session_state.current_filters = {}
            if 'segmentation_results' in st.session_state:
                del st.session_state.segmentation_results
            st.rerun()
    
    # Hızlı İstatistikler
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
        processing_time = "Hızlı" if len(df) < 100000 else "Orta" if len(df) < 500000 else "Yavaş"
        st.metric("İşlem Hızı", processing_time)
    
    # API ve Entegrasyon
    st.markdown('<h3 class="subsection-title">🔗 API ve Entegrasyon</h3>', unsafe_allow_html=True)
    
    with st.expander("📡 API Erişimi"):
        st.code("""
        # PharmaIntelligence Pro API Örneği
        import requests
        
        API_KEY = "your_api_key_here"
        BASE_URL = "https://api.pharmaintelligence.com/v1"
        
        # Pazar metriklerini çek
        response = requests.get(
            f"{BASE_URL}/market-metrics",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        # Segmentasyon analizi
        data = {"n_clusters": 4, "method": "kmeans"}
        response = requests.post(
            f"{BASE_URL}/segmentation",
            json=data,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        """, language="python")
        
        st.info("API erişimi için lütfen bizimle iletişime geçin.")

# ================================================
# 8. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    try:
        # Performans optimizasyonu
        gc.enable()
        
        # Exception handling
        main()
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        st.error("Detaylı hata bilgisi:")
        st.code(traceback.format_exc())
        
        # Yeniden deneme butonu
        if st.button("🔄 Sayfayı Yenile"):
            st.rerun()
