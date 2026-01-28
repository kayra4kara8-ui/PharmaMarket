# app.py - Profesyonel İlaç Pazarı Dashboard (Optimize Edilmiş)
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
    
    /* === TABLES === */
    .data-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: var(--bg-card);
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    .data-table th {
        background: var(--bg-hover);
        color: var(--text-primary);
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        padding: 1rem;
        text-align: left;
        border-bottom: 2px solid var(--accent-blue);
    }
    
    .data-table td {
        padding: 0.75rem 1rem;
        color: var(--text-secondary);
        border-bottom: 1px solid var(--bg-hover);
        font-size: 0.9rem;
    }
    
    .data-table tr:hover {
        background: rgba(59, 130, 246, 0.1);
    }
    
    /* === BUTTONS === */
    .custom-button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: var(--radius-sm);
        font-weight: 600;
        cursor: pointer;
        transition: all var(--transition-normal);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.95rem;
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
    }
    
    .custom-button.secondary {
        background: var(--bg-hover);
        color: var(--text-primary);
    }
    
    .custom-button.secondary:hover {
        background: var(--bg-surface);
    }
    
    /* === PROGRESS BARS === */
    .progress-bar {
        height: 6px;
        background: var(--bg-hover);
        border-radius: 3px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        border-radius: 3px;
        transition: width 1s ease-in-out;
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
            
            # Sütun isimlerini standardize et
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
                
                # Sütun isimlerini standardize et
                mapping = {
                    'MAT Q3 2022 USD MNF': 'Sales_2022',
                    'MAT Q3 2023 USD MNF': 'Sales_2023',
                    'MAT Q3 2024 USD MNF': 'Sales_2024',
                    'MAT Q3 2022 Units': 'Units_2022',
                    'MAT Q3 2023 Units': 'Units_2023',
                    'MAT Q3 2024 Units': 'Units_2024',
                    'MAT Q3 2024 Unit Avg Price USD MNF': 'Avg_Price_2024',
                    'MAT Q3 2023 Unit Avg Price USD MNF': 'Avg_Price_2023',
                    'MAT Q3 2022 Unit Avg Price USD MNF': 'Avg_Price_2022',
                }
                
                col = mapping.get(col, col)
            cleaned.append(str(col).strip())
        
        return cleaned
    
    @staticmethod
    def prepare_analytics_data(df):
        """Analiz için veriyi hazırla"""
        try:
            # Growth oranlarını hesapla
            if 'Sales_2022' in df.columns and 'Sales_2023' in df.columns:
                df['Growth_22_23'] = ((df['Sales_2023'] - df['Sales_2022']) / 
                                      df['Sales_2022'].replace(0, np.nan)) * 100
            
            if 'Sales_2023' in df.columns and 'Sales_2024' in df.columns:
                df['Growth_23_24'] = ((df['Sales_2024'] - df['Sales_2023']) / 
                                      df['Sales_2023'].replace(0, np.nan)) * 100
            
            # Ortalama fiyat sütunlarını birleştir
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if price_cols:
                df['Avg_Price_Overall'] = df[price_cols].mean(axis=1, skipna=True)
            
            # CAGR hesapla (3 yıllık)
            if all(col in df.columns for col in ['Sales_2022', 'Sales_2024']):
                df['CAGR_3Y'] = ((df['Sales_2024'] / df['Sales_2022'].replace(0, np.nan)) ** (1/2) - 1) * 100
            
            # Market Share hesapla (toplam satışa göre)
            if 'Sales_2024' in df.columns:
                total_sales = df['Sales_2024'].sum()
                if total_sales > 0:
                    df['Market_Share_2024'] = (df['Sales_2024'] / total_sales) * 100
            
            # Price-Volume Ratio
            if 'Avg_Price_2024' in df.columns and 'Units_2024' in df.columns:
                df['Price_Volume_Ratio'] = df['Avg_Price_2024'] * df['Units_2024']
            
            # Performance Score
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 3:
                # Standartlaştırılmış skor
                scaler = StandardScaler()
                numeric_data = df[numeric_cols].fillna(0)
                scaled_data = scaler.fit_transform(numeric_data)
                df['Performance_Score'] = scaled_data.mean(axis=1)
            
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
                help="Tüm sütunlarda arama yapın"
            )
            
            # Kategori bazlı filtreler
            filter_config = {}
            
            if 'Country' in df.columns:
                countries = sorted(df['Country'].dropna().unique())
                selected_countries = AdvancedFilterSystem.create_searchable_multiselect(
                    "🌍 Ülkeler",
                    countries,
                    key="countries_filter"
                )
                if selected_countries:
                    filter_config['Country'] = selected_countries
            
            if 'Corporation' in df.columns:
                companies = sorted(df['Corporation'].dropna().unique())
                selected_companies = AdvancedFilterSystem.create_searchable_multiselect(
                    "🏢 Şirketler",
                    companies,
                    key="companies_filter"
                )
                if selected_companies:
                    filter_config['Corporation'] = selected_companies
            
            if 'Molecule' in df.columns:
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
            
            if 'Sales_2024' in df.columns:
                min_sales, max_sales = float(df['Sales_2024'].min()), float(df['Sales_2024'].max())
                sales_range = st.slider(
                    "Satış Aralığı (2024)",
                    min_value=min_sales,
                    max_value=max_sales,
                    value=(min_sales, max_sales),
                    format="$%.0f"
                )
                filter_config['Sales_2024_range'] = sales_range
            
            if 'Growth_23_24' in df.columns:
                growth_range = st.slider(
                    "Büyüme Oranı (%)",
                    min_value=float(df['Growth_23_24'].min()),
                    max_value=float(df['Growth_23_24'].max()),
                    value=(-50.0, 150.0),
                    format="%.1f%%"
                )
                filter_config['Growth_range'] = growth_range
            
            # Filtreleme butonları
            col1, col2 = st.columns(2)
            with col1:
                apply_filter = st.button("✅ Filtre Uygula", use_container_width=True)
            with col2:
                clear_filter = st.button("🗑️ Filtreleri Temizle", use_container_width=True)
            
            return search_term, filter_config, apply_filter, clear_filter
    
    @staticmethod
    def create_searchable_multiselect(label, options, key):
        """Arama yapılabilir multiselect"""
        # Search box
        search_query = st.text_input(f"{label} Ara", key=f"{key}_search", placeholder="Arama yapın...")
        
        # Filtrelenmiş seçenekler
        if search_query:
            filtered_options = [opt for opt in options if search_query.lower() in str(opt).lower()]
        else:
            filtered_options = options
        
        # "Tümünü Seç" butonu
        col1, col2 = st.columns(2)
        with col1:
            select_all = st.button("✅ Tümü", key=f"{key}_select_all", use_container_width=True)
        with col2:
            select_none = st.button("❌ Hiçbiri", key=f"{key}_select_none", use_container_width=True)
        
        # Multiselect
        if select_all:
            default = filtered_options
        elif select_none:
            default = []
        else:
            default = filtered_options[:min(5, len(filtered_options))] if filtered_options else []
        
        selected = st.multiselect(
            label,
            options=filtered_options,
            default=default,
            key=key
        )
        
        st.caption(f"{len(selected)} / {len(options)} seçildi")
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
                    if filtered_df[col].dtype == 'object':
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
        if 'Sales_2024_range' in filter_config:
            min_val, max_val = filter_config['Sales_2024_range']
            filtered_df = filtered_df[
                (filtered_df['Sales_2024'] >= min_val) & 
                (filtered_df['Sales_2024'] <= max_val)
            ]
        
        if 'Growth_range' in filter_config:
            min_val, max_val = filter_config['Growth_range']
            filtered_df = filtered_df[
                (filtered_df['Growth_23_24'] >= min_val) & 
                (filtered_df['Growth_23_24'] <= max_val)
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
            if 'Sales_2024' in df.columns:
                metrics['Total_Market_Value_2024'] = df['Sales_2024'].sum()
                metrics['Avg_Sales_Per_Product'] = df['Sales_2024'].mean()
                metrics['Median_Sales_2024'] = df['Sales_2024'].median()
                metrics['Sales_Std_Dev'] = df['Sales_2024'].std()
                
                # Çeyreklikler
                metrics['Sales_Q1'] = df['Sales_2024'].quantile(0.25)
                metrics['Sales_Q3'] = df['Sales_2024'].quantile(0.75)
                metrics['Sales_IQR'] = metrics['Sales_Q3'] - metrics['Sales_Q1']
            
            # Büyüme metrikleri
            if 'Growth_23_24' in df.columns:
                metrics['Avg_Growth_Rate'] = df['Growth_23_24'].mean()
                metrics['Growth_Std_Dev'] = df['Growth_23_24'].std()
                metrics['Positive_Growth_Products'] = (df['Growth_23_24'] > 0).sum()
                metrics['Negative_Growth_Products'] = (df['Growth_23_24'] < 0).sum()
                metrics['High_Growth_Products'] = (df['Growth_23_24'] > 20).sum()
            
            # Pazar konsantrasyonu
            if 'Corporation' in df.columns and 'Sales_2024' in df.columns:
                corp_sales = df.groupby('Corporation')['Sales_2024'].sum().sort_values(ascending=False)
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
                if 'Sales_2024' in df.columns:
                    mol_sales = df.groupby('Molecule')['Sales_2024'].sum()
                    total_mol_sales = mol_sales.sum()
                    if total_mol_sales > 0:
                        metrics['Top_10_Molecule_Share'] = mol_sales.nlargest(10).sum() / total_mol_sales * 100
                        metrics['Gini_Coefficient_Molecules'] = AdvancedPharmaAnalytics.calculate_gini(mol_sales.values)
            
            # Coğrafi dağılım
            if 'Country' in df.columns:
                metrics['Country_Coverage'] = df['Country'].nunique()
                if 'Sales_2024' in df.columns:
                    country_sales = df.groupby('Country')['Sales_2024'].sum()
                    metrics['Top_5_Country_Share'] = country_sales.nlargest(5).sum() / country_sales.sum() * 100
            
            # Fiyat analizleri
            if 'Avg_Price_2024' in df.columns:
                metrics['Avg_Price_2024'] = df['Avg_Price_2024'].mean()
                metrics['Price_Variance'] = df['Avg_Price_2024'].var()
                metrics['Price_CV'] = (df['Avg_Price_2024'].std() / df['Avg_Price_2024'].mean()) * 100 if df['Avg_Price_2024'].mean() > 0 else 0
                
                # Price segments
                price_quartiles = df['Avg_Price_2024'].quantile([0.25, 0.5, 0.75])
                metrics['Price_Q1'] = price_quartiles[0.25]
                metrics['Price_Median'] = price_quartiles[0.5]
                metrics['Price_Q3'] = price_quartiles[0.75]
            
            # CAGR analizi
            if 'CAGR_3Y' in df.columns:
                metrics['Avg_CAGR_3Y'] = df['CAGR_3Y'].mean()
                metrics['High_CAGR_Products'] = (df['CAGR_3Y'] > 15).sum()
            
            # Veri kalitesi
            metrics['Total_Rows'] = len(df)
            metrics['Total_Columns'] = len(df.columns)
            metrics['Missing_Values'] = df.isnull().sum().sum()
            metrics['Missing_Percentage'] = (metrics['Missing_Values'] / (len(df) * len(df.columns))) * 100
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_gini(array):
        """Gini katsayısını hesapla"""
        array = array.flatten()
        if np.amin(array) < 0:
            array -= np.amin(array)
        array = array + 0.0000001
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))
    
    @staticmethod
    def analyze_market_trends(df, period='yearly'):
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
            
            # Aylık/çeyreklik trendler (varsa)
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'month' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    monthly_trend = df.groupby(df[date_col].dt.to_period('M'))['Sales_2024'].sum()
                    trends['Monthly_Trend'] = monthly_trend
                except:
                    pass
            
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
            sales_features = [col for col in df.columns if 'Sales_' in col]
            if sales_features:
                features.extend(sales_features[-2:])  # Son 2 yıl
            
            # Büyüme özellikleri
            if 'Growth_23_24' in df.columns:
                features.append('Growth_23_24')
            
            # Fiyat özellikleri
            price_features = [col for col in df.columns if 'Avg_Price' in col]
            if price_features:
                features.append(price_features[-1])
            
            if len(features) < 2:
                st.warning("Segmentasyon için yeterli özellik bulunamadı")
                return None
            
            # Veriyi hazırla
            segmentation_data = df[features].fillna(0)
            
            # Outlier'ları temizle
            Q1 = segmentation_data.quantile(0.25)
            Q3 = segmentation_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = ~((segmentation_data < (Q1 - 1.5 * IQR)) | (segmentation_data > (Q3 + 1.5 * IQR))).any(axis=1)
            segmentation_data = segmentation_data[outlier_mask]
            
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
                silhouette = silhouette_score(features_scaled, clusters)
                calinski = calinski_harabasz_score(features_scaled, clusters)
            else:
                silhouette = None
                calinski = None
            
            # Sonuçları birleştir
            result_df = df.loc[outlier_mask].copy()
            result_df['Segment'] = clusters
            
            # Segment profilleri
            segment_profiles = result_df.groupby('Segment').agg({
                'Sales_2024': ['mean', 'median', 'sum', 'count'],
                'Growth_23_24': ['mean', 'median'],
                'Avg_Price_2024': ['mean', 'median'] if 'Avg_Price_2024' in df.columns else None
            }).round(2)
            
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
                'profiles': segment_profiles,
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
            # 1. Yüksek büyüyen ama düşük market share olan ürünler
            if 'Growth_23_24' in df.columns and 'Market_Share_2024' in df.columns:
                high_growth_low_share = df[
                    (df['Growth_23_24'] > 30) & 
                    (df['Market_Share_2024'] < 1)
                ]
                
                if len(high_growth_low_share) > 0:
                    insights.append({
                        'type': 'opportunity',
                        'title': '🚀 Yüksek Büyüme Potansiyeli',
                        'description': f"{len(high_growth_low_share)} ürün %30+ büyüme oranına sahip ama < %1 pazar payında. Yatırım fırsatı!",
                        'data': high_growth_low_share.head(10)
                    })
            
            # 2. Yüksek fiyat ama yüksek büyüme
            if 'Avg_Price_2024' in df.columns and 'Growth_23_24' in df.columns:
                premium_growth = df[
                    (df['Avg_Price_2024'] > df['Avg_Price_2024'].quantile(0.75)) &
                    (df['Growth_23_24'] > 20)
                ]
                
                if len(premium_growth) > 0:
                    insights.append({
                        'type': 'premium',
                        'title': '💰 Premium Büyüme Segmenti',
                        'description': f"{len(premium_growth)} premium ürün %20+ büyüme gösteriyor. Fiyat esnekliği yüksek.",
                        'data': premium_growth.head(10)
                    })
            
            # 3. Büyük pazar payı ama negatif büyüme
            if 'Market_Share_2024' in df.columns and 'Growth_23_24' in df.columns:
                declining_giants = df[
                    (df['Market_Share_2024'] > 5) &
                    (df['Growth_23_24'] < 0)
                ]
                
                if len(declining_giants) > 0:
                    insights.append({
                        'type': 'warning',
                        'title': '⚠️ Olgun Ürünlerde Düşüş',
                        'description': f"{len(declining_giants)} büyük pazar paylı ürün negatif büyüme gösteriyor. Yeniden konumlandırma gerekli.",
                        'data': declining_giants.head(10)
                    })
            
            # 4. Coğrafi fırsatlar
            if 'Country' in df.columns and 'Growth_23_24' in df.columns:
                country_growth = df.groupby('Country')['Growth_23_24'].mean().nlargest(5)
                if len(country_growth) > 0:
                    insights.append({
                        'type': 'geographic',
                        'title': '🌍 En Hızlı Büyüyen Pazarlar',
                        'description': f"En yüksek büyüme oranları: {', '.join([f'{c} (%{g:.1f})' for c, g in country_growth.items()])}",
                        'data': None
                    })
            
            # 5. Fiyat-hacim ilişkisi
            if 'Avg_Price_2024' in df.columns and 'Units_2024' in df.columns:
                correlation = df[['Avg_Price_2024', 'Units_2024']].corr().iloc[0, 1]
                
                if correlation < -0.3:
                    insights.append({
                        'type': 'price_elasticity',
                        'title': '📉 Yüksek Fiyat Esnekliği',
                        'description': f"Fiyat-hacim korelasyonu: {correlation:.2f}. Fiyat artışları satış hacmini önemli ölçüde düşürüyor.",
                        'data': None
                    })
                elif correlation > 0.3:
                    insights.append({
                        'type': 'veblen',
                        'title': '📈 Veblen Etkisi',
                        'description': f"Fiyat-hacim korelasyonu: {correlation:.2f}. Yüksek fiyat daha yüksek talep anlamına geliyor (premium etkisi).",
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
                total_sales = metrics.get('Total_Market_Value_2024', 0)
                st.markdown(f"""
                <div class="custom-metric-card premium">
                    <div class="custom-metric-label">TOPLAM PAZAR DEĞERİ</div>
                    <div class="custom-metric-value">${total_sales/1e9:.2f}B</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">2024</span>
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
                        <span>2023 → 2024</span>
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
                avg_price = metrics.get('Avg_Price_2024', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">ORTALAMA FİYAT</div>
                    <div class="custom-metric-value">${avg_price:.2f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">2024</span>
                        <span>Per Unit</span>
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
            if 'Corporation' in df.columns and 'Sales_2024' in df.columns:
                # Şirket bazlı pazar payı
                company_sales = df.groupby('Corporation')['Sales_2024'].sum().sort_values(ascending=False)
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
                        x=top_companies.values,
                        y=top_companies.index,
                        orientation='h',
                        marker_color='#3b82f6',
                        text=[f'${x/1e6:.1f}M' for x in top_companies.values],
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
            if 'Country' in df.columns and 'Sales_2024' in df.columns:
                # Ülke bazlı satışlar
                country_sales = df.groupby('Country')['Sales_2024'].sum().reset_index()
                country_sales = country_sales.sort_values('Sales_2024', ascending=False)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Coğrafi Satış Dağılımı', 'Top 15 Ülke',
                                   'Bölgesel Konsantrasyon', 'Satış Yoğunluğu'),
                    specs=[[{'type': 'choropleth'}, {'type': 'bar'}],
                           [{'type': 'treemap'}, {'type': 'densitymapbox'}]],
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )
                
                # Choropleth map
                fig.add_trace(
                    go.Choropleth(
                        locations=country_sales['Country'],
                        locationmode='country names',
                        z=country_sales['Sales_2024'],
                        colorscale='Blues',
                        colorbar_title="Satış (USD)",
                        hoverinfo='location+z'
                    ),
                    row=1, col=1
                )
                
                # Top 15 ülke bar chart
                top_countries = country_sales.head(15)
                fig.add_trace(
                    go.Bar(
                        x=top_countries['Sales_2024'],
                        y=top_countries['Country'],
                        orientation='h',
                        marker_color='#8b5cf6',
                        text=[f'${x/1e6:.1f}M' for x in top_countries['Sales_2024']],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
                
                # Treemap (bölgesel konsantrasyon)
                if 'Region' in df.columns:
                    region_sales = df.groupby('Region')['Sales_2024'].sum().reset_index()
                    fig.add_trace(
                        go.Treemap(
                            labels=region_sales['Region'],
                            parents=[''] * len(region_sales),
                            values=region_sales['Sales_2024'],
                            textinfo="label+value+percent parent",
                            marker_colorscale='Viridis'
                        ),
                        row=2, col=1
                    )
                
                # Density mapbox (varsa koordinatlar)
                fig.update_layout(
                    height=800,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9',
                    showlegend=False,
                    title_text="Coğrafi Analiz",
                    title_x=0.5,
                    geo=dict(
                        bgcolor='rgba(0,0,0,0)',
                        lakecolor='rgba(0,0,0,0)',
                        landcolor='rgba(100,100,100,0.2)',
                        projection_type='natural earth'
                    )
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
            if 'Avg_Price_2024' in df.columns and 'Units_2024' in df.columns:
                # Sample veri (büyük datasetler için)
                sample_df = df[
                    (df['Avg_Price_2024'] > 0) & 
                    (df['Units_2024'] > 0)
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
                        x=sample_df['Avg_Price_2024'],
                        y=sample_df['Units_2024'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=sample_df['Growth_23_24'] if 'Growth_23_24' in sample_df.columns else sample_df['Sales_2024'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Büyüme %" if 'Growth_23_24' in sample_df.columns else "Satış")
                        ),
                        text=sample_df['Molecule'] if 'Molecule' in sample_df.columns else None,
                        hoverinfo='text+x+y'
                    ),
                    row=1, col=1
                )
                
                # Fiyat dağılımı
                fig.add_trace(
                    go.Histogram(
                        x=df['Avg_Price_2024'],
                        nbinsx=50,
                        marker_color='#3b82f6',
                        name='Fiyat Dağılımı'
                    ),
                    row=1, col=2
                )
                
                # Hacim dağılımı
                fig.add_trace(
                    go.Histogram(
                        x=df['Units_2024'],
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
                            y=company_data['Avg_Price_2024'],
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
            
            return None
            
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
                if 'Corporation' in df.columns and 'Sales_2024' in df.columns:
                    market_share = df.groupby('Corporation')['Sales_2024'].sum().sort_values(ascending=False)
                    market_share_df = market_share.reset_index()
                    market_share_df.columns = ['ŞİRKET', 'SATIŞ_2024']
                    market_share_df['PAY (%)'] = (market_share_df['SATIŞ_2024'] / market_share_df['SATIŞ_2024'].sum()) * 100
                    market_share_df['KÜMÜLATİF_PAY'] = market_share_df['PAY (%)'].cumsum()
                    market_share_df.to_excel(writer, sheet_name='PAZAR_PAYI', index=False)
                
                # 4. Ülke analizi
                if 'Country' in df.columns:
                    country_analysis = df.groupby('Country').agg({
                        'Sales_2024': ['sum', 'mean', 'count'],
                        'Growth_23_24': 'mean' if 'Growth_23_24' in df.columns else None
                    }).round(2)
                    country_analysis.to_excel(writer, sheet_name='ULKE_ANALIZI')
                
                # 5. Molekül analizi
                if 'Molecule' in df.columns:
                    molecule_analysis = df.groupby('Molecule').agg({
                        'Sales_2024': ['sum', 'mean', 'count'],
                        'Avg_Price_2024': 'mean' if 'Avg_Price_2024' in df.columns else None
                    }).round(2)
                    molecule_analysis.nlargest(50, ('Sales_2024', 'sum')).to_excel(
                        writer, sheet_name='MOLEKUL_ANALIZI'
                    )
                
                # 6. İçgörüler
                insights_data = []
                for insight in insights:
                    insights_data.append({
                        'TİP': insight['type'],
                        'BAŞLIK': insight['title'],
                        'AÇIKLAMA': insight['description']
                    })
                
                if insights_data:
                    insights_df = pd.DataFrame(insights_data)
                    insights_df.to_excel(writer, sheet_name='STRATEJIK_ICGORULER', index=False)
                
                # 7. Büyüme analizi
                if 'Growth_23_24' in df.columns:
                    growth_analysis = pd.DataFrame({
                        'KATEGORİ': ['< -20%', '-20% to 0%', '0% to 20%', '20% to 50%', '> 50%'],
                        'ÜRÜN_SAYISI': [
                            (df['Growth_23_24'] < -20).sum(),
                            ((df['Growth_23_24'] >= -20) & (df['Growth_23_24'] < 0)).sum(),
                            ((df['Growth_23_24'] >= 0) & (df['Growth_23_24'] < 20)).sum(),
                            ((df['Growth_23_24'] >= 20) & (df['Growth_23_24'] < 50)).sum(),
                            (df['Growth_23_24'] >= 50).sum()
                        ]
                    })
                    growth_analysis.to_excel(writer, sheet_name='BUYUME_ANALIZI', index=False)
                
                writer.save()
            
            output.seek(0)
            return output
            
        except Exception as e:
            st.error(f"Excel rapor oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def generate_pdf_summary(df, metrics, insights):
        """PDF özet raporu oluştur (basit versiyon)"""
        try:
            summary = f"""
            PHARMAINTELLIGENCE PRO - PAZAR ANALİZ RAPORU
            Oluşturulma Tarihi: {datetime.now().strftime('%d/%m/%Y %H:%M')}
            
            GENEL BAKIŞ
            ----------
            Toplam Ürün Sayısı: {metrics.get('Total_Rows', 0):,}
            Toplam Pazar Değeri (2024): ${metrics.get('Total_Market_Value_2024', 0)/1e9:.2f}B
            Ortalama Büyüme Oranı: {metrics.get('Avg_Growth_Rate', 0):.1f}%
            Pazar Konsantrasyonu (HHI): {metrics.get('HHI_Index', 0):.0f}
            
            ANA PERFORMANS GÖSTERGELERİ
            ---------------------------
            • Top 3 Şirket Pazar Payı: {metrics.get('Top_3_Share', 0):.1f}%
            • Molekül Çeşitliliği: {metrics.get('Unique_Molecules', 0):,}
            • Coğrafi Kapsam: {metrics.get('Country_Coverage', 0)} ülke
            • Ortalama Fiyat: ${metrics.get('Avg_Price_2024', 0):.2f}
            
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
            4. Fiyat esnekliğine göre strateji belirleyin
            
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
                
                visualization_quality = st.select_slider(
                    "Görselleştirme Kalitesi",
                    options=['Hızlı', 'Orta', 'Yüksek'],
                    value='Orta'
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
                elif insight['type'] == 'opportunity':
                    icon = "🚀"
                elif insight['type'] == 'premium':
                    icon = "💰"
                elif insight['type'] == 'geographic':
                    icon = "🌍"
                
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <div class="insight-icon">{icon}</div>
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-content">{insight['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if insight['data'] is not None and not insight['data'].empty:
                    with st.expander("📋 Detaylı Liste"):
                        st.dataframe(
                            insight['data'][['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_23_24']].head(10),
                            use_container_width=True
                        )
    else:
        st.info("Henüz stratejik içgörü tespit edilmedi. Verilerinizi analiz ediyoruz...")
    
    # Hızlı Veri Önizleme
    st.markdown('<h3 class="subsection-title">📋 Veri Önizleme</h3>', unsafe_allow_html=True)
    
    preview_col1, preview_col2 = st.columns([1, 3])
    
    with preview_col1:
        rows_to_show = st.slider("Gösterilecek Satır Sayısı", 10, 1000, 100, 10)
        show_columns = st.multiselect(
            "Gösterilecek Sütunlar",
            options=df.columns.tolist(),
            default=['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_23_24'][:min(5, len(df.columns))]
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
        status_color = "success" if missing_pct < 5 else "warning" if missing_pct < 20 else "danger"
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
        col1, col2 = st.columns(2)
        
        with col1:
            # Top moleküller
            top_molecules = df.groupby('Molecule')['Sales_2024'].sum().nlargest(15)
            fig = px.bar(
                top_molecules,
                orientation='h',
                title='Top 15 Molekül - Satış Performansı',
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
            if 'Growth_23_24' in df.columns:
                molecule_growth = df.groupby('Molecule')['Growth_23_24'].mean().nlargest(15)
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
    
    if 'Avg_Price_2024' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Fiyat segmentleri
            price_segments = pd.cut(
                df['Avg_Price_2024'],
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
            if 'Growth_23_24' in df.columns:
                df['Price_Segment'] = price_segments
                segment_growth = df.groupby('Price_Segment')['Growth_23_24'].mean().dropna()
                
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
    
    if 'Avg_Price_2024' in df.columns and 'Units_2024' in df.columns:
        # Korelasyon analizi
        correlation = df[['Avg_Price_2024', 'Units_2024']].corr().iloc[0, 1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fiyat-Hacim Korelasyonu", f"{correlation:.3f}")
        
        with col2:
            elasticity_status = "Yüksek Esneklik" if correlation < -0.3 else "Düşük Esneklik" if correlation > 0.3 else "Nötr"
            st.metric("Esneklik Durumu", elasticity_status)
        
        with col3:
            recommendation = "Fiyat Artışı Riskli" if correlation < -0.3 else "Fiyat Artışı Mümkün" if correlation > 0.3 else "Limitli Fiyat Artışı"
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
    
    # Rekabet metrikleri
    st.markdown('<h3 class="subsection-title">📊 Rekabet Yoğunluğu Metrikleri</h3>', unsafe_allow_html=True)
    
    comp_cols = st.columns(4)
    
    with comp_cols[0]:
        hhi = metrics.get('HHI_Index', 0)
        hhi_status = "Monopolistik" if hhi > 2500 else "Oligopol" if hhi > 1800 else "Rekabetçi"
        st.metric("HHI İndeksi", f"{hhi:.0f}", hhi_status)
    
    with comp_cols[1]:
        top3_share = metrics.get('Top_3_Share', 0)
        concentration = "Yüksek" if top3_share > 50 else "Orta" if top3_share > 30 else "Düşük"
        st.metric("Top 3 Payı", f"{top3_share:.1f}%", concentration)
    
    with comp_cols[2]:
        cr4 = metrics.get('Top_4_Share', 0)
        st.metric("CR4 Oranı", f"{cr4:.1f}%")
    
    with comp_cols[3]:
        gini = metrics.get('Gini_Coefficient_Molecules', 0)
        inequality = "Yüksek" if gini > 0.6 else "Orta" if gini > 0.4 else "Düşük"
        st.metric("Gini Katsayısı", f"{gini:.3f}", inequality)
    
    # Şirket performans karşılaştırması
    st.markdown('<h3 class="subsection-title">📈 Şirket Performans Analizi</h3>', unsafe_allow_html=True)
    
    if 'Corporation' in df.columns:
        # Performans matrisi
        company_metrics = df.groupby('Corporation').agg({
            'Sales_2024': ['sum', 'mean', 'count'],
            'Growth_23_24': 'mean',
            'Avg_Price_2024': 'mean'
        }).round(2)
        
        company_metrics.columns = ['_'.join(col).strip() for col in company_metrics.columns.values]
        company_metrics = company_metrics.sort_values('Sales_2024_sum', ascending=False)
        
        # Top 20 şirket
        top_companies = company_metrics.head(20)
        
        # Heatmap
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
        n_clusters = st.slider("Segment Sayısı", 2, 8, 4)
        method = st.selectbox("Segmentasyon Metodu", ['kmeans', 'dbscan'])
        
        if st.button("🔍 Segmentasyon Analizi Yap", type="primary", use_container_width=True):
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
            
            # Segment metrikleri
            st.markdown("**Segment Performans Metrikleri:**")
            st.dataframe(
                results['profiles'],
                use_container_width=True
            )
            
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
        opportunity_insights = [i for i in insights if i['type'] in ['opportunity', 'premium']]
        
        if opportunity_insights:
            for insight in opportunity_insights[:3]:
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-content">{insight['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if insight['data'] is not None and not insight['data'].empty:
                    with st.expander("🚀 Bu Fırsattaki Ürünler"):
                        st.dataframe(
                            insight['data'][['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_23_24']],
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
        horizontal=True
    )
    
    # Rapor parametreleri
    with st.expander("⚙️ Rapor Ayarları"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_charts = st.checkbox("Grafikleri Dahil Et", value=True)
            include_insights = st.checkbox("Stratejik İçgörüleri Dahil Et", value=True)
        
        with col2:
            data_since = st.date_input(
                "Veri Başlangıç Tarihi",
                value=datetime.now() - timedelta(days=365)
            )
            report_language = st.selectbox("Rapor Dili", ['Türkçe', 'İngilizce'])
    
    # Rapor oluşturma
    st.markdown('<h3 class="subsection-title">🛠️ Rapor Oluşturma</h3>', unsafe_allow_html=True)
    
    report_cols = st.columns(3)
    
    with report_cols[0]:
        if st.button("📈 Excel Raporu Oluştur", use_container_width=True):
            with st.spinner("Excel raporu oluşturuluyor..."):
                reporting = ProfessionalReporting()
                excel_report = reporting.generate_excel_report(df, metrics, insights)
                
                if excel_report:
                    st.download_button(
                        label="⬇️ Excel İndir",
                        data=excel_report,
                        file_name=f"pharma_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
    
    with report_cols[1]:
        if st.button("📄 PDF Özet Oluştur", use_container_width=True):
            with st.spinner("PDF özeti oluşturuluyor..."):
                reporting = ProfessionalReporting()
                pdf_summary = reporting.generate_pdf_summary(df, metrics, insights)
                
                st.download_button(
                    label="⬇️ PDF İndir",
                    data=pdf_summary.encode('utf-8'),
                    file_name=f"pharma_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    with report_cols[2]:
        if st.button("🔄 Analizi Sıfırla", use_container_width=True):
            st.session_state.df = None
            st.session_state.filtered_df = None
            st.session_state.metrics = None
            st.session_state.insights = []
            st.session_state.current_filters = {}
            st.rerun()
    
    # Hızlı İstatistikler
    st.markdown('<h3 class="subsection-title">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)
    
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with stat_cols[1]:
        st.metric("Toplam Sütun", len(df.columns))
    
    with stat_cols[2]:
        st.metric("Bellek Kullanımı", f"{df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
    
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
        try:
            main()
        except Exception as e:
            st.error(f"Uygulama hatası: {str(e)}")
            st.error("Detaylı hata bilgisi:")
            st.code(traceback.format_exc())
            
            # Yeniden deneme butonu
            if st.button("🔄 Sayfayı Yenile"):
                st.rerun()
    finally:
        # Cleanup
        gc.collect()
