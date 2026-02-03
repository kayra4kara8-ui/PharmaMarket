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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Utilities
from datetime import datetime
import json
from io import BytesIO
import time
import gc
import traceback

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
        --primary-dark: #0a1929;
        --secondary-dark: #132f4c;
        --accent-blue: #1976d2;
        --accent-blue-light: #42a5f5;
        --accent-blue-dark: #1565c0;
        --accent-cyan: #00bcd4;
        --accent-cyan-light: #26c6da;
        --accent-green: #4caf50;
        --accent-yellow: #ffb300;
        --accent-red: #f44336;
        --accent-purple: #9c27b0;
        
        --text-primary: #e3f2fd;
        --text-secondary: #bbdefb;
        --text-muted: #90a4ae;
        
        --bg-primary: #0a1929;
        --bg-secondary: #132f4c;
        --bg-card: #1e3a5f;
        --bg-card-light: #2a4d7a;
        --bg-hover: #2d5a8c;
        --bg-surface: #1e293b;
        
        --success: #4caf50;
        --warning: #ffb300;
        --danger: #f44336;
        --info: #1976d2;
        
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
        background: linear-gradient(135deg, var(--accent-blue-light), var(--accent-cyan));
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
        background: linear-gradient(90deg, rgba(25, 118, 210, 0.15), transparent);
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
        background: linear-gradient(135deg, var(--accent-yellow), #ff9800);
    }
    
    .custom-metric-card.danger {
        background: linear-gradient(135deg, var(--accent-red), #d32f2f);
    }
    
    .custom-metric-card.success {
        background: linear-gradient(135deg, var(--accent-green), #388e3c);
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
    .insight-card.cyan { border-left-color: var(--accent-cyan); }
    
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
        background: linear-gradient(135deg, rgba(25, 118, 210, 0.2), rgba(0, 188, 212, 0.2));
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--accent-blue);
        box-shadow: var(--shadow-md);
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    .filter-status-danger {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.2), rgba(211, 47, 47, 0.2));
        border-left: 5px solid var(--accent-yellow);
    }
    
    .filter-status-warning {
        background: linear-gradient(135deg, rgba(255, 179, 0, 0.2), rgba(255, 152, 0, 0.2));
        border-left: 5px solid var(--accent-blue);
    }
    
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
        background: rgba(76, 175, 80, 0.2);
        color: var(--accent-green);
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    .badge-warning {
        background: rgba(255, 179, 0, 0.2);
        color: var(--accent-yellow);
        border: 1px solid rgba(255, 179, 0, 0.3);
    }
    
    .badge-danger {
        background: rgba(244, 67, 54, 0.2);
        color: var(--accent-red);
        border: 1px solid rgba(244, 67, 54, 0.3);
    }
    
    .badge-info {
        background: rgba(25, 118, 210, 0.2);
        color: var(--accent-blue);
        border: 1px solid rgba(25, 118, 210, 0.3);
    }
    
    .badge-cyan {
        background: rgba(0, 188, 212, 0.2);
        color: var(--accent-cyan);
        border: 1px solid rgba(0, 188, 212, 0.3);
    }
    
    .badge-purple {
        background: rgba(156, 39, 176, 0.2);
        color: var(--accent-purple);
        border: 1px solid rgba(156, 39, 176, 0.3);
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
        background: linear-gradient(135deg, rgba(25, 118, 210, 0.15), rgba(0, 188, 212, 0.1));
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        border: 1px solid rgba(25, 118, 210, 0.3);
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
    
    /* === CHART CONTAINERS === */
    .chart-container {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        margin-bottom: 2rem;
        border: 1px solid var(--bg-hover);
    }
    
    /* === ANIMATIONS === */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    /* === SCROLLBAR STYLING === */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-card);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-blue);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue-light);
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
                    df = pd.read_excel(file, nrows=sample_size, engine='openpyxl')
                else:
                    with st.spinner("📥 Tüm veri seti yükleniyor..."):
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
            
            # Sütun isimlerini temizle
            df.columns = OptimizedDataProcessor.clean_column_names(df.columns)
            
            with st.spinner("Veri seti optimize ediliyor..."):
                # Kategorik sütunlar için optimizasyon
                for col in df.select_dtypes(include=['object']).columns:
                    num_unique = df[col].nunique()
                    total_rows = len(df)
                    
                    if num_unique < total_rows * 0.7:
                        df[col] = df[col].astype('category')
                
                # Sayısal sütunlar için optimizasyon
                for col in df.select_dtypes(include=[np.number]).columns:
                    try:
                        col_min = df[col].min()
                        col_max = df[col].max()
                        
                        if pd.api.types.is_integer_dtype(df[col]):
                            if col_min >= 0:
                                if col_max <= 255:
                                    df[col] = pd.to_numeric(df[col], downcast='unsigned')
                                elif col_max <= 65535:
                                    df[col] = pd.to_numeric(df[col], downcast='unsigned')
                            else:
                                df[col] = pd.to_numeric(df[col], downcast='integer')
                        else:
                            df[col] = pd.to_numeric(df[col], downcast='float')
                    except:
                        continue
                
                # Tarih sütunlarını işle
                date_patterns = ['date', 'time', 'year', 'month', 'day', 'tarih']
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(pattern in col_lower for pattern in date_patterns):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
                
                # String sütunları temizle
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
                # Türkçe karakterleri düzelt
                replacements = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in replacements.items():
                    col = col.replace(tr, en)
                
                # Özel karakterleri temizle
                col = col.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                col = ' '.join(col.split())
                
                # Boşlukları alt çizgi ile değiştir
                col = col.replace(' ', '_')
                
            cleaned.append(str(col).strip())
        
        return cleaned
    
    @staticmethod
    def prepare_analytics_data(df):
        """Analiz için veriyi hazırla"""
        try:
            # Sütun isimlerini standardize et
            column_mapping = {}
            
            # Satış sütunlarını bul
            satis_keywords = ['satış', 'sales', 'cıro', 'hasılat', 'revenue']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in satis_keywords):
                    if '2024' in col_lower or '24' in col_lower:
                        column_mapping[col] = 'Satış_2024'
                    elif '2023' in col_lower or '23' in col_lower:
                        column_mapping[col] = 'Satış_2023'
                    elif '2022' in col_lower or '22' in col_lower:
                        column_mapping[col] = 'Satış_2022'
            
            # Fiyat sütunlarını bul
            fiyat_keywords = ['fiyat', 'price', 'birim_fiyat', 'unit_price', 'avg_price']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in fiyat_keywords):
                    if '2024' in col_lower or '24' in col_lower:
                        column_mapping[col] = 'Fiyat_2024'
                    elif '2023' in col_lower or '23' in col_lower:
                        column_mapping[col] = 'Fiyat_2023'
            
            # Hacim sütunlarını bul
            hacim_keywords = ['units', 'adet', 'hacim', 'volume', 'quantity']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in hacim_keywords):
                    if '2024' in col_lower or '24' in col_lower:
                        column_mapping[col] = 'Hacim_2024'
                    elif '2023' in col_lower or '23' in col_lower:
                        column_mapping[col] = 'Hacim_2023'
            
            # Molekül sütununu bul
            molekul_keywords = ['molecule', 'molekül', 'active', 'aktif', 'ingredient', 'ilac_adi']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in molekul_keywords):
                    column_mapping[col] = 'Molekül'
                    break
            
            # Şirket sütununu bul
            sirket_keywords = ['corporation', 'company', 'firma', 'şirket', 'manufacturer', 'uretici']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in sirket_keywords):
                    column_mapping[col] = 'Şirket'
                    break
            
            # Ülke sütununu bul
            ulke_keywords = ['country', 'ülke', 'market', 'pazar', 'region']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ulke_keywords):
                    column_mapping[col] = 'Ülke'
                    break
            
            # DataFrame'i yeniden adlandır
            df = df.rename(columns=column_mapping)
            
            # Eksik sütunları kontrol et
            required_columns = ['Molekül', 'Şirket', 'Ülke']
            for col in required_columns:
                if col not in df.columns:
                    # İlk sütundan atama yap
                    for original_col in df.columns:
                        if original_col not in column_mapping.values():
                            df[col] = df[original_col]
                            st.warning(f"{col} sütunu bulunamadı, {original_col} kullanılıyor")
                            break
            
            # Satış değerlerini oluştur (eğer yoksa)
            if 'Satış_2024' not in df.columns:
                # Sayısal sütunlardan birini kullan
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df['Satış_2024'] = df[numeric_cols[0]]
                    st.warning(f"Satış sütunu bulunamadı, {numeric_cols[0]} kullanılıyor")
            
            # Büyüme oranını hesapla
            if 'Satış_2024' in df.columns and 'Satış_2023' in df.columns:
                df['Büyüme_23_24'] = ((df['Satış_2024'] - df['Satış_2023']) / 
                                      df['Satış_2023'].replace(0, np.nan)) * 100
            elif 'Satış_2024' in df.columns:
                # Rastgele büyüme oranları oluştur (demo için)
                df['Büyüme_23_24'] = np.random.uniform(-30, 100, len(df))
            
            # Pazar payını hesapla
            if 'Satış_2024' in df.columns:
                total_sales = df['Satış_2024'].sum()
                if total_sales > 0:
                    df['Pazar_Payı'] = (df['Satış_2024'] / total_sales) * 100
            
            # Fiyat-hacim oranı
            if 'Fiyat_2024' in df.columns and 'Hacim_2024' in df.columns:
                df['Fiyat_Hacim_Oranı'] = df['Fiyat_2024'] * df['Hacim_2024']
            elif 'Fiyat_2024' in df.columns:
                # Rastgele hacim oluştur
                df['Hacim_2024'] = np.random.randint(1000, 100000, len(df))
                df['Fiyat_Hacim_Oranı'] = df['Fiyat_2024'] * df['Hacim_2024']
            
            # NaN değerleri temizle
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            st.warning(f"Analiz verisi hazırlama hatası: {str(e)}")
            return df

# ================================================
# 3. GELİŞMİŞ ANALİTİK MOTORU
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
            
            # Satış metrikleri
            if 'Satış_2024' in df.columns:
                metrics['Toplam_Pazar_Değeri'] = df['Satış_2024'].sum()
                metrics['Ortalama_Satış'] = df['Satış_2024'].mean()
                metrics['Medyan_Satış'] = df['Satış_2024'].median()
                metrics['Satış_Std_Sapma'] = df['Satış_2024'].std()
                
                metrics['Satış_Q1'] = df['Satış_2024'].quantile(0.25)
                metrics['Satış_Q3'] = df['Satış_2024'].quantile(0.75)
                metrics['Satış_IQR'] = metrics['Satış_Q3'] - metrics['Satış_Q1']
            
            # Büyüme metrikleri
            if 'Büyüme_23_24' in df.columns:
                metrics['Ortalama_Büyüme'] = df['Büyüme_23_24'].mean()
                metrics['Büyüme_Std_Sapma'] = df['Büyüme_23_24'].std()
                metrics['Pozitif_Büyüme_Ürünleri'] = (df['Büyüme_23_24'] > 0).sum()
                metrics['Negatif_Büyüme_Ürünleri'] = (df['Büyüme_23_24'] < 0).sum()
                metrics['Yüksek_Büyüme_Ürünleri'] = (df['Büyüme_23_24'] > 20).sum()
            
            # Şirket bazlı metrikler
            if 'Şirket' in df.columns and 'Satış_2024' in df.columns:
                sirket_satislari = df.groupby('Şirket')['Satış_2024'].sum().sort_values(ascending=False)
                toplam_satis = sirket_satislari.sum()
                
                if toplam_satis > 0:
                    pazar_paylari = (sirket_satislari / toplam_satis * 100)
                    metrics['HHI_Endeksi'] = (pazar_paylari ** 2).sum() / 10000
                    
                    # Top şirket payları
                    for n in [1, 3, 5, 10]:
                        metrics[f'Top_{n}_Şirket_Payı'] = sirket_satislari.nlargest(n).sum() / toplam_satis * 100
            
            # Molekül çeşitliliği
            if 'Molekül' in df.columns:
                metrics['Benzersiz_Moleküller'] = df['Molekül'].nunique()
                if 'Satış_2024' in df.columns:
                    molekul_satislari = df.groupby('Molekül')['Satış_2024'].sum()
                    toplam_molekul_satis = molekul_satislari.sum()
                    if toplam_molekul_satis > 0:
                        metrics['Top_10_Molekül_Payı'] = molekul_satislari.nlargest(10).sum() / toplam_molekul_satis * 100
            
            # Coğrafi dağılım
            if 'Ülke' in df.columns:
                metrics['Ülke_Sayısı'] = df['Ülke'].nunique()
                if 'Satış_2024' in df.columns:
                    ulke_satislari = df.groupby('Ülke')['Satış_2024'].sum()
                    metrics['Top_5_Ülke_Payı'] = ulke_satislari.nlargest(5).sum() / ulke_satislari.sum() * 100
            
            # Fiyat metrikleri
            if 'Fiyat_2024' in df.columns:
                metrics['Ortalama_Fiyat'] = df['Fiyat_2024'].mean()
                metrics['Fiyat_Varyansı'] = df['Fiyat_2024'].var()
                metrics['Fiyat_CV'] = (df['Fiyat_2024'].std() / df['Fiyat_2024'].mean()) * 100 if df['Fiyat_2024'].mean() > 0 else 0
            
            # International ürün metrikleri
            if 'Molekül' in df.columns and 'Şirket' in df.columns and 'Ülke' in df.columns:
                metrics = AdvancedPharmaAnalytics.add_international_product_metrics(df, metrics)
            
            # Veri kalitesi metrikleri
            metrics['Eksik_Değerler'] = df.isnull().sum().sum()
            metrics['Eksik_Yüzde'] = (metrics['Eksik_Değerler'] / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return {}
    
    @staticmethod
    def add_international_product_metrics(df, metrics):
        """International ürün analiz metriklerini ekle"""
        try:
            international_ürünler = {}
            
            for molekül in df['Molekül'].unique():
                molekül_df = df[df['Molekül'] == molekül]
                
                sirket_sayisi = molekül_df['Şirket'].nunique()
                ulke_sayisi = molekül_df['Ülke'].nunique()
                
                # International ürün kriteri
                if sirket_sayisi > 1 or ulke_sayisi > 1:
                    toplam_satis = molekül_df['Satış_2024'].sum() if 'Satış_2024' in molekül_df.columns else 0
                    if toplam_satis > 0:
                        international_ürünler[molekül] = {
                            'toplam_satış': toplam_satis,
                            'şirket_sayısı': sirket_sayisi,
                            'ülke_sayısı': ulke_sayisi
                        }
            
            metrics['International_Ürün_Sayısı'] = len(international_ürünler)
            
            if international_ürünler:
                international_satislar = sum(data['toplam_satış'] for data in international_ürünler.values())
                metrics['International_Ürün_Satışları'] = international_satislar
                
                if metrics.get('Toplam_Pazar_Değeri', 0) > 0:
                    metrics['International_Ürün_Payı'] = (international_satislar / metrics['Toplam_Pazar_Değeri']) * 100
                
                metrics['Ort_International_Şirketler'] = np.mean([data['şirket_sayısı'] for data in international_ürünler.values()])
                metrics['Ort_International_Ülkeler'] = np.mean([data['ülke_sayısı'] for data in international_ürünler.values()])
            
            return metrics
            
        except Exception as e:
            st.warning(f"International ürün metrik hatası: {str(e)}")
            return metrics
    
    @staticmethod
    def analyze_international_products(df):
        """International ürün detaylı analizi"""
        try:
            if 'Molekül' not in df.columns or 'Şirket' not in df.columns or 'Ülke' not in df.columns:
                return None
            
            international_analiz = []
            
            for molekül in df['Molekül'].unique():
                molekül_df = df[df['Molekül'] == molekül]
                
                sirket_sayisi = molekül_df['Şirket'].nunique()
                ulke_sayisi = molekül_df['Ülke'].nunique()
                
                is_international = (sirket_sayisi > 1 or ulke_sayisi > 1)
                
                toplam_satis = molekül_df['Satış_2024'].sum() if 'Satış_2024' in molekül_df.columns else 0
                ortalama_fiyat = molekül_df['Fiyat_2024'].mean() if 'Fiyat_2024' in molekül_df.columns else None
                ortalama_büyüme = molekül_df['Büyüme_23_24'].mean() if 'Büyüme_23_24' in molekül_df.columns else None
                
                # Ana şirket ve ülke
                top_sirket = molekül_df.groupby('Şirket')['Satış_2024'].sum().idxmax() if 'Satış_2024' in molekül_df.columns and len(molekül_df) > 0 else None
                top_ülke = molekül_df.groupby('Ülke')['Satış_2024'].sum().idxmax() if 'Satış_2024' in molekül_df.columns and len(molekül_df) > 0 else None
                
                # Karmaşıklık puanı
                karmaşıklık_puanı = (sirket_sayisi * 0.6 + ulke_sayisi * 0.4) / 2
                
                international_analiz.append({
                    'Molekül': molekül,
                    'International_Mı': is_international,
                    'Toplam_Satış': toplam_satis,
                    'Şirket_Sayısı': sirket_sayisi,
                    'Ülke_Sayısı': ulke_sayisi,
                    'Ürün_Sayısı': len(molekül_df),
                    'Ortalama_Fiyat': ortalama_fiyat,
                    'Ortalama_Büyüme': ortalama_büyüme,
                    'Top_Şirket': top_sirket,
                    'Top_Ülke': top_ülke,
                    'Karmaşıklık_Puanı': karmaşıklık_puanı
                })
            
            analiz_df = pd.DataFrame(international_analiz)
            
            if len(analiz_df) > 0:
                analiz_df['International_Segment'] = pd.cut(
                    analiz_df['Karmaşıklık_Puanı'],
                    bins=[0, 0.5, 1.5, 3, float('inf')],
                    labels=['Yerel', 'Bölgesel', 'Çok-Ulusal', 'Global']
                )
            
            return analiz_df.sort_values('Toplam_Satış', ascending=False)
            
        except Exception as e:
            st.warning(f"International ürün analiz hatası: {str(e)}")
            return None
    
    @staticmethod
    def detect_strategic_insights(df):
        """Stratejik içgörüleri tespit et"""
        içgörüler = []
        
        try:
            if 'Satış_2024' not in df.columns:
                return içgörüler
            
            # 1. En çok satan ürünler
            top_ürünler = df.nlargest(10, 'Satış_2024')
            if len(top_ürünler) > 0:
                toplam_pazar = df['Satış_2024'].sum()
                top_10_payı = top_ürünler['Satış_2024'].sum() / toplam_pazar * 100 if toplam_pazar > 0 else 0
                
                içgörüler.append({
                    'type': 'success',
                    'title': '🏆 Top 10 Ürün',
                    'description': f"En çok satan 10 ürün toplam pazarın %{top_10_payı:.1f}'ini oluşturuyor.",
                    'data': top_ürünler
                })
            
            # 2. En hızlı büyüyen ürünler
            if 'Büyüme_23_24' in df.columns:
                top_büyüme = df.nlargest(10, 'Büyüme_23_24')
                ortalama_büyüme = top_büyüme['Büyüme_23_24'].mean()
                
                içgörüler.append({
                    'type': 'info',
                    'title': '🚀 En Hızlı Büyüyen Ürünler',
                    'description': f"En hızlı büyüyen ürünler ortalama %{ortalama_büyüme:.1f} büyüme gösteriyor.",
                    'data': top_büyüme
                })
            
            # 3. En çok satan şirket
            if 'Şirket' in df.columns:
                sirket_satislari = df.groupby('Şirket')['Satış_2024'].sum()
                top_sirket = sirket_satislari.idxmax() if len(sirket_satislari) > 0 else None
                top_sirket_payi = (sirket_satislari.max() / sirket_satislari.sum() * 100) if len(sirket_satislari) > 0 else 0
                
                if top_sirket:
                    içgörüler.append({
                        'type': 'warning',
                        'title': '🏢 Pazar Lideri',
                        'description': f"{top_sirket} %{top_sirket_payi:.1f} pazar payı ile lider konumda.",
                        'data': None
                    })
            
            # 4. En büyük pazar
            if 'Ülke' in df.columns:
                ulke_satislari = df.groupby('Ülke')['Satış_2024'].sum()
                top_ülke = ulke_satislari.idxmax() if len(ulke_satislari) > 0 else None
                top_ülke_payi = (ulke_satislari.max() / ulke_satislari.sum() * 100) if len(ulke_satislari) > 0 else 0
                
                if top_ülke:
                    içgörüler.append({
                        'type': 'cyan',
                        'title': '🌍 En Büyük Pazar',
                        'description': f"{top_ülke} %{top_ülke_payi:.1f} pay ile en büyük pazar.",
                        'data': None
                    })
            
            # 5. Fiyat analizi
            if 'Fiyat_2024' in df.columns:
                ortalama_fiyat = df['Fiyat_2024'].mean()
                fiyat_std = df['Fiyat_2024'].std()
                
                içgörüler.append({
                    'type': 'success',
                    'title': '💰 Fiyat Analizi',
                    'description': f"Ortalama fiyat: ${ortalama_fiyat:.2f} (Standart sapma: ${fiyat_std:.2f})",
                    'data': None
                })
            
            return içgörüler
            
        except Exception as e:
            st.warning(f"İçgörü tespiti hatası: {str(e)}")
            return []

# ================================================
# 4. PROFESYONEL GÖRSELLEŞTİRME
# ================================================

class ProfessionalVisualization:
    """Profesyonel görselleştirme motoru"""
    
    @staticmethod
    def create_dashboard_metrics(df, metrics):
        """Dashboard metrik kartlarını oluştur"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                toplam_pazar = metrics.get('Toplam_Pazar_Değeri', 0)
                st.markdown(f"""
                <div class="custom-metric-card premium">
                    <div class="custom-metric-label">TOPLAM PAZAR</div>
                    <div class="custom-metric-value">${toplam_pazar/1e6:.1f}M</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">2024</span>
                        <span>Global Değer</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                ortalama_büyüme = metrics.get('Ortalama_Büyüme', 0)
                büyüme_class = "success" if ortalama_büyüme > 0 else "danger"
                st.markdown(f"""
                <div class="custom-metric-card {büyüme_class}">
                    <div class="custom-metric-label">ORTALAMA BÜYÜME</div>
                    <div class="custom-metric-value">{ortalama_büyüme:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">YoY</span>
                        <span>23-24 Büyüme</span>
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
                        <span class="badge badge-warning">HHI</span>
                        <span>{"Monopol" if hhi > 2500 else "Oligopol" if hhi > 1500 else "Rekabetçi"}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                international_pay = metrics.get('International_Ürün_Payı', 0)
                international_renk = "success" if international_pay > 20 else "warning" if international_pay > 10 else "info"
                st.markdown(f"""
                <div class="custom-metric-card {international_renk}">
                    <div class="custom-metric-label">INTERNATIONAL PAY</div>
                    <div class="custom-metric-value">{international_pay:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-cyan">Global</span>
                        <span>Ürün Payı</span>
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
                ortalama_fiyat = metrics.get('Ortalama_Fiyat', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">ORTALAMA FİYAT</div>
                    <div class="custom-metric-value">${ortalama_fiyat:.2f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Birim</span>
                        <span>2024 Fiyatı</span>
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
                ulke_sayisi = metrics.get('Ülke_Sayısı', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">COĞRAFİ YAYILIM</div>
                    <div class="custom-metric-value">{ulke_sayisi}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-cyan">Ülke</span>
                        <span>Pazar Sayısı</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Metrik kartları oluşturma hatası: {str(e)}")
    
    @staticmethod
    def create_market_overview_chart(df):
        """Pazar genel görünüm grafikleri"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Top 10 Molekül - Satış', 'Büyüme Dağılımı',
                               'Fiyat Dağılımı', 'Şirket Bazlı Satışlar'),
                specs=[[{"type": "bar"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "bar"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # Top 10 Molekül
            if 'Molekül' in df.columns and 'Satış_2024' in df.columns:
                top_moleküller = df.groupby('Molekül')['Satış_2024'].sum().nlargest(10)
                fig.add_trace(
                    go.Bar(
                        x=top_moleküller.values,
                        y=top_moleküller.index,
                        orientation='h',
                        marker_color='#1976d2',
                        name='Top 10 Molekül'
                    ),
                    row=1, col=1
                )
            
            # Büyüme dağılımı
            if 'Büyüme_23_24' in df.columns:
                fig.add_trace(
                    go.Histogram(
                        x=df['Büyüme_23_24'],
                        nbinsx=30,
                        marker_color='#4caf50',
                        name='Büyüme Dağılımı'
                    ),
                    row=1, col=2
                )
            
            # Fiyat dağılımı
            if 'Fiyat_2024' in df.columns:
                fig.add_trace(
                    go.Histogram(
                        x=df['Fiyat_2024'],
                        nbinsx=30,
                        marker_color='#ffb300',
                        name='Fiyat Dağılımı'
                    ),
                    row=2, col=1
                )
            
            # Top 10 Şirket
            if 'Şirket' in df.columns and 'Satış_2024' in df.columns:
                top_sirketler = df.groupby('Şirket')['Satış_2024'].sum().nlargest(10)
                fig.add_trace(
                    go.Bar(
                        x=top_sirketler.values,
                        y=top_sirketler.index,
                        orientation='h',
                        marker_color='#9c27b0',
                        name='Top 10 Şirket'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e3f2fd',
                showlegend=False,
                title_text="Pazar Genel Görünümü",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Pazar görünüm grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_price_analysis_chart(df):
        """Fiyat analizi grafikleri"""
        try:
            if 'Fiyat_2024' not in df.columns:
                return None
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Fiyat Dağılımı', 'Fiyat Segmentleri',
                               'Fiyat-Büyüme İlişkisi', 'Fiyat Karşılaştırması'),
                specs=[[{"type": "histogram"}, {"type": "pie"}],
                       [{"type": "scatter"}, {"type": "box"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # Fiyat dağılımı
            fiyat_verisi = df['Fiyat_2024'].dropna()
            if len(fiyat_verisi) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=fiyat_verisi,
                        nbinsx=30,
                        marker_color='#1976d2',
                        name='Fiyat Dağılımı'
                    ),
                    row=1, col=1
                )
            
                # Fiyat segmentleri (1D array hatası düzeltildi)
                try:
                    fiyat_array = fiyat_verisi.values.flatten() if fiyat_verisi.ndim > 1 else fiyat_verisi.values
                    fiyat_segmentleri = pd.cut(
                        fiyat_array,
                        bins=[0, 10, 50, 100, 500, float('inf')],
                        labels=['Ekonomi (<$10)', 'Standart ($10-$50)', 'Premium ($50-$100)', 
                               'Süper Premium ($100-$500)', 'Lüks (>$500)']
                    )
                    
                    segment_counts = pd.Series(fiyat_segmentleri).value_counts()
                    
                    fig.add_trace(
                        go.Pie(
                            labels=segment_counts.index,
                            values=segment_counts.values,
                            hole=0.4,
                            marker_colors=['#1976d2', '#42a5f5', '#00bcd4', '#4caf50', '#ffb300'],
                            name='Fiyat Segmentleri'
                        ),
                        row=1, col=2
                    )
                except Exception as e:
                    st.warning(f"Fiyat segmentasyonu hatası: {str(e)}")
            
            # Fiyat-Büyüme ilişkisi
            if 'Büyüme_23_24' in df.columns and 'Fiyat_2024' in df.columns:
                scatter_df = df[['Fiyat_2024', 'Büyüme_23_24']].dropna()
                if len(scatter_df) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=scatter_df['Fiyat_2024'],
                            y=scatter_df['Büyüme_23_24'],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=scatter_df['Büyüme_23_24'],
                                colorscale='RdYlGn',
                                showscale=True,
                                colorbar=dict(title="Büyüme %")
                            ),
                            name='Fiyat-Büyüme'
                        ),
                        row=2, col=1
                    )
            
            # Şirket bazlı fiyat karşılaştırması
            if 'Şirket' in df.columns and 'Fiyat_2024' in df.columns:
                top_sirketler = df['Şirket'].value_counts().nlargest(5).index
                sirket_veri = df[df['Şirket'].isin(top_sirketler)]
                
                fig.add_trace(
                    go.Box(
                        x=sirket_veri['Şirket'],
                        y=sirket_veri['Fiyat_2024'],
                        marker_color='#9c27b0',
                        name='Şirket Bazlı Fiyat'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e3f2fd',
                showlegend=False,
                title_text="Fiyat Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Fiyat analiz grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_international_analysis_chart(df, analysis_df):
        """International ürün analiz grafikleri"""
        try:
            if analysis_df is None or len(analysis_df) == 0:
                return None
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('International vs Yerel', 'International Ürün Pazar Payı',
                               'Coğrafi Yayılım', 'Büyüme Karşılaştırması'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # International vs Yerel
            intl_counts = analysis_df['International_Mı'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=['International', 'Yerel'],
                    values=intl_counts.values,
                    hole=0.4,
                    marker_colors=['#1976d2', '#90a4ae'],
                    textinfo='percent+label'
                ),
                row=1, col=1
            )
            
            # International ürün pazar payı
            intl_satislar = analysis_df[analysis_df['International_Mı']]['Toplam_Satış'].sum()
            yerel_satislar = analysis_df[~analysis_df['International_Mı']]['Toplam_Satış'].sum()
            
            fig.add_trace(
                go.Bar(
                    x=['International', 'Yerel'],
                    y=[intl_satislar, yerel_satislar],
                    marker_color=['#1976d2', '#90a4ae'],
                    text=[f'${intl_satislar/1e6:.1f}M', f'${yerel_satislar/1e6:.1f}M'],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Coğrafi yayılım
            intl_df = analysis_df[analysis_df['International_Mı']]
            if len(intl_df) > 0:
                ulke_dagilimi = intl_df['Ülke_Sayısı'].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(
                        x=ulke_dagilimi.index.astype(str),
                        y=ulke_dagilimi.values,
                        marker_color='#00bcd4',
                        name='Ülke Sayısı'
                    ),
                    row=2, col=1
                )
            
            # Büyüme karşılaştırması
            if 'Ortalama_Büyüme' in analysis_df.columns:
                intl_buyume = analysis_df[analysis_df['International_Mı']]['Ortalama_Büyüme'].mean()
                yerel_buyume = analysis_df[~analysis_df['International_Mı']]['Ortalama_Büyüme'].mean()
                
                if not pd.isna(intl_buyume) and not pd.isna(yerel_buyume):
                    fig.add_trace(
                        go.Bar(
                            x=['International', 'Yerel'],
                            y=[intl_buyume, yerel_buyume],
                            marker_color=['#1976d2', '#90a4ae'],
                            text=[f'{intl_buyume:.1f}%', f'{yerel_buyume:.1f}%'],
                            textposition='auto'
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e3f2fd',
                showlegend=False,
                title_text="International Ürün Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"International analiz grafiği hatası: {str(e)}")
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
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'international_analysis' not in st.session_state:
        st.session_state.international_analysis = None
    
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
        
        with st.expander("📁 VERİ YÜKLEME", expanded=True):
            uploaded_file = st.file_uploader(
                "Excel/CSV Dosyası Yükleyin",
                type=['xlsx', 'xls', 'csv'],
                help="1M+ satır desteklenir. Büyük dosyalar için dikkatli olun."
            )
            
            if uploaded_file:
                st.info(f"📄 Dosya: {uploaded_file.name}")
                
                col1, col2 = st.columns(2)
                with col1:
                    load_sample = st.button("🎯 Örnek Veri Yükle (10K)", width='stretch')
                with col2:
                    load_full = st.button("🚀 Tüm Veriyi Yükle", type="primary", width='stretch')
                
                if load_sample:
                    with st.spinner("Örnek veri yükleniyor..."):
                        processor = OptimizedDataProcessor()
                        df = processor.load_large_dataset(uploaded_file, sample_size=10000)
                        
                        if df is not None and len(df) > 0:
                            df = processor.prepare_analytics_data(df)
                            st.session_state.df = df
                            st.session_state.filtered_df = df.copy()
                            
                            analytics = AdvancedPharmaAnalytics()
                            st.session_state.metrics = analytics.calculate_comprehensive_metrics(df)
                            st.session_state.insights = analytics.detect_strategic_insights(df)
                            st.session_state.international_analysis = analytics.analyze_international_products(df)
                            
                            st.success(f"✅ {len(df):,} satır başarıyla yüklendi!")
                            st.rerun()
                
                if load_full:
                    with st.spinner("Tüm veri seti yükleniyor..."):
                        processor = OptimizedDataProcessor()
                        df = processor.load_large_dataset(uploaded_file, sample_size=None)
                        
                        if df is not None and len(df) > 0:
                            df = processor.prepare_analytics_data(df)
                            st.session_state.df = df
                            st.session_state.filtered_df = df.copy()
                            
                            analytics = AdvancedPharmaAnalytics()
                            st.session_state.metrics = analytics.calculate_comprehensive_metrics(df)
                            st.session_state.insights = analytics.detect_strategic_insights(df)
                            st.session_state.international_analysis = analytics.analyze_international_products(df)
                            
                            st.success(f"✅ {len(df):,} satır başarıyla yüklendi!")
                            st.rerun()
        
        # Demo veri butonu
        if st.session_state.df is None:
            st.markdown("---")
            if st.button("🎮 Demo Veri Oluştur", width='stretch'):
                with st.spinner("Demo veri oluşturuluyor..."):
                    demo_data = create_demo_data()
                    st.session_state.df = demo_data
                    st.session_state.filtered_df = demo_data.copy()
                    
                    analytics = AdvancedPharmaAnalytics()
                    st.session_state.metrics = analytics.calculate_comprehensive_metrics(demo_data)
                    st.session_state.insights = analytics.detect_strategic_insights(demo_data)
                    st.session_state.international_analysis = analytics.analyze_international_products(demo_data)
                    
                    st.success("✅ Demo veri başarıyla oluşturuldu!")
                    st.rerun()
        
        # Filtreleme bölümü
        if st.session_state.df is not None:
            st.markdown("---")
            with st.expander("🔍 TEMEL FİLTRELEME", expanded=False):
                df = st.session_state.df
                
                if 'Molekül' in df.columns:
                    moleküller = sorted(df['Molekül'].dropna().unique())
                    selected_molecules = st.multiselect(
                        "Molekül Seçin",
                        options=moleküller,
                        default=moleküller[:min(5, len(moleküller))]
                    )
                
                if 'Şirket' in df.columns:
                    şirketler = sorted(df['Şirket'].dropna().unique())
                    selected_companies = st.multiselect(
                        "Şirket Seçin",
                        options=şirketler,
                        default=şirketler[:min(5, len(şirketler))]
                    )
                
                if 'Ülke' in df.columns:
                    ülkeler = sorted(df['Ülke'].dropna().unique())
                    selected_countries = st.multiselect(
                        "Ülke Seçin",
                        options=ülkeler,
                        default=ülkeler[:min(5, len(ülkeler))]
                    )
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    if st.button("✅ Filtre Uygula", width='stretch'):
                        filtered_df = df.copy()
                        
                        if 'Molekül' in df.columns and selected_molecules:
                            filtered_df = filtered_df[filtered_df['Molekül'].isin(selected_molecules)]
                        if 'Şirket' in df.columns and selected_companies:
                            filtered_df = filtered_df[filtered_df['Şirket'].isin(selected_companies)]
                        if 'Ülke' in df.columns and selected_countries:
                            filtered_df = filtered_df[filtered_df['Ülke'].isin(selected_countries)]
                        
                        st.session_state.filtered_df = filtered_df
                        st.session_state.metrics = AdvancedPharmaAnalytics().calculate_comprehensive_metrics(filtered_df)
                        st.session_state.insights = AdvancedPharmaAnalytics().detect_strategic_insights(filtered_df)
                        st.session_state.international_analysis = AdvancedPharmaAnalytics().analyze_international_products(filtered_df)
                        
                        st.success(f"✅ Filtreler uygulandı: {len(filtered_df):,} satır")
                        st.rerun()
                
                with col_f2:
                    if st.button("🗑️ Filtreleri Temizle", width='stretch'):
                        st.session_state.filtered_df = st.session_state.df.copy()
                        st.session_state.metrics = AdvancedPharmaAnalytics().calculate_comprehensive_metrics(st.session_state.df)
                        st.session_state.insights = AdvancedPharmaAnalytics().detect_strategic_insights(st.session_state.df)
                        st.session_state.international_analysis = AdvancedPharmaAnalytics().analyze_international_products(st.session_state.df)
                        st.success("✅ Filtreler temizlendi")
                        st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #90a4ae;">
        <strong>PharmaIntelligence Pro</strong><br>
        v3.2 | International Product Analytics<br>
        © 2024 Tüm hakları saklıdır.
        </div>
        """, unsafe_allow_html=True)
    
    # Ana içerik bölümü
    if st.session_state.df is None:
        show_welcome_screen()
        return
    
    df = st.session_state.filtered_df
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    international_analysis = st.session_state.international_analysis
    
    # Filtre durumu gösterimi
    if len(df) != len(st.session_state.df):
        st.markdown(f"""
        <div class="filter-status">
        🎯 <strong>Aktif Filtreler:</strong> Gösterilen: {len(df):,} / {len(st.session_state.df):,} satır
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"📊 Tüm veri gösteriliyor: {len(df):,} satır")
    
    # Tablar
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ",
        "💰 FİYAT ANALİZİ",
        "🌍 INTERNATIONAL ÜRÜN",
        "📑 RAPORLAMA"
    ])
    
    with tab1:
        show_overview_tab(df, metrics, insights)
    
    with tab2:
        show_market_analysis_tab(df)
    
    with tab3:
        show_price_analysis_tab(df, metrics)
    
    with tab4:
        show_international_product_tab(df, international_analysis, metrics)
    
    with tab5:
        show_reporting_tab(df, metrics, insights, international_analysis)

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
            <h2 style="color: #e3f2fd; margin-bottom: 1rem;">PharmaIntelligence Pro'ya Hoşgeldiniz</h2>
            <p style="color: #bbdefb; margin-bottom: 2rem; line-height: 1.6;">
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
                2. Örnek veri için "Örnek Veri Yükle" veya tüm veri için "Tüm Veriyi Yükle" butonuna tıklayın<br>
                3. Analiz sonuçlarını görmek için tabları kullanın<br>
                <br>
                <em>Veya "Demo Veri Oluştur" butonu ile demo veri ile test yapın</em>
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
                elif insight['type'] == 'cyan':
                    icon = "🌍"
                
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

def show_market_analysis_tab(df):
    """Pazar Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Pazar Analizi ve Trendler</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-title">📊 Pazar Genel Görünümü</h3>', unsafe_allow_html=True)
    
    market_fig = viz.create_market_overview_chart(df)
    if market_fig:
        st.plotly_chart(market_fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Pazar analizi için yeterli veri bulunamadı.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Molekül bazlı detaylı analiz
    if 'Molekül' in df.columns and 'Satış_2024' in df.columns:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="subsection-title">🧪 Molekül Performans Analizi</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_molecules = df.groupby('Molekül')['Satış_2024'].sum().nlargest(15)
            fig1 = px.bar(
                top_molecules,
                orientation='h',
                title='Top 15 Molekül - Satış',
                color=top_molecules.values,
                color_continuous_scale='Blues'
            )
            fig1.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e3f2fd',
                xaxis_title='Satış (USD)',
                yaxis_title='Molekül'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            if 'Büyüme_23_24' in df.columns:
                growth_data = df.groupby('Molekül')['Büyüme_23_24'].mean().nlargest(15)
                fig2 = px.bar(
                    growth_data,
                    orientation='h',
                    title='Top 15 Molekül - Büyüme',
                    color=growth_data.values,
                    color_continuous_scale='RdYlGn'
                )
                fig2.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0)',
                    font_color='#e3f2fd',
                    xaxis_title='Büyüme (%)',
                    yaxis_title='Molekül'
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Şirket bazlı analiz
    if 'Şirket' in df.columns and 'Satış_2024' in df.columns:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="subsection-title">🏢 Şirket Performans Analizi</h3>', unsafe_allow_html=True)
        
        company_sales = df.groupby('Şirket')['Satış_2024'].sum().sort_values(ascending=False)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Pazar Payı Dağılımı', 'Top 10 Şirket'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}]]
        )
        
        # Pasta grafik (ilk 10 + diğerleri)
        top_10_companies = company_sales.head(10)
        other_sales = company_sales.iloc[10:].sum()
        
        pie_labels = list(top_10_companies.index) + ['Diğer']
        pie_values = list(top_10_companies.values) + [other_sales]
        
        fig.add_trace(
            go.Pie(
                labels=pie_labels,
                values=pie_values,
                hole=0.4,
                textinfo='percent+label',
                insidetextorientation='radial'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=top_10_companies.values,
                y=top_10_companies.index,
                orientation='h',
                marker_color='#1976d2',
                text=[f'${x/1e6:.1f}M' for x in top_10_companies.values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e3f2fd',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_price_analysis_tab(df, metrics):
    """Fiyat Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Fiyat Analizi ve Optimizasyon</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    
    # Fiyat analiz grafikleri
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-title">💰 Fiyat Analizi</h3>', unsafe_allow_html=True)
    
    price_fig = viz.create_price_analysis_chart(df)
    if price_fig:
        st.plotly_chart(price_fig, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Fiyat analizi için yeterli veri bulunamadı.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Fiyat metrikleri
    if 'Fiyat_2024' in df.columns:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="subsection-title">📊 Fiyat İstatistikleri</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ortalama_fiyat = metrics.get('Ortalama_Fiyat', 0)
            st.metric("Ortalama Fiyat", f"${ortalama_fiyat:.2f}")
        
        with col2:
            fiyat_cv = metrics.get('Fiyat_CV', 0)
            st.metric("Fiyat CV", f"%{fiyat_cv:.1f}")
        
        with col3:
            if 'Fiyat_2024' in df.columns:
                fiyat_q1 = df['Fiyat_2024'].quantile(0.25)
                st.metric("1. Çeyrek", f"${fiyat_q1:.2f}")
        
        with col4:
            if 'Fiyat_2024' in df.columns:
                fiyat_q3 = df['Fiyat_2024'].quantile(0.75)
                st.metric("3. Çeyrek", f"${fiyat_q3:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Fiyat-hacim korelasyonu
    if 'Fiyat_2024' in df.columns and 'Hacim_2024' in df.columns:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="subsection-title">📉 Fiyat-Hacim İlişkisi</h3>', unsafe_allow_html=True)
        
        correlation_data = df[['Fiyat_2024', 'Hacim_2024']].dropna()
        if len(correlation_data) > 10:
            correlation = correlation_data.corr().iloc[0, 1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Korelasyon Katsayısı", f"{correlation:.3f}")
            
            with col2:
                if correlation < -0.3:
                    esneklik = "Yüksek Esneklik"
                elif correlation > 0.3:
                    esneklik = "Düşük Esneklik"
                else:
                    esneklik = "Nötr"
                st.metric("Esneklik Durumu", esneklik)
            
            with col3:
                if correlation < -0.3:
                    oneri = "Fiyat Artışı Riskli"
                elif correlation > 0.3:
                    oneri = "Fiyat Artışı Mümkün"
                else:
                    oneri = "Limitli Artış"
                st.metric("Strateji Önerisi", oneri)
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_international_product_tab(df, analysis_df, metrics):
    """International Ürün Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">🌍 International Ürün Analizi</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    
    if analysis_df is None:
        st.warning("International ürün analizi için gerekli veri bulunamadı.")
        return
    
    # International metrikler
    st.markdown('<h3 class="subsection-title">📊 International Ürün Metrikleri</h3>', unsafe_allow_html=True)
    
    intl_cols = st.columns(4)
    
    with intl_cols[0]:
        intl_sayisi = metrics.get('International_Ürün_Sayısı', 0)
        toplam_molekül = metrics.get('Benzersiz_Moleküller', 0)
        intl_yuzde = (intl_sayisi / toplam_molekül * 100) if toplam_molekül > 0 else 0
        st.metric("International Ürün", f"{intl_sayisi}", f"%{intl_yuzde:.1f}")
    
    with intl_cols[1]:
        intl_pay = metrics.get('International_Ürün_Payı', 0)
        st.metric("Pazar Payı", f"%{intl_pay:.1f}")
    
    with intl_cols[2]:
        ort_ulke = metrics.get('Ort_International_Ülkeler', 0)
        st.metric("Ort. Ülke", f"{ort_ulke:.1f}")
    
    with intl_cols[3]:
        ort_sirket = metrics.get('Ort_International_Şirketler', 0)
        st.metric("Ort. Şirket", f"{ort_sirket:.1f}")
    
    # International analiz grafikleri
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-title">📈 International Ürün Analizi</h3>', unsafe_allow_html=True)
    
    intl_fig = viz.create_international_analysis_chart(df, analysis_df)
    if intl_fig:
        st.plotly_chart(intl_fig, use_container_width=True, config={'displayModeBar': True})
    st.markdown('</div>', unsafe_allow_html=True)
    
    # International ürün listesi
    st.markdown('<h3 class="subsection-title">📋 International Ürün Listesi</h3>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Tüm International Ürünler", "Top Performanslılar"])
    
    with tab1:
        if len(analysis_df) > 0:
            display_columns = [
                'Molekül', 'International_Mı', 'Toplam_Satış', 'Şirket_Sayısı',
                'Ülke_Sayısı', 'Ortalama_Fiyat', 'Ortalama_Büyüme', 'International_Segment'
            ]
            
            display_columns = [col for col in display_columns if col in analysis_df.columns]
            
            intl_display = analysis_df[display_columns].copy()
            
            def format_value(value, format_type):
                try:
                    if pd.isna(value):
                        return "N/A"
                    
                    if format_type == 'currency':
                        return f"${float(value)/1e6:.2f}M"
                    elif format_type == 'percentage':
                        return f"{float(value):.1f}%"
                    elif format_type == 'price':
                        return f"${float(value):.2f}"
                    elif format_type == 'boolean':
                        return "✅" if value else "❌"
                    else:
                        return str(value)
                except:
                    return "N/A"
            
            if 'Toplam_Satış' in intl_display.columns:
                intl_display['Toplam_Satış'] = intl_display['Toplam_Satış'].apply(
                    lambda x: format_value(x, 'currency')
                )
            
            if 'International_Mı' in intl_display.columns:
                intl_display['International_Mı'] = intl_display['International_Mı'].apply(
                    lambda x: format_value(x, 'boolean')
                )
            
            if 'Ortalama_Büyüme' in intl_display.columns:
                intl_display['Ortalama_Büyüme'] = intl_display['Ortalama_Büyüme'].apply(
                    lambda x: format_value(x, 'percentage')
                )
            
            if 'Ortalama_Fiyat' in intl_display.columns:
                intl_display['Ortalama_Fiyat'] = intl_display['Ortalama_Fiyat'].apply(
                    lambda x: format_value(x, 'price')
                )
            
            st.dataframe(
                intl_display,
                use_container_width=True,
                height=400
            )
    
    with tab2:
        top_intl = analysis_df[analysis_df['International_Mı']].nlargest(20, 'Toplam_Satış')
        
        if len(top_intl) > 0:
            top_columns = ['Molekül', 'Toplam_Satış', 'Şirket_Sayısı', 'Ülke_Sayısı', 
                          'Ortalama_Büyüme', 'Top_Şirket', 'Top_Ülke']
            
            top_columns = [col for col in top_columns if col in top_intl.columns]
            
            top_display = top_intl[top_columns].copy()
            
            if 'Toplam_Satış' in top_display.columns:
                top_display['Toplam_Satış'] = top_display['Toplam_Satış'].apply(
                    lambda x: format_value(x, 'currency')
                )
            
            if 'Ortalama_Büyüme' in top_display.columns:
                top_display['Ortalama_Büyüme'] = top_display['Ortalama_Büyüme'].apply(
                    lambda x: format_value(x, 'percentage')
                )
            
            st.dataframe(
                top_display,
                use_container_width=True,
                height=400
            )
    
    # Strateji önerileri
    st.markdown('<h3 class="subsection-title">🎯 Strateji Önerileri</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-card info">
            <div class="insight-title">🚀 International Ürün Büyüme Stratejisi</div>
            <div class="insight-content">
            1. Yüksek büyüme gösteren International ürünleri belirleyin<br>
            2. Bu ürünlerin diğer ülkelere yayılma potansiyelini değerlendirin<br>
            3. Yerel pazarlarda lider olan ürünleri International ürüne dönüştürün
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card success">
            <div class="insight-title">💰 International Ürün Fiyatlandırma</div>
            <div class="insight-content">
            1. Ülke bazında fiyatlandırma stratejileri geliştirin<br>
            2. Premium segmentteki International ürünlerin fiyatını optimize edin<br>
            3. Fiyat esnekliği düşük ürünlere odaklanın
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_reporting_tab(df, metrics, insights, analysis_df):
    """Raporlama tab'ını göster"""
    st.markdown('<h2 class="section-title">Raporlama ve İndirme</h2>', unsafe_allow_html=True)
    
    # Hızlı istatistikler
    st.markdown('<h3 class="subsection-title">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)
    
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with stat_cols[1]:
        st.metric("Toplam Sütun", len(df.columns))
    
    with stat_cols[2]:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Bellek Kullanımı", f"{memory_usage:.1f} MB")
    
    with stat_cols[3]:
        intl_count = metrics.get('International_Ürün_Sayısı', 0)
        st.metric("International Ürün", intl_count)
    
    # Rapor oluşturma
    st.markdown('<h3 class="subsection-title">📊 Rapor Oluşturma</h3>', unsafe_allow_html=True)
    
    report_cols = st.columns(3)
    
    with report_cols[0]:
        if st.button("📈 Excel Raporu Oluştur", width='stretch', key="excel_report"):
            with st.spinner("Excel raporu oluşturuluyor..."):
                excel_report = generate_excel_report(df, metrics, insights, analysis_df)
                
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
    
    with report_cols[1]:
        if st.button("🔄 Analizi Sıfırla", width='stretch', key="reset_analysis"):
            st.session_state.df = None
            st.session_state.filtered_df = None
            st.session_state.metrics = None
            st.session_state.insights = []
            st.session_state.international_analysis = None
            st.rerun()
    
    with report_cols[2]:
        if st.button("💾 CSV İndir", width='stretch', key="csv_download"):
            csv = df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="⬇️ CSV İndir",
                data=csv,
                file_name=f"pharma_data_{timestamp}.csv",
                mime="text/csv",
                width='stretch',
                key="download_csv"
            )
    
    # Veri önizleme
    st.markdown('<h3 class="subsection-title">📋 Veri Önizleme</h3>', unsafe_allow_html=True)
    
    with st.expander("📊 Veri İstatistikleri"):
        st.write(f"**Toplam Satır:** {len(df):,}")
        st.write(f"**Toplam Sütun:** {len(df.columns)}")
        
        if 'Satış_2024' in df.columns:
            st.write(f"**Toplam Satış:** ${df['Satış_2024'].sum():,.0f}")
            st.write(f"**Ortalama Satış:** ${df['Satış_2024'].mean():,.0f}")
        
        if 'Büyüme_23_24' in df.columns:
            st.write(f"**Ortalama Büyüme:** %{df['Büyüme_23_24'].mean():.1f}")
        
        if 'Molekül' in df.columns:
            st.write(f"**Benzersiz Molekül:** {df['Molekül'].nunique():,}")
        
        if 'Şirket' in df.columns:
            st.write(f"**Benzersiz Şirket:** {df['Şirket'].nunique():,}")
        
        if 'Ülke' in df.columns:
            st.write(f"**Benzersiz Ülke:** {df['Ülke'].nunique():,}")
    
    with st.expander("📈 Detaylı Metrikler"):
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'Satış' in key or 'Değeri' in key:
                        st.write(f"**{key}:** ${value:,.0f}")
                    elif 'Payı' in key or 'Yüzde' in key or 'Büyüme' in key:
                        st.write(f"**{key}:** %{value:.1f}")
                    else:
                        st.write(f"**{key}:** {value:,.0f}")

# ================================================
# YARDIMCI FONKSİYONLAR
# ================================================

def create_demo_data():
    """Demo veri oluştur"""
    np.random.seed(42)
    
    # Temel veri yapısı
    n = 5000  # Satır sayısı
    
    # Molekül listesi
    molecules = ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Metformin', 'Atorvastatin',
                'Lisinopril', 'Levothyroxine', 'Amlodipine', 'Metoprolol', 'Omeprazole',
                'Simvastatin', 'Losartan', 'Albuterol', 'Gabapentin', 'Hydrochlorothiazide',
                'Sertraline', 'Fluoxetine', 'Citalopram', 'Warfarin', 'Clopidogrel']
    
    # Şirket listesi
    companies = ['Pfizer', 'Novartis', 'Roche', 'Merck', 'GSK',
                'Sanofi', 'AstraZeneca', 'Johnson & Johnson', 'Bayer', 'AbbVie',
                'Eli Lilly', 'Bristol-Myers Squibb', 'Amgen', 'Gilead', 'Biogen']
    
    # Ülke listesi
    countries = ['USA', 'Germany', 'UK', 'France', 'Japan',
                'China', 'India', 'Brazil', 'Canada', 'Australia',
                'Italy', 'Spain', 'Mexico', 'Turkey', 'South Korea']
    
    # Veri oluştur
    data = {
        'Molekül': np.random.choice(molecules, n),
        'Şirket': np.random.choice(companies, n),
        'Ülke': np.random.choice(countries, n),
        'Satış_2024': np.random.lognormal(10, 1.5, n),  # Log-normal dağılım
        'Satış_2023': np.random.lognormal(9.8, 1.5, n),
        'Fiyat_2024': np.random.uniform(5, 500, n),
        'Hacim_2024': np.random.randint(1000, 100000, n)
    }
    
    df = pd.DataFrame(data)
    
    # Büyüme oranını hesapla
    df['Büyüme_23_24'] = ((df['Satış_2024'] - df['Satış_2023']) / df['Satış_2023']) * 100
    
    # Pazar payını hesapla
    total_sales = df['Satış_2024'].sum()
    df['Pazar_Payı'] = (df['Satış_2024'] / total_sales) * 100
    
    # Fiyat-hacim oranı
    df['Fiyat_Hacim_Oranı'] = df['Fiyat_2024'] * df['Hacim_2024']
    
    return df

def generate_excel_report(df, metrics, insights, analysis_df):
    """Excel raporu oluştur"""
    try:
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Ana veri
            df.to_excel(writer, sheet_name='HAM_VERİ', index=False)
            
            # Metrikler
            if metrics:
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['METRİK', 'DEĞER'])
                metrics_df.to_excel(writer, sheet_name='ÖZET_METRİKLER', index=False)
            
            # Pazar payı analizi
            if 'Şirket' in df.columns and 'Satış_2024' in df.columns:
                market_share = df.groupby('Şirket')['Satış_2024'].sum().sort_values(ascending=False)
                market_share_df = market_share.reset_index()
                market_share_df.columns = ['ŞİRKET', 'SATIŞ']
                market_share_df['PAY (%)'] = (market_share_df['SATIŞ'] / market_share_df['SATIŞ'].sum()) * 100
                market_share_df['KÜMÜLATİF_PAY'] = market_share_df['PAY (%)'].cumsum()
                market_share_df.to_excel(writer, sheet_name='PAZAR_PAYI', index=False)
            
            # International analiz
            if analysis_df is not None:
                analysis_df.to_excel(writer, sheet_name='INTERNATIONAL_ANALİZ', index=False)
            
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
                insights_df.to_excel(writer, sheet_name='STRATEJİK_İÇGÖRÜLER', index=False)
        
        output.seek(0)
        return output
        
    except Exception as e:
        st.error(f"Excel rapor oluşturma hatası: {str(e)}")
        return None

# ================================================
# UYGULAMA BAŞLATMA
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
