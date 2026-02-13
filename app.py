"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                         PHARMAINTELLIGENCE PRO - KURUMSAL EDİSYON v7.0                    ║
║                                                                                          ║
║              • Gelişmiş Makine Öğrenimi (Scikit-learn)                                   ║
║              • Otomatik Segmentasyon & Kümeleme                                          ║
║              • Zaman Serisi Analizi & Tahminleme                                        ║
║              • Anomali Tespiti & Risk Analizi                                           ║
║              • İnteraktif Gösterge Panelleri                                            ║
║              • Profesyonel Raporlama (Excel/PDF/HTML)                                   ║
║                                                                                          ║
║                         Streamlit Cloud Optimize Edilmiş Sürüm                           ║
║                         © 2024 PharmaIntelligence Inc.                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# 1. TEMEL KÜTÜPHANELER - Streamlit Cloud Uyumlu
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_dendrogram
import warnings
warnings.filterwarnings('ignore')

# === Makine Öğrenimi - Scikit-learn ===
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
)
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering
)
from sklearn.ensemble import (
    IsolationForest, RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, mean_squared_error, r2_score,
    mean_absolute_error, explained_variance_score
)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# === Zaman Serisi ===
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# === İstatistik ===
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr, spearmanr

# === Veri İşleme ===
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
import re
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import lru_cache
import base64

# === Raporlama ===
import xlsxwriter
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ============================================================================
# 2. KURUMSAL TEMA & CSS - Profesyonel Tasarım
# ============================================================================

st.set_page_config(
    page_title="PharmaIntelligence Pro | Kurumsal Analitik",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROFESSIONAL_CSS = """
<style>
    /* === KÖK DEĞİŞKENLER === */
    :root {
        --primary: #0a1e3c;
        --secondary: #1e3a5f;
        --accent: #2d7dd2;
        --accent-light: #4a9fe3;
        --accent-dark: #1a5fa0;
        --success: #2dd2a3;
        --warning: #f2c94c;
        --danger: #eb5757;
        --info: #2d7dd2;
        
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        
        --bg-primary: #0a1e3c;
        --bg-secondary: #1e3a5f;
        --bg-card: rgba(30, 58, 95, 0.85);
        --bg-hover: rgba(45, 125, 210, 0.15);
        
        --border: rgba(255, 255, 255, 0.1);
        --border-hover: rgba(45, 125, 210, 0.3);
        
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 32px rgba(0,0,0,0.5);
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        
        --transition: 300ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* === GENEL STİLLER === */
    .stApp {
        background: radial-gradient(circle at 0% 0%, var(--primary), #0f1a2b);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    /* === GLASMORFİZM KART === */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.75rem;
        transition: all var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--success));
        transform: scaleX(0);
        transition: transform var(--transition);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: var(--accent);
        box-shadow: var(--shadow-lg);
    }
    
    .glass-card:hover::before {
        transform: scaleX(1);
    }
    
    /* === BAŞLIKLAR === */
    .pharma-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -1px;
        animation: fadeInUp 0.8s ease;
    }
    
    .pharma-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        font-weight: 400;
        line-height: 1.6;
        max-width: 900px;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1.5rem 0;
        padding-left: 1rem;
        border-left: 5px solid;
        border-image: linear-gradient(to bottom, var(--accent), var(--success));
        border-image-slice: 1;
        background: linear-gradient(90deg, rgba(45,125,210,0.1), transparent);
        padding: 1rem;
        border-radius: var(--radius-md);
        color: var(--text-primary);
    }
    
    /* === METRİK KARTLARI === */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--border);
        transition: all var(--transition);
    }
    
    .metric-card:hover {
        border-color: var(--accent);
        transform: scale(1.02);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    
    .metric-trend {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        color: var(--text-muted);
    }
    
    /* === İÇGÖRÜ KARTLARI === */
    .insight-card {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        transition: all var(--transition);
    }
    
    .insight-card:hover {
        transform: translateX(5px);
        background: var(--bg-hover);
    }
    
    .insight-success { border-left-color: var(--success); }
    .insight-warning { border-left-color: var(--warning); }
    .insight-danger { border-left-color: var(--danger); }
    .insight-info { border-left-color: var(--accent); }
    
    /* === ROZETLER === */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-primary {
        background: rgba(45,125,210,0.2);
        color: var(--accent-light);
        border: 1px solid rgba(45,125,210,0.3);
    }
    
    .badge-success {
        background: rgba(45,210,163,0.2);
        color: var(--success);
        border: 1px solid rgba(45,210,163,0.3);
    }
    
    .badge-warning {
        background: rgba(242,201,76,0.2);
        color: var(--warning);
        border: 1px solid rgba(242,201,76,0.3);
    }
    
    .badge-danger {
        background: rgba(235,87,87,0.2);
        color: var(--danger);
        border: 1px solid rgba(235,87,87,0.3);
    }
    
    /* === FİLTRE BİLGİSİ === */
    .filter-info {
        background: linear-gradient(135deg, rgba(45,125,210,0.15), rgba(45,210,163,0.15));
        padding: 1rem 1.5rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--success);
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    /* === ANİMASYONLAR === */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .animate-fadeIn {
        animation: fadeInUp 0.6s ease;
    }
    
    .animate-slideIn {
        animation: slideIn 0.6s ease;
    }
    
    /* === RESPONSIVE === */
    @media (max-width: 1200px) {
        .pharma-title { font-size: 2.5rem; }
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
    }
    
    @media (max-width: 768px) {
        .pharma-title { font-size: 2rem; }
        .metric-grid { grid-template-columns: 1fr; }
    }
    
    /* === SCROLLBAR === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--accent), var(--success));
        border-radius: 9999px;
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ============================================================================
# 3. VERİ İŞLEME MOTORU
# ============================================================================

class DataProcessor:
    """Gelişmiş veri işleme ve optimizasyon motoru"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_data(file) -> Optional[pd.DataFrame]:
        """Veri yükleme - Streamlit Cloud optimizasyonu"""
        try:
            start_time = time.time()
            
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, low_memory=False)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, engine='openpyxl')
            else:
                st.error("❌ Desteklenmeyen dosya formatı. CSV veya Excel yükleyin.")
                return None
            
            # Sütun isimlerini temizle
            df.columns = DataProcessor._clean_column_names(df.columns)
            
            # Veri tipi optimizasyonu
            df = DataProcessor._optimize_dtypes(df)
            
            load_time = time.time() - start_time
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            st.success(f"""
            ✅ **Veri başarıyla yüklendi!**
            - Satır: {len(df):,}
            - Sütun: {len(df.columns)}
            - Bellek: {memory_usage:.1f} MB
            - Süre: {load_time:.2f}s
            """)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            return None
    
    @staticmethod
    def _clean_column_names(columns: List[str]) -> List[str]:
        """Sütun isimlerini temizle ve standardize et"""
        cleaned = []
        seen = {}
        
        for col in columns:
            if isinstance(col, str):
                # Türkçe karakter dönüşümü
                char_map = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in char_map.items():
                    col = col.replace(tr, en)
                
                # Özel karakterleri temizle
                col = re.sub(r'[^\w\s-]', '', col)
                # Boşlukları alt çizgiye çevir
                col = re.sub(r'\s+', '_', col)
                # Birden fazla alt çizgiyi teke indir
                col = re.sub(r'_+', '_', col)
                col = col.strip('_')
                
                # Domain-spesifik haritalama
                col = DataProcessor._apply_mapping(col)
            
            # Duplicate handling
            if col in seen:
                seen[col] += 1
                col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
            
            cleaned.append(str(col))
        
        return cleaned
    
    @staticmethod
    def _apply_mapping(col: str) -> str:
        """Alan-spesifik sütun haritalama"""
        
        # Satış
        if re.search(r'MAT.*Q3.*2022.*USD.*MNF', col, re.I):
            return 'Sales_2022'
        elif re.search(r'MAT.*Q3.*2023.*USD.*MNF', col, re.I):
            return 'Sales_2023'
        elif re.search(r'MAT.*Q3.*2024.*USD.*MNF', col, re.I):
            return 'Sales_2024'
        
        # Birim
        elif re.search(r'MAT.*Q3.*2022.*Units', col, re.I):
            return 'Units_2022'
        elif re.search(r'MAT.*Q3.*2023.*Units', col, re.I):
            return 'Units_2023'
        elif re.search(r'MAT.*Q3.*2024.*Units', col, re.I):
            return 'Units_2024'
        
        # Fiyat
        elif re.search(r'MAT.*Q3.*2022.*Unit.*Avg.*Price', col, re.I):
            return 'Price_2022'
        elif re.search(r'MAT.*Q3.*2023.*Unit.*Avg.*Price', col, re.I):
            return 'Price_2023'
        elif re.search(r'MAT.*Q3.*2024.*Unit.*Avg.*Price', col, re.I):
            return 'Price_2024'
        
        # Standart
        mapping = {
            'Country': 'Country',
            'Sector': 'Sector',
            'Corporation': 'Company',
            'Manufacturer': 'Manufacturer',
            'Molecule': 'Molecule',
            'International Product': 'International_Product',
            'Specialty Product': 'Specialty_Product',
            'Region': 'Region',
            'Sub-Region': 'Sub_Region'
        }
        
        for k, v in mapping.items():
            if k in col:
                return v
        
        return col
    
    @staticmethod
    def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Veri tiplerini optimize et - bellek tasarrufu"""
        
        for col in df.columns:
            # Object -> Category
            if df[col].dtype == 'object':
                n_unique = df[col].nunique()
                if n_unique < len(df) * 0.5:
                    df[col] = df[col].astype('category')
            
            # Integer optimizasyonu
            elif df[col].dtype in ['int64', 'int32']:
                try:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    if pd.isna(min_val) or pd.isna(max_val):
                        continue
                    
                    if min_val >= 0:
                        if max_val <= 255:
                            df[col] = df[col].astype('uint8')
                        elif max_val <= 65535:
                            df[col] = df[col].astype('uint16')
                        elif max_val <= 4294967295:
                            df[col] = df[col].astype('uint32')
                    else:
                        if min_val >= -128 and max_val <= 127:
                            df[col] = df[col].astype('int8')
                        elif min_val >= -32768 and max_val <= 32767:
                            df[col] = df[col].astype('int16')
                        elif min_val >= -2147483648 and max_val <= 2147483647:
                            df[col] = df[col].astype('int32')
                except:
                    pass
            
            # Float optimizasyonu
            elif df[col].dtype == 'float64':
                try:
                    df[col] = df[col].astype('float32')
                except:
                    pass
        
        return df
    
    @staticmethod
    def prepare_analytics(df: pd.DataFrame) -> pd.DataFrame:
        """Analitik için veri hazırlama - hesaplanmış metrikler"""
        
        df = df.copy()
        
        # Satış sütunlarını bul
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        
        if len(sales_cols) >= 2:
            # Büyüme hesapla
            for i in range(1, len(sales_cols)):
                prev = sales_cols[i-1]
                curr = sales_cols[i]
                
                growth_col = f'Growth_{curr.split("_")[1]}'
                df[growth_col] = np.where(
                    df[prev] > 0,
                    ((df[curr] - df[prev]) / df[prev]) * 100,
                    np.nan
                )
            
            # CAGR hesapla (son yıl / ilk yıl)
            first_year = sales_cols[0]
            last_year = sales_cols[-1]
            years_diff = int(last_year.split('_')[1]) - int(first_year.split('_')[1])
            
            if years_diff > 0:
                df['CAGR'] = np.where(
                    df[first_year] > 0,
                    (np.power(df[last_year] / df[first_year], 1/years_diff) - 1) * 100,
                    np.nan
                )
            
            # Pazar payı
            total_sales = df[last_year].sum()
            if total_sales > 0:
                df['Market_Share'] = (df[last_year] / total_sales) * 100
        
        # Fiyat hesapla (eğer yoksa)
        price_cols = [col for col in df.columns if 'Price_' in col]
        if not price_cols and sales_cols:
            for col in sales_cols:
                year = col.split('_')[1]
                units_col = f'Units_{year}'
                if units_col in df.columns:
                    df[f'Price_{year}'] = np.where(
                        df[units_col] > 0,
                        df[col] / df[units_col],
                        np.nan
                    )
        
        return df
    
    @staticmethod
    def extract_year(column_name: str) -> Optional[int]:
        """Sütun adından yıl çıkar"""
        match = re.search(r'\b(20\d{2})\b', column_name)
        if match:
            return int(match.group(1))
        return None

# ============================================================================
# 4. GELİŞMİŞ FİLTRELEME SİSTEMİ
# ============================================================================

class FilterSystem:
    """Çok kriterli gelişmiş filtreleme sistemi"""
    
    @staticmethod
    def render_filters(df: pd.DataFrame) -> Tuple[str, Dict, bool, bool]:
        """Filtre arayüzünü oluştur"""
        
        with st.sidebar.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
            st.markdown("### 🔍 Global Arama")
            
            search_term = st.text_input(
                "Ara",
                placeholder="Molekül, şirket, ülke...",
                label_visibility="collapsed"
            )
            
            filter_config = {}
            
            # Kategorik filtreler
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            priority_cols = ['Country', 'Company', 'Molecule', 'Region']
            
            for col in priority_cols:
                if col in cat_cols:
                    values = sorted(df[col].dropna().unique())
                    if len(values) > 0:
                        selected = st.multiselect(
                            f"📌 {col}",
                            options=values,
                            default=[],
                            key=f"filter_{col}"
                        )
                        if selected:
                            filter_config[col] = selected
            
            # Sayısal filtreler
            st.markdown("### 📊 Sayısal Filtreler")
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                latest_sales = sales_cols[-1]
                min_val = float(df[latest_sales].min())
                max_val = float(df[latest_sales].max())
                
                if min_val < max_val:
                    sales_range = st.slider(
                        f"Satış ({latest_sales})",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                    filter_config['sales_range'] = (sales_range, latest_sales)
            
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                latest_growth = growth_cols[-1]
                min_g = float(df[latest_growth].min())
                max_g = float(df[latest_growth].max())
                
                if min_g < max_g:
                    growth_range = st.slider(
                        f"Büyüme ({latest_growth})",
                        min_value=min_g,
                        max_value=max_g,
                        value=(min_g, max_g)
                    )
                    filter_config['growth_range'] = (growth_range, latest_growth)
            
            # Uluslararası ürün filtresi
            if 'International_Product' in df.columns:
                intl_filter = st.selectbox(
                    "🌐 Uluslararası Ürün",
                    ["Hepsi", "Sadece Uluslararası", "Sadece Yerel"]
                )
                if intl_filter != "Hepsi":
                    filter_config['international'] = intl_filter
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                apply_filters = st.button("✅ Uygula", use_container_width=True, type="primary")
            with col2:
                clear_filters = st.button("🗑️ Temizle", use_container_width=True)
            
            return search_term, filter_config, apply_filters, clear_filters
    
    @staticmethod
    def apply_filters(
        df: pd.DataFrame,
        search_term: str,
        filter_config: Dict
    ) -> pd.DataFrame:
        """Filtreleri DataFrame'e uygula"""
        
        filtered = df.copy()
        
        # Global arama
        if search_term:
            mask = pd.Series(False, index=filtered.index)
            for col in filtered.columns:
                try:
                    mask |= filtered[col].astype(str).str.contains(
                        search_term, case=False, na=False
                    )
                except:
                    continue
            filtered = filtered[mask]
        
        # Kategorik filtreler
        for col, values in filter_config.items():
            if col in filtered.columns and isinstance(values, list):
                filtered = filtered[filtered[col].isin(values)]
        
        # Satış aralığı
        if 'sales_range' in filter_config:
            (min_val, max_val), col = filter_config['sales_range']
            filtered = filtered[
                (filtered[col] >= min_val) & 
                (filtered[col] <= max_val)
            ]
        
        # Büyüme aralığı
        if 'growth_range' in filter_config:
            (min_val, max_val), col = filter_config['growth_range']
            filtered = filtered[
                (filtered[col] >= min_val) & 
                (filtered[col] <= max_val)
            ]
        
        # Uluslararası ürün
        if 'international' in filter_config:
            if filter_config['international'] == "Sadece Uluslararası":
                filtered = filtered[filtered['International_Product'] == 1]
            elif filter_config['international'] == "Sadece Yerel":
                filtered = filtered[filtered['International_Product'] == 0]
        
        return filtered

# ============================================================================
# 5. ANALİTİK MOTORU - Scikit-learn Tabanlı
# ============================================================================

class AnalyticsEngine:
    """Makine öğrenimi ve istatistiksel analiz motoru"""
    
    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Kapsamlı pazar metrikleri hesapla"""
        
        metrics = {}
        
        try:
            metrics['total_rows'] = len(df)
            metrics['total_columns'] = len(df.columns)
            
            # Satış metrikleri
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                latest = sales_cols[-1]
                year = DataProcessor.extract_year(latest) or "Son"
                
                metrics['latest_year'] = year
                metrics['total_market_value'] = float(df[latest].sum())
                metrics['avg_sales'] = float(df[latest].mean())
                metrics['median_sales'] = float(df[latest].median())
                metrics['std_sales'] = float(df[latest].std())
                metrics['q1_sales'] = float(df[latest].quantile(0.25))
                metrics['q3_sales'] = float(df[latest].quantile(0.75))
                metrics['iqr_sales'] = metrics['q3_sales'] - metrics['q1_sales']
                
                # Top N ürünlerin pazar payı
                top_10_sum = df.nlargest(10, latest)[latest].sum()
                metrics['top10_share'] = (top_10_sum / metrics['total_market_value'] * 100) if metrics['total_market_value'] > 0 else 0
                
                top_20_sum = df.nlargest(20, latest)[latest].sum()
                metrics['top20_share'] = (top_20_sum / metrics['total_market_value'] * 100) if metrics['total_market_value'] > 0 else 0
            
            # Büyüme metrikleri
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                latest_growth = growth_cols[-1]
                metrics['avg_growth'] = float(df[latest_growth].mean())
                metrics['median_growth'] = float(df[latest_growth].median())
                metrics['positive_growth'] = int((df[latest_growth] > 0).sum())
                metrics['negative_growth'] = int((df[latest_growth] < 0).sum())
                metrics['high_growth'] = int((df[latest_growth] > 20).sum())
                metrics['high_growth_pct'] = (metrics['high_growth'] / metrics['total_rows'] * 100) if metrics['total_rows'] > 0 else 0
            
            # Şirket metrikleri
            if 'Company' in df.columns and sales_cols:
                latest = sales_cols[-1]
                company_sales = df.groupby('Company')[latest].sum().sort_values(ascending=False)
                total = company_sales.sum()
                
                metrics['total_companies'] = len(company_sales)
                
                if total > 0:
                    shares = (company_sales / total * 100)
                    metrics['hhi_index'] = float((shares ** 2).sum())
                    
                    for n in [1, 3, 5, 10]:
                        if len(company_sales) >= n:
                            top_n_sum = company_sales.nlargest(n).sum()
                            metrics[f'top{n}_share'] = (top_n_sum / total * 100)
            
            # Molekül metrikleri
            if 'Molecule' in df.columns:
                metrics['unique_molecules'] = df['Molecule'].nunique()
                
                if sales_cols:
                    latest = sales_cols[-1]
                    mol_sales = df.groupby('Molecule')[latest].sum()
                    total = mol_sales.sum()
                    
                    if total > 0:
                        metrics['top10_molecule_share'] = (mol_sales.nlargest(10).sum() / total * 100)
            
            # Ülke metrikleri
            if 'Country' in df.columns:
                metrics['country_count'] = df['Country'].nunique()
                
                if sales_cols:
                    latest = sales_cols[-1]
                    country_sales = df.groupby('Country')[latest].sum()
                    total = country_sales.sum()
                    
                    if total > 0:
                        metrics['top5_country_share'] = (country_sales.nlargest(5).sum() / total * 100)
            
            # Fiyat metrikleri
            price_cols = [col for col in df.columns if 'Price_' in col]
            if price_cols:
                latest_price = price_cols[-1]
                metrics['avg_price'] = float(df[latest_price].mean())
                metrics['median_price'] = float(df[latest_price].median())
                metrics['std_price'] = float(df[latest_price].std())
                metrics['min_price'] = float(df[latest_price].min())
                metrics['max_price'] = float(df[latest_price].max())
            
            # Uluslararası ürün metrikleri
            if 'International_Product' in df.columns and sales_cols:
                latest = sales_cols[-1]
                
                intl_df = df[df['International_Product'] == 1]
                local_df = df[df['International_Product'] == 0]
                
                metrics['intl_product_count'] = len(intl_df)
                metrics['local_product_count'] = len(local_df)
                
                if len(intl_df) > 0:
                    metrics['intl_sales'] = float(intl_df[latest].sum())
                    metrics['local_sales'] = float(local_df[latest].sum())
                    
                    total = metrics['total_market_value']
                    if total > 0:
                        metrics['intl_share'] = (metrics['intl_sales'] / total * 100)
                        metrics['local_share'] = (metrics['local_sales'] / total * 100)
                
                if growth_cols and len(intl_df) > 0:
                    latest_growth = growth_cols[-1]
                    metrics['intl_avg_growth'] = float(intl_df[latest_growth].mean())
                    metrics['local_avg_growth'] = float(local_df[latest_growth].mean())
                
                if price_cols and len(intl_df) > 0:
                    latest_price = price_cols[-1]
                    metrics['intl_avg_price'] = float(intl_df[latest_price].mean())
                    metrics['local_avg_price'] = float(local_df[latest_price].mean())
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
        
        return metrics
    
    @staticmethod
    def anomaly_detection(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Isolation Forest ile anomali tespiti"""
        
        try:
            # Özellikleri seç
            features = []
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                features.append(sales_cols[-1])
            
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                features.append(growth_cols[-1])
            
            price_cols = [col for col in df.columns if 'Price_' in col]
            if price_cols:
                features.append(price_cols[-1])
            
            if 'Market_Share' in df.columns:
                features.append('Market_Share')
            
            if len(features) < 2:
                return None
            
            X = df[features].fillna(0)
            
            if len(X) < 10:
                return None
            
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100,
                n_jobs=-1
            )
            
            predictions = iso_forest.fit_predict(X)
            scores = iso_forest.score_samples(X)
            
            result_df = df.copy()
            result_df['Anomaly'] = predictions
            result_df['Anomaly_Score'] = scores
            
            # Risk kategorileri
            result_df['Risk_Level'] = pd.cut(
                scores,
                bins=[-np.inf, -0.5, -0.2, np.inf],
                labels=['Yüksek Risk', 'Orta Risk', 'Düşük Risk']
            )
            
            return result_df
            
        except Exception as e:
            st.warning(f"Anomali tespiti hatası: {str(e)}")
            return None
    
    @staticmethod
    def market_segmentation(
        df: pd.DataFrame,
        n_clusters: int = 4,
        method: str = 'kmeans'
    ) -> Optional[pd.DataFrame]:
        """Pazar segmentasyonu - çoklu algoritmalar"""
        
        try:
            # Özellikleri seç
            features = []
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                features.extend(sales_cols[-2:])
            
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                features.append(growth_cols[-1])
            
            price_cols = [col for col in df.columns if 'Price_' in col]
            if price_cols:
                features.append(price_cols[-1])
            
            if 'Market_Share' in df.columns:
                features.append('Market_Share')
            
            if len(features) < 2:
                return None
            
            X = df[features].fillna(0)
            
            if len(X) < n_clusters * 10:
                return None
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Clustering
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
            else:  # hierarchical
                model = AgglomerativeClustering(n_clusters=n_clusters)
            
            clusters = model.fit_predict(X_scaled)
            
            result_df = df.copy()
            result_df['Cluster'] = clusters
            
            # Cluster isimlendirme
            cluster_names = {
                0: '📈 Büyüyen',
                1: '💎 Premium',
                2: '📦 Hacimli',
                3: '⚠️ Riskli',
                4: '🎯 Niş',
                5: '💰 Kârlı',
                6: '📉 Düşüşte',
                7: '🆕 Yeni'
            }
            
            result_df['Segment'] = result_df['Cluster'].map(
                lambda x: cluster_names.get(x, f'Segment {x}')
            )
            
            # Kalite metrikleri
            try:
                if len(set(clusters)) > 1 and -1 not in set(clusters):
                    silhouette = silhouette_score(X_scaled, clusters)
                    result_df.attrs['silhouette_score'] = silhouette
            except:
                pass
            
            return result_df
            
        except Exception as e:
            st.warning(f"Segmentasyon hatası: {str(e)}")
            return None
    
    @staticmethod
    def forecast_trends(df: pd.DataFrame, periods: int = 3) -> Optional[pd.DataFrame]:
        """Holt-Winters ile trend tahmini"""
        
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if len(sales_cols) < 3:
                return None
            
            # Yıllık toplamları hesapla
            yearly_data = {}
            for col in sorted(sales_cols):
                year = DataProcessor.extract_year(col)
                if year:
                    yearly_data[year] = df[col].sum()
            
            if len(yearly_data) < 3:
                return None
            
            series = pd.Series(yearly_data)
            
            # Holt-Winters modeli
            try:
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal=None,
                    initialization_method='estimated'
                )
                fitted_model = model.fit()
                
                # Tahmin
                last_year = max(yearly_data.keys())
                forecast_years = [last_year + i + 1 for i in range(periods)]
                forecast_values = fitted_model.forecast(periods)
                
                # Basit güven aralığı
                residuals = fitted_model.fittedvalues - series
                std_residuals = np.std(residuals)
                conf_int = 1.96 * std_residuals
                
                forecast_df = pd.DataFrame({
                    'Yıl': forecast_years,
                    'Tahmin': forecast_values.values,
                    'Alt_Sınır': forecast_values.values - conf_int,
                    'Üst_Sınır': forecast_values.values + conf_int
                })
                
                # Büyüme oranı
                last_actual = series.iloc[-1]
                first_forecast = forecast_values.iloc[0]
                forecast_df['Büyüme'] = ((first_forecast - last_actual) / last_actual * 100)
                
                return forecast_df
                
            except:
                # Basit lineer trend
                x = np.arange(len(series))
                y = series.values
                
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                forecast_values = p(np.arange(len(series), len(series) + periods))
                last_year = max(yearly_data.keys())
                forecast_years = [last_year + i + 1 for i in range(periods)]
                
                forecast_df = pd.DataFrame({
                    'Yıl': forecast_years,
                    'Tahmin': forecast_values,
                    'Alt_Sınır': forecast_values * 0.9,
                    'Üst_Sınır': forecast_values * 1.1
                })
                
                return forecast_df
            
        except Exception as e:
            st.warning(f"Tahminleme hatası: {str(e)}")
            return None
    
    @staticmethod
    def generate_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
        """Stratejik içgörüler üret"""
        
        insights = []
        
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if not sales_cols:
                return insights
            
            latest = sales_cols[-1]
            year = DataProcessor.extract_year(latest) or "Son Yıl"
            total_market = df[latest].sum()
            
            # En iyi ürünler
            if 'Molecule' in df.columns:
                top_molecules = df.nlargest(5, latest)[['Molecule', latest]]
                top_share = (top_molecules[latest].sum() / total_market * 100) if total_market > 0 else 0
                
                insights.append({
                    'type': 'success',
                    'title': f'🏆 En İyi 5 Molekül - {year}',
                    'description': f"İlk 5 molekül pazarın %{top_share:.1f}'ini oluşturuyor."
                })
            
            # Büyüme liderleri
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            if growth_cols:
                latest_growth = growth_cols[-1]
                top_growth = df.nlargest(5, latest_growth)
                avg_top_growth = top_growth[latest_growth].mean()
                
                insights.append({
                    'type': 'info',
                    'title': '🚀 En Hızlı Büyüyen Ürünler',
                    'description': f"En hızlı büyüyen 5 ürün ortalama %{avg_top_growth:.1f} büyüme gösteriyor."
                })
            
            # Pazar lideri
            if 'Company' in df.columns:
                company_sales = df.groupby('Company')[latest].sum().nlargest(1)
                if len(company_sales) > 0:
                    top_company = company_sales.index[0]
                    top_share = (company_sales.iloc[0] / total_market * 100) if total_market > 0 else 0
                    
                    insights.append({
                        'type': 'warning',
                        'title': f'🏢 Pazar Lideri - {year}',
                        'description': f"{top_company} %{top_share:.1f} pazar payı ile lider."
                    })
            
            # En büyük pazar
            if 'Country' in df.columns:
                country_sales = df.groupby('Country')[latest].sum().nlargest(1)
                if len(country_sales) > 0:
                    top_country = country_sales.index[0]
                    top_share = (country_sales.iloc[0] / total_market * 100) if total_market > 0 else 0
                    
                    insights.append({
                        'type': 'geographic',
                        'title': '🌍 En Büyük Pazar',
                        'description': f"{top_country} %{top_share:.1f} pay ile en büyük pazar."
                    })
            
            # Fiyat analizi
            price_cols = [col for col in df.columns if 'Price_' in col]
            if price_cols:
                latest_price = price_cols[-1]
                avg_price = df[latest_price].mean()
                price_std = df[latest_price].std()
                
                insights.append({
                    'type': 'price',
                    'title': f'💰 Fiyat Analizi - {year}',
                    'description': f"Ortalama fiyat: ${avg_price:.2f} (Standart sapma: ${price_std:.2f})"
                })
            
            # Uluslararası analiz
            if 'International_Product' in df.columns:
                intl_count = (df['International_Product'] == 1).sum()
                intl_share = (intl_count / len(df) * 100) if len(df) > 0 else 0
                
                intl_sales = df[df['International_Product'] == 1][latest].sum()
                intl_sales_share = (intl_sales / total_market * 100) if total_market > 0 else 0
                
                insights.append({
                    'type': 'international',
                    'title': '🌐 Uluslararası Ürünler',
                    'description': f"%{intl_share:.1f} ürün uluslararası, pazar payı: %{intl_sales_share:.1f}"
                })
            
            # Yoğunlaşma analizi
            if len(sales_cols) >= 2:
                prev = sales_cols[-2]
                growth = ((total_market - df[prev].sum()) / df[prev].sum() * 100) if df[prev].sum() > 0 else 0
                
                trend = "büyüme" if growth > 0 else "daralma"
                insights.append({
                    'type': 'warning' if growth < 0 else 'success',
                    'title': f'📈 Pazar Trendi',
                    'description': f"Pazar %{abs(growth):.1f} oranında {trend} gösteriyor."
                })
            
        except Exception as e:
            st.warning(f"İçgörü üretme hatası: {str(e)}")
        
        return insights

# ============================================================================
# 6. PROFESYONEL GÖRSELLEŞTİRME MOTORU
# ============================================================================

class VisualizationEngine:
    """İleri seviye görselleştirme motoru"""
    
    @staticmethod
    def create_metrics_dashboard(df: pd.DataFrame, metrics: Dict[str, Any]):
        """Metrik gösterge paneli oluştur"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            market_value = metrics.get('total_market_value', 0)
            year = metrics.get('latest_year', '')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💰 TOPLAM PAZAR</div>
                <div class="metric-value">${market_value/1e6:.1f}M</div>
                <div class="metric-trend">
                    <span class="badge badge-primary">{year}</span>
                    <span>Yıllık ciro</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_growth = metrics.get('avg_growth', 0)
            growth_class = "badge-success" if avg_growth > 0 else "badge-danger" if avg_growth < 0 else "badge-warning"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📈 ORTALAMA BÜYÜME</div>
                <div class="metric-value">%{avg_growth:.1f}</div>
                <div class="metric-trend">
                    <span class="badge {growth_class}">Yıllık</span>
                    <span>YoY büyüme</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            hhi = metrics.get('hhi_index', 0)
            hhi_class = "badge-danger" if hhi > 2500 else "badge-warning" if hhi > 1500 else "badge-success"
            hhi_text = "Tekelleşmiş" if hhi > 2500 else "Yoğun" if hhi > 1500 else "Rekabetçi"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📊 HHI İNDEKSİ</div>
                <div class="metric-value">{hhi:.0f}</div>
                <div class="metric-trend">
                    <span class="badge {hhi_class}">{hhi_text}</span>
                    <span>Pazar yoğunluğu</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            intl_share = metrics.get('intl_share', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🌐 ULUSLARARASI PAY</div>
                <div class="metric-value">%{intl_share:.1f}</div>
                <div class="metric-trend">
                    <span class="badge badge-primary">Küresel</span>
                    <span>Toplam pazar</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # İkinci sıra
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            molecules = metrics.get('unique_molecules', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🧪 MOLEKÜLLER</div>
                <div class="metric-value">{molecules:,}</div>
                <div class="metric-trend">
                    <span class="badge badge-success">Benzersiz</span>
                    <span>Farklı molekül</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            avg_price = metrics.get('avg_price', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💰 ORTALAMA FİYAT</div>
                <div class="metric-value">${avg_price:.2f}</div>
                <div class="metric-trend">
                    <span class="badge badge-info">Birim</span>
                    <span>Ortalama</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            high_growth = metrics.get('high_growth_pct', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚀 YÜKSEK BÜYÜME</div>
                <div class="metric-value">%{high_growth:.1f}</div>
                <div class="metric-trend">
                    <span class="badge badge-success">>%20</span>
                    <span>Hızlı büyüyen</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            countries = metrics.get('country_count', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🌍 ÜLKE SAYISI</div>
                <div class="metric-value">{countries}</div>
                <div class="metric-trend">
                    <span class="badge badge-primary">Küresel</span>
                    <span>Pazar kapsamı</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def create_sales_trend(df: pd.DataFrame) -> Optional[go.Figure]:
        """Satış trend grafiği"""
        
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if len(sales_cols) < 2:
                return None
            
            trend_data = []
            for col in sorted(sales_cols):
                year = DataProcessor.extract_year(col)
                if year:
                    trend_data.append({
                        'Yıl': year,
                        'Toplam Satış': df[col].sum(),
                        'Ortalama Satış': df[col].mean(),
                        'Ürün Sayısı': (df[col] > 0).sum()
                    })
            
            if len(trend_data) < 2:
                return None
            
            trend_df = pd.DataFrame(trend_data)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Bar chart
            fig.add_trace(
                go.Bar(
                    x=trend_df['Yıl'],
                    y=trend_df['Toplam Satış'],
                    name='Toplam Satış',
                    marker_color='#2d7dd2',
                    text=[f'${x/1e6:.1f}M' for x in trend_df['Toplam Satış']],
                    textposition='auto'
                ),
                secondary_y=False
            )
            
            # Line chart
            fig.add_trace(
                go.Scatter(
                    x=trend_df['Yıl'],
                    y=trend_df['Ortalama Satış'],
                    name='Ortalama Satış',
                    mode='lines+markers',
                    line=dict(color='#2dd2a3', width=3),
                    marker=dict(size=10)
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title='📈 Pazar Büyüklüğü ve Ortalama Satış Trendi',
                height=500,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            fig.update_xaxes(title_text='Yıl', gridcolor='rgba(100,116,139,0.2)')
            fig.update_yaxes(title_text='Toplam Satış (USD)', secondary_y=False, gridcolor='rgba(100,116,139,0.2)')
            fig.update_yaxes(title_text='Ortalama Satış (USD)', secondary_y=True, gridcolor='rgba(100,116,139,0.2)')
            
            return fig
            
        except Exception as e:
            st.warning(f"Trend grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_market_share_chart(df: pd.DataFrame) -> Optional[go.Figure]:
        """Pazar payı dağılımı"""
        
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if not sales_cols or 'Company' not in df.columns:
                return None
            
            latest = sales_cols[-1]
            
            company_sales = df.groupby('Company')[latest].sum().sort_values(ascending=False)
            top_companies = company_sales.nlargest(10)
            other_sum = company_sales.iloc[10:].sum() if len(company_sales) > 10 else 0
            
            pie_data = top_companies.copy()
            if other_sum > 0:
                pie_data['Diğerleri'] = other_sum
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Pazar Payı Dağılımı', 'İlk 10 Şirket'),
                specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                column_widths=[0.4, 0.6]
            )
            
            fig.add_trace(
                go.Pie(
                    labels=pie_data.index,
                    values=pie_data.values,
                    hole=0.4,
                    textinfo='percent+label',
                    textposition='outside',
                    marker_colors=px.colors.qualitative.Set3
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=top_companies.values,
                    y=top_companies.index,
                    orientation='h',
                    marker_color='#2d7dd2',
                    text=[f'${x/1e6:.1f}M' for x in top_companies.values],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title='🏢 Pazar Yoğunlaşma Analizi',
                height=500,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            fig.update_xaxes(title_text='Satış (USD)', row=1, col=2, gridcolor='rgba(100,116,139,0.2)')
            fig.update_yaxes(title_text='', row=1, col=2, gridcolor='rgba(100,116,139,0.2)')
            
            return fig
            
        except Exception as e:
            st.warning(f"Pazar payı grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_world_map(df: pd.DataFrame) -> Optional[go.Figure]:
        """Dünya haritası görselleştirmesi"""
        
        try:
            if 'Country' not in df.columns:
                return None
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if not sales_cols:
                return None
            
            latest = sales_cols[-1]
            
            country_data = df.groupby('Country')[latest].sum().reset_index()
            country_data.columns = ['Ülke', 'Satış']
            
            # Ülke ismi düzeltmeleri
            country_mapping = {
                'USA': 'United States',
                'US': 'United States',
                'U.S.A': 'United States',
                'UK': 'United Kingdom',
                'U.K': 'United Kingdom',
                'UAE': 'United Arab Emirates',
                'Turkey': 'Türkiye',
                'Turkiye': 'Türkiye',
                'South Korea': 'South Korea'
            }
            
            country_data['Ülke'] = country_data['Ülke'].replace(country_mapping)
            
            fig = px.choropleth(
                country_data,
                locations='Ülke',
                locationmode='country names',
                color='Satış',
                hover_name='Ülke',
                color_continuous_scale='Viridis',
                title='🌍 Global Pazar Dağılımı'
            )
            
            fig.update_layout(
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                geo=dict(
                    bgcolor='rgba(0,0,0,0)',
                    lakecolor='#1e3a5f',
                    landcolor='#2d4a7a',
                    subunitcolor='#64748b'
                ),
                coloraxis_colorbar=dict(
                    title='Satış (USD)',
                    tickprefix='$'
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Harita hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_growth_matrix(df: pd.DataFrame) -> Optional[go.Figure]:
        """Büyüme-Pazar payı matrisi (BCG benzeri)"""
        
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            
            if not sales_cols or not growth_cols:
                return None
            
            latest_sales = sales_cols[-1]
            latest_growth = growth_cols[-1]
            
            plot_df = df.copy()
            
            # Pazar payı hesapla
            total_sales = plot_df[latest_sales].sum()
            plot_df['Pazar_Payı'] = (plot_df[latest_sales] / total_sales * 100) if total_sales > 0 else 0
            
            # Örnekleme (performans için)
            if len(plot_df) > 1000:
                plot_df = plot_df.sample(1000, random_state=42)
            
            fig = px.scatter(
                plot_df,
                x='Pazar_Payı',
                y=latest_growth,
                size=plot_df[latest_sales] / plot_df[latest_sales].max() * 100,
                color=latest_growth,
                hover_name='Molecule' if 'Molecule' in plot_df.columns else None,
                color_continuous_scale='RdYlGn',
                title='📊 Büyüme-Pazar Payı Matrisi',
                labels={
                    'Pazar_Payı': 'Pazar Payı (%)',
                    latest_growth: 'Büyüme Oranı (%)'
                }
            )
            
            # Referans çizgileri
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
            fig.add_vline(x=plot_df['Pazar_Payı'].mean(), line_dash="dash", line_color="white", opacity=0.3)
            
            fig.update_layout(
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                coloraxis_colorbar=dict(
                    title='Büyüme',
                    tickfont=dict(color='#f8fafc')
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Matris grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_anomaly_plot(anomaly_df: pd.DataFrame) -> Optional[go.Figure]:
        """Anomali görselleştirmesi"""
        
        try:
            sales_cols = [col for col in anomaly_df.columns if 'Sales_' in col]
            growth_cols = [col for col in anomaly_df.columns if 'Growth_' in col]
            
            if not sales_cols or not growth_cols or 'Risk_Level' not in anomaly_df.columns:
                return None
            
            latest_sales = sales_cols[-1]
            latest_growth = growth_cols[-1]
            
            # Örnekleme
            plot_df = anomaly_df
            if len(plot_df) > 2000:
                plot_df = plot_df.sample(2000, random_state=42)
            
            fig = px.scatter(
                plot_df,
                x=latest_sales,
                y=latest_growth,
                color='Risk_Level',
                size=abs(plot_df['Anomaly_Score']) * 10,
                hover_name='Molecule' if 'Molecule' in plot_df.columns else None,
                title='⚠️ Anomali Tespiti - Risk Analizi',
                labels={
                    latest_sales: 'Satış (USD)',
                    latest_growth: 'Büyüme (%)'
                },
                color_discrete_map={
                    'Yüksek Risk': '#eb5757',
                    'Orta Risk': '#f2c94c',
                    'Düşük Risk': '#2dd2a3'
                }
            )
            
            fig.update_layout(
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            fig.update_xaxes(type='log', gridcolor='rgba(100,116,139,0.2)')
            fig.update_yaxes(gridcolor='rgba(100,116,139,0.2)')
            
            return fig
            
        except Exception as e:
            st.warning(f"Anomali grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_forecast_plot(
        historical_df: pd.DataFrame,
        forecast_df: Optional[pd.DataFrame]
    ) -> Optional[go.Figure]:
        """Tahmin görselleştirmesi"""
        
        try:
            if forecast_df is None or len(forecast_df) == 0:
                return None
            
            sales_cols = [col for col in historical_df.columns if 'Sales_' in col]
            if not sales_cols:
                return None
            
            # Tarihsel veri
            historical = []
            for col in sorted(sales_cols):
                year = DataProcessor.extract_year(col)
                if year:
                    historical.append({
                        'Yıl': year,
                        'Gerçek': historical_df[col].sum(),
                        'Tür': 'Tarihsel'
                    })
            
            hist_df = pd.DataFrame(historical)
            
            # Tahmin verisi
            forecast_display = forecast_df.copy()
            forecast_display['Tür'] = 'Tahmin'
            
            fig = go.Figure()
            
            # Tarihsel
            fig.add_trace(go.Scatter(
                x=hist_df['Yıl'],
                y=hist_df['Gerçek'],
                mode='lines+markers',
                name='Tarihsel',
                line=dict(color='#2d7dd2', width=3),
                marker=dict(size=10)
            ))
            
            # Tahmin
            fig.add_trace(go.Scatter(
                x=forecast_display['Yıl'],
                y=forecast_display['Tahmin'],
                mode='lines+markers',
                name='Tahmin',
                line=dict(color='#2dd2a3', width=3, dash='dash'),
                marker=dict(size=10)
            ))
            
            # Güven aralığı
            fig.add_trace(go.Scatter(
                x=forecast_display['Yıl'].tolist() + forecast_display['Yıl'].tolist()[::-1],
                y=forecast_display['Üst_Sınır'].tolist() + forecast_display['Alt_Sınır'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(45,210,163,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='%95 Güven Aralığı'
            ))
            
            fig.update_layout(
                title='🔮 Pazar Tahmini - Holt-Winters',
                xaxis_title='Yıl',
                yaxis_title='Toplam Satış (USD)',
                height=500,
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Tahmin grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_cluster_plot(clustered_df: pd.DataFrame) -> Optional[go.Figure]:
        """Küme görselleştirmesi (PCA ile)"""
        
        try:
            if 'Cluster' not in clustered_df.columns:
                return None
            
            # Özellikleri seç
            numeric_cols = clustered_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if any(x in col for x in ['Sales_', 'Price_', 'Growth_', 'Market_Share'])]
            feature_cols = feature_cols[:5]  # İlk 5 özellik
            
            if len(feature_cols) < 2:
                return None
            
            X = clustered_df[feature_cols].fillna(0)
            
            # PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            plot_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Segment': clustered_df['Segment'] if 'Segment' in clustered_df.columns else clustered_df['Cluster'].astype(str),
                'Molecule': clustered_df['Molecule'] if 'Molecule' in clustered_df.columns else ''
            })
            
            fig = px.scatter(
                plot_df,
                x='PC1',
                y='PC2',
                color='Segment',
                hover_data=['Molecule'],
                title='🎯 PCA ile Segment Görselleştirmesi',
                labels={'PC1': 'Birinci Bileşen', 'PC2': 'İkinci Bileşen'}
            )
            
            fig.update_layout(
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Küme grafiği hatası: {str(e)}")
            return None

# ============================================================================
# 7. RAPORLAMA MOTORU
# ============================================================================

class ReportingEngine:
    """Profesyonel raporlama motoru"""
    
    @staticmethod
    def create_excel_report(
        df: pd.DataFrame,
        metrics: Dict[str, Any],
        insights: List[Dict[str, str]]
    ) -> BytesIO:
        """Excel raporu oluştur"""
        
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Formatlar
                header_format = workbook.add_format({
                    'bold': True,
                    'font_color': 'white',
                    'bg_color': '#2d7dd2',
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                
                money_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1})
                percent_format = workbook.add_format({'num_format': '0.00%', 'border': 1})
                
                # 1. Yönetici Özeti
                summary_data = [
                    ['Rapor Tarihi', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    ['Toplam Ürün Sayısı', f"{metrics.get('total_rows', 0):,}"],
                    ['Toplam Pazar Değeri', f"${metrics.get('total_market_value', 0)/1e6:.2f}M"],
                    ['Ortalama Büyüme', f"%{metrics.get('avg_growth', 0):.2f}"],
                    ['HHI İndeksi', f"{metrics.get('hhi_index', 0):.0f}"],
                    ['Benzersiz Molekül', f"{metrics.get('unique_molecules', 0):,}"],
                    ['Ülke Sayısı', metrics.get('country_count', 0)],
                    ['Uluslararası Ürün Payı', f"%{metrics.get('intl_share', 0):.2f}"]
                ]
                
                summary_df = pd.DataFrame(summary_data, columns=['Metrik', 'Değer'])
                summary_df.to_excel(writer, sheet_name='Yönetici Özeti', index=False)
                
                worksheet = writer.sheets['Yönetici Özeti']
                worksheet.set_column('A:A', 30)
                worksheet.set_column('B:B', 25)
                
                # 2. Detaylı Veri
                df.to_excel(writer, sheet_name='Ham Veri', index=False)
                worksheet = writer.sheets['Ham Veri']
                
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # 3. Stratejik İçgörüler
                if insights:
                    insights_df = pd.DataFrame(insights)
                    insights_df.columns = ['Tür', 'Başlık', 'Açıklama']
                    insights_df.to_excel(writer, sheet_name='İçgörüler', index=False)
                    
                    worksheet = writer.sheets['İçgörüler']
                    worksheet.set_column('A:A', 15)
                    worksheet.set_column('B:B', 30)
                    worksheet.set_column('C:C', 50)
                
                # 4. En İyi Ürünler
                sales_cols = [col for col in df.columns if 'Sales_' in col]
                if sales_cols:
                    latest = sales_cols[-1]
                    
                    if 'Molecule' in df.columns and 'Company' in df.columns:
                        top_products = df.nlargest(50, latest)[['Molecule', 'Company', latest]]
                    elif 'Molecule' in df.columns:
                        top_products = df.nlargest(50, latest)[['Molecule', latest]]
                    else:
                        top_products = df.nlargest(50, latest)[[latest]]
                    
                    top_products.to_excel(writer, sheet_name='İlk 50 Ürün', index=False)
                    
                    worksheet = writer.sheets['İlk 50 Ürün']
                    worksheet.set_column('A:A', 30)
                    worksheet.set_column('B:B', 30)
                    worksheet.set_column('C:C', 20, money_format)
                
                # 5. Şirket Analizi
                if 'Company' in df.columns and sales_cols:
                    company_analysis = df.groupby('Company')[latest].agg(['sum', 'mean', 'count']).round(2)
                    company_analysis.columns = ['Toplam Satış', 'Ortalama Satış', 'Ürün Sayısı']
                    company_analysis = company_analysis.sort_values('Toplam Satış', ascending=False)
                    company_analysis.to_excel(writer, sheet_name='Şirket Analizi')
                    
                    worksheet = writer.sheets['Şirket Analizi']
                    worksheet.set_column('A:A', 30)
                    worksheet.set_column('B:B', 20, money_format)
                    worksheet.set_column('C:C', 20, money_format)
                    worksheet.set_column('D:D', 15)
                
        except Exception as e:
            st.error(f"Excel raporu oluşturma hatası: {str(e)}")
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_html_report(
        df: pd.DataFrame,
        metrics: Dict[str, Any],
        insights: List[Dict[str, str]]
    ) -> str:
        """HTML raporu oluştur"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>PharmaIntelligence Pro Raporu</title>
            <style>
                body {{
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                    margin: 0;
                    padding: 40px;
                    background: linear-gradient(135deg, #0a1e3c, #1e3a5f);
                    color: #f8fafc;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: rgba(30, 58, 95, 0.8);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 40px;
                    border: 1px solid rgba(255,255,255,0.1);
                }}
                h1 {{
                    font-size: 2.5rem;
                    background: linear-gradient(135deg, #f8fafc, #cbd5e1);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 10px;
                }}
                .date {{
                    color: #2d7dd2;
                    margin-bottom: 30px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .metric-card {{
                    background: rgba(45,125,210,0.1);
                    border: 1px solid rgba(45,125,210,0.3);
                    border-radius: 12px;
                    padding: 20px;
                }}
                .metric-label {{
                    color: #cbd5e1;
                    font-size: 0.85rem;
                    text-transform: uppercase;
                    margin-bottom: 10px;
                }}
                .metric-value {{
                    font-size: 1.8rem;
                    font-weight: 700;
                }}
                .insights {{
                    margin-top: 30px;
                }}
                .insight-card {{
                    background: rgba(45,125,210,0.05);
                    border-left: 4px solid #2dd2a3;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 15px;
                }}
                .insight-title {{
                    font-weight: 600;
                    color: #2dd2a3;
                    margin-bottom: 5px;
                }}
                .footer {{
                    margin-top: 40px;
                    text-align: center;
                    color: #64748b;
                    font-size: 0.85rem;
                    border-top: 1px solid rgba(255,255,255,0.1);
                    padding-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>💊 PharmaIntelligence Pro</h1>
                <div class="date">Rapor Tarihi: {datetime.now().strftime('%d %B %Y, %H:%M')}</div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Toplam Pazar</div>
                        <div class="metric-value">${metrics.get('total_market_value', 0)/1e6:.1f}M</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Ortalama Büyüme</div>
                        <div class="metric-value">%{metrics.get('avg_growth', 0):.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">HHI İndeksi</div>
                        <div class="metric-value">{metrics.get('hhi_index', 0):.0f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Ülke Sayısı</div>
                        <div class="metric-value">{metrics.get('country_count', 0)}</div>
                    </div>
                </div>
        """
        
        # İçgörüler
        if insights:
            html += '<div class="insights"><h2>Stratejik İçgörüler</h2>'
            
            for insight in insights[:8]:
                html += f"""
                <div class="insight-card">
                    <div class="insight-title">{insight.get('title', 'İçgörü')}</div>
                    <div>{insight.get('description', '')}</div>
                </div>
                """
            
            html += '</div>'
        
        # Footer
        html += f"""
                <div class="footer">
                    © 2024 PharmaIntelligence Inc. | Versiyon 7.0<br>
                    Rapor ID: {hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12].upper()}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    @staticmethod
    def create_pdf_report(
        df: pd.DataFrame,
        metrics: Dict[str, Any],
        insights: List[Dict[str, str]]
    ) -> Optional[BytesIO]:
        """PDF raporu oluştur (reportlab varsa)"""
        
        if not REPORTLAB_AVAILABLE:
            return None
        
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            story = []
            
            # Başlık
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2d7dd2'),
                spaceAfter=20,
                alignment=1
            )
            
            title = Paragraph("PharmaIntelligence Pro", title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Tarih
            date_text = Paragraph(
                f"Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
                styles['Normal']
            )
            story.append(date_text)
            story.append(Spacer(1, 30))
            
            # Metrik tablosu
            metric_data = [
                ['Metrik', 'Değer'],
                ['Toplam Pazar', f"${metrics.get('total_market_value', 0)/1e6:.1f}M"],
                ['Ortalama Büyüme', f"%{metrics.get('avg_growth', 0):.1f}"],
                ['HHI İndeksi', f"{metrics.get('hhi_index', 0):.0f}"],
                ['Ülke Sayısı', str(metrics.get('country_count', 0))],
                ['Toplam Ürün', f"{metrics.get('total_rows', 0):,}"]
            ]
            
            metric_table = Table(metric_data, colWidths=[200, 200])
            metric_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#2d7dd2')),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#1e3a5f')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2d7dd2')),
            ]))
            
            story.append(metric_table)
            story.append(Spacer(1, 30))
            
            # İçgörüler
            if insights:
                story.append(Paragraph("Stratejik İçgörüler", styles['Heading2']))
                story.append(Spacer(1, 15))
                
                for insight in insights[:5]:
                    insight_text = Paragraph(
                        f"<b>{insight.get('title', 'İçgörü')}:</b> {insight.get('description', '')}",
                        styles['Normal']
                    )
                    story.append(insight_text)
                    story.append(Spacer(1, 10))
            
            # PDF oluştur
            doc.build(story)
            buffer.seek(0)
            
            return buffer
            
        except Exception as e:
            st.warning(f"PDF raporu oluşturulamadı: {str(e)}")
            return None

# ============================================================================
# 8. ANA UYGULAMA
# ============================================================================

def main():
    """Ana uygulama fonksiyonu"""
    
    # Session state başlat
    if 'data' not in st.session_state:
        st.session_state.data = None
        st.session_state.filtered_data = None
        st.session_state.metrics = {}
        st.session_state.insights = []
        st.session_state.anomalies = None
        st.session_state.clusters = None
        st.session_state.forecast = None
        st.session_state.active_filters = {}
    
    # Header
    st.markdown("""
    <div class="animate-fadeIn">
        <h1 class="pharma-title">💊 PHARMAINTELLIGENCE PRO</h1>
        <p class="pharma-subtitle">
        Yapay zeka destekli pazar analitiği, anomali tespiti, segmentasyon 
        ve profesyonel raporlama ile kurumsal ilaç pazarı istihbaratı.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 VERİ YÜKLEME")
        
        uploaded_file = st.file_uploader(
            "Excel veya CSV dosyası seçin",
            type=['xlsx', 'xls', 'csv'],
            help="Büyük veri setleri için optimize edilmiştir"
        )
        
        if uploaded_file:
            if st.button("🚀 VERİYİ YÜKLE & ANALİZ ET", type="primary", use_container_width=True):
                with st.spinner("📊 Veri işleniyor..."):
                    df = DataProcessor.load_data(uploaded_file)
                    
                    if df is not None:
                        df = DataProcessor.prepare_analytics(df)
                        
                        st.session_state.data = df
                        st.session_state.filtered_data = df.copy()
                        
                        analytics = AnalyticsEngine()
                        st.session_state.metrics = analytics.calculate_metrics(df)
                        st.session_state.insights = analytics.generate_insights(df)
                        
                        st.rerun()
        
        # Filtreler
        if st.session_state.data is not None:
            st.markdown("---")
            search_term, filter_config, apply, clear = FilterSystem.render_filters(
                st.session_state.data
            )
            
            if apply:
                filtered = FilterSystem.apply_filters(
                    st.session_state.data,
                    search_term,
                    filter_config
                )
                st.session_state.filtered_data = filtered
                st.session_state.active_filters = filter_config
                
                analytics = AnalyticsEngine()
                st.session_state.metrics = analytics.calculate_metrics(filtered)
                st.session_state.insights = analytics.generate_insights(filtered)
                
                st.rerun()
            
            if clear:
                st.session_state.filtered_data = st.session_state.data.copy()
                st.session_state.active_filters = {}
                
                analytics = AnalyticsEngine()
                st.session_state.metrics = analytics.calculate_metrics(st.session_state.data)
                st.session_state.insights = analytics.generate_insights(st.session_state.data)
                
                st.rerun()
        
        # Sistem bilgisi
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; font-size:0.8rem; color:#64748b;">
            <strong>PharmaIntelligence Pro v7.0</strong><br>
            Streamlit Cloud Optimized<br>
            © 2024 Tüm hakları saklıdır
        </div>
        """, unsafe_allow_html=True)
    
    # Ana içerik
    if st.session_state.data is None:
        # Hoşgeldin ekranı
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align:center; padding:3rem;">
                <div style="font-size:6rem; margin-bottom:1rem;">💊</div>
                <h2 style="color:#f8fafc; margin-bottom:1rem;">
                    PharmaIntelligence Pro'ya Hoş Geldiniz
                </h2>
                <p style="color:#cbd5e1; font-size:1.1rem; margin-bottom:2rem;">
                    Sol panelden bir veri seti yükleyerek gelişmiş analitik 
                    platformunu kullanmaya başlayın.
                </p>
                <div style="display:flex; justify-content:center; gap:1rem;">
                    <span class="badge badge-primary">Makine Öğrenimi</span>
                    <span class="badge badge-success">Segmentasyon</span>
                    <span class="badge badge-warning">Tahminleme</span>
                    <span class="badge badge-info">Anomali Tespiti</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    df = st.session_state.filtered_data
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    
    # Filtre durumu
    if st.session_state.active_filters:
        filter_text = "🎯 **Aktif Filtreler:** "
        filter_items = []
        
        for k, v in st.session_state.active_filters.items():
            if k == 'sales_range':
                (min_v, max_v), col = v
                filter_items.append(f"Satış: ${min_v:,.0f}-${max_v:,.0f}")
            elif k == 'growth_range':
                (min_v, max_v), col = v
                filter_items.append(f"Büyüme: %{min_v:.1f}-%{max_v:.1f}")
            elif k == 'international':
                filter_items.append(f"Uluslararası: {v}")
            elif isinstance(v, list):
                if len(v) > 3:
                    filter_items.append(f"{k}: {len(v)} seçim")
                else:
                    filter_items.append(f"{k}: {', '.join(v[:3])}")
        
        filter_text += " | ".join(filter_items)
        filter_text += f" | **Gösterilen:** {len(df):,} / {len(st.session_state.data):,} satır"
        
        st.markdown(f'<div class="filter-info">{filter_text}</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col2:
            if st.button("🧹 Filtreleri Temizle", use_container_width=True):
                st.session_state.filtered_data = st.session_state.data.copy()
                st.session_state.active_filters = {}
                
                analytics = AnalyticsEngine()
                st.session_state.metrics = analytics.calculate_metrics(st.session_state.data)
                st.session_state.insights = analytics.generate_insights(st.session_state.data)
                
                st.rerun()
    else:
        st.info(f"📊 **Gösterilen:** {len(df):,} satır (filtre uygulanmamış)")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ",
        "💰 FİYAT ANALİZİ",
        "🏢 REKABET",
        "🌍 ULUSLARARASI",
        "🔮 TAHMİNLEME",
        "⚠️ ANOMALİ TESPİTİ"
    ])
    
    with tab1:
        st.markdown('<h2 class="section-title">📊 Pazar Özet Metrikleri</h2>', unsafe_allow_html=True)
        
        viz = VisualizationEngine()
        viz.create_metrics_dashboard(df, metrics)
        
        st.markdown('<h2 class="section-title">💡 Stratejik İçgörüler</h2>', unsafe_allow_html=True)
        
        if insights:
            insight_cols = st.columns(2)
            
            for i, insight in enumerate(insights[:8]):
                with insight_cols[i % 2]:
                    insight_class = {
                        'success': 'insight-success',
                        'warning': 'insight-warning',
                        'danger': 'insight-danger',
                        'info': 'insight-info',
                        'geographic': 'insight-info',
                        'price': 'insight-info',
                        'international': 'insight-info'
                    }.get(insight.get('type', 'info'), 'insight-info')
                    
                    st.markdown(f"""
                    <div class="insight-card {insight_class}">
                        <h4 style="margin-bottom:0.5rem;">{insight.get('title', '')}</h4>
                        <p style="margin:0; color:#cbd5e1;">{insight.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="section-title">📋 Veri Önizleme</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_rows = st.slider("Gösterilecek satır", 10, 500, 50, 10)
            
            available_cols = df.columns.tolist()
            default_cols = []
            
            for col in ['Molecule', 'Company', 'Country', 'Sales_2024', 'Growth_2024']:
                if col in available_cols:
                    default_cols.append(col)
                    if len(default_cols) >= 5:
                        break
            
            if len(default_cols) < 5:
                default_cols = available_cols[:5]
            
            selected_cols = st.multiselect(
                "Sütunlar",
                options=available_cols,
                default=default_cols
            )
        
        with col2:
            if selected_cols:
                st.dataframe(df[selected_cols].head(n_rows), width='stretch', height=400)
            else:
                st.dataframe(df.head(n_rows), width='stretch', height=400)
    
    with tab2:
        st.markdown('<h2 class="section-title">📈 Pazar Trendleri</h2>', unsafe_allow_html=True)
        
        viz = VisualizationEngine()
        
        trend_fig = viz.create_sales_trend(df)
        if trend_fig:
            st.plotly_chart(trend_fig, width='stretch', config={'displayModeBar': True})
        
        st.markdown('<h2 class="section-title">📊 Büyüme-Pazar Payı Matrisi</h2>', unsafe_allow_html=True)
        
        matrix_fig = viz.create_growth_matrix(df)
        if matrix_fig:
            st.plotly_chart(matrix_fig, width='stretch', config={'displayModeBar': True})
        else:
            st.info("Büyüme matrisi için yeterli veri yok.")
    
    with tab3:
        st.markdown('<h2 class="section-title">💰 Fiyat Analizi</h2>', unsafe_allow_html=True)
        
        price_cols = [col for col in df.columns if 'Price_' in col]
        
        if not price_cols:
            st.warning("Fiyat analizi için veri setinde fiyat sütunları bulunamadı.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                latest_price = price_cols[-1]
                
                fig = px.histogram(
                    df,
                    x=latest_price,
                    nbins=50,
                    title='Fiyat Dağılımı',
                    labels={latest_price: 'Fiyat (USD)'}
                )
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Fiyat segmentleri
                price_data = df[latest_price].dropna()
                
                if len(price_data) > 0:
                    segments = pd.cut(
                        price_data,
                        bins=[0, 10, 50, 100, 500, float('inf')],
                        labels=['Ekonomik (<$10)', 'Standart ($10-50)', 'Premium ($50-100)',
                               'Süper Premium ($100-500)', 'Lüks (>$500)']
                    )
                    
                    segment_counts = segments.value_counts()
                    
                    fig = px.bar(
                        x=segment_counts.index,
                        y=segment_counts.values,
                        title='Fiyat Segmentleri',
                        labels={'x': 'Segment', 'y': 'Ürün Sayısı'}
                    )
                    fig.update_layout(
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f8fafc'
                    )
                    st.plotly_chart(fig, width='stretch')
    
    with tab4:
        st.markdown('<h2 class="section-title">🏢 Rekabet Analizi</h2>', unsafe_allow_html=True)
        
        viz = VisualizationEngine()
        
        share_fig = viz.create_market_share_chart(df)
        if share_fig:
            st.plotly_chart(share_fig, width='stretch', config={'displayModeBar': True})
        else:
            st.info("Pazar payı analizi için yeterli veri yok.")
        
        # Segmentasyon
        st.markdown('<h2 class="section-title">🎯 Pazar Segmentasyonu</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Segment Sayısı", 2, 8, 4)
        with col2:
            method = st.selectbox(
                "Algoritma",
                ["kmeans", "hierarchical", "dbscan"],
                format_func=lambda x: {
                    'kmeans': 'K-Means', 
                    'hierarchical': 'Hiyerarşik', 
                    'dbscan': 'DBSCAN'
                }[x]
            )
        
        if st.button("🔍 SEGMENTASYONU BAŞLAT", type="primary", use_container_width=True):
            with st.spinner("Segmentasyon yapılıyor..."):
                analytics = AnalyticsEngine()
                clustered = analytics.market_segmentation(df, n_clusters, method)
                
                if clustered is not None:
                    st.session_state.clusters = clustered
                    st.success(f"✅ {n_clusters} segment oluşturuldu!")
                    
                    if 'silhouette_score' in clustered.attrs:
                        st.metric("Silhouette Skoru", f"{clustered.attrs['silhouette_score']:.3f}")
                else:
                    st.error("Segmentasyon yapılamadı. Daha fazla veri gerekli.")
        
        if st.session_state.clusters is not None:
            cluster_fig = viz.create_cluster_plot(st.session_state.clusters)
            if cluster_fig:
                st.plotly_chart(cluster_fig, width='stretch', config={'displayModeBar': True})
            
            # Segment istatistikleri
            st.markdown("### 📊 Segment İstatistikleri")
            
            cluster_df = st.session_state.clusters
            segment_col = 'Segment' if 'Segment' in cluster_df.columns else 'Cluster'
            
            stats = []
            for segment in sorted(cluster_df[segment_col].unique()):
                seg_df = cluster_df[cluster_df[segment_col] == segment]
                
                stat = {'Segment': segment, 'Ürün Sayısı': len(seg_df)}
                
                sales_cols = [col for col in df.columns if 'Sales_' in col]
                if sales_cols:
                    latest = sales_cols[-1]
                    stat['Ortalama Satış'] = seg_df[latest].mean()
                
                stats.append(stat)
            
            stats_df = pd.DataFrame(stats)
            
            if 'Ortalama Satış' in stats_df.columns:
                stats_df['Ortalama Satış'] = stats_df['Ortalama Satış'].apply(
                    lambda x: f'${x:,.0f}' if pd.notnull(x) else 'N/A'
                )
            
            st.dataframe(stats_df, width='stretch')
    
    with tab5:
        st.markdown('<h2 class="section-title">🌍 Uluslararası Pazar Analizi</h2>', unsafe_allow_html=True)
        
        if 'International_Product' not in df.columns:
            st.warning("Veri setinde 'International_Product' sütunu bulunamadı.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                intl_count = (df['International_Product'] == 1).sum()
                st.metric("Uluslararası Ürün", f"{intl_count:,}")
            
            with col2:
                local_count = (df['International_Product'] == 0).sum()
                st.metric("Yerel Ürün", f"{local_count:,}")
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                latest = sales_cols[-1]
                
                intl_sales = df[df['International_Product'] == 1][latest].sum()
                local_sales = df[df['International_Product'] == 0][latest].sum()
                total = intl_sales + local_sales
                
                with col3:
                    if total > 0:
                        intl_share = (intl_sales / total * 100)
                        st.metric("Uluslararası Pay", f"%{intl_share:.1f}")
                
                with col4:
                    if total > 0:
                        local_share = (local_sales / total * 100)
                        st.metric("Yerel Pay", f"%{local_share:.1f}")
                
                # Karşılaştırmalı grafik
                comp_data = pd.DataFrame({
                    'Kategori': ['Uluslararası', 'Yerel'],
                    'Satış': [intl_sales, local_sales],
                    'Ürün Sayısı': [intl_count, local_count]
                })
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Satış Karşılaştırması', 'Ürün Sayısı Karşılaştırması'),
                    specs=[[{'type': 'pie'}, {'type': 'bar'}]]
                )
                
                fig.add_trace(
                    go.Pie(
                        labels=comp_data['Kategori'],
                        values=comp_data['Satış'],
                        hole=0.4,
                        textinfo='percent+label'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=comp_data['Kategori'],
                        y=comp_data['Ürün Sayısı'],
                        marker_color=['#2d7dd2', '#2dd2a3'],
                        text=comp_data['Ürün Sayısı'],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    showlegend=False
                )
                
                st.plotly_chart(fig, width='stretch')
            
            # Dünya haritası
            world_fig = viz.create_world_map(df)
            if world_fig:
                st.markdown('<h2 class="section-title">🌍 Coğrafi Dağılım</h2>', unsafe_allow_html=True)
                st.plotly_chart(world_fig, width='stretch', config={'displayModeBar': True})
    
    with tab6:
        st.markdown('<h2 class="section-title">🔮 Pazar Tahminleme</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-card insight-info">
            <h4 style="margin-bottom:0.5rem;">📊 Holt-Winters Tahmin Metodu</h4>
            <p style="margin:0; color:#cbd5e1;">
            Tarihsel trendlere dayalı olarak gelecek pazar değerlerini tahmin eder.
            En az 3 yıllık veri gereklidir.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            periods = st.slider("Tahmin Dönemi (Yıl)", 1, 5, 2)
            
            if st.button("🔮 TAHMİN OLUŞTUR", type="primary", use_container_width=True):
                with st.spinner("Tahmin yapılıyor..."):
                    analytics = AnalyticsEngine()
                    forecast = analytics.forecast_trends(df, periods)
                    
                    if forecast is not None:
                        st.session_state.forecast = forecast
                        st.success("✅ Tahmin oluşturuldu!")
                    else:
                        st.error("Tahmin yapılamadı. En az 3 yıllık veri gerekli.")
        
        with col2:
            if st.session_state.forecast is not None:
                forecast_fig = viz.create_forecast_plot(df, st.session_state.forecast)
                if forecast_fig:
                    st.plotly_chart(forecast_fig, width='stretch', config={'displayModeBar': True})
        
        if st.session_state.forecast is not None:
            st.markdown("### 📊 Tahmin Detayları")
            
            forecast_display = st.session_state.forecast.copy()
            forecast_display['Tahmin'] = forecast_display['Tahmin'].apply(lambda x: f'${x/1e6:.2f}M')
            forecast_display['Alt_Sınır'] = forecast_display['Alt_Sınır'].apply(lambda x: f'${x/1e6:.2f}M')
            forecast_display['Üst_Sınır'] = forecast_display['Üst_Sınır'].apply(lambda x: f'${x/1e6:.2f}M')
            
            if 'Büyüme' in forecast_display.columns:
                forecast_display['Büyüme'] = forecast_display['Büyüme'].apply(lambda x: f'%{x:.1f}')
            
            st.dataframe(forecast_display, width='stretch')
    
    with tab7:
        st.markdown('<h2 class="section-title">⚠️ Anomali Tespiti</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-card insight-warning">
            <h4 style="margin-bottom:0.5rem;">🔍 Isolation Forest Algoritması</h4>
            <p style="margin:0; color:#cbd5e1;">
            Pazardaki aykırı değerleri ve olağandışı kalıpları tespit eder.
            Yüksek riskli ürünler özel inceleme gerektirir.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            contamination = st.slider(
                "Anomali Oranı",
                min_value=0.01,
                max_value=0.3,
                value=0.1,
                step=0.01,
                format="%.2f"
            )
            
            if st.button("🔍 ANOMALİ TESPİT ET", type="primary", use_container_width=True):
                with st.spinner("Anomaliler tespit ediliyor..."):
                    analytics = AnalyticsEngine()
                    anomalies = analytics.anomaly_detection(df)
                    
                    if anomalies is not None:
                        st.session_state.anomalies = anomalies
                        
                        n_anomalies = (anomalies['Anomaly'] == -1).sum()
                        st.success(f"✅ {n_anomalies} anomali tespit edildi (%{n_anomalies/len(anomalies)*100:.1f})")
                    else:
                        st.error("Anomali tespiti yapılamadı. Yeterli veri yok.")
        
        with col2:
            if st.session_state.anomalies is not None:
                anomaly_fig = viz.create_anomaly_plot(st.session_state.anomalies)
                if anomaly_fig:
                    st.plotly_chart(anomaly_fig, width='stretch', config={'displayModeBar': True})
        
        if st.session_state.anomalies is not None:
            st.markdown("### ⚠️ Yüksek Riskli Ürünler")
            
            anomaly_df = st.session_state.anomalies
            high_risk = anomaly_df[anomaly_df['Risk_Level'] == 'Yüksek Risk']
            
            if len(high_risk) > 0:
                display_cols = []
                
                for col in ['Molecule', 'Company', 'Country', 'Risk_Level', 'Anomaly_Score']:
                    if col in high_risk.columns:
                        display_cols.append(col)
                
                sales_cols = [col for col in anomaly_df.columns if 'Sales_' in col]
                if sales_cols:
                    display_cols.append(sales_cols[-1])
                
                st.dataframe(
                    high_risk[display_cols].sort_values('Anomaly_Score').head(20),
                    width='stretch',
                    height=400
                )
            else:
                st.info("Yüksek riskli anomali tespit edilmedi.")
    
    # Footer'da raporlama butonları
    st.markdown("---")
    st.markdown('<h2 class="section-title">📑 RAPORLAMA</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Excel Raporu", use_container_width=True):
            with st.spinner("Excel raporu oluşturuluyor..."):
                report_engine = ReportingEngine()
                excel_data = report_engine.create_excel_report(df, metrics, insights)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="⬇️ Excel'i İndir",
                    data=excel_data,
                    file_name=f"pharma_rapor_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    with col2:
        if st.button("🌐 HTML Raporu", use_container_width=True):
            with st.spinner("HTML raporu oluşturuluyor..."):
                report_engine = ReportingEngine()
                html_data = report_engine.create_html_report(df, metrics, insights)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="⬇️ HTML'i İndir",
                    data=html_data.encode('utf-8'),
                    file_name=f"pharma_rapor_{timestamp}.html",
                    mime="text/html",
                    use_container_width=True
                )
    
    with col3:
        if st.button("📄 PDF Raporu", use_container_width=True):
            if not REPORTLAB_AVAILABLE:
                st.warning("PDF raporu için reportlab kütüphanesi gerekli.")
            else:
                with st.spinner("PDF raporu oluşturuluyor..."):
                    report_engine = ReportingEngine()
                    pdf_data = report_engine.create_pdf_report(df, metrics, insights)
                    
                    if pdf_data:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        st.download_button(
                            label="⬇️ PDF'i İndir",
                            data=pdf_data,
                            file_name=f"pharma_rapor_{timestamp}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
    
    with col4:
        if st.button("🔄 Sıfırla", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != 'app_started':
                    del st.session_state[key]
            st.rerun()

# ============================================================================
# 9. UYGULAMA BAŞLATMA
# ============================================================================

if __name__ == "__main__":
    try:
        gc.enable()
        st.session_state.setdefault('app_started', True)
        main()
    except Exception as e:
        st.error(f"❌ Uygulama hatası: {str(e)}")
        st.error("Detaylı hata bilgisi:")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Uygulamayı Yeniden Başlat", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
