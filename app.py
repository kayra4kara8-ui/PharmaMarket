# app.py - PROFESYONEL İLAÇ PAZARI ANALİTİK PLATFORMU
# Enterprise Level - International Product Analizi Dahil
# Kod Uzunluğu: ~4200 satır
# Tarih: 2024
# Geliştirici: PharmaIntelligence AI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Gelişmiş analitik
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from scipy import stats, signal, interpolate
import ruptures as rpt  # Change point detection

# Yardımcı araçlar
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
import re
import os
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import base64
from pathlib import Path
import inspect
from functools import wraps, lru_cache
import concurrent.futures
from collections import defaultdict, Counter
import itertools

# Excel export için
import xlsxwriter
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side

# ================================================
# 0. SABİTLER VE KONFİGÜRASYON
# ================================================

class Config:
    """Uygulama konfigürasyon sabitleri"""
    # Performans ayarları
    MAX_ROWS_PREVIEW = 5000
    MAX_ROWS_PROCESSING = 1000000
    CHUNK_SIZE = 50000
    CACHE_TTL = 3600  # seconds
    MAX_CACHE_ENTRIES = 20
    
    # Analiz ayarları
    FORECAST_YEARS = 2
    CLUSTER_RANGE = (2, 8)
    ANOMALY_THRESHOLD = 0.95
    SIGNIFICANCE_LEVEL = 0.05
    
    # Renk şeması
    COLORS = {
        'primary_dark': '#0c1a32',
        'secondary_dark': '#14274e',
        'accent_blue': '#2d7dd2',
        'accent_cyan': '#2acaea',
        'accent_teal': '#30c9c9',
        'success': '#2dd2a3',
        'warning': '#f2c94c',
        'danger': '#eb5757',
        'info': '#2d7dd2'
    }
    
    # Stil ayarları
    CHART_TEMPLATE = 'plotly_dark'
    CHART_HEIGHT = 500
    CHART_WIDTH = None  # Responsive

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
        ### PharmaIntelligence Enterprise v6.0
        • International Product Analytics
        • Predictive Modeling & Forecasting
        • Anomaly Detection & Monitoring
        • Advanced Segmentation
        • Automated Reporting
        • Machine Learning Integration
        © 2024 PharmaIntelligence Inc. All Rights Reserved
        """
    }
)

# PROFESYONEL MAVİ TEMA CSS STYLES - GELİŞTİRİLMİŞ
PROFESSIONAL_CSS = f"""
<style>
    /* === ROOT VARIABLES === */
    :root {{
        --primary-dark: {Config.COLORS['primary_dark']};
        --secondary-dark: {Config.COLORS['secondary_dark']};
        --accent-blue: {Config.COLORS['accent_blue']};
        --accent-blue-light: #4a9fe3;
        --accent-blue-dark: #1a5fa0;
        --accent-cyan: {Config.COLORS['accent_cyan']};
        --accent-teal: {Config.COLORS['accent_teal']};
        --success: {Config.COLORS['success']};
        --warning: {Config.COLORS['warning']};
        --danger: {Config.COLORS['danger']};
        --info: {Config.COLORS['info']};
        
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        
        --bg-primary: {Config.COLORS['primary_dark']};
        --bg-secondary: {Config.COLORS['secondary_dark']};
        --bg-card: #1e3a5f;
        --bg-hover: #2d4a7a;
        --bg-surface: {Config.COLORS['secondary_dark']};
        
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.4);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.5);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.6);
        --shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.7);
        --shadow-glass: 0 8px 32px rgba(31, 38, 135, 0.37);
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        
        --transition-fast: 200ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
        
        --glass-blur: blur(10px);
        --glass-border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* === GLOBAL STYLES === */
    .stApp {{
        background: linear-gradient(135deg, var(--primary-dark), var(--secondary-dark));
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text-primary);
        min-height: 100vh;
    }}
    
    /* Glassmorphism effects */
    .glass-card {{
        background: rgba(30, 58, 95, 0.7);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border-radius: var(--radius-lg);
        border: var(--glass-border);
        box-shadow: var(--shadow-glass);
    }}
    
    /* Streamlit component overrides */
    .stDataFrame, .stTable {{
        background: var(--bg-card) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--bg-hover) !important;
    }}
    
    /* === TYPOGRAPHY === */
    .pharma-title {{
        font-size: 2.8rem;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan), var(--accent-teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }}
    
    /* === ANIMATIONS === */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .animate-fade-in {{
        animation: fadeIn 0.6s ease-out;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}
    
    .pulse-animation {{
        animation: pulse 2s ease-in-out infinite;
    }}
    
    /* === CUSTOM COMPONENTS === */
    .dashboard-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    .alert-box {{
        padding: 1rem;
        border-radius: var(--radius-md);
        margin: 1rem 0;
        border-left: 4px solid;
        background: rgba(255, 255, 255, 0.05);
    }}
    
    .alert-info {{ border-left-color: var(--accent-blue); }}
    .alert-success {{ border-left-color: var(--success); }}
    .alert-warning {{ border-left-color: var(--warning); }}
    .alert-danger {{ border-left-color: var(--danger); }}
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ================================================
# 2. MİMARİ SINIFLAR
# ================================================

class DataProcessor:
    """Veri işleme ve optimizasyon sınıfı"""
    
    def __init__(self):
        self.column_mappings = {
            'SATIŞ': ['Satış', 'Sales', 'Revenue', 'Turnover'],
            'BİRİM': ['Birim', 'Units', 'Quantity', 'Volume'],
            'FİYAT': ['Fiyat', 'Price', 'Cost', 'Value'],
            'MOLEKÜL': ['Molekül', 'Molecule', 'Active Ingredient'],
            'ŞİRKET': ['Şirket', 'Company', 'Corporation', 'Manufacturer'],
            'ÜLKE': ['Ülke', 'Country', 'Region', 'Market']
        }
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, max_entries=Config.MAX_CACHE_ENTRIES, show_spinner=False)
    def load_large_dataset(file, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Büyük veri setlerini optimize şekilde yükle
        
        Parameters:
        -----------
        file : UploadedFile
            Yüklenen dosya
        sample_size : int, optional
            Örneklem boyutu
            
        Returns:
        --------
        pd.DataFrame
            Yüklenen veri seti
        """
        try:
            start_time = time.time()
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                if sample_size:
                    # Chunk-based reading for large CSV files
                    chunks = []
                    chunk_size = min(sample_size, Config.CHUNK_SIZE)
                    with st.spinner(f"📥 CSV verisi yükleniyor..."):
                        for chunk in pd.read_csv(file, chunksize=chunk_size, low_memory=False):
                            chunks.append(chunk)
                            if sum(len(c) for c in chunks) >= sample_size:
                                break
                    df = pd.concat(chunks, ignore_index=True).iloc[:sample_size]
                else:
                    # Use optimized reading
                    with st.spinner("📥 CSV verisi yükleniyor..."):
                        df = pd.read_csv(file, low_memory=False, 
                                         encoding_errors='ignore',
                                         on_bad_lines='warn')
            
            elif file_extension in ['xlsx', 'xls']:
                if sample_size:
                    # Read Excel in chunks
                    chunks = []
                    rows_read = 0
                    
                    with st.spinner(f"📥 Büyük Excel verisi yükleniyor..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # First, get total rows
                        total_rows = pd.read_excel(file, nrows=1).shape[0]
                        chunk_size = min(Config.CHUNK_SIZE, sample_size)
                        
                        for chunk_start in range(0, sample_size, chunk_size):
                            chunk = pd.read_excel(
                                file,
                                skiprows=chunk_start,
                                nrows=chunk_size,
                                engine='openpyxl'
                            )
                            
                            if chunk.empty:
                                break
                            
                            chunks.append(chunk)
                            rows_read += len(chunk)
                            
                            progress = min(rows_read / sample_size, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"📊 {rows_read:,} satır yüklendi...")
                            
                            if rows_read >= sample_size:
                                break
                        
                        df = pd.concat(chunks, ignore_index=True)
                        progress_bar.progress(1.0)
                        status_text.text(f"✅ {len(df):,} satır başarıyla yüklendi")
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                else:
                    # Read entire file
                    with st.spinner("📥 Excel verisi yükleniyor..."):
                        df = pd.read_excel(file, engine='openpyxl')
            
            else:
                st.error(f"Desteklenmeyen dosya formatı: {file_extension}")
                return None
            
            # Optimize dataframe
            df = DataProcessor.optimize_dataframe(df)
            
            load_time = time.time() - start_time
            st.success(f"✅ Veri yükleme tamamlandı: {len(df):,} satır, {len(df.columns)} sütun ({load_time:.2f}s)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.error(f"Detay: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame'i bellek kullanımı açısından optimize et
        
        Parameters:
        -----------
        df : pd.DataFrame
            Optimize edilecek dataframe
            
        Returns:
        --------
        pd.DataFrame
            Optimize edilmiş dataframe
        """
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Clean column names
            df.columns = DataProcessor.clean_column_names(df.columns)
            
            # Optimize data types
            with st.spinner("Veri seti optimize ediliyor..."):
                
                # String columns
                for col in df.select_dtypes(include=['object']).columns:
                    unique_count = df[col].nunique()
                    total_rows = len(df)
                    
                    if unique_count / total_rows < 0.5:  # Low cardinality
                        df[col] = df[col].astype('category')
                    else:
                        # Clean strings
                        df[col] = df[col].astype(str).str.strip()
                
                # Numeric columns - FIXED: Use pd.api.types for safe checking
                for col in df.select_dtypes(include=[np.number]).columns:
                    try:
                        col_min = df[col].min()
                        col_max = df[col].max()
                        
                        # Safe type checking using pd.api.types
                        if pd.api.types.is_integer_dtype(df[col]):
                            if col_min >= 0:
                                if col_max <= 255:
                                    df[col] = pd.to_numeric(df[col], downcast='unsigned')
                                elif col_max <= 65535:
                                    df[col] = pd.to_numeric(df[col], downcast='unsigned')
                            else:
                                if col_min >= -128 and col_max <= 127:
                                    df[col] = pd.to_numeric(df[col], downcast='signed')
                                elif col_min >= -32768 and col_max <= 32767:
                                    df[col] = pd.to_numeric(df[col], downcast='signed')
                        else:
                            # Float columns
                            df[col] = pd.to_numeric(df[col], downcast='float')
                    except Exception as e:
                        st.warning(f"Sütun {col} optimizasyonu başarısız: {str(e)}")
                        continue
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_saving = original_memory - optimized_memory
            
            if memory_saving > 0:
                st.info(f"💾 Bellek optimizasyonu: {original_memory:.1f}MB → {optimized_memory:.1f}MB "
                       f"(%{(memory_saving/original_memory*100):.1f} tasarruf)")
            
            return df
            
        except Exception as e:
            st.warning(f"Optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def clean_column_names(columns: List[str]) -> List[str]:
        """
        Sütun isimlerini temizle ve standardize et
        
        Parameters:
        -----------
        columns : List[str]
            Orijinal sütun isimleri
            
        Returns:
        --------
        List[str]
            Temizlenmiş sütun isimleri
        """
        cleaned = []
        seen_names = {}
        
        for col in columns:
            if not isinstance(col, str):
                col = str(col)
            
            # Turkish characters
            turkish_to_english = {
                'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
            }
            
            for tr, en in turkish_to_english.items():
                col = col.replace(tr, en)
            
            # Clean special characters and whitespace
            col = re.sub(r'[\n\r\t]', ' ', col)
            col = ' '.join(col.split())
            col = re.sub(r'[^\w\s]', '_', col)
            col = re.sub(r'\s+', '_', col)
            col = col.strip('_')
            
            # Extract year using regex (FIXED)
            year_match = re.search(r'\b(20\d{2})\b', col)
            if year_match:
                year = year_match.group(1)
                # Remove year from base name
                base_name = re.sub(r'\b(20\d{2})\b', '', col).strip('_')
                col = f"{base_name}_{year}" if base_name else f"Year_{year}"
            else:
                # No year found, keep as is
                pass
            
            # Ensure uniqueness (FIXED: De-duplication mechanism)
            original_col = col
            counter = 1
            while col in seen_names:
                col = f"{original_col}_{counter}"
                counter += 1
            seen_names[col] = True
            
            # Standardize common column names
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['sales', 'satış', 'revenue']):
                col = 'Satış'
            elif any(keyword in col_lower for keyword in ['units', 'birim', 'quantity']):
                col = 'Birim'
            elif any(keyword in col_lower for keyword in ['price', 'fiyat', 'cost']):
                col = 'Fiyat'
            elif any(keyword in col_lower for keyword in ['molecule', 'molekül', 'ingredient']):
                col = 'Molekül'
            elif any(keyword in col_lower for keyword in ['company', 'şirket', 'corporation']):
                col = 'Şirket'
            elif any(keyword in col_lower for keyword in ['country', 'ülke', 'market']):
                col = 'Ülke'
            elif any(keyword in col_lower for keyword in ['region', 'bölge']):
                col = 'Bölge'
            elif any(keyword in col_lower for keyword in ['product', 'ürün']):
                col = 'Ürün'
            
            cleaned.append(col)
        
        return cleaned
    
    @staticmethod
    def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analiz için veriyi hazırla
        
        Parameters:
        -----------
        df : pd.DataFrame
            Ham veri
            
        Returns:
        --------
        pd.DataFrame
            Analiz için hazırlanmış veri
        """
        try:
            # Identify sales columns using regex
            sales_columns = [col for col in df.columns 
                           if re.search(r'(satış|sales|revenue).*\d{4}', col, re.IGNORECASE)]
            
            if not sales_columns:
                # Try alternative patterns
                sales_columns = [col for col in df.columns 
                               if re.search(r'\d{4}', col) and 
                               any(keyword in col.lower() for keyword in ['value', 'amount', 'total'])]
            
            if not sales_columns:
                st.warning("⚠️ Satış sütunları bulunamadı. Veri yapınızı kontrol edin.")
                return df
            
            # Extract years from column names
            years = []
            for col in sales_columns:
                year_match = re.search(r'\b(20\d{2})\b', col)
                if year_match:
                    try:
                        years.append(int(year_match.group(1)))
                    except:
                        continue
            
            if not years:
                st.warning("⚠️ Sütun isimlerinden yıl bilgisi çıkarılamadı.")
                return df
            
            years = sorted(set(years))
            
            # Calculate growth rates
            for i in range(1, len(years)):
                prev_year = str(years[i-1])
                curr_year = str(years[i])
                
                prev_col = next((col for col in sales_columns if prev_year in col), None)
                curr_col = next((col for col in sales_columns if curr_year in col), None)
                
                if prev_col and curr_col and prev_col in df.columns and curr_col in df.columns:
                    # Safe division avoiding zero
                    with np.errstate(divide='ignore', invalid='ignore'):
                        growth = np.where(
                            df[prev_col] != 0,
                            ((df[curr_col] - df[prev_col]) / df[prev_col]) * 100,
                            np.nan
                        )
                    
                    df[f'Büyüme_{prev_year}_{curr_year}'] = pd.Series(growth, dtype=np.float32)
            
            # Calculate CAGR
            if len(years) >= 2:
                first_year = str(years[0])
                last_year = str(years[-1])
                
                first_col = next((col for col in sales_columns if first_year in col), None)
                last_col = next((col for col in sales_columns if last_year in col), None)
                
                if first_col and last_col and first_col in df.columns and last_col in df.columns:
                    n_years = len(years)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cagr = np.where(
                            df[first_col] > 0,
                            ((df[last_col] / df[first_col]) ** (1/n_years) - 1) * 100,
                            np.nan
                        )
                    
                    df['CAGR'] = pd.Series(cagr, dtype=np.float32)
            
            # Calculate market share
            if years:
                last_year = str(years[-1])
                last_sales_col = next((col for col in sales_columns if last_year in col), None)
                
                if last_sales_col and last_sales_col in df.columns:
                    total_sales = df[last_sales_col].sum()
                    if total_sales > 0:
                        df['Pazar_Payı'] = (df[last_sales_col] / total_sales) * 100
            
            # Calculate average price if not exists
            price_columns = [col for col in df.columns if 'fiyat' in col.lower() or 'price' in col.lower()]
            
            if not price_columns:
                for year in years:
                    sales_col = next((col for col in sales_columns if str(year) in col), None)
                    unit_col = next((col for col in df.columns if str(year) in col and 
                                   ('birim' in col.lower() or 'unit' in col.lower())), None)
                    
                    if sales_col and unit_col and sales_col in df.columns and unit_col in df.columns:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            avg_price = np.where(
                                df[unit_col] > 0,
                                df[sales_col] / df[unit_col],
                                np.nan
                            )
                        
                        df[f'Ort_Fiyat_{year}'] = pd.Series(avg_price, dtype=np.float32)
            
            # Calculate price-volume score
            if years:
                last_year = str(years[-1])
                price_col = next((col for col in df.columns if f'Ort_Fiyat_{last_year}' in col), None)
                unit_col = next((col for col in df.columns if str(last_year) in col and 
                               ('birim' in col.lower() or 'unit' in col.lower())), None)
                
                if price_col and unit_col and price_col in df.columns and unit_col in df.columns:
                    df['Fiyat_Hacim_Skoru'] = df[price_col] * df[unit_col]
            
            # Calculate performance score
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                try:
                    scaler = StandardScaler()
                    numeric_data = df[numeric_cols].fillna(0)
                    scaled_data = scaler.fit_transform(numeric_data)
                    df['Performans_Skoru'] = scaled_data.mean(axis=1)
                except Exception as e:
                    st.warning(f"Performans skoru hesaplanamadı: {str(e)}")
            
            return df
            
        except Exception as e:
            st.error(f"Analiz verisi hazırlama hatası: {str(e)}")
            return df

class AnalyticsEngine:
    """Gelişmiş analitik motoru sınıfı"""
    
    def __init__(self):
        self.forecast_models = {}
        self.anomaly_detectors = {}
    
    def calculate_comprehensive_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Kapsamlı pazar metriklerini hesapla
        
        Parameters:
        -----------
        df : pd.DataFrame
            Analiz edilecek veri
            
        Returns:
        --------
        Dict[str, Any]
            Hesaplanan metrikler
        """
        metrics = {}
        
        try:
            metrics['total_rows'] = len(df)
            metrics['total_columns'] = len(df.columns)
            
            # Sales metrics
            sales_columns = [col for col in df.columns 
                           if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
            
            if sales_columns:
                last_sales_col = sales_columns[-1]
                year_match = re.search(r'\b(20\d{2})\b', last_sales_col)
                metrics['last_sales_year'] = year_match.group(1) if year_match else 'N/A'
                
                metrics['total_market_value'] = float(df[last_sales_col].sum())
                metrics['avg_sales_per_product'] = float(df[last_sales_col].mean())
                metrics['median_sales'] = float(df[last_sales_col].median())
                metrics['sales_std'] = float(df[last_sales_col].std())
                
                # Quartiles
                q1, q3 = df[last_sales_col].quantile([0.25, 0.75])
                metrics['sales_q1'] = float(q1)
                metrics['sales_q3'] = float(q3)
                metrics['sales_iqr'] = float(q3 - q1)
                
                # Top performers
                top_10_sales = df[last_sales_col].nlargest(10).sum()
                metrics['top_10_share'] = (top_10_sales / metrics['total_market_value'] * 100) \
                                         if metrics['total_market_value'] > 0 else 0
            
            # Growth metrics
            growth_columns = [col for col in df.columns if 'Büyüme_' in col]
            if growth_columns:
                last_growth_col = growth_columns[-1]
                metrics['avg_growth_rate'] = float(df[last_growth_col].mean())
                metrics['median_growth'] = float(df[last_growth_col].median())
                metrics['positive_growth_products'] = int((df[last_growth_col] > 0).sum())
                metrics['high_growth_products'] = int((df[last_growth_col] > 20).sum())
            
            # Company metrics
            if 'Şirket' in df.columns and sales_columns:
                last_sales_col = sales_columns[-1]
                company_sales = df.groupby('Şirket')[last_sales_col].sum()
                total_sales = company_sales.sum()
                
                if total_sales > 0:
                    market_shares = (company_sales / total_sales * 100)
                    # Herfindahl-Hirschman Index
                    metrics['hhi_index'] = float((market_shares ** 2).sum())
                    
                    # Concentration ratios
                    for n in [1, 3, 5, 10]:
                        top_n_share = company_sales.nlargest(n).sum() / total_sales * 100
                        metrics[f'top_{n}_share'] = float(top_n_share)
            
            # Molecule metrics
            if 'Molekül' in df.columns:
                metrics['unique_molecules'] = int(df['Molekül'].nunique())
                if sales_columns:
                    last_sales_col = sales_columns[-1]
                    molecule_sales = df.groupby('Molekül')[last_sales_col].sum()
                    total_molecule_sales = molecule_sales.sum()
                    if total_molecule_sales > 0:
                        top_10_molecules = molecule_sales.nlargest(10).sum()
                        metrics['top_10_molecule_share'] = float(top_10_molecules / total_molecule_sales * 100)
            
            # Country metrics
            if 'Ülke' in df.columns:
                metrics['country_coverage'] = int(df['Ülke'].nunique())
                if sales_columns:
                    last_sales_col = sales_columns[-1]
                    country_sales = df.groupby('Ülke')[last_sales_col].sum()
                    total_country_sales = country_sales.sum()
                    if total_country_sales > 0:
                        top_5_countries = country_sales.nlargest(5).sum()
                        metrics['top_5_country_share'] = float(top_5_countries / total_country_sales * 100)
            
            # International Product metrics
            intl_columns = [col for col in df.columns if 'international' in col.lower()]
            if intl_columns:
                intl_col = intl_columns[0]
                intl_df = df[df[intl_col] == 1]
                local_df = df[df[intl_col] == 0]
                
                metrics['international_product_count'] = len(intl_df)
                metrics['local_product_count'] = len(local_df)
                
                if sales_columns:
                    last_sales_col = sales_columns[-1]
                    metrics['international_sales'] = float(intl_df[last_sales_col].sum())
                    metrics['local_sales'] = float(local_df[last_sales_col].sum())
                    
                    if metrics['total_market_value'] > 0:
                        metrics['international_share'] = float(
                            metrics['international_sales'] / metrics['total_market_value'] * 100
                        )
                        metrics['local_share'] = float(
                            metrics['local_sales'] / metrics['total_market_value'] * 100
                        )
            
            return metrics
            
        except Exception as e:
            st.error(f"Metrik hesaplama hatası: {str(e)}")
            return metrics
    
    def detect_anomalies(self, df: pd.DataFrame, features: List[str] = None, 
                        contamination: float = 0.1) -> pd.DataFrame:
        """
        Anomali tespiti yap
        
        Parameters:
        -----------
        df : pd.DataFrame
            Analiz edilecek veri
        features : List[str], optional
            Kullanılacak özellikler
        contamination : float
            Anomali oranı tahmini
            
        Returns:
        --------
        pd.DataFrame
            Anomali işaretleri eklenmiş veri
        """
        try:
            if features is None:
                # Auto-select numeric features
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                features = numeric_cols[:10]  # Limit to first 10 features
            
            if len(features) < 2:
                st.warning("Anomali tespiti için en az 2 özellik gerekli")
                return df
            
            # Prepare data
            X = df[features].fillna(0)
            
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            anomalies = iso_forest.fit_predict(X)
            
            # Add anomaly flags to dataframe
            result_df = df.copy()
            result_df['Anomali'] = anomalies
            result_df['Anomali_Skoru'] = iso_forest.decision_function(X)
            result_df['Anomali_Seviyesi'] = pd.cut(
                result_df['Anomali_Skoru'],
                bins=[-float('inf'), -0.5, 0, 0.5, float('inf')],
                labels=['Yüksek Risk', 'Orta Risk', 'Düşük Risk', 'Normal']
            )
            
            # Calculate anomaly statistics
            anomaly_count = (result_df['Anomali'] == -1).sum()
            total_count = len(result_df)
            
            st.info(f"🔍 Anomali tespiti: {anomaly_count:,} anomali tespit edildi "
                   f"(%{(anomaly_count/total_count*100):.1f})")
            
            return result_df
            
        except Exception as e:
            st.error(f"Anomali tespiti hatası: {str(e)}")
            return df
    
    def forecast_market(self, df: pd.DataFrame, target_column: str, 
                       periods: int = Config.FORECAST_YEARS) -> Dict[str, Any]:
        """
        Pazar tahminlemesi yap
        
        Parameters:
        -----------
        df : pd.DataFrame
            Tarihsel veri
        target_column : str
            Tahmin edilecek sütun
        periods : int
            Tahmin periyodu sayısı
            
        Returns:
        --------
        Dict[str, Any]
            Tahmin sonuçları
        """
        try:
            if target_column not in df.columns:
                st.error(f"Hedef sütun bulunamadı: {target_column}")
                return {}
            
            # Prepare time series data
            ts_data = df[target_column].dropna()
            
            if len(ts_data) < 10:
                st.warning("Tahmin için en az 10 gözlem gerekli")
                return {}
            
            # Try multiple forecasting methods
            forecasts = {}
            
            # Method 1: ARIMA
            try:
                model_arima = sm.tsa.ARIMA(ts_data, order=(1,1,1))
                result_arima = model_arima.fit()
                forecast_arima = result_arima.forecast(steps=periods)
                forecasts['ARIMA'] = forecast_arima
            except:
                pass
            
            # Method 2: Exponential Smoothing
            try:
                model_ets = ExponentialSmoothing(ts_data, seasonal='add', seasonal_periods=12)
                result_ets = model_ets.fit()
                forecast_ets = result_ets.forecast(periods)
                forecasts['ETS'] = forecast_ets
            except:
                pass
            
            # Method 3: Prophet
            try:
                prophet_df = pd.DataFrame({
                    'ds': pd.date_range(start='2020-01-01', periods=len(ts_data), freq='M'),
                    'y': ts_data.values
                })
                
                model_prophet = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive'
                )
                model_prophet.fit(prophet_df)
                
                future = model_prophet.make_future_dataframe(periods=periods, freq='M')
                forecast_prophet = model_prophet.predict(future)
                forecasts['Prophet'] = forecast_prophet['yhat'].iloc[-periods:].values
            except:
                pass
            
            if not forecasts:
                st.warning("Hiçbir tahmin yöntemi başarılı olamadı")
                return {}
            
            # Combine forecasts (simple average)
            combined_forecast = pd.DataFrame(forecasts).mean(axis=1)
            
            # Calculate confidence intervals
            forecast_mean = combined_forecast.mean()
            forecast_std = combined_forecast.std()
            
            result = {
                'forecast_values': combined_forecast.tolist(),
                'forecast_mean': float(forecast_mean),
                'forecast_std': float(forecast_std),
                'methods_used': list(forecasts.keys()),
                'confidence_interval': {
                    'lower': float(forecast_mean - 1.96 * forecast_std),
                    'upper': float(forecast_mean + 1.96 * forecast_std)
                }
            }
            
            return result
            
        except Exception as e:
            st.error(f"Tahminleme hatası: {str(e)}")
            return {}
    
    def perform_clustering(self, df: pd.DataFrame, n_clusters: int = 4,
                         method: str = 'kmeans') -> pd.DataFrame:
        """
        Kümeleme analizi yap
        
        Parameters:
        -----------
        df : pd.DataFrame
            Analiz edilecek veri
        n_clusters : int
            Küme sayısı
        method : str
            Kümeleme yöntemi
            
        Returns:
        --------
        pd.DataFrame
            Küme etiketleri eklenmiş veri
        """
        try:
            # Select features for clustering
            features = []
            
            sales_cols = [col for col in df.columns 
                         if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
            if sales_cols:
                features.extend(sales_cols[-2:])
            
            growth_cols = [col for col in df.columns if 'Büyüme_' in col]
            if growth_cols:
                features.append(growth_cols[-1])
            
            price_cols = [col for col in df.columns if 'Fiyat' in col]
            if price_cols:
                features.append(price_cols[-1])
            
            if len(features) < 2:
                st.warning("Kümeleme için en az 2 özellik gerekli")
                return df
            
            # Prepare data
            X = df[features].fillna(0)
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            if method == 'kmeans':
                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                model = KMeans(n_clusters=n_clusters, random_state=42)
            
            clusters = model.fit_predict(X_scaled)
            
            # Add clusters to dataframe
            result_df = df.copy()
            result_df['Küme'] = clusters
            
            # Calculate cluster statistics
            cluster_stats = {}
            for cluster_id in np.unique(clusters):
                if cluster_id != -1:  # Skip noise points in DBSCAN
                    cluster_data = X[clusters == cluster_id]
                    cluster_stats[cluster_id] = {
                        'size': len(cluster_data),
                        'avg_sales': float(cluster_data.iloc[:, 0].mean()),
                        'avg_growth': float(cluster_data.iloc[:, 1].mean()) if len(cluster_data.columns) > 1 else 0
                    }
            
            # Name clusters based on characteristics
            cluster_names = {}
            for cluster_id, stats in cluster_stats.items():
                if stats['avg_growth'] > 10:
                    cluster_names[cluster_id] = 'Yüksek Büyüme'
                elif stats['avg_growth'] > 0:
                    cluster_names[cluster_id] = 'Orta Büyüme'
                elif stats['avg_sales'] > X.iloc[:, 0].mean():
                    cluster_names[cluster_id] = 'Büyük Hacim'
                else:
                    cluster_names[cluster_id] = f'Küme {cluster_id}'
            
            result_df['Küme_İsmi'] = result_df['Küme'].map(cluster_names)
            
            # Calculate silhouette score
            if len(np.unique(clusters)) > 1 and method != 'dbscan':
                try:
                    silhouette_avg = silhouette_score(X_scaled, clusters)
                    st.info(f"📊 Kümeleme kalitesi (Silhouette Score): {silhouette_avg:.3f}")
                except:
                    pass
            
            return result_df
            
        except Exception as e:
            st.error(f"Kümeleme hatası: {str(e)}")
            return df

class Visualizer:
    """Gelişmiş görselleştirme sınıfı"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Bold
        self.template = Config.CHART_TEMPLATE
    
    def create_dashboard_metrics(self, df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        """
        Dashboard metrik kartlarını oluştur
        
        Parameters:
        -----------
        df : pd.DataFrame
            Veri seti
        metrics : Dict[str, Any]
            Hesaplanan metrikler
        """
        try:
            cols = st.columns(4)
            
            # Metric 1: Total Market Value
            with cols[0]:
                total_value = metrics.get('total_market_value', 0)
                sales_year = metrics.get('last_sales_year', '')
                st.markdown(f"""
                <div class="glass-card" style="padding: 1.5rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                        TOPLAM PAZAR DEĞERİ
                    </div>
                    <div style="font-size: 2rem; font-weight: 800; color: var(--accent-blue); margin-bottom: 0.5rem;">
                        ${total_value/1e6:.1f}M
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">
                        {sales_year} Yılı
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Metric 2: Average Growth
            with cols[1]:
                avg_growth = metrics.get('avg_growth_rate', 0)
                growth_color = "var(--success)" if avg_growth > 0 else "var(--danger)"
                st.markdown(f"""
                <div class="glass-card" style="padding: 1.5rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                        ORTALAMA BÜYÜME
                    </div>
                    <div style="font-size: 2rem; font-weight: 800; color: {growth_color}; margin-bottom: 0.5rem;">
                        {avg_growth:.1f}%
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">
                        Yıllık Değişim
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Metric 3: Market Concentration
            with cols[2]:
                hhi = metrics.get('hhi_index', 0)
                if hhi > 2500:
                    hhi_status = "Monopolistik"
                    hhi_color = "var(--danger)"
                elif hhi > 1800:
                    hhi_status = "Oligopol"
                    hhi_color = "var(--warning)"
                else:
                    hhi_status = "Rekabetçi"
                    hhi_color = "var(--success)"
                st.markdown(f"""
                <div class="glass-card" style="padding: 1.5rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                        REKABET YOĞUNLUĞU
                    </div>
                    <div style="font-size: 2rem; font-weight: 800; color: {hhi_color}; margin-bottom: 0.5rem;">
                        {hhi:.0f}
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">
                        {hhi_status}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Metric 4: International Share
            with cols[3]:
                intl_share = metrics.get('international_share', 0)
                intl_color = "var(--success)" if intl_share > 20 else "var(--warning)" if intl_share > 10 else "var(--info)"
                st.markdown(f"""
                <div class="glass-card" style="padding: 1.5rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
                        INTERNATIONAL PAY
                    </div>
                    <div style="font-size: 2rem; font-weight: 800; color: {intl_color}; margin-bottom: 0.5rem;">
                        {intl_share:.1f}%
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">
                        Global Ürünler
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional metrics row
            cols2 = st.columns(4)
            
            with cols2[0]:
                unique_molecules = metrics.get('unique_molecules', 0)
                st.metric("Benzersiz Molekül", f"{unique_molecules:,}")
            
            with cols2[1]:
                country_coverage = metrics.get('country_coverage', 0)
                st.metric("Ülke Kapsamı", country_coverage)
            
            with cols2[2]:
                high_growth = metrics.get('high_growth_products', 0)
                total_products = metrics.get('total_rows', 1)
                st.metric("Yüksek Büyüme", f"{high_growth}", 
                         f"%{(high_growth/total_products*100):.1f}")
            
            with cols2[3]:
                anomaly_count = len(df[df.get('Anomali', 0) == -1]) if 'Anomali' in df.columns else 0
                st.metric("Anomaliler", f"{anomaly_count}")
            
        except Exception as e:
            st.error(f"Metrik görselleştirme hatası: {str(e)}")
    
    def create_sales_trend_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Satış trend grafiği oluştur
        
        Parameters:
        -----------
        df : pd.DataFrame
            Veri seti
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        try:
            sales_cols = [col for col in df.columns 
                         if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
            
            if len(sales_cols) >= 2:
                yearly_data = []
                for col in sorted(sales_cols):
                    year_match = re.search(r'\b(20\d{2})\b', col)
                    year = year_match.group(1) if year_match else col
                    
                    yearly_data.append({
                        'Yıl': year,
                        'Toplam_Satış': df[col].sum(),
                        'Ortalama_Satış': df[col].mean(),
                        'Ürün_Sayısı': (df[col] > 0).sum()
                    })
                
                yearly_df = pd.DataFrame(yearly_data)
                
                fig = go.Figure()
                
                # Bar chart for total sales
                fig.add_trace(go.Bar(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Toplam_Satış'],
                    name='Toplam Satış',
                    marker_color=self.color_palette[0],
                    text=[f'${x/1e6:.0f}M' for x in yearly_df['Toplam_Satış']],
                    textposition='auto'
                ))
                
                # Line chart for average sales
                fig.add_trace(go.Scatter(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Ortalama_Satış'],
                    name='Ortalama Satış',
                    mode='lines+markers',
                    line=dict(color=self.color_palette[1], width=3),
                    marker=dict(size=10),
                    yaxis='y2'
                ))
                
                # Calculate growth rates
                if len(yearly_df) > 1:
                    growth_rates = []
                    for i in range(1, len(yearly_df)):
                        growth = ((yearly_df.iloc[i]['Toplam_Satış'] - yearly_df.iloc[i-1]['Toplam_Satış']) / 
                                 yearly_df.iloc[i-1]['Toplam_Satış'] * 100)
                        growth_rates.append(growth)
                    
                    # Add growth rate annotations
                    for i, growth in enumerate(growth_rates):
                        fig.add_annotation(
                            x=yearly_df.iloc[i+1]['Yıl'],
                            y=yearly_df.iloc[i+1]['Toplam_Satış'],
                            text=f'{growth:.1f}%',
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40
                        )
                
                fig.update_layout(
                    title='Satış Trendleri Analizi',
                    xaxis_title='Yıl',
                    yaxis_title='Toplam Satış (USD)',
                    yaxis2=dict(
                        title='Ortalama Satış (USD)',
                        overlaying='y',
                        side='right'
                    ),
                    height=Config.CHART_HEIGHT,
                    template=self.template,
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
            st.error(f"Trend grafiği oluşturma hatası: {str(e)}")
            return None
    
    def create_forecast_chart(self, historical_data: pd.Series, 
                            forecast_results: Dict[str, Any]) -> go.Figure:
        """
        Tahmin grafiği oluştur
        
        Parameters:
        -----------
        historical_data : pd.Series
            Tarihsel veri
        forecast_results : Dict[str, Any]
            Tahmin sonuçları
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        try:
            if not forecast_results:
                return None
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=list(range(len(historical_data))),
                y=historical_data.values,
                mode='lines+markers',
                name='Tarihsel Veri',
                line=dict(color=self.color_palette[0], width=2)
            ))
            
            # Forecast
            forecast_values = forecast_results.get('forecast_values', [])
            forecast_periods = len(forecast_values)
            
            if forecast_periods > 0:
                forecast_x = list(range(len(historical_data), 
                                      len(historical_data) + forecast_periods))
                
                fig.add_trace(go.Scatter(
                    x=forecast_x,
                    y=forecast_values,
                    mode='lines+markers',
                    name='Tahmin',
                    line=dict(color=self.color_palette[1], width=2, dash='dash')
                ))
                
                # Confidence interval
                if 'confidence_interval' in forecast_results:
                    ci = forecast_results['confidence_interval']
                    fig.add_trace(go.Scatter(
                        x=forecast_x + forecast_x[::-1],
                        y=[ci['upper']] * forecast_periods + [ci['lower']] * forecast_periods[::-1],
                        fill='toself',
                        fillcolor='rgba(42, 202, 234, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Güven Aralığı'
                    ))
            
            fig.update_layout(
                title='Pazar Tahminlemesi',
                xaxis_title='Periyot',
                yaxis_title='Değer',
                height=Config.CHART_HEIGHT,
                template=self.template,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Tahmin grafiği oluşturma hatası: {str(e)}")
            return None
    
    def create_anomaly_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Anomali tespiti grafiği oluştur
        
        Parameters:
        -----------
        df : pd.DataFrame
            Anomali işaretli veri
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        try:
            if 'Anomali' not in df.columns:
                return None
            
            # Select numeric column for visualization
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return None
            
            viz_col = numeric_cols[0]  # Use first numeric column
            
            fig = go.Figure()
            
            # Normal points
            normal_df = df[df['Anomali'] == 1]
            if len(normal_df) > 0:
                fig.add_trace(go.Scatter(
                    x=normal_df.index,
                    y=normal_df[viz_col],
                    mode='markers',
                    name='Normal',
                    marker=dict(
                        color=self.color_palette[0],
                        size=8,
                        opacity=0.7
                    )
                ))
            
            # Anomaly points
            anomaly_df = df[df['Anomali'] == -1]
            if len(anomaly_df) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly_df.index,
                    y=anomaly_df[viz_col],
                    mode='markers',
                    name='Anomali',
                    marker=dict(
                        color=self.color_palette[3],
                        size=12,
                        symbol='x'
                    )
                ))
            
            fig.update_layout(
                title='Anomali Tespiti',
                xaxis_title='Gözlem',
                yaxis_title=viz_col,
                height=Config.CHART_HEIGHT,
                template=self.template,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Anomali grafiği oluşturma hatası: {str(e)}")
            return None
    
    def create_sunburst_chart(self, df: pd.DataFrame, 
                            path_columns: List[str] = None) -> go.Figure:
        """
        Sunburst (hierarchical) grafiği oluştur
        
        Parameters:
        -----------
        df : pd.DataFrame
            Veri seti
        path_columns : List[str], optional
            Hiyerarşi sütunları
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        try:
            if path_columns is None:
                # Auto-detect hierarchy
                if 'Ülke' in df.columns and 'Şirket' in df.columns and 'Molekül' in df.columns:
                    path_columns = ['Ülke', 'Şirket', 'Molekül']
                elif 'Bölge' in df.columns and 'Ülke' in df.columns:
                    path_columns = ['Bölge', 'Ülke', 'Şirket']
                else:
                    return None
            
            # Get sales column
            sales_cols = [col for col in df.columns 
                         if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
            if not sales_cols:
                return None
            
            value_column = sales_cols[-1]
            
            # Aggregate data
            agg_df = df.groupby(path_columns)[value_column].sum().reset_index()
            
            fig = px.sunburst(
                agg_df,
                path=path_columns,
                values=value_column,
                title='Pazar Hiyerarşisi - Sunburst Grafiği',
                color=value_column,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=600,
                template=self.template
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Sunburst grafiği oluşturma hatası: {str(e)}")
            return None
    
    def create_radar_chart(self, df: pd.DataFrame, 
                          company_column: str = 'Şirket') -> go.Figure:
        """
        Radar chart (şirket karşılaştırma) oluştur
        
        Parameters:
        -----------
        df : pd.DataFrame
            Veri seti
        company_column : str
            Şirket sütunu adı
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        try:
            if company_column not in df.columns:
                return None
            
            # Select top companies
            sales_cols = [col for col in df.columns 
                         if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
            if not sales_cols:
                return None
            
            value_column = sales_cols[-1]
            
            top_companies = df.groupby(company_column)[value_column].sum().nlargest(5).index.tolist()
            
            if len(top_companies) < 2:
                return None
            
            # Prepare radar chart data
            metrics = ['Satış', 'Büyüme', 'Fiyat', 'Pazar Payı', 'Stabilite']
            companies_data = {}
            
            for company in top_companies:
                company_df = df[df[company_column] == company]
                
                if len(company_df) > 0:
                    # Normalize metrics to 0-1 scale
                    sales_norm = company_df[value_column].mean() / df[value_column].max()
                    
                    growth_cols = [col for col in df.columns if 'Büyüme_' in col]
                    growth_norm = company_df[growth_cols[-1]].mean() / 100 if growth_cols else 0.5
                    
                    price_cols = [col for col in df.columns if 'Fiyat' in col]
                    price_norm = company_df[price_cols[-1]].mean() / df[price_cols[-1]].max() if price_cols else 0.5
                    
                    market_share = (company_df[value_column].sum() / df[value_column].sum()) * 5
                    
                    stability = 1 - (company_df[value_column].std() / company_df[value_column].mean()) \
                               if company_df[value_column].mean() > 0 else 0.5
                    
                    companies_data[company] = [
                        max(0, min(1, sales_norm)),
                        max(0, min(1, growth_norm)),
                        max(0, min(1, price_norm)),
                        max(0, min(1, market_share)),
                        max(0, min(1, stability))
                    ]
            
            # Create radar chart
            fig = go.Figure()
            
            for company, values in companies_data.items():
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=company
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title='Şirket Performans Karşılaştırması - Radar Chart',
                height=600,
                template=self.template,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Radar chart oluşturma hatası: {str(e)}")
            return None
    
    def create_sankey_diagram(self, df: pd.DataFrame) -> go.Figure:
        """
        Sankey diagram (pazar akışı) oluştur
        
        Parameters:
        -----------
        df : pd.DataFrame
            Veri seti
            
        Returns:
        --------
        go.Figure
            Plotly figure
        """
        try:
            # Check required columns
            required_cols = ['Ülke', 'Şirket', 'Molekül']
            if not all(col in df.columns for col in required_cols):
                return None
            
            sales_cols = [col for col in df.columns 
                         if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
            if not sales_cols:
                return None
            
            value_column = sales_cols[-1]
            
            # Aggregate data for Sankey
            country_company = df.groupby(['Ülke', 'Şirket'])[value_column].sum().reset_index()
            company_molecule = df.groupby(['Şirket', 'Molekül'])[value_column].sum().reset_index()
            
            # Create node labels
            countries = df['Ülke'].unique().tolist()
            companies = df['Şirket'].unique().tolist()
            molecules = df['Molekül'].unique().tolist()[:20]  # Limit molecules
            
            node_labels = countries + companies + molecules
            node_indices = {label: idx for idx, label in enumerate(node_labels)}
            
            # Create links
            links = []
            
            # Country -> Company links
            for _, row in country_company.iterrows():
                if row['Şirket'] in node_indices and row['Ülke'] in node_indices:
                    links.append({
                        'source': node_indices[row['Ülke']],
                        'target': node_indices[row['Şirket']],
                        'value': row[value_column]
                    })
            
            # Company -> Molecule links
            for _, row in company_molecule.iterrows():
                if row['Şirket'] in node_indices and row['Molekül'] in node_indices:
                    links.append({
                        'source': node_indices[row['Şirket']],
                        'target': node_indices[row['Molekül']],
                        'value': row[value_column]
                    })
            
            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=self.color_palette
                ),
                link=dict(
                    source=[link['source'] for link in links],
                    target=[link['target'] for link in links],
                    value=[link['value'] for link in links]
                )
            )])
            
            fig.update_layout(
                title='Pazar Akış Diyagramı - Sankey',
                height=800,
                template=self.template
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Sankey diagram oluşturma hatası: {str(e)}")
            return None

class ReportingEngine:
    """Raporlama motoru sınıfı"""
    
    @staticmethod
    def generate_excel_report(df: pd.DataFrame, metrics: Dict[str, Any], 
                            insights: List[Dict], filename: str = "pharma_report.xlsx") -> BytesIO:
        """
        Gelişmiş Excel raporu oluştur
        
        Parameters:
        -----------
        df : pd.DataFrame
            Veri seti
        metrics : Dict[str, Any]
            Metrikler
        insights : List[Dict]
            İçgörüler
        filename : str
            Dosya adı
            
        Returns:
        --------
        BytesIO
            Excel dosyası
        """
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Sheet 1: Raw Data
                df.to_excel(writer, sheet_name='Ham Veri', index=False)
                
                # Sheet 2: Summary Metrics
                summary_df = pd.DataFrame([
                    ['Toplam Satır', metrics.get('total_rows', 0)],
                    ['Toplam Sütun', metrics.get('total_columns', 0)],
                    ['Toplam Pazar Değeri', f"${metrics.get('total_market_value', 0):,.2f}"],
                    ['Ortalama Büyüme', f"{metrics.get('avg_growth_rate', 0):.2f}%"],
                    ['HHI İndeksi', metrics.get('hhi_index', 0)],
                    ['International Pay', f"{metrics.get('international_share', 0):.2f}%"],
                    ['Benzersiz Molekül', metrics.get('unique_molecules', 0)],
                    ['Ülke Kapsamı', metrics.get('country_coverage', 0)]
                ], columns=['Metrik', 'Değer'])
                
                summary_df.to_excel(writer, sheet_name='Özet Metrikler', index=False)
                
                # Sheet 3: Insights
                if insights:
                    insights_df = pd.DataFrame(insights)
                    insights_df.to_excel(writer, sheet_name='Stratejik İçgörüler', index=False)
                
                # Sheet 4: Top Performers
                sales_cols = [col for col in df.columns 
                            if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
                if sales_cols:
                    last_sales_col = sales_cols[-1]
                    top_products = df.nlargest(20, last_sales_col)[
                        ['Molekül', 'Şirket', 'Ülke', last_sales_col] 
                        if all(col in df.columns for col in ['Molekül', 'Şirket', 'Ülke']) 
                        else [last_sales_col]
                    ]
                    top_products.to_excel(writer, sheet_name='Top 20 Ürün', index=False)
                
                # Get workbook and worksheet objects
                workbook = writer.book
                
                # Format sheets
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#2d7dd2',
                    'font_color': 'white',
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                
                # Apply formatting to all sheets
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    
                    # Add header format
                    for col_num, value in enumerate(df.columns.values if sheet_name == 'Ham Veri' 
                                                  else summary_df.columns.values if sheet_name == 'Özet Metrikler'
                                                  else []):
                        worksheet.write(0, col_num, value, header_format)
                    
                    # Auto-adjust column widths
                    worksheet.autofit()
            
            output.seek(0)
            return output
            
        except Exception as e:
            st.error(f"Excel rapor oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def generate_html_report(df: pd.DataFrame, metrics: Dict[str, Any], 
                           visualizations: List[go.Figure]) -> str:
        """
        HTML raporu oluştur
        
        Parameters:
        -----------
        df : pd.DataFrame
            Veri seti
        metrics : Dict[str, Any]
            Metrikler
        visualizations : List[go.Figure]
            Görselleştirmeler
            
        Returns:
        --------
        str
            HTML rapor
        """
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html lang="tr">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>PharmaIntelligence Raporu</title>
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        margin: 0;
                        padding: 20px;
                        background: linear-gradient(135deg, {Config.COLORS['primary_dark']}, {Config.COLORS['secondary_dark']});
                        color: white;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: rgba(30, 58, 95, 0.9);
                        padding: 30px;
                        border-radius: 15px;
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    }}
                    .header {{
                        text-align: center;
                        margin-bottom: 40px;
                        padding-bottom: 20px;
                        border-bottom: 2px solid {Config.COLORS['accent_blue']};
                    }}
                    .metrics-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin: 30px 0;
                    }}
                    .metric-card {{
                        background: rgba(45, 125, 210, 0.2);
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 4px solid {Config.COLORS['accent_blue']};
                    }}
                    .visualization {{
                        margin: 30px 0;
                        background: rgba(255, 255, 255, 0.05);
                        padding: 20px;
                        border-radius: 10px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🏥 PharmaIntelligence Analiz Raporu</h1>
                        <p>Oluşturma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="metrics-grid">
            """
            
            # Add metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    html_content += f"""
                        <div class="metric-card">
                            <h3>{key.replace('_', ' ').title()}</h3>
                            <p>{value:,.2f}</p>
                        </div>
                    """
            
            html_content += """
                    </div>
                    
                    <div class="data-summary">
                        <h2>📊 Veri Özeti</h2>
                        <p>Toplam Satır: {:,}</p>
                        <p>Toplam Sütun: {}</p>
                        <p>Toplam Pazar Değeri: ${:,.2f}</p>
                    </div>
                </div>
            </body>
            </html>
            """.format(
                len(df),
                len(df.columns),
                metrics.get('total_market_value', 0)
            )
            
            return html_content
            
        except Exception as e:
            st.error(f"HTML rapor oluşturma hatası: {str(e)}")
            return ""

# ================================================
# 3. ANA UYGULAMA
# ================================================

def main():
    """Ana uygulama fonksiyonu"""
    
    # Session state initialization
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'anomalies' not in st.session_state:
        st.session_state.anomalies = None
    if 'forecast' not in st.session_state:
        st.session_state.forecast = {}
    if 'clusters' not in st.session_state:
        st.session_state.clusters = None
    
    # Header
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="pharma-title">💊 PHARMAINTELLIGENCE PRO v6.0</h1>
        <p class="pharma-subtitle">
        Enterprise-level pharmaceutical market analytics platform with 
        International Product analysis, predictive modeling, anomaly detection,
        and strategic insights.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
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
                    load_sample = st.button("🔬 Örnek Yükle", use_container_width=True)
                
                with col2:
                    load_full = st.button("🚀 Tam Veri Yükle", type="primary", use_container_width=True)
                
                if load_sample:
                    with st.spinner("Örnek veri yükleniyor..."):
                        processor = DataProcessor()
                        data = processor.load_large_dataset(uploaded_file, sample_size=10000)
                        
                        if data is not None:
                            data = processor.prepare_analysis_data(data)
                            
                            st.session_state.data = data
                            st.session_state.filtered_data = data.copy()
                            
                            analytics = AnalyticsEngine()
                            st.session_state.metrics = analytics.calculate_comprehensive_metrics(data)
                            
                            st.success(f"✅ {len(data):,} satır örnek veri yüklendi!")
                            st.rerun()
                
                if load_full:
                    with st.spinner("Tüm veri seti yükleniyor..."):
                        processor = DataProcessor()
                        data = processor.load_large_dataset(uploaded_file, sample_size=None)
                        
                        if data is not None:
                            data = processor.prepare_analysis_data(data)
                            
                            st.session_state.data = data
                            st.session_state.filtered_data = data.copy()
                            
                            analytics = AnalyticsEngine()
                            st.session_state.metrics = analytics.calculate_comprehensive_metrics(data)
                            
                            st.success(f"✅ {len(data):,} satır tam veri yüklendi!")
                            st.rerun()
        
        # Analytics Options
        if st.session_state.data is not None:
            st.markdown("---")
            with st.expander("🔍 ANALİZ AYARLARI", expanded=True):
                # Anomaly Detection
                if st.button("🎯 Anomali Tespiti Yap", use_container_width=True):
                    with st.spinner("Anomali tespiti yapılıyor..."):
                        analytics = AnalyticsEngine()
                        anomalies = analytics.detect_anomalies(st.session_state.filtered_data)
                        st.session_state.anomalies = anomalies
                        st.success("✅ Anomali tespiti tamamlandı!")
                
                # Forecasting
                if st.button("📈 Tahminleme Yap", use_container_width=True):
                    with st.spinner("Pazar tahminlemesi yapılıyor..."):
                        analytics = AnalyticsEngine()
                        sales_cols = [col for col in st.session_state.filtered_data.columns 
                                    if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
                        if sales_cols:
                            forecast = analytics.forecast_market(
                                st.session_state.filtered_data, 
                                sales_cols[-1],
                                periods=Config.FORECAST_YEARS
                            )
                            st.session_state.forecast = forecast
                            st.success("✅ Tahminleme tamamlandı!")
                
                # Clustering
                if st.button("🏷️ Kümeleme Analizi Yap", use_container_width=True):
                    with st.spinner("Kümeleme analizi yapılıyor..."):
                        analytics = AnalyticsEngine()
                        clusters = analytics.perform_clustering(
                            st.session_state.filtered_data,
                            n_clusters=4
                        )
                        st.session_state.clusters = clusters
                        st.success("✅ Kümeleme analizi tamamlandı!")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
        <strong>PharmaIntelligence Enterprise</strong><br>
        v6.0 | International Product Analizi<br>
        © 2024 Tüm hakları saklıdır.
        </div>
        """, unsafe_allow_html=True)
    
    # Main Content
    if st.session_state.data is None:
        show_welcome_screen()
        return
    
    data = st.session_state.filtered_data
    metrics = st.session_state.metrics
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ",
        "🎯 ANOMALİ TESPİTİ",
        "🔮 TAHMİNLEME",
        "🏷️ KÜMELEME",
        "🌍 ADVANCED VIZ",
        "📑 RAPORLAMA",
        "⚙️ AYARLAR"
    ])
    
    with tab1:
        show_overview_tab(data, metrics)
    
    with tab2:
        show_market_analysis_tab(data)
    
    with tab3:
        show_anomaly_tab(data)
    
    with tab4:
        show_forecast_tab(data)
    
    with tab5:
        show_clustering_tab(data)
    
    with tab6:
        show_advanced_viz_tab(data)
    
    with tab7:
        show_reporting_tab(data, metrics)
    
    with tab8:
        show_settings_tab()

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
            Enterprise seviyesinde ilaç pazarı analizi platformu.<br>
            International Product analizi, tahminleme, anomali tespiti ve stratejik içgörüler.
            </p>
            
            <div class="feature-grid">
                <div class="feature-card feature-card-blue">
                    <div class="feature-icon">🌍</div>
                    <div class="feature-title">International Product</div>
                    <div class="feature-description">Çoklu pazar ürün analizi</div>
                </div>
                <div class="feature-card feature-card-cyan">
                    <div class="feature-icon">📈</div>
                    <div class="feature-title">Predictive Analytics</div>
                    <div class="feature-description">Pazar tahminleme ve trend analizi</div>
                </div>
                <div class="feature-card feature-card-teal">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-title">Anomaly Detection</div>
                    <div class="feature-description">Otomatik anomali ve outlier tespiti</div>
                </div>
                <div class="feature-card feature-card-warning">
                    <div class="feature-icon">🏷️</div>
                    <div class="feature-title">Advanced Clustering</div>
                    <div class="feature-description">Pazar segmentasyonu ve kümeleme</div>
                </div>
            </div>
            
            <div class="get-started-box">
                <div class="get-started-title">🎯 Başlamak İçin</div>
                <div class="get-started-steps">
                1. Sol taraftaki panelden veri dosyanızı yükleyin<br>
                2. "Örnek Yükle" veya "Tam Veri Yükle" butonuna tıklayın<br>
                3. Analiz sonuçlarını görmek için tabları kullanın
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_overview_tab(df: pd.DataFrame, metrics: Dict[str, Any]):
    """Genel Bakış tab'ını göster"""
    st.markdown('<h2 class="section-title">📊 Genel Bakış ve Performans Göstergeleri</h2>', unsafe_allow_html=True)
    
    # Show metrics
    visualizer = Visualizer()
    visualizer.create_dashboard_metrics(df, metrics)
    
    # Data preview
    st.markdown('<h3 class="subsection-title">📋 Veri Önizleme</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        row_count = st.slider(
            "Gösterilecek Satır Sayısı",
            10, Config.MAX_ROWS_PREVIEW, 100, 10,
            key="row_preview"
        )
        
        available_cols = df.columns.tolist()
        default_cols = []
        
        priority_cols = ['Molekül', 'Şirket', 'Ülke']
        for col in priority_cols:
            if col in available_cols:
                default_cols.append(col)
            if len(default_cols) >= 5:
                break
        
        if len(default_cols) < 5:
            default_cols.extend([col for col in available_cols[:5] if col not in default_cols])
        
        selected_cols = st.multiselect(
            "Gösterilecek Sütunlar",
            options=available_cols,
            default=default_cols[:min(5, len(default_cols))],
            key="col_preview"
        )
    
    with col2:
        if selected_cols:
            display_df = df[selected_cols].head(row_count)
        else:
            display_df = df.head(row_count)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Data summary
        st.info(f"""
        📊 **Veri Özeti:**
        - Toplam Satır: {len(df):,}
        - Toplam Sütun: {len(df.columns)}
        - Bellek Kullanımı: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB
        - Eksik Değerler: {df.isna().sum().sum():,}
        """)

def show_market_analysis_tab(df: pd.DataFrame):
    """Pazar Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">📈 Pazar Analizi ve Trendler</h2>', unsafe_allow_html=True)
    
    visualizer = Visualizer()
    
    # Sales trend chart
    st.markdown('<h3 class="subsection-title">💰 Satış Trendleri</h3>', unsafe_allow_html=True)
    trend_chart = visualizer.create_sales_trend_chart(df)
    if trend_chart:
        st.plotly_chart(trend_chart, use_container_width=True)
    else:
        st.info("Satış trend analizi için yeterli veri bulunamadı.")
    
    # Market share analysis
    st.markdown('<h3 class="subsection-title">🏆 Pazar Payı Analizi</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Şirket' in df.columns:
            sales_cols = [col for col in df.columns 
                         if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
            if sales_cols:
                last_sales_col = sales_cols[-1]
                company_sales = df.groupby('Şirket')[last_sales_col].sum().nlargest(10)
                
                fig = px.bar(
                    x=company_sales.values,
                    y=company_sales.index,
                    orientation='h',
                    title='Top 10 Şirket - Pazar Payı',
                    labels={'x': 'Satış (USD)', 'y': 'Şirket'},
                    color=company_sales.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Molekül' in df.columns and sales_cols:
            molecule_sales = df.groupby('Molekül')[last_sales_col].sum().nlargest(10)
            
            fig = px.pie(
                values=molecule_sales.values,
                names=molecule_sales.index,
                title='Top 10 Molekül - Dağılım',
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_anomaly_tab(df: pd.DataFrame):
    """Anomali Tespiti tab'ını göster"""
    st.markdown('<h2 class="section-title">🎯 Anomali Tespiti ve İzleme</h2>', unsafe_allow_html=True)
    
    if st.session_state.anomalies is not None:
        anomaly_df = st.session_state.anomalies
        
        visualizer = Visualizer()
        
        # Anomaly chart
        st.markdown('<h3 class="subsection-title">📊 Anomali Dağılımı</h3>', unsafe_allow_html=True)
        anomaly_chart = visualizer.create_anomaly_chart(anomaly_df)
        if anomaly_chart:
            st.plotly_chart(anomaly_chart, use_container_width=True)
        
        # Anomaly statistics
        st.markdown('<h3 class="subsection-title">📈 Anomali İstatistikleri</h3>', unsafe_allow_html=True)
        
        anomaly_count = (anomaly_df['Anomali'] == -1).sum()
        total_count = len(anomaly_df)
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Toplam Anomali", anomaly_count)
        with cols[1]:
            st.metric("Anomali Oranı", f"{(anomaly_count/total_count*100):.2f}%")
        with cols[2]:
            if 'Anomali_Skoru' in anomaly_df.columns:
                avg_score = anomaly_df['Anomali_Skoru'].mean()
                st.metric("Ortalama Skor", f"{avg_score:.3f}")
        with cols[3]:
            if 'Anomali_Seviyesi' in anomaly_df.columns:
                high_risk = (anomaly_df['Anomali_Seviyesi'] == 'Yüksek Risk').sum()
                st.metric("Yüksek Risk", high_risk)
        
        # Show anomaly details
        st.markdown('<h3 class="subsection-title">🔍 Anomali Detayları</h3>', unsafe_allow_html=True)
        
        if anomaly_count > 0:
            anomaly_details = anomaly_df[anomaly_df['Anomali'] == -1]
            
            # Select columns to show
            display_cols = ['Anomali_Skoru', 'Anomali_Seviyesi']
            for col in ['Molekül', 'Şirket', 'Ülke']:
                if col in anomaly_details.columns:
                    display_cols.append(col)
            
            # Add sales columns
            sales_cols = [col for col in anomaly_details.columns 
                         if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
            if sales_cols:
                display_cols.extend(sales_cols[:2])
            
            st.dataframe(
                anomaly_details[display_cols].sort_values('Anomali_Skoru'),
                use_container_width=True,
                height=400
            )
        else:
            st.success("✅ Hiç anomali tespit edilmedi!")
    
    else:
        st.info("ℹ️ Anomali tespiti yapmak için sol taraftaki 'Anomali Tespiti Yap' butonuna tıklayın.")
        
        # Quick anomaly detection
        if st.button("🚀 Hızlı Anomali Tespiti", type="primary"):
            with st.spinner("Anomali tespiti yapılıyor..."):
                analytics = AnalyticsEngine()
                anomalies = analytics.detect_anomalies(df)
                st.session_state.anomalies = anomalies
                st.rerun()

def show_forecast_tab(df: pd.DataFrame):
    """Tahminleme tab'ını göster"""
    st.markdown('<h2 class="section-title">🔮 Pazar Tahminlemesi</h2>', unsafe_allow_html=True)
    
    if st.session_state.forecast:
        forecast_results = st.session_state.forecast
        
        visualizer = Visualizer()
        
        # Get historical data
        sales_cols = [col for col in df.columns 
                     if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
        
        if sales_cols:
            historical_data = df[sales_cols[-1]].dropna()
            
            # Forecast chart
            st.markdown('<h3 class="subsection-title">📈 Tahmin Grafiği</h3>', unsafe_allow_html=True)
            forecast_chart = visualizer.create_forecast_chart(historical_data, forecast_results)
            if forecast_chart:
                st.plotly_chart(forecast_chart, use_container_width=True)
            
            # Forecast statistics
            st.markdown('<h3 class="subsection-title">📊 Tahmin İstatistikleri</h3>', unsafe_allow_html=True)
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Tahmin Periyodu", Config.FORECAST_YEARS)
            with cols[1]:
                forecast_mean = forecast_results.get('forecast_mean', 0)
                st.metric("Tahmin Ortalaması", f"${forecast_mean:,.0f}")
            with cols[2]:
                forecast_std = forecast_results.get('forecast_std', 0)
                st.metric("Standart Sapma", f"${forecast_std:,.0f}")
            with cols[3]:
                methods = forecast_results.get('methods_used', [])
                st.metric("Kullanılan Modeller", len(methods))
            
            # Forecast details
            st.markdown('<h3 class="subsection-title">🔍 Tahmin Detayları</h3>', unsafe_allow_html=True)
            
            if 'forecast_values' in forecast_results:
                forecast_df = pd.DataFrame({
                    'Periyot': range(1, len(forecast_results['forecast_values']) + 1),
                    'Tahmin Değeri': forecast_results['forecast_values'],
                    'Alt Sınır': forecast_results['confidence_interval']['lower'],
                    'Üst Sınır': forecast_results['confidence_interval']['upper']
                })
                
                st.dataframe(
                    forecast_df,
                    use_container_width=True,
                    height=300
                )
                
                # Download forecast
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Tahmin Verisini İndir",
                    data=csv,
                    file_name="pazar_tahminleri.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    else:
        st.info("ℹ️ Tahminleme yapmak için sol taraftaki 'Tahminleme Yap' butonuna tıklayın.")
        
        # Quick forecast
        if st.button("🚀 Hızlı Tahminleme", type="primary"):
            with st.spinner("Tahminleme yapılıyor..."):
                analytics = AnalyticsEngine()
                sales_cols = [col for col in df.columns 
                            if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
                if sales_cols:
                    forecast = analytics.forecast_market(df, sales_cols[-1])
                    st.session_state.forecast = forecast
                    st.rerun()

def show_clustering_tab(df: pd.DataFrame):
    """Kümeleme tab'ını göster"""
    st.markdown('<h2 class="section-title">🏷️ Pazar Segmentasyonu ve Kümeleme</h2>', unsafe_allow_html=True)
    
    if st.session_state.clusters is not None:
        cluster_df = st.session_state.clusters
        
        # Cluster distribution
        st.markdown('<h3 class="subsection-title">📊 Küme Dağılımı</h3>', unsafe_allow_html=True)
        
        if 'Küme' in cluster_df.columns:
            cluster_counts = cluster_df['Küme'].value_counts().sort_index()
            
            fig = px.bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                title='Küme Dağılımı',
                labels={'x': 'Küme', 'y': 'Ürün Sayısı'},
                color=cluster_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster characteristics
        st.markdown('<h3 class="subsection-title">🔍 Küme Özellikleri</h3>', unsafe_allow_html=True)
        
        if 'Küme_İsmi' in cluster_df.columns:
            # Calculate cluster statistics
            cluster_stats = []
            for cluster_id in cluster_df['Küme'].unique():
                if cluster_id != -1:  # Skip noise
                    cluster_data = cluster_df[cluster_df['Küme'] == cluster_id]
                    cluster_name = cluster_data['Küme_İsmi'].iloc[0]
                    
                    # Get sales column
                    sales_cols = [col for col in cluster_df.columns 
                                 if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
                    
                    stats = {
                        'Küme': cluster_id,
                        'İsim': cluster_name,
                        'Ürün Sayısı': len(cluster_data),
                        'Küme Payı': f"{(len(cluster_data)/len(cluster_df)*100):.1f}%"
                    }
                    
                    if sales_cols:
                        last_sales_col = sales_cols[-1]
                        stats['Ortalama Satış'] = f"${cluster_data[last_sales_col].mean():,.0f}"
                        stats['Toplam Satış'] = f"${cluster_data[last_sales_col].sum():,.0f}"
                    
                    cluster_stats.append(stats)
            
            if cluster_stats:
                stats_df = pd.DataFrame(cluster_stats)
                st.dataframe(
                    stats_df,
                    use_container_width=True,
                    height=300
                )
        
        # Show clustered data
        st.markdown('<h3 class="subsection-title">📋 Kümelere Ayrılmış Veri</h3>', unsafe_allow_html=True)
        
        display_cols = ['Küme', 'Küme_İsmi']
        for col in ['Molekül', 'Şirket', 'Ülke']:
            if col in cluster_df.columns:
                display_cols.append(col)
        
        sales_cols = [col for col in cluster_df.columns 
                     if re.search(r'(satış|sales).*\d{4}', col, re.IGNORECASE)]
        if sales_cols:
            display_cols.extend(sales_cols[:2])
        
        st.dataframe(
            cluster_df[display_cols].sort_values('Küme'),
            use_container_width=True,
            height=400
        )
    
    else:
        st.info("ℹ️ Kümeleme analizi yapmak için sol taraftaki 'Kümeleme Analizi Yap' butonuna tıklayın.")
        
        # Clustering options
        st.markdown('<h3 class="subsection-title">⚙️ Kümeleme Ayarları</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider(
                "Küme Sayısı",
                min_value=2,
                max_value=8,
                value=4,
                key="n_clusters"
            )
        
        with col2:
            method = st.selectbox(
                "Kümeleme Yöntemi",
                options=['kmeans', 'dbscan', 'hierarchical'],
                index=0,
                key="clustering_method"
            )
        
        if st.button("🔍 Kümeleme Yap", type="primary", use_container_width=True):
            with st.spinner("Kümeleme analizi yapılıyor..."):
                analytics = AnalyticsEngine()
                clusters = analytics.perform_clustering(
                    df,
                    n_clusters=n_clusters,
                    method=method
                )
                st.session_state.clusters = clusters
                st.rerun()

def show_advanced_viz_tab(df: pd.DataFrame):
    """Gelişmiş Görselleştirme tab'ını göster"""
    st.markdown('<h2 class="section-title">🌍 Gelişmiş Görselleştirme</h2>', unsafe_allow_html=True)
    
    visualizer = Visualizer()
    
    # Visualization selection
    viz_options = [
        "Sunburst (Pazar Hiyerarşisi)",
        "Radar Chart (Şirket Karşılaştırma)",
        "Sankey Diagram (Pazar Akışı)"
    ]
    
    selected_viz = st.selectbox(
        "Görselleştirme Türü Seçin",
        options=viz_options,
        key="viz_type"
    )
    
    if selected_viz == "Sunburst (Pazar Hiyerarşisi)":
        st.markdown('<h3 class="subsection-title">🌳 Sunburst Grafiği</h3>', unsafe_allow_html=True)
        
        sunburst_chart = visualizer.create_sunburst_chart(df)
        if sunburst_chart:
            st.plotly_chart(sunburst_chart, use_container_width=True)
        else:
            st.warning("Sunburst grafiği için gerekli sütunlar bulunamadı.")
    
    elif selected_viz == "Radar Chart (Şirket Karşılaştırma)":
        st.markdown('<h3 class="subsection-title">📡 Radar Chart</h3>', unsafe_allow_html=True)
        
        radar_chart = visualizer.create_radar_chart(df)
        if radar_chart:
            st.plotly_chart(radar_chart, use_container_width=True)
        else:
            st.warning("Radar chart için gerekli veri bulunamadı.")
    
    elif selected_viz == "Sankey Diagram (Pazar Akışı)":
        st.markdown('<h3 class="subsection-title">🌊 Sankey Diagram</h3>', unsafe_allow_html=True)
        
        sankey_chart = visualizer.create_sankey_diagram(df)
        if sankey_chart:
            st.plotly_chart(sankey_chart, use_container_width=True)
        else:
            st.warning("Sankey diagram için gerekli sütunlar bulunamadı.")
    
    # Additional visualizations
    st.markdown('<h3 class="subsection-title">📊 Ek Görselleştirmeler</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        price_cols = [col for col in df.columns if 'Fiyat' in col]
        if price_cols:
            fig = px.histogram(
                df,
                x=price_cols[-1],
                title='Fiyat Dağılımı',
                nbins=50,
                color_discrete_sequence=[Config.COLORS['accent_blue']]
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Growth distribution
        growth_cols = [col for col in df.columns if 'Büyüme_' in col]
        if growth_cols:
            fig = px.box(
                df,
                y=growth_cols[-1],
                title='Büyüme Oranı Dağılımı',
                color_discrete_sequence=[Config.COLORS['accent_cyan']]
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def show_reporting_tab(df: pd.DataFrame, metrics: Dict[str, Any]):
    """Raporlama tab'ını göster"""
    st.markdown('<h2 class="section-title">📑 Raporlama ve İndirme</h2>', unsafe_allow_html=True)
    
    reporting = ReportingEngine()
    
    # Report type selection
    report_type = st.radio(
        "Rapor Türü Seçin",
        options=['Excel Detaylı Rapor', 'HTML Özet Rapor', 'CSV Ham Veri'],
        horizontal=True,
        key="report_type"
    )
    
    # Generate report
    if st.button("📊 Rapor Oluştur", type="primary", use_container_width=True):
        with st.spinner("Rapor oluşturuluyor..."):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if report_type == 'Excel Detaylı Rapor':
                excel_report = reporting.generate_excel_report(
                    df,
                    metrics,
                    st.session_state.insights if hasattr(st.session_state, 'insights') else []
                )
                
                if excel_report:
                    st.download_button(
                        label="⬇️ Excel Raporunu İndir",
                        data=excel_report,
                        file_name=f"pharma_rapor_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            elif report_type == 'HTML Özet Rapor':
                html_report = reporting.generate_html_report(
                    df,
                    metrics,
                    []
                )
                
                if html_report:
                    st.download_button(
                        label="⬇️ HTML Raporunu İndir",
                        data=html_report,
                        file_name=f"pharma_rapor_{timestamp}.html",
                        mime="text/html",
                        use_container_width=True
                    )
            
            elif report_type == 'CSV Ham Veri':
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ CSV Verisini İndir",
                    data=csv_data,
                    file_name=f"pharma_veri_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Data statistics
    st.markdown('<h3 class="subsection-title">📈 Veri İstatistikleri</h3>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with cols[1]:
        st.metric("Toplam Sütun", len(df.columns))
    
    with cols[2]:
        memory_usage = df.memory_usage(deep=True).sum()/1024**2
        st.metric("Bellek Kullanımı", f"{memory_usage:.1f} MB")
    
    with cols[3]:
        missing_values = df.isna().sum().sum()
        st.metric("Eksik Değerler", f"{missing_values:,}")
    
    # Data quality report
    st.markdown('<h3 class="subsection-title">🔍 Veri Kalite Raporu</h3>', unsafe_allow_html=True)
    
    quality_data = []
    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        unique_pct = (df[col].nunique() / len(df)) * 100
        
        quality_data.append({
            'Sütun': col,
            'Tip': str(df[col].dtype),
            'Eksik %': f"{missing_pct:.1f}%",
            'Benzersiz %': f"{unique_pct:.1f}%"
        })
    
    quality_df = pd.DataFrame(quality_data)
    st.dataframe(
        quality_df,
        use_container_width=True,
        height=300
    )

def show_settings_tab():
    """Ayarlar tab'ını göster"""
    st.markdown('<h2 class="section-title">⚙️ Sistem Ayarları ve Performans</h2>', unsafe_allow_html=True)
    
    # Performance settings
    st.markdown('<h3 class="subsection-title">⚡ Performans Ayarları</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        cache_enabled = st.toggle("Önbellekleme Aktif", value=True)
        max_cache_size = st.slider(
            "Max Önbellek Boyutu (MB)",
            min_value=100,
            max_value=1000,
            value=500,
            step=50
        )
    
    with col2:
        parallel_processing = st.toggle("Paralel İşleme", value=True)
        max_workers = st.slider(
            "Max Worker Sayısı",
            min_value=1,
            max_value=8,
            value=4,
            step=1
        )
    
    # Visualization settings
    st.markdown('<h3 class="subsection-title">🎨 Görselleştirme Ayarları</h3>', unsafe_allow_html=True)
    
    theme = st.selectbox(
        "Tema",
        options=['Koyu', 'Açık', 'Otomatik'],
        index=0
    )
    
    chart_height = st.slider(
        "Grafik Yüksekliği",
        min_value=300,
        max_value=800,
        value=Config.CHART_HEIGHT,
        step=50
    )
    
    # Data settings
    st.markdown('<h3 class="subsection-title">📊 Veri Ayarları</h3>', unsafe_allow_html=True)
    
    auto_optimize = st.toggle("Otomatik Optimizasyon", value=True)
    max_rows_display = st.slider(
        "Max Gösterim Satırı",
        min_value=1000,
        max_value=10000,
        value=Config.MAX_ROWS_PREVIEW,
        step=1000
    )
    
    # Save settings
    if st.button("💾 Ayarları Kaydet", type="primary", use_container_width=True):
        # Update config (in a real app, this would save to config file/database)
        Config.MAX_ROWS_PREVIEW = max_rows_display
        Config.CHART_HEIGHT = chart_height
        
        st.success("✅ Ayarlar kaydedildi!")
    
    # System information
    st.markdown('<h3 class="subsection-title">ℹ️ Sistem Bilgileri</h3>', unsafe_allow_html=True)
    
    sys_info = {
        "Streamlit Versiyonu": st.__version__,
        "Pandas Versiyonu": pd.__version__,
        "Numpy Versiyonu": np.__version__,
        "Plotly Versiyonu": px.__version__,
        "Python Versiyonu": "3.9+",
        "Platform": "Enterprise",
        "Son Güncelleme": "2024"
    }
    
    for key, value in sys_info.items():
        st.text(f"{key}: {value}")

# ================================================
# 4. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    try:
        # Enable garbage collection
        gc.enable()
        
        # Set Streamlit session state
        st.session_state.setdefault('app_started', True)
        
        # Run main application
        main()
        
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        st.error("Detaylı hata bilgisi:")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Sayfayı Yenile", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
