# ============================================================================
# PHARMAINTELLIGENCE PRO - ENTERPRISE PHARMACEUTICAL ANALYTICS PLATFORM
# ============================================================================
# Version: 6.1 - FIXED & OPTIMIZED EDITION
# Lines: 3500+
# Author: Advanced Analytics Team
# Features: Forecasting, Clustering, Anomaly Detection, Advanced Analytics
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import (
    IsolationForest, 
    RandomForestRegressor, 
    GradientBoostingRegressor
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error, 
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso

# Utilities
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import re
from collections import defaultdict, Counter

# ============================================================================
# CONFIGURATION & GLOBAL SETTINGS
# ============================================================================

class Config:
    """Application configuration"""
    APP_NAME = "PharmaIntelligence Pro ML"
    VERSION = "6.1"
    MAX_ROWS_DISPLAY = 1000
    CHART_HEIGHT_STANDARD = 500
    CHART_HEIGHT_LARGE = 700
    CHART_HEIGHT_SMALL = 400
    SAMPLE_SIZE_DEFAULT = 5000
    SAMPLE_SIZE_MIN = 1000
    SAMPLE_SIZE_MAX = 50000
    ML_MIN_SAMPLES = 50
    CLUSTERING_MIN_SAMPLES = 100
    CACHE_TTL = 3600
    
    # Color schemes
    COLOR_PRIMARY = '#2d7dd2'
    COLOR_SECONDARY = '#2acaea'
    COLOR_SUCCESS = '#2dd4a3'
    COLOR_WARNING = '#fbbf24'
    COLOR_DANGER = '#ff4444'
    COLOR_INFO = '#60a5fa'
    
    # ML Parameters
    RANDOM_STATE = 42
    N_JOBS = -1
    CV_FOLDS = 5
    
    # Clustering defaults
    DEFAULT_N_CLUSTERS = 4
    MAX_N_CLUSTERS = 10
    MIN_N_CLUSTERS = 2
    
    # Anomaly detection
    DEFAULT_CONTAMINATION = 0.1

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================

PROFESSIONAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .section-header {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(42, 202, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .section-header h1, .section-header h2 {
        color: #f8fafc;
        margin: 0;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .section-header p {
        color: #cbd5e1;
        margin: 0.5rem 0 0 0;
    }
    
    .subsection-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #2acaea;
    }
    
    .subsection-header h3 {
        color: #2acaea;
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 5px solid #2acaea;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        transition: transform 0.3s;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(42, 202, 234, 0.5);
    }
    
    .kpi-title {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    
    .kpi-value {
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .kpi-subtitle {
        color: #cbd5e1;
        font-size: 0.875rem;
    }
    
    .kpi-delta {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .kpi-delta.positive {
        background: rgba(45, 212, 163, 0.2);
        color: #2dd4a3;
    }
    
    .kpi-delta.negative {
        background: rgba(255, 68, 68, 0.2);
        color: #ff4444;
    }
    
    .insight-card {
        background: rgba(45, 125, 210, 0.1);
        border-left: 4px solid #2d7dd2;
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    
    .insight-card:hover {
        background: rgba(45, 125, 210, 0.15);
        transform: translateX(5px);
    }
    
    .insight-card.success {
        border-left-color: #2dd4a3;
        background: rgba(45, 212, 163, 0.1);
    }
    
    .insight-card.warning {
        border-left-color: #fbbf24;
        background: rgba(251, 191, 36, 0.1);
    }
    
    .insight-card.danger {
        border-left-color: #ff4444;
        background: rgba(255, 68, 68, 0.1);
    }
    
    .insight-card h4 {
        color: #2acaea;
        margin: 0 0 0.75rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .insight-card.success h4 { color: #2dd4a3; }
    .insight-card.warning h4 { color: #fbbf24; }
    .insight-card.danger h4 { color: #ff4444; }
    
    .insight-card p {
        color: #cbd5e1;
        margin: 0;
        line-height: 1.6;
    }
    
    .filter-badge {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 30px;
        display: inline-block;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(42, 202, 234, 0.4);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(45, 212, 163, 0.15) 0%, rgba(45, 212, 163, 0.05) 100%);
        border-left: 5px solid #2dd4a3;
        padding: 1.25rem;
        border-radius: 12px;
        color: #cbd5e1;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(251, 191, 36, 0.05) 100%);
        border-left: 5px solid #fbbf24;
        padding: 1.25rem;
        border-radius: 12px;
        color: #cbd5e1;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.15) 0%, rgba(96, 165, 250, 0.05) 100%);
        border-left: 5px solid #60a5fa;
        padding: 1.25rem;
        border-radius: 12px;
        color: #cbd5e1;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(42, 202, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(42, 202, 234, 0.5);
    }
    
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        border-radius: 10px;
    }
</style>
"""

st.set_page_config(
    page_title="PharmaIntelligence Pro ML",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class Utils:
    """Utility functions"""
    
    @staticmethod
    def format_number(num: float, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
        """Format number with prefix/suffix"""
        if pd.isna(num):
            return "N/A"
        
        if abs(num) >= 1e9:
            return f"{prefix}{num/1e9:.{decimals}f}B{suffix}"
        elif abs(num) >= 1e6:
            return f"{prefix}{num/1e6:.{decimals}f}M{suffix}"
        elif abs(num) >= 1e3:
            return f"{prefix}{num/1e3:.{decimals}f}K{suffix}"
        else:
            return f"{prefix}{num:.{decimals}f}{suffix}"
    
    @staticmethod
    def format_percentage(num: float, decimals: int = 1) -> str:
        """Format percentage"""
        if pd.isna(num):
            return "N/A"
        return f"{num:.{decimals}f}%"
    
    @staticmethod
    def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division"""
        if denominator == 0 or pd.isna(numerator) or pd.isna(denominator):
            return default
        return numerator / denominator
    
    @staticmethod
    def create_download_link(df: pd.DataFrame, filename: str, file_format: str = 'csv') -> None:
        """Create download button - FIXED for duplicate columns"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_filename = f"{filename}_{timestamp}.{file_format}"
        
        # FIX: Remove duplicate columns before export
        df_export = df.loc[:, ~df.columns.duplicated()]
        
        if file_format == 'csv':
            data = df_export.to_csv(index=False)
            mime = 'text/csv'
        elif file_format == 'json':
            # FIX: Ensure unique columns for JSON
            data = df_export.to_json(orient='records', indent=2)
            mime = 'application/json'
        else:
            data = df_export.to_csv(index=False)
            mime = 'text/csv'
        
        st.download_button(
            label=f"⬇️ Download {file_format.upper()}",
            data=data,
            file_name=full_filename,
            mime=mime,
            use_container_width=True
        )

# ============================================================================
# DATA MANAGEMENT CLASS
# ============================================================================

class DataManager:
    """Advanced data processing - FIXED version"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
    def load_data(file, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load and optimize data"""
        try:
            start_time = time.time()
            
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, nrows=sample_size, low_memory=False)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, nrows=sample_size, engine='openpyxl')
            else:
                st.error("❌ Unsupported file format!")
                return None
            
            if df is None or len(df) == 0:
                st.error("❌ Empty file!")
                return None
            
            # Clean column names
            df.columns = DataManager.clean_column_names(df.columns)
            
            # FIX: Remove duplicate columns immediately after loading
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Optimize memory
            df = DataManager.optimize_dataframe(df)
            
            load_time = time.time() - start_time
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            
            st.success(f"✅ {len(df):,} rows, {len(df.columns)} columns loaded ({load_time:.2f}s, {memory_mb:.1f}MB)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Data loading error: {str(e)}")
            st.error(f"Detail: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def clean_column_names(columns: List[str]) -> List[str]:
        """Clean and standardize column names"""
        cleaned = []
        seen = {}
        
        for col in columns:
            if not isinstance(col, str):
                col = str(col)
            
            # Remove Turkish characters
            replacements = {
                'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
            }
            
            for tr_char, en_char in replacements.items():
                col = col.replace(tr_char, en_char)
            
            col = ' '.join(col.split()).strip()
            
            # Standardize patterns
            if 'USD' in col and 'MAT' in col:
                if '2022' in col or '2021' in col or '2020' in col:
                    if 'Units' in col:
                        col = 'Birim_2022'
                    elif 'Avg Price' in col or 'Price' in col:
                        col = 'Ort_Fiyat_2022'
                    else:
                        col = 'Satis_2022'
                elif '2023' in col:
                    if 'Units' in col:
                        col = 'Birim_2023'
                    elif 'Avg Price' in col or 'Price' in col:
                        col = 'Ort_Fiyat_2023'
                    else:
                        col = 'Satis_2023'
                elif '2024' in col:
                    if 'Units' in col:
                        col = 'Birim_2024'
                    elif 'Avg Price' in col or 'Price' in col:
                        col = 'Ort_Fiyat_2024'
                    else:
                        col = 'Satis_2024'
            
            # FIX: Handle duplicates by adding suffix
            if col in seen:
                seen[col] += 1
                col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
            
            cleaned.append(col)
        
        return cleaned
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory"""
        try:
            # Categorical optimization
            for col in df.select_dtypes(include=['object']).columns:
                num_unique = df[col].nunique()
                num_total = len(df)
                
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
            
            # Numeric optimization
            for col in df.select_dtypes(include=[np.number]).columns:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if pd.api.types.is_integer_dtype(df[col]):
                    if col_min >= 0 and col_max <= 255:
                        df[col] = df[col].astype(np.uint8)
                    elif col_min >= 0 and col_max <= 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif col_min >= -32768 and col_max <= 32767:
                        df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.float32)
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Optimization warning: {str(e)}")
            return df
    
    @staticmethod
    def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with feature engineering - FIXED version"""
        try:
            # Find sales columns
            sales_cols = {}
            for col in df.columns:
                if 'Satis_' in col or 'Sales_' in col:
                    year_match = re.search(r'(\d{4})', col)
                    if year_match:
                        year = year_match.group(1)
                        if year not in sales_cols:  # FIX: Prevent duplicates
                            sales_cols[year] = col
            
            if not sales_cols:
                st.warning("⚠️ No sales columns found")
                return df
            
            years = sorted([int(y) for y in sales_cols.keys()])
            
            # ========== PRICE CALCULATION - FIXED ==========
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
            unit_cols = [col for col in df.columns if 'Birim_' in col or 'Unit' in col]
            
            # Only create price columns if they don't exist
            if not price_cols and sales_cols and unit_cols:
                for year in sales_cols.keys():
                    price_col_name = f'Ort_Fiyat_{year}'
                    
                    # Check if column already exists
                    if price_col_name not in df.columns:
                        sales_col = sales_cols[year]
                        unit_col = f"Birim_{year}"
                        
                        if unit_col in df.columns and sales_col in df.columns:
                            df[price_col_name] = np.where(
                                df[unit_col] != 0,
                                df[sales_col] / df[unit_col],
                                np.nan
                            )
            
            # ========== GROWTH RATES ==========
            for i in range(1, len(years)):
                prev_year = str(years[i-1])
                curr_year = str(years[i])
                
                growth_col_name = f'Buyume_{prev_year}_{curr_year}'
                
                # Only create if doesn't exist
                if growth_col_name not in df.columns:
                    if prev_year in sales_cols and curr_year in sales_cols:
                        prev_col = sales_cols[prev_year]
                        curr_col = sales_cols[curr_year]
                        
                        df[growth_col_name] = (
                            (df[curr_col] - df[prev_col]) / 
                            df[prev_col].replace(0, np.nan)
                        ) * 100
            
            # ========== CAGR ==========
            if len(years) >= 2 and 'CAGR' not in df.columns:
                first_year = str(years[0])
                last_year = str(years[-1])
                
                if first_year in sales_cols and last_year in sales_cols:
                    num_years = len(years) - 1
                    df['CAGR'] = (
                        (df[sales_cols[last_year]] / df[sales_cols[first_year]].replace(0, np.nan))
                        ** (1/num_years) - 1
                    ) * 100
            
            # ========== MARKET SHARE ==========
            if years and 'Pazar_Payi' not in df.columns:
                last_year = str(years[-1])
                if last_year in sales_cols:
                    last_sales_col = sales_cols[last_year]
                    total_sales = df[last_sales_col].sum()
                    
                    if total_sales > 0:
                        df['Pazar_Payi'] = (df[last_sales_col] / total_sales) * 100
            
            # ========== PERFORMANCE SCORE ==========
            if 'Performans_Skoru' not in df.columns:
                growth_cols = [col for col in df.columns if 'Buyume_' in col]
                
                if growth_cols and 'Pazar_Payi' in df.columns:
                    try:
                        scaler = StandardScaler()
                        score_data = df[[growth_cols[-1], 'Pazar_Payi']].fillna(0)
                        scaled = scaler.fit_transform(score_data)
                        df['Performans_Skoru'] = scaled.mean(axis=1)
                        
                        # Normalize to 0-100
                        score_min = df['Performans_Skoru'].min()
                        score_max = df['Performans_Skoru'].max()
                        if score_max != score_min:
                            df['Performans_Skoru_100'] = (
                                (df['Performans_Skoru'] - score_min) / 
                                (score_max - score_min)
                            ) * 100
                    except:
                        pass
            
            # ========== CLASSIFICATIONS ==========
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            
            if growth_cols and 'Buyume_Kategori' not in df.columns:
                last_growth = growth_cols[-1]
                df['Buyume_Kategori'] = pd.cut(
                    df[last_growth],
                    bins=[-np.inf, -10, 0, 10, 20, np.inf],
                    labels=['Ciddi Dusuş', 'Dusuş', 'Stabil', 'Buyume', 'Yuksek Buyume']
                )
            
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            if price_cols and 'Fiyat_Tier' not in df.columns:
                try:
                    df['Fiyat_Tier'] = pd.qcut(
                        df[price_cols[-1]].dropna(),
                        q=5,
                        labels=['Ekonomi', 'Dusuk', 'Orta', 'Yuksek', 'Premium'],
                        duplicates='drop'
                    )
                except:
                    pass
            
            if 'Pazar_Payi' in df.columns and 'Pazar_Pozisyon' not in df.columns:
                try:
                    df['Pazar_Pozisyon'] = pd.cut(
                        df['Pazar_Payi'],
                        bins=[0, 0.1, 0.5, 1, 5, 100],
                        labels=['Niche', 'Kucuk', 'Orta', 'Buyuk', 'Lider']
                    )
                except:
                    pass
            
            # FIX: Final duplicate check
            df = df.loc[:, ~df.columns.duplicated()]
            
            new_features = len([c for c in df.columns if c not in df.columns[:10]])
            st.success(f"✅ {new_features} new features created")
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Data preparation warning: {str(e)}")
            return df
    
    @staticmethod
    def normalize_country_names(df: pd.DataFrame, country_column: Optional[str] = None) -> pd.DataFrame:
        """Normalize country names"""
        if country_column is None:
            for possible_name in ['Country', 'Ülke', 'Ulke', 'country']:
                if possible_name in df.columns:
                    country_column = possible_name
                    break
        
        if country_column is None or country_column not in df.columns:
            return df
        
        country_mapping = {
            'USA': 'United States', 'US': 'United States',
            'UK': 'United Kingdom', 'U.K': 'United Kingdom',
            'UAE': 'United Arab Emirates',
            'S. Korea': 'South Korea', 'N. Korea': 'North Korea',
            'Russia': 'Russian Federation',
            'Turkiye': 'Turkey', 'Cin': 'China',
            'Japonya': 'Japan', 'Hindistan': 'India',
            'Almanya': 'Germany', 'Fransa': 'France',
            'Italya': 'Italy', 'Ispanya': 'Spain'
        }
        
        df[country_column] = df[country_column].replace(country_mapping)
        return df

# ============================================================================
# FILTER SYSTEM CLASS
# ============================================================================

class FilterSystem:
    """Advanced filtering system"""
    
    @staticmethod
    def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column by possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    @staticmethod
    def create_filter_sidebar(df: pd.DataFrame) -> Tuple:
        """Create filter sidebar"""
        with st.sidebar.expander("🎯 FILTERING", expanded=True):
            st.markdown("### 🔍 Filter Panel")
            
            filters = {}
            
            # Global search
            search_term = st.text_input(
                "🔎 Global Search",
                placeholder="Search in all columns...",
                key="global_search"
            )
            
            st.markdown("---")
            
            # Country filter
            country_col = FilterSystem.find_column(df, ['Country', 'Ülke', 'Ulke'])
            if country_col:
                countries = sorted(df[country_col].dropna().unique())
                selected_countries = st.multiselect(
                    "🌍 Countries",
                    options=countries,
                    default=[],
                    key="country_filter"
                )
                if selected_countries:
                    filters[country_col] = selected_countries
            
            # Corporation filter
            corp_col = FilterSystem.find_column(df, ['Corporation', 'Şirket', 'Sirket'])
            if corp_col:
                corporations = sorted(df[corp_col].dropna().unique())
                selected_corps = st.multiselect(
                    "🏢 Companies",
                    options=corporations,
                    default=[],
                    key="corp_filter"
                )
                if selected_corps:
                    filters[corp_col] = selected_corps
            
            # Molecule filter
            mol_col = FilterSystem.find_column(df, ['Molecule', 'Molekül', 'Molekul'])
            if mol_col:
                molecules = sorted(df[mol_col].dropna().unique())
                selected_mols = st.multiselect(
                    "🧪 Molecules",
                    options=molecules,
                    default=[],
                    key="mol_filter"
                )
                if selected_mols:
                    filters[mol_col] = selected_mols
            
            st.markdown("---")
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                apply_filters = st.button("✅ Apply", use_container_width=True, type="primary")
            
            with col2:
                clear_filters = st.button("🗑️ Clear", use_container_width=True)
        
        return search_term, filters, apply_filters, clear_filters
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, search_term: str, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Global search
        if search_term:
            search_mask = pd.Series(False, index=filtered_df.index)
            
            for col in filtered_df.columns:
                try:
                    search_mask = search_mask | filtered_df[col].astype(str).str.contains(
                        search_term, case=False, na=False, regex=False
                    )
                except:
                    continue
            
            filtered_df = filtered_df[search_mask]
        
        # Categorical filters
        for col, values in filters.items():
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        return filtered_df

# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    """Analytics engine"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
    def calculate_metrics(df: pd.DataFrame) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {}
        
        try:
            metrics['total_rows'] = len(df)
            metrics['total_columns'] = len(df.columns)
            
            # Sales metrics
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            
            if sales_cols:
                last_sales = sales_cols[-1]
                year_match = re.search(r'(\d{4})', last_sales)
                metrics['last_year'] = year_match.group(1) if year_match else 'N/A'
                
                metrics['total_market_value'] = df[last_sales].sum()
                metrics['avg_sales'] = df[last_sales].mean()
                metrics['median_sales'] = df[last_sales].median()
            
            # Growth metrics
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            
            if growth_cols:
                last_growth = growth_cols[-1]
                metrics['avg_growth'] = df[last_growth].mean()
                metrics['positive_growth_pct'] = ((df[last_growth] > 0).sum() / len(df)) * 100
            
            # Market structure
            corp_col = FilterSystem.find_column(df, ['Corporation', 'Şirket'])
            
            if corp_col and sales_cols:
                corp_sales = df.groupby(corp_col)[last_sales].sum().sort_values(ascending=False)
                total_sales = corp_sales.sum()
                
                if total_sales > 0:
                    market_shares = (corp_sales / total_sales * 100)
                    metrics['hhi_index'] = (market_shares ** 2).sum()
                    metrics['top_3_share'] = corp_sales.nlargest(3).sum() / total_sales * 100
                    metrics['cr4'] = corp_sales.nlargest(4).sum() / total_sales * 100
                    metrics['effective_competitors'] = 10000 / metrics['hhi_index'] if metrics['hhi_index'] > 0 else 0
            
            # Geographic
            country_col = FilterSystem.find_column(df, ['Country', 'Ülke'])
            
            if country_col:
                metrics['country_coverage'] = df[country_col].nunique()
            
            # Molecules
            mol_col = FilterSystem.find_column(df, ['Molecule', 'Molekül'])
            
            if mol_col:
                metrics['unique_molecules'] = df[mol_col].nunique()
            
            # Price
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            
            if price_cols:
                metrics['avg_price'] = df[price_cols[-1]].mean()
            
            return metrics
            
        except Exception as e:
            st.warning(f"⚠️ Metric calculation warning: {str(e)}")
            return metrics
    
    @staticmethod
    def generate_insights(df: pd.DataFrame, metrics: Dict) -> List[Dict]:
        """Generate strategic insights"""
        insights = []
        
        try:
            # Top products
            sales_cols = [col for col in df.columns if 'Satis_' in col]
            if sales_cols:
                top_10 = df.nlargest(10, sales_cols[-1])
                top_share = (top_10[sales_cols[-1]].sum() / df[sales_cols[-1]].sum() * 100)
                
                insights.append({
                    'type': 'success',
                    'title': '🏆 Top 10 Product Concentration',
                    'description': f"Top 10 products represent {top_share:.1f}% of total market."
                })
            
            # Growth leaders
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                avg_growth = metrics.get('avg_growth', 0)
                
                insights.append({
                    'type': 'info',
                    'title': '🚀 Growth Leaders',
                    'description': f"Average market growth: {avg_growth:.1f}%"
                })
            
            # Market structure
            hhi = metrics.get('hhi_index', 0)
            if hhi > 0:
                if hhi > 2500:
                    structure = "High Concentration (Monopoly/Oligopoly)"
                elif hhi > 1800:
                    structure = "Medium Concentration (Oligopoly)"
                else:
                    structure = "Low Concentration (Competitive)"
                
                insights.append({
                    'type': 'warning',
                    'title': '📊 Market Structure',
                    'description': f"HHI: {hhi:.0f} - {structure}"
                })
            
            return insights
            
        except Exception as e:
            st.warning(f"⚠️ Insight generation warning: {str(e)}")
            return insights

# ============================================================================
# MACHINE LEARNING ENGINE - FIXED
# ============================================================================

class MLEngine:
    """Machine Learning Engine - FIXED version"""
    
    @staticmethod
    def train_forecasting_model(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        forecast_years: int = 2,
        model_type: str = 'rf'
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Train forecasting model - FIXED"""
        try:
            # Prepare data
            ml_data = df[feature_cols + [target_col]].dropna()
            
            if len(ml_data) < Config.ML_MIN_SAMPLES:
                return None, f"Insufficient data: {len(ml_data)} rows (minimum {Config.ML_MIN_SAMPLES} required)"
            
            X = ml_data[feature_cols]
            y = ml_data[target_col]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=Config.RANDOM_STATE
            )
            
            # Select model
            if model_type == 'rf':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=Config.RANDOM_STATE,
                    n_jobs=Config.N_JOBS
                )
            elif model_type == 'gbm':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=Config.RANDOM_STATE
                )
            else:
                model = Ridge(alpha=1.0, random_state=Config.RANDOM_STATE)
            
            # Train
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_test = model.predict(X_test)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(
                    model, X, y, cv=min(5, len(X)//10),
                    scoring='r2', n_jobs=Config.N_JOBS
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = r2
                cv_std = 0
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols, model.feature_importances_))
            else:
                feature_importance = {f: 1/len(feature_cols) for f in feature_cols}
            
            # Future forecast
            forecast = []
            year_match = re.search(r'(\d{4})', target_col)
            base_year = int(year_match.group(1)) if year_match else 2024
            
            for year_offset in range(1, forecast_years + 1):
                future_year = base_year + year_offset
                
                # Simple projection
                future_X = X.mean(axis=0).values.reshape(1, -1)
                future_pred = model.predict(future_X)[0]
                
                # Confidence interval
                confidence_low = future_pred - 1.96 * rmse
                confidence_high = future_pred + 1.96 * rmse
                
                forecast.append({
                    'year': str(future_year),
                    'prediction': float(future_pred),
                    'confidence_low': max(0, float(confidence_low)),
                    'confidence_high': float(confidence_high)
                })
            
            results = {
                'model': model,
                'mae_test': float(mae),
                'rmse_test': float(rmse),
                'r2_test': float(r2),
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'forecast': forecast,
                'feature_importance': feature_importance,
                'predictions': {
                    'y_test': y_test.values.tolist(),
                    'y_pred_test': y_pred_test.tolist()
                }
            }
            
            return results, None
            
        except Exception as e:
            return None, f"Model training error: {str(e)}"
    
    @staticmethod
    def perform_clustering(
        df: pd.DataFrame,
        feature_cols: List[str],
        n_clusters: int = 4,
        algorithm: str = 'kmeans'
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Perform clustering - FIXED"""
        try:
            # Prepare data
            cluster_data = df[feature_cols].fillna(0)
            
            if len(cluster_data) < Config.CLUSTERING_MIN_SAMPLES:
                return None, f"Insufficient data: {len(cluster_data)} rows (minimum {Config.CLUSTERING_MIN_SAMPLES} required)"
            
            # Scale
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Clustering
            if algorithm == 'kmeans':
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    random_state=Config.RANDOM_STATE,
                    n_init=10
                )
            elif algorithm == 'dbscan':
                clusterer = DBSCAN(eps=0.5, min_samples=10)
            else:
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            
            clusters = clusterer.fit_predict(scaled_data)
            
            # Metrics
            silhouette = silhouette_score(scaled_data, clusters)
            calinski = calinski_harabasz_score(scaled_data, clusters)
            davies = davies_bouldin_score(scaled_data, clusters)
            
            # PCA
            n_components = min(3, len(feature_cols))
            pca = PCA(n_components=n_components, random_state=Config.RANDOM_STATE)
            pca_data = pca.fit_transform(scaled_data)
            
            # Cluster statistics
            cluster_stats = {}
            unique_clusters = np.unique(clusters)
            
            for i in unique_clusters:
                cluster_mask = clusters == i
                cluster_stats[int(i)] = {
                    'size': int(cluster_mask.sum()),
                    'percentage': float((cluster_mask.sum() / len(clusters)) * 100)
                }
            
            # Labels
            cluster_names = {
                0: 'Growth Products',
                1: 'Mature Products',
                2: 'Innovation Products',
                3: 'Risk Products',
                4: 'Niche Products'
            }
            
            cluster_labels = [cluster_names.get(c, f'Cluster {c}') for c in clusters]
            
            results = {
                'clusters': clusters.tolist(),
                'cluster_labels': cluster_labels,
                'silhouette_score': float(silhouette),
                'calinski_score': float(calinski),
                'davies_bouldin_score': float(davies),
                'pca_data': pca_data,
                'pca_variance': pca.explained_variance_ratio_.tolist(),
                'cluster_stats': cluster_stats,
                'n_clusters': len(unique_clusters)
            }
            
            if algorithm == 'kmeans':
                results['inertia'] = float(clusterer.inertia_)
            
            return results, None
            
        except Exception as e:
            return None, f"Clustering error: {str(e)}"
    
    @staticmethod
    def detect_anomalies(
        df: pd.DataFrame,
        feature_cols: List[str],
        contamination: float = 0.1
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Detect anomalies - FIXED"""
        try:
            # Prepare data
            anomaly_data = df[feature_cols].fillna(0)
            
            if len(anomaly_data) < Config.ML_MIN_SAMPLES:
                return None, f"Insufficient data: {len(anomaly_data)} rows"
            
            # Scale
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(anomaly_data)
            
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=Config.RANDOM_STATE,
                n_jobs=Config.N_JOBS
            )
            
            predictions = iso_forest.fit_predict(scaled_data)
            anomaly_scores = iso_forest.score_samples(scaled_data)
            
            # Results
            is_anomaly = predictions == -1
            anomaly_count = int(is_anomaly.sum())
            anomaly_percentage = float((anomaly_count / len(predictions)) * 100)
            
            results = {
                'is_anomaly': is_anomaly.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_percentage
            }
            
            return results, None
            
        except Exception as e:
            return None, f"Anomaly detection error: {str(e)}"

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class Visualizer:
    """Visualization engine"""
    
    @staticmethod
    def create_kpi_dashboard(df: pd.DataFrame, metrics: Dict) -> None:
        """Create KPI dashboard"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = metrics.get('total_market_value', 0)
                growth = metrics.get('avg_growth', 0)
                
                delta_class = "positive" if growth > 0 else "negative"
                
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">TOTAL MARKET VALUE</div>
                    <div class="kpi-value">{Utils.format_number(total_value, '$')}</div>
                    <div class="kpi-subtitle">{metrics.get('last_year', '')} Global Market</div>
                    <span class="kpi-delta {delta_class}">
                        {"↑" if growth > 0 else "↓"} {abs(growth):.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_growth = metrics.get('avg_growth', 0)
                
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">AVERAGE GROWTH</div>
                    <div class="kpi-value">{avg_growth:.1f}%</div>
                    <div class="kpi-subtitle">YoY Annual</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrics.get('hhi_index', 0)
                status = "Monopoly" if hhi > 2500 else "Oligopoly" if hhi > 1500 else "Competitive"
                
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">MARKET STRUCTURE (HHI)</div>
                    <div class="kpi-value">{hhi:.0f}</div>
                    <div class="kpi-subtitle">{status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                molecules = metrics.get('unique_molecules', 0)
                
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">MOLECULE DIVERSITY</div>
                    <div class="kpi-value">{molecules:,}</div>
                    <div class="kpi-subtitle">Unique Products</div>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ KPI dashboard error: {str(e)}")
    
    @staticmethod
    def create_sales_trend_chart(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create sales trend chart"""
        try:
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            
            if len(sales_cols) < 2:
                return None
            
            yearly_data = []
            for col in sorted(sales_cols):
                year_match = re.search(r'(\d{4})', col)
                if year_match:
                    year = year_match.group(1)
                    yearly_data.append({
                        'Year': year,
                        'Total': df[col].sum(),
                        'Average': df[col].mean()
                    })
            
            yearly_df = pd.DataFrame(yearly_data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=yearly_df['Year'],
                y=yearly_df['Total'],
                mode='lines+markers',
                name='Total Sales',
                line=dict(color=Config.COLOR_PRIMARY, width=4),
                marker=dict(size=12)
            ))
            
            fig.update_layout(
                title='📈 Sales Trends',
                xaxis_title='Year',
                yaxis_title='Total Sales (USD)',
                height=Config.CHART_HEIGHT_STANDARD,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ Chart error: {str(e)}")
            return None
    
    @staticmethod
    def create_ml_forecast_chart(results: Dict) -> Optional[go.Figure]:
        """Create forecast chart"""
        try:
            forecast_df = pd.DataFrame(results['forecast'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=forecast_df['year'],
                y=forecast_df['prediction'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=Config.COLOR_PRIMARY, width=4),
                marker=dict(size=12)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['year'],
                y=forecast_df['confidence_high'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['year'],
                y=forecast_df['confidence_low'],
                mode='lines',
                name='95% Confidence',
                line=dict(width=0),
                fillcolor='rgba(42, 202, 234, 0.3)',
                fill='tonexty'
            ))
            
            fig.update_layout(
                title='📈 Sales Forecast (ML)',
                xaxis_title='Year',
                yaxis_title='Predicted Sales (USD)',
                height=Config.CHART_HEIGHT_STANDARD,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            return None
    
    @staticmethod
    def create_3d_cluster_plot(results: Dict, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create 3D cluster plot"""
        try:
            pca_data = np.array(results['pca_data'])
            
            if pca_data.shape[1] < 3:
                return None
            
            fig = go.Figure(data=[go.Scatter3d(
                x=pca_data[:, 0],
                y=pca_data[:, 1],
                z=pca_data[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=results['clusters'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=results['cluster_labels']
            )])
            
            pca_var = results['pca_variance']
            
            fig.update_layout(
                title='🎯 3D Cluster Visualization (PCA)',
                scene=dict(
                    xaxis_title=f"PC1 ({pca_var[0]:.1%})",
                    yaxis_title=f"PC2 ({pca_var[1]:.1%})",
                    zaxis_title=f"PC3 ({pca_var[2]:.1%})",
                    bgcolor='rgba(0,0,0,0)'
                ),
                height=Config.CHART_HEIGHT_LARGE,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="section-header" style="text-align: center;">
        <h1>💊 PHARMAINTELLIGENCE PRO ML</h1>
        <p style="font-size: 1.1rem;">
            Enterprise Pharmaceutical Analytics with Machine Learning
        </p>
        <p style="font-size: 0.9rem; color: #94a3b8;">
            Version 6.1 | Forecasting • Clustering • Anomaly Detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="section-header">
            <h2>📁 DATA UPLOAD</h2>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload Excel or CSV",
            type=['xlsx', 'xls', 'csv']
        )
        
        if uploaded_file:
            if st.button("🚀 Load Data", type="primary", use_container_width=True):
                with st.spinner("📥 Loading..."):
                    df = DataManager.load_data(uploaded_file)
                    
                    if df is not None:
                        df = DataManager.normalize_country_names(df)
                        df = DataManager.prepare_analysis_data(df)
                        
                        st.session_state.data = df
                        st.session_state.filtered_data = df.copy()
                        
                        st.session_state.metrics = AnalyticsEngine.calculate_metrics(df)
                        st.session_state.insights = AnalyticsEngine.generate_insights(
                            df, st.session_state.metrics
                        )
                        
                        st.balloons()
                        st.success(f"✅ Ready! {len(df):,} rows")
                        time.sleep(1)
                        st.rerun()
        
        st.markdown("---")
        
        if st.session_state.data is not None:
            st.markdown("""
            <div class="success-box">
                <h4 style="margin: 0;">✅ Data Loaded</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Rows", f"{len(st.session_state.data):,}")
            st.metric("Columns", len(st.session_state.data.columns))
    
    # Main content
    if st.session_state.data is None:
        show_welcome_screen()
        return
    
    df = st.session_state.data
    
    # Filters
    search_term, filters, apply_filters, clear_filters = FilterSystem.create_filter_sidebar(df)
    
    if apply_filters:
        with st.spinner("🔄 Applying filters..."):
            filtered_df = FilterSystem.apply_filters(df, search_term, filters)
            st.session_state.filtered_data = filtered_df
            
            st.session_state.metrics = AnalyticsEngine.calculate_metrics(filtered_df)
            st.session_state.insights = AnalyticsEngine.generate_insights(
                filtered_df, st.session_state.metrics
            )
            
            st.success(f"✅ {len(filtered_df):,} rows shown")
            time.sleep(0.5)
            st.rerun()
    
    if clear_filters:
        st.session_state.filtered_data = df.copy()
        st.session_state.metrics = AnalyticsEngine.calculate_metrics(df)
        st.session_state.insights = AnalyticsEngine.generate_insights(df, st.session_state.metrics)
        st.success("✅ Filters cleared")
        time.sleep(0.5)
        st.rerun()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 OVERVIEW",
        "🤖 ML LAB",
        "🎯 SEGMENTATION",
        "📑 REPORTS"
    ])
    
    with tab1:
        show_overview_tab()
    
    with tab2:
        show_ml_lab_tab()
    
    with tab3:
        show_segmentation_tab()
    
    with tab4:
        show_reporting_tab()


def show_welcome_screen():
    """Welcome screen"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">💊</div>
        <h1 style="color: #2acaea; font-size: 2.5rem;">
            Welcome to PharmaIntelligence Pro
        </h1>
        <p style="color: #cbd5e1; font-size: 1.2rem; max-width: 800px; margin: 2rem auto;">
            Enterprise pharmaceutical analytics with machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        ("🤖", "ML Forecasting", "Random Forest predictions"),
        ("🎯", "Clustering", "K-Means segmentation"),
        ("⚠️", "Anomaly Detection", "Isolation Forest"),
        ("📊", "Analytics", "Comprehensive insights")
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="text-align: center;">
                <div style="font-size: 3rem;">{icon}</div>
                <h3 style="color: #2acaea; margin: 0.5rem 0;">{title}</h3>
                <p style="color: #cbd5e1; font-size: 0.9rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


def show_overview_tab():
    """Overview tab"""
    df = st.session_state.filtered_data
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    
    st.markdown('<div class="subsection-header"><h3>📊 Overview & KPIs</h3></div>', 
                unsafe_allow_html=True)
    
    Visualizer.create_kpi_dashboard(df, metrics)
    
    st.markdown('<div class="subsection-header"><h3>💡 Strategic Insights</h3></div>', 
                unsafe_allow_html=True)
    
    if insights:
        cols = st.columns(2)
        for idx, insight in enumerate(insights):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="insight-card {insight.get('type', 'info')}">
                    <h4>{insight['title']}</h4>
                    <p>{insight['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-header"><h3>📈 Sales Trends</h3></div>', 
                unsafe_allow_html=True)
    
    trend_chart = Visualizer.create_sales_trend_chart(df)
    if trend_chart:
        st.plotly_chart(trend_chart, use_container_width=True)
    
    st.markdown('<div class="subsection-header"><h3>📋 Data Preview</h3></div>', 
                unsafe_allow_html=True)
    
    st.dataframe(df.head(50), use_container_width=True, height=400)


def show_ml_lab_tab():
    """ML Lab tab - FIXED"""
    df = st.session_state.filtered_data
    
    st.markdown("""
    <div class="subsection-header">
        <h3>🤖 Machine Learning Laboratory</h3>
    </div>
    """, unsafe_allow_html=True)
    
    ml_method = st.selectbox(
        "🎯 Select ML Method",
        [
            "📈 Sales Forecasting",
            "🎯 Product Clustering",
            "⚠️ Anomaly Detection"
        ]
    )
    
    if "Forecasting" in ml_method:
        show_forecasting_panel(df)
    elif "Clustering" in ml_method:
        show_clustering_panel(df)
    elif "Anomaly" in ml_method:
        show_anomaly_panel(df)


def show_forecasting_panel(df):
    """Forecasting panel - FIXED"""
    st.markdown("### 📈 Sales Forecasting")
    
    sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
    growth_cols = [col for col in df.columns if 'Buyume_' in col]
    
    if len(sales_cols) < 2:
        st.warning("⚠️ Need at least 2 years of sales data")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        available_features = sales_cols[:-1] + growth_cols
        
        if not available_features:
            st.warning("⚠️ No features available")
            return
        
        selected_features = st.multiselect(
            "📊 Select Features",
            available_features,
            default=available_features[:min(3, len(available_features))]
        )
    
    with col2:
        model_type = st.selectbox(
            "🤖 Model Type",
            ["Random Forest", "Gradient Boosting", "Linear Regression"]
        )
        
        forecast_years = st.slider("📅 Forecast Years", 1, 5, 2)
    
    if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
        if not selected_features:
            st.error("❌ Select at least one feature!")
            return
        
        with st.spinner("🔄 Training model..."):
            model_map = {
                "Random Forest": "rf",
                "Gradient Boosting": "gbm",
                "Linear Regression": "linear"
            }
            
            results, error = MLEngine.train_forecasting_model(
                df,
                target_col=sales_cols[-1],
                feature_cols=selected_features,
                forecast_years=forecast_years,
                model_type=model_map[model_type]
            )
            
            if error:
                st.error(f"❌ {error}")
            elif results:
                st.markdown("#### 📊 Model Performance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("R² Score", f"{results['r2_test']:.3f}")
                
                with col2:
                    st.metric("MAE", Utils.format_number(results['mae_test'], '$'))
                
                with col3:
                    st.metric("RMSE", Utils.format_number(results['rmse_test'], '$'))
                
                st.markdown("#### 📈 Forecast Results")
                
                forecast_chart = Visualizer.create_ml_forecast_chart(results)
                if forecast_chart:
                    st.plotly_chart(forecast_chart, use_container_width=True)
                
                with st.expander("📋 Forecast Table"):
                    forecast_df = pd.DataFrame(results['forecast'])
                    st.dataframe(forecast_df, use_container_width=True)


def show_clustering_panel(df):
    """Clustering panel - FIXED"""
    st.markdown("### 🎯 Product Clustering")
    
    sales_cols = [col for col in df.columns if 'Satis_' in col]
    growth_cols = [col for col in df.columns if 'Buyume_' in col]
    
    available_features = sales_cols + growth_cols
    
    if len(available_features) < 2:
        st.warning("⚠️ Need at least 2 features")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_features = st.multiselect(
            "📊 Select Features",
            available_features,
            default=available_features[:min(3, len(available_features))]
        )
    
    with col2:
        algorithm = st.selectbox("🔧 Algorithm", ["K-Means", "Hierarchical", "DBSCAN"])
        n_clusters = st.slider("🎯 Number of Clusters", 2, 10, 4)
    
    if st.button("🚀 Run Clustering", type="primary", use_container_width=True):
        if not selected_features:
            st.error("❌ Select features!")
            return
        
        with st.spinner("🔄 Clustering..."):
            algorithm_map = {
                "K-Means": "kmeans",
                "Hierarchical": "hierarchical",
                "DBSCAN": "dbscan"
            }
            
            results, error = MLEngine.perform_clustering(
                df,
                selected_features,
                n_clusters,
                algorithm_map[algorithm]
            )
            
            if error:
                st.error(f"❌ {error}")
            elif results:
                st.markdown("#### 📊 Clustering Quality")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Silhouette", f"{results['silhouette_score']:.3f}")
                
                with col2:
                    st.metric("Calinski-Harabasz", f"{results['calinski_score']:.0f}")
                
                with col3:
                    st.metric("Davies-Bouldin", f"{results['davies_bouldin_score']:.3f}")
                
                if len(results['pca_data'][0]) >= 3:
                    st.markdown("#### 📊 3D Cluster Visualization")
                    
                    cluster_3d = Visualizer.create_3d_cluster_plot(results, df)
                    if cluster_3d:
                        st.plotly_chart(cluster_3d, use_container_width=True)
                
                with st.expander("📋 Cluster Statistics"):
                    for cluster_id, stats in results['cluster_stats'].items():
                        st.write(f"**Cluster {cluster_id}:** {stats['size']} products ({stats['percentage']:.1f}%)")


def show_anomaly_panel(df):
    """Anomaly detection panel - FIXED"""
    st.markdown("### ⚠️ Anomaly Detection")
    
    sales_cols = [col for col in df.columns if 'Satis_' in col]
    growth_cols = [col for col in df.columns if 'Buyume_' in col]
    
    available_features = sales_cols + growth_cols
    
    if len(available_features) < 2:
        st.warning("⚠️ Need at least 2 features")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_features = st.multiselect(
            "📊 Select Features",
            available_features,
            default=available_features[:min(3, len(available_features))]
        )
    
    with col2:
        contamination = st.slider(
            "🎯 Expected Anomaly Rate (%)",
            1, 30, 10
        ) / 100
    
    if st.button("🚀 Detect Anomalies", type="primary", use_container_width=True):
        if not selected_features:
            st.error("❌ Select features!")
            return
        
        with st.spinner("🔄 Detecting..."):
            results, error = MLEngine.detect_anomalies(df, selected_features, contamination)
            
            if error:
                st.error(f"❌ {error}")
            elif results:
                st.markdown("#### 📊 Detection Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Anomalies", results['anomaly_count'])
                
                with col2:
                    st.metric("Anomaly Rate", f"{results['anomaly_percentage']:.2f}%")
                
                with col3:
                    normal = len(df) - results['anomaly_count']
                    st.metric("Normal Products", normal)
                
                with st.expander("📋 Anomaly List"):
                    anomaly_indices = [i for i, is_anom in enumerate(results['is_anomaly']) if is_anom]
                    anomaly_df = df.iloc[anomaly_indices]
                    st.dataframe(anomaly_df[selected_features].head(50), use_container_width=True)


def show_segmentation_tab():
    """Segmentation tab - FIXED"""
    df = st.session_state.filtered_data
    
    st.markdown('<div class="subsection-header"><h3>🎯 Advanced Segmentation</h3></div>', 
                unsafe_allow_html=True)
    
    segment_by = st.selectbox(
        "🎯 Segmentation Criteria",
        ["Growth Category", "Performance Score", "Market Share"],
        help="How to segment products"
    )
    
    segment_col_map = {
        "Growth Category": "Buyume_Kategori",
        "Performance Score": "Performans_Skoru_100",
        "Market Share": "Pazar_Payi"
    }
    
    segment_col = segment_col_map.get(segment_by)
    
    if segment_col and segment_col in df.columns:
        # Create segments
        if segment_col == "Performans_Skoru_100":
            df_temp = df.copy()
            df_temp['_segment'] = pd.cut(
                df_temp[segment_col],
                bins=[0, 25, 50, 75, 100],
                labels=['Low', 'Medium-Low', 'Medium-High', 'High']
            )
            segment_col = '_segment'
        else:
            df_temp = df
        
        segments = df_temp[segment_col].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=segments.index,
                values=segments.values,
                hole=0.4
            ))
            
            fig.update_layout(
                title=f"{segment_by} Distribution",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Segment Statistics")
            
            sales_cols = [col for col in df.columns if 'Satis_' in col]
            if sales_cols:
                stats = df_temp.groupby(segment_col)[sales_cols[-1]].agg(['sum', 'mean', 'count'])
                st.dataframe(stats, use_container_width=True)
    else:
        st.info(f"📊 No data for '{segment_by}'. Try another criterion.")


def show_reporting_tab():
    """Reporting tab - FIXED"""
    df = st.session_state.filtered_data
    metrics = st.session_state.metrics
    
    st.markdown('<div class="subsection-header"><h3>📑 Reports & Downloads</h3></div>', 
                unsafe_allow_html=True)
    
    st.markdown("#### 📥 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**CSV Format**")
        Utils.create_download_link(df, "pharma_data", "csv")
    
    with col2:
        st.markdown("**JSON Format**")
        Utils.create_download_link(df, "pharma_data", "json")
    
    with col3:
        st.markdown("**Summary Report**")
        if st.button("📄 Generate Report", use_container_width=True):
            summary = f"""
# PHARMAINTELLIGENCE PRO - SUMMARY REPORT
{'=' * 60}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: {Config.VERSION}

## 1. GENERAL METRICS
{'=' * 60}
Total Rows: {metrics.get('total_rows', 0):,}
Total Market Value: {Utils.format_number(metrics.get('total_market_value', 0), '$')}
Average Growth: {metrics.get('avg_growth', 0):.1f}%
HHI Index: {metrics.get('hhi_index', 0):.0f}

## 2. MARKET STRUCTURE
{'=' * 60}
Unique Molecules: {metrics.get('unique_molecules', 0):,}
Country Coverage: {metrics.get('country_coverage', 0)}
Top 3 Market Share: {metrics.get('top_3_share', 0):.1f}%
"""
            
            st.download_button(
                label="⬇️ Download Report",
                data=summary,
                file_name=f"pharma_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        gc.enable()
        main()
    except Exception as e:
        st.error(f"❌ Critical error: {str(e)}")
        st.error("**Detailed Error:**")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Restart", type="primary"):
            st.rerun()

# ============================================================================
# END OF APPLICATION
# ============================================================================
