# ============================================================================
# PHARMAINTELLIGENCE PRO v7.0 - ENTERPRISE PHARMACEUTICAL ANALYTICS PLATFORM
# ============================================================================
# Version: 7.0 - Advanced Pharma Analytics
# Features: Excel Mapping, Price Analysis, ML Forecasting, Clustering, Anomaly Detection
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
from prophet import Prophet
import holidays

# Utilities
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple
import math
import re

# Country normalization
try:
    import pycountry
    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False
    st.warning("⚠️ pycountry kütüphanesi bulunamadı. Ülke normalizasyonu sınırlı olacak.")

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="PharmaIntelligence Pro v7.0 | Advanced Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling - Dark Theme Enhanced
DARK_THEME_CSS = """
<style>
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 100%);
        color: #e0e0e0;
    }
    
    /* Headers */
    .section-header {
        background: linear-gradient(135deg, #2d7dd2 0%, #6a11cb 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 16px rgba(106, 17, 203, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .section-header h2 {
        color: #ffffff;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* KPI Cards - Glass Morphism */
    .kpi-card {
        background: rgba(30, 30, 46, 0.7);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: 0.5s;
    }
    
    .kpi-card:hover::before {
        left: 100%;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(106, 17, 203, 0.4);
        border-color: rgba(106, 17, 203, 0.5);
    }
    
    .kpi-title {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .kpi-subtitle {
        color: #cbd5e1;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Insight Cards */
    .insight-card {
        background: rgba(45, 125, 210, 0.15);
        border-left: 4px solid #2d7dd2;
        padding: 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .insight-card:hover {
        transform: translateX(5px);
        background: rgba(45, 125, 210, 0.25);
    }
    
    .insight-card h4 {
        color: #2acaea;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .insight-card p {
        color: #e2e8f0;
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Filter Badge */
    .filter-badge {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        display: inline-block;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(106, 17, 203, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Buttons - Modern Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(106, 17, 203, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(106, 17, 203, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Success/Warning/Info boxes */
    .success-box {
        background: rgba(45, 212, 163, 0.15);
        border-left: 4px solid #2dd4a3;
        padding: 1.25rem;
        border-radius: 10px;
        color: #cbd5e1;
        border: 1px solid rgba(45, 212, 163, 0.2);
    }
    
    .warning-box {
        background: rgba(251, 191, 36, 0.15);
        border-left: 4px solid #fbbf24;
        padding: 1.25rem;
        border-radius: 10px;
        color: #cbd5e1;
        border: 1px solid rgba(251, 191, 36, 0.2);
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.15);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 10px;
        color: #cbd5e1;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 30, 46, 0.7);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(106, 17, 203, 0.2);
        border-color: rgba(106, 17, 203, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(106, 17, 203, 0.3);
    }
    
    /* Dataframe Styling */
    .dataframe {
        background-color: rgba(30, 30, 46, 0.7) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 46, 0.7);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2575fc 0%, #6a11cb 100%);
    }
    
    /* Streamlit specific overrides */
    .stMetric {
        background: transparent !important;
    }
    
    .stMetric label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #666;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: rgba(30, 30, 46, 0.95);
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

# ============================================================================
# ENHANCED DATA PROCESSING CLASS
# ============================================================================

class EnhancedDataManager:
    """Advanced data processing with Excel structure mapping"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_and_clean_data(file, sample_size=None):
        """Load and clean data with Excel column mapping"""
        try:
            start_time = time.time()
            
            # Load based on file type
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, nrows=sample_size, encoding='utf-8')
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, nrows=sample_size, engine='openpyxl')
            else:
                st.error("❌ Desteklenmeyen dosya formatı!")
                return None
            
            st.info(f"📥 Ham veri yüklendi: {len(df)} satır, {len(df.columns)} sütun")
            
            # Clean column names with advanced mapping
            df = EnhancedDataManager.clean_and_map_columns(df)
            
            # Create calculated columns
            df = EnhancedDataManager.create_calculated_columns(df)
            
            # Optimize memory
            df = EnhancedDataManager.optimize_dataframe(df)
            
            # Normalize country names
            df = EnhancedDataManager.normalize_country_names(df)
            
            load_time = time.time() - start_time
            
            # Show mapping summary
            st.success(f"""
            ✅ **Veri Başarıyla Yüklendi:**
            - **{len(df):,}** satır
            - **{len(df.columns):,}** sütun
            - **{load_time:.2f}s** yükleme süresi
            - **{df.memory_usage(deep=True).sum()/1024**2:.1f}MB** bellek kullanımı
            """)
            
            # Show column mapping summary
            with st.expander("📋 Kolon Dönüşüm Özeti", expanded=False):
                st.markdown("**Orjinal Kolonlar → Yeni Kolonlar:**")
                for orig_col, new_col in EnhancedDataManager.get_mapping_summary(df).items():
                    st.write(f"`{orig_col}` → `{new_col}`")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.error(traceback.format_exc())
            return None
    
    @staticmethod
    def clean_and_map_columns(df):
        """Clean and map Excel columns to standardized names"""
        column_mapping = {}
        
        # Text columns mapping
        text_mappings = {
            'Source.Name': 'Source',
            'Country': 'Country',
            'Sector': 'Sector',
            'Panel': 'Panel',
            'Region': 'Region',
            'Sub-Region': 'Sub_Region',
            'Corporation': 'Corporation',
            'Manufacturer': 'Manufacturer',
            'Molecule List': 'Molecule_List',
            'Molecule': 'Molecule',
            'Chemical Salt': 'Chemical_Salt',
            'International Product': 'International_Product',
            'Specialty Product': 'Specialty_Product',
            'NFC123': 'NFC123',
            'International Pack': 'International_Pack',
            'International Strength': 'International_Strength',
            'International Size': 'International_Size',
            'International Volume': 'International_Volume',
            'International Prescription': 'International_Prescription'
        }
        
        # Process each column
        for col in df.columns:
            col_str = str(col).strip()
            
            # Remove Turkish characters and normalize
            replacements = {
                'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c',
                '\n': ' ', '\r': ' ', '\t': ' '
            }
            
            for tr, en in replacements.items():
                col_str = col_str.replace(tr, en)
            
            # Clean whitespace
            col_str = ' '.join(col_str.split())
            
            # Check for MAT patterns (Sales, Volume, Price)
            mat_patterns = [
                (r'MAT Q\d+ 2022.*USD.*MNF', 'Sales_2022'),
                (r'MAT Q\d+ 2022.*Units', 'Volume_2022'),
                (r'MAT Q\d+ 2022.*Unit Avg Price USD.*MNF', 'Price_2022'),
                (r'MAT Q\d+ 2023.*USD.*MNF', 'Sales_2023'),
                (r'MAT Q\d+ 2023.*Units', 'Volume_2023'),
                (r'MAT Q\d+ 2023.*Unit Avg Price USD.*MNF', 'Price_2023'),
                (r'MAT Q\d+ 2024.*USD.*MNF', 'Sales_2024'),
                (r'MAT Q\d+ 2024.*Units', 'Volume_2024'),
                (r'MAT Q\d+ 2024.*Unit Avg Price USD.*MNF', 'Price_2024')
            ]
            
            mapped = False
            for pattern, new_name in mat_patterns:
                if re.search(pattern, col_str, re.IGNORECASE):
                    column_mapping[col] = new_name
                    mapped = True
                    break
            
            # Map text columns
            if not mapped:
                for orig, new in text_mappings.items():
                    if orig.lower() in col_str.lower():
                        column_mapping[col] = new
                        mapped = True
                        break
            
            # If not mapped, use cleaned name
            if not mapped:
                # Create safe column name
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', col_str)
                safe_name = re.sub(r'_+', '_', safe_name).strip('_')
                if safe_name:
                    column_mapping[col] = safe_name
                else:
                    column_mapping[col] = f'Column_{list(df.columns).index(col)}'
        
        # Apply mapping
        df = df.rename(columns=column_mapping)
        
        # Ensure all required columns exist
        required_columns = [
            'Sales_2022', 'Volume_2022', 'Price_2022',
            'Sales_2023', 'Volume_2023', 'Price_2023', 
            'Sales_2024', 'Volume_2024', 'Price_2024',
            'Country', 'Region', 'Corporation', 'Molecule'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.warning(f"⚠️ Bazı gerekli kolonlar eksik: {missing_cols}")
        
        return df
    
    @staticmethod
    def get_mapping_summary(df):
        """Get column mapping summary for display"""
        # This would track original to new mapping
        # For now, return current columns
        return {col: col for col in df.columns}
    
    @staticmethod
    def create_calculated_columns(df):
        """Create calculated columns for analysis"""
        
        # Calculate missing price columns
        for year in ['2022', '2023', '2024']:
            sales_col = f'Sales_{year}'
            volume_col = f'Volume_{year}'
            price_col = f'Price_{year}'
            
            if sales_col in df.columns and volume_col in df.columns:
                if price_col not in df.columns:
                    df[price_col] = np.nan
                
                # Calculate price where missing and volume > 0
                mask = (df[price_col].isna()) & (df[volume_col] > 0)
                df.loc[mask, price_col] = df.loc[mask, sales_col] / df.loc[mask, volume_col]
                
                # Handle infinite values
                df[price_col] = df[price_col].replace([np.inf, -np.inf], np.nan)
        
        # Calculate growth rates
        for i in range(2, 5):  # 2022-2023, 2023-2024
            prev_year = str(2020 + i)
            curr_year = str(2021 + i)
            
            prev_sales = f'Sales_{prev_year}'
            curr_sales = f'Sales_{curr_year}'
            
            if prev_sales in df.columns and curr_sales in df.columns:
                growth_col = f'Growth_{prev_year}_{curr_year}'
                df[growth_col] = ((df[curr_sales] - df[prev_sales]) / 
                                 df[prev_sales].replace(0, np.nan)) * 100
        
        # Calculate CAGR if we have multiple years
        sales_years = [col for col in df.columns if col.startswith('Sales_')]
        if len(sales_years) >= 2:
            first_year = min(sales_years, key=lambda x: int(x.split('_')[1]))
            last_year = max(sales_years, key=lambda x: int(x.split('_')[1]))
            
            periods = int(last_year.split('_')[1]) - int(first_year.split('_')[1])
            if periods > 0:
                df['CAGR'] = ((df[last_year] / df[first_year].replace(0, np.nan)) ** (1/periods) - 1) * 100
        
        # Calculate market share for latest year
        latest_sales = max([col for col in df.columns if col.startswith('Sales_')], 
                          key=lambda x: int(x.split('_')[1]), default=None)
        if latest_sales and df[latest_sales].sum() > 0:
            df['Market_Share'] = (df[latest_sales] / df[latest_sales].sum()) * 100
        
        # Price segments
        if 'Price_2024' in df.columns:
            price_median = df['Price_2024'].median()
            price_q1 = df['Price_2024'].quantile(0.25)
            price_q3 = df['Price_2024'].quantile(0.75)
            
            df['Price_Segment'] = pd.cut(
                df['Price_2024'],
                bins=[-np.inf, price_q1, price_median, price_q3, np.inf],
                labels=['Economy', 'Mainstream', 'Premium', 'Luxury']
            )
        
        return df
    
    @staticmethod
    def optimize_dataframe(df):
        """Optimize dataframe memory usage"""
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Optimize categorical columns
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
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
                        if col_min >= -128 and col_max <= 127:
                            df[col] = df[col].astype(np.int8)
                        elif col_min >= -32768 and col_max <= 32767:
                            df[col] = df[col].astype(np.int16)
                        elif col_min >= -2147483648 and col_max <= 2147483647:
                            df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.float32)
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            savings = original_memory - optimized_memory
            
            if savings > 0:
                st.info(f"💾 Bellek optimizasyonu: {original_memory:.1f}MB → {optimized_memory:.1f}MB (↓%{savings/original_memory*100:.1f})")
            
            return df
            
        except Exception as e:
            st.warning(f"Optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def normalize_country_names(df):
        """Normalize country names for choropleth maps"""
        country_col = None
        for possible in ['Country', 'country', 'Ülke', 'Ulke']:
            if possible in df.columns:
                country_col = possible
                break
        
        if not country_col:
            return df
        
        # Manual mapping for common variations
        country_mapping = {
            'USA': 'United States',
            'US': 'United States',
            'U.S.A': 'United States',
            'United States of America': 'United States',
            'UK': 'United Kingdom',
            'U.K': 'United Kingdom',
            'UAE': 'United Arab Emirates',
            'U.A.E': 'United Arab Emirates',
            'S. Korea': 'South Korea',
            'N. Korea': 'North Korea',
            'Russia': 'Russian Federation',
            'Iran': 'Iran, Islamic Republic of',
            'Vietnam': 'Viet Nam',
            'Syria': 'Syrian Arab Republic',
            'Laos': 'Lao People\'s Democratic Republic',
            'Bolivia': 'Bolivia, Plurinational State of',
            'Venezuela': 'Venezuela, Bolivarian Republic of',
            'Tanzania': 'Tanzania, United Republic of',
            'Moldova': 'Moldova, Republic of',
            'Macedonia': 'North Macedonia',
            'Congo': 'Congo, Republic of the',
            'DR Congo': 'Congo, Democratic Republic of the',
            'Ivory Coast': 'Côte d\'Ivoire',
            'Czech Republic': 'Czechia',
            'Turkey': 'Türkiye',
            'Turkiye': 'Türkiye'
        }
        
        df[country_col] = df[country_col].replace(country_mapping)
        
        return df

# ============================================================================
# ENHANCED ANALYTICS ENGINE
# ============================================================================

class EnhancedAnalytics:
    """Enhanced analytics with strategic insights"""
    
    @staticmethod
    def calculate_comprehensive_metrics(df):
        """Calculate comprehensive market metrics"""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['total_rows'] = len(df)
            metrics['total_columns'] = len(df.columns)
            
            # Latest year metrics
            latest_year = '2024'  # Assuming 2024 is latest
            sales_col = f'Sales_{latest_year}'
            volume_col = f'Volume_{latest_year}'
            price_col = f'Price_{latest_year}'
            
            if sales_col in df.columns:
                metrics['total_market_value'] = df[sales_col].sum()
                metrics['avg_sales'] = df[sales_col].mean()
                metrics['median_sales'] = df[sales_col].median()
                metrics['sales_q1'] = df[sales_col].quantile(0.25)
                metrics['sales_q3'] = df[sales_col].quantile(0.75)
                metrics['sales_iqr'] = metrics['sales_q3'] - metrics['sales_q1']
                metrics['sales_std'] = df[sales_col].std()
            
            if volume_col in df.columns:
                metrics['total_volume'] = df[volume_col].sum()
                metrics['avg_volume'] = df[volume_col].mean()
            
            if price_col in df.columns:
                metrics['avg_price'] = df[price_col].mean()
                metrics['median_price'] = df[price_col].median()
                metrics['price_std'] = df[price_col].std()
                metrics['price_cv'] = (metrics['price_std'] / metrics['avg_price'] * 100) if metrics['avg_price'] > 0 else 0
            
            # Growth metrics
            growth_col = 'Growth_2023_2024' if 'Growth_2023_2024' in df.columns else None
            if growth_col:
                metrics['avg_growth'] = df[growth_col].mean()
                metrics['positive_growth_count'] = (df[growth_col] > 0).sum()
                metrics['negative_growth_count'] = (df[growth_col] < 0).sum()
                metrics['high_growth_count'] = (df[growth_col] > 20).sum()
                metrics['declining_count'] = (df[growth_col] < -10).sum()
            
            # Corporation metrics
            if 'Corporation' in df.columns and sales_col in df.columns:
                corp_metrics = df.groupby('Corporation')[sales_col].agg(['sum', 'count', 'mean']).reset_index()
                corp_metrics = corp_metrics.sort_values('sum', ascending=False)
                
                metrics['unique_corporations'] = corp_metrics.shape[0]
                metrics['top_corp'] = corp_metrics.iloc[0]['Corporation']
                metrics['top_corp_share'] = (corp_metrics.iloc[0]['sum'] / metrics['total_market_value']) * 100 if metrics['total_market_value'] > 0 else 0
                
                # HHI Index
                market_shares = (corp_metrics['sum'] / metrics['total_market_value']) * 100
                metrics['hhi_index'] = (market_shares ** 2).sum()
                
                # Concentration ratios
                for n in [1, 3, 5, 10]:
                    top_n_share = corp_metrics.head(n)['sum'].sum() / metrics['total_market_value'] * 100
                    metrics[f'top_{n}_corp_share'] = top_n_share
            
            # Country metrics
            if 'Country' in df.columns and sales_col in df.columns:
                country_metrics = df.groupby('Country')[sales_col].agg(['sum', 'count']).reset_index()
                country_metrics = country_metrics.sort_values('sum', ascending=False)
                
                metrics['unique_countries'] = country_metrics.shape[0]
                metrics['top_country'] = country_metrics.iloc[0]['Country']
                metrics['top_country_share'] = (country_metrics.iloc[0]['sum'] / metrics['total_market_value']) * 100 if metrics['total_market_value'] > 0 else 0
                
                # Regional analysis
                if 'Region' in df.columns:
                    region_metrics = df.groupby('Region')[sales_col].sum().reset_index()
                    metrics['top_region'] = region_metrics.loc[region_metrics[sales_col].idxmax(), 'Region']
            
            # Molecule metrics
            if 'Molecule' in df.columns:
                metrics['unique_molecules'] = df['Molecule'].nunique()
                
                if sales_col in df.columns:
                    mol_metrics = df.groupby('Molecule')[sales_col].sum().reset_index()
                    mol_metrics = mol_metrics.sort_values(sales_col, ascending=False)
                    
                    metrics['top_molecule'] = mol_metrics.iloc[0]['Molecule']
                    metrics['top_molecule_share'] = (mol_metrics.iloc[0][sales_col] / metrics['total_market_value']) * 100 if metrics['total_market_value'] > 0 else 0
            
            # Price segment metrics
            if 'Price_Segment' in df.columns and sales_col in df.columns:
                segment_metrics = df.groupby('Price_Segment')[sales_col].sum().reset_index()
                segment_metrics['share'] = (segment_metrics[sales_col] / metrics['total_market_value']) * 100
                metrics['price_segments'] = segment_metrics.to_dict('records')
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return metrics
    
    @staticmethod
    def generate_strategic_insights(df, metrics):
        """Generate strategic insights from data"""
        insights = []
        
        try:
            latest_year = '2024'
            sales_col = f'Sales_{latest_year}'
            growth_col = 'Growth_2023_2024'
            
            # Insight 1: Market size and growth
            if 'total_market_value' in metrics:
                market_size = metrics['total_market_value'] / 1e9  # Convert to billions
                insights.append({
                    'type': 'success',
                    'title': f'💰 {latest_year} Pazar Büyüklüğü',
                    'description': f"Toplam pazar değeri **${market_size:.2f}B**. "
                })
            
            # Insight 2: Growth rate
            if 'avg_growth' in metrics:
                growth_rate = metrics['avg_growth']
                if growth_rate > 10:
                    growth_desc = "💹 **Yüksek Büyüme** - Pazar dinamik ve genişliyor"
                elif growth_rate > 0:
                    growth_desc = "📈 **Orta Büyüme** - Sağlıklı gelişim"
                else:
                    growth_desc = "⚠️ **Düşüş** - Pazar daralıyor"
                
                insights.append({
                    'type': 'info',
                    'title': '📊 Büyüme Trendi',
                    'description': f"Ortalama yıllık büyüme: **%{growth_rate:.1f}**. {growth_desc}"
                })
            
            # Insight 3: Market concentration
            if 'hhi_index' in metrics:
                hhi = metrics['hhi_index']
                if hhi > 2500:
                    concentration = "🔴 **Monopolistik** - Yüksek konsantrasyon riski"
                    rec = "Rekabet artırıcı tedbirler gerekli"
                elif hhi > 1500:
                    concentration = "🟡 **Oligopol** - Orta düzey konsantrasyon"
                    rec = "Pazar girişleri için fırsatlar var"
                else:
                    concentration = "🟢 **Rekabetçi** - Sağlıklı dağılım"
                    rec = "Çeşitlilik ve yenilik için uygun ortam"
                
                insights.append({
                    'type': 'warning' if hhi > 2500 else 'info',
                    'title': '🏢 Pazar Yapısı',
                    'description': f"HHI İndeksi: **{hhi:.0f}**. {concentration}. {rec}"
                })
            
            # Insight 4: Top performer
            if 'top_corp' in metrics and 'top_corp_share' in metrics:
                top_corp = metrics['top_corp']
                top_share = metrics['top_corp_share']
                
                if top_share > 30:
                    dominance = "⏫ **Baskın Oyuncu** - Pazar lideri güçlü konumda"
                elif top_share > 20:
                    dominance = "⬆️ **Güçlü Lider** - Önemli pazar payı"
                else:
                    dominance = "↔️ **Dağılmış Liderlik** - Rekabet yoğun"
                
                insights.append({
                    'type': 'success',
                    'title': '🏆 Pazar Lideri',
                    'description': f"**{top_corp}** %{top_share:.1f} pazar payı ile lider. {dominance}"
                })
            
            # Insight 5: Geographic concentration
            if 'top_country' in metrics and 'top_country_share' in metrics:
                top_country = metrics['top_country']
                country_share = metrics['top_country_share']
                
                if country_share > 40:
                    geo_risk = "⚠️ **Yüksek Coğrafi Risk** - Aşırı bağımlılık"
                elif country_share > 25:
                    geo_risk = "🔸 **Orta Risk** - Çeşitlendirme gerekli"
                else:
                    geo_risk = "✅ **Dağılmış Risk** - Sağlıklı dağılım"
                
                insights.append({
                    'type': 'geographic',
                    'title': '🌍 Coğrafi Yoğunlaşma',
                    'description': f"**{top_country}** %{country_share:.1f} pay ile en büyük pazar. {geo_risk}"
                })
            
            # Insight 6: Product diversity
            if 'unique_molecules' in metrics:
                mol_count = metrics['unique_molecules']
                if mol_count > 100:
                    diversity = "🌿 **Yüksek Çeşitlilik** - Geniş ürün portföyü"
                elif mol_count > 50:
                    diversity = "🍃 **Orta Çeşitlilik** - Dengeli portföy"
                else:
                    diversity = "🍂 **Sınırlı Çeşitlilik** - Odaklanmış portföy"
                
                insights.append({
                    'type': 'info',
                    'title': '🧪 Ürün Çeşitliliği',
                    'description': f"**{mol_count}** farklı molekül. {diversity}"
                })
            
            # Insight 7: Price analysis
            if 'price_segments' in metrics:
                segments = metrics['price_segments']
                premium_segment = next((s for s in segments if s['Price_Segment'] == 'Premium'), None)
                economy_segment = next((s for s in segments if s['Price_Segment'] == 'Economy'), None)
                
                if premium_segment and economy_segment:
                    premium_share = premium_segment['share']
                    economy_share = economy_segment['share']
                    
                    if premium_share > economy_share:
                        price_insight = "💎 **Premium Odaklı** - Yüksek değer segmenti baskın"
                    else:
                        price_insight = "💰 **Hacim Odaklı** - Ekonomi segmenti baskın"
                    
                    insights.append({
                        'type': 'price',
                        'title': '🏷️ Fiyat Segmentasyonu',
                        'description': f"Premium: %{premium_share:.1f}, Economy: %{economy_share:.1f}. {price_insight}"
                    })
            
            return insights
            
        except Exception as e:
            st.warning(f"İçgörü oluşturma hatası: {str(e)}")
            return insights

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class EnhancedVisualizer:
    """Professional visualization engine with pharma focus"""
    
    @staticmethod
    def create_dashboard_kpis(df, metrics):
        """Create enhanced KPI cards for dashboard"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = metrics.get('total_market_value', 0)
                year = '2024'
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">TOPLAM PAZAR DEĞERİ</div>
                    <div class="kpi-value">${total_value/1e9:.2f}B</div>
                    <div class="kpi-subtitle">{year} Global İlaç Pazarı</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_growth = metrics.get('avg_growth', 0)
                growth_icon = "📈" if avg_growth > 0 else "📉" if avg_growth < 0 else "➡️"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">BÜYÜME ORANI</div>
                    <div class="kpi-value">{growth_icon} {avg_growth:.1f}%</div>
                    <div class="kpi-subtitle">2023 → 2024 YoY Değişim</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                top_corp = metrics.get('top_corp', 'N/A')
                top_share = metrics.get('top_corp_share', 0)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">PAZAR LİDERİ</div>
                    <div class="kpi-value">{top_corp[:15]}{'...' if len(top_corp) > 15 else ''}</div>
                    <div class="kpi-subtitle">%{top_share:.1f} Pazar Payı</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                top_country = metrics.get('top_country', 'N/A')
                country_share = metrics.get('top_country_share', 0)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">EN BÜYÜK PAZAR</div>
                    <div class="kpi-value">{top_country[:15]}{'...' if len(top_country) > 15 else ''}</div>
                    <div class="kpi-subtitle">%{country_share:.1f} Coğrafi Pay</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional KPIs in second row
            st.markdown("<br>", unsafe_allow_html=True)
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                hhi = metrics.get('hhi_index', 0)
                hhi_status = "Monopol" if hhi > 2500 else "Oligopol" if hhi > 1500 else "Rekabetçi"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">HHI İNDEKSİ</div>
                    <div class="kpi-value">{hhi:.0f}</div>
                    <div class="kpi-subtitle">{hhi_status} Pazar</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                unique_mols = metrics.get('unique_molecules', 0)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">ÜRÜN ÇEŞİTLİLİĞİ</div>
                    <div class="kpi-value">{unique_mols}</div>
                    <div class="kpi-subtitle">Farklı Molekül</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                avg_price = metrics.get('avg_price', 0)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">ORTALAMA FİYAT</div>
                    <div class="kpi-value">${avg_price:.2f}</div>
                    <div class="kpi-subtitle">2024 Birim Fiyat</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                positive_growth = metrics.get('positive_growth_count', 0)
                total_rows = metrics.get('total_rows', 1)
                growth_percentage = (positive_growth / total_rows) * 100
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">POZİTİF BÜYÜME</div>
                    <div class="kpi-value">%{growth_percentage:.1f}</div>
                    <div class="kpi-subtitle">{positive_growth}/{total_rows} Ürün</div>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"KPI kartları oluşturma hatası: {str(e)}")
    
    @staticmethod
    def create_sales_trend_chart(df):
        """Create sales trend visualization for 2022-2024"""
        try:
            # Collect yearly data
            yearly_data = []
            for year in ['2022', '2023', '2024']:
                sales_col = f'Sales_{year}'
                volume_col = f'Volume_{year}'
                price_col = f'Price_{year}'
                
                if sales_col in df.columns:
                    yearly_data.append({
                        'Yıl': year,
                        'Toplam Satış': df[sales_col].sum(),
                        'Ortalama Satış': df[sales_col].mean(),
                        'Ürün Sayısı': (df[sales_col] > 0).sum()
                    })
                    
                if volume_col in df.columns:
                    if 'Hacim' not in yearly_data[-1] if yearly_data else True:
                        continue
                    yearly_data[-1]['Toplam Hacim'] = df[volume_col].sum()
                
                if price_col in df.columns:
                    if 'Ortalama Fiyat' not in yearly_data[-1] if yearly_data else True:
                        continue
                    yearly_data[-1]['Ortalama Fiyat'] = df[price_col].mean()
            
            if len(yearly_data) < 2:
                st.info("📊 Trend analizi için en az 2 yıllık veri gerekli")
                return None
            
            yearly_df = pd.DataFrame(yearly_data)
            
            # Create dual-axis chart
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]]
            )
            
            # Total sales bar (primary y-axis)
            fig.add_trace(
                go.Bar(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Toplam Satış'],
                    name='Toplam Satış',
                    marker_color='#2d7dd2',
                    text=[f'${x/1e6:.0f}M' for x in yearly_df['Toplam Satış']],
                    textposition='outside',
                    opacity=0.9
                ),
                secondary_y=False
            )
            
            # Average price line (secondary y-axis)
            if 'Ortalama Fiyat' in yearly_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=yearly_df['Yıl'],
                        y=yearly_df['Ortalama Fiyat'],
                        name='Ortalama Fiyat',
                        mode='lines+markers',
                        line=dict(color='#ff6b6b', width=3),
                        marker=dict(size=12, symbol='diamond'),
                        yaxis='y2'
                    ),
                    secondary_y=True
                )
            
            fig.update_layout(
                title='📈 Satış Trendleri & Fiyat Gelişimi (2022-2024)',
                xaxis_title='Yıl',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                showlegend=True,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_yaxes(
                title_text="Toplam Satış (USD)", 
                secondary_y=False,
                gridcolor='rgba(255,255,255,0.1)'
            )
            fig.update_yaxes(
                title_text="Ortalama Fiyat (USD)", 
                secondary_y=True,
                gridcolor='rgba(255,255,255,0.1)'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Trend grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_market_share_chart(df):
        """Create market share visualization for corporations"""
        try:
            sales_col = 'Sales_2024' if 'Sales_2024' in df.columns else \
                       'Sales_2023' if 'Sales_2023' in df.columns else None
            
            if not sales_col or 'Corporation' not in df.columns:
                return None
            
            # Aggregate corporation sales
            corp_sales = df.groupby('Corporation')[sales_col].sum().sort_values(ascending=False)
            
            if len(corp_sales) == 0:
                return None
            
            # Take top 15 and group others
            top_n = 15
            top_corps = corp_sales.head(top_n)
            other_sales = corp_sales.iloc[top_n:].sum()
            
            # Prepare data for pie chart
            pie_labels = list(top_corps.index) + ['Diğer']
            pie_values = list(top_corps.values) + [other_sales]
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Pazar Payı Dağılımı', 'Top 10 Şirket Satışları'),
                specs=[[{'type': 'domain'}, {'type': 'bar'}]],
                column_widths=[0.4, 0.6]
            )
            
            # Pie chart with custom colors
            colors = px.colors.qualitative.Bold + px.colors.qualitative.Vivid
            
            fig.add_trace(
                go.Pie(
                    labels=pie_labels,
                    values=pie_values,
                    hole=0.5,
                    marker_colors=colors,
                    textinfo='label+percent',
                    textposition='outside',
                    hoverinfo='label+value+percent',
                    name='Pazar Payı'
                ),
                row=1, col=1
            )
            
            # Horizontal bar chart for top 10
            top_10 = corp_sales.head(10)
            fig.add_trace(
                go.Bar(
                    y=top_10.index,
                    x=top_10.values,
                    orientation='h',
                    marker_color='#6a11cb',
                    text=[f'${x/1e6:.1f}M' for x in top_10.values],
                    textposition='auto',
                    name='Satışlar'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                showlegend=False,
                title_text="🏢 Şirket Bazlı Pazar Konsantrasyonu"
            )
            
            fig.update_yaxes(autorange="reversed", row=1, col=2)
            
            return fig
            
        except Exception as e:
            st.warning(f"Pazar payı grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_sunburst_chart(df):
        """Create hierarchical sunburst chart"""
        try:
            sales_col = 'Sales_2024' if 'Sales_2024' in df.columns else None
            if not sales_col:
                return None
            
            # Required columns for hierarchy
            hierarchy_cols = []
            for col in ['Region', 'Country', 'Corporation', 'Molecule']:
                if col in df.columns:
                    hierarchy_cols.append(col)
            
            if len(hierarchy_cols) < 2:
                return None
            
            # Aggregate data for hierarchy
            agg_df = df.groupby(hierarchy_cols)[sales_col].sum().reset_index()
            agg_df = agg_df[agg_df[sales_col] > 0]  # Remove zero sales
            
            if len(agg_df) == 0:
                return None
            
            # Create sunburst chart
            fig = px.sunburst(
                agg_df,
                path=hierarchy_cols,
                values=sales_col,
                color=sales_col,
                color_continuous_scale='Viridis',
                title='🌐 Hiyerarşik Pazar Yapısı',
                hover_data={sales_col: ':.2f'}
            )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                margin=dict(t=50, l=0, r=0, b=0)
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Sunburst grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_price_volume_chart(df, sample_size=10000):
        """Create price-volume scatter plot with segmentation"""
        try:
            # Check required columns
            price_col = 'Price_2024'
            volume_col = 'Volume_2024'
            sales_col = 'Sales_2024'
            
            if not all(col in df.columns for col in [price_col, volume_col, sales_col]):
                st.warning("⚠️ Fiyat-hacim analizi için gerekli sütunlar bulunamadı")
                return None
            
            # Filter valid data
            valid_data = df[
                (df[price_col] > 0) & 
                (df[volume_col] > 0) & 
                (df[sales_col] > 0)
            ].copy()
            
            if len(valid_data) == 0:
                st.info("📊 Geçerli fiyat ve hacim verisi bulunamadı")
                return None
            
            # Sample for performance
            if len(valid_data) > sample_size:
                valid_data = valid_data.sample(sample_size, random_state=42)
                st.info(f"ℹ️ Performans için {sample_size:,} satır örneklendi")
            
            # Determine color column
            color_col = None
            for col in ['Region', 'Country', 'Price_Segment']:
                if col in valid_data.columns:
                    color_col = col
                    break
            
            # Create scatter plot
            fig = px.scatter(
                valid_data,
                x=price_col,
                y=volume_col,
                size=sales_col,
                color=color_col if color_col else sales_col,
                hover_name='Molecule' if 'Molecule' in valid_data.columns else None,
                hover_data={
                    'Corporation': True,
                    'Country': True,
                    'Sales_2024': ':.2f',
                    'Price_2024': ':.2f',
                    'Volume_2024': ':,.0f'
                },
                title='💰 Fiyat-Hacim İlişkisi Analizi',
                labels={
                    price_col: 'Fiyat (USD)',
                    volume_col: 'Hacim (Birim)',
                    sales_col: 'Satış (USD)'
                },
                color_continuous_scale='Viridis' if not color_col else None,
                size_max=50
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Fiyat-hacim grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_price_segmentation_chart(df):
        """Create price segmentation pie chart"""
        try:
            if 'Price_Segment' not in df.columns or 'Sales_2024' not in df.columns:
                return None
            
            # Aggregate by price segment
            segment_data = df.groupby('Price_Segment')['Sales_2024'].agg(['sum', 'count']).reset_index()
            segment_data['share'] = (segment_data['sum'] / segment_data['sum'].sum()) * 100
            
            # Sort by share
            segment_data = segment_data.sort_values('share', ascending=False)
            
            # Create pie chart
            fig = px.pie(
                segment_data,
                values='share',
                names='Price_Segment',
                title='🏷️ Fiyat Segmentasyonu - Pazar Payları',
                hover_data=['sum', 'count'],
                labels={'share': 'Pazar Payı (%)', 'sum': 'Toplam Satış'},
                color='Price_Segment',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>Pazar Payı: %{percent}<br>Toplam Satış: $%{value:,.0f}<extra></extra>"
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Fiyat segmentasyonu grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_choropleth_map(df):
        """Create world choropleth map for sales distribution"""
        try:
            sales_col = 'Sales_2024' if 'Sales_2024' in df.columns else None
            country_col = 'Country'
            
            if not sales_col or country_col not in df.columns:
                return None
            
            # Aggregate by country
            country_sales = df.groupby(country_col)[sales_col].sum().reset_index()
            country_sales.columns = ['Country', 'Total_Sales']
            
            # Remove NaN countries
            country_sales = country_sales.dropna(subset=['Country'])
            
            if len(country_sales) == 0:
                return None
            
            # Create choropleth map
            fig = px.choropleth(
                country_sales,
                locations='Country',
                locationmode='country names',
                color='Total_Sales',
                hover_name='Country',
                hover_data={'Total_Sales': ':,.0f'},
                color_continuous_scale='Viridis',
                title='🌍 Global İlaç Pazarı Dağılımı',
                projection='natural earth'
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                geo=dict(
                    bgcolor='rgba(0,0,0,0)',
                    lakecolor='#1a1a2e',
                    landcolor='#2d4a7a',
                    subunitcolor='#64748b',
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor='#64748b'
                ),
                coloraxis_colorbar=dict(
                    title="Satış (USD)",
                    thickness=20,
                    len=0.75
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Harita oluşturma hatası: {str(e)}")
            return None

# ============================================================================
# ADVANCED ML ENGINE
# ============================================================================

class AdvancedMLEngine:
    """Advanced Machine Learning models for pharmaceutical analytics"""
    
    @staticmethod
    def forecast_sales_prophet(df, entity_type, entity_name, forecast_years=2):
        """Forecast sales using Prophet"""
        try:
            # Prepare time series data
            years = ['2022', '2023', '2024']
            sales_cols = [f'Sales_{year}' for year in years if f'Sales_{year}' in df.columns]
            
            if len(sales_cols) < 2:
                st.warning("⚠️ Tahmin için en az 2 yıllık veri gerekli")
                return None, None
            
            # Filter data based on entity
            if entity_type == 'Molecule' and 'Molecule' in df.columns:
                entity_data = df[df['Molecule'] == entity_name]
            elif entity_type == 'Country' and 'Country' in df.columns:
                entity_data = df[df['Country'] == entity_name]
            else:
                st.warning("⚠️ Seçilen entity tipi veya ismi geçerli değil")
                return None, None
            
            if len(entity_data) == 0:
                st.warning(f"⚠️ {entity_name} için veri bulunamadı")
                return None, None
            
            # Aggregate yearly sales
            yearly_sales = []
            for year, col in zip(years, sales_cols):
                if col in entity_data.columns:
                    total_sales = entity_data[col].sum()
                    yearly_sales.append({
                        'ds': pd.to_datetime(f'{year}-12-31'),
                        'y': total_sales
                    })
            
            if len(yearly_sales) < 2:
                st.warning("⚠️ Yeterli tarihsel veri yok")
                return None, None
            
            # Create DataFrame for Prophet
            prophet_df = pd.DataFrame(yearly_sales)
            
            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_years, freq='Y')
            
            # Make predictions
            forecast = model.predict(future)
            
            # Prepare results
            results = {
                'model': model,
                'forecast': forecast,
                'historical': prophet_df,
                'entity_type': entity_type,
                'entity_name': entity_name
            }
            
            return results, None
            
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def create_clustering_analysis(df, n_clusters=4):
        """Perform K-Means clustering analysis"""
        try:
            # Select features for clustering
            feature_cols = []
            
            if 'Price_2024' in df.columns:
                feature_cols.append('Price_2024')
            if 'Volume_2024' in df.columns:
                feature_cols.append('Volume_2024')
            if 'Growth_2023_2024' in df.columns:
                feature_cols.append('Growth_2023_2024')
            if 'Market_Share' in df.columns:
                feature_cols.append('Market_Share')
            
            if len(feature_cols) < 2:
                st.warning("⚠️ Kümeleme için yeterli özellik yok")
                return None, None
            
            # Prepare data
            cluster_data = df[feature_cols].fillna(0)
            
            if len(cluster_data) < n_clusters * 10:
                st.warning(f"⚠️ Kümeleme için yeterli veri yok (minimum {n_clusters * 10} satır)")
                return None, None
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Perform K-Means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=20,
                max_iter=300,
                tol=1e-4
            )
            
            clusters = kmeans.fit_predict(scaled_data)
            
            # Calculate metrics
            silhouette = silhouette_score(scaled_data, clusters)
            
            # PCA for 3D visualization
            pca = PCA(n_components=3)
            pca_data = pca.fit_transform(scaled_data)
            
            # Assign cluster names based on characteristics
            cluster_names = {
                0: 'Nakit İnekleri (Cash Cows)',
                1: 'Yıldızlar (Stars)',
                2: 'Soru İşaretleri (Question Marks)',
                3: 'Köpekler (Dogs)',
                4: 'Gelişen Ürünler',
                5: 'Olgun Ürünler',
                6: 'Yenilikçi Ürünler',
                7: 'Riskli Ürünler'
            }
            
            # Assign names to clusters
            assigned_names = []
            for i in range(n_clusters):
                assigned_names.append(cluster_names.get(i, f'Küme {i+1}'))
            
            # Prepare results
            results = {
                'clusters': clusters,
                'cluster_names': [assigned_names[c] for c in clusters],
                'silhouette_score': silhouette,
                'pca_data': pca_data,
                'pca_variance': pca.explained_variance_ratio_,
                'cluster_centers': kmeans.cluster_centers_,
                'features': feature_cols,
                'kmeans_model': kmeans,
                'scaler': scaler
            }
            
            return results, None
            
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def detect_anomalies_isolation(df, contamination=0.1):
        """Detect anomalies using Isolation Forest"""
        try:
            # Select features for anomaly detection
            feature_cols = []
            
            if 'Price_2024' in df.columns:
                feature_cols.append('Price_2024')
            if 'Sales_2024' in df.columns:
                feature_cols.append('Sales_2024')
            if 'Growth_2023_2024' in df.columns:
                feature_cols.append('Growth_2023_2024')
            if 'Market_Share' in df.columns:
                feature_cols.append('Market_Share')
            
            if len(feature_cols) < 2:
                st.warning("⚠️ Anomali tespiti için yeterli özellik yok")
                return None, None
            
            # Prepare data
            anomaly_data = df[feature_cols].fillna(0)
            
            if len(anomaly_data) < 50:
                st.warning("⚠️ Anomali tespiti için yeterli veri yok (minimum 50 satır)")
                return None, None
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(anomaly_data)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                n_jobs=-1
            )
            
            predictions = iso_forest.fit_predict(scaled_data)
            anomaly_scores = iso_forest.score_samples(scaled_data)
            
            # Prepare results
            results = {
                'is_anomaly': predictions == -1,
                'anomaly_scores': anomaly_scores,
                'anomaly_count': (predictions == -1).sum(),
                'anomaly_percentage': (predictions == -1).sum() / len(predictions) * 100,
                'normal_count': (predictions == 1).sum(),
                'features': feature_cols,
                'model': iso_forest,
                'scaler': scaler
            }
            
            return results, None
            
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def perform_what_if_analysis(df, what_if_scenarios):
        """Perform what-if analysis based on scenarios"""
        try:
            results = []
            
            for scenario in what_if_scenarios:
                # Apply scenario to dataframe
                scenario_df = df.copy()
                
                # Apply price changes
                if 'price_change' in scenario:
                    price_col = 'Price_2024'
                    if price_col in scenario_df.columns:
                        scenario_df[f'{price_col}_scenario'] = scenario_df[price_col] * (1 + scenario['price_change'] / 100)
                        
                        # Recalculate sales if volume stays constant
                        if 'Volume_2024' in scenario_df.columns:
                            scenario_df['Sales_scenario'] = scenario_df[f'{price_col}_scenario'] * scenario_df['Volume_2024']
                
                # Apply volume changes
                if 'volume_change' in scenario:
                    volume_col = 'Volume_2024'
                    if volume_col in scenario_df.columns:
                        scenario_df[f'{volume_col}_scenario'] = scenario_df[volume_col] * (1 + scenario['volume_change'] / 100)
                        
                        # Recalculate sales if price stays constant
                        if 'Price_2024' in scenario_df.columns:
                            scenario_df['Sales_scenario'] = scenario_df['Price_2024'] * scenario_df[f'{volume_col}_scenario']
                
                # Calculate impact
                if 'Sales_scenario' in scenario_df.columns and 'Sales_2024' in scenario_df.columns:
                    total_current = df['Sales_2024'].sum()
                    total_scenario = scenario_df['Sales_scenario'].sum()
                    impact = ((total_scenario - total_current) / total_current) * 100
                    
                    results.append({
                        'scenario': scenario.get('name', 'Senaryo'),
                        'description': scenario.get('description', ''),
                        'current_sales': total_current,
                        'scenario_sales': total_scenario,
                        'impact_percentage': impact,
                        'impact_absolute': total_scenario - total_current
                    })
            
            return results, None
            
        except Exception as e:
            return None, str(e)

# ============================================================================
# FILTER SYSTEM
# ============================================================================

class EnhancedFilterSystem:
    """Enhanced filter system with advanced controls"""
    
    @staticmethod
    def create_filter_sidebar(df):
        """Create comprehensive filter sidebar"""
        with st.sidebar.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
            st.markdown('<div class="section-header"><h2>🔍 Filtreler</h2></div>', unsafe_allow_html=True)
            
            # Year selection
            st.markdown("### 📅 Yıl Seçimi")
            available_years = []
            for year in ['2022', '2023', '2024']:
                if f'Sales_{year}' in df.columns:
                    available_years.append(year)
            
            if available_years:
                selected_year = st.selectbox(
                    "Analiz Yılı",
                    available_years,
                    index=len(available_years)-1
                )
            else:
                selected_year = None
            
            # Global search
            search_term = st.text_input(
                "🔎 Genel Arama",
                placeholder="Molekül, şirket, ülke ara...",
                help="Tüm sütunlarda arama yapın"
            )
            
            filters = {}
            
            # Country filter
            if 'Country' in df.columns:
                countries = sorted(df['Country'].dropna().unique())
                selected_countries = EnhancedFilterSystem.searchable_multiselect(
                    "🌍 Ülkeler",
                    countries,
                    key="country_filter"
                )
                if selected_countries and "Tümü" not in selected_countries:
                    filters['Country'] = selected_countries
            
            # Corporation filter
            if 'Corporation' in df.columns:
                corporations = sorted(df['Corporation'].dropna().unique())
                selected_corps = EnhancedFilterSystem.searchable_multiselect(
                    "🏢 Şirketler",
                    corporations,
                    key="corp_filter"
                )
                if selected_corps and "Tümü" not in selected_corps:
                    filters['Corporation'] = selected_corps
            
            # Molecule filter
            if 'Molecule' in df.columns:
                molecules = sorted(df['Molecule'].dropna().unique())
                selected_mols = EnhancedFilterSystem.searchable_multiselect(
                    "🧪 Moleküller",
                    molecules,
                    key="mol_filter"
                )
                if selected_mols and "Tümü" not in selected_mols:
                    filters['Molecule'] = selected_mols
            
            # Region filter
            if 'Region' in df.columns:
                regions = sorted(df['Region'].dropna().unique())
                selected_regions = EnhancedFilterSystem.searchable_multiselect(
                    "🗺️ Bölgeler",
                    regions,
                    key="region_filter"
                )
                if selected_regions and "Tümü" not in selected_regions:
                    filters['Region'] = selected_regions
            
            st.markdown("---")
            st.markdown("### 📊 Sayısal Filtreler")
            
            # Sales filter
            if selected_year and f'Sales_{selected_year}' in df.columns:
                sales_col = f'Sales_{selected_year}'
                min_sales = float(df[sales_col].min())
                max_sales = float(df[sales_col].max())
                
                sales_range = st.slider(
                    "Satış Aralığı (USD)",
                    min_value=min_sales,
                    max_value=max_sales,
                    value=(min_sales, max_sales),
                    format="$%.0f"
                )
                filters['sales_range'] = (sales_range, sales_col)
            
            # Price filter
            if selected_year and f'Price_{selected_year}' in df.columns:
                price_col = f'Price_{selected_year}'
                min_price = float(df[price_col].min())
                max_price = float(df[price_col].max())
                
                price_range = st.slider(
                    "Fiyat Aralığı (USD)",
                    min_value=min_price,
                    max_value=max_price,
                    value=(min_price, max_price),
                    format="$%.2f"
                )
                filters['price_range'] = (price_range, price_col)
            
            # Growth filter
            if 'Growth_2023_2024' in df.columns:
                growth_col = 'Growth_2023_2024'
                min_growth = float(df[growth_col].min())
                max_growth = float(df[growth_col].max())
                
                growth_range = st.slider(
                    "Büyüme Oranı (%)",
                    min_value=min_growth,
                    max_value=max_growth,
                    value=(min_growth, max_growth),
                    format="%.1f%%"
                )
                filters['growth_range'] = (growth_range, growth_col)
            
            st.markdown("---")
            
            # Additional filters
            col1, col2 = st.columns(2)
            with col1:
                positive_growth_only = st.checkbox("📈 Pozitif Büyüme", value=False)
            with col2:
                high_market_share = st.checkbox("🏆 Yüksek Pazar Payı (>5%)", value=False)
            
            if positive_growth_only and 'Growth_2023_2024' in df.columns:
                filters['positive_growth'] = True
            
            if high_market_share and 'Market_Share' in df.columns:
                filters['high_market_share'] = True
            
            st.markdown("---")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                apply_filters = st.button("✅ Filtrele", use_container_width=True, type="primary")
            with col2:
                clear_filters = st.button("🗑️ Temizle", use_container_width=True)
        
        return selected_year, search_term, filters, apply_filters, clear_filters
    
    @staticmethod
    def searchable_multiselect(label, options, key, default_all=True):
        """Create searchable multiselect widget"""
        if not options:
            return []
        
        all_options = ["Tümü"] + list(options)
        
        search_query = st.text_input(
            f"🔍 {label} Ara",
            key=f"{key}_search",
            placeholder="Filtrele..."
        )
        
        if search_query:
            filtered_options = ["Tümü"] + [
                opt for opt in options 
                if search_query.lower() in str(opt).lower()
            ]
        else:
            filtered_options = all_options
        
        default = ["Tümü"] if default_all else []
        
        selected = st.multiselect(
            label,
            options=filtered_options,
            default=default,
            key=key
        )
        
        if "Tümü" in selected:
            if len(selected) > 1:
                selected = [opt for opt in selected if opt != "Tümü"]
            else:
                selected = list(options)
        
        if selected and "Tümü" not in selected:
            st.caption(f"✅ {len(selected)} / {len(options)} seçildi")
        
        return selected
    
    @staticmethod
    def apply_filters(df, search_term, filters):
        """Apply all filters to dataframe"""
        filtered_df = df.copy()
        
        # Global search
        if search_term:
            search_mask = pd.Series(False, index=filtered_df.index)
            for col in filtered_df.columns:
                try:
                    if filtered_df[col].dtype == 'object' or filtered_df[col].dtype.name == 'category':
                        search_mask = search_mask | filtered_df[col].astype(str).str.contains(
                            search_term, case=False, na=False
                        )
                except:
                    continue
            filtered_df = filtered_df[search_mask]
        
        # Column filters
        for col, values in filters.items():
            if col in ['sales_range', 'price_range', 'growth_range', 'positive_growth', 'high_market_share']:
                continue
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        # Sales range
        if 'sales_range' in filters:
            (min_val, max_val), col_name = filters['sales_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        # Price range
        if 'price_range' in filters:
            (min_val, max_val), col_name = filters['price_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        # Growth range
        if 'growth_range' in filters:
            (min_val, max_val), col_name = filters['growth_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        # Positive growth only
        if filters.get('positive_growth', False) and 'Growth_2023_2024' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Growth_2023_2024'] > 0]
        
        # High market share
        if filters.get('high_market_share', False) and 'Market_Share' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Market_Share'] > 5]
        
        return filtered_df

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown("""
    <div class="section-header" style="text-align: center;">
        <h1>💊 PHARMAINTELLIGENCE PRO v7.0</h1>
        <p style="margin: 0; color: #cbd5e1; font-size: 1.1rem;">Advanced Pharmaceutical Analytics with Machine Learning</p>
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
    if 'active_filters' not in st.session_state:
        st.session_state.active_filters = {}
    if 'selected_year' not in st.session_state:
        st.session_state.selected_year = '2024'
    
    # Sidebar - Data Upload
    with st.sidebar:
        st.markdown('<div class="section-header"><h2>📁 VERİ YÜKLEME</h2></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Excel/CSV Dosyası Yükleyin",
            type=['xlsx', 'xls', 'csv'],
            help="İlaç pazarı verilerinizi yükleyin (Excel veya CSV formatında)"
        )
        
        if uploaded_file:
            sample_option = st.checkbox("Örnek veri ile test et (ilk 10,000 satır)", value=True)
            sample_size = 10000 if sample_option else None
            
            if st.button("🚀 Veriyi Yükle & Analiz Et", type="primary", use_container_width=True):
                with st.spinner("Veri yükleniyor ve analiz ediliyor..."):
                    # Load and clean data
                    df = EnhancedDataManager.load_and_clean_data(uploaded_file, sample_size)
                    
                    if df is not None:
                        # Store in session state
                        st.session_state.data = df
                        st.session_state.filtered_data = df.copy()
                        
                        # Calculate metrics
                        st.session_state.metrics = EnhancedAnalytics.calculate_comprehensive_metrics(df)
                        st.session_state.insights = EnhancedAnalytics.generate_strategic_insights(df, st.session_state.metrics)
                        
                        st.success(f"✅ **{len(df):,}** satır başarıyla yüklendi!")
                        st.rerun()
    
    # Main content
    if st.session_state.data is None:
        show_welcome_screen()
        return
    
    df = st.session_state.data
    
    # Filters
    selected_year, search_term, filters, apply_filters, clear_filters = EnhancedFilterSystem.create_filter_sidebar(df)
    st.session_state.selected_year = selected_year
    
    if apply_filters:
        with st.spinner("Filtreler uygulanıyor..."):
            filtered_df = EnhancedFilterSystem.apply_filters(df, search_term, filters)
            st.session_state.filtered_data = filtered_df
            st.session_state.active_filters = filters
            st.session_state.metrics = EnhancedAnalytics.calculate_comprehensive_metrics(filtered_df)
            st.session_state.insights = EnhancedAnalytics.generate_strategic_insights(filtered_df, st.session_state.metrics)
            st.success(f"✅ **{len(filtered_df):,}** / {len(df):,} satır gösteriliyor")
            st.rerun()
    
    if clear_filters:
        st.session_state.filtered_data = df.copy()
        st.session_state.active_filters = {}
        st.session_state.metrics = EnhancedAnalytics.calculate_comprehensive_metrics(df)
        st.session_state.insights = EnhancedAnalytics.generate_strategic_insights(df, st.session_state.metrics)
        st.success("✅ Filtreler temizlendi")
        st.rerun()
    
    # Show filter status
    if st.session_state.active_filters:
        active_filter_count = len([k for k in st.session_state.active_filters.keys() 
                                 if k not in ['sales_range', 'price_range', 'growth_range', 'positive_growth', 'high_market_share']])
        
        st.markdown(f"""
        <div class="filter-badge">
            🎯 Aktif Filtreler: {active_filter_count} | 
            Gösterilen: {len(st.session_state.filtered_data):,} / {len(df):,} satır
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 GENEL BAKIŞ",
        "💰 FİYAT ANALİZİ",
        "🌍 COĞRAFİ ANALİZ",
        "🏆 REKABET ANALİZİ",
        "🤖 ML LABORATUVARI",
        "📑 RAPORLAMA"
    ])
    
    with tab1:
        show_overview_tab(st.session_state.filtered_data, st.session_state.metrics, st.session_state.insights)
    
    with tab2:
        show_price_analysis_tab(st.session_state.filtered_data)
    
    with tab3:
        show_geographic_tab(st.session_state.filtered_data)
    
    with tab4:
        show_competition_tab(st.session_state.filtered_data, st.session_state.metrics)
    
    with tab5:
        show_ml_lab_tab(st.session_state.filtered_data)
    
    with tab6:
        show_reporting_tab(st.session_state.filtered_data, st.session_state.metrics, st.session_state.insights)


def show_welcome_screen():
    """Show welcome screen"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 0;">
            <h1 style="color: #2acaea; font-size: 4rem;">💊</h1>
            <h2 style="color: #ffffff; margin-bottom: 1rem;">PharmaIntelligence Pro v7.0</h2>
            <p style="color: #cbd5e1; font-size: 1.2rem; margin-bottom: 2rem;">
                Makine öğrenmesi destekli ilaç pazarı analiz platformu
            </p>
            
            <div style="background: rgba(30, 30, 46, 0.7); padding: 2rem; border-radius: 15px; 
                       border: 1px solid rgba(255, 255, 255, 0.1); margin-top: 2rem;">
                <h3 style="color: #2acaea; margin-bottom: 1rem;">🚀 Başlamak İçin:</h3>
                <ol style="text-align: left; color: #e0e0e0; padding-left: 1.5rem;">
                    <li style="margin-bottom: 0.5rem;">Sol taraftan Excel/CSV dosyanızı yükleyin</li>
                    <li style="margin-bottom: 0.5rem;">Veri otomatik olarak temizlenecek ve analiz edilecek</li>
                    <li style="margin-bottom: 0.5rem;">Gelişmiş filtrelerle verinizi keşfedin</li>
                    <li>ML modülleri ile tahminler ve içgörüler elde edin</li>
                </ol>
            </div>
            
            <div style="margin-top: 3rem; color: #94a3b8;">
                <p>⚠️ Excel dosyanızın aşağıdaki kolonları içerdiğinden emin olun:</p>
                <p style="font-size: 0.9rem; font-family: monospace;">
                    Source.Name, Country, Corporation, Molecule,<br>
                    MAT Q3 2022\\nUSD MNF, MAT Q3 2022\\nUnits, ...
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_overview_tab(df, metrics, insights):
    """Overview tab with KPIs and insights"""
    st.markdown('<div class="section-header"><h2>📊 Genel Bakış & Stratejik İçgörüler</h2></div>', unsafe_allow_html=True)
    
    # Display KPIs
    EnhancedVisualizer.create_dashboard_kpis(df, metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display insights in columns
    if insights:
        st.markdown('<div class="section-header"><h3>💡 Stratejik İçgörüler</h3></div>', unsafe_allow_html=True)
        
        cols = st.columns(2)
        for idx, insight in enumerate(insights):
            with cols[idx % 2]:
                insight_class = {
                    'success': 'success-box',
                    'warning': 'warning-box',
                    'info': 'info-box',
                    'price': 'info-box',
                    'geographic': 'info-box'
                }.get(insight['type'], 'info-box')
                
                st.markdown(f"""
                <div class="{insight_class}">
                    <h4>{insight['title']}</h4>
                    <p>{insight['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Satış Trendleri")
        trend_chart = EnhancedVisualizer.create_sales_trend_chart(df)
        if trend_chart:
            st.plotly_chart(trend_chart, use_container_width=True)
        else:
            st.info("Trend analizi için yeterli veri yok")
    
    with col2:
        st.markdown("### 🏢 Pazar Payı Dağılımı")
        market_chart = EnhancedVisualizer.create_market_share_chart(df)
        if market_chart:
            st.plotly_chart(market_chart, use_container_width=True)
        else:
            st.info("Pazar payı analizi için yeterli veri yok")
    
    # Sunburst chart
    st.markdown("### 🌐 Hiyerarşik Pazar Yapısı")
    sunburst_chart = EnhancedVisualizer.create_sunburst_chart(df)
    if sunburst_chart:
        st.plotly_chart(sunburst_chart, use_container_width=True)
    else:
        st.info("Sunburst grafiği için yeterli veri yok")


def show_price_analysis_tab(df):
    """Price analysis tab"""
    st.markdown('<div class="section-header"><h2>💰 Fiyat Analizi & Segmentasyon</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Price-Volume scatter plot
        sample_size = st.slider("Örnek Sayısı", 1000, 20000, 10000, 1000, 
                              help="Büyük veri setleri için performansı iyileştirir")
        
        st.markdown("### 📊 Fiyat-Hacim İlişkisi")
        price_volume_chart = EnhancedVisualizer.create_price_volume_chart(df, sample_size)
        if price_volume_chart:
            st.plotly_chart(price_volume_chart, use_container_width=True)
        else:
            st.warning("""
            ⚠️ Fiyat-hacim analizi için gerekli veriler bulunamadı.
            
            **Gerekli kolonlar:**
            - `Price_2024` veya `Price_2023`
            - `Volume_2024` veya `Volume_2023`
            - `Sales_2024` veya `Sales_2023`
            
            Veri yükleme sırasında bu kolonlar otomatik oluşturulmalıdır.
            """)
    
    with col2:
        st.markdown("### 🏷️ Fiyat Segmentasyonu")
        price_seg_chart = EnhancedVisualizer.create_price_segmentation_chart(df)
        if price_seg_chart:
            st.plotly_chart(price_seg_chart, use_container_width=True)
        else:
            st.info("Fiyat segmentasyonu analizi için yeterli veri yok")
        
        # Price statistics
        if 'Price_2024' in df.columns:
            st.markdown("### 📈 Fiyat İstatistikleri")
            
            price_stats = df['Price_2024'].describe()
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Ortalama", f"${price_stats['mean']:.2f}")
                st.metric("Minimum", f"${price_stats['min']:.2f}")
            
            with col_b:
                st.metric("Medyan", f"${price_stats['50%']:.2f}")
                st.metric("Maximum", f"${price_stats['max']:.2f}")
            
            # Price distribution
            st.markdown("### 📊 Fiyat Dağılımı")
            
            # Create histogram
            fig = px.histogram(
                df, 
                x='Price_2024',
                nbins=50,
                title='Fiyat Dağılımı',
                labels={'Price_2024': 'Fiyat (USD)', 'count': 'Ürün Sayısı'}
            )
            
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                showlegend=False,
                margin=dict(t=30, l=0, r=0, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)


def show_geographic_tab(df):
    """Geographic analysis tab"""
    st.markdown('<div class="section-header"><h2>🌍 Coğrafi Analiz & Global Dağılım</h2></div>', unsafe_allow_html=True)
    
    # Choropleth map
    st.markdown("### 🗺️ Global Pazar Dağılımı")
    choropleth = EnhancedVisualizer.create_choropleth_map(df)
    if choropleth:
        st.plotly_chart(choropleth, use_container_width=True)
    else:
        st.info("Harita oluşturmak için ülke ve satış verileri gerekli")
    
    # Regional analysis
    if 'Region' in df.columns and 'Sales_2024' in df.columns:
        st.markdown("### 📊 Bölgesel Analiz")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional sales breakdown
            region_sales = df.groupby('Region')['Sales_2024'].agg(['sum', 'count']).reset_index()
            region_sales = region_sales.sort_values('sum', ascending=False)
            
            fig1 = px.bar(
                region_sales,
                x='sum',
                y='Region',
                orientation='h',
                title='Bölgelere Göre Satışlar',
                labels={'sum': 'Toplam Satış (USD)', 'Region': 'Bölge'},
                color='sum',
                color_continuous_scale='Viridis'
            )
            
            fig1.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                showlegend=False,
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Regional product count
            fig2 = px.pie(
                region_sales,
                values='count',
                names='Region',
                title='Bölgelere Göre Ürün Dağılımı',
                hole=0.4
            )
            
            fig2.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0',
                showlegend=True
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    # Top countries table
    if 'Country' in df.columns and 'Sales_2024' in df.columns:
        st.markdown("### 🏆 En Büyük 10 Ülke")
        
        country_sales = df.groupby('Country')['Sales_2024'].agg(['sum', 'count']).reset_index()
        country_sales = country_sales.sort_values('sum', ascending=False).head(10)
        country_sales['sum_formatted'] = country_sales['sum'].apply(lambda x: f"${x/1e6:.1f}M")
        country_sales['share'] = (country_sales['sum'] / country_sales['sum'].sum()) * 100
        
        # Display as styled table
        st.dataframe(
            country_sales[['Country', 'sum_formatted', 'share', 'count']].rename(
                columns={'Country': 'Ülke', 'sum_formatted': 'Toplam Satış', 
                        'share': 'Pazar Payı (%)', 'count': 'Ürün Sayısı'}
            ),
            use_container_width=True,
            hide_index=True
        )


def show_competition_tab(df, metrics):
    """Competition analysis tab"""
    st.markdown('<div class="section-header"><h2>🏆 Rekabet Analizi & Pazar Yapısı</h2></div>', unsafe_allow_html=True)
    
    # Market concentration metrics
    st.markdown("### 📊 Pazar Konsantrasyon Metrikleri")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hhi = metrics.get('hhi_index', 0)
        if hhi > 2500:
            status = "🔴 Monopolistik"
        elif hhi > 1500:
            status = "🟡 Oligopol"
        else:
            status = "🟢 Rekabetçi"
        
        st.metric("HHI İndeksi", f"{hhi:.0f}", status)
    
    with col2:
        top_share = metrics.get('top_corp_share', 0)
        st.metric("Lider Şirket Payı", f"%{top_share:.1f}")
    
    with col3:
        top_3_share = metrics.get('top_3_corp_share', 0)
        st.metric("Top 3 Şirket Payı", f"%{top_3_share:.1f}")
    
    with col4:
        unique_corps = metrics.get('unique_corporations', 0)
        st.metric("Toplam Şirket", unique_corps)
    
    # Market share evolution
    if all(col in df.columns for col in ['Corporation', 'Sales_2023', 'Sales_2024']):
        st.markdown("### 📈 Şirket Performansı (2023 → 2024)")
        
        # Calculate growth by corporation
        corp_growth = df.groupby('Corporation').agg({
            'Sales_2023': 'sum',
            'Sales_2024': 'sum'
        }).reset_index()
        
        corp_growth['Growth'] = ((corp_growth['Sales_2024'] - corp_growth['Sales_2023']) / 
                                corp_growth['Sales_2023'].replace(0, np.nan)) * 100
        
        # Filter top corporations
        top_corps = corp_growth.nlargest(15, 'Sales_2024')
        
        # Create bubble chart
        fig = px.scatter(
            top_corps,
            x='Sales_2024',
            y='Growth',
            size='Sales_2024',
            color='Growth',
            hover_name='Corporation',
            hover_data={'Sales_2023': ':.2f', 'Sales_2024': ':.2f', 'Growth': ':.2f'},
            title='Şirket Büyüme Performansı',
            labels={'Sales_2024': '2024 Satışları (USD)', 'Growth': 'Büyüme (%)'},
            color_continuous_scale='RdYlGn',
            size_max=60
        )
        
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', type='log'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.add_vline(x=top_corps['Sales_2024'].median(), line_dash="dash", line_color="white", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Competitive landscape table
    if 'Corporation' in df.columns and 'Sales_2024' in df.columns:
        st.markdown("### 📋 Rekabetçi Lansman Tablosu")
        
        corp_analysis = df.groupby('Corporation').agg({
            'Sales_2024': ['sum', 'count'],
            'Price_2024': 'mean',
            'Growth_2023_2024': 'mean',
            'Market_Share': 'sum'
        }).reset_index()
        
        # Flatten column names
        corp_analysis.columns = ['Corporation', 'Total_Sales', 'Product_Count', 
                                'Avg_Price', 'Avg_Growth', 'Market_Share']
        
        corp_analysis = corp_analysis.sort_values('Total_Sales', ascending=False).head(20)
        
        # Format columns
        corp_analysis['Total_Sales_Formatted'] = corp_analysis['Total_Sales'].apply(
            lambda x: f"${x/1e6:.1f}M"
        )
        corp_analysis['Avg_Price_Formatted'] = corp_analysis['Avg_Price'].apply(
            lambda x: f"${x:.2f}"
        )
        corp_analysis['Avg_Growth_Formatted'] = corp_analysis['Avg_Growth'].apply(
            lambda x: f"%{x:.1f}"
        )
        corp_analysis['Market_Share_Formatted'] = corp_analysis['Market_Share'].apply(
            lambda x: f"%{x:.1f}"
        )
        
        # Display table
        st.dataframe(
            corp_analysis[[
                'Corporation', 'Total_Sales_Formatted', 'Product_Count',
                'Avg_Price_Formatted', 'Avg_Growth_Formatted', 'Market_Share_Formatted'
            ]].rename(columns={
                'Corporation': 'Şirket',
                'Total_Sales_Formatted': 'Toplam Satış',
                'Product_Count': 'Ürün Sayısı',
                'Avg_Price_Formatted': 'Ort. Fiyat',
                'Avg_Growth_Formatted': 'Ort. Büyüme',
                'Market_Share_Formatted': 'Pazar Payı'
            }),
            use_container_width=True,
            hide_index=True
        )


def show_ml_lab_tab(df):
    """ML Laboratory tab with interactive features"""
    st.markdown('<div class="section-header"><h2>🤖 ML Laboratuvarı - İleri Analitik</h2></div>', unsafe_allow_html=True)
    
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        "🔮 Satış Tahmini",
        "🎯 Ürün Kümeleme",
        "⚠️ Anomali Tespiti",
        "🧪 What-If Analizi"
    ])
    
    with ml_tab1:
        show_forecasting_tab(df)
    
    with ml_tab2:
        show_clustering_tab(df)
    
    with ml_tab3:
        show_anomaly_tab(df)
    
    with ml_tab4:
        show_whatif_tab(df)


def show_forecasting_tab(df):
    """Sales forecasting tab"""
    st.markdown("### 🔮 Satış Tahmini (Prophet)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Entity type selection
        entity_type = st.selectbox(
            "Tahmin Entity Tipi",
            ["Molecule", "Country"],
            help="Molekül veya Ülke bazında tahmin yapın"
        )
    
    with col2:
        # Entity selection
        if entity_type == "Molecule" and "Molecule" in df.columns:
            entities = sorted(df['Molecule'].dropna().unique())
            entity_name = st.selectbox(
                f"{entity_type} Seçin",
                entities,
                index=min(10, len(entities)-1) if entities else 0
            )
        elif entity_type == "Country" and "Country" in df.columns:
            entities = sorted(df['Country'].dropna().unique())
            entity_name = st.selectbox(
                f"{entity_type} Seçin",
                entities,
                index=0 if entities else 0
            )
        else:
            st.warning(f"⚠️ {entity_type} kolonu veride bulunamadı")
            return
    
    # Forecast parameters
    forecast_years = st.slider("Tahmin Yılı Sayısı", 1, 5, 2)
    
    if st.button("🎯 Tahmini Başlat", type="primary", use_container_width=True):
        with st.spinner(f"{entity_name} için tahmin yapılıyor..."):
            results, error = AdvancedMLEngine.forecast_sales_prophet(
                df, entity_type, entity_name, forecast_years
            )
            
            if error:
                st.error(f"❌ Tahmin hatası: {error}")
            elif results:
                # Display forecast results
                st.success(f"✅ {entity_name} için tahmin tamamlandı!")
                
                # Plot forecast
                fig = results['model'].plot(results['forecast'])
                
                # Customize plot
                fig.update_layout(
                    title=f"{entity_name} - Satış Tahmini",
                    xaxis_title="Tarih",
                    yaxis_title="Satış (USD)",
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e0e0'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast table
                st.markdown("### 📋 Tahmin Tablosu")
                
                forecast_table = results['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_years)
                forecast_table['ds'] = forecast_table['ds'].dt.year
                forecast_table = forecast_table.rename(columns={
                    'ds': 'Yıl',
                    'yhat': 'Tahmin',
                    'yhat_lower': 'Alt Sınır',
                    'yhat_upper': 'Üst Sınır'
                })
                
                st.dataframe(
                    forecast_table.style.format({
                        'Tahmin': '${:,.0f}',
                        'Alt Sınır': '${:,.0f}',
                        'Üst Sınır': '${:,.0f}'
                    }),
                    use_container_width=True
                )
                
                # Growth rate calculation
                if len(results['historical']) >= 2:
                    last_historical = results['historical']['y'].iloc[-1]
                    last_forecast = results['forecast']['yhat'].iloc[-1]
                    cagr = ((last_forecast / last_historical) ** (1/forecast_years) - 1) * 100
                    
                    st.metric(
                        "Tahmini CAGR",
                        f"%{cagr:.1f}",
                        f"${last_forecast-last_historical:,.0f}"
                    )
            else:
                st.warning("Tahmin sonuçları alınamadı")


def show_clustering_tab(df):
    """Product clustering tab"""
    st.markdown("### 🎯 Ürün Kümeleme (K-Means)")
    
    # Clustering parameters
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider(
            "Küme Sayısı",
            min_value=2,
            max_value=8,
            value=4,
            help="BCG Matrisi için 4 küme önerilir (Cash Cows, Stars, Question Marks, Dogs)"
        )
    
    with col2:
        sample_size = st.slider(
            "Örnek Boyutu",
            min_value=1000,
            max_value=20000,
            value=min(10000, len(df)),
            step=1000
        )
    
    if st.button("🎯 Kümeleri Oluştur", type="primary", use_container_width=True):
        with st.spinner("Ürünler kümelendiriliyor..."):
            # Sample data for performance
            sample_df = df.sample(min(sample_size, len(df)), random_state=42)
            
            results, error = AdvancedMLEngine.create_clustering_analysis(sample_df, n_clusters)
            
            if error:
                st.error(f"❌ Kümeleme hatası: {error}")
            elif results:
                st.success(f"✅ {len(sample_df)} ürün {n_clusters} kümeye ayrıldı")
                
                # Display silhouette score
                silhouette = results['silhouette_score']
                st.metric("Silhouette Score", f"{silhouette:.3f}")
                
                # Create 3D scatter plot
                if results['pca_data'].shape[1] >= 3:
                    pca_df = pd.DataFrame({
                        'PC1': results['pca_data'][:, 0],
                        'PC2': results['pca_data'][:, 1],
                        'PC3': results['pca_data'][:, 2],
                        'Cluster': results['cluster_names'],
                        'Molecule': sample_df['Molecule'].values if 'Molecule' in sample_df.columns else None,
                        'Corporation': sample_df['Corporation'].values if 'Corporation' in sample_df.columns else None,
                        'Sales_2024': sample_df['Sales_2024'].values if 'Sales_2024' in sample_df.columns else None
                    })
                    
                    fig = px.scatter_3d(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        z='PC3',
                        color='Cluster',
                        hover_name='Molecule',
                        hover_data=['Corporation', 'Sales_2024'],
                        title='3D Kümeleme Görünümü',
                        labels={'PC1': f"PC1 (%{results['pca_variance'][0]*100:.1f})",
                               'PC2': f"PC2 (%{results['pca_variance'][1]*100:.1f})",
                               'PC3': f"PC3 (%{results['pca_variance'][2]*100:.1f})"}
                    )
                    
                    fig.update_layout(
                        height=600,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e0e0e0'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster analysis
                st.markdown("### 📊 Küme Analizi")
                
                # Add cluster labels to sample data
                sample_df_with_clusters = sample_df.copy()
                sample_df_with_clusters['Cluster'] = results['cluster_names']
                
                # Analyze clusters
                cluster_analysis = sample_df_with_clusters.groupby('Cluster').agg({
                    'Sales_2024': ['count', 'sum', 'mean'],
                    'Price_2024': 'mean',
                    'Growth_2023_2024': 'mean',
                    'Market_Share': 'sum'
                }).reset_index()
                
                # Flatten column names
                cluster_analysis.columns = ['Cluster', 'Product_Count', 'Total_Sales', 
                                          'Avg_Sales', 'Avg_Price', 'Avg_Growth', 'Total_Market_Share']
                
                # Format for display
                cluster_analysis['Total_Sales_Formatted'] = cluster_analysis['Total_Sales'].apply(
                    lambda x: f"${x/1e6:.1f}M"
                )
                cluster_analysis['Avg_Sales_Formatted'] = cluster_analysis['Avg_Sales'].apply(
                    lambda x: f"${x:,.0f}"
                )
                cluster_analysis['Avg_Price_Formatted'] = cluster_analysis['Avg_Price'].apply(
                    lambda x: f"${x:.2f}"
                )
                cluster_analysis['Avg_Growth_Formatted'] = cluster_analysis['Avg_Growth'].apply(
                    lambda x: f"%{x:.1f}"
                )
                cluster_analysis['Total_Market_Share_Formatted'] = cluster_analysis['Total_Market_Share'].apply(
                    lambda x: f"%{x:.1f}"
                )
                
                # Display cluster analysis
                st.dataframe(
                    cluster_analysis[[
                        'Cluster', 'Product_Count', 'Total_Sales_Formatted',
                        'Avg_Sales_Formatted', 'Avg_Price_Formatted',
                        'Avg_Growth_Formatted', 'Total_Market_Share_Formatted'
                    ]].rename(columns={
                        'Cluster': 'Küme',
                        'Product_Count': 'Ürün Sayısı',
                        'Total_Sales_Formatted': 'Toplam Satış',
                        'Avg_Sales_Formatted': 'Ort. Satış',
                        'Avg_Price_Formatted': 'Ort. Fiyat',
                        'Avg_Growth_Formatted': 'Ort. Büyüme',
                        'Total_Market_Share_Formatted': 'Top. Pazar Payı'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Strategic recommendations based on BCG Matrix
                if n_clusters == 4:
                    st.markdown("### 💡 BCG Matrisi Stratejik Öneriler")
                    
                    recommendations = {
                        'Nakit İnekleri (Cash Cows)': "✅ **Yatırımı Koru**: Düşük büyüme ancak yüksek pazar payı. Nakit akışı sağlar, minimum yatırım yap.",
                        'Yıldızlar (Stars)': "🚀 **Yatırımı Artır**: Yüksek büyüme ve yüksek pazar payı. Gelecekteki nakit inekleri, yoğun yatırım yap.",
                        'Soru İşaretleri (Question Marks)': "🤔 **Seçici Yatırım**: Yüksek büyüme ancak düşük pazar payı. Ya yıldıza dönüşür ya da köpek olur. Dikkatli yatırım yap.",
                        'Köpekler (Dogs)': "🔄 **Yeniden Değerlendir**: Düşük büyüme ve düşük pazar payı. Satmayı, kapatmayı veya yeniden konumlandırmayı düşün."
                    }
                    
                    for cluster_name in cluster_analysis['Cluster'].unique():
                        if cluster_name in recommendations:
                            with st.expander(f"{cluster_name} - Strateji", expanded=True):
                                st.info(recommendations[cluster_name])
            else:
                st.warning("Kümeleme sonuçları alınamadı")


def show_anomaly_tab(df):
    """Anomaly detection tab"""
    st.markdown("### ⚠️ Anomali Tespiti (Isolation Forest)")
    
    # Anomaly detection parameters
    col1, col2 = st.columns(2)
    
    with col1:
        contamination = st.slider(
            "Anomali Oranı",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Beklenen anomali oranı (contamination)"
        )
    
    with col2:
        sample_size = st.slider(
            "Analiz Örnek Boyutu",
            min_value=1000,
            max_value=20000,
            value=min(10000, len(df)),
            step=1000
        )
    
    if st.button("🔍 Anomalileri Tespit Et", type="primary", use_container_width=True):
        with st.spinner("Anomali analizi yapılıyor..."):
            # Sample data for performance
            sample_df = df.sample(min(sample_size, len(df)), random_state=42)
            
            results, error = AdvancedMLEngine.detect_anomalies_isolation(sample_df, contamination)
            
            if error:
                st.error(f"❌ Anomali tespit hatası: {error}")
            elif results:
                st.success(f"✅ Anomali analizi tamamlandı")
                
                # Display anomaly metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Tespit Edilen Anomali",
                        f"{results['anomaly_count']}",
                        f"%{results['anomaly_percentage']:.1f}"
                    )
                
                with col2:
                    st.metric(
                        "Normal Veri",
                        f"{results['normal_count']}",
                        f"%{100 - results['anomaly_percentage']:.1f}"
                    )
                
                with col3:
                    avg_score = results['anomaly_scores'].mean()
                    st.metric(
                        "Ortalama Anomali Skoru",
                        f"{avg_score:.3f}"
                    )
                
                # Create anomaly visualization
                st.markdown("### 📊 Anomali Dağılımı")
                
                # Prepare data for visualization
                anomaly_df = sample_df.copy()
                anomaly_df['Is_Anomaly'] = results['is_anomaly']
                anomaly_df['Anomaly_Score'] = results['anomaly_scores']
                
                # Scatter plot for anomalies
                if 'Price_2024' in anomaly_df.columns and 'Sales_2024' in anomaly_df.columns:
                    fig = px.scatter(
                        anomaly_df,
                        x='Price_2024',
                        y='Sales_2024',
                        color='Is_Anomaly',
                        size='Anomaly_Score',
                        hover_name='Molecule' if 'Molecule' in anomaly_df.columns else None,
                        hover_data=['Corporation', 'Country', 'Anomaly_Score'],
                        title='Anomali Dağılımı - Fiyat vs Satış',
                        labels={'Is_Anomaly': 'Anomali', 'Price_2024': 'Fiyat (USD)', 'Sales_2024': 'Satış (USD)'},
                        color_discrete_map={True: '#ff6b6b', False: '#2d7dd2'}
                    )
                    
                    fig.update_layout(
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#e0e0e0',
                        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show anomaly details table
                st.markdown("### 📋 Anomali Detayları")
                
                anomalies = anomaly_df[anomaly_df['Is_Anomaly']].copy()
                
                if len(anomalies) > 0:
                    # Sort by anomaly score
                    anomalies = anomalies.sort_values('Anomaly_Score', ascending=True)
                    
                    # Select relevant columns for display
                    display_cols = ['Anomaly_Score']
                    for col in ['Molecule', 'Corporation', 'Country', 'Price_2024', 'Sales_2024', 
                               'Volume_2024', 'Growth_2023_2024', 'Market_Share']:
                        if col in anomalies.columns:
                            display_cols.append(col)
                    
                    # Format for display
                    display_df = anomalies[display_cols].head(50)  # Show top 50 anomalies
                    
                    # Create formatted version for display
                    formatted_df = display_df.copy()
                    if 'Price_2024' in formatted_df.columns:
                        formatted_df['Price_2024'] = formatted_df['Price_2024'].apply(lambda x: f"${x:.2f}")
                    if 'Sales_2024' in formatted_df.columns:
                        formatted_df['Sales_2024'] = formatted_df['Sales_2024'].apply(lambda x: f"${x:,.0f}")
                    if 'Market_Share' in formatted_df.columns:
                        formatted_df['Market_Share'] = formatted_df['Market_Share'].apply(lambda x: f"%{x:.2f}")
                    if 'Growth_2023_2024' in formatted_df.columns:
                        formatted_df['Growth_2023_2024'] = formatted_df['Growth_2023_2024'].apply(lambda x: f"%{x:.1f}")
                    
                    st.dataframe(
                        formatted_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Anomaly insights
                    st.markdown("### 💡 Anomali İçgörüleri")
                    
                    if len(anomalies) > 0:
                        # Common characteristics of anomalies
                        if 'Price_2024' in anomalies.columns:
                            price_stats = anomalies['Price_2024'].describe()
                            st.write(f"**Fiyat Dağılımı**: Min ${price_stats['min']:.2f}, Max ${price_stats['max']:.2f}, Ortalama ${price_stats['mean']:.2f}")
                        
                        if 'Corporation' in anomalies.columns:
                            top_corp_anomalies = anomalies['Corporation'].value_counts().head(5)
                            st.write(f"**En Çok Anomalili Şirketler**: {', '.join(top_corp_anomalies.index.tolist())}")
                        
                        if 'Country' in anomalies.columns:
                            top_country_anomalies = anomalies['Country'].value_counts().head(5)
                            st.write(f"**En Çok Anomalili Ülkeler**: {', '.join(top_country_anomalies.index.tolist())}")
                        
                        # Recommendations
                        st.markdown("""
                        #### 🛡️ Önerilen Aksiyonlar:
                        1. **Yüksek fiyat anomalileri**: Fiyatlandırma stratejisini gözden geçir
                        2. **Yüksek büyüme anomalileri**: Büyüme trendlerini doğrula ve yatırım fırsatlarını değerlendir
                        3. **Düşük satış anomalileri**: Ürün performansını yeniden değerlendir
                        4. **Şirket bazlı anomaliler**: İç kontrolleri ve raporlama süreçlerini gözden geçir
                        """)
                else:
                    st.info("🎉 Anomali tespit edilmedi. Veri seti normal görünüyor.")
            else:
                st.warning("Anomali tespit sonuçları alınamadı")


def show_whatif_tab(df):
    """What-If analysis tab"""
    st.markdown("### 🧪 What-If Analizi")
    
    st.markdown("""
    Senaryo bazlı analiz ile farklı stratejilerin pazar üzerindeki etkisini simüle edin.
    """)
    
    # Create scenario builder
    with st.form("what_if_scenarios"):
        st.markdown("#### 📋 Senaryo Oluştur")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_name = st.text_input("Senaryo Adı", "Fiyat Artışı Senaryosu")
            
            price_change = st.slider(
                "Fiyat Değişimi (%)",
                min_value=-50,
                max_value=100,
                value=10,
                step=5,
                help="Ürün fiyatlarındaki değişim oranı"
            )
        
        with col2:
            scenario_desc = st.text_area(
                "Senaryo Açıklaması",
                "Tüm ürünlerde %10 fiyat artışı senaryosu"
            )
            
            volume_change = st.slider(
                "Hacim Değişimi (%)",
                min_value=-50,
                max_value=100,
                value=0,
                step=5,
                help="Satış hacmindeki değişim oranı"
            )
        
        submitted = st.form_submit_button("🎯 Senaryoyu Çalıştır", type="primary")
    
    if submitted:
        # Define scenarios
        scenarios = [{
            'name': scenario_name,
            'description': scenario_desc,
            'price_change': price_change,
            'volume_change': volume_change
        }]
        
        with st.spinner("Senaryo analiz ediliyor..."):
            results, error = AdvancedMLEngine.perform_what_if_analysis(df, scenarios)
            
            if error:
                st.error(f"❌ Senaryo analiz hatası: {error}")
            elif results:
                st.success("✅ Senaryo analizi tamamlandı")
                
                for result in results:
                    # Display results in metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Mevcut Satışlar",
                            f"${result['current_sales']/1e9:.2f}B"
                        )
                    
                    with col2:
                        st.metric(
                            "Senaryo Satışları",
                            f"${result['scenario_sales']/1e9:.2f}B"
                        )
                    
                    with col3:
                        impact = result['impact_percentage']
                        impact_color = "green" if impact > 0 else "red" if impact < 0 else "gray"
                        st.metric(
                            "Toplam Etki",
                            f"%{impact:.1f}",
                            delta=f"${result['impact_absolute']/1e6:.1f}M",
                            delta_color="normal"
                        )
                    
                    # Detailed analysis
                    st.markdown(f"#### 📊 {result['scenario']} - Detaylı Analiz")
                    
                    # Create impact breakdown
                    impact_data = []
                    
                    if 'Corporation' in df.columns:
                        # Corporate impact
                        current_corp = df.groupby('Corporation')['Sales_2024'].sum()
                        scenario_corp = df.copy()
                        
                        if price_change != 0:
                            scenario_corp['Price_scenario'] = scenario_corp['Price_2024'] * (1 + price_change/100)
                            scenario_corp['Sales_scenario'] = scenario_corp['Price_scenario'] * scenario_corp['Volume_2024']
                        elif volume_change != 0:
                            scenario_corp['Volume_scenario'] = scenario_corp['Volume_2024'] * (1 + volume_change/100)
                            scenario_corp['Sales_scenario'] = scenario_corp['Price_2024'] * scenario_corp['Volume_scenario']
                        
                        scenario_corp_agg = scenario_corp.groupby('Corporation')['Sales_scenario'].sum()
                        
                        # Calculate top 5 impacts
                        for corp in current_corp.index:
                            if corp in scenario_corp_agg.index:
                                current = current_corp[corp]
                                scenario = scenario_corp_agg[corp]
                                impact = ((scenario - current) / current) * 100 if current > 0 else 0
                                
                                impact_data.append({
                                    'Entity': corp,
                                    'Type': 'Şirket',
                                    'Current': current,
                                    'Scenario': scenario,
                                    'Impact': impact
                                })
                    
                    # Create impact chart
                    if impact_data:
                        impact_df = pd.DataFrame(impact_data)
                        impact_df = impact_df.sort_values('Impact', ascending=False).head(10)
                        
                        fig = px.bar(
                            impact_df,
                            x='Impact',
                            y='Entity',
                            orientation='h',
                            title='En Çok Etkilenen Şirketler',
                            labels={'Impact': 'Etki (%)', 'Entity': 'Şirket'},
                            color='Impact',
                            color_continuous_scale='RdYlGn'
                        )
                        
                        fig.update_layout(
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#e0e0e0',
                            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                            yaxis=dict(autorange="reversed")
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Strategic recommendations
                    st.markdown("#### 💡 Stratejik Öneriler")
                    
                    if result['impact_percentage'] > 10:
                        st.success(f"""
                        **🚀 Büyük Fırsat**: Senaryo %{result['impact_percentage']:.1f} pozitif etki gösteriyor.
                        
                        **Öneriler:**
                        1. Bu senaryoyu pilot bölgelerde test edin
                        2. Müşteri tepkilerini ölçün
                        3. Rekabetçi tepkileri değerlendirin
                        4. Kademeli olarak uygulamaya geçin
                        """)
                    elif result['impact_percentage'] > 0:
                        st.info(f"""
                        **📈 Olumlu Etki**: Senaryo %{result['impact_percentage']:.1f} pozitif etki gösteriyor.
                        
                        **Öneriler:**
                        1. Dikkatli bir şekilde uygulamayı değerlendirin
                        2. Pazarlama stratejilerini güçlendirin
                        3. Müşteri sadakatini koruyun
                        """)
                    elif result['impact_percentage'] < -10:
                        st.error(f"""
                        **⚠️ Büyük Risk**: Senaryo %{abs(result['impact_percentage']):.1f} negatif etki gösteriyor.
                        
                        **Öneriler:**
                        1. Bu senaryodan kaçının
                        2. Alternatif stratejiler geliştirin
                        3. Müşteri kaybı riskini minimize edin
                        4. Fiyatlandırma stratejinizi yeniden değerlendirin
                        """)
                    else:
                        st.warning(f"""
                        **⚖️ Sınırlı Etki**: Senaryo %{abs(result['impact_percentage']):.1f} etki gösteriyor.
                        
                        **Öneriler:**
                        1. Küçük ölçekte test edin
                        2. Geri bildirimleri dikkate alın
                        3. Rekabet avantajını koruyun
                        """)
            else:
                st.warning("Senaryo analiz sonuçları alınamadı")


def show_reporting_tab(df, metrics, insights):
    """Reporting tab"""
    st.markdown('<div class="section-header"><h2>📑 Raporlama & İndirme</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Veri İndirme")
        
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 CSV Olarak İndir",
            data=csv,
            file_name="pharma_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("### 📈 Raporlar")
        
        # Excel report
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Veri', index=False)
            
            # Create summary sheet
            summary_data = {
                'Metrik': ['Toplam Satır', 'Toplam Sütun', 'Toplam Pazar Değeri', 
                          'Ortalama Büyüme', 'HHI İndeksi', 'Şirket Sayısı',
                          'Ülke Sayısı', 'Molekül Sayısı'],
                'Değer': [
                    metrics.get('total_rows', 0),
                    metrics.get('total_columns', 0),
                    f"${metrics.get('total_market_value', 0)/1e9:.2f}B",
                    f"%{metrics.get('avg_growth', 0):.1f}",
                    metrics.get('hhi_index', 0),
                    metrics.get('unique_corporations', 0),
                    metrics.get('unique_countries', 0),
                    metrics.get('unique_molecules', 0)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Özet', index=False)
        
        excel_buffer.seek(0)
        
        st.download_button(
            label="📊 Excel Raporu İndir",
            data=excel_buffer,
            file_name="pharma_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        st.markdown("### 🖼️ Görseller")
        
        # Generate and offer chart downloads
        if st.button("📷 Grafikleri Kaydet", use_container_width=True):
            st.info("""
            Grafikleri kaydetmek için:
            1. Her grafiğin üzerindeki kamera ikonuna tıklayın
            2. PNG formatında indirin
            3. Sunumlarınızda kullanın
            """)
    
    # Generate executive summary
    st.markdown("### 📋 Yönetici Özeti")
    
    if insights:
        with st.expander("📄 Tam Raporu Görüntüle", expanded=True):
            # Executive Summary
            st.markdown("#### 🎯 Yönetici Özeti")
            
            summary_text = f"""
            **Analiz Tarihi**: {datetime.now().strftime('%d %B %Y')}
            **Veri Seti**: {metrics.get('total_rows', 0):,} satır, {metrics.get('total_columns', 0):,} sütun
            
            ### 🏆 Ana Bulgular:
            
            1. **Pazar Büyüklüğü**: Toplam pazar değeri ${metrics.get('total_market_value', 0)/1e9:.2f}B
            2. **Büyüme Trendi**: Yıllık ortalama büyüme %{metrics.get('avg_growth', 0):.1f}
            3. **Pazar Lideri**: {metrics.get('top_corp', 'N/A')} - %{metrics.get('top_corp_share', 0):.1f} pazar payı
            4. **Pazar Yapısı**: HHI İndeksi {metrics.get('hhi_index', 0):.0f} - {
                "Monopolistik" if metrics.get('hhi_index', 0) > 2500 else 
                "Oligopol" if metrics.get('hhi_index', 0) > 1500 else 
                "Rekabetçi"
            }
            5. **Coğrafi Dağılım**: {metrics.get('top_country', 'N/A')} - %{metrics.get('top_country_share', 0):.1f} pazar payı
            
            ### 💡 Stratejik Öneriler:
            """
            
            st.markdown(summary_text)
            
            # Add insights as recommendations
            for insight in insights[:5]:  # Show top 5 insights
                st.markdown(f"- **{insight['title']}**: {insight['description']}")
            
            # Recommendations based on analysis
            st.markdown("""
            ### 🚀 Önerilen Aksiyonlar:
            
            1. **Büyüme Odaklı Strateji**: Pozitif büyüme gösteren ürünlere yatırım yapın
            2. **Risk Yönetimi**: Coğrafi ve şirket bazlı riskleri dağıtın
            3. **Fiyat Optimizasyonu**: Fiyat segmentasyonu ile karlılığı artırın
            4. **Rekabet Analizi**: Pazar payı değişimlerini yakından izleyin
            5. **İnovasyon**: Yüksek büyüme potansiyelli segmentlere odaklanın
            """)
            
            # Performance metrics table
            st.markdown("#### 📊 Performans Metrikleri")
            
            perf_metrics = pd.DataFrame({
                'Metrik': [
                    'Toplam Pazar Değeri (2024)',
                    'Ortalama Yıllık Büyüme',
                    'Pazar Lideri Payı',
                    'HHI Konsantrasyon İndeksi',
                    'Ürün Çeşitliliği',
                    'Coğrafi Kapsam',
                    'Pozitif Büyüyen Ürünler',
                    'Ortalama Fiyat'
                ],
                'Değer': [
                    f"${metrics.get('total_market_value', 0)/1e9:.2f}B",
                    f"%{metrics.get('avg_growth', 0):.1f}",
                    f"%{metrics.get('top_corp_share', 0):.1f}",
                    f"{metrics.get('hhi_index', 0):.0f}",
                    f"{metrics.get('unique_molecules', 0):,}",
                    f"{metrics.get('unique_countries', 0):,} ülke",
                    f"%{metrics.get('positive_growth_count', 0)/metrics.get('total_rows', 1)*100:.1f}",
                    f"${metrics.get('avg_price', 0):.2f}"
                ],
                'Durum': [
                    '✅' if metrics.get('total_market_value', 0) > 0 else '⚠️',
                    '✅' if metrics.get('avg_growth', 0) > 0 else '⚠️',
                    '⚠️' if metrics.get('top_corp_share', 0) > 30 else '✅',
                    '✅' if metrics.get('hhi_index', 0) < 1500 else '⚠️',
                    '✅' if metrics.get('unique_molecules', 0) > 50 else '⚠️',
                    '✅' if metrics.get('unique_countries', 0) > 10 else '⚠️',
                    '✅' if (metrics.get('positive_growth_count', 0)/metrics.get('total_rows', 1)*100) > 50 else '⚠️',
                    '✅' if metrics.get('avg_price', 0) > 0 else '⚠️'
                ]
            })
            
            st.dataframe(
                perf_metrics,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Rapor oluşturmak için önce veri yükleyin ve analiz edin.")


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Kritik Hata: {str(e)}")
        st.error("Lütfen sayfayı yenileyin veya destek ekibiyle iletişime geçin.")
        
        if st.button("🔄 Sayfayı Yenile"):
            st.rerun()
