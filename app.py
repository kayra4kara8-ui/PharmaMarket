# ============================================================================
# PHARMAINTELLIGENCE PRO - ENTERPRISE PHARMACEUTICAL ANALYTICS PLATFORM
# ============================================================================
# Version: 6.0 - Full Featured ML Edition
# Lines: 3000+
# Author: Senior Data Science Team
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import (
    IsolationForest, 
    RandomForestRegressor, 
    GradientBoostingRegressor,
    RandomForestClassifier
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error, 
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage

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

# Country normalization
try:
    import pycountry
    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False

# ============================================================================
# CONFIGURATION & GLOBAL SETTINGS
# ============================================================================

class Config:
    """Application configuration"""
    APP_NAME = "PharmaIntelligence Pro ML"
    VERSION = "6.0"
    MAX_ROWS_DISPLAY = 1000
    CHART_HEIGHT_STANDARD = 500
    CHART_HEIGHT_LARGE = 700
    CHART_HEIGHT_SMALL = 400
    SAMPLE_SIZE_DEFAULT = 5000
    SAMPLE_SIZE_MIN = 1000
    SAMPLE_SIZE_MAX = 50000
    ML_MIN_SAMPLES = 50
    CLUSTERING_MIN_SAMPLES_PER_CLUSTER = 10
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
    MIN_CONTAMINATION = 0.01
    MAX_CONTAMINATION = 0.3

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================

PROFESSIONAL_CSS = """
<style>
    /* ========== GLOBAL STYLES ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 0;
    }
    
    /* ========== HEADERS ========== */
    .section-header {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(42, 202, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, 
            rgba(255,255,255,0.1) 0%, 
            rgba(255,255,255,0.05) 100%);
        pointer-events: none;
    }
    
    .section-header h1, .section-header h2 {
        color: #f8fafc;
        margin: 0;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        position: relative;
        z-index: 1;
    }
    
    .section-header h1 {
        font-size: 2.5rem;
    }
    
    .section-header h2 {
        font-size: 1.8rem;
    }
    
    .section-header p {
        color: #cbd5e1;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .subsection-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #2acaea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    .subsection-header h3 {
        color: #2acaea;
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    /* ========== KPI CARDS ========== */
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 5px solid #2acaea;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(42, 202, 234, 0.1) 0%, transparent 70%);
        transition: all 0.5s;
        pointer-events: none;
    }
    
    .kpi-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 24px rgba(42, 202, 234, 0.5);
        border-left-color: #60a5fa;
    }
    
    .kpi-card:hover::before {
        top: -25%;
        right: -25%;
    }
    
    .kpi-title {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
        position: relative;
        z-index: 1;
    }
    
    .kpi-value {
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        line-height: 1;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .kpi-subtitle {
        color: #cbd5e1;
        font-size: 0.875rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
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
    
    /* ========== INSIGHT CARDS ========== */
    .insight-card {
        background: rgba(45, 125, 210, 0.1);
        border-left: 4px solid #2d7dd2;
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        transition: all 0.3s;
        backdrop-filter: blur(10px);
    }
    
    .insight-card:hover {
        background: rgba(45, 125, 210, 0.15);
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(42, 202, 234, 0.2);
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
    
    .insight-card.success h4 {
        color: #2dd4a3;
    }
    
    .insight-card.warning h4 {
        color: #fbbf24;
    }
    
    .insight-card.danger h4 {
        color: #ff4444;
    }
    
    .insight-card p {
        color: #cbd5e1;
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* ========== FILTER BADGE ========== */
    .filter-badge {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 30px;
        display: inline-block;
        margin: 1rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(42, 202, 234, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 4px 12px rgba(42, 202, 234, 0.4);
        }
        50% {
            box-shadow: 0 4px 20px rgba(42, 202, 234, 0.6);
        }
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(42, 202, 234, 0.3);
        letter-spacing: 0.025em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 8px 20px rgba(42, 202, 234, 0.5);
        background: linear-gradient(135deg, #3d8de2 0%, #3adcfa 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(1);
    }
    
    /* ========== INFO/WARNING/SUCCESS BOXES ========== */
    .success-box {
        background: linear-gradient(135deg, rgba(45, 212, 163, 0.15) 0%, rgba(45, 212, 163, 0.05) 100%);
        border-left: 5px solid #2dd4a3;
        padding: 1.25rem;
        border-radius: 12px;
        color: #cbd5e1;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(45, 212, 163, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(251, 191, 36, 0.05) 100%);
        border-left: 5px solid #fbbf24;
        padding: 1.25rem;
        border-radius: 12px;
        color: #cbd5e1;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.2);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.15) 0%, rgba(96, 165, 250, 0.05) 100%);
        border-left: 5px solid #60a5fa;
        padding: 1.25rem;
        border-radius: 12px;
        color: #cbd5e1;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(96, 165, 250, 0.2);
    }
    
    .danger-box {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.15) 0%, rgba(255, 68, 68, 0.05) 100%);
        border-left: 5px solid #ff4444;
        padding: 1.25rem;
        border-radius: 12px;
        color: #cbd5e1;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 68, 68, 0.2);
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 58, 95, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(42, 202, 234, 0.1);
        color: #2acaea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(42, 202, 234, 0.4);
    }
    
    /* ========== METRICS ========== */
    .stMetric {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stMetric label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.875rem !important;
    }
    
    /* ========== DATAFRAME ========== */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* ========== PROGRESS BAR ========== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2d7dd2 0%, #2acaea 100%);
        border-radius: 10px;
    }
    
    /* ========== SIDEBAR ========== */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        background: rgba(45, 125, 210, 0.1);
        border-radius: 8px;
        font-weight: 600;
        color: #2acaea;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(45, 125, 210, 0.2);
    }
    
    /* ========== SCROLLBAR ========== */
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
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #3d8de2 0%, #3adcfa 100%);
    }
    
    /* ========== SPINNER ========== */
    .stSpinner > div {
        border-top-color: #2acaea !important;
    }
    
    /* ========== SELECT BOX ========== */
    .stSelectbox > div > div {
        background-color: rgba(30, 58, 95, 0.5);
        border-radius: 8px;
        color: #f8fafc;
    }
    
    /* ========== MULTISELECT ========== */
    .stMultiSelect > div > div {
        background-color: rgba(30, 58, 95, 0.5);
        border-radius: 8px;
    }
    
    /* ========== TEXT INPUT ========== */
    .stTextInput > div > div > input {
        background-color: rgba(30, 58, 95, 0.5);
        border-radius: 8px;
        color: #f8fafc;
        border: 1px solid rgba(42, 202, 234, 0.3);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2acaea;
        box-shadow: 0 0 0 2px rgba(42, 202, 234, 0.2);
    }
    
    /* ========== SLIDER ========== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #2d7dd2 0%, #2acaea 100%);
    }
    
    /* ========== ANIMATIONS ========== */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .slide-in-right {
        animation: slideInRight 0.5s ease-out;
    }
    
    /* ========== RESPONSIVE ========== */
    @media (max-width: 768px) {
        .section-header h1 {
            font-size: 2rem;
        }
        
        .section-header h2 {
            font-size: 1.5rem;
        }
        
        .kpi-value {
            font-size: 2rem;
        }
    }
</style>
"""

st.set_page_config(
    page_title="PharmaIntelligence Pro ML | Enterprise Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://pharmaintelligence.com/support',
        'Report a bug': 'https://pharmaintelligence.com/bug-report',
        'About': f"""
        ### {Config.APP_NAME} v{Config.VERSION}
        
        Enterprise-Grade Pharmaceutical Analytics
        
        **Features:**
        - Machine Learning Forecasting
        - Advanced Clustering & Segmentation
        - Anomaly Detection
        - Geographic Analysis
        - Competitive Intelligence
        - What-If Simulation
        
        © 2024 PharmaIntelligence Inc.
        """
    }
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
    def calculate_cagr(start_value: float, end_value: float, periods: int) -> float:
        """Calculate Compound Annual Growth Rate"""
        if start_value <= 0 or end_value <= 0 or periods <= 0:
            return np.nan
        return ((end_value / start_value) ** (1 / periods) - 1) * 100
    
    @staticmethod
    def get_color_scale(value: float, min_val: float, max_val: float) -> str:
        """Get color based on value scale"""
        if pd.isna(value):
            return Config.COLOR_INFO
        
        normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        
        if normalized < 0.33:
            return Config.COLOR_DANGER
        elif normalized < 0.67:
            return Config.COLOR_WARNING
        else:
            return Config.COLOR_SUCCESS
    
    @staticmethod
    def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value"""
        if denominator == 0 or pd.isna(numerator) or pd.isna(denominator):
            return default
        return numerator / denominator
    
    @staticmethod
    def create_download_link(df: pd.DataFrame, filename: str, file_format: str = 'csv') -> None:
        """Create download button for dataframe"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_filename = f"{filename}_{timestamp}.{file_format}"
        
        if file_format == 'csv':
            data = df.to_csv(index=False)
            mime = 'text/csv'
        elif file_format == 'json':
            data = df.to_json(orient='records', indent=2)
            mime = 'application/json'
        else:
            data = df.to_csv(index=False)
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
    """Advanced data processing and management"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
    def load_data(file, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load and optimize data from file"""
        try:
            start_time = time.time()
            
            # Determine file type and load
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, nrows=sample_size, low_memory=False)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, nrows=sample_size, engine='openpyxl')
            else:
                st.error("❌ Desteklenmeyen dosya formatı!")
                return None
            
            if df is None or len(df) == 0:
                st.error("❌ Dosya boş veya okunamadı!")
                return None
            
            # Clean column names
            df.columns = DataManager.clean_column_names(df.columns)
            
            # Optimize memory
            df = DataManager.optimize_dataframe(df)
            
            load_time = time.time() - start_time
            
            # Display load statistics
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.success(f"✅ {len(df):,} satır, {len(df.columns)} sütun yüklendi ({load_time:.2f}s, {memory_mb:.1f}MB)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.error(f"Detay: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def clean_column_names(columns: List[str]) -> List[str]:
        """Clean and standardize column names"""
        cleaned = []
        
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
            
            # Clean whitespace
            col = ' '.join(col.split())
            
            # Standardize specific patterns
            original_col = col
            
            # Sales columns
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
            
            # If no match, keep cleaned original
            if col == original_col:
                col = col.strip()
            
            cleaned.append(col)
        
        return cleaned
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage"""
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Optimize categorical columns
            for col in df.select_dtypes(include=['object']).columns:
                num_unique = df[col].nunique()
                num_total = len(df)
                
                if num_unique / num_total < 0.5:
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
                    # For floats, use float32
                    df[col] = df[col].astype(np.float32)
            
            # Clean string columns
            for col in df.select_dtypes(include=['object', 'category']).columns:
                try:
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.strip()
                except:
                    pass
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            savings = original_memory - optimized_memory
            
            if savings > 0:
                st.info(f"💾 Bellek optimizasyonu: {original_memory:.1f}MB → {optimized_memory:.1f}MB (↓{savings:.1f}MB / {savings/original_memory*100:.1f}%)")
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def normalize_country_names(df: pd.DataFrame, country_column: Optional[str] = None) -> pd.DataFrame:
        """Normalize country names for choropleth maps"""
        # Find country column
        if country_column is None:
            for possible_name in ['Country', 'Ülke', 'Ulke', 'country', 'COUNTRY']:
                if possible_name in df.columns:
                    country_column = possible_name
                    break
        
        if country_column is None or country_column not in df.columns:
            return df
        
        # Manual mapping dictionary
        country_mapping = {
            # United States variations
            'USA': 'United States',
            'US': 'United States',
            'U.S.A': 'United States',
            'U.S.A.': 'United States',
            'United States of America': 'United States',
            'U.S': 'United States',
            'Amerika': 'United States',
            
            # United Kingdom variations
            'UK': 'United Kingdom',
            'U.K': 'United Kingdom',
            'U.K.': 'United Kingdom',
            'Great Britain': 'United Kingdom',
            'United Kingdom of Great Britain': 'United Kingdom',
            'Ingiltere': 'United Kingdom',
            
            # UAE variations
            'UAE': 'United Arab Emirates',
            'U.A.E': 'United Arab Emirates',
            'U.A.E.': 'United Arab Emirates',
            'Emirlikleri': 'United Arab Emirates',
            
            # Korea variations
            'S. Korea': 'South Korea',
            'South Korea': 'Korea, Republic of',
            'N. Korea': 'North Korea',
            'North Korea': 'Korea, Democratic People\'s Republic of',
            'Guney Kore': 'Korea, Republic of',
            'Kuzey Kore': 'Korea, Democratic People\'s Republic of',
            
            # Russia variations
            'Russia': 'Russian Federation',
            'Rusya': 'Russian Federation',
            
            # Other common variations
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
            'Ceska': 'Czechia',
            'Holland': 'Netherlands',
            'Hollanda': 'Netherlands',
            'Almanya': 'Germany',
            'Fransa': 'France',
            'Italya': 'Italy',
            'Ispanya': 'Spain',
            'Yunanistan': 'Greece',
            'Turkiye': 'Turkey',
            'Cin': 'China',
            'Japonya': 'Japan',
            'Hindistan': 'India',
            'Brezilya': 'Brazil',
            'Arjantin': 'Argentina',
            'Meksika': 'Mexico',
            'Kanada': 'Canada',
            'Avustralya': 'Australia',
            'Yeni Zelanda': 'New Zealand'
        }
        
        # Apply manual mapping
        df[country_column] = df[country_column].replace(country_mapping)
        
        # Use pycountry if available for fuzzy matching
        if PYCOUNTRY_AVAILABLE:
            def fuzzy_country_match(name):
                if pd.isna(name):
                    return name
                
                # First check if already standardized
                if str(name) in country_mapping.values():
                    return name
                
                try:
                    # Try fuzzy search
                    countries = pycountry.countries.search_fuzzy(str(name))
                    if countries:
                        return countries[0].name
                except:
                    pass
                
                return name
            
            df[country_column] = df[country_column].apply(fuzzy_country_match)
        
        return df
    
    @staticmethod
    def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis with comprehensive feature engineering"""
        try:
            # Find sales columns
            sales_cols = {}
            for col in df.columns:
                if 'Satis_' in col or 'Sales_' in col:
                    # Extract year from column name
                    year_match = re.search(r'(\d{4})', col)
                    if year_match:
                        year = year_match.group(1)
                        sales_cols[year] = col
            
            if not sales_cols:
                st.warning("⚠️ Satış sütunları bulunamadı")
                return df
            
            years = sorted([int(y) for y in sales_cols.keys()])
            
            # ========== PRICE CALCULATION ==========
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
            unit_cols = [col for col in df.columns if 'Birim_' in col or 'Unit' in col]
            
            if not price_cols and sales_cols and unit_cols:
                st.info("💡 Fiyat sütunları oluşturuluyor (Satış / Birim)...")
                for year in sales_cols.keys():
                    sales_col = sales_cols[year]
                    unit_col = f"Birim_{year}" if f"Birim_{year}" in df.columns else None
                    
                    if unit_col is None:
                        # Try alternative names
                        for possible_unit in [f"Unit_{year}", f"Units_{year}", f"Birim{year}"]:
                            if possible_unit in df.columns:
                                unit_col = possible_unit
                                break
                    
                    if unit_col and sales_col in df.columns and unit_col in df.columns:
                        df[f'Ort_Fiyat_{year}'] = np.where(
                            df[unit_col] != 0,
                            df[sales_col] / df[unit_col],
                            np.nan
                        )
            
            # ========== GROWTH RATES (YoY) ==========
            for i in range(1, len(years)):
                prev_year = str(years[i-1])
                curr_year = str(years[i])
                
                if prev_year in sales_cols and curr_year in sales_cols:
                    prev_col = sales_cols[prev_year]
                    curr_col = sales_cols[curr_year]
                    
                    # YoY Growth
                    df[f'Buyume_{prev_year}_{curr_year}'] = (
                        (df[curr_col] - df[prev_col]) / 
                        df[prev_col].replace(0, np.nan)
                    ) * 100
                    
                    # Absolute change
                    df[f'Degisim_{prev_year}_{curr_year}'] = df[curr_col] - df[prev_col]
            
            # ========== PRICE ANALYSIS ==========
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
            
            if price_cols:
                # Average price across years
                df['Ort_Fiyat_Genel'] = df[price_cols].mean(axis=1, skipna=True)
                
                # Price volatility (std dev)
                df['Fiyat_Volatilite'] = df[price_cols].std(axis=1, skipna=True)
                
                # Price trend (simple linear regression slope)
                if len(price_cols) >= 2:
                    def calculate_price_trend(row):
                        prices = row[price_cols].dropna()
                        if len(prices) < 2:
                            return np.nan
                        x = np.arange(len(prices))
                        y = prices.values
                        slope = np.polyfit(x, y, 1)[0]
                        return slope
                    
                    df['Fiyat_Trend'] = df.apply(calculate_price_trend, axis=1)
            
            # ========== CAGR ==========
            if len(years) >= 2:
                first_year = str(years[0])
                last_year = str(years[-1])
                
                if first_year in sales_cols and last_year in sales_cols:
                    num_years = len(years) - 1
                    df['CAGR'] = (
                        (df[sales_cols[last_year]] / df[sales_cols[first_year]].replace(0, np.nan))
                        ** (1/num_years) - 1
                    ) * 100
            
            # ========== MARKET SHARE ==========
            if years:
                last_year = str(years[-1])
                if last_year in sales_cols:
                    last_sales_col = sales_cols[last_year]
                    total_sales = df[last_sales_col].sum()
                    
                    if total_sales > 0:
                        df['Pazar_Payi'] = (df[last_sales_col] / total_sales) * 100
                        
                        # Cumulative market share
                        df_sorted = df.sort_values(last_sales_col, ascending=False).reset_index(drop=True)
                        df_sorted['Kumulatif_Pazar_Payi'] = df_sorted['Pazar_Payi'].cumsum()
                        df = df.merge(
                            df_sorted[['Kumulatif_Pazar_Payi']],
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
            
            # ========== SALES MOMENTUM ==========
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                # Average growth momentum
                df['Momentum_Ortalama'] = df[growth_cols].mean(axis=1, skipna=True)
                
                # Growth acceleration (change in growth rate)
                if len(growth_cols) >= 2:
                    df['Ivmelenme'] = df[growth_cols[-1]] - df[growth_cols[-2]]
            
            # ========== VOLUME ANALYSIS ==========
            unit_cols = [col for col in df.columns if 'Birim_' in col or 'Unit' in col]
            if unit_cols:
                # Total volume
                df['Toplam_Hacim'] = df[unit_cols].sum(axis=1, skipna=True)
                
                # Volume growth
                if len(unit_cols) >= 2:
                    df['Hacim_Buyume'] = (
                        (df[unit_cols[-1]] - df[unit_cols[-2]]) / 
                        df[unit_cols[-2]].replace(0, np.nan)
                    ) * 100
            
            # ========== PERFORMANCE SCORE ==========
            # Composite score based on multiple factors
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 3:
                try:
                    scaler = StandardScaler()
                    
                    # Select key metrics for performance score
                    score_features = []
                    
                    if 'CAGR' in df.columns:
                        score_features.append('CAGR')
                    if growth_cols:
                        score_features.append(growth_cols[-1])
                    if 'Pazar_Payi' in df.columns:
                        score_features.append('Pazar_Payi')
                    if price_cols:
                        score_features.append(price_cols[-1])
                    
                    if score_features:
                        score_data = df[score_features].fillna(0)
                        scaled_scores = scaler.fit_transform(score_data)
                        df['Performans_Skoru'] = scaled_scores.mean(axis=1)
                        
                        # Normalize to 0-100 scale
                        score_min = df['Performans_Skoru'].min()
                        score_max = df['Performans_Skoru'].max()
                        if score_max != score_min:
                            df['Performans_Skoru_100'] = (
                                (df['Performans_Skoru'] - score_min) / 
                                (score_max - score_min)
                            ) * 100
                except Exception as e:
                    st.warning(f"⚠️ Performans skoru hesaplanamadı: {str(e)}")
            
            # ========== CLASSIFICATION LABELS ==========
            # Growth classification
            if growth_cols:
                last_growth = growth_cols[-1]
                df['Buyume_Kategori'] = pd.cut(
                    df[last_growth],
                    bins=[-np.inf, -10, 0, 10, 20, np.inf],
                    labels=['Ciddi Dusuş', 'Dusuş', 'Stabil', 'Buyume', 'Yuksek Buyume']
                )
            
            # Price tier classification
            if price_cols:
                last_price = price_cols[-1]
                df['Fiyat_Tier'] = pd.qcut(
                    df[last_price].dropna(),
                    q=5,
                    labels=['Ekonomi', 'Dusuk', 'Orta', 'Yuksek', 'Premium'],
                    duplicates='drop'
                )
            
            # Market share classification
            if 'Pazar_Payi' in df.columns:
                df['Pazar_Pozisyon'] = pd.cut(
                    df['Pazar_Payi'],
                    bins=[0, 0.1, 0.5, 1, 5, 100],
                    labels=['Niche', 'Kucuk', 'Orta', 'Buyuk', 'Lider']
                )
            
            st.success(f"✅ {len([c for c in df.columns if c not in df.columns[:len(df.columns)]])} yeni özellik oluşturuldu")
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Veri hazırlama hatası: {str(e)}")
            st.error(traceback.format_exc())
            return df

# ============================================================================
# FILTER SYSTEM CLASS
# ============================================================================

class FilterSystem:
    """Advanced filtering with search capabilities"""
    
    @staticmethod
    def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column by possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    @staticmethod
    def searchable_multiselect(
        label: str,
        options: List,
        key: str,
        default_all: bool = True,
        help_text: Optional[str] = None
    ) -> List:
        """Create searchable multiselect widget"""
        if not options:
            return []
        
        all_options = ["Tümü"] + list(options)
        
        # Search box
        search_query = st.text_input(
            f"🔍 {label} Ara",
            key=f"{key}_search",
            placeholder="Filtrele...",
            help="Aramak istediğiniz değeri yazın"
        )
        
        # Filter options based on search
        if search_query:
            filtered_options = ["Tümü"] + [
                opt for opt in options 
                if search_query.lower() in str(opt).lower()
            ]
        else:
            filtered_options = all_options
        
        # Default selection
        if default_all:
            default = ["Tümü"]
        else:
            default = []
        
        # Multiselect
        selected = st.multiselect(
            label,
            options=filtered_options,
            default=default,
            key=key,
            help=help_text
        )
        
        # Handle "Tümü" selection
        if "Tümü" in selected:
            if len(selected) > 1:
                # If other items selected, remove "Tümü"
                selected = [opt for opt in selected if opt != "Tümü"]
            else:
                # If only "Tümü", return all options
                selected = list(options)
        
        # Display selection count
        if selected:
            if len(selected) == len(options):
                st.caption(f"✅ TÜMÜ seçildi ({len(options)} öğe)")
            else:
                st.caption(f"✅ {len(selected)} / {len(options)} seçildi")
        
        return selected
    
    @staticmethod
    def create_filter_sidebar(df: pd.DataFrame) -> Tuple:
        """Create comprehensive filter sidebar"""
        with st.sidebar.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
            st.markdown("""
            <div class="subsection-header">
                <h3>🔍 Filtreleme Paneli</h3>
            </div>
            """, unsafe_allow_html=True)
            
            filters = {}
            
            # ========== GLOBAL SEARCH ==========
            st.markdown("#### 🔎 Genel Arama")
            search_term = st.text_input(
                "Tüm sütunlarda ara",
                placeholder="Molekül, şirket, ülke vb...",
                help="Arama terimi tüm text sütunlarında aranacak",
                key="global_search"
            )
            
            st.markdown("---")
            
            # ========== CATEGORICAL FILTERS ==========
            st.markdown("#### 📑 Kategorik Filtreler")
            
            # Country filter
            country_col = FilterSystem.find_column(df, ['Country', 'Ülke', 'Ulke', 'COUNTRY'])
            if country_col:
                countries = sorted(df[country_col].dropna().unique())
                selected_countries = FilterSystem.searchable_multiselect(
                    "🌍 Ülkeler",
                    countries,
                    key="country_filter",
                    help_text=f"Toplam {len(countries)} ülke"
                )
                if selected_countries and len(selected_countries) < len(countries):
                    filters[country_col] = selected_countries
            
            # Corporation filter
            corp_col = FilterSystem.find_column(df, ['Corporation', 'Şirket', 'Sirket', 'Company'])
            if corp_col:
                corporations = sorted(df[corp_col].dropna().unique())
                selected_corps = FilterSystem.searchable_multiselect(
                    "🏢 Şirketler",
                    corporations,
                    key="corp_filter",
                    help_text=f"Toplam {len(corporations)} şirket"
                )
                if selected_corps and len(selected_corps) < len(corporations):
                    filters[corp_col] = selected_corps
            
            # Molecule filter
            mol_col = FilterSystem.find_column(df, ['Molecule', 'Molekül', 'Molekul', 'Product'])
            if mol_col:
                molecules = sorted(df[mol_col].dropna().unique())
                selected_mols = FilterSystem.searchable_multiselect(
                    "🧪 Moleküller",
                    molecules,
                    key="mol_filter",
                    help_text=f"Toplam {len(molecules)} molekül"
                )
                if selected_mols and len(selected_mols) < len(molecules):
                    filters[mol_col] = selected_mols
            
            # Sector filter (if exists)
            sector_col = FilterSystem.find_column(df, ['Sector', 'Sektor', 'Category'])
            if sector_col:
                sectors = sorted(df[sector_col].dropna().unique())
                selected_sectors = FilterSystem.searchable_multiselect(
                    "🏭 Sektörler",
                    sectors,
                    key="sector_filter",
                    help_text=f"Toplam {len(sectors)} sektör"
                )
                if selected_sectors and len(selected_sectors) < len(sectors):
                    filters[sector_col] = selected_sectors
            
            st.markdown("---")
            
            # ========== NUMERICAL FILTERS ==========
            st.markdown("#### 📊 Sayısal Filtreler")
            
            # Sales filter
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            if sales_cols:
                last_sales_col = sales_cols[-1]
                sales_data = df[last_sales_col].dropna()
                
                if len(sales_data) > 0:
                    min_sales = float(sales_data.min())
                    max_sales = float(sales_data.max())
                    
                    st.write(f"**Satış Aralığı:** {Utils.format_number(min_sales, '$')} - {Utils.format_number(max_sales, '$')}")
                    
                    sales_range = st.slider(
                        "Satış Filtresi",
                        min_value=min_sales,
                        max_value=max_sales,
                        value=(min_sales, max_sales),
                        format="$%.0f",
                        help="Satış değeri bu aralıkta olan ürünleri filtrele",
                        key="sales_filter"
                    )
                    
                    if sales_range != (min_sales, max_sales):
                        filters['sales_range'] = (sales_range, last_sales_col)
            
            # Growth filter
            growth_cols = [col for col in df.columns if 'Buyume_' in col or 'Growth_' in col]
            if growth_cols:
                last_growth_col = growth_cols[-1]
                growth_data = df[last_growth_col].dropna()
                
                if len(growth_data) > 0:
                    min_growth = float(growth_data.min())
                    max_growth = float(growth_data.max())
                    
                    st.write(f"**Büyüme Aralığı:** {Utils.format_percentage(min_growth)} - {Utils.format_percentage(max_growth)}")
                    
                    growth_range = st.slider(
                        "Büyüme Filtresi (%)",
                        min_value=min_growth,
                        max_value=max_growth,
                        value=(min(min_growth, -50.0), max(max_growth, 150.0)),
                        format="%.1f%%",
                        help="Büyüme oranı bu aralıkta olan ürünleri filtrele",
                        key="growth_filter"
                    )
                    
                    if growth_range != (min_growth, max_growth):
                        filters['growth_range'] = (growth_range, last_growth_col)
            
            # Price filter
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
            if price_cols:
                last_price_col = price_cols[-1]
                price_data = df[last_price_col].dropna()
                
                if len(price_data) > 0:
                    min_price = float(price_data.min())
                    max_price = float(price_data.max())
                    
                    st.write(f"**Fiyat Aralığı:** ${min_price:.2f} - ${max_price:.2f}")
                    
                    price_range = st.slider(
                        "Fiyat Filtresi",
                        min_value=min_price,
                        max_value=max_price,
                        value=(min_price, max_price),
                        format="$%.2f",
                        help="Fiyat bu aralıkta olan ürünleri filtrele",
                        key="price_filter"
                    )
                    
                    if price_range != (min_price, max_price):
                        filters['price_range'] = (price_range, last_price_col)
            
            # Market share filter
            if 'Pazar_Payi' in df.columns:
                share_data = df['Pazar_Payi'].dropna()
                
                if len(share_data) > 0:
                    min_share = float(share_data.min())
                    max_share = float(share_data.max())
                    
                    st.write(f"**Pazar Payı:** {Utils.format_percentage(min_share)} - {Utils.format_percentage(max_share)}")
                    
                    share_range = st.slider(
                        "Pazar Payı Filtresi (%)",
                        min_value=min_share,
                        max_value=max_share,
                        value=(min_share, max_share),
                        format="%.2f%%",
                        help="Pazar payı bu aralıkta olan ürünleri filtrele",
                        key="share_filter"
                    )
                    
                    if share_range != (min_share, max_share):
                        filters['share_range'] = share_range
            
            st.markdown("---")
            
            # ========== ADVANCED FILTERS ==========
            st.markdown("#### ⚙️ Gelişmiş Filtreler")
            
            col1, col2 = st.columns(2)
            
            with col1:
                positive_growth_only = st.checkbox(
                    "📈 Pozitif Büyüme",
                    value=False,
                    help="Sadece büyüyen ürünleri göster"
                )
                if positive_growth_only and growth_cols:
                    filters['positive_growth'] = True
                
                high_growth_only = st.checkbox(
                    "🚀 Yüksek Büyüme (>20%)",
                    value=False,
                    help="20% üzeri büyüyen ürünleri göster"
                )
                if high_growth_only and growth_cols:
                    filters['high_growth'] = True
            
            with col2:
                top_performers_only = st.checkbox(
                    "🏆 Top Performerlar",
                    value=False,
                    help="En yüksek satışa sahip ürünler"
                )
                if top_performers_only:
                    filters['top_performers'] = True
                
                exclude_outliers = st.checkbox(
                    "🎯 Outlier Hariç",
                    value=False,
                    help="Aşırı uç değerleri çıkar"
                )
                if exclude_outliers:
                    filters['exclude_outliers'] = True
            
            st.markdown("---")
            
            # ========== CLASSIFICATION FILTERS ==========
            if 'Buyume_Kategori' in df.columns:
                st.markdown("#### 🏷️ Kategori Filtreleri")
                
                growth_categories = df['Buyume_Kategori'].dropna().unique()
                selected_growth_cats = st.multiselect(
                    "📊 Büyüme Kategorisi",
                    options=sorted(growth_categories),
                    default=sorted(growth_categories),
                    key="growth_cat_filter"
                )
                if selected_growth_cats and len(selected_growth_cats) < len(growth_categories):
                    filters['growth_category'] = selected_growth_cats
            
            if 'Fiyat_Tier' in df.columns:
                price_tiers = df['Fiyat_Tier'].dropna().unique()
                selected_price_tiers = st.multiselect(
                    "💰 Fiyat Segmenti",
                    options=sorted(price_tiers),
                    default=sorted(price_tiers),
                    key="price_tier_filter"
                )
                if selected_price_tiers and len(selected_price_tiers) < len(price_tiers):
                    filters['price_tier'] = selected_price_tiers
            
            if 'Pazar_Pozisyon' in df.columns:
                market_positions = df['Pazar_Pozisyon'].dropna().unique()
                selected_positions = st.multiselect(
                    "🎯 Pazar Pozisyonu",
                    options=sorted(market_positions),
                    default=sorted(market_positions),
                    key="position_filter"
                )
                if selected_positions and len(selected_positions) < len(market_positions):
                    filters['market_position'] = selected_positions
            
            st.markdown("---")
            
            # ========== ACTION BUTTONS ==========
            col1, col2 = st.columns(2)
            
            with col1:
                apply_filters = st.button(
                    "✅ Filtreleri Uygula",
                    use_container_width=True,
                    type="primary",
                    help="Seçili filtreleri uygula"
                )
            
            with col2:
                clear_filters = st.button(
                    "🗑️ Filtreleri Temizle",
                    use_container_width=True,
                    help="Tüm filtreleri sıfırla"
                )
        
        return search_term, filters, apply_filters, clear_filters
    
    @staticmethod
    def apply_filters(
        df: pd.DataFrame,
        search_term: str,
        filters: Dict
    ) -> pd.DataFrame:
        """Apply all filters to dataframe"""
        filtered_df = df.copy()
        original_count = len(filtered_df)
        
        # ========== GLOBAL SEARCH ==========
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
            st.info(f"🔍 Arama: '{search_term}' → {len(filtered_df):,} / {original_count:,} satır")
        
        # ========== CATEGORICAL FILTERS ==========
        for col, values in filters.items():
            if col in ['sales_range', 'growth_range', 'price_range', 'share_range',
                      'positive_growth', 'high_growth', 'top_performers', 'exclude_outliers',
                      'growth_category', 'price_tier', 'market_position']:
                continue
            
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        # ========== NUMERICAL RANGE FILTERS ==========
        if 'sales_range' in filters:
            (min_val, max_val), col_name = filters['sales_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        if 'growth_range' in filters:
            (min_val, max_val), col_name = filters['growth_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        if 'price_range' in filters:
            (min_val, max_val), col_name = filters['price_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        if 'share_range' in filters:
            (min_val, max_val) = filters['share_range']
            if 'Pazar_Payi' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['Pazar_Payi'] >= min_val) & 
                    (filtered_df['Pazar_Payi'] <= max_val)
                ]
        
        # ========== BOOLEAN FILTERS ==========
        if filters.get('positive_growth', False):
            growth_cols = [col for col in filtered_df.columns if 'Buyume_' in col or 'Growth_' in col]
            if growth_cols:
                filtered_df = filtered_df[filtered_df[growth_cols[-1]] > 0]
        
        if filters.get('high_growth', False):
            growth_cols = [col for col in filtered_df.columns if 'Buyume_' in col or 'Growth_' in col]
            if growth_cols:
                filtered_df = filtered_df[filtered_df[growth_cols[-1]] > 20]
        
        if filters.get('top_performers', False):
            sales_cols = [col for col in filtered_df.columns if 'Satis_' in col or 'Sales_' in col]
            if sales_cols:
                # Top 10% by sales
                threshold = filtered_df[sales_cols[-1]].quantile(0.9)
                filtered_df = filtered_df[filtered_df[sales_cols[-1]] >= threshold]
        
        if filters.get('exclude_outliers', False):
            # Remove outliers using IQR method
            sales_cols = [col for col in filtered_df.columns if 'Satis_' in col or 'Sales_' in col]
            if sales_cols:
                Q1 = filtered_df[sales_cols[-1]].quantile(0.25)
                Q3 = filtered_df[sales_cols[-1]].quantile(0.75)
                IQR = Q3 - Q1
                filtered_df = filtered_df[
                    (filtered_df[sales_cols[-1]] >= Q1 - 1.5 * IQR) &
                    (filtered_df[sales_cols[-1]] <= Q3 + 1.5 * IQR)
                ]
        
        # ========== CATEGORY FILTERS ==========
        if 'growth_category' in filters and filters['growth_category']:
            if 'Buyume_Kategori' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Buyume_Kategori'].isin(filters['growth_category'])]
        
        if 'price_tier' in filters and filters['price_tier']:
            if 'Fiyat_Tier' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Fiyat_Tier'].isin(filters['price_tier'])]
        
        if 'market_position' in filters and filters['market_position']:
            if 'Pazar_Pozisyon' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Pazar_Pozisyon'].isin(filters['market_position'])]
        
        return filtered_df

# Due to length constraints, I'll continue in the next message with:
# - Analytics Engine
# - ML Engine
# - Visualization Engine
# - Main Application and all Tabs

# Please let me know when ready for the continuation!

# ============================================================================
# ANALYTICS ENGINE CLASS
# ============================================================================

class AnalyticsEngine:
    """Comprehensive pharmaceutical analytics"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
    def calculate_comprehensive_metrics(df: pd.DataFrame) -> Dict:
        """Calculate all market metrics"""
        metrics = {}
        
        try:
            # Basic metrics
            metrics['total_rows'] = len(df)
            metrics['total_columns'] = len(df.columns)
            metrics['memory_mb'] = df.memory_usage(deep=True).sum() / 1024**2
            
            # Sales metrics
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            
            if sales_cols:
                last_sales_col = sales_cols[-1]
                metrics['last_year'] = re.search(r'(\d{4})', last_sales_col).group(1) if re.search(r'(\d{4})', last_sales_col) else 'N/A'
                metrics['total_market_value'] = df[last_sales_col].sum()
                metrics['avg_sales'] = df[last_sales_col].mean()
                metrics['median_sales'] = df[last_sales_col].median()
                metrics['sales_std'] = df[last_sales_col].std()
                metrics['sales_q1'] = df[last_sales_col].quantile(0.25)
                metrics['sales_q3'] = df[last_sales_col].quantile(0.75)
                metrics['sales_iqr'] = metrics['sales_q3'] - metrics['sales_q1']
                metrics['sales_cv'] = (metrics['sales_std'] / metrics['avg_sales']) * 100 if metrics['avg_sales'] > 0 else 0
            
            # Growth metrics
            growth_cols = [col for col in df.columns if 'Buyume_' in col or 'Growth_' in col]
            
            if growth_cols:
                last_growth_col = growth_cols[-1]
                metrics['avg_growth'] = df[last_growth_col].mean()
                metrics['median_growth'] = df[last_growth_col].median()
                metrics['growth_std'] = df[last_growth_col].std()
                metrics['positive_growth_count'] = (df[last_growth_col] > 0).sum()
                metrics['negative_growth_count'] = (df[last_growth_col] < 0).sum()
                metrics['high_growth_count'] = (df[last_growth_col] > 20).sum()
                metrics['decline_count'] = (df[last_growth_col] < -10).sum()
                metrics['positive_growth_pct'] = (metrics['positive_growth_count'] / len(df)) * 100
            
            # Corporation analysis
            corp_col = FilterSystem.find_column(df, ['Corporation', 'Şirket'])
            
            if corp_col and sales_cols:
                corp_sales = df.groupby(corp_col)[last_sales_col].sum().sort_values(ascending=False)
                total_sales = corp_sales.sum()
                
                if total_sales > 0:
                    # Market shares
                    market_shares = (corp_sales / total_sales * 100)
                    
                    # HHI (Herfindahl-Hirschman Index)
                    metrics['hhi_index'] = (market_shares ** 2).sum()
                    
                    # Top N shares
                    for n in [1, 3, 5, 10]:
                        if len(corp_sales) >= n:
                            metrics[f'top_{n}_share'] = corp_sales.nlargest(n).sum() / total_sales * 100
                    
                    # Concentration ratio
                    metrics['cr4'] = corp_sales.nlargest(4).sum() / total_sales * 100 if len(corp_sales) >= 4 else 100
                    
                    # Number of competitors
                    metrics['num_competitors'] = len(corp_sales)
                    
                    # Effective number of competitors
                    metrics['effective_competitors'] = 10000 / metrics['hhi_index'] if metrics['hhi_index'] > 0 else 0
            
            # Molecule diversity
            mol_col = FilterSystem.find_column(df, ['Molecule', 'Molekül'])
            
            if mol_col:
                metrics['unique_molecules'] = df[mol_col].nunique()
                
                if sales_cols:
                    mol_sales = df.groupby(mol_col)[last_sales_col].sum()
                    total_mol_sales = mol_sales.sum()
                    
                    if total_mol_sales > 0:
                        metrics['top_10_molecule_share'] = mol_sales.nlargest(10).sum() / total_mol_sales * 100
                        metrics['top_molecule'] = mol_sales.idxmax()
                        metrics['top_molecule_share'] = (mol_sales.max() / total_mol_sales) * 100
            
            # Geographic metrics
            country_col = FilterSystem.find_column(df, ['Country', 'Ülke'])
            
            if country_col:
                metrics['country_coverage'] = df[country_col].nunique()
                
                if sales_cols:
                    country_sales = df.groupby(country_col)[last_sales_col].sum()
                    total_country_sales = country_sales.sum()
                    
                    if total_country_sales > 0:
                        metrics['top_5_country_share'] = country_sales.nlargest(5).sum() / total_country_sales * 100
                        metrics['top_country'] = country_sales.idxmax()
                        metrics['top_country_share'] = (country_sales.max() / total_country_sales) * 100
            
            # Price metrics
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
            
            if price_cols:
                last_price_col = price_cols[-1]
                metrics['avg_price'] = df[last_price_col].mean()
                metrics['median_price'] = df[last_price_col].median()
                metrics['price_std'] = df[last_price_col].std()
                metrics['price_variance'] = df[last_price_col].var()
                metrics['price_cv'] = (metrics['price_std'] / metrics['avg_price']) * 100 if metrics['avg_price'] > 0 else 0
                
                price_quartiles = df[last_price_col].quantile([0.25, 0.5, 0.75])
                metrics['price_q1'] = price_quartiles[0.25]
                metrics['price_median'] = price_quartiles[0.5]
                metrics['price_q3'] = price_quartiles[0.75]
            
            # CAGR metrics
            if 'CAGR' in df.columns:
                metrics['avg_cagr'] = df['CAGR'].mean()
                metrics['median_cagr'] = df['CAGR'].median()
                metrics['positive_cagr_count'] = (df['CAGR'] > 0).sum()
            
            # Market share metrics
            if 'Pazar_Payi' in df.columns:
                metrics['avg_market_share'] = df['Pazar_Payi'].mean()
                metrics['market_share_gini'] = AnalyticsEngine.calculate_gini_coefficient(df['Pazar_Payi'])
            
            # International products
            if mol_col and corp_col and sales_cols:
                intl_count = 0
                intl_sales = 0
                
                for molecule in df[mol_col].unique():
                    mol_df = df[df[mol_col] == molecule]
                    unique_corps = mol_df[corp_col].nunique()
                    
                    if unique_corps > 1:
                        intl_count += 1
                        intl_sales += mol_df[last_sales_col].sum()
                
                metrics['intl_product_count'] = intl_count
                metrics['intl_product_sales'] = intl_sales
                
                if metrics.get('total_market_value', 0) > 0:
                    metrics['intl_product_share'] = (intl_sales / metrics['total_market_value']) * 100
            
            # Performance distribution
            if 'Performans_Skoru_100' in df.columns:
                metrics['avg_performance_score'] = df['Performans_Skoru_100'].mean()
                metrics['high_performers'] = (df['Performans_Skoru_100'] > 75).sum()
                metrics['low_performers'] = (df['Performans_Skoru_100'] < 25).sum()
            
            return metrics
            
        except Exception as e:
            st.warning(f"⚠️ Metrik hesaplama hatası: {str(e)}")
            return metrics
    
    @staticmethod
    def calculate_gini_coefficient(data: pd.Series) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        try:
            data = data.dropna()
            if len(data) == 0:
                return 0.0
            
            sorted_data = np.sort(data)
            n = len(sorted_data)
            cumsum = np.cumsum(sorted_data)
            
            return (2 * np.sum((np.arange(1, n+1)) * sorted_data)) / (n * cumsum[-1]) - (n + 1) / n
            
        except:
            return 0.0
    
    @staticmethod
    def generate_strategic_insights(df: pd.DataFrame, metrics: Dict) -> List[Dict]:
        """Generate strategic business insights"""
        insights = []
        
        try:
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            if not sales_cols:
                return insights
            
            last_sales_col = sales_cols[-1]
            year = metrics.get('last_year', 'N/A')
            
            # ========== TOP PRODUCTS INSIGHT ==========
            top_products = df.nlargest(10, last_sales_col)
            top_share = (top_products[last_sales_col].sum() / df[last_sales_col].sum() * 100)
            
            insights.append({
                'type': 'success',
                'title': f'🏆 Top 10 Ürün Konsantrasyonu - {year}',
                'description': f"En çok satan 10 ürün toplam pazarın %{top_share:.1f}'ini oluşturuyor. " +
                              f"Ortalama satış: {Utils.format_number(top_products[last_sales_col].mean(), '$')}"
            })
            
            # ========== GROWTH LEADERS ==========
            growth_cols = [col for col in df.columns if 'Buyume_' in col or 'Growth_' in col]
            if growth_cols:
                last_growth_col = growth_cols[-1]
                top_growth = df.nlargest(10, last_growth_col)
                avg_top_growth = top_growth[last_growth_col].mean()
                
                insights.append({
                    'type': 'info',
                    'title': f'🚀 Büyüme Liderleri',
                    'description': f"En hızlı büyüyen 10 ürün ortalama %{avg_top_growth:.1f} büyüme gösteriyor. " +
                                  f"Pazar ortalaması: %{metrics.get('avg_growth', 0):.1f}"
                })
                
                # Decline warning
                decline_count = metrics.get('decline_count', 0)
                if decline_count > 0:
                    decline_pct = (decline_count / len(df)) * 100
                    insights.append({
                        'type': 'warning',
                        'title': '⚠️ Düşüş Trendi',
                        'description': f"{decline_count:,} ürün (% {decline_pct:.1f}) %10'dan fazla düşüş gösteriyor. " +
                                      "Risk analizi önerilir."
                    })
            
            # ========== MARKET LEADER ==========
            corp_col = FilterSystem.find_column(df, ['Corporation', 'Şirket'])
            if corp_col:
                top_corps = df.groupby(corp_col)[last_sales_col].sum().nlargest(5)
                leader = top_corps.index[0]
                leader_share = (top_corps.iloc[0] / df[last_sales_col].sum()) * 100
                
                insights.append({
                    'type': 'warning',
                    'title': f'🏢 Pazar Lideri - {year}',
                    'description': f"{leader} %{leader_share:.1f} pazar payı ile lider konumda. " +
                                  f"CR4: %{metrics.get('cr4', 0):.1f}"
                })
            
            # ========== MARKET STRUCTURE ==========
            hhi = metrics.get('hhi_index', 0)
            if hhi > 0:
                if hhi > 2500:
                    structure = "Yüksek Konsantrasyon (Monopol/Oligopol)"
                    risk = "Yüksek giriş bariyeri, rekabet riski düşük"
                elif hhi > 1800:
                    structure = "Orta Konsantrasyon (Oligopol)"
                    risk = "Dengeli rekabet ortamı"
                else:
                    structure = "Düşük Konsantrasyon (Rekabetçi)"
                    risk = "Yoğun rekabet, fiyat baskısı riski"
                
                insights.append({
                    'type': 'info',
                    'title': '📊 Pazar Yapısı Analizi',
                    'description': f"HHI: {hhi:.0f} - {structure}. {risk}. " +
                                  f"Efektif rakip sayısı: {metrics.get('effective_competitors', 0):.1f}"
                })
            
            # ========== GEOGRAPHIC INSIGHT ==========
            country_col = FilterSystem.find_column(df, ['Country', 'Ülke'])
            if country_col:
                top_country = metrics.get('top_country', 'N/A')
                country_share = metrics.get('top_country_share', 0)
                
                insights.append({
                    'type': 'geographic',
                    'title': f'🌍 Coğrafi Lider - {year}',
                    'description': f"{top_country} %{country_share:.1f} pay ile en büyük pazar. " +
                                  f"Toplam {metrics.get('country_coverage', 0)} ülkede faaliyet."
                })
            
            # ========== PRICE ANALYSIS ==========
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
            if price_cols:
                avg_price = metrics.get('avg_price', 0)
                price_cv = metrics.get('price_cv', 0)
                
                price_tier = "Yüksek" if avg_price > 100 else "Orta" if avg_price > 10 else "Düşük"
                
                insights.append({
                    'type': 'price',
                    'title': f'💰 Fiyatlandırma Profili - {year}',
                    'description': f"Ortalama fiyat: ${avg_price:.2f} ({price_tier} segment). " +
                                  f"Fiyat varyasyon katsayısı: %{price_cv:.1f}"
                })
            
            # ========== INTERNATIONAL PRODUCTS ==========
            intl_count = metrics.get('intl_product_count', 0)
            if intl_count > 0:
                intl_share = metrics.get('intl_product_share', 0)
                
                insights.append({
                    'type': 'success',
                    'title': '🌐 International Product Analizi',
                    'description': f"{intl_count} ürün çoklu pazar/şirket yapısında. " +
                                  f"Toplam satış içindeki payı: %{intl_share:.1f}"
                })
            
            # ========== OPPORTUNITY INSIGHT ==========
            if growth_cols and 'Pazar_Payi' in df.columns:
                # High growth + low market share = opportunity
                last_growth_col = growth_cols[-1]
                opportunities = df[
                    (df[last_growth_col] > 20) & 
                    (df['Pazar_Payi'] < 1)
                ]
                
                if len(opportunities) > 0:
                    insights.append({
                        'type': 'success',
                        'title': '💎 Büyüme Fırsatları',
                        'description': f"{len(opportunities)} ürün yüksek büyüme + düşük pazar payı " +
                                      "kombinasyonunda. Yatırım fırsatı potansiyeli yüksek."
                    })
            
            # ========== RISK INSIGHT ==========
            if growth_cols and sales_cols:
                # High market share + negative growth = risk
                last_growth_col = growth_cols[-1]
                if 'Pazar_Payi' in df.columns:
                    risks = df[
                        (df[last_growth_col] < -5) & 
                        (df['Pazar_Payi'] > 5)
                    ]
                    
                    if len(risks) > 0:
                        risk_sales = risks[last_sales_col].sum()
                        risk_share = (risk_sales / df[last_sales_col].sum()) * 100
                        
                        insights.append({
                            'type': 'danger',
                            'title': '⚠️ Risk Alanları',
                            'description': f"{len(risks)} büyük ürün düşüş trendinde. " +
                                          f"Risk altındaki satış: {Utils.format_number(risk_sales, '$')} (%{risk_share:.1f})"
                        })
            
            return insights
            
        except Exception as e:
            st.warning(f"⚠️ İçgörü oluşturma hatası: {str(e)}")
            return insights

# ============================================================================
# MACHINE LEARNING ENGINE CLASS (Full Implementation)
# ============================================================================

class MLEngine:
    """Comprehensive machine learning capabilities"""
    
    @staticmethod
    def train_forecasting_model(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        forecast_years: int = 2,
        model_type: str = 'rf'
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Train forecasting model with multiple algorithms"""
        try:
            # Prepare data
            ml_data = df[feature_cols + [target_col]].dropna()
            
            if len(ml_data) < Config.ML_MIN_SAMPLES:
                return None, f"Yetersiz veri: {len(ml_data)} satır (minimum {Config.ML_MIN_SAMPLES} gerekli)"
            
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
                    min_samples_split=5,
                    random_state=Config.RANDOM_STATE,
                    n_jobs=Config.N_JOBS
                )
            elif model_type == 'gbm':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=Config.RANDOM_STATE
                )
            elif model_type == 'linear':
                model = Ridge(alpha=1.0, random_state=Config.RANDOM_STATE)
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=Config.RANDOM_STATE
                )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y, cv=min(Config.CV_FOLDS, len(X)//10),
                scoring='r2', n_jobs=Config.N_JOBS
            )
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols, model.feature_importances_))
            else:
                feature_importance = {f: 1/len(feature_cols) for f in feature_cols}
            
            # Future forecast
            forecast = []
            base_year = int(re.search(r'(\d{4})', target_col).group(1)) if re.search(r'(\d{4})', target_col) else 2024
            
            for year_offset in range(1, forecast_years + 1):
                future_year = base_year + year_offset
                
                # Simple projection using mean features
                future_X = X.mean(axis=0).values.reshape(1, -1)
                future_pred = model.predict(future_X)[0]
                
                # Confidence interval (simplified)
                std_error = rmse_test
                confidence_low = future_pred - 1.96 * std_error
                confidence_high = future_pred + 1.96 * std_error
                
                forecast.append({
                    'year': str(future_year),
                    'prediction': future_pred,
                    'confidence_low': max(0, confidence_low),
                    'confidence_high': confidence_high
                })
            
            results = {
                'model': model,
                'model_type': model_type,
                'mae_train': mae_train,
                'mae_test': mae_test,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'forecast': forecast,
                'feature_importance': feature_importance,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'predictions': {
                    'y_test': y_test.values,
                    'y_pred_test': y_pred_test
                }
            }
            
            return results, None
            
        except Exception as e:
            return None, f"Model eğitim hatası: {str(e)}"
    
    @staticmethod
    def perform_clustering(
        df: pd.DataFrame,
        feature_cols: List[str],
        n_clusters: int = 4,
        algorithm: str = 'kmeans'
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Perform clustering with multiple algorithms"""
        try:
            # Prepare data
            cluster_data = df[feature_cols].fillna(0)
            
            min_samples = Config.CLUSTERING_MIN_SAMPLES_PER_CLUSTER * n_clusters
            if len(cluster_data) < min_samples:
                return None, f"Yetersiz veri: {len(cluster_data)} satır (minimum {min_samples} gerekli)"
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Clustering algorithm
            if algorithm == 'kmeans':
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    random_state=Config.RANDOM_STATE,
                    n_init=10,
                    max_iter=300
                )
            elif algorithm == 'dbscan':
                clusterer = DBSCAN(eps=0.5, min_samples=10)
            elif algorithm == 'hierarchical':
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                clusterer = KMeans(n_clusters=n_clusters, random_state=Config.RANDOM_STATE)
            
            # Fit
            clusters = clusterer.fit_predict(scaled_data)
            
            # Metrics
            silhouette = silhouette_score(scaled_data, clusters)
            calinski = calinski_harabasz_score(scaled_data, clusters)
            davies = davies_bouldin_score(scaled_data, clusters)
            
            # PCA for visualization
            n_components = min(3, len(feature_cols))
            pca = PCA(n_components=n_components, random_state=Config.RANDOM_STATE)
            pca_data = pca.fit_transform(scaled_data)
            
            # Cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_mask = clusters == i
                cluster_stats[i] = {
                    'size': cluster_mask.sum(),
                    'percentage': (cluster_mask.sum() / len(clusters)) * 100,
                    'mean_features': df[feature_cols][cluster_mask].mean().to_dict()
                }
            
            # Cluster names
            cluster_names = {
                0: 'Gelişen Ürünler',
                1: 'Olgun Ürünler',
                2: 'Yenilikçi Ürünler',
                3: 'Riskli Ürünler',
                4: 'Niş Ürünler',
                5: 'Hacim Ürünleri',
                6: 'Premium Ürünler',
                7: 'Ekonomi Ürünleri',
                8: 'Büyüme Yıldızları',
                9: 'Düşüş Trendinde'
            }
            
            cluster_labels = [cluster_names.get(c, f'Küme {c}') for c in clusters]
            
            results = {
                'clusters': clusters,
                'cluster_labels': cluster_labels,
                'silhouette_score': silhouette,
                'calinski_score': calinski,
                'davies_bouldin_score': davies,
                'pca_data': pca_data,
                'pca_variance': pca.explained_variance_ratio_,
                'cluster_stats': cluster_stats,
                'n_clusters': len(np.unique(clusters)),
                'algorithm': algorithm,
                'feature_cols': feature_cols
            }
            
            if algorithm == 'kmeans':
                results['centers'] = clusterer.cluster_centers_
                results['inertia'] = clusterer.inertia_
            
            return results, None
            
        except Exception as e:
            return None, f"Kümeleme hatası: {str(e)}"
    
    @staticmethod
    def detect_anomalies(
        df: pd.DataFrame,
        feature_cols: List[str],
        contamination: float = 0.1
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Detect anomalies using Isolation Forest"""
        try:
            # Prepare data
            anomaly_data = df[feature_cols].fillna(0)
            
            if len(anomaly_data) < Config.ML_MIN_SAMPLES:
                return None, f"Yetersiz veri: {len(anomaly_data)} satır (minimum {Config.ML_MIN_SAMPLES} gerekli)"
            
            # Scale features
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
            
            # Identify anomalies
            is_anomaly = predictions == -1
            anomaly_count = is_anomaly.sum()
            anomaly_percentage = (anomaly_count / len(predictions)) * 100
            
            # Anomaly severity (based on score)
            severity = np.where(
                anomaly_scores < np.percentile(anomaly_scores, 5),
                'Kritik',
                np.where(
                    anomaly_scores < np.percentile(anomaly_scores, 10),
                    'Yüksek',
                    'Orta'
                )
            )
            
            results = {
                'is_anomaly': is_anomaly,
                'anomaly_scores': anomaly_scores,
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_percentage,
                'severity': severity,
                'contamination': contamination,
                'feature_cols': feature_cols
            }
            
            return results, None
            
        except Exception as e:
            return None, f"Anomali tespiti hatası: {str(e)}"
    
    @staticmethod
    def calculate_optimal_clusters(
        df: pd.DataFrame,
        feature_cols: List[str],
        max_k: int = 10
    ) -> Optional[Dict]:
        """Calculate optimal number of clusters using elbow and silhouette methods"""
        try:
            cluster_data = df[feature_cols].fillna(0)
            
            if len(cluster_data) < Config.CLUSTERING_MIN_SAMPLES_PER_CLUSTER * Config.MIN_N_CLUSTERS:
                return None
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            max_k = min(max_k, len(cluster_data) // Config.CLUSTERING_MIN_SAMPLES_PER_CLUSTER)
            k_range = range(Config.MIN_N_CLUSTERS, max_k + 1)
            
            inertias = []
            silhouettes = []
            calinski_scores = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_STATE, n_init=10)
                labels = kmeans.fit_predict(scaled_data)
                
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(scaled_data, labels))
                calinski_scores.append(calinski_harabasz_score(scaled_data, labels))
            
            # Find elbow point
            if len(inertias) >= 3:
                # Calculate second derivative
                diffs = np.diff(inertias)
                diffs2 = np.diff(diffs)
                elbow_idx = np.argmax(diffs2) + 2 if len(diffs2) > 0 else 2
                optimal_k_elbow = list(k_range)[elbow_idx] if elbow_idx < len(k_range) else Config.DEFAULT_N_CLUSTERS
            else:
                optimal_k_elbow = Config.DEFAULT_N_CLUSTERS
            
            # Find optimal by silhouette
            optimal_k_silhouette = list(k_range)[np.argmax(silhouettes)]
            
            return {
                'k_values': list(k_range),
                'inertias': inertias,
                'silhouettes': silhouettes,
                'calinski_scores': calinski_scores,
                'optimal_k_elbow': optimal_k_elbow,
                'optimal_k_silhouette': optimal_k_silhouette
            }
            
        except Exception as e:
            st.warning(f"⚠️ Optimal küme hesaplama hatası: {str(e)}")
            return None


# ============================================================================
# VISUALIZATION ENGINE CLASS (Complete Implementation)
# ============================================================================

class Visualizer:
    """Professional pharmaceutical visualization engine"""
    
    @staticmethod
    def create_kpi_dashboard(df: pd.DataFrame, metrics: Dict) -> None:
        """Create comprehensive KPI dashboard"""
        try:
            # Row 1: Primary Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = metrics.get('total_market_value', 0)
                year = metrics.get('last_year', '')
                growth = metrics.get('avg_growth', 0)
                
                delta_html = f'<span class="kpi-delta {"positive" if growth > 0 else "negative"}">' \
                            f'{"↑" if growth > 0 else "↓"} {abs(growth):.1f}%</span>'
                
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">TOPLAM PAZAR DEĞERİ</div>
                    <div class="kpi-value">{Utils.format_number(total_value, '$')}</div>
                    <div class="kpi-subtitle">{year} Global Pazar</div>
                    {delta_html}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_growth = metrics.get('avg_growth', 0)
                growth_std = metrics.get('growth_std', 0)
                
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">ORTALAMA BÜYÜME</div>
                    <div class="kpi-value">{avg_growth:.1f}%</div>
                    <div class="kpi-subtitle">Yıllık YoY (±{growth_std:.1f}%)</div>
                    <span class="kpi-delta {"positive" if avg_growth > 0 else "negative"}">
                        Medyan: {metrics.get('median_growth', 0):.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrics.get('hhi_index', 0)
                eff_competitors = metrics.get('effective_competitors', 0)
                hhi_status = "Monopol" if hhi > 2500 else "Oligopol" if hhi > 1500 else "Rekabetçi"
                
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">PAZAR YAPISI (HHI)</div>
                    <div class="kpi-value">{hhi:.0f}</div>
                    <div class="kpi-subtitle">{hhi_status}</div>
                    <span class="kpi-delta info">
                        Efektif Rakip: {eff_competitors:.1f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                intl_share = metrics.get('intl_product_share', 0)
                intl_count = metrics.get('intl_product_count', 0)
                
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">INTERNATIONAL PRODUCTS</div>
                    <div class="kpi-value">{intl_share:.1f}%</div>
                    <div class="kpi-subtitle">Global Çoklu Pazar</div>
                    <span class="kpi-delta positive">
                        {intl_count} Ürün
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            # Row 2: Secondary Metrics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                unique_mols = metrics.get('unique_molecules', 0)
                top_mol_share = metrics.get('top_10_molecule_share', 0)
                
                st.metric(
                    "Molekül Çeşitliliği",
                    f"{unique_mols:,}",
                    f"Top 10: {top_mol_share:.1f}%"
                )
            
            with col6:
                avg_price = metrics.get('avg_price', 0)
                price_cv = metrics.get('price_cv', 0)
                
                st.metric(
                    "Ortalama Fiyat",
                    f"${avg_price:.2f}",
                    f"CV: {price_cv:.1f}%"
                )
            
            with col7:
                positive_pct = metrics.get('positive_growth_pct', 0)
                high_growth = metrics.get('high_growth_count', 0)
                
                st.metric(
                    "Büyüyen Ürünler",
                    f"{positive_pct:.1f}%",
                    f"{high_growth} >20% büyüme"
                )
            
            with col8:
                country_coverage = metrics.get('country_coverage', 0)
                top_country_share = metrics.get('top_country_share', 0)
                
                st.metric(
                    "Coğrafi Kapsam",
                    f"{country_coverage} Ülke",
                    f"Lider: {top_country_share:.1f}%"
                )
            
        except Exception as e:
            st.error(f"❌ KPI dashboard hatası: {str(e)}")
    
    @staticmethod
    def create_sales_trend_chart(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create comprehensive sales trend visualization"""
        try:
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            
            if len(sales_cols) < 2:
                st.info("📊 Trend analizi için en az 2 yıllık veri gerekli")
                return None
            
            # Aggregate yearly data
            yearly_data = []
            for col in sorted(sales_cols):
                year_match = re.search(r'(\d{4})', col)
                if year_match:
                    year = year_match.group(1)
                    yearly_data.append({
                        'Yıl': year,
                        'Toplam': df[col].sum(),
                        'Ortalama': df[col].mean(),
                        'Medyan': df[col].median(),
                        'Ürün_Sayısı': (df[col] > 0).sum()
                    })
            
            yearly_df = pd.DataFrame(yearly_data)
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Toplam Satış Trendi',
                    'Yıllık Büyüme Oranı',
                    'Ortalama vs Medyan Satış',
                    'Aktif Ürün Sayısı'
                ),
                specs=[
                    [{"secondary_y": False}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # 1. Total sales trend
            fig.add_trace(
                go.Scatter(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Toplam'],
                    mode='lines+markers+text',
                    name='Toplam Satış',
                    line=dict(color=Config.COLOR_PRIMARY, width=4),
                    marker=dict(size=12),
                    text=[Utils.format_number(x, '$') for x in yearly_df['Toplam']],
                    textposition='top center',
                    hovertemplate='<b>%{x}</b><br>Toplam: %{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. YoY growth rate
            if len(yearly_df) > 1:
                yoy_growth = yearly_df['Toplam'].pct_change() * 100
                colors = [Config.COLOR_SUCCESS if x > 0 else Config.COLOR_DANGER for x in yoy_growth]
                
                fig.add_trace(
                    go.Bar(
                        x=yearly_df['Yıl'][1:],
                        y=yoy_growth[1:],
                        name='Büyüme %',
                        marker_color=colors[1:],
                        text=[f'{x:.1f}%' for x in yoy_growth[1:]],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Büyüme: %{y:.1f}%<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # 3. Average vs Median
            fig.add_trace(
                go.Scatter(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Ortalama'],
                    mode='lines+markers',
                    name='Ortalama',
                    line=dict(color=Config.COLOR_SECONDARY, width=3)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Medyan'],
                    mode='lines+markers',
                    name='Medyan',
                    line=dict(color=Config.COLOR_INFO, width=3, dash='dash')
                ),
                row=2, col=1
            )
            
            # 4. Active products
            fig.add_trace(
                go.Bar(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Ürün_Sayısı'],
                    name='Ürün Sayısı',
                    marker_color=Config.COLOR_WARNING,
                    text=yearly_df['Ürün_Sayısı'],
                    textposition='outside'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=Config.CHART_HEIGHT_LARGE,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True,
                hovermode='closest',
                title_text="📈 Satış Trendleri ve Büyüme Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ Trend grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_market_share_analysis(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create comprehensive market share visualization"""
        try:
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            corp_col = FilterSystem.find_column(df, ['Corporation', 'Şirket'])
            
            if not sales_cols or not corp_col:
                return None
            
            last_sales = sales_cols[-1]
            corp_sales = df.groupby(corp_col)[last_sales].sum().sort_values(ascending=False)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Top 15 Şirket - Pazar Payları',
                    'Pazar Konsantrasyonu (Pareto)',
                    'Satış Dağılımı',
                    'CR Analizi (Concentration Ratios)'
                ),
                specs=[
                    [{'type': 'bar'}, {'type': 'scatter'}],
                    [{'type': 'domain'}, {'type': 'bar'}]
                ],
                vertical_spacing=0.15
            )
            
            # 1. Top companies horizontal bar
            top_15 = corp_sales.nlargest(15)
            top_15_pct = (top_15 / corp_sales.sum()) * 100
            
            fig.add_trace(
                go.Bar(
                    y=top_15.index,
                    x=top_15_pct.values,
                    orientation='h',
                    marker_color=Config.COLOR_PRIMARY,
                    text=[f'{x:.1f}%' for x in top_15_pct.values],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Pay: %{x:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. Pareto chart (cumulative)
            sorted_sales = corp_sales.sort_values(ascending=False)
            cumulative_pct = (sorted_sales.cumsum() / sorted_sales.sum()) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cumulative_pct) + 1)),
                    y=cumulative_pct.values,
                    mode='lines+markers',
                    line=dict(color=Config.COLOR_SECONDARY, width=3),
                    marker=dict(size=6),
                    name='Kümülatif %'
                ),
                row=1, col=2
            )
            
            # Add 80% line
            fig.add_hline(
                y=80, line_dash="dash", line_color=Config.COLOR_WARNING,
                annotation_text="80% Eşiği",
                row=1, col=2
            )
            
            # 3. Pie chart - Top 10 + Others
            top_10 = corp_sales.nlargest(10)
            others = corp_sales.iloc[10:].sum()
            
            pie_data = top_10.copy()
            if others > 0:
                pie_data['Diğer'] = others
            
            fig.add_trace(
                go.Pie(
                    labels=pie_data.index,
                    values=pie_data.values,
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Bold
                ),
                row=2, col=1
            )
            
            # 4. Concentration ratios
            cr_values = []
            cr_labels = []
            for n in [1, 4, 8, 20]:
                if len(corp_sales) >= n:
                    cr = (corp_sales.nlargest(n).sum() / corp_sales.sum()) * 100
                    cr_values.append(cr)
                    cr_labels.append(f'CR{n}')
            
            fig.add_trace(
                go.Bar(
                    x=cr_labels,
                    y=cr_values,
                    marker_color=[Config.COLOR_DANGER, Config.COLOR_WARNING, 
                                 Config.COLOR_INFO, Config.COLOR_SUCCESS],
                    text=[f'{x:.1f}%' for x in cr_values],
                    textposition='outside'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=Config.CHART_HEIGHT_LARGE,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                title_text="🏆 Pazar Payı ve Konsantrasyon Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ Pazar payı grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_price_volume_chart(df: pd.DataFrame, sample_size: int = 5000) -> Optional[go.Figure]:
        """Create optimized price-volume scatter plot"""
        try:
            # Find columns
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
            unit_cols = [col for col in df.columns if 'Birim_' in col or 'Unit' in col]
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            
            # Calculate price if needed
            if not price_cols and sales_cols and unit_cols:
                last_sales = sales_cols[-1]
                last_unit = unit_cols[-1]
                df['_temp_price'] = np.where(
                    df[last_unit] != 0,
                    df[last_sales] / df[last_unit],
                    np.nan
                )
                price_cols = ['_temp_price']
            
            if not price_cols or not unit_cols:
                st.info("💡 Fiyat-hacim analizi için gerekli sütunlar bulunamadı")
                return None
            
            last_price = price_cols[-1]
            last_unit = unit_cols[-1]
            
            # Filter valid data
            plot_df = df[(df[last_price] > 0) & (df[last_unit] > 0)].copy()
            
            if len(plot_df) == 0:
                st.info("📊 Geçerli fiyat-hacim verisi bulunamadı")
                return None
            
            # CRITICAL: Sample for performance
            if len(plot_df) > sample_size:
                plot_df = plot_df.sample(sample_size, random_state=Config.RANDOM_STATE)
                st.caption(f"ℹ️ Performans için {sample_size:,} örnek gösteriliyor (toplam: {len(df):,})")
            
            # Create figure
            mol_col = FilterSystem.find_column(df, ['Molecule', 'Molekül'])
            hover_name = mol_col if mol_col and mol_col in plot_df.columns else None
            
            fig = px.scatter(
                plot_df,
                x=last_price,
                y=last_unit,
                size=last_unit,
                color=last_price,
                hover_name=hover_name,
                title='Fiyat-Hacim İlişkisi',
                labels={
                    last_price: 'Fiyat (USD)',
                    last_unit: 'Hacim (Birim)'
                },
                color_continuous_scale='Viridis',
                opacity=0.6
            )
            
            # Add trendline
            if len(plot_df) >= 10:
                from scipy.stats import linregress
                slope, intercept, r_value, _, _ = linregress(
                    plot_df[last_price].fillna(0),
                    plot_df[last_unit].fillna(0)
                )
                
                x_trend = np.array([plot_df[last_price].min(), plot_df[last_price].max()])
                y_trend = slope * x_trend + intercept
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        name=f'Trend (R²={r_value**2:.3f})',
                        line=dict(color='red', width=2, dash='dash')
                    )
                )
            
            fig.update_layout(
                height=Config.CHART_HEIGHT_STANDARD,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ Fiyat-hacim grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_geographic_choropleth(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create world choropleth map"""
        try:
            country_col = FilterSystem.find_column(df, ['Country', 'Ülke'])
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            
            if not country_col or not sales_cols:
                return None
            
            last_sales = sales_cols[-1]
            
            # Aggregate by country
            country_sales = df.groupby(country_col)[last_sales].sum().reset_index()
            country_sales.columns = ['Country', 'Total_Sales']
            
            # Create map
            fig = px.choropleth(
                country_sales,
                locations='Country',
                locationmode='country names',
                color='Total_Sales',
                hover_name='Country',
                hover_data={
                    'Total_Sales': ':,.0f',
                    'Country': False
                },
                color_continuous_scale='Viridis',
                title='🗺️ Global İlaç Pazarı Dağılımı',
                projection='natural earth'
            )
            
            fig.update_layout(
                height=Config.CHART_HEIGHT_STANDARD,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                geo=dict(
                    bgcolor='rgba(0,0,0,0)',
                    lakecolor='#1e3a5f',
                    landcolor='#2d4a7a',
                    subunitcolor='#64748b',
                    showframe=False
                ),
                coloraxis_colorbar=dict(
                    title="Toplam Satış",
                    tickprefix="$",
                    tickformat=".2s"
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ Harita oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_ml_forecast_chart(results: Dict) -> Optional[go.Figure]:
        """Create ML forecast visualization"""
        try:
            forecast_df = pd.DataFrame(results['forecast'])
            
            fig = go.Figure()
            
            # Prediction line
            fig.add_trace(go.Scatter(
                x=forecast_df['year'],
                y=forecast_df['prediction'],
                mode='lines+markers',
                name='Tahmin',
                line=dict(color=Config.COLOR_PRIMARY, width=4),
                marker=dict(size=12, symbol='diamond')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df['year'],
                y=forecast_df['confidence_high'],
                mode='lines',
                name='Üst Limit',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['year'],
                y=forecast_df['confidence_low'],
                mode='lines',
                name='Güven Aralığı (95%)',
                line=dict(width=0),
                fillcolor='rgba(42, 202, 234, 0.3)',
                fill='tonexty'
            ))
            
            fig.update_layout(
                title='📈 Satış Tahmini (ML Forecasting)',
                xaxis_title='Yıl',
                yaxis_title='Tahmini Satış (USD)',
                height=Config.CHART_HEIGHT_STANDARD,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ Tahmin grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_3d_cluster_plot(results: Dict, df: pd.DataFrame) -> Optional[go.Figure]:
        """Create 3D cluster visualization"""
        try:
            if results['pca_data'].shape[1] < 3:
                return None
            
            mol_col = FilterSystem.find_column(df, ['Molecule', 'Molekül'])
            hover_text = df[mol_col] if mol_col else None
            
            fig = go.Figure(data=[go.Scatter3d(
                x=results['pca_data'][:, 0],
                y=results['pca_data'][:, 1],
                z=results['pca_data'][:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=results['clusters'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Küme"),
                    line=dict(width=0.5, color='white')
                ),
                text=results['cluster_labels'],
                hovertext=hover_text,
                hovertemplate='<b>%{text}</b><br>%{hovertext}<extra></extra>'
            )])
            
            fig.update_layout(
                title='🎯 3D Küme Görselleştirmesi (PCA)',
                scene=dict(
                    xaxis_title=f"PC1 ({results['pca_variance'][0]:.1%})",
                    yaxis_title=f"PC2 ({results['pca_variance'][1]:.1%})",
                    zaxis_title=f"PC3 ({results['pca_variance'][2]:.1%})",
                    bgcolor='rgba(0,0,0,0)'
                ),
                height=Config.CHART_HEIGHT_LARGE,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ 3D grafik hatası: {str(e)}")
            return None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Application header
    st.markdown("""
    <div class="section-header" style="text-align: center;">
        <h1>💊 PHARMAINTELLIGENCE PRO ML</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Enterprise Pharmaceutical Analytics with Advanced Machine Learning
        </p>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem; color: #94a3b8;">
            Version 6.0 | Forecasting • Clustering • Anomaly Detection • Geographic Analysis
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
    if 'active_filters' not in st.session_state:
        st.session_state.active_filters = {}
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = {}
    
    # Sidebar - Data upload
    with st.sidebar:
        st.markdown("""
        <div class="section-header">
            <h2>📁 VERİ YÜKLEME</h2>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Excel veya CSV Dosyası Yükleyin",
            type=['xlsx', 'xls', 'csv'],
            help="İlaç pazarı verilerinizi yükleyin (100K+ satır desteklenir)"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🚀 Veriyi Yükle", type="primary", use_container_width=True):
                    with st.spinner("📥 Veri yükleniyor ve hazırlanıyor..."):
                        # Load data
                        df = DataManager.load_data(uploaded_file)
                        
                        if df is not None and len(df) > 0:
                            # Normalize countries
                            df = DataManager.normalize_country_names(df)
                            
                            # Prepare analysis data
                            df = DataManager.prepare_analysis_data(df)
                            
                            # Store in session state
                            st.session_state.data = df
                            st.session_state.filtered_data = df.copy()
                            
                            # Calculate metrics and insights
                            with st.spinner("📊 Metrikler hesaplanıyor..."):
                                st.session_state.metrics = AnalyticsEngine.calculate_comprehensive_metrics(df)
                                st.session_state.insights = AnalyticsEngine.generate_strategic_insights(
                                    df, st.session_state.metrics
                                )
                            
                            st.balloons()
                            st.success(f"✅ Analiz tamamlandı! {len(df):,} satır hazır.")
                            time.sleep(1)
                            st.rerun()
            
            with col2:
                if st.button("ℹ️ Yardım", use_container_width=True):
                    st.info("""
                    **Desteklenen Format:**
                    - Excel (.xlsx, .xls)
                    - CSV (.csv)
                    
                    **Beklenen Sütunlar:**
                    - Satış_YYYY veya Sales_YYYY
                    - Country/Ülke
                    - Corporation/Şirket
                    - Molecule/Molekül
                    """)
        
        st.markdown("---")
        
        # Display current data info
        if st.session_state.data is not None:
            df_info = st.session_state.data
            
            st.markdown("""
            <div class="success-box">
                <h4 style="margin: 0 0 0.5rem 0; color: #2dd4a3;">✅ Veri Yüklendi</h4>
            </div>
            """, unsafe_allow_html=True)
            
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Satırlar", f"{len(df_info):,}")
                st.metric("Sütunlar", f"{len(df_info.columns)}")
            with info_col2:
                memory_mb = df_info.memory_usage(deep=True).sum() / 1024**2
                st.metric("Bellek", f"{memory_mb:.1f} MB")
                st.metric("Filtrelenmiş", f"{len(st.session_state.filtered_data):,}")
    
    # Main content
    if st.session_state.data is None:
        show_welcome_screen()
        return
    
    df = st.session_state.data
    
    # Filters in sidebar
    search_term, filters, apply_filters, clear_filters = FilterSystem.create_filter_sidebar(df)
    
    # Handle filter actions
    if apply_filters:
        with st.spinner("🔄 Filtreler uygulanıyor..."):
            filtered_df = FilterSystem.apply_filters(df, search_term, filters)
            st.session_state.filtered_data = filtered_df
            st.session_state.active_filters = filters
            
            # Recalculate metrics
            st.session_state.metrics = AnalyticsEngine.calculate_comprehensive_metrics(filtered_df)
            st.session_state.insights = AnalyticsEngine.generate_strategic_insights(
                filtered_df, st.session_state.metrics
            )
            
            st.success(f"✅ {len(filtered_df):,} satır gösteriliyor")
            time.sleep(0.5)
            st.rerun()
    
    if clear_filters:
        st.session_state.filtered_data = df.copy()
        st.session_state.active_filters = {}
        st.session_state.metrics = AnalyticsEngine.calculate_comprehensive_metrics(df)
        st.session_state.insights = AnalyticsEngine.generate_strategic_insights(df, st.session_state.metrics)
        st.success("✅ Filtreler temizlendi")
        time.sleep(0.5)
        st.rerun()
    
    # Show active filters badge
    if st.session_state.active_filters:
        filter_count = len([k for k in st.session_state.active_filters.keys() 
                           if not k.endswith('_range') and k not in ['positive_growth', 'high_growth', 
                                                                      'top_performers', 'exclude_outliers']])
        
        st.markdown(f"""
        <div class="filter-badge">
            🎯 {filter_count} Aktif Filtre | 
            Gösterilen: {len(st.session_state.filtered_data):,} / {len(df):,} satır
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 GENEL BAKIŞ",
        "🌍 COĞRAFİ ANALİZ",
        "💰 FİYAT ANALİZİ",
        "🏆 REKABET ANALİZİ",
        "🤖 ML LABORATUVARI",
        "🎯 SEGMENTASYON",
        "📑 RAPORLAMA"
    ])
    
    with tab1:
        show_overview_tab()
    
    with tab2:
        show_geographic_tab()
    
    with tab3:
        show_price_tab()
    
    with tab4:
        show_competition_tab()
    
    with tab5:
        show_ml_lab_tab()
    
    with tab6:
        show_segmentation_tab()
    
    with tab7:
        show_reporting_tab()


def show_welcome_screen():
    """Display welcome screen"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 5rem; margin-bottom: 1rem;">💊</div>
        <h1 style="color: #2acaea; font-size: 2.5rem; margin-bottom: 1rem;">
            PharmaIntelligence Pro'ya Hoşgeldiniz
        </h1>
        <p style="color: #cbd5e1; font-size: 1.2rem; max-width: 800px; margin: 0 auto 2rem auto;">
            Enterprise-grade ilaç pazarı analiz platformu ile verilerinizi makine öğrenmesi 
            gücüyle analiz edin ve stratejik kararlar alın.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="kpi-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
            <h3 style="color: #2acaea; margin-bottom: 0.5rem;">ML Forecasting</h3>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                Random Forest ile gelecek satış tahminleri ve güven aralıkları
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="kpi-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
            <h3 style="color: #2acaea; margin-bottom: 0.5rem;">Clustering</h3>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                K-Means ile akıllı segmentasyon ve 3D görselleştirme
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="kpi-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
            <h3 style="color: #2acaea; margin-bottom: 0.5rem;">Anomaly Detection</h3>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                Isolation Forest ile anormal satış ve fiyat tespiti
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="kpi-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌍</div>
            <h3 style="color: #2acaea; margin-bottom: 0.5rem;">Geo Analysis</h3>
            <p style="color: #cbd5e1; font-size: 0.9rem;">
                İnteraktif dünya haritaları ve ülke bazlı analiz
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started
    st.markdown("""
    <div class="info-box" style="margin-top: 3rem; max-width: 800px; margin-left: auto; margin-right: auto;">
        <h3 style="margin: 0 0 1rem 0; color: #60a5fa;">🚀 Hemen Başlayın</h3>
        <ol style="color: #cbd5e1; line-height: 2;">
            <li><strong>Sol panelden</strong> Excel veya CSV dosyanızı yükleyin</li>
            <li><strong>"Veriyi Yükle"</strong> butonuna tıklayın</li>
            <li><strong>Sekmeleri kullanarak</strong> analizleri keşfedin</li>
            <li><strong>Filtreleyerek</strong> derin içgörüler elde edin</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


def show_overview_tab():
    """Overview tab with comprehensive KPIs and insights"""
    df = st.session_state.filtered_data
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    
    st.markdown('<div class="subsection-header"><h3>📊 Genel Bakış ve Performans Göstergeleri</h3></div>', 
                unsafe_allow_html=True)
    
    # KPI Dashboard
    Visualizer.create_kpi_dashboard(df, metrics)
    
    # Insights section
    st.markdown('<div class="subsection-header"><h3>💡 Stratejik İçgörüler</h3></div>', 
                unsafe_allow_html=True)
    
    if insights:
        cols = st.columns(2)
        for idx, insight in enumerate(insights):
            with cols[idx % 2]:
                insight_class = insight.get('type', 'info')
                st.markdown(f"""
                <div class="insight-card {insight_class}">
                    <h4>{insight['title']}</h4>
                    <p>{insight['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📊 İçgörüler analiz ediliyor...")
    
    # Sales trends
    st.markdown('<div class="subsection-header"><h3>📈 Satış Trendleri ve Büyüme</h3></div>', 
                unsafe_allow_html=True)
    
    trend_chart = Visualizer.create_sales_trend_chart(df)
    if trend_chart:
        st.plotly_chart(trend_chart, use_container_width=True, config={'displayModeBar': True})
    
    # Data preview
    st.markdown('<div class="subsection-header"><h3>📋 Veri Önizleme</h3></div>', 
                unsafe_allow_html=True)
    
    preview_col1, preview_col2 = st.columns([1, 3])
    
    with preview_col1:
        n_rows = st.slider("Gösterilecek Satır", 10, 100, 25, 5)
        
        # Column selector
        all_cols = df.columns.tolist()
        priority_cols = []
        for col_pattern in ['Molekül', 'Şirket', 'Ülke', 'Satis_', 'Buyume_', 'Pazar']:
            priority_cols.extend([c for c in all_cols if col_pattern in c])
        
        default_cols = list(dict.fromkeys(priority_cols[:5]))  # Remove duplicates, keep order
        
        selected_cols = st.multiselect(
            "Gösterilecek Sütunlar",
            options=all_cols,
            default=default_cols if default_cols else all_cols[:5]
        )
    
    with preview_col2:
        if selected_cols:
            st.dataframe(
                df[selected_cols].head(n_rows),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(df.head(n_rows), use_container_width=True, height=400)


def show_geographic_tab():
    """Geographic analysis tab"""
    df = st.session_state.filtered_data
    
    st.markdown('<div class="subsection-header"><h3>🌍 Coğrafi Analiz ve Pazar Dağılımı</h3></div>', 
                unsafe_allow_html=True)
    
    # World map
    st.markdown("#### 🗺️ Dünya Haritası - Satış Dağılımı")
    
    choropleth = Visualizer.create_geographic_choropleth(df)
    if choropleth:
        st.plotly_chart(choropleth, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("🌍 Harita için gerekli veri bulunamadı")
    
    # Country analysis
    country_col = FilterSystem.find_column(df, ['Country', 'Ülke'])
    sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
    
    if country_col and sales_cols:
        st.markdown("#### 📊 Ülke Bazlı Detaylı Analiz")
        
        last_sales = sales_cols[-1]
        country_sales = df.groupby(country_col)[last_sales].sum().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top countries by sales
            fig = go.Figure()
            top_20 = country_sales.nlargest(20)
            
            fig.add_trace(go.Bar(
                x=top_20.values,
                y=top_20.index,
                orientation='h',
                marker=dict(
                    color=top_20.values,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[Utils.format_number(x, '$') for x in top_20.values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Top 20 Ülke - Toplam Satış",
                xaxis_title="Toplam Satış (USD)",
                yaxis_title="Ülke",
                height=Config.CHART_HEIGHT_LARGE,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Growth by country
            growth_cols = [col for col in df.columns if 'Buyume_' in col or 'Growth_' in col]
            
            if growth_cols:
                last_growth = growth_cols[-1]
                country_growth = df.groupby(country_col)[last_growth].mean().sort_values(ascending=False)
                
                fig = go.Figure()
                top_20_growth = country_growth.nlargest(20)
                
                colors = [Config.COLOR_SUCCESS if x > 0 else Config.COLOR_DANGER 
                         for x in top_20_growth.values]
                
                fig.add_trace(go.Bar(
                    x=top_20_growth.values,
                    y=top_20_growth.index,
                    orientation='h',
                    marker_color=colors,
                    text=[f'{x:.1f}%' for x in top_20_growth.values],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Top 20 Ülke - Ortalama Büyüme",
                    xaxis_title="Büyüme Oranı (%)",
                    yaxis_title="Ülke",
                    height=Config.CHART_HEIGHT_LARGE,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                
                st.plotly_chart(fig, use_container_width=True)


def show_price_tab():
    """Price analysis tab with optimized performance"""
    df = st.session_state.filtered_data
    
    st.markdown('<div class="subsection-header"><h3>💰 Fiyat Analizi ve Optimizasyon</h3></div>', 
                unsafe_allow_html=True)
    
    # Performance control
    st.markdown("#### ⚙️ Performans Ayarları")
    sample_size = st.slider(
        "Gösterilecek Örnek Sayısı (Performans için)",
        min_value=Config.SAMPLE_SIZE_MIN,
        max_value=min(Config.SAMPLE_SIZE_MAX, len(df)),
        value=min(Config.SAMPLE_SIZE_DEFAULT, len(df)),
        step=1000,
        help="Büyük veri setlerinde performans için örnekleme yapılır"
    )
    
    # Price-volume analysis
    st.markdown("#### 📊 Fiyat-Hacim İlişkisi")
    
    price_volume_chart = Visualizer.create_price_volume_chart(df, sample_size=sample_size)
    if price_volume_chart:
        st.plotly_chart(price_volume_chart, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("💡 Fiyat-hacim analizi için gerekli sütunlar bulunamadı")
    
    # Price distribution and segments
    st.markdown("#### 📈 Fiyat Dağılımı ve Segmentleri")
    
    price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
    
    if price_cols:
        last_price = price_cols[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[last_price],
                nbinsx=50,
                marker_color=Config.COLOR_PRIMARY,
                name='Fiyat Dağılımı'
            ))
            
            fig.update_layout(
                title="Fiyat Dağılımı",
                xaxis_title="Fiyat (USD)",
                yaxis_title="Ürün Sayısı",
                height=Config.CHART_HEIGHT_SMALL,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price segments
            price_data = df[last_price].dropna()
            
            if len(price_data) > 0:
                segments = pd.cut(
                    price_data,
                    bins=[0, 10, 50, 100, 500, float('inf')],
                    labels=['Ekonomi\n(<$10)', 'Standart\n($10-$50)', 
                           'Premium\n($50-$100)', 'Süper Premium\n($100-$500)', 
                           'Lüks\n(>$500)']
                )
                
                segment_counts = segments.value_counts()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=segment_counts.index,
                    y=segment_counts.values,
                    marker_color=Config.COLOR_SECONDARY,
                    text=segment_counts.values,
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Fiyat Segmentleri",
                    xaxis_title="Segment",
                    yaxis_title="Ürün Sayısı",
                    height=Config.CHART_HEIGHT_SMALL,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                
                st.plotly_chart(fig, use_container_width=True)


def show_competition_tab():
    """Competition analysis tab"""
    df = st.session_state.filtered_data
    metrics = st.session_state.metrics
    
    st.markdown('<div class="subsection-header"><h3>🏆 Rekabet Analizi ve Pazar Yapısı</h3></div>', 
                unsafe_allow_html=True)
    
    # Competition metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hhi = metrics.get('hhi_index', 0)
        if hhi > 2500:
            status = "Monopol"
            delta_color = "inverse"
        elif hhi > 1500:
            status = "Oligopol"
            delta_color = "normal"
        else:
            status = "Rekabetçi"
            delta_color = "normal"
        
        st.metric("HHI İndeksi", f"{hhi:.0f}", status)
    
    with col2:
        cr4 = metrics.get('cr4', 0)
        st.metric("CR4 (Top 4 Payı)", f"{cr4:.1f}%")
    
    with col3:
        top_3 = metrics.get('top_3_share', 0)
        st.metric("Top 3 Payı", f"{top_3:.1f}%")
    
    with col4:
        eff_comp = metrics.get('effective_competitors', 0)
        st.metric("Efektif Rakip Sayısı", f"{eff_comp:.1f}")
    
    # Market share analysis
    st.markdown("#### 📊 Pazar Payı Analizi")
    
    market_share_chart = Visualizer.create_market_share_analysis(df)
    if market_share_chart:
        st.plotly_chart(market_share_chart, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("📊 Pazar payı analizi için gerekli veri bulunamadı")


def show_ml_lab_tab():
    """Machine Learning Laboratory tab - Complete implementation"""
    df = st.session_state.filtered_data
    
    st.markdown("""
    <div class="subsection-header">
        <h3>🤖 Makine Öğrenmesi Laboratuvarı</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0;">
            <strong>💡 ML Özellikleri:</strong> Bu bölümde satış tahminleme, ürün kümeleme ve 
            anomali tespiti yapabilirsiniz. Her analiz için gerekli özellikleri seçin ve modeli çalıştırın.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ML method selector
    ml_method = st.selectbox(
        "🎯 ML Analiz Türü Seçin",
        [
            "📈 Satış Tahmini (Forecasting)",
            "🎯 Ürün Kümeleme (Clustering)",
            "⚠️ Anomali Tespiti (Anomaly Detection)"
        ],
        help="Yapılacak makine öğrenmesi analiz türünü seçin"
    )
    
    if "Forecasting" in ml_method:
        show_forecasting_panel(df)
    elif "Clustering" in ml_method:
        show_clustering_panel(df)
    elif "Anomali" in ml_method:
        show_anomaly_panel(df)


def show_forecasting_panel(df):
    """Forecasting analysis panel"""
    st.markdown("### 📈 Satış Tahmini - ML Forecasting")
    
    sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
    growth_cols = [col for col in df.columns if 'Buyume_' in col or 'Growth_' in col]
    price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
    
    if len(sales_cols) < 2:
        st.warning("⚠️ Tahmin için en az 2 yıllık satış verisi gerekli")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature selection
        available_features = sales_cols[:-1] + growth_cols + price_cols
        
        if len(available_features) < 1:
            st.warning("⚠️ Yeterli özellik bulunamadı")
            return
        
        selected_features = st.multiselect(
            "📊 Tahmin için kullanılacak özellikler",
            available_features,
            default=available_features[:min(3, len(available_features))],
            help="Model eğitiminde kullanılacak özellikler"
        )
    
    with col2:
        # Model parameters
        model_type = st.selectbox(
            "🤖 Model Türü",
            ["Random Forest", "Gradient Boosting", "Linear Regression"],
            help="Kullanılacak ML algoritması"
        )
        
        forecast_years = st.slider(
            "📅 Kaç yıl ilerisi tahmin edilsin?",
            1, 5, 2,
            help="Gelecek için tahmin edilecek yıl sayısı"
        )
    
    if st.button("🚀 Tahmin Modelini Çalıştır", type="primary", use_container_width=True):
        if not selected_features:
            st.error("❌ En az bir özellik seçmelisiniz!")
            return
        
        with st.spinner("🔄 Model eğitiliyor... (Bu birkaç saniye sürebilir)"):
            model_type_map = {
                "Random Forest": "rf",
                "Gradient Boosting": "gbm",
                "Linear Regression": "linear"
            }
            
            results, error = MLEngine.train_forecasting_model(
                df,
                target_col=sales_cols[-1],
                feature_cols=selected_features,
                forecast_years=forecast_years,
                model_type=model_type_map[model_type]
            )
            
            if error:
                st.error(f"❌ {error}")
            elif results:
                # Store results
                st.session_state.ml_results['forecasting'] = results
                
                # Performance metrics
                st.markdown("#### 📊 Model Performansı")
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                with perf_col1:
                    st.metric("R² Skoru (Test)", f"{results['r2_test']:.3f}")
                
                with perf_col2:
                    st.metric("MAE (Test)", Utils.format_number(results['mae_test'], '$'))
                
                with perf_col3:
                    st.metric("RMSE (Test)", Utils.format_number(results['rmse_test'], '$'))
                
                with perf_col4:
                    st.metric("CV Mean", f"{results['cv_mean']:.3f}")
                
                # Forecast chart
                st.markdown("#### 📈 Tahmin Sonuçları")
                
                forecast_chart = Visualizer.create_ml_forecast_chart(results)
                if forecast_chart:
                    st.plotly_chart(forecast_chart, use_container_width=True)
                
                # Feature importance
                st.markdown("#### 🎯 Özellik Önemi")
                
                importance_df = pd.DataFrame({
                    'Özellik': list(results['feature_importance'].keys()),
                    'Önem': list(results['feature_importance'].values())
                }).sort_values('Önem', ascending=False)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance_df['Önem'],
                    y=importance_df['Özellik'],
                    orientation='h',
                    marker_color=Config.COLOR_PRIMARY
                ))
                
                fig.update_layout(
                    title="Özellik Önemi Analizi",
                    xaxis_title="Önem Skoru",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                with st.expander("📋 Detaylı Tahmin Tablosu"):
                    forecast_df = pd.DataFrame(results['forecast'])
                    forecast_df['prediction'] = forecast_df['prediction'].apply(lambda x: Utils.format_number(x, '$'))
                    forecast_df['confidence_low'] = forecast_df['confidence_low'].apply(lambda x: Utils.format_number(x, '$'))
                    forecast_df['confidence_high'] = forecast_df['confidence_high'].apply(lambda x: Utils.format_number(x, '$'))
                    st.dataframe(forecast_df, use_container_width=True)


def show_clustering_panel(df):
    """Clustering analysis panel"""
    st.markdown("### 🎯 Ürün Kümeleme - ML Clustering")
    
    sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
    growth_cols = [col for col in df.columns if 'Buyume_' in col or 'Growth_' in col]
    price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
    
    available_features = sales_cols + growth_cols + price_cols
    
    if len(available_features) < 2:
        st.warning("⚠️ Kümeleme için en az 2 özellik gerekli")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature selection
        selected_features = st.multiselect(
            "📊 Kümeleme için kullanılacak özellikler",
            available_features,
            default=available_features[:min(3, len(available_features))],
            help="Kümeleme analizinde kullanılacak özellikler"
        )
    
    with col2:
        # Clustering parameters
        algorithm = st.selectbox(
            "🔧 Kümeleme Algoritması",
            ["K-Means", "Hierarchical", "DBSCAN"],
            help="Kullanılacak kümeleme algoritması"
        )
        
        n_clusters = st.slider(
            "🎯 Küme Sayısı",
            Config.MIN_N_CLUSTERS,
            Config.MAX_N_CLUSTERS,
            Config.DEFAULT_N_CLUSTERS,
            help="Oluşturulacak küme sayısı"
        )
    
    button_col1, button_col2 = st.columns([2, 1])
    
    with button_col1:
        run_clustering = st.button("🚀 Kümeleme Analizi Yap", type="primary", use_container_width=True)
    
    with button_col2:
        show_elbow = st.button("📊 Optimal Küme Bul", use_container_width=True)
    
    if show_elbow and selected_features:
        with st.spinner("🔄 Optimal küme sayısı hesaplanıyor..."):
            elbow_results = MLEngine.calculate_optimal_clusters(df, selected_features)
            
            if elbow_results:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Elbow Method - Inertia', 'Silhouette Score')
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=elbow_results['k_values'],
                        y=elbow_results['inertias'],
                        mode='lines+markers',
                        name='Inertia',
                        line=dict(color=Config.COLOR_PRIMARY, width=3),
                        marker=dict(size=10)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=elbow_results['k_values'],
                        y=elbow_results['silhouettes'],
                        mode='lines+markers',
                        name='Silhouette',
                        line=dict(color=Config.COLOR_SECONDARY, width=3),
                        marker=dict(size=10)
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"💡 Önerilen küme sayısı: **{elbow_results['optimal_k_silhouette']}** (Silhouette) | " +
                          f"**{elbow_results['optimal_k_elbow']}** (Elbow)")
    
    if run_clustering:
        if not selected_features:
            st.error("❌ En az bir özellik seçmelisiniz!")
            return
        
        with st.spinner("🔄 Kümeleme analizi yapılıyor..."):
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
                # Store results
                st.session_state.ml_results['clustering'] = results
                
                # Performance metrics
                st.markdown("#### 📊 Kümeleme Kalitesi")
                
                qual_col1, qual_col2, qual_col3 = st.columns(3)
                
                with qual_col1:
                    st.metric("Silhouette Skoru", f"{results['silhouette_score']:.3f}")
                
                with qual_col2:
                    st.metric("Calinski-Harabasz", f"{results['calinski_score']:.0f}")
                
                with qual_col3:
                    st.metric("Davies-Bouldin", f"{results['davies_bouldin_score']:.3f}")
                
                # 3D visualization
                if results['pca_data'].shape[1] >= 3:
                    st.markdown("#### 📊 3D Küme Görselleştirmesi")
                    
                    cluster_3d = Visualizer.create_3d_cluster_plot(results, df)
                    if cluster_3d:
                        st.plotly_chart(cluster_3d, use_container_width=True)
                
                # Cluster distribution
                st.markdown("#### 📊 Küme Dağılımı")
                
                cluster_counts = pd.Series(results['cluster_labels']).value_counts()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    marker_color=Config.COLOR_PRIMARY,
                    text=cluster_counts.values,
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Küme Başına Ürün Sayısı",
                    xaxis_title="Küme",
                    yaxis_title="Ürün Sayısı",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster statistics
                with st.expander("📋 Küme İstatistikleri"):
                    for cluster_id, stats in results['cluster_stats'].items():
                        st.markdown(f"**Küme {cluster_id}:** {stats['size']} ürün ({stats['percentage']:.1f}%)")


def show_anomaly_panel(df):
    """Anomaly detection panel"""
    st.markdown("### ⚠️ Anomali Tespiti - ML Anomaly Detection")
    
    sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
    growth_cols = [col for col in df.columns if 'Buyume_' in col or 'Growth_' in col]
    price_cols = [col for col in df.columns if 'Ort_Fiyat' in col or 'Price' in col]
    
    available_features = sales_cols + growth_cols + price_cols
    
    if len(available_features) < 2:
        st.warning("⚠️ Anomali tespiti için en az 2 özellik gerekli")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_features = st.multiselect(
            "📊 Anomali tespiti için kullanılacak özellikler",
            available_features,
            default=available_features[:min(3, len(available_features))],
            help="Anomali tespitinde kullanılacak özellikler"
        )
    
    with col2:
        contamination = st.slider(
            "🎯 Beklenen Anomali Oranı (%)",
            min_value=int(Config.MIN_CONTAMINATION * 100),
            max_value=int(Config.MAX_CONTAMINATION * 100),
            value=int(Config.DEFAULT_CONTAMINATION * 100),
            help="Veri setindeki anormal veri oranı tahmini"
        ) / 100
    
    if st.button("🚀 Anomali Tespiti Yap", type="primary", use_container_width=True):
        if not selected_features:
            st.error("❌ En az bir özellik seçmelisiniz!")
            return
        
        with st.spinner("🔄 Anomaliler tespit ediliyor..."):
            results, error = MLEngine.detect_anomalies(df, selected_features, contamination)
            
            if error:
                st.error(f"❌ {error}")
            elif results:
                # Store results
                st.session_state.ml_results['anomaly'] = results
                
                # Results
                st.markdown("#### 📊 Anomali Tespit Sonuçları")
                
                anom_col1, anom_col2, anom_col3 = st.columns(3)
                
                with anom_col1:
                    st.metric("Toplam Anomali", results['anomaly_count'])
                
                with anom_col2:
                    st.metric("Anomali Oranı", f"{results['anomaly_percentage']:.2f}%")
                
                with anom_col3:
                    normal_count = len(df) - results['anomaly_count']
                    st.metric("Normal Ürün", normal_count)
                
                # Visualization
                if len(selected_features) >= 2:
                    st.markdown("#### 📊 Anomali Dağılımı")
                    
                    anomaly_df = df.copy()
                    anomaly_df['Anomali'] = results['is_anomaly']
                    anomaly_df['Anomali_Skoru'] = results['anomaly_scores']
                    
                    fig = px.scatter(
                        anomaly_df,
                        x=selected_features[0],
                        y=selected_features[1],
                        color='Anomali',
                        color_discrete_map={True: Config.COLOR_DANGER, False: Config.COLOR_SUCCESS},
                        title="Anomali Tespiti - İlk İki Özellik",
                        labels={'Anomali': 'Anomali mi?'},
                        hover_data=['Anomali_Skoru'],
                        opacity=0.6
                    )
                    
                    fig.update_layout(
                        height=600,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#f8fafc'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly list
                with st.expander("📋 Tespit Edilen Anomaliler (Top 50)"):
                    anomaly_indices = np.where(results['is_anomaly'])[0]
                    anomaly_df = df.iloc[anomaly_indices].copy()
                    anomaly_df['Anomali_Skoru'] = results['anomaly_scores'][anomaly_indices]
                    anomaly_df['Şiddet'] = results['severity'][anomaly_indices]
                    
                    display_cols = selected_features + ['Anomali_Skoru', 'Şiddet']
                    mol_col = FilterSystem.find_column(df, ['Molecule', 'Molekül'])
                    if mol_col and mol_col in anomaly_df.columns:
                        display_cols = [mol_col] + display_cols
                    
                    st.dataframe(
                        anomaly_df[display_cols].sort_values('Anomali_Skoru').head(50),
                        use_container_width=True,
                        height=400
                    )


def show_segmentation_tab():
    """Advanced segmentation and profiling tab"""
    df = st.session_state.filtered_data
    
    st.markdown('<div class="subsection-header"><h3>🎯 İleri Seviye Segmentasyon</h3></div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0;">
            Bu bölümde verilerinizi farklı kriterlere göre segmentlere ayırabilir ve 
            her segmentin özelliklerini detaylı olarak inceleyebilirsiniz.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Segmentation options
    segment_by = st.selectbox(
        "🎯 Segmentasyon Kriteri",
        ["Büyüme Kategorisi", "Fiyat Segmenti", "Pazar Pozisyonu", "Performans Skoru"],
        help="Ürünlerin nasıl segmentlere ayrılacağını seçin"
    )
    
    segment_col_map = {
        "Büyüme Kategorisi": "Buyume_Kategori",
        "Fiyat Segmenti": "Fiyat_Tier",
        "Pazar Pozisyonu": "Pazar_Pozisyon",
        "Performans Skoru": "Performans_Skoru_100"
    }
    
    segment_col = segment_col_map.get(segment_by)
    
    if segment_col and segment_col in df.columns:
        if segment_col == "Performans_Skoru_100":
            # For performance score, create bins
            df['_segment'] = pd.cut(
                df[segment_col],
                bins=[0, 25, 50, 75, 100],
                labels=['Düşük Performans', 'Orta-Alt', 'Orta-Üst', 'Yüksek Performans']
            )
            segment_col = '_segment'
        
        segments = df[segment_col].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Segment distribution
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=segments.index,
                values=segments.values,
                hole=0.4,
                marker_colors=px.colors.qualitative.Bold
            ))
            
            fig.update_layout(
                title=f"{segment_by} Dağılımı",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Segment statistics
            st.markdown("#### 📊 Segment İstatistikleri")
            
            sales_cols = [col for col in df.columns if 'Satis_' in col or 'Sales_' in col]
            if sales_cols:
                last_sales = sales_cols[-1]
                
                segment_stats = df.groupby(segment_col).agg({
                    last_sales: ['sum', 'mean', 'count']
                }).round(2)
                
                st.dataframe(segment_stats, use_container_width=True)
    else:
        st.info(f"📊 '{segment_by}' için veri bulunamadı. Lütfen başka bir kriter seçin.")


def show_reporting_tab():
    """Comprehensive reporting and export tab"""
    df = st.session_state.filtered_data
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    
    st.markdown('<div class="subsection-header"><h3>📑 Raporlama ve İndirme</h3></div>', 
                unsafe_allow_html=True)
    
    # Export options
    st.markdown("#### 📥 Veri İndirme Seçenekleri")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**CSV Format**")
        Utils.create_download_link(df, "pharma_data", "csv")
    
    with col2:
        st.markdown("**JSON Format**")
        Utils.create_download_link(df, "pharma_data", "json")
    
    with col3:
        st.markdown("**Özet Rapor**")
        if st.button("📄 Rapor Oluştur", use_container_width=True):
            summary = f"""
# PHARMAINTELLIGENCE PRO - ÖZET RAPOR
{'=' * 60}
Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Rapor Versiyonu: {Config.VERSION}

## 1. GENEL METRİKLER
{'=' * 60}
Toplam Satır: {metrics.get('total_rows', 0):,}
Toplam Pazar Değeri: {Utils.format_number(metrics.get('total_market_value', 0), '$')}
Ortalama Büyüme: {metrics.get('avg_growth', 0):.1f}%
HHI İndeksi: {metrics.get('hhi_index', 0):.0f}

## 2. PAZAR YAPISI
{'=' * 60}
Benzersiz Molekül: {metrics.get('unique_molecules', 0):,}
Ülke Kapsamı: {metrics.get('country_coverage', 0)}
International Product: {metrics.get('intl_product_count', 0)}
Top 3 Pazar Payı: {metrics.get('top_3_share', 0):.1f}%

## 3. STRATEJİK İÇGÖRÜLER
{'=' * 60}
"""
            for i, insight in enumerate(insights[:10], 1):
                summary += f"\n{i}. {insight['title']}\n   {insight['description']}\n"
            
            st.download_button(
                label="⬇️ Raporu İndir",
                data=summary,
                file_name=f"pharma_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Quick statistics
    st.markdown("#### 📈 Hızlı İstatistikler")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with stat_col2:
        st.metric("Toplam Sütun", len(df.columns))
    
    with stat_col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Bellek", f"{memory_mb:.1f} MB")
    
    with stat_col4:
        intl_count = metrics.get('intl_product_count', 0)
        st.metric("International", intl_count)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        # Enable garbage collection
        gc.enable()
        
        # Run main application
        main()
        
    except Exception as e:
        st.error(f"❌ Kritik uygulama hatası: {str(e)}")
        st.error("**Detaylı Hata Bilgisi:**")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Uygulamayı Yeniden Başlat", type="primary"):
            st.rerun()

# ============================================================================
# END OF APPLICATION
# ============================================================================
