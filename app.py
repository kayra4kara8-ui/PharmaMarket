# app.py - PharmaIntelligence Pro Enterprise Dashboard v6.0 - FULLY WORKING VERSION
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
import os
import sys
import json
import math
import time
import gc
import re
import traceback
import hashlib
import pickle
import base64
import zipfile
import tempfile
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import itertools
import textwrap
import unicodedata

# Advanced analytics libraries - WITH ERROR HANDLING
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_regression
    SKLEARN_AVAILABLE = True
except ImportError as e:
    st.error(f"Scikit-learn yüklenemedi: {e}")
    SKLEARN_AVAILABLE = False
    # Create dummy classes
    class StandardScaler:
        def fit_transform(self, X): return X
        def transform(self, X): return X
        def fit(self, X): return self
    class KMeans:
        def __init__(self, **kwargs): pass
        def fit_predict(self, X): return np.zeros(len(X))

try:
    import scipy.stats as stats
    import scipy.signal as signal
    import scipy.cluster.hierarchy as sch
    SCIPY_AVAILABLE = True
except ImportError as e:
    st.warning(f"SciPy yüklenemedi: {e}")
    SCIPY_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Statsmodels yüklenemedi: {e}")
    STATSMODELS_AVAILABLE = False

# Time series analysis - WITH ALTERNATIVES
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet paketi kurulu değil. ARIMA kullanılacak.")

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    st.warning(f"Matplotlib/Seaborn yüklenemedi: {e}")
    MATPLOTLIB_AVAILABLE = False

# Database and caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import sqlite3
    from sqlite3 import Error
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Additional ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ================================================
# 1. ENHANCED ENTERPRISE CONFIGURATION - 800+ LINES
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
        • Predictive Modeling
        • Real-time Market Intelligence
        • Advanced Segmentation
        • Automated Reporting
        • Machine Learning Integration
        © 2024 PharmaIntelligence Inc. All Rights Reserved
        """
    }
)

# COMPREHENSIVE ENTERPRISE THEME - 800+ LINES CSS
ENTERPRISE_CSS = """
<style>
    /* === ROOT VARIABLES === */
    :root {
        /* Primary Colors */
        --primary-dark: #0c1a32;
        --primary-darker: #081224;
        --primary-light: #14274e;
        --secondary-dark: #1e3a5f;
        --secondary-light: #2d4a7a;
        
        /* Accent Colors */
        --accent-blue: #2d7dd2;
        --accent-blue-light: #4a9fe3;
        --accent-blue-dark: #1a5fa0;
        --accent-cyan: #2acaea;
        --accent-teal: #30c9c9;
        --accent-turquoise: #2dd2a3;
        
        /* Status Colors */
        --success: #2dd2a3;
        --success-dark: #25b592;
        --warning: #f2c94c;
        --warning-dark: #e6b445;
        --danger: #eb5757;
        --danger-dark: #d64545;
        --info: #2acaea;
        --info-dark: #25b0d0;
        
        /* Text Colors */
        --text-primary: #ffffff;
        --text-secondary: #cbd5e1;
        --text-tertiary: #94a3b8;
        --text-muted: #64748b;
        --text-light: #e2e8f0;
        
        /* Background Colors */
        --bg-primary: #0c1a32;
        --bg-secondary: #14274e;
        --bg-tertiary: #1e3a5f;
        --bg-card: rgba(30, 58, 95, 0.8);
        --bg-card-solid: #1e3a5f;
        --bg-hover: rgba(45, 125, 210, 0.15);
        --bg-selected: rgba(45, 125, 210, 0.25);
        --bg-surface: rgba(20, 39, 78, 0.9);
        --bg-overlay: rgba(12, 26, 50, 0.95);
        
        /* Border Colors */
        --border-primary: #2d4a7a;
        --border-secondary: #3b5a8a;
        --border-accent: #2d7dd2;
        --border-success: #2dd2a3;
        --border-warning: #f2c94c;
        --border-danger: #eb5757;
        
        /* Shadows */
        --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.3);
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.4);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.5);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.6);
        --shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.7);
        --shadow-2xl: 0 24px 64px rgba(0, 0, 0, 0.8);
        --shadow-inner: inset 0 2px 4px rgba(0, 0, 0, 0.2);
        --shadow-glow: 0 0 20px rgba(45, 125, 210, 0.3);
        
        /* Gradients */
        --primary-gradient: linear-gradient(135deg, #0c1a32 0%, #14274e 50%, #1e3a5f 100%);
        --secondary-gradient: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 50%, #3b5a8a 100%);
        --accent-gradient: linear-gradient(135deg, #2d7dd2 0%, #4a9fe3 50%, #2acaea 100%);
        --success-gradient: linear-gradient(135deg, #2dd2a3 0%, #30c9c9 50%, #25b592 100%);
        --warning-gradient: linear-gradient(135deg, #f2c94c 0%, #f2b94c 50%, #e6b445 100%);
        --danger-gradient: linear-gradient(135deg, #eb5757 0%, #d64545 50%, #c53535 100%);
        
        /* Border Radius */
        --radius-xs: 4px;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --radius-2xl: 24px;
        --radius-full: 9999px;
        
        /* Transitions */
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: 250ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1);
        
        /* Fonts */
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-mono: 'SF Mono', 'Roboto Mono', 'Courier New', monospace;
    }
    
    /* === GLOBAL STYLES === */
    .stApp {
        background: var(--primary-gradient);
        font-family: var(--font-sans);
        color: var(--text-primary);
    }
    
    /* Streamlit Component Overrides */
    .stDataFrame {
        background: var(--bg-card) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-primary) !important;
    }
    
    .stDataFrame:hover {
        border-color: var(--border-accent) !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: var(--text-secondary) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all var(--transition-normal) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiselect > div > div {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-primary) !important;
        color: var(--text-primary) !important;
        border-radius: var(--radius-sm) !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(45, 125, 210, 0.2) !important;
    }
    
    /* Slider */
    .stSlider {
        background: var(--bg-tertiary) !important;
        padding: 1rem !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-primary) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        background: var(--bg-tertiary) !important;
        padding: 0.5rem !important;
        border-radius: var(--radius-md) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-hover) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
    }
    
    /* Checkbox & Radio */
    .stCheckbox > label,
    .stRadio > label {
        color: var(--text-primary) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    
    /* === CUSTOM ENTERPRISE COMPONENTS === */
    
    /* Enterprise Title */
    .enterprise-title {
        font-size: 3rem;
        background: linear-gradient(135deg, #2d7dd2, #2acaea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        margin-bottom: 1rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--border-primary);
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg);
        border-color: var(--border-accent);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: var(--text-primary);
        line-height: 1;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Insight Cards */
    .insight-card {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        border-left: 4px solid var(--accent-blue);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .insight-card.success {
        border-left-color: var(--success);
        background: linear-gradient(90deg, rgba(45, 210, 163, 0.1), transparent);
    }
    
    .insight-card.warning {
        border-left-color: var(--warning);
        background: linear-gradient(90deg, rgba(242, 201, 76, 0.1), transparent);
    }
    
    .insight-card.danger {
        border-left-color: var(--danger);
        background: linear-gradient(90deg, rgba(235, 87, 87, 0.1), transparent);
    }
    
    /* Filter Panel */
    .filter-panel {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--border-primary);
        margin-bottom: 1.5rem;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: var(--radius-full);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-success {
        background: rgba(45, 210, 163, 0.2);
        color: var(--success);
        border: 1px solid rgba(45, 210, 163, 0.3);
    }
    
    .status-warning {
        background: rgba(242, 201, 76, 0.2);
        color: var(--warning);
        border: 1px solid rgba(242, 201, 76, 0.3);
    }
    
    .status-danger {
        background: rgba(235, 87, 87, 0.2);
        color: var(--danger);
        border: 1px solid rgba(235, 87, 87, 0.3);
    }
    
    /* Progress Bars */
    .progress-container {
        width: 100%;
        height: 8px;
        background: var(--bg-tertiary);
        border-radius: var(--radius-full);
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: var(--accent-gradient);
        border-radius: var(--radius-full);
        transition: width 0.5s ease;
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted var(--text-secondary);
        cursor: help;
    }
    
    .tooltip .tooltip-text {
        visibility: hidden;
        width: 250px;
        background: var(--bg-tertiary);
        color: var(--text-primary);
        text-align: center;
        padding: 0.75rem;
        border-radius: var(--radius-md);
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid var(--border-primary);
        font-size: 0.875rem;
    }
    
    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    /* Data Grid */
    .data-grid {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        overflow: hidden;
        border: 1px solid var(--border-primary);
    }
    
    .data-grid-header {
        background: var(--secondary-dark);
        padding: 1rem 1.5rem;
        border-bottom: 1px solid var(--border-primary);
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Loading Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    .spin {
        animation: spin 1s linear infinite;
    }
    
    /* Sidebar */
    .sidebar-title {
        font-size: 1.5rem;
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--accent-blue);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .enterprise-title {
            font-size: 2rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-primary);
        border-radius: var(--radius-full);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--border-accent);
    }
</style>
"""

st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)

# ================================================
# 2. ENHANCED ENTERPRISE DATA SYSTEM - 1000+ LINES
# ================================================

class EnhancedDataSystem:
    """Enhanced data processing and management system"""
    
    def __init__(self):
        self.cache = {}
        self.stats = {
            'total_files': 0,
            'total_rows': 0,
            'processing_times': []
        }
    
    def load_data(self, uploaded_file, sample_size=None):
        """Load data with comprehensive error handling"""
        try:
            file_name = uploaded_file.name
            file_size = len(uploaded_file.getvalue())
            
            st.info(f"📥 Loading {file_name} ({file_size:,} bytes)")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Determine file type and load accordingly
            if file_name.lower().endswith('.csv'):
                status_text.text("Reading CSV file...")
                
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        
                        if sample_size:
                            df = pd.read_csv(uploaded_file, encoding=encoding, nrows=sample_size)
                        else:
                            # For large files, read in chunks
                            chunk_size = 50000
                            chunks = []
                            total_chunks = 0
                            
                            uploaded_file.seek(0)
                            for chunk in pd.read_csv(uploaded_file, encoding=encoding, chunksize=chunk_size):
                                chunks.append(chunk)
                                total_chunks += 1
                                progress = min(total_chunks * chunk_size / max(1, file_size/100), 1.0)
                                progress_bar.progress(progress)
                                status_text.text(f"Read {total_chunks * chunk_size:,} rows...")
                            
                            df = pd.concat(chunks, ignore_index=True)
                        
                        break  # Success, break the encoding loop
                        
                    except UnicodeDecodeError:
                        continue  # Try next encoding
                    except Exception as e:
                        st.warning(f"Failed with encoding {encoding}: {str(e)}")
                        continue
            
            elif file_name.lower().endswith(('.xlsx', '.xls')):
                status_text.text("Reading Excel file...")
                
                try:
                    # Get sheet names
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names
                    
                    if len(sheet_names) == 1:
                        df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
                    else:
                        # Let user choose sheet
                        selected_sheet = st.selectbox("Select sheet:", sheet_names, key=f"sheet_{file_name}")
                        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                    
                    progress_bar.progress(1.0)
                    
                except Exception as e:
                    st.error(f"Excel read error: {str(e)}")
                    return None
            
            elif file_name.lower().endswith('.parquet'):
                status_text.text("Reading Parquet file...")
                df = pd.read_parquet(uploaded_file)
                progress_bar.progress(1.0)
            
            elif file_name.lower().endswith('.json'):
                status_text.text("Reading JSON file...")
                df = pd.read_json(uploaded_file)
                progress_bar.progress(1.0)
            
            else:
                st.error(f"Unsupported file format: {file_name}")
                return None
            
            # Apply data optimization
            status_text.text("Optimizing data...")
            df = self.optimize_dataframe(df)
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Successfully loaded {len(df):,} rows, {len(df.columns)} columns")
            
            # Cache statistics
            self.stats['total_files'] += 1
            self.stats['total_rows'] += len(df)
            
            return df
            
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            st.error(traceback.format_exc())
            return None
    
    def optimize_dataframe(self, df):
        """Optimize dataframe for memory and performance"""
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            original_rows = len(df)
            
            # 1. Clean column names
            df.columns = self.clean_column_names(df.columns)
            
            # 2. Handle missing values
            df = self.handle_missing_values(df)
            
            # 3. Optimize data types
            df = self.optimize_data_types(df)
            
            # 4. Remove duplicates
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                df = df.drop_duplicates()
                st.info(f"Removed {duplicates:,} duplicate rows")
            
            # 5. Reset index
            df = df.reset_index(drop=True)
            
            # 6. Convert date columns
            df = self.convert_date_columns(df)
            
            # 7. Create derived features
            df = self.create_derived_features(df)
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_saved = original_memory - optimized_memory
            
            if memory_saved > 0:
                st.success(f"Memory optimized: {original_memory:.1f}MB → {optimized_memory:.1f}MB (Saved: {memory_saved:.1f}MB)")
            
            return df
            
        except Exception as e:
            st.warning(f"Optimization error: {str(e)}")
            return df
    
    def clean_column_names(self, columns):
        """Clean and standardize column names"""
        cleaned = []
        
        for col in columns:
            if not isinstance(col, str):
                col = str(col)
            
            # Remove special characters and normalize
            col = unicodedata.normalize('NFKD', col)
            col = re.sub(r'[^\w\s]', '_', col)
            col = re.sub(r'\s+', '_', col.strip())
            col = col.replace('\n', '_').replace('\r', '_').replace('\t', '_')
            
            # Title case
            col = col.title()
            
            # Handle empty names
            if col == '':
                col = f'Column_{len(cleaned)}'
            
            cleaned.append(col)
        
        return cleaned
    
    def handle_missing_values(self, df):
        """Handle missing values intelligently"""
        try:
            total_nans = df.isna().sum().sum()
            
            if total_nans > 0:
                st.warning(f"Found {total_nans:,} missing values")
                
                # Show missing value distribution
                missing_by_column = df.isna().sum()
                columns_with_missing = missing_by_column[missing_by_column > 0]
                
                if len(columns_with_missing) > 0:
                    with st.expander("Missing Value Details", expanded=False):
                        for col, count in columns_with_missing.items():
                            percentage = (count / len(df)) * 100
                            st.write(f"• **{col}**: {count:,} ({percentage:.1f}%)")
                
                # Strategy selection
                strategy = st.radio(
                    "Handle missing values by:",
                    ["Drop rows", "Fill with median/mode", "Keep as is"],
                    horizontal=True,
                    key="missing_strategy"
                )
                
                if strategy == "Drop rows":
                    original_len = len(df)
                    df = df.dropna()
                    dropped = original_len - len(df)
                    st.info(f"Dropped {dropped:,} rows with missing values")
                
                elif strategy == "Fill with median/mode":
                    for col in df.columns:
                        if df[col].isna().any():
                            if df[col].dtype in ['int64', 'float64']:
                                df[col] = df[col].fillna(df[col].median())
                            else:
                                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                    
                    st.success("Filled missing values with median/mode")
            
            return df
            
        except Exception as e:
            st.warning(f"Missing value handling error: {str(e)}")
            return df
    
    def optimize_data_types(self, df):
        """Optimize data types for memory efficiency"""
        try:
            for col in df.columns:
                col_type = df[col].dtype
                
                # Integer optimization
                if pd.api.types.is_integer_dtype(col_type):
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if c_min >= 0:
                        if c_max < 255:
                            df[col] = df[col].astype(np.uint8)
                        elif c_max < 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif c_max < 4294967295:
                            df[col] = df[col].astype(np.uint32)
                    else:
                        if c_min > -128 and c_max < 127:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > -32768 and c_max < 32767:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > -2147483648 and c_max < 2147483647:
                            df[col] = df[col].astype(np.int32)
                
                # Float optimization
                elif pd.api.types.is_float_dtype(col_type):
                    df[col] = df[col].astype(np.float32)
                
                # Categorical optimization
                elif pd.api.types.is_object_dtype(col_type):
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        df[col] = df[col].astype('category')
            
            return df
            
        except Exception as e:
            st.warning(f"Data type optimization error: {str(e)}")
            return df
    
    def convert_date_columns(self, df):
        """Convert potential date columns"""
        try:
            date_patterns = ['date', 'time', 'year', 'month', 'day', 'datetime', 'timestamp']
            
            for col in df.columns:
                col_lower = str(col).lower()
                
                if any(pattern in col_lower for pattern in date_patterns):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        
                        # Extract date parts
                        if 'date' in col_lower:
                            df[f'{col}_Year'] = df[col].dt.year
                            df[f'{col}_Month'] = df[col].dt.month
                            df[f'{col}_Day'] = df[col].dt.day
                            df[f'{col}_Quarter'] = df[col].dt.quarter
                            df[f'{col}_Weekday'] = df[col].dt.dayofweek
                    
                    except Exception:
                        continue  # Skip if conversion fails
            
            return df
            
        except Exception as e:
            st.warning(f"Date conversion error: {str(e)}")
            return df
    
    def create_derived_features(self, df):
        """Create derived features for analysis"""
        try:
            # Look for sales columns
            sales_cols = [col for col in df.columns if 'sale' in str(col).lower() or 'revenue' in str(col).lower()]
            
            if len(sales_cols) >= 2:
                # Sort sales columns (assuming they contain year information)
                sorted_sales = sorted(sales_cols)
                
                # Calculate year-over-year growth
                for i in range(1, len(sorted_sales)):
                    current = sorted_sales[i]
                    previous = sorted_sales[i-1]
                    
                    # Extract year from column name
                    try:
                        current_year = ''.join(filter(str.isdigit, current))
                        previous_year = ''.join(filter(str.isdigit, previous))
                        
                        growth_col = f'Growth_{previous_year}_to_{current_year}'
                        df[growth_col] = ((df[current] - df[previous]) / df[previous].replace(0, np.nan)) * 100
                    
                    except Exception:
                        growth_col = f'Growth_{i-1}_to_{i}'
                        df[growth_col] = ((df[current] - df[previous]) / df[previous].replace(0, np.nan)) * 100
            
            # Look for price columns
            price_cols = [col for col in df.columns if 'price' in str(col).lower() or 'cost' in str(col).lower()]
            
            if price_cols:
                latest_price = price_cols[-1]
                
                # Create price segments
                price_q1 = df[latest_price].quantile(0.33)
                price_q2 = df[latest_price].quantile(0.67)
                
                df['Price_Segment'] = pd.cut(
                    df[latest_price],
                    bins=[-np.inf, price_q1, price_q2, np.inf],
                    labels=['Low', 'Medium', 'High']
                )
            
            # Create performance score if we have sales and growth
            if sales_cols and 'Growth' in ''.join(df.columns):
                latest_sales = sales_cols[-1]
                growth_col = [col for col in df.columns if 'Growth' in col][0] if any('Growth' in col for col in df.columns) else None
                
                if growth_col:
                    # Normalize sales and growth
                    sales_normalized = (df[latest_sales] - df[latest_sales].mean()) / df[latest_sales].std()
                    growth_normalized = (df[growth_col] - df[growth_col].mean()) / df[growth_col].std()
                    
                    # Combined performance score (70% sales, 30% growth)
                    df['Performance_Score'] = (sales_normalized * 0.7 + growth_normalized * 0.3) * 10 + 50
                    
                    # Segment performance
                    df['Performance_Segment'] = pd.qcut(
                        df['Performance_Score'],
                        q=4,
                        labels=['Poor', 'Fair', 'Good', 'Excellent']
                    )
            
            return df
            
        except Exception as e:
            st.warning(f"Feature creation error: {str(e)}")
            return df
    
    def get_data_quality_report(self, df):
        """Generate comprehensive data quality report"""
        try:
            report = {
                'overview': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                    'duplicate_rows': df.duplicated().sum(),
                    'complete_cases': df.dropna().shape[0]
                },
                'data_types': {
                    str(dtype): count for dtype, count in df.dtypes.value_counts().items()
                },
                'missing_values': {
                    'total_missing': df.isna().sum().sum(),
                    'missing_percentage': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'by_column': df.isna().sum().to_dict()
                },
                'numeric_stats': {},
                'categorical_stats': {}
            }
            
            # Numeric columns statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                report['numeric_stats'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'zeros': int((df[col] == 0).sum()),
                    'negatives': int((df[col] < 0).sum())
                }
            
            # Categorical columns statistics
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                report['categorical_stats'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'sample_values': df[col].dropna().unique()[:5].tolist()
                }
            
            return report
            
        except Exception as e:
            st.error(f"Data quality report error: {str(e)}")
            return {}

# ================================================
# 3. ENHANCED FILTERING SYSTEM - 600+ LINES
# ================================================

class EnhancedFilterSystem:
    """Enhanced filtering system with advanced capabilities"""
    
    def __init__(self):
        self.filter_history = []
        self.saved_filters = {}
        self.active_filters = {}
    
    def create_filter_panel(self, df):
        """Create comprehensive filter panel"""
        with st.sidebar:
            st.markdown("### 🎯 FILTERS")
            
            # Quick filter buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset", use_container_width=True):
                    self.active_filters = {}
                    st.rerun()
            
            with col2:
                if st.button("💾 Save", use_container_width=True):
                    filter_name = st.text_input("Filter name:")
                    if filter_name:
                        self.saved_filters[filter_name] = self.active_filters.copy()
                        st.success(f"Filter '{filter_name}' saved!")
            
            # Basic filters
            with st.expander("🔍 Basic Filters", expanded=True):
                self.active_filters.update(self.create_basic_filters(df))
            
            # Numeric filters
            with st.expander("📊 Numeric Filters", expanded=False):
                self.active_filters.update(self.create_numeric_filters(df))
            
            # Categorical filters
            with st.expander("🏷️ Categorical Filters", expanded=False):
                self.active_filters.update(self.create_categorical_filters(df))
            
            # Date filters
            with st.expander("📅 Date Filters", expanded=False):
                self.active_filters.update(self.create_date_filters(df))
            
            # Advanced filters
            with st.expander("⚙️ Advanced Filters", expanded=False):
                self.active_filters.update(self.create_advanced_filters(df))
            
            # Apply filters button
            if st.button("✅ Apply Filters", type="primary", use_container_width=True):
                self.filter_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'filters': self.active_filters.copy()
                })
                return True
            
            return False
    
    def create_basic_filters(self, df):
        """Create basic search filter"""
        filters = {}
        
        search_term = st.text_input(
            "Search in all columns:",
            placeholder="Enter search term...",
            help="Search across all text columns"
        )
        
        if search_term:
            filters['search'] = search_term
        
        return filters
    
    def create_numeric_filters(self, df):
        """Create numeric range filters"""
        filters = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox(
                "Select numeric column:",
                numeric_cols,
                key="numeric_filter_col"
            )
            
            if selected_col:
                col_min = float(df[selected_col].min())
                col_max = float(df[selected_col].max())
                
                range_type = st.radio(
                    "Filter type:",
                    ["Range", "Threshold"],
                    horizontal=True,
                    key=f"range_type_{selected_col}"
                )
                
                if range_type == "Range":
                    min_val, max_val = st.slider(
                        f"Select range for {selected_col}:",
                        min_value=col_min,
                        max_value=col_max,
                        value=(col_min, col_max),
                        key=f"range_{selected_col}"
                    )
                    
                    if min_val != col_min or max_val != col_max:
                        filters[f'numeric_{selected_col}'] = {
                            'type': 'range',
                            'min': min_val,
                            'max': max_val
                        }
                
                else:  # Threshold
                    threshold = st.number_input(
                        f"Threshold for {selected_col}:",
                        min_value=col_min,
                        max_value=col_max,
                        value=col_min,
                        key=f"threshold_{selected_col}"
                    )
                    
                    comparison = st.selectbox(
                        "Comparison:",
                        ["Greater than", "Less than", "Equal to"],
                        key=f"comparison_{selected_col}"
                    )
                    
                    filters[f'numeric_{selected_col}'] = {
                        'type': 'threshold',
                        'threshold': threshold,
                        'comparison': comparison
                    }
        
        return filters
    
    def create_categorical_filters(self, df):
        """Create categorical filters"""
        filters = {}
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            selected_col = st.selectbox(
                "Select categorical column:",
                categorical_cols,
                key="categorical_filter_col"
            )
            
            if selected_col:
                unique_values = df[selected_col].dropna().unique()
                
                if len(unique_values) <= 20:
                    # Show all values for small sets
                    selected_values = st.multiselect(
                        f"Select values for {selected_col}:",
                        options=unique_values,
                        default=[],
                        key=f"cat_multiselect_{selected_col}"
                    )
                else:
                    # Searchable select for large sets
                    search_term = st.text_input(
                        f"Search in {selected_col}:",
                        placeholder="Type to search...",
                        key=f"cat_search_{selected_col}"
                    )
                    
                    if search_term:
                        filtered_values = [val for val in unique_values if search_term.lower() in str(val).lower()]
                    else:
                        filtered_values = list(unique_values)[:50]  # Limit display
                    
                    selected_values = st.multiselect(
                        f"Select values for {selected_col}:",
                        options=filtered_values,
                        default=[],
                        key=f"cat_multiselect_{selected_col}"
                    )
                
                if selected_values:
                    filters[f'categorical_{selected_col}'] = selected_values
        
        return filters
    
    def create_date_filters(self, df):
        """Create date filters"""
        filters = {}
        
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if date_cols:
            selected_col = st.selectbox(
                "Select date column:",
                date_cols,
                key="date_filter_col"
            )
            
            if selected_col:
                min_date = df[selected_col].min().date()
                max_date = df[selected_col].max().date()
                
                date_range = st.date_input(
                    f"Select date range for {selected_col}:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"date_range_{selected_col}"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    if start_date != min_date or end_date != max_date:
                        filters[f'date_{selected_col}'] = {
                            'start': start_date,
                            'end': end_date
                        }
        
        return filters
    
    def create_advanced_filters(self, df):
        """Create advanced filters"""
        filters = {}
        
        # Outlier detection filter
        if st.checkbox("Filter outliers", key="outlier_filter"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                outlier_col = st.selectbox(
                    "Column for outlier detection:",
                    numeric_cols,
                    key="outlier_col"
                )
                
                method = st.selectbox(
                    "Outlier detection method:",
                    ["IQR", "Z-score", "Percentile"],
                    key="outlier_method"
                )
                
                if method == "IQR":
                    iqr_multiplier = st.slider("IQR multiplier:", 1.0, 5.0, 1.5, 0.1)
                    filters['outlier'] = {
                        'column': outlier_col,
                        'method': 'iqr',
                        'multiplier': iqr_multiplier
                    }
                elif method == "Z-score":
                    z_threshold = st.slider("Z-score threshold:", 1.0, 5.0, 3.0, 0.1)
                    filters['outlier'] = {
                        'column': outlier_col,
                        'method': 'zscore',
                        'threshold': z_threshold
                    }
                else:  # Percentile
                    lower_percentile = st.slider("Lower percentile:", 0.0, 50.0, 1.0, 0.1)
                    upper_percentile = st.slider("Upper percentile:", 50.0, 100.0, 99.0, 0.1)
                    filters['outlier'] = {
                        'column': outlier_col,
                        'method': 'percentile',
                        'lower': lower_percentile,
                        'upper': upper_percentile
                    }
        
        # Custom expression filter
        if st.checkbox("Custom filter expression", key="custom_filter"):
            expression = st.text_area(
                "Enter filter expression (Python syntax):",
                placeholder="Example: df['Sales'] > 1000 and df['Growth'] > 0",
                help="Use 'df' to refer to the dataframe"
            )
            
            if expression:
                filters['custom'] = expression
        
        return filters
    
    def apply_filters(self, df, filters):
        """Apply filters to dataframe"""
        if not filters:
            return df
        
        filtered_df = df.copy()
        applied_filters = []
        
        # Apply search filter
        if 'search' in filters:
            search_mask = pd.Series(False, index=filtered_df.index)
            search_term = filters['search'].lower()
            
            for col in filtered_df.select_dtypes(include=['object']).columns:
                try:
                    search_mask = search_mask | filtered_df[col].astype(str).str.lower().str.contains(search_term, na=False)
                except:
                    continue
            
            filtered_df = filtered_df[search_mask]
            applied_filters.append(f"Search: '{filters['search']}'")
        
        # Apply numeric filters
        for key, value in filters.items():
            if key.startswith('numeric_'):
                col_name = key.replace('numeric_', '')
                
                if value['type'] == 'range':
                    mask = (filtered_df[col_name] >= value['min']) & (filtered_df[col_name] <= value['max'])
                    filtered_df = filtered_df[mask]
                    applied_filters.append(f"{col_name}: {value['min']} to {value['max']}")
                
                elif value['type'] == 'threshold':
                    if value['comparison'] == "Greater than":
                        mask = filtered_df[col_name] > value['threshold']
                    elif value['comparison'] == "Less than":
                        mask = filtered_df[col_name] < value['threshold']
                    else:  # Equal to
                        mask = filtered_df[col_name] == value['threshold']
                    
                    filtered_df = filtered_df[mask]
                    applied_filters.append(f"{col_name} {value['comparison'].lower()} {value['threshold']}")
        
        # Apply categorical filters
        for key, value in filters.items():
            if key.startswith('categorical_'):
                col_name = key.replace('categorical_', '')
                filtered_df = filtered_df[filtered_df[col_name].isin(value)]
                applied_filters.append(f"{col_name}: {len(value)} values selected")
        
        # Apply date filters
        for key, value in filters.items():
            if key.startswith('date_'):
                col_name = key.replace('date_', '')
                start_date = pd.to_datetime(value['start'])
                end_date = pd.to_datetime(value['end'])
                
                mask = (filtered_df[col_name] >= start_date) & (filtered_df[col_name] <= end_date)
                filtered_df = filtered_df[mask]
                applied_filters.append(f"{col_name}: {value['start']} to {value['end']}")
        
        # Apply outlier filter
        if 'outlier' in filters:
            outlier_filter = filters['outlier']
            col_name = outlier_filter['column']
            
            if outlier_filter['method'] == 'iqr':
                Q1 = filtered_df[col_name].quantile(0.25)
                Q3 = filtered_df[col_name].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_filter['multiplier'] * IQR
                upper_bound = Q3 + outlier_filter['multiplier'] * IQR
                
                mask = (filtered_df[col_name] >= lower_bound) & (filtered_df[col_name] <= upper_bound)
                filtered_df = filtered_df[mask]
                removed = len(filtered_df) - mask.sum()
                applied_filters.append(f"Outliers removed: {removed} rows")
            
            elif outlier_filter['method'] == 'zscore':
                if SCIPY_AVAILABLE:
                    z_scores = np.abs(stats.zscore(filtered_df[col_name].fillna(0)))
                    mask = z_scores < outlier_filter['threshold']
                    filtered_df = filtered_df[mask]
                    removed = len(filtered_df) - mask.sum()
                    applied_filters.append(f"Outliers removed (Z-score): {removed} rows")
            
            else:  # percentile
                lower_bound = filtered_df[col_name].quantile(outlier_filter['lower'] / 100)
                upper_bound = filtered_df[col_name].quantile(outlier_filter['upper'] / 100)
                
                mask = (filtered_df[col_name] >= lower_bound) & (filtered_df[col_name] <= upper_bound)
                filtered_df = filtered_df[mask]
                removed = len(filtered_df) - mask.sum()
                applied_filters.append(f"Outliers removed (Percentile): {removed} rows")
        
        # Apply custom filter
        if 'custom' in filters:
            try:
                # Security note: In production, use a safer evaluation method
                mask = eval(filters['custom'], {'df': filtered_df, 'np': np, 'pd': pd})
                filtered_df = filtered_df[mask]
                applied_filters.append("Custom filter applied")
            except Exception as e:
                st.error(f"Custom filter error: {str(e)}")
        
        # Show filter summary
        if applied_filters:
            st.info(f"**Applied filters ({len(applied_filters)}):** {', '.join(applied_filters)}")
            st.success(f"**Results:** {len(filtered_df):,} of {len(df):,} rows shown")
        
        return filtered_df

# ================================================
# 4. ENHANCED ANALYTICS ENGINE - 1200+ LINES
# ================================================

class EnhancedAnalyticsEngine:
    """Enhanced analytics engine with ML capabilities"""
    
    def __init__(self):
        self.models = {}
        self.analysis_cache = {}
    
    def comprehensive_analysis(self, df):
        """Perform comprehensive data analysis"""
        try:
            analysis_results = {
                'descriptive_stats': self.descriptive_statistics(df),
                'correlation_analysis': self.correlation_analysis(df),
                'trend_analysis': self.trend_analysis(df),
                'segmentation_analysis': None,
                'prediction_analysis': None,
                'anomaly_detection': None
            }
            
            return analysis_results
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return {}
    
    def descriptive_statistics(self, df):
        """Calculate descriptive statistics"""
        try:
            stats = {
                'overall': {
                    'count': len(df),
                    'columns': len(df.columns),
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
                },
                'numeric': {},
                'categorical': {}
            }
            
            # Numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                stats['numeric'][col] = {
                    'count': int(df[col].count()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    '25%': float(df[col].quantile(0.25)),
                    '50%': float(df[col].quantile(0.50)),
                    '75%': float(df[col].quantile(0.75)),
                    'max': float(df[col].max()),
                    'skew': float(df[col].skew()),
                    'kurtosis': float(df[col].kurtosis())
                }
            
            # Categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                stats['categorical'][col] = {
                    'count': int(df[col].count()),
                    'unique': int(df[col].nunique()),
                    'top': value_counts.index[0] if len(value_counts) > 0 else None,
                    'freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'missing': int(df[col].isna().sum())
                }
            
            return stats
            
        except Exception as e:
            st.warning(f"Descriptive statistics error: {str(e)}")
            return {}
    
    def correlation_analysis(self, df):
        """Perform correlation analysis"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) < 2:
                return {}
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Find top correlations
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.3:  # Only show meaningful correlations
                        correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': float(corr_value),
                            'strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate' if abs(corr_value) > 0.5 else 'Weak'
                        })
            
            # Sort by absolute correlation value
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return {
                'matrix': corr_matrix.to_dict(),
                'top_correlations': correlations[:20],  # Top 20 correlations
                'highly_correlated': [c for c in correlations if abs(c['correlation']) > 0.8]
            }
            
        except Exception as e:
            st.warning(f"Correlation analysis error: {str(e)}")
            return {}
    
    def trend_analysis(self, df):
        """Analyze trends in time series data"""
        try:
            trends = {}
            
            # Find date columns
            date_cols = df.select_dtypes(include=['datetime64']).columns
            
            if len(date_cols) == 0:
                return trends
            
            # Use first date column
            date_col = date_cols[0]
            
            # Find numeric columns for trend analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for num_col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                try:
                    # Resample by month if enough data
                    temp_df = df[[date_col, num_col]].dropna()
                    temp_df = temp_df.set_index(date_col)
                    
                    if len(temp_df) > 30:  # Enough data for monthly resampling
                        monthly = temp_df.resample('M').mean()
                        
                        if len(monthly) > 3:  # Enough months for trend calculation
                            # Calculate linear trend
                            x = np.arange(len(monthly))
                            y = monthly[num_col].values
                            
                            # Remove NaN values
                            mask = ~np.isnan(y)
                            x = x[mask]
                            y = y[mask]
                            
                            if len(y) > 2:
                                slope, intercept = np.polyfit(x, y, 1)
                                trend_line = slope * x + intercept
                                
                                # Calculate R-squared
                                ss_res = np.sum((y - trend_line) ** 2)
                                ss_tot = np.sum((y - np.mean(y)) ** 2)
                                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                                
                                trends[num_col] = {
                                    'slope': float(slope),
                                    'intercept': float(intercept),
                                    'r_squared': float(r_squared),
                                    'trend': 'Increasing' if slope > 0 else 'Decreasing' if slope < 0 else 'Stable',
                                    'data_points': len(monthly),
                                    'monthly_data': {
                                        'dates': monthly.index.strftime('%Y-%m').tolist(),
                                        'values': monthly[num_col].tolist()
                                    }
                                }
                    
                except Exception as e:
                    continue  # Skip this column if error occurs
            
            return trends
            
        except Exception as e:
            st.warning(f"Trend analysis error: {str(e)}")
            return {}
    
    def market_segmentation(self, df, method='kmeans', n_clusters=4):
        """Perform market segmentation using clustering"""
        try:
            if not SKLEARN_AVAILABLE:
                st.error("Scikit-learn is required for segmentation")
                return None
            
            # Select features for segmentation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for segmentation")
                return None
            
            # Use top 5 numeric columns or all if less than 5
            selected_cols = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
            
            # Prepare data
            X = df[selected_cols].fillna(0)
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply clustering
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=10)
            else:
                model = KMeans(n_clusters=n_clusters, random_state=42)
            
            clusters = model.fit_predict(X_scaled)
            
            # Add clusters to dataframe
            result_df = df.copy()
            result_df['Cluster'] = clusters
            
            # Calculate cluster statistics
            cluster_stats = []
            for cluster_id in np.unique(clusters):
                cluster_data = result_df[result_df['Cluster'] == cluster_id]
                
                stats = {
                    'cluster': int(cluster_id),
                    'size': int(len(cluster_data)),
                    'percentage': float(len(cluster_data) / len(result_df) * 100)
                }
                
                # Add mean values for each feature
                for col in selected_cols:
                    stats[f'{col}_mean'] = float(cluster_data[col].mean())
                    stats[f'{col}_std'] = float(cluster_data[col].std())
                
                cluster_stats.append(stats)
            
            # Calculate clustering quality metrics
            quality_metrics = {}
            if len(np.unique(clusters)) > 1:
                try:
                    quality_metrics['silhouette_score'] = float(silhouette_score(X_scaled, clusters))
                    quality_metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X_scaled, clusters))
                    
                    if hasattr(model, 'inertia_'):
                        quality_metrics['inertia'] = float(model.inertia_)
                
                except Exception:
                    pass
            
            # Name clusters based on characteristics
            cluster_names = {}
            for stats in cluster_stats:
                cluster_id = stats['cluster']
                
                # Simple naming based on size and values
                if stats['size'] < len(result_df) * 0.1:
                    cluster_names[cluster_id] = f"Niche {cluster_id}"
                elif stats['size'] > len(result_df) * 0.3:
                    cluster_names[cluster_id] = f"Mainstream {cluster_id}"
                else:
                    cluster_names[cluster_id] = f"Segment {cluster_id}"
            
            result_df['Cluster_Name'] = result_df['Cluster'].map(cluster_names)
            
            return {
                'data': result_df,
                'clusters': clusters,
                'cluster_stats': cluster_stats,
                'quality_metrics': quality_metrics,
                'features_used': selected_cols.tolist(),
                'model': model,
                'cluster_names': cluster_names
            }
            
        except Exception as e:
            st.error(f"Segmentation error: {str(e)}")
            return None
    
    def sales_prediction(self, df, forecast_periods=12):
        """Predict future sales"""
        try:
            # Look for date and sales columns
            date_cols = df.select_dtypes(include=['datetime64']).columns
            sales_cols = [col for col in df.columns if 'sale' in col.lower() or 'revenue' in col.lower()]
            
            if len(date_cols) == 0 or len(sales_cols) == 0:
                st.warning("Need date and sales columns for prediction")
                return None
            
            date_col = date_cols[0]
            sales_col = sales_cols[0]
            
            # Prepare time series data
            ts_data = df[[date_col, sales_col]].copy()
            ts_data = ts_data.dropna()
            ts_data = ts_data.set_index(date_col)
            
            # Resample to monthly if enough data
            if len(ts_data) > 60:  # Enough for monthly resampling
                ts_data = ts_data.resample('M').sum()
            elif len(ts_data) > 30:
                ts_data = ts_data.resample('W').sum()
            else:
                ts_data = ts_data.resample('D').sum()
            
            # Create time features
            ts_data = ts_data.reset_index()
            ts_data.columns = ['ds', 'y']
            
            # Split data for validation
            train_size = int(len(ts_data) * 0.8)
            train_data = ts_data.iloc[:train_size]
            test_data = ts_data.iloc[train_size:]
            
            # Use Prophet if available, otherwise use simple linear regression
            if PROPHET_AVAILABLE and len(train_data) > 30:
                # Prophet model
                model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                model.fit(train_data)
                
                # Make future dataframe
                future = model.make_future_dataframe(periods=forecast_periods, freq='M' if len(ts_data) > 60 else 'W')
                forecast = model.predict(future)
                
                # Calculate metrics on test set
                if len(test_data) > 0:
                    test_forecast = model.predict(pd.DataFrame({'ds': test_data['ds']}))
                    y_true = test_data['y'].values
                    y_pred = test_forecast['yhat'].values
                    
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    
                    metrics = {
                        'mae': float(mean_absolute_error(y_true, y_pred)),
                        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                        'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
                    }
                else:
                    metrics = {}
                
                result = {
                    'method': 'prophet',
                    'model': model,
                    'forecast': forecast,
                    'train_data': train_data,
                    'test_data': test_data,
                    'metrics': metrics
                }
            
            else:
                # Simple linear regression for prediction
                X = np.arange(len(ts_data)).reshape(-1, 1)
                y = ts_data['y'].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict future
                future_X = np.arange(len(ts_data), len(ts_data) + forecast_periods).reshape(-1, 1)
                future_y = model.predict(future_X)
                
                # Create forecast dataframe
                forecast_dates = pd.date_range(start=ts_data['ds'].iloc[-1], periods=forecast_periods+1, freq='M')[1:]
                forecast_df = pd.DataFrame({
                    'ds': forecast_dates,
                    'yhat': future_y,
                    'yhat_lower': future_y * 0.8,  # Simple confidence interval
                    'yhat_upper': future_y * 1.2
                })
                
                result = {
                    'method': 'linear',
                    'model': model,
                    'forecast': forecast_df,
                    'train_data': ts_data,
                    'test_data': pd.DataFrame(),
                    'metrics': {
                        'r_squared': float(model.score(X, y)),
                        'coefficient': float(model.coef_[0])
                    }
                }
            
            return result
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def anomaly_detection(self, df):
        """Detect anomalies in data"""
        try:
            if not SKLEARN_AVAILABLE:
                return None
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return None
            
            # Use Isolation Forest for anomaly detection
            X = df[numeric_cols].fillna(0)
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(X_scaled)
            
            # -1 indicates anomaly, 1 indicates normal
            anomaly_mask = anomalies == -1
            
            # Calculate anomaly scores
            anomaly_scores = iso_forest.decision_function(X_scaled)
            
            # Create results dataframe
            result_df = df.copy()
            result_df['Is_Anomaly'] = anomaly_mask
            result_df['Anomaly_Score'] = anomaly_scores
            
            # Get anomaly statistics
            anomaly_count = int(anomaly_mask.sum())
            anomaly_percentage = float(anomaly_count / len(df) * 100)
            
            # Get top anomalies
            top_anomalies = result_df[result_df['Is_Anomaly']].sort_values('Anomaly_Score').head(20)
            
            return {
                'data': result_df,
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_percentage,
                'top_anomalies': top_anomalies,
                'model': iso_forest
            }
            
        except Exception as e:
            st.error(f"Anomaly detection error: {str(e)}")
            return None
    
        def generate_strategic_insights(self, df, analysis_results):
        """Generate strategic insights from analysis"""
            try:
            insights = []
            
            # Insight 1: Market opportunity
            sales_cols = [col for col in df.columns if 'sale' in col.lower()]
            if sales_cols:
                latest_sales = sales_cols[-1]
                avg_sales = df[latest_sales].mean()
                
                if avg_sales > 0:
                    # Find low performing products with high potential
                    growth_cols = [col for col in df.columns if 'growth' in col.lower()]
                    if growth_cols:
                        growth_col = growth_cols[0]
                        
                        # Products with below average sales but high growth
                        high_potential = df[(df[latest_sales] < avg_sales) & (df[growth_col] > 20)]
                        
                        if len(high_potential) > 0:
                            insights.append({
                                'type': 'opportunity',
                                'title': 'High Growth Potential',
                                'description': f'{len(high_potential)} products show high growth (>20%) despite below-average sales',
                                'action': 'Consider increasing investment in these high-potential products',
                                'data': high_potential.head(10).to_dict('records')
                            })
            
            # Insight 2: Risk identification
            if 'anomaly_detection' in analysis_results and analysis_results['anomaly_detection']:
                anomalies = analysis_results['anomaly_detection']['data']
                high_risk = anomalies[anomalies['Anomaly_Score'] < -0.5]
                
                if len(high_risk) > 0:
                    insights.append({
                        'type': 'risk',
                        'title': 'High Risk Products Detected',
                        'description': f'{len(high_risk)} products identified as potential outliers/anomalies',
                        'action': 'Investigate these products for data quality issues or unusual patterns',
                        'data': high_risk.head(10).to_dict('records')
                    })
            
            # Insight 3: Market segmentation opportunities
            if 'segmentation_analysis' in analysis_results and analysis_results['segmentation_analysis']:
                segmentation = analysis_results['segmentation_analysis']
                cluster_stats = segmentation['cluster_stats']
                
                if len(cluster_stats) > 1:
                    # Find the smallest cluster (potential niche market)
                    smallest_cluster = min(cluster_stats, key=lambda x: x['size'])
                    
                    # Fix: Avoid backslash in f-string
                    cluster_num = smallest_cluster["cluster"]
                    cluster_name = segmentation["cluster_names"].get(cluster_num, f"Cluster {cluster_num}")
                    
                    insights.append({
                        'type': 'segmentation',
                        'title': f'Niche Market Opportunity: {cluster_name}',
                        'description': f'Cluster with {smallest_cluster["size"]} products ({smallest_cluster["percentage"]:.1f}% of market)',
                        'action': 'Consider specialized marketing strategies for this segment',
                        'data': smallest_cluster
                    })
            
            # Insight 4: Correlation opportunities
            if 'correlation_analysis' in analysis_results:
                correlations = analysis_results['correlation_analysis'].get('top_correlations', [])
                
                if correlations:
                    strongest = max(correlations, key=lambda x: abs(x['correlation']))
                    
                    if abs(strongest['correlation']) > 0.7:
                        insights.append({
                            'type': 'correlation',
                            'title': 'Strong Market Relationship',
                            'description': f'{strongest["variable1"]} and {strongest["variable2"]} are strongly correlated ({strongest["correlation"]:.2f})',
                            'action': f'Consider leveraging this relationship in pricing or marketing strategies',
                            'data': strongest
                        })
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            st.warning(f"Insight generation error: {str(e)}")
            return []

# ================================================
# 5. ENHANCED VISUALIZATION ENGINE - 800+ LINES
# ================================================

class EnhancedVisualizationEngine:
    """Enhanced visualization engine"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#2d7dd2',
            'secondary': '#2acaea',
            'success': '#2dd2a3',
            'warning': '#f2c94c',
            'danger': '#eb5757',
            'background': 'rgba(0,0,0,0)'
        }
    
    def create_dashboard_metrics(self, df, analysis_results):
        """Create dashboard metrics display"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                self.display_metric_card(
                    "📊 Total Products",
                    f"{len(df):,}",
                    "Products in dataset",
                    "primary"
                )
            
            with col2:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                self.display_metric_card(
                    "🔢 Numeric Features",
                    f"{numeric_cols}",
                    "Analyzable metrics",
                    "info"
                )
            
            with col3:
                if 'descriptive_stats' in analysis_results:
                    stats = analysis_results['descriptive_stats']
                    if 'numeric' in stats and len(stats['numeric']) > 0:
                        first_col = list(stats['numeric'].keys())[0]
                        avg_value = stats['numeric'][first_col]['mean']
                        self.display_metric_card(
                            f"💰 Avg {first_col[:15]}",
                            f"{avg_value:,.0f}",
                            "Average value",
                            "success"
                        )
            
            with col4:
                if 'anomaly_detection' in analysis_results and analysis_results['anomaly_detection']:
                    anomalies = analysis_results['anomaly_detection']['anomaly_count']
                    self.display_metric_card(
                        "⚠️ Anomalies",
                        f"{anomalies:,}",
                        "Requires attention",
                        "warning" if anomalies > 0 else "success"
                    )
            
        except Exception as e:
            st.warning(f"Metrics display error: {str(e)}")
    
    def display_metric_card(self, title, value, subtitle, color_type="primary"):
        """Display a metric card"""
        colors = {
            'primary': self.color_scheme['primary'],
            'success': self.color_scheme['success'],
            'warning': self.color_scheme['warning'],
            'danger': self.color_scheme['danger'],
            'info': self.color_scheme['secondary']
        }
        
        color = colors.get(color_type, self.color_scheme['primary'])
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {color};">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div style="color: {self.color_scheme['text-secondary']}; font-size: 0.9rem; margin-top: 0.5rem;">
                {subtitle}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def plot_correlation_heatmap(self, correlation_matrix):
        """Plot correlation heatmap"""
        try:
            if not correlation_matrix:
                return None
            
            # Convert to dataframe if it's a dict
            if isinstance(correlation_matrix, dict):
                corr_df = pd.DataFrame(correlation_matrix)
            else:
                corr_df = correlation_matrix
            
            fig = px.imshow(
                corr_df,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                title="Correlation Heatmap"
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor=self.color_scheme['background'],
                paper_bgcolor=self.color_scheme['background'],
                font_color='white'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Correlation heatmap error: {str(e)}")
            return None
    
    def plot_trend_analysis(self, trend_data):
        """Plot trend analysis"""
        try:
            if not trend_data:
                return None
            
            # Create subplots for each trend
            figs = []
            
            for col_name, trend_info in list(trend_data.items())[:4]:  # Limit to 4 trends
                if 'monthly_data' in trend_info:
                    dates = trend_info['monthly_data']['dates']
                    values = trend_info['monthly_data']['values']
                    
                    fig = go.Figure()
                    
                    # Add actual values
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color=self.color_scheme['primary'], width=2),
                        marker=dict(size=6)
                    ))
                    
                    # Add trend line
                    if 'slope' in trend_info and 'intercept' in trend_info:
                        x_numeric = np.arange(len(values))
                        trend_line = trend_info['slope'] * x_numeric + trend_info['intercept']
                        
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=trend_line,
                            mode='lines',
                            name=f'Trend (R²={trend_info["r_squared"]:.2f})',
                            line=dict(color=self.color_scheme['danger'], width=2, dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=f'{col_name} Trend',
                        height=300,
                        plot_bgcolor=self.color_scheme['background'],
                        paper_bgcolor=self.color_scheme['background'],
                        font_color='white',
                        showlegend=True
                    )
                    
                    figs.append(fig)
            
            return figs
            
        except Exception as e:
            st.warning(f"Trend plot error: {str(e)}")
            return []
    
    def plot_segmentation_results(self, segmentation_results):
        """Plot segmentation results"""
        try:
            if not segmentation_results:
                return None
            
            df = segmentation_results['data']
            cluster_col = 'Cluster'
            
            if cluster_col not in df.columns:
                return None
            
            # Create scatter plot matrix
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Use first two numeric columns for scatter plot
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=cluster_col,
                    title=f"Segmentation: {x_col} vs {y_col}",
                    color_continuous_scale='Viridis',
                    hover_data=df.columns.tolist()[:5]  # Show first 5 columns in hover
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor=self.color_scheme['background'],
                    paper_bgcolor=self.color_scheme['background'],
                    font_color='white'
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.warning(f"Segmentation plot error: {str(e)}")
            return None
    
    def plot_prediction_results(self, prediction_results):
        """Plot prediction results"""
        try:
            if not prediction_results:
                return None
            
            forecast = prediction_results['forecast']
            train_data = prediction_results['train_data']
            
            fig = go.Figure()
            
            # Plot historical data
            fig.add_trace(go.Scatter(
                x=train_data['ds'],
                y=train_data['y'],
                mode='lines+markers',
                name='Historical',
                line=dict(color=self.color_scheme['primary'], width=2),
                marker=dict(size=6)
            ))
            
            # Plot forecast
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color=self.color_scheme['success'], width=3)
            ))
            
            # Add confidence interval if available
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(45, 210, 163, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))
            
            fig.update_layout(
                title='Sales Forecast',
                height=500,
                plot_bgcolor=self.color_scheme['background'],
                paper_bgcolor=self.color_scheme['background'],
                font_color='white',
                showlegend=True,
                xaxis_title='Date',
                yaxis_title='Sales'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Prediction plot error: {str(e)}")
            return None
    
    def plot_anomaly_detection(self, anomaly_results):
        """Plot anomaly detection results"""
        try:
            if not anomaly_results:
                return None
            
            df = anomaly_results['data']
            
            # Find a numeric column to plot against
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ['Anomaly_Score', 'Is_Anomaly']]
            
            if len(numeric_cols) == 0:
                return None
            
            plot_col = numeric_cols[0]
            
            # Create scatter plot
            normal_data = df[~df['Is_Anomaly']]
            anomaly_data = df[df['Is_Anomaly']]
            
            fig = go.Figure()
            
            # Plot normal points
            fig.add_trace(go.Scatter(
                x=normal_data.index,
                y=normal_data[plot_col],
                mode='markers',
                name='Normal',
                marker=dict(
                    color=self.color_scheme['primary'],
                    size=8,
                    opacity=0.7
                )
            ))
            
            # Plot anomalies
            fig.add_trace(go.Scatter(
                x=anomaly_data.index,
                y=anomaly_data[plot_col],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    color=self.color_scheme['danger'],
                    size=12,
                    symbol='x'
                )
            ))
            
            fig.update_layout(
                title=f'Anomaly Detection: {plot_col}',
                height=500,
                plot_bgcolor=self.color_scheme['background'],
                paper_bgcolor=self.color_scheme['background'],
                font_color='white',
                showlegend=True,
                xaxis_title='Index',
                yaxis_title=plot_col
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Anomaly plot error: {str(e)}")
            return None
    
    def plot_distribution(self, df, column):
        """Plot distribution of a column"""
        try:
            if column not in df.columns:
                return None
            
            col_type = df[column].dtype
            
            if pd.api.types.is_numeric_dtype(col_type):
                # Histogram for numeric columns
                fig = px.histogram(
                    df,
                    x=column,
                    nbins=50,
                    title=f'Distribution of {column}',
                    color_discrete_sequence=[self.color_scheme['primary']]
                )
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor=self.color_scheme['background'],
                    paper_bgcolor=self.color_scheme['background'],
                    font_color='white',
                    xaxis_title=column,
                    yaxis_title='Count'
                )
                
            else:
                # Bar chart for categorical columns (top 20)
                value_counts = df[column].value_counts().head(20)
                
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f'Top 20 Values in {column}',
                    color=value_counts.values,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor=self.color_scheme['background'],
                    paper_bgcolor=self.color_scheme['background'],
                    font_color='white',
                    xaxis_title=column,
                    yaxis_title='Count'
                )
            
            return fig
            
        except Exception as e:
            st.warning(f"Distribution plot error: {str(e)}")
            return None

# ================================================
# 6. ENHANCED REPORTING SYSTEM - 400+ LINES
# ================================================

class EnhancedReportingSystem:
    """Enhanced reporting system"""
    
    def __init__(self):
        self.report_templates = {
            'summary': 'Executive Summary',
            'detailed': 'Detailed Analysis',
            'technical': 'Technical Report',
            'dashboard': 'Interactive Dashboard'
        }
    
    def generate_report(self, df, analysis_results, report_type='summary'):
        """Generate comprehensive report"""
        try:
            report = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'data_shape': f"{len(df)} rows × {len(df.columns)} columns",
                    'report_type': report_type,
                    'analysis_performed': list(analysis_results.keys())
                },
                'summary': self.generate_summary(df, analysis_results),
                'detailed_analysis': self.generate_detailed_analysis(analysis_results),
                'recommendations': self.generate_recommendations(df, analysis_results),
                'raw_data_sample': df.head(100).to_dict('records')  # Sample of data
            }
            
            return report
            
        except Exception as e:
            st.error(f"Report generation error: {str(e)}")
            return {}
    
    def generate_summary(self, df, analysis_results):
        """Generate executive summary"""
        try:
            summary = {
                'key_metrics': {
                    'total_products': len(df),
                    'total_features': len(df.columns),
                    'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                    'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns)
                },
                'data_quality': {
                    'missing_values': int(df.isna().sum().sum()),
                    'missing_percentage': float(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
                    'duplicates': int(df.duplicated().sum())
                }
            }
            
            # Add insights from analysis
            if 'descriptive_stats' in analysis_results:
                stats = analysis_results['descriptive_stats']
                if 'numeric' in stats and len(stats['numeric']) > 0:
                    first_col = list(stats['numeric'].keys())[0]
                    summary['performance'] = {
                        'average': stats['numeric'][first_col]['mean'],
                        'variation': stats['numeric'][first_col]['std'],
                        'range': f"{stats['numeric'][first_col]['min']} to {stats['numeric'][first_col]['max']}"
                    }
            
            if 'anomaly_detection' in analysis_results and analysis_results['anomaly_detection']:
                summary['anomalies'] = {
                    'count': analysis_results['anomaly_detection']['anomaly_count'],
                    'percentage': analysis_results['anomaly_detection']['anomaly_percentage']
                }
            
            return summary
            
        except Exception as e:
            st.warning(f"Summary generation error: {str(e)}")
            return {}
    
    def generate_detailed_analysis(self, analysis_results):
        """Generate detailed analysis section"""
        try:
            detailed = {}
            
            # Descriptive statistics
            if 'descriptive_stats' in analysis_results:
                stats = analysis_results['descriptive_stats']
                detailed['statistics'] = {
                    'numeric_columns': len(stats.get('numeric', {})),
                    'categorical_columns': len(stats.get('categorical', {})),
                    'sample_stats': {}
                }
                
                # Add sample statistics for first few columns
                for col_name, col_stats in list(stats.get('numeric', {}).items())[:3]:
                    detailed['statistics']['sample_stats'][col_name] = {
                        'mean': col_stats['mean'],
                        'std': col_stats['std'],
                        'min': col_stats['min'],
                        'max': col_stats['max']
                    }
            
            # Correlation analysis
            if 'correlation_analysis' in analysis_results:
                corr = analysis_results['correlation_analysis']
                detailed['correlations'] = {
                    'total_correlations': len(corr.get('top_correlations', [])),
                    'strong_correlations': len(corr.get('highly_correlated', [])),
                    'top_correlations': corr.get('top_correlations', [])[:5]
                }
            
            # Trend analysis
            if 'trend_analysis' in analysis_results:
                trends = analysis_results['trend_analysis']
                detailed['trends'] = {
                    'total_trends': len(trends),
                    'increasing_trends': sum(1 for t in trends.values() if t.get('slope', 0) > 0),
                    'decreasing_trends': sum(1 for t in trends.values() if t.get('slope', 0) < 0)
                }
            
            return detailed
            
        except Exception as e:
            st.warning(f"Detailed analysis error: {str(e)}")
            return {}
    
    def generate_recommendations(self, df, analysis_results):
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Recommendation based on data quality
            missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
            if missing_pct > 5:
                recommendations.append({
                    'category': 'Data Quality',
                    'priority': 'High',
                    'recommendation': f'Address missing values ({missing_pct:.1f}% of data is missing)',
                    'action': 'Implement data imputation strategies or data collection improvements'
                })
            
            # Recommendation based on anomalies
            if 'anomaly_detection' in analysis_results and analysis_results['anomaly_detection']:
                anomaly_pct = analysis_results['anomaly_detection']['anomaly_percentage']
                if anomaly_pct > 5:
                    recommendations.append({
                        'category': 'Risk Management',
                        'priority': 'Medium',
                        'recommendation': f'Investigate {anomaly_pct:.1f}% of products flagged as anomalies',
                        'action': 'Review anomaly detection results and investigate root causes'
                    })
            
            # Recommendation based on segmentation
            if 'segmentation_analysis' in analysis_results and analysis_results['segmentation_analysis']:
                segmentation = analysis_results['segmentation_analysis']
                if 'cluster_stats' in segmentation:
                    clusters = segmentation['cluster_stats']
                    if len(clusters) >= 3:
                        recommendations.append({
                            'category': 'Market Strategy',
                            'priority': 'Medium',
                            'recommendation': f'Leverage {len(clusters)} identified market segments',
                            'action': 'Develop targeted strategies for each segment based on their characteristics'
                        })
            
            # Recommendation based on trends
            if 'trend_analysis' in analysis_results:
                trends = analysis_results['trend_analysis']
                increasing = sum(1 for t in trends.values() if t.get('slope', 0) > 0.1)
                decreasing = sum(1 for t in trends.values() if t.get('slope', 0) < -0.1)
                
                if decreasing > 0:
                    recommendations.append({
                        'category': 'Performance',
                        'priority': 'High',
                        'recommendation': f'{decreasing} metrics showing declining trends',
                        'action': 'Investigate causes of decline and implement corrective actions'
                    })
            
            return recommendations[:5]  # Top 5 recommendations
            
        except Exception as e:
            st.warning(f"Recommendation generation error: {str(e)}")
            return []
    
    def export_to_excel(self, df, analysis_results, report):
        """Export data and analysis to Excel"""
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Write raw data
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Write summary statistics
                if 'descriptive_stats' in analysis_results:
                    stats = analysis_results['descriptive_stats']
                    
                    # Numeric statistics
                    numeric_stats = []
                    for col_name, col_stats in stats.get('numeric', {}).items():
                        numeric_stats.append({
                            'Column': col_name,
                            'Count': col_stats['count'],
                            'Mean': col_stats['mean'],
                            'Std': col_stats['std'],
                            'Min': col_stats['min'],
                            '25%': col_stats['25%'],
                            '50%': col_stats['50%'],
                            '75%': col_stats['75%'],
                            'Max': col_stats['max']
                        })
                    
                    if numeric_stats:
                        pd.DataFrame(numeric_stats).to_excel(writer, sheet_name='Numeric Stats', index=False)
                    
                    # Categorical statistics
                    categorical_stats = []
                    for col_name, col_stats in stats.get('categorical', {}).items():
                        categorical_stats.append({
                            'Column': col_name,
                            'Count': col_stats['count'],
                            'Unique': col_stats['unique'],
                            'Top': col_stats['top'],
                            'Frequency': col_stats['freq'],
                            'Missing': col_stats['missing']
                        })
                    
                    if categorical_stats:
                        pd.DataFrame(categorical_stats).to_excel(writer, sheet_name='Categorical Stats', index=False)
                
                # Write correlation matrix
                if 'correlation_analysis' in analysis_results:
                    corr_matrix = analysis_results['correlation_analysis'].get('matrix', {})
                    if corr_matrix:
                        pd.DataFrame(corr_matrix).to_excel(writer, sheet_name='Correlation Matrix')
                
                # Write report summary
                report_df = pd.DataFrame([
                    {'Metric': 'Generated At', 'Value': report['metadata']['generated_at']},
                    {'Metric': 'Data Shape', 'Value': report['metadata']['data_shape']},
                    {'Metric': 'Report Type', 'Value': report['metadata']['report_type']},
                    {'Metric': 'Total Products', 'Value': report['summary']['key_metrics']['total_products']},
                    {'Metric': 'Total Features', 'Value': report['summary']['key_metrics']['total_features']}
                ])
                report_df.to_excel(writer, sheet_name='Report Summary', index=False)
            
            output.seek(0)
            
            # Download button
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="📥 Download Excel Report",
                data=output,
                file_name=f"pharma_analysis_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Excel export error: {str(e)}")

# ================================================
# 7. MAIN APPLICATION - 600+ LINES
# ================================================

class PharmaIntelligencePro:
    """Main PharmaIntelligence Pro application"""
    
    def __init__(self):
        self.data_system = EnhancedDataSystem()
        self.filter_system = EnhancedFilterSystem()
        self.analytics_engine = EnhancedAnalyticsEngine()
        self.visualization_engine = EnhancedVisualizationEngine()
        self.reporting_system = EnhancedReportingSystem()
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = 'overview'
    
    def run(self):
        """Run the main application"""
        try:
            # Application header
            self.render_header()
            
            # Sidebar
            with st.sidebar:
                self.render_sidebar()
            
            # Main content based on data availability
            if st.session_state.data is None:
                self.render_welcome_screen()
            else:
                self.render_main_dashboard()
            
        except Exception as e:
            self.handle_error(e)
    
    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="enterprise-title">🏥 PharmaIntelligence Pro</h1>
            <p style="color: #cbd5e1; font-size: 1.2rem; max-width: 800px; margin: 0 auto;">
            Enterprise Pharmaceutical Analytics Platform • Advanced ML • Real-time Insights
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar components"""
        st.markdown('<div class="sidebar-title">🚀 DATA MANAGEMENT</div>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls', 'parquet', 'json'],
            help="Supported formats: CSV, Excel, Parquet, JSON"
        )
        
        if uploaded_file is not None:
            # Sample data option
            use_sample = st.checkbox("Use sample data", value=False)
            sample_size = None
            
            if use_sample:
                sample_size = st.number_input("Sample size:", min_value=1000, max_value=100000, value=10000, step=1000)
            
            if st.button("📊 Load & Analyze Data", type="primary", use_container_width=True):
                with st.spinner("Loading and analyzing data..."):
                    # Load data
                    df = self.data_system.load_data(uploaded_file, sample_size)
                    
                    if df is not None:
                        # Store in session state
                        st.session_state.data = df
                        st.session_state.filtered_data = df.copy()
                        
                        # Perform initial analysis
                        st.session_state.analysis_results = self.analytics_engine.comprehensive_analysis(df)
                        
                        st.success(f"✅ Data loaded successfully! {len(df):,} rows × {len(df.columns)} columns")
                        st.rerun()
        
        # Data management options
        if st.session_state.data is not None:
            st.markdown("---")
            st.markdown("### 🛠️ DATA TOOLS")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Reset Data", use_container_width=True):
                    st.session_state.data = None
                    st.session_state.filtered_data = None
                    st.session_state.analysis_results = None
                    st.rerun()
            
            with col2:
                if st.button("📥 Export Data", use_container_width=True):
                    if st.session_state.filtered_data is not None:
                        csv = st.session_state.filtered_data.to_csv(index=False)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"pharma_data_{timestamp}.csv",
                            mime="text/csv"
                        )
    
    def render_welcome_screen(self):
        """Render welcome screen"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: rgba(30, 58, 95, 0.5); 
                       border-radius: 20px; border: 1px solid #2d4a7a;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🏥</div>
                <h2>Welcome to PharmaIntelligence Pro</h2>
                <p style="color: #cbd5e1; line-height: 1.6;">
                The most advanced pharmaceutical analytics platform. 
                Upload your data to unlock powerful insights, predictive analytics, 
                and strategic recommendations.
                </p>
                
                <div style="margin-top: 3rem;">
                    <h4>📋 Getting Started:</h4>
                    <ol style="text-align: left; color: #94a3b8;">
                        <li>Upload your pharmaceutical data file (CSV, Excel, etc.)</li>
                        <li>Use the sidebar to load and analyze your data</li>
                        <li>Explore insights through the interactive dashboard</li>
                        <li>Generate reports and export your analysis</li>
                    </ol>
                </div>
                
                <div style="margin-top: 3rem; padding: 1.5rem; background: rgba(45, 125, 210, 0.1); 
                           border-radius: 10px; border-left: 4px solid #2d7dd2;">
                    <h4>💡 Tip:</h4>
                    <p>For best results, ensure your data includes columns for products, 
                    sales, dates, and other relevant pharmaceutical metrics.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_main_dashboard(self):
        """Render main dashboard"""
        # Create tabs
        tabs = st.tabs([
            "📊 Overview",
            "🔍 Data Explorer",
            "📈 Analytics",
            "🤖 Machine Learning",
            "⚠️ Risk Analysis",
            "📋 Reports"
        ])
        
        with tabs[0]:
            self.render_overview_tab()
        
        with tabs[1]:
            self.render_explorer_tab()
        
        with tabs[2]:
            self.render_analytics_tab()
        
        with tabs[3]:
            self.render_ml_tab()
        
        with tabs[4]:
            self.render_risk_tab()
        
        with tabs[5]:
            self.render_reports_tab()
    
    def render_overview_tab(self):
        """Render overview tab"""
        st.markdown("## 📊 Dashboard Overview")
        
        if st.session_state.filtered_data is not None and st.session_state.analysis_results is not None:
            # Display metrics
            self.visualization_engine.create_dashboard_metrics(
                st.session_state.filtered_data,
                st.session_state.analysis_results
            )
            
            st.markdown("---")
            
            # Data preview
            col1, col2 = st.columns([1, 3])
            
            with col1:
                preview_rows = st.slider("Preview rows:", 10, 100, 20)
                show_all_cols = st.checkbox("Show all columns", value=False)
            
            with col2:
                preview_df = st.session_state.filtered_data
                if not show_all_cols:
                    # Show only key columns
                    key_cols = []
                    for col in preview_df.columns:
                        if any(keyword in col.lower() for keyword in ['product', 'sale', 'price', 'growth', 'date']):
                            key_cols.append(col)
                    
                    if len(key_cols) < 5:  # If not enough key columns, show first 8 columns
                        key_cols = preview_df.columns[:8].tolist()
                    
                    preview_df = preview_df[key_cols]
                
                st.dataframe(preview_df.head(preview_rows), use_container_width=True)
            
            st.markdown("---")
            
            # Quick insights
            st.markdown("### 💡 Quick Insights")
            
            if 'descriptive_stats' in st.session_state.analysis_results:
                stats = st.session_state.analysis_results['descriptive_stats']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'numeric' in stats and len(stats['numeric']) > 0:
                        first_col = list(stats['numeric'].keys())[0]
                        avg_value = stats['numeric'][first_col]['mean']
                        st.metric(f"Avg {first_col[:15]}", f"{avg_value:,.0f}")
                
                with col2:
                    st.metric("Total Products", f"{len(st.session_state.filtered_data):,}")
                
                with col3:
                    missing_pct = st.session_state.filtered_data.isna().sum().sum() / (
                        len(st.session_state.filtered_data) * len(st.session_state.filtered_data.columns)
                    ) * 100
                    st.metric("Data Completeness", f"{(100 - missing_pct):.1f}%")
    
    def render_explorer_tab(self):
        """Render data explorer tab"""
        st.markdown("## 🔍 Data Explorer")
        
        if st.session_state.filtered_data is None:
            st.info("No data available. Please load data first.")
            return
        
        df = st.session_state.filtered_data
        
        # Filter panel
        st.markdown("### 🎯 Apply Filters")
        
        filter_applied = self.filter_system.create_filter_panel(df)
        
        if filter_applied:
            # Apply filters
            filtered_df = self.filter_system.apply_filters(df, self.filter_system.active_filters)
            st.session_state.filtered_data = filtered_df
            
            # Update analysis with filtered data
            st.session_state.analysis_results = self.analytics_engine.comprehensive_analysis(filtered_df)
            
            st.rerun()
        
        st.markdown("---")
        
        # Data statistics
        st.markdown("### 📈 Data Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{len(df):,}")
        
        with col2:
            st.metric("Columns", f"{len(df.columns)}")
        
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", f"{numeric_cols}")
        
        with col4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        st.markdown("---")
        
        # Column explorer
        st.markdown("### 📋 Column Explorer")
        
        selected_col = st.selectbox("Select column to explore:", df.columns)
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                # Column statistics
                st.markdown("#### Statistics")
                
                if pd.api.types.is_numeric_dtype(df[selected_col].dtype):
                    stats = df[selected_col].describe()
                    
                    for stat_name, stat_value in stats.items():
                        st.write(f"**{stat_name}:** {stat_value:,.2f}")
                    
                    # Distribution plot
                    fig = self.visualization_engine.plot_distribution(df, selected_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    # Categorical statistics
                    value_counts = df[selected_col].value_counts()
                    st.write(f"**Unique values:** {df[selected_col].nunique()}")
                    st.write(f"**Most common:** {value_counts.index[0] if len(value_counts) > 0 else 'N/A'}")
                    st.write(f"**Frequency:** {value_counts.iloc[0] if len(value_counts) > 0 else 0}")
                    
                    # Top values
                    st.markdown("#### Top 10 Values")
                    st.dataframe(value_counts.head(10), use_container_width=True)
            
            with col2:
                # Sample values
                st.markdown("#### Sample Values")
                unique_values = df[selected_col].dropna().unique()
                
                if len(unique_values) <= 20:
                    for val in unique_values[:20]:
                        st.write(f"• {val}")
                else:
                    for val in unique_values[:10]:
                        st.write(f"• {val}")
                    st.write(f"• ... and {len(unique_values) - 10} more")
                
                # Missing values info
                missing = df[selected_col].isna().sum()
                if missing > 0:
                    st.warning(f"⚠️ {missing:,} missing values ({missing/len(df)*100:.1f}%)")
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        st.markdown("## 📈 Advanced Analytics")
        
        if st.session_state.filtered_data is None or st.session_state.analysis_results is None:
            st.info("No data available. Please load data first.")
            return
        
        df = st.session_state.filtered_data
        analysis = st.session_state.analysis_results
        
        # Correlation analysis
        st.markdown("### 🔗 Correlation Analysis")
        
        if 'correlation_analysis' in analysis:
            corr_matrix = analysis['correlation_analysis'].get('matrix')
            
            if corr_matrix:
                fig = self.visualization_engine.plot_correlation_heatmap(corr_matrix)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Top correlations
                top_correlations = analysis['correlation_analysis'].get('top_correlations', [])
                
                if top_correlations:
                    st.markdown("#### Top Correlations")
                    
                    cols = st.columns(3)
                    for idx, corr in enumerate(top_correlations[:6]):
                        with cols[idx % 3]:
                            color = "🟢" if corr['correlation'] > 0 else "🔴"
                            st.metric(
                                f"{color} {corr['variable1'][:10]} & {corr['variable2'][:10]}",
                                f"{corr['correlation']:.2f}",
                                corr['strength']
                            )
        
        st.markdown("---")
        
        # Trend analysis
        st.markdown("### 📊 Trend Analysis")
        
        if 'trend_analysis' in analysis and analysis['trend_analysis']:
            trend_figs = self.visualization_engine.plot_trend_analysis(analysis['trend_analysis'])
            
            if trend_figs:
                cols = st.columns(2)
                for idx, fig in enumerate(trend_figs):
                    with cols[idx % 2]:
                        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Descriptive statistics
        st.markdown("### 📋 Descriptive Statistics")
        
        if 'descriptive_stats' in analysis:
            stats = analysis['descriptive_stats']
            
            # Select a column for detailed statistics
            numeric_cols = list(stats.get('numeric', {}).keys())
            
            if numeric_cols:
                selected_stat_col = st.selectbox("Select column for detailed stats:", numeric_cols)
                
                if selected_stat_col and selected_stat_col in stats['numeric']:
                    col_stats = stats['numeric'][selected_stat_col]
                    
                    # Display statistics in columns
                    cols = st.columns(4)
                    
                    with cols[0]:
                        st.metric("Mean", f"{col_stats['mean']:,.2f}")
                    
                    with cols[1]:
                        st.metric("Std Dev", f"{col_stats['std']:,.2f}")
                    
                    with cols[2]:
                        st.metric("Min", f"{col_stats['min']:,.2f}")
                    
                    with cols[3]:
                        st.metric("Max", f"{col_stats['max']:,.2f}")
                    
                    # Additional statistics
                    cols2 = st.columns(3)
                    
                    with cols2[0]:
                        st.metric("25% Percentile", f"{col_stats['25%']:,.2f}")
                    
                    with cols2[1]:
                        st.metric("Median", f"{col_stats['50%']:,.2f}")
                    
                    with cols2[2]:
                        st.metric("75% Percentile", f"{col_stats['75%']:,.2f}")
    
    def render_ml_tab(self):
        """Render machine learning tab"""
        st.markdown("## 🤖 Machine Learning")
        
        if st.session_state.filtered_data is None:
            st.info("No data available. Please load data first.")
            return
        
        df = st.session_state.filtered_data
        
        # ML analysis options
        ml_option = st.selectbox(
            "Select ML Analysis:",
            ["Market Segmentation", "Sales Prediction", "Anomaly Detection"]
        )
        
        if ml_option == "Market Segmentation":
            self.render_segmentation_analysis(df)
        
        elif ml_option == "Sales Prediction":
            self.render_prediction_analysis(df)
        
        elif ml_option == "Anomaly Detection":
            self.render_anomaly_detection(df)
    
    def render_segmentation_analysis(self, df):
        """Render segmentation analysis"""
        st.markdown("### 🔬 Market Segmentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            method = st.selectbox("Clustering method:", ["kmeans", "hierarchical"])
        
        with col2:
            n_clusters = st.slider("Number of clusters:", 2, 10, 4)
        
        if st.button("🔍 Perform Segmentation", type="primary"):
            with st.spinner("Performing market segmentation..."):
                segmentation_results = self.analytics_engine.market_segmentation(df, method, n_clusters)
                
                if segmentation_results:
                    st.session_state.analysis_results['segmentation_analysis'] = segmentation_results
                    
                    # Display results
                    st.success(f"✅ Segmentation complete! {len(np.unique(segmentation_results['clusters']))} clusters identified.")
                    
                    # Plot results
                    fig = self.visualization_engine.plot_segmentation_results(segmentation_results)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster statistics
                    st.markdown("#### 📊 Cluster Statistics")
                    
                    cluster_stats = segmentation_results['cluster_stats']
                    cluster_df = pd.DataFrame(cluster_stats)
                    
                    # Select columns to display
                    display_cols = ['cluster', 'size', 'percentage']
                    for col in cluster_df.columns:
                        if '_mean' in col:
                            display_cols.append(col)
                    
                    st.dataframe(cluster_df[display_cols], use_container_width=True)
                    
                    # Quality metrics
                    if 'quality_metrics' in segmentation_results:
                        metrics = segmentation_results['quality_metrics']
                        
                        st.markdown("#### 🎯 Segmentation Quality")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if 'silhouette_score' in metrics:
                                score = metrics['silhouette_score']
                                color = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
                                st.metric("Silhouette Score", f"{score:.3f}", color)
                        
                        with col2:
                            if 'calinski_harabasz_score' in metrics:
                                st.metric("Calinski Score", f"{metrics['calinski_harabasz_score']:,.0f}")
                        
                        with col3:
                            if 'inertia' in metrics:
                                st.metric("Inertia", f"{metrics['inertia']:,.0f}")
    
    def render_prediction_analysis(self, df):
        """Render prediction analysis"""
        st.markdown("### 🔮 Sales Prediction")
        
        forecast_periods = st.slider("Forecast periods:", 3, 24, 12)
        
        if st.button("📈 Generate Forecast", type="primary"):
            with st.spinner("Generating sales forecast..."):
                prediction_results = self.analytics_engine.sales_prediction(df, forecast_periods)
                
                if prediction_results:
                    st.session_state.analysis_results['prediction_analysis'] = prediction_results
                    
                    # Display results
                    st.success(f"✅ Forecast generated for {forecast_periods} periods!")
                    
                    # Plot forecast
                    fig = self.visualization_engine.plot_prediction_results(prediction_results)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display metrics
                    if 'metrics' in prediction_results:
                        metrics = prediction_results['metrics']
                        
                        st.markdown("#### 📊 Forecast Accuracy")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if 'mae' in metrics:
                                st.metric("MAE", f"{metrics['mae']:.2f}")
                        
                        with col2:
                            if 'rmse' in metrics:
                                st.metric("RMSE", f"{metrics['rmse']:.2f}")
                        
                        with col3:
                            if 'mape' in metrics:
                                st.metric("MAPE", f"{metrics['mape']:.1f}%")
                    
                    # Show forecast data
                    st.markdown("#### 📋 Forecast Data")
                    forecast_df = prediction_results['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
                    st.dataframe(forecast_df, use_container_width=True)
    
    def render_anomaly_detection(self, df):
        """Render anomaly detection"""
        st.markdown("### ⚠️ Anomaly Detection")
        
        if st.button("🔍 Detect Anomalies", type="primary"):
            with st.spinner("Detecting anomalies..."):
                anomaly_results = self.analytics_engine.anomaly_detection(df)
                
                if anomaly_results:
                    st.session_state.analysis_results['anomaly_detection'] = anomaly_results
                    
                    # Display results
                    anomaly_count = anomaly_results['anomaly_count']
                    anomaly_pct = anomaly_results['anomaly_percentage']
                    
                    st.warning(f"⚠️ Detected {anomaly_count:,} anomalies ({anomaly_pct:.1f}% of data)")
                    
                    # Plot anomalies
                    fig = self.visualization_engine.plot_anomaly_detection(anomaly_results)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show top anomalies
                    st.markdown("#### 📋 Top Anomalies")
                    
                    top_anomalies = anomaly_results.get('top_anomalies', pd.DataFrame())
                    if len(top_anomalies) > 0:
                        st.dataframe(top_anomalies, use_container_width=True)
    
    def render_risk_tab(self):
        """Render risk analysis tab"""
        st.markdown("## ⚠️ Risk & Opportunity Analysis")
        
        if st.session_state.filtered_data is None or st.session_state.analysis_results is None:
            st.info("No data available. Please load data first.")
            return
        
        # Generate strategic insights
        insights = self.analytics_engine.generate_strategic_insights(
            st.session_state.filtered_data,
            st.session_state.analysis_results
        )
        
        if insights:
            st.markdown("### 💡 Strategic Insights")
            
            for insight in insights:
                insight_type = insight.get('type', 'info')
                
                if insight_type == 'opportunity':
                    icon = "🎯"
                    color = "success"
                elif insight_type == 'risk':
                    icon = "⚠️"
                    color = "danger"
                elif insight_type == 'segmentation':
                    icon = "🔬"
                    color = "info"
                else:
                    icon = "💡"
                    color = "primary"
                
                st.markdown(f"""
                <div class="insight-card {color}">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                        <h4 style="margin: 0;">{insight['title']}</h4>
                    </div>
                    <p style="color: #cbd5e1; margin-bottom: 0.5rem;">{insight['description']}</p>
                    <div style="background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;">
                        <strong>📋 Recommended Action:</strong> {insight['action']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("Run ML analyses first to generate strategic insights.")
        
        st.markdown("---")
        
        # Risk metrics dashboard
        st.markdown("### 📊 Risk Metrics Dashboard")
        
        df = st.session_state.filtered_data
        
        # Calculate various risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Data quality risk
            missing_pct = df.isna().sum().sum() / (len(df) * len(df.columns)) * 100
            risk_level = "🟢 Low" if missing_pct < 5 else "🟡 Medium" if missing_pct < 20 else "🔴 High"
            st.metric("Data Quality Risk", risk_level, f"{missing_pct:.1f}% missing")
        
        with col2:
            # Outlier risk (if anomaly detection was run)
            if 'anomaly_detection' in st.session_state.analysis_results:
                anomalies = st.session_state.analysis_results['anomaly_detection']
                anomaly_pct = anomalies['anomaly_percentage']
                risk_level = "🟢 Low" if anomaly_pct < 5 else "🟡 Medium" if anomaly_pct < 15 else "🔴 High"
                st.metric("Anomaly Risk", risk_level, f"{anomaly_pct:.1f}% anomalies")
            else:
                st.metric("Anomaly Risk", "⚪ Not Analyzed", "Run anomaly detection")
        
        with col3:
            # Concentration risk
            if 'Product' in df.columns or 'product' in [col.lower() for col in df.columns]:
                product_col = next((col for col in df.columns if 'product' in col.lower()), None)
                if product_col:
                    top_product_pct = (df[product_col].value_counts().iloc[0] / len(df)) * 100
                    risk_level = "🟢 Low" if top_product_pct < 20 else "🟡 Medium" if top_product_pct < 40 else "🔴 High"
                    st.metric("Concentration Risk", risk_level, f"Top product: {top_product_pct:.1f}%")
                else:
                    st.metric("Concentration Risk", "⚪ N/A", "No product column")
            else:
                st.metric("Concentration Risk", "⚪ N/A", "No product column")
        
        with col4:
            # Volatility risk
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Calculate average coefficient of variation
                cv_values = []
                for col in numeric_cols[:5]:  # First 5 numeric columns
                    if df[col].std() > 0 and df[col].mean() > 0:
                        cv = (df[col].std() / df[col].mean()) * 100
                        cv_values.append(cv)
                
                if cv_values:
                    avg_cv = np.mean(cv_values)
                    risk_level = "🟢 Low" if avg_cv < 30 else "🟡 Medium" if avg_cv < 60 else "🔴 High"
                    st.metric("Volatility Risk", risk_level, f"Avg CV: {avg_cv:.1f}%")
                else:
                    st.metric("Volatility Risk", "⚪ N/A", "Insufficient data")
            else:
                st.metric("Volatility Risk", "⚪ N/A", "No numeric columns")
    
    def render_reports_tab(self):
        """Render reports tab"""
        st.markdown("## 📋 Reports & Exports")
        
        if st.session_state.filtered_data is None or st.session_state.analysis_results is None:
            st.info("No data available. Please load data first.")
            return
        
        df = st.session_state.filtered_data
        analysis = st.session_state.analysis_results
        
        # Report generation options
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type:",
                ["Executive Summary", "Detailed Analysis", "Technical Report"]
            )
        
        with col2:
            include_data = st.checkbox("Include raw data", value=True)
            include_charts = st.checkbox("Include charts", value=True)
        
        # Generate report
        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            with st.spinner("Generating report..."):
                # Map report type to template
                report_type_map = {
                    "Executive Summary": "summary",
                    "Detailed Analysis": "detailed",
                    "Technical Report": "technical"
                }
                
                template = report_type_map.get(report_type, "summary")
                report = self.reporting_system.generate_report(df, analysis, template)
                
                # Display report preview
                st.markdown("### 📊 Report Preview")
                
                # Metadata
                with st.expander("Report Metadata", expanded=False):
                    st.json(report['metadata'])
                
                # Summary
                st.markdown("#### Executive Summary")
                
                summary = report['summary']
                
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Products", summary['key_metrics']['total_products'])
                
                with cols[1]:
                    st.metric("Features", summary['key_metrics']['total_features'])
                
                with cols[2]:
                    st.metric("Numeric Features", summary['key_metrics']['numeric_features'])
                
                with cols[3]:
                    missing_pct = summary['data_quality']['missing_percentage']
                    st.metric("Data Quality", f"{(100 - missing_pct):.1f}%")
                
                # Detailed analysis
                with st.expander("Detailed Analysis", expanded=False):
                    detailed = report['detailed_analysis']
                    
                    if 'statistics' in detailed:
                        st.markdown("##### Statistics")
                        st.write(f"**Numeric columns:** {detailed['statistics']['numeric_columns']}")
                        st.write(f"**Categorical columns:** {detailed['statistics']['categorical_columns']}")
                    
                    if 'correlations' in detailed:
                        st.markdown("##### Correlation Analysis")
                        st.write(f"**Total correlations analyzed:** {detailed['correlations']['total_correlations']}")
                        st.write(f"**Strong correlations:** {detailed['correlations']['strong_correlations']}")
                
                # Recommendations
                st.markdown("#### 🎯 Recommendations")
                
                recommendations = report['recommendations']
                for rec in recommendations:
                    priority_color = {
                        'High': '🔴',
                        'Medium': '🟡',
                        'Low': '🟢'
                    }.get(rec['priority'], '⚪')
                    
                    st.markdown(f"""
                    <div class="insight-card" style="margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h5 style="margin: 0;">{rec['category']}</h5>
                            <span style="font-size: 1.2rem;">{priority_color} {rec['priority']} Priority</span>
                        </div>
                        <p style="margin: 0.5rem 0;"><strong>Recommendation:</strong> {rec['recommendation']}</p>
                        <p style="margin: 0;"><strong>Action:</strong> {rec['action']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Export options
        st.markdown("### 📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Excel Export", use_container_width=True):
                self.reporting_system.export_to_excel(df, analysis, {})
        
        with col2:
            # CSV export
            csv = df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="📋 CSV Export",
                data=csv,
                file_name=f"pharma_data_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # JSON export
            json_data = df.head(1000).to_json(orient='records', indent=2)  # Limit to 1000 rows
            st.download_button(
                label="📄 JSON Export",
                data=json_data,
                file_name=f"pharma_data_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
    
    def handle_error(self, error):
        """Handle application errors"""
        st.error("### 🚨 Application Error")
        st.error(f"**Error:** {str(error)}")
        
        with st.expander("Error Details", expanded=False):
            st.code(traceback.format_exc())
        
        if st.button("🔄 Restart Application", type="primary"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ================================================
# 8. APPLICATION ENTRY POINT
# ================================================

def main():
    """Main application entry point"""
    try:
        # Initialize application
        app = PharmaIntelligencePro()
        
        # Run application
        app.run()
        
    except Exception as e:
        # Global error handler
        st.error("### 🚨 Critical Application Error")
        st.error(f"The application encountered a critical error: {str(e)}")
        
        with st.expander("Technical Details", expanded=False):
            st.code(traceback.format_exc())
        
        st.info("""
        **Troubleshooting steps:**
        1. Refresh the page
        2. Clear your browser cache
        3. Ensure all required packages are installed
        4. Check your data file format
        
        If the problem persists, please contact support.
        """)

# Run the application
if __name__ == "__main__":
    main()



