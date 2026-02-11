"""
PharmaIntelligence Pro - Enterprise Pharmaceutical Analytics Platform
Version: 6.0.0
Author: PharmaIntelligence Inc.
License: Enterprise

Advanced pharmaceutical market analytics with AI-powered insights,
forecasting, anomaly detection, and comprehensive reporting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Advanced Analytics
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
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
import hashlib
from collections import defaultdict

# Excel/PDF Export
import xlsxwriter
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ================================================
# 1. CONFIGURATION & THEMING
# ================================================

st.set_page_config(
    page_title="PharmaIntelligence Pro | Enterprise Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://pharmaintelligence.com/enterprise-support',
        'Report a bug': "https://pharmaintelligence.com/enterprise-bug-report",
        'About': """
        ### PharmaIntelligence Enterprise v6.0
        • AI-Powered Forecasting
        • Anomaly Detection
        • International Product Analytics
        • Advanced Segmentation
        • Automated Reporting
        • Machine Learning Integration
        © 2024 PharmaIntelligence Inc. All Rights Reserved
        """
    }
)

# Professional Theme CSS
PROFESSIONAL_CSS = """
<style>
    /* === ROOT VARIABLES === */
    :root {
        --primary-dark: #0c1a32;
        --secondary-dark: #14274e;
        --accent-blue: #2d7dd2;
        --accent-blue-light: #4a9fe3;
        --accent-blue-dark: #1a5fa0;
        --accent-cyan: #2acaea;
        --accent-teal: #30c9c9;
        --success: #2dd2a3;
        --warning: #f2c94c;
        --danger: #eb5757;
        --info: #2d7dd2;
        
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        
        --bg-primary: #0c1a32;
        --bg-secondary: #14274e;
        --bg-card: #1e3a5f;
        --bg-hover: #2d4a7a;
        --bg-surface: #14274e;
        
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
    
    /* === GLASSMORPHISM CARDS === */
    .glass-card {
        background: rgba(30, 58, 95, 0.6);
        backdrop-filter: blur(10px);
        border-radius: var(--radius-lg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: var(--shadow-lg);
        padding: 1.5rem;
        transition: all var(--transition-normal);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-xl);
        border-color: var(--accent-blue);
    }
    
    /* === TYPOGRAPHY === */
    .pharma-title {
        font-size: 3rem;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan), var(--accent-teal));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0%, 100% { filter: hue-rotate(0deg); }
        50% { filter: hue-rotate(20deg); }
    }
    
    .pharma-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 900px;
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
        background: linear-gradient(90deg, rgba(45, 125, 210, 0.15), transparent);
        padding: 1rem;
        border-radius: var(--radius-sm);
    }
    
    /* === METRIC CARDS === */
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
        transform: translateY(-5px) scale(1.02);
        box-shadow: var(--shadow-xl);
        border-color: var(--accent-blue);
    }
    
    .custom-metric-value {
        font-size: 2.4rem;
        font-weight: 900;
        margin: 0.5rem 0;
        color: var(--text-primary);
        line-height: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .custom-metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
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
    .insight-card.success { border-left-color: var(--success); }
    .insight-card.warning { border-left-color: var(--warning); }
    .insight-card.danger { border-left-color: var(--danger); }
    
    /* === ANIMATIONS === */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* === STREAMLIT OVERRIDES === */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
    }
    
    .stDataFrame, .stTable {
        background: var(--bg-card) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--bg-hover) !important;
    }
    
    /* === FILTER SECTION === */
    .filter-status {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.2), rgba(42, 202, 234, 0.2));
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--success);
        box-shadow: var(--shadow-md);
        color: var(--text-primary);
        font-size: 0.95rem;
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
        background: rgba(45, 210, 163, 0.2);
        color: var(--success);
        border: 1px solid rgba(45, 210, 163, 0.3);
    }
    
    .badge-warning {
        background: rgba(242, 201, 76, 0.2);
        color: var(--warning);
        border: 1px solid rgba(242, 201, 76, 0.3);
    }
    
    .badge-info {
        background: rgba(45, 125, 210, 0.2);
        color: var(--accent-blue);
        border: 1px solid rgba(45, 125, 210, 0.3);
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ================================================
# 2. DATA PROCESSOR CLASS
# ================================================

class DataProcessor:
    """
    Enterprise-grade data processing engine with advanced optimization,
    cleaning, and transformation capabilities.
    """
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
    def load_large_dataset(
        file: Any,
        sample_size: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load large datasets with intelligent chunking and progress tracking.
        
        Args:
            file: Uploaded file object
            sample_size: Optional row limit for sampling
            
        Returns:
            Optimized DataFrame or None if error occurs
        """
        try:
            start_time = time.time()
            
            if file.name.endswith('.csv'):
                if sample_size:
                    df = pd.read_csv(file, nrows=sample_size)
                else:
                    with st.spinner("📥 Loading CSV data..."):
                        df = pd.read_csv(file, low_memory=False)
                        
            elif file.name.endswith(('.xlsx', '.xls')):
                if sample_size:
                    chunks = []
                    chunk_size = 50000
                    total_chunks = (sample_size // chunk_size) + 1
                    
                    with st.spinner(f"📥 Loading large dataset..."):
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
                            progress = min(loaded_rows / sample_size, 1.0)
                            
                            progress_bar.progress(progress)
                            status_text.text(f"📊 {loaded_rows:,} rows loaded...")
                            
                            if loaded_rows >= sample_size:
                                break
                        
                        df = pd.concat(chunks, ignore_index=True)
                        progress_bar.progress(1.0)
                        status_text.text(f"✅ {len(df):,} rows loaded successfully")
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                else:
                    with st.spinner(f"📥 Loading entire dataset..."):
                        df = pd.read_excel(file, engine='openpyxl')
            else:
                st.error("❌ Unsupported file format")
                return None
            
            # Optimize dataframe
            df = DataProcessor.optimize_dataframe(df)
            
            load_time = time.time() - start_time
            st.success(f"✅ Data loaded: {len(df):,} rows, {len(df.columns)} columns ({load_time:.2f}s)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Data loading error: {str(e)}")
            st.error(f"Details: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced DataFrame optimization with intelligent type conversion.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Clean column names
            df.columns = DataProcessor.clean_column_names(df.columns)
            
            with st.spinner("Optimizing dataset..."):
                
                # Optimize categorical columns
                for col in df.select_dtypes(include=['object']).columns:
                    unique_count = df[col].nunique()
                    total_rows = len(df)
                    
                    # Convert to category if cardinality < 70%
                    if unique_count < total_rows * 0.7:
                        df[col] = df[col].astype('category')
                
                # Optimize numeric columns with safe conversion
                for col in df.select_dtypes(include=[np.number]).columns:
                    try:
                        col_data = df[col]
                        
                        # Skip if all NaN
                        if col_data.isna().all():
                            continue
                        
                        col_min = col_data.min()
                        col_max = col_data.max()
                        
                        # Skip if min/max are NaN
                        if pd.isna(col_min) or pd.isna(col_max):
                            continue
                        
                        # Integer optimization
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
                            # Float optimization
                            df[col] = df[col].astype(np.float32)
                    except Exception:
                        continue
                
                # String cleaning
                for col in df.select_dtypes(include=['object', 'category']).columns:
                    try:
                        df[col] = df[col].astype(str).str.strip()
                    except:
                        pass
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_savings = original_memory - optimized_memory
            
            if memory_savings > 0:
                st.success(f"💾 Memory optimization: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({memory_savings/original_memory*100:.1f}% savings)")
            
            return df
            
        except Exception as e:
            st.warning(f"Optimization warning: {str(e)}")
            return df
    
    @staticmethod
    def clean_column_names(columns: List[str]) -> List[str]:
        """
        Clean and standardize column names with deduplication.
        
        Args:
            columns: List of column names
            
        Returns:
            List of cleaned, unique column names
        """
        cleaned = []
        seen_names = {}
        
        for col in columns:
            if isinstance(col, str):
                # Turkish character mapping
                tr_map = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in tr_map.items():
                    col = col.replace(tr, en)
                
                # Clean whitespace
                col = col.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                col = ' '.join(col.split())
                
                # Standardized mappings
                col = DataProcessor._apply_column_mappings(col)
                
                col = col.strip()
            
            # Deduplication
            original_col = str(col)
            if original_col in seen_names:
                seen_names[original_col] += 1
                col = f"{original_col}_{seen_names[original_col]}"
            else:
                seen_names[original_col] = 0
            
            cleaned.append(str(col).strip())
        
        return cleaned
    
    @staticmethod
    def _apply_column_mappings(col: str) -> str:
        """Apply domain-specific column name mappings."""
        
        # Sales columns
        if "MAT Q3 2022 USD MNF" in col:
            return "Satış_2022"
        elif "MAT Q3 2023 USD MNF" in col:
            return "Satış_2023"
        elif "MAT Q3 2024 USD MNF" in col:
            return "Satış_2024"
        
        # Unit columns
        elif "MAT Q3 2022 Units" in col:
            return "Birim_2022"
        elif "MAT Q3 2023 Units" in col:
            return "Birim_2023"
        elif "MAT Q3 2024 Units" in col:
            return "Birim_2024"
        
        # Price columns
        elif "MAT Q3 2022 Unit Avg Price USD MNF" in col:
            return "Ort_Fiyat_2022"
        elif "MAT Q3 2023 Unit Avg Price USD MNF" in col:
            return "Ort_Fiyat_2023"
        elif "MAT Q3 2024 Unit Avg Price USD MNF" in col:
            return "Ort_Fiyat_2024"
        
        # Standard Units
        elif "MAT Q3 2022 Standard Units" in col:
            return "Standard_Units_2022"
        elif "MAT Q3 2023 Standard Units" in col:
            return "Standard_Units_2023"
        elif "MAT Q3 2024 Standard Units" in col:
            return "Standard_Units_2024"
        
        # SU Average Price
        elif "MAT Q3 2022 SU Avg Price USD MNF" in col:
            return "SU_Ort_Fiyat_2022"
        elif "MAT Q3 2023 SU Avg Price USD MNF" in col:
            return "SU_Ort_Fiyat_2023"
        elif "MAT Q3 2024 SU Avg Price USD MNF" in col:
            return "SU_Ort_Fiyat_2024"
        
        # Other columns
        elif "Source.Name" in col:
            return "Kaynak"
        elif "Country" in col:
            return "Ulke"
        elif "Sector" in col:
            return "Sektor"
        elif "Corporation" in col:
            return "Sirket"
        elif "Manufacturer" in col:
            return "Uretici"
        elif "Molecule List" in col:
            return "Molekul_Listesi"
        elif "Molecule" in col:
            return "Molekul"
        elif "Chemical Salt" in col:
            return "Kimyasal_Tuz"
        elif "International Product" in col:
            return "International_Product"
        elif "Specialty Product" in col:
            return "Ozel_Urun"
        elif "NFC123" in col:
            return "NFC123"
        elif "International Pack" in col:
            return "International_Pack"
        elif "International Strength" in col:
            return "International_Strength"
        elif "International Size" in col:
            return "International_Size"
        elif "International Volume" in col:
            return "International_Volume"
        elif "International Prescription" in col:
            return "International_Prescription"
        elif "Panel" in col:
            return "Panel"
        elif "Region" in col and "Sub-Region" not in col:
            return "Bolge"
        elif "Sub-Region" in col:
            return "Alt_Bolge"
        
        return col
    
    @staticmethod
    def extract_year_from_column(col_name: str) -> Optional[int]:
        """
        Extract 4-digit year from column name using regex.
        
        Args:
            col_name: Column name
            
        Returns:
            Extracted year or None
        """
        match = re.search(r'\b(20\d{2})\b', col_name)
        if match:
            return int(match.group(1))
        return None
    
    @staticmethod
    def prepare_analytics_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for analytics with calculated metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Enhanced DataFrame with calculated columns
        """
        try:
            # Find sales columns using regex
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            
            if not sales_cols:
                st.warning("⚠️ No sales columns found")
                return df
            
            # Extract years safely
            years = []
            for col in sales_cols:
                year = DataProcessor.extract_year_from_column(col)
                if year:
                    years.append(year)
            
            years = sorted(set(years))
            
            if len(years) < 2:
                st.info("ℹ️ Need at least 2 years for growth calculation")
                return df
            
            # Calculate growth rates
            for i in range(1, len(years)):
                prev_year = years[i-1]
                curr_year = years[i]
                
                prev_col = f"Satış_{prev_year}"
                curr_col = f"Satış_{curr_year}"
                
                if prev_col in df.columns and curr_col in df.columns:
                    growth_col = f'Buyume_{prev_year}_{curr_year}'
                    df[growth_col] = np.where(
                        df[prev_col] != 0,
                        ((df[curr_col] - df[prev_col]) / df[prev_col]) * 100,
                        np.nan
                    )
            
            # CAGR calculation
            if len(years) >= 2:
                first_year = years[0]
                last_year = years[-1]
                first_col = f"Satış_{first_year}"
                last_col = f"Satış_{last_year}"
                
                if first_col in df.columns and last_col in df.columns:
                    num_periods = last_year - first_year
                    df['CAGR'] = np.where(
                        df[first_col] > 0,
                        (np.power(df[last_col] / df[first_col], 1/num_periods) - 1) * 100,
                        np.nan
                    )
            
            # Market share
            if years:
                last_year = years[-1]
                last_sales_col = f"Satış_{last_year}"
                
                if last_sales_col in df.columns:
                    total_sales = df[last_sales_col].sum()
                    if total_sales > 0:
                        df['Pazar_Payi'] = (df[last_sales_col] / total_sales) * 100
            
            # Calculate average prices if not present
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            if not price_cols:
                for year in years:
                    sales_col = f"Satış_{year}"
                    unit_col = f"Birim_{year}"
                    
                    if sales_col in df.columns and unit_col in df.columns:
                        df[f'Ort_Fiyat_{year}'] = np.where(
                            df[unit_col] > 0,
                            df[sales_col] / df[unit_col],
                            np.nan
                        )
            
            # Price-Volume ratio
            if years:
                last_year = years[-1]
                price_col = f"Ort_Fiyat_{last_year}"
                unit_col = f"Birim_{last_year}"
                
                if price_col in df.columns and unit_col in df.columns:
                    df['Fiyat_Hacim_Orani'] = df[price_col] * df[unit_col]
            
            # Performance score
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                try:
                    scaler = StandardScaler()
                    numeric_data = df[numeric_cols].fillna(0)
                    scaled_data = scaler.fit_transform(numeric_data)
                    df['Performans_Skoru'] = scaled_data.mean(axis=1)
                except Exception:
                    pass
            
            # International Product handling
            if 'International_Product' in df.columns:
                df['International_Product'] = df['International_Product'].fillna(0).astype(int)
            
            return df
            
        except Exception as e:
            st.warning(f"Analytics preparation warning: {str(e)}")
            return df

# ================================================
# 3. ADVANCED FILTER SYSTEM
# ================================================

class AdvancedFilterSystem:
    """
    Sophisticated filtering system with multi-criteria support
    and intelligent search capabilities.
    """
    
    @staticmethod
    def create_filter_sidebar(df: pd.DataFrame) -> Tuple[str, Dict, bool, bool]:
        """
        Create advanced filtering sidebar.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (search_term, filter_config, apply_filters, clear_filters)
        """
        with st.sidebar.expander("🎯 ADVANCED FILTERING", expanded=True):
            st.markdown('<div class="filter-title">🔍 Search & Filter</div>', unsafe_allow_html=True)
            
            search_term = st.text_input(
                "🔎 Global Search",
                placeholder="Molecule, Company, Country...",
                help="Search across all columns",
                key="global_search"
            )
            
            filter_config = {}
            available_cols = df.columns.tolist()
            
            # Country filter
            if 'Ulke' in available_cols:
                countries = sorted(df['Ulke'].dropna().unique())
                selected_countries = AdvancedFilterSystem._searchable_multiselect(
                    "🌍 Countries",
                    countries,
                    key="countries_filter",
                    select_all_default=True
                )
                if selected_countries and "All" not in selected_countries:
                    filter_config['Ulke'] = selected_countries
            
            # Company filter
            if 'Sirket' in available_cols:
                companies = sorted(df['Sirket'].dropna().unique())
                selected_companies = AdvancedFilterSystem._searchable_multiselect(
                    "🏢 Companies",
                    companies,
                    key="companies_filter",
                    select_all_default=True
                )
                if selected_companies and "All" not in selected_companies:
                    filter_config['Sirket'] = selected_companies
            
            # Molecule filter
            if 'Molekul' in available_cols:
                molecules = sorted(df['Molekul'].dropna().unique())
                selected_molecules = AdvancedFilterSystem._searchable_multiselect(
                    "🧪 Molecules",
                    molecules,
                    key="molecules_filter",
                    select_all_default=True
                )
                if selected_molecules and "All" not in selected_molecules:
                    filter_config['Molekul'] = selected_molecules
            
            st.markdown("---")
            st.markdown('<div class="filter-title">📊 Numeric Filters</div>', unsafe_allow_html=True)
            
            # Sales filter
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if sales_cols:
                last_sales_col = sales_cols[-1]
                min_sales = float(df[last_sales_col].min())
                max_sales = float(df[last_sales_col].max())
                
                sales_range = st.slider(
                    f"Sales Filter ({last_sales_col})",
                    min_value=min_sales,
                    max_value=max_sales,
                    value=(min_sales, max_sales),
                    key="sales_filter"
                )
                filter_config['sales_range'] = (sales_range, last_sales_col)
            
            # Growth filter
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                last_growth_col = growth_cols[-1]
                min_growth = float(df[last_growth_col].min())
                max_growth = float(df[last_growth_col].max())
                
                growth_range = st.slider(
                    f"Growth Filter ({last_growth_col})",
                    min_value=min_growth,
                    max_value=max_growth,
                    value=(min(min_growth, -50.0), max(max_growth, 150.0)),
                    key="growth_filter"
                )
                filter_config['growth_range'] = (growth_range, last_growth_col)
            
            st.markdown("---")
            st.markdown('<div class="filter-title">⚙️ Additional Filters</div>', unsafe_allow_html=True)
            
            # International Product filter
            if 'International_Product' in df.columns:
                intl_filter = st.selectbox(
                    "International Product",
                    ["All", "International Only", "Local Only"],
                    key="intl_filter"
                )
                if intl_filter != "All":
                    filter_config['international_filter'] = intl_filter
            
            # Positive growth only
            only_positive = st.checkbox("📈 Positive Growth Only", value=False)
            if only_positive and growth_cols:
                filter_config['positive_growth'] = True
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                apply_filters = st.button("✅ Apply", use_container_width=True, key="apply_filters")
            with col2:
                clear_filters = st.button("🗑️ Clear", use_container_width=True, key="clear_filters")
            
            return search_term, filter_config, apply_filters, clear_filters
    
    @staticmethod
    def _searchable_multiselect(
        label: str,
        options: List[str],
        key: str,
        select_all_default: bool = False
    ) -> List[str]:
        """Create searchable multiselect widget."""
        
        if not options:
            return []
        
        all_options = ["All"] + list(options)
        
        search_query = st.text_input(
            f"{label} Search",
            key=f"{key}_search",
            placeholder="Search..."
        )
        
        if search_query:
            filtered_options = ["All"] + [
                opt for opt in options
                if search_query.lower() in str(opt).lower()
            ]
        else:
            filtered_options = all_options
        
        default_selection = ["All"] if select_all_default else filtered_options[:min(5, len(filtered_options))]
        
        selected = st.multiselect(
            label,
            options=filtered_options,
            default=default_selection,
            key=key,
            help="'All' selects everything"
        )
        
        if "All" in selected and len(selected) > 1:
            selected = [opt for opt in selected if opt != "All"]
        elif "All" in selected:
            selected = list(options)
        
        if selected:
            if len(selected) == len(options):
                st.caption(f"✅ ALL selected ({len(options)} items)")
            else:
                st.caption(f"✅ {len(selected)} / {len(options)} selected")
        
        return selected
    
    @staticmethod
    def apply_filters(
        df: pd.DataFrame,
        search_term: str,
        filter_config: Dict
    ) -> pd.DataFrame:
        """
        Apply all configured filters to DataFrame.
        
        Args:
            df: Input DataFrame
            search_term: Global search term
            filter_config: Dictionary of filter configurations
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Global search
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
        
        # Column filters
        for col, values in filter_config.items():
            if col in filtered_df.columns and values and col not in [
                'sales_range', 'growth_range', 'positive_growth', 'international_filter'
            ]:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        # Sales range
        if 'sales_range' in filter_config:
            (min_val, max_val), col_name = filter_config['sales_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) &
                    (filtered_df[col_name] <= max_val)
                ]
        
        # Growth range
        if 'growth_range' in filter_config:
            (min_val, max_val), col_name = filter_config['growth_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) &
                    (filtered_df[col_name] <= max_val)
                ]
        
        # International filter
        if 'international_filter' in filter_config and 'International_Product' in filtered_df.columns:
            if filter_config['international_filter'] == "International Only":
                filtered_df = filtered_df[filtered_df['International_Product'] == 1]
            elif filter_config['international_filter'] == "Local Only":
                filtered_df = filtered_df[filtered_df['International_Product'] == 0]
        
        # Positive growth
        if 'positive_growth' in filter_config and filter_config['positive_growth']:
            growth_cols = [col for col in filtered_df.columns if 'Buyume_' in col]
            if growth_cols:
                filtered_df = filtered_df[filtered_df[growth_cols[-1]] > 0]
        
        return filtered_df

# ================================================
# 4. ANALYTICS ENGINE
# ================================================

class AnalyticsEngine:
    """
    Advanced analytics engine with comprehensive market intelligence,
    forecasting, and anomaly detection capabilities.
    """
    
    @staticmethod
    def calculate_comprehensive_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive market metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        try:
            metrics['Total_Rows'] = len(df)
            metrics['Total_Columns'] = len(df.columns)
            
            # Sales metrics
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if sales_cols:
                last_sales_col = sales_cols[-1]
                year = DataProcessor.extract_year_from_column(last_sales_col)
                
                metrics['Last_Sales_Year'] = year
                metrics['Total_Market_Value'] = df[last_sales_col].sum()
                metrics['Avg_Sales_Per_Product'] = df[last_sales_col].mean()
                metrics['Median_Sales'] = df[last_sales_col].median()
                metrics['Sales_Std'] = df[last_sales_col].std()
                metrics['Sales_Q1'] = df[last_sales_col].quantile(0.25)
                metrics['Sales_Q3'] = df[last_sales_col].quantile(0.75)
                metrics['Sales_IQR'] = metrics['Sales_Q3'] - metrics['Sales_Q1']
            
            # Growth metrics
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                last_growth_col = growth_cols[-1]
                metrics['Avg_Growth_Rate'] = df[last_growth_col].mean()
                metrics['Median_Growth'] = df[last_growth_col].median()
                metrics['Positive_Growth_Products'] = (df[last_growth_col] > 0).sum()
                metrics['Negative_Growth_Products'] = (df[last_growth_col] < 0).sum()
                metrics['High_Growth_Products'] = (df[last_growth_col] > 20).sum()
                metrics['High_Growth_Percentage'] = (metrics['High_Growth_Products'] / metrics['Total_Rows']) * 100
            
            # Company-based metrics
            if 'Sirket' in df.columns and sales_cols:
                last_sales_col = sales_cols[-1]
                company_sales = df.groupby('Sirket')[last_sales_col].sum().sort_values(ascending=False)
                total_sales = company_sales.sum()
                
                if total_sales > 0:
                    market_shares = (company_sales / total_sales * 100)
                    metrics['HHI_Index'] = (market_shares ** 2).sum()
                    
                    for n in [1, 3, 5, 10]:
                        if len(company_sales) >= n:
                            metrics[f'Top_{n}_Share'] = company_sales.nlargest(n).sum() / total_sales * 100
            
            # Molecule metrics
            if 'Molekul' in df.columns:
                metrics['Unique_Molecules'] = df['Molekul'].nunique()
                if sales_cols:
                    molecule_sales = df.groupby('Molekul')[last_sales_col].sum()
                    total_molecule_sales = molecule_sales.sum()
                    if total_molecule_sales > 0:
                        metrics['Top_10_Molecule_Share'] = molecule_sales.nlargest(10).sum() / total_molecule_sales * 100
            
            # Country metrics
            if 'Ulke' in df.columns:
                metrics['Country_Coverage'] = df['Ulke'].nunique()
                if sales_cols:
                    country_sales = df.groupby('Ulke')[last_sales_col].sum()
                    total_country_sales = country_sales.sum()
                    if total_country_sales > 0:
                        metrics['Top_5_Country_Share'] = country_sales.nlargest(5).sum() / total_country_sales * 100
            
            # Price metrics
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            if price_cols:
                last_price_col = price_cols[-1]
                metrics['Avg_Price'] = df[last_price_col].mean()
                metrics['Price_Variance'] = df[last_price_col].var()
                metrics['Price_Q1'] = df[last_price_col].quantile(0.25)
                metrics['Price_Median'] = df[last_price_col].quantile(0.5)
                metrics['Price_Q3'] = df[last_price_col].quantile(0.75)
            
            # International Product metrics
            if 'International_Product' in df.columns and sales_cols:
                last_sales_col = sales_cols[-1]
                
                intl_df = df[df['International_Product'] == 1]
                local_df = df[df['International_Product'] == 0]
                
                metrics['International_Product_Count'] = len(intl_df)
                metrics['Local_Product_Count'] = len(local_df)
                metrics['International_Product_Sales'] = intl_df[last_sales_col].sum()
                metrics['Local_Product_Sales'] = local_df[last_sales_col].sum()
                
                total_sales = metrics.get('Total_Market_Value', 0)
                if total_sales > 0:
                    metrics['International_Product_Share'] = (metrics['International_Product_Sales'] / total_sales) * 100
                    metrics['Local_Product_Share'] = (metrics['Local_Product_Sales'] / total_sales) * 100
                
                if len(intl_df) > 0 and growth_cols:
                    last_growth_col = growth_cols[-1]
                    metrics['International_Avg_Growth'] = intl_df[last_growth_col].mean()
                    metrics['Local_Avg_Growth'] = local_df[last_growth_col].mean()
                
                if len(intl_df) > 0 and price_cols:
                    last_price_col = price_cols[-1]
                    metrics['International_Avg_Price'] = intl_df[last_price_col].mean()
                    metrics['Local_Avg_Price'] = local_df[last_price_col].mean()
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrics calculation warning: {str(e)}")
            return metrics
    
    @staticmethod
    def international_product_analysis(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Detailed International Product analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Analysis DataFrame or None
        """
        try:
            if 'International_Product' not in df.columns:
                return None
            
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if not sales_cols:
                return None
            
            last_sales_col = sales_cols[-1]
            analysis_data = []
            
            # Molecule-based analysis
            if 'Molekul' in df.columns:
                for molecule in df['Molekul'].unique():
                    molecule_df = df[df['Molekul'] == molecule]
                    
                    is_international = (molecule_df['International_Product'] == 1).any()
                    total_sales = molecule_df[last_sales_col].sum()
                    
                    company_count = molecule_df['Sirket'].nunique() if 'Sirket' in molecule_df.columns else 1
                    country_count = molecule_df['Ulke'].nunique() if 'Ulke' in molecule_df.columns else 1
                    
                    growth_cols = [col for col in df.columns if 'Buyume_' in col]
                    avg_growth = molecule_df[growth_cols[-1]].mean() if growth_cols else None
                    
                    price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
                    avg_price = molecule_df[price_cols[-1]].mean() if price_cols else None
                    
                    analysis_data.append({
                        'Molekul': molecule,
                        'International': is_international,
                        'Total_Sales': total_sales,
                        'Company_Count': company_count,
                        'Country_Count': country_count,
                        'Product_Count': len(molecule_df),
                        'Avg_Price': avg_price,
                        'Avg_Growth': avg_growth,
                        'Complexity_Score': (company_count * 0.6 + country_count * 0.4) / 2
                    })
            
            elif 'Sirket' in df.columns:
                # Company-based analysis
                for company in df['Sirket'].unique():
                    company_df = df[df['Sirket'] == company]
                    
                    is_international = (company_df['International_Product'] == 1).any()
                    total_sales = company_df[last_sales_col].sum()
                    
                    analysis_data.append({
                        'Sirket': company,
                        'International': is_international,
                        'Total_Sales': total_sales,
                        'Product_Count': len(company_df),
                        'International_Product_Count': (company_df['International_Product'] == 1).sum()
                    })
            
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                
                # Segmentation
                if 'Complexity_Score' in analysis_df.columns:
                    analysis_df['Segment'] = pd.cut(
                        analysis_df['Complexity_Score'],
                        bins=[0, 0.5, 1.5, 3, float('inf')],
                        labels=['Local', 'Regional', 'Multi-National', 'Global']
                    )
                
                return analysis_df.sort_values('Total_Sales', ascending=False)
            
            return None
            
        except Exception as e:
            st.warning(f"International Product analysis error: {str(e)}")
            return None
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Detect market anomalies using Isolation Forest.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with anomaly scores
        """
        try:
            # Select numeric features
            numeric_cols = []
            
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if sales_cols:
                numeric_cols.extend(sales_cols[-2:])
            
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                numeric_cols.append(growth_cols[-1])
            
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            if price_cols:
                numeric_cols.append(price_cols[-1])
            
            if len(numeric_cols) < 2:
                return None
            
            anomaly_data = df[numeric_cols].fillna(0)
            
            if len(anomaly_data) < 10:
                return None
            
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            anomaly_scores = iso_forest.fit_predict(anomaly_data)
            anomaly_scores_continuous = iso_forest.score_samples(anomaly_data)
            
            result_df = df.copy()
            result_df['Anomaly'] = anomaly_scores
            result_df['Anomaly_Score'] = anomaly_scores_continuous
            
            # Categorize
            result_df['Anomaly_Category'] = pd.cut(
                result_df['Anomaly_Score'],
                bins=[-np.inf, -0.5, -0.2, 0],
                labels=['High Risk', 'Medium Risk', 'Normal']
            )
            
            return result_df
            
        except Exception as e:
            st.warning(f"Anomaly detection error: {str(e)}")
            return None
    
    @staticmethod
    def forecast_market(df: pd.DataFrame, periods: int = 2) -> Optional[pd.DataFrame]:
        """
        Forecast future market values using time series analysis.
        
        Args:
            df: Input DataFrame
            periods: Number of periods to forecast
            
        Returns:
            Forecast DataFrame or None
        """
        try:
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if len(sales_cols) < 3:
                return None
            
            # Extract years
            years = []
            for col in sorted(sales_cols):
                year = DataProcessor.extract_year_from_column(col)
                if year:
                    years.append(year)
            
            years = sorted(set(years))
            
            if len(years) < 3:
                return None
            
            # Aggregate sales by year
            yearly_sales = {}
            for year in years:
                col = f"Satış_{year}"
                if col in df.columns:
                    yearly_sales[year] = df[col].sum()
            
            # Create time series
            ts_data = pd.Series(yearly_sales)
            
            # Exponential Smoothing forecast
            model = ExponentialSmoothing(
                ts_data,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            fitted_model = model.fit()
            
            # Forecast
            last_year = years[-1]
            forecast_years = [last_year + i + 1 for i in range(periods)]
            forecast_values = fitted_model.forecast(steps=periods)
            
            # Calculate confidence intervals (simplified)
            residuals_std = np.std(fitted_model.fittedvalues - ts_data)
            confidence_interval = 1.96 * residuals_std
            
            forecast_df = pd.DataFrame({
                'Year': forecast_years,
                'Forecast': forecast_values.values,
                'Lower_Bound': forecast_values.values - confidence_interval,
                'Upper_Bound': forecast_values.values + confidence_interval
            })
            
            # Calculate growth rates
            if len(forecast_df) > 0:
                last_actual = ts_data.iloc[-1]
                first_forecast = forecast_df['Forecast'].iloc[0]
                forecast_df['Growth_From_Last_Year'] = ((first_forecast - last_actual) / last_actual) * 100
            
            return forecast_df
            
        except Exception as e:
            st.warning(f"Forecasting error: {str(e)}")
            return None
    
    @staticmethod
    def advanced_segmentation(
        df: pd.DataFrame,
        n_clusters: int = 4,
        method: str = 'kmeans'
    ) -> Optional[pd.DataFrame]:
        """
        Advanced market segmentation with multiple algorithms.
        
        Args:
            df: Input DataFrame
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            
        Returns:
            Segmented DataFrame or None
        """
        try:
            # Select features
            features = []
            
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if sales_cols:
                features.extend(sales_cols[-2:])
            
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                features.append(growth_cols[-1])
            
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            if price_cols:
                features.append(price_cols[-1])
            
            if 'Pazar_Payi' in df.columns:
                features.append('Pazar_Payi')
            
            if len(features) < 2:
                return None
            
            segmentation_data = df[features].fillna(0)
            
            if len(segmentation_data) < n_clusters * 10:
                return None
            
            # Scale features
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(segmentation_data)
            
            # Apply clustering
            if method == 'kmeans':
                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=10)
            elif method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                model = KMeans(n_clusters=n_clusters, random_state=42)
            
            clusters = model.fit_predict(features_scaled)
            
            result_df = df.copy()
            result_df['Cluster'] = clusters
            
            # Cluster naming
            cluster_names = {
                0: 'Growing Products',
                1: 'Mature Products',
                2: 'Innovative Products',
                3: 'Risky Products',
                4: 'Niche Products',
                5: 'Volume Products',
                6: 'Premium Products',
                7: 'Economy Products'
            }
            
            result_df['Cluster_Name'] = result_df['Cluster'].map(
                lambda x: cluster_names.get(x, f'Cluster_{x}')
            )
            
            # Calculate cluster metrics
            try:
                sil_score = silhouette_score(features_scaled, clusters)
                result_df.attrs['silhouette_score'] = sil_score
            except:
                pass
            
            return result_df
            
        except Exception as e:
            st.warning(f"Segmentation error: {str(e)}")
            return None
    
    @staticmethod
    def generate_strategic_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Generate strategic insights from data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        try:
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if not sales_cols:
                return insights
            
            last_sales_col = sales_cols[-1]
            year = DataProcessor.extract_year_from_column(last_sales_col)
            
            # Top products
            top_products = df.nlargest(10, last_sales_col)
            top_share = (top_products[last_sales_col].sum() / df[last_sales_col].sum() * 100) if df[last_sales_col].sum() > 0 else 0
            
            insights.append({
                'type': 'success',
                'title': f'🏆 Top 10 Products - {year}',
                'description': f"Top 10 products account for {top_share:.1f}% of total market."
            })
            
            # Fast-growing products
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                last_growth_col = growth_cols[-1]
                top_growth = df.nlargest(10, last_growth_col)
                avg_growth = top_growth[last_growth_col].mean()
                
                insights.append({
                    'type': 'info',
                    'title': f'🚀 Fastest Growing 10 Products',
                    'description': f"Fastest growing products show {avg_growth:.1f}% average growth."
                })
            
            # Market leader
            if 'Sirket' in df.columns:
                top_companies = df.groupby('Sirket')[last_sales_col].sum().nlargest(5)
                top_company = top_companies.index[0]
                top_company_share = (top_companies.iloc[0] / df[last_sales_col].sum()) * 100
                
                insights.append({
                    'type': 'warning',
                    'title': f'🏢 Market Leader - {year}',
                    'description': f"{top_company} leads with {top_company_share:.1f}% market share."
                })
            
            # Largest market
            if 'Ulke' in df.columns:
                top_countries = df.groupby('Ulke')[last_sales_col].sum().nlargest(5)
                top_country = top_countries.index[0]
                top_country_share = (top_countries.iloc[0] / df[last_sales_col].sum()) * 100
                
                insights.append({
                    'type': 'geographic',
                    'title': f'🌍 Largest Market - {year}',
                    'description': f"{top_country} is the largest market with {top_country_share:.1f}% share."
                })
            
            # Price analysis
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            if price_cols:
                avg_price = df[price_cols[-1]].mean()
                price_std = df[price_cols[-1]].std()
                
                insights.append({
                    'type': 'price',
                    'title': f'💰 Price Analysis - {year}',
                    'description': f"Average price: ${avg_price:.2f} (Std: ${price_std:.2f})"
                })
            
            # International Product
            if 'International_Product' in df.columns:
                intl_df = df[df['International_Product'] == 1]
                local_df = df[df['International_Product'] == 0]
                
                intl_count = len(intl_df)
                intl_share = (intl_df[last_sales_col].sum() / df[last_sales_col].sum() * 100) if df[last_sales_col].sum() > 0 else 0
                
                insights.append({
                    'type': 'international',
                    'title': f'🌐 International Product Analysis',
                    'description': f"{intl_count} International Products account for {intl_share:.1f}% of market."
                })
            
            return insights
            
        except Exception as e:
            st.warning(f"Insight generation warning: {str(e)}")
            return []

# ================================================
# 5. PROFESSIONAL VISUALIZER
# ================================================

class ProfessionalVisualizer:
    """
    Advanced visualization engine with enterprise-grade charts
    and interactive dashboards.
    """
    
    @staticmethod
    def create_dashboard_metrics(df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        """Create dashboard metric cards."""
        
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sales = metrics.get('Total_Market_Value', 0)
                sales_year = metrics.get('Last_Sales_Year', '')
                st.markdown(f"""
                <div class="custom-metric-card primary">
                    <div class="custom-metric-label">TOTAL MARKET VALUE</div>
                    <div class="custom-metric-value">${total_sales/1e6:.1f}M</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">{sales_year}</span>
                        <span>Total Market</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_growth = metrics.get('Avg_Growth_Rate', 0)
                growth_class = "success" if avg_growth > 0 else "danger"
                st.markdown(f"""
                <div class="custom-metric-card {growth_class}">
                    <div class="custom-metric-label">AVERAGE GROWTH</div>
                    <div class="custom-metric-value">{avg_growth:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">YoY</span>
                        <span>Annual Growth</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrics.get('HHI_Index', 0)
                hhi_status = "danger" if hhi > 2500 else "warning" if hhi > 1500 else "success"
                hhi_text = "Monopolistic" if hhi > 2500 else "Oligopoly" if hhi > 1500 else "Competitive"
                st.markdown(f"""
                <div class="custom-metric-card {hhi_status}">
                    <div class="custom-metric-label">COMPETITION INTENSITY</div>
                    <div class="custom-metric-value">{hhi:.0f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-warning">HHI Index</span>
                        <span>{hhi_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                intl_share = metrics.get('International_Product_Share', 0)
                intl_color = "success" if intl_share > 20 else "warning" if intl_share > 10 else "info"
                st.markdown(f"""
                <div class="custom-metric-card {intl_color}">
                    <div class="custom-metric-label">INTERNATIONAL PRODUCTS</div>
                    <div class="custom-metric-value">{intl_share:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Global</span>
                        <span>Multi-Market</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Second row
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                unique_molecules = metrics.get('Unique_Molecules', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">MOLECULE DIVERSITY</div>
                    <div class="custom-metric-value">{unique_molecules:,}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">Unique</span>
                        <span>Different Molecules</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                avg_price = metrics.get('Avg_Price', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">AVERAGE PRICE</div>
                    <div class="custom-metric-value">${avg_price:.2f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Per Unit</span>
                        <span>Average</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                high_growth_pct = metrics.get('High_Growth_Percentage', 0)
                st.markdown(f"""
                <div class="custom-metric-card success">
                    <div class="custom-metric-label">HIGH GROWTH</div>
                    <div class="custom-metric-value">{high_growth_pct:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">20%+</span>
                        <span>Fast Growing</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                country_coverage = metrics.get('Country_Coverage', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">GEOGRAPHIC SPREAD</div>
                    <div class="custom-metric-value">{country_coverage}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Countries</span>
                        <span>Global Coverage</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Metric card creation error: {str(e)}")
    
    @staticmethod
    def sales_trend_chart(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create sales trend visualization."""
        
        try:
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if len(sales_cols) < 2:
                return None
            
            yearly_data = []
            for col in sorted(sales_cols):
                year = DataProcessor.extract_year_from_column(col)
                if year:
                    yearly_data.append({
                        'Year': year,
                        'Total_Sales': df[col].sum(),
                        'Avg_Sales': df[col].mean(),
                        'Product_Count': (df[col] > 0).sum()
                    })
            
            yearly_df = pd.DataFrame(yearly_data)
            
            fig = go.Figure()
            
            # Total sales bar
            fig.add_trace(go.Bar(
                x=yearly_df['Year'],
                y=yearly_df['Total_Sales'],
                name='Total Sales',
                marker_color='#2d7dd2',
                text=[f'${x/1e6:.0f}M' for x in yearly_df['Total_Sales']],
                textposition='auto'
            ))
            
            # Average sales line (secondary axis)
            fig.add_trace(go.Scatter(
                x=yearly_df['Year'],
                y=yearly_df['Avg_Sales'],
                name='Average Sales',
                mode='lines+markers',
                line=dict(color='#2acaea', width=3),
                marker=dict(size=10),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='Sales Trend Analysis',
                xaxis_title='Year',
                yaxis_title='Total Sales (USD)',
                yaxis2=dict(
                    title='Average Sales (USD)',
                    overlaying='y',
                    side='right'
                ),
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Sales trend chart error: {str(e)}")
            return None
    
    @staticmethod
    def market_share_analysis(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create market share visualization."""
        
        try:
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if not sales_cols or 'Sirket' not in df.columns:
                return None
            
            last_sales_col = sales_cols[-1]
            
            company_sales = df.groupby('Sirket')[last_sales_col].sum().sort_values(ascending=False)
            top_companies = company_sales.nlargest(15)
            other_sales = company_sales.iloc[15:].sum() if len(company_sales) > 15 else 0
            
            pie_data = top_companies.copy()
            if other_sales > 0:
                pie_data['Others'] = other_sales
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Market Share Distribution', 'Top 10 Company Sales'),
                specs=[[{'type': 'domain'}, {'type': 'bar'}]],
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
                    textposition='outside'
                ),
                row=1, col=1
            )
            
            # Bar chart
            fig.add_trace(
                go.Bar(
                    x=top_companies.values[:10],
                    y=top_companies.index[:10],
                    orientation='h',
                    marker_color='#2d7dd2',
                    text=[f'${x/1e6:.1f}M' for x in top_companies.values[:10]],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                title_text="Market Concentration Analysis",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Market share chart error: {str(e)}")
            return None
    
    @staticmethod
    def world_map_visualization(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create world map visualization."""
        
        try:
            if 'Ulke' not in df.columns:
                return None
            
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if not sales_cols:
                return None
            
            last_sales_col = sales_cols[-1]
            
            country_sales = df.groupby('Ulke')[last_sales_col].sum().reset_index()
            country_sales.columns = ['Country', 'Total_Sales']
            
            # Country name mapping
            country_mapping = {
                'USA': 'United States',
                'US': 'United States',
                'U.S.A': 'United States',
                'UK': 'United Kingdom',
                'U.K': 'United Kingdom',
                'UAE': 'United Arab Emirates',
                'U.A.E': 'United Arab Emirates',
                'S. Korea': 'South Korea',
                'South Korea': 'Korea, Republic of',
                'Russia': 'Russian Federation',
                'Iran': 'Iran, Islamic Republic of',
                'Vietnam': 'Viet Nam',
                'Syria': 'Syrian Arab Republic',
                'Laos': 'Lao People\'s Democratic Republic',
                'Bolivia': 'Bolivia, Plurinational State of',
                'Venezuela': 'Venezuela, Bolivarian Republic of',
                'Tanzania': 'Tanzania, United Republic of',
                'Moldova': 'Moldova, Republic of',
                'Macedonia': 'North Macedonia'
            }
            
            country_sales['Country'] = country_sales['Country'].replace(country_mapping)
            
            fig = px.choropleth(
                country_sales,
                locations='Country',
                locationmode='country names',
                color='Total_Sales',
                hover_name='Country',
                hover_data={'Total_Sales': ':.2f'},
                color_continuous_scale='Viridis',
                title='Global Pharmaceutical Market Distribution',
                projection='natural earth'
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                geo=dict(
                    bgcolor='rgba(0,0,0,0)',
                    lakecolor='#1e3a5f',
                    landcolor='#2d4a7a',
                    subunitcolor='#64748b'
                ),
                coloraxis_colorbar=dict(
                    title="Total Sales (USD)",
                    tickprefix="$"
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"World map error: {str(e)}")
            return None
    
    @staticmethod
    def sunburst_hierarchy_chart(df: pd.DataFrame) -> Optional[go.Figure]:
        """Create sunburst hierarchy visualization."""
        
        try:
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            if not sales_cols:
                return None
            
            last_sales_col = sales_cols[-1]
            
            # Prepare hierarchy data
            hierarchy_data = []
            
            if 'Sirket' in df.columns and 'Molekul' in df.columns:
                for _, row in df.iterrows():
                    hierarchy_data.append({
                        'labels': row['Molekul'],
                        'parents': row['Sirket'],
                        'values': row[last_sales_col]
                    })
                
                # Add company-level
                company_totals = df.groupby('Sirket')[last_sales_col].sum()
                for company, total in company_totals.items():
                    hierarchy_data.append({
                        'labels': company,
                        'parents': '',
                        'values': total
                    })
                
                hierarchy_df = pd.DataFrame(hierarchy_data)
                
                fig = go.Figure(go.Sunburst(
                    labels=hierarchy_df['labels'],
                    parents=hierarchy_df['parents'],
                    values=hierarchy_df['values'],
                    branchvalues="total",
                    marker=dict(
                        colorscale='Viridis',
                        cmid=hierarchy_df['values'].median()
                    ),
                    hovertemplate='<b>%{label}</b><br>Sales: $%{value:.2f}<br><extra></extra>'
                ))
                
                fig.update_layout(
                    title='Market Hierarchy - Company > Molecule',
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.warning(f"Sunburst chart error: {str(e)}")
            return None
    
    @staticmethod
    def radar_comparison_chart(df: pd.DataFrame, entities: List[str]) -> Optional[go.Figure]:
        """Create radar chart for entity comparison."""
        
        try:
            if 'Sirket' not in df.columns or len(entities) == 0:
                return None
            
            # Metrics for comparison
            metrics = []
            sales_cols = [col for col in df.columns if 'Satış_' in col]
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            
            if sales_cols:
                metrics.append(('Market Share', sales_cols[-1]))
            if growth_cols:
                metrics.append(('Growth Rate', growth_cols[-1]))
            if 'Pazar_Payi' in df.columns:
                metrics.append(('Market Position', 'Pazar_Payi'))
            
            if len(metrics) < 3:
                return None
            
            fig = go.Figure()
            
            for entity in entities[:5]:  # Limit to 5 for readability
                entity_df = df[df['Sirket'] == entity]
                
                if len(entity_df) == 0:
                    continue
                
                values = []
                for _, metric_col in metrics:
                    if metric_col in entity_df.columns:
                        values.append(entity_df[metric_col].mean())
                    else:
                        values.append(0)
                
                # Normalize values to 0-100 scale
                max_vals = [df[col].max() for _, col in metrics]
                normalized_values = [(v / m * 100) if m > 0 else 0 for v, m in zip(values, max_vals)]
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=[name for name, _ in metrics],
                    fill='toself',
                    name=entity
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title='Company Performance Comparison',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Radar chart error: {str(e)}")
            return None
    
    @staticmethod
    def forecast_visualization(
        historical_df: pd.DataFrame,
        forecast_df: Optional[pd.DataFrame]
    ) -> Optional[go.Figure]:
        """Visualize historical data and forecasts."""
        
        try:
            if forecast_df is None:
                return None
            
            sales_cols = [col for col in historical_df.columns if 'Satış_' in col]
            if not sales_cols:
                return None
            
            # Historical data
            years = []
            values = []
            for col in sorted(sales_cols):
                year = DataProcessor.extract_year_from_column(col)
                if year:
                    years.append(year)
                    values.append(historical_df[col].sum())
            
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines+markers',
                name='Historical',
                line=dict(color='#2d7dd2', width=3),
                marker=dict(size=10)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['Year'],
                y=forecast_df['Forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#2acaea', width=3, dash='dash'),
                marker=dict(size=10)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=list(forecast_df['Year']) + list(forecast_df['Year'][::-1]),
                y=list(forecast_df['Upper_Bound']) + list(forecast_df['Lower_Bound'][::-1]),
                fill='toself',
                fillcolor='rgba(42, 202, 234, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='95% Confidence'
            ))
            
            fig.update_layout(
                title='Market Forecast with Confidence Intervals',
                xaxis_title='Year',
                yaxis_title='Total Sales (USD)',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Forecast visualization error: {str(e)}")
            return None
    
    @staticmethod
    def anomaly_scatter_plot(anomaly_df: pd.DataFrame) -> Optional[go.Figure]:
        """Create anomaly detection scatter plot."""
        
        try:
            if anomaly_df is None or 'Anomaly_Score' not in anomaly_df.columns:
                return None
            
            sales_cols = [col for col in anomaly_df.columns if 'Satış_' in col]
            growth_cols = [col for col in anomaly_df.columns if 'Buyume_' in col]
            
            if not sales_cols or not growth_cols:
                return None
            
            last_sales_col = sales_cols[-1]
            last_growth_col = growth_cols[-1]
            
            # Sample if too large
            plot_df = anomaly_df if len(anomaly_df) <= 5000 else anomaly_df.sample(5000, random_state=42)
            
            fig = px.scatter(
                plot_df,
                x=last_sales_col,
                y=last_growth_col,
                color='Anomaly_Category',
                size=abs(plot_df['Anomaly_Score']),
                hover_name='Molekul' if 'Molekul' in plot_df.columns else None,
                title='Anomaly Detection - Sales vs Growth',
                labels={
                    last_sales_col: 'Sales (USD)',
                    last_growth_col: 'Growth Rate (%)'
                },
                color_discrete_map={
                    'High Risk': '#eb5757',
                    'Medium Risk': '#f2c94c',
                    'Normal': '#2dd2a3'
                }
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Anomaly plot error: {str(e)}")
            return None

# ================================================
# 6. REPORT GENERATOR
# ================================================

class ReportGenerator:
    """
    Advanced report generation with Excel and PDF export capabilities.
    """
    
    @staticmethod
    def generate_excel_report(
        df: pd.DataFrame,
        metrics: Dict[str, Any],
        insights: List[Dict[str, str]],
        filename: str = "pharma_report.xlsx"
    ) -> BytesIO:
        """
        Generate comprehensive Excel report.
        
        Args:
            df: Input DataFrame
            metrics: Calculated metrics
            insights: Strategic insights
            filename: Output filename
            
        Returns:
            BytesIO object containing Excel file
        """
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'font_color': 'white',
                    'bg_color': '#2d7dd2',
                    'border': 1
                })
                
                number_format = workbook.add_format({
                    'num_format': '#,##0.00',
                    'border': 1
                })
                
                percent_format = workbook.add_format({
                    'num_format': '0.00%',
                    'border': 1
                })
                
                # Sheet 1: Executive Summary
                summary_data = pd.DataFrame([
                    ['Total Market Value', f"${metrics.get('Total_Market_Value', 0)/1e6:.2f}M"],
                    ['Average Growth Rate', f"{metrics.get('Avg_Growth_Rate', 0):.2f}%"],
                    ['HHI Index', f"{metrics.get('HHI_Index', 0):.2f}"],
                    ['Unique Molecules', metrics.get('Unique_Molecules', 0)],
                    ['Country Coverage', metrics.get('Country_Coverage', 0)],
                    ['International Product Share', f"{metrics.get('International_Product_Share', 0):.2f}%"]
                ], columns=['Metric', 'Value'])
                
                summary_data.to_excel(writer, sheet_name='Executive Summary', index=False)
                worksheet = writer.sheets['Executive Summary']
                worksheet.set_column('A:A', 30)
                worksheet.set_column('B:B', 20)
                
                # Sheet 2: Detailed Data
                df.to_excel(writer, sheet_name='Detailed Data', index=False)
                worksheet = writer.sheets['Detailed Data']
                
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Sheet 3: Strategic Insights
                insights_data = pd.DataFrame(insights)
                if not insights_data.empty:
                    insights_data.to_excel(writer, sheet_name='Strategic Insights', index=False)
                
                # Sheet 4: Top Products
                sales_cols = [col for col in df.columns if 'Satış_' in col]
                if sales_cols:
                    top_products = df.nlargest(50, sales_cols[-1])[
                        ['Molekul', 'Sirket', sales_cols[-1]] if 'Molekul' in df.columns and 'Sirket' in df.columns else [sales_cols[-1]]
                    ]
                    top_products.to_excel(writer, sheet_name='Top 50 Products', index=False)
                
                # Sheet 5: Company Analysis
                if 'Sirket' in df.columns and sales_cols:
                    company_analysis = df.groupby('Sirket').agg({
                        sales_cols[-1]: ['sum', 'mean', 'count']
                    }).round(2)
                    company_analysis.columns = ['Total Sales', 'Avg Sales', 'Product Count']
                    company_analysis = company_analysis.sort_values('Total Sales', ascending=False)
                    company_analysis.to_excel(writer, sheet_name='Company Analysis')
            
            output.seek(0)
            return output
            
        except Exception as e:
            st.error(f"Excel report generation error: {str(e)}")
            return BytesIO()
    
    @staticmethod
    def generate_html_report(
        df: pd.DataFrame,
        metrics: Dict[str, Any],
        insights: List[Dict[str, str]]
    ) -> str:
        """Generate HTML report."""
        
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>PharmaIntelligence Pro Report</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background: linear-gradient(135deg, #0c1a32, #14274e);
                        color: #f8fafc;
                    }}
                    .header {{
                        text-align: center;
                        padding: 30px;
                        background: rgba(30, 58, 95, 0.8);
                        border-radius: 10px;
                        margin-bottom: 30px;
                    }}
                    .metric-grid {{
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .metric-card {{
                        background: rgba(30, 58, 95, 0.6);
                        padding: 20px;
                        border-radius: 10px;
                        border: 1px solid #2d7dd2;
                    }}
                    .metric-value {{
                        font-size: 2rem;
                        font-weight: bold;
                        color: #2acaea;
                    }}
                    .insights {{
                        margin-top: 30px;
                    }}
                    .insight {{
                        background: rgba(30, 58, 95, 0.6);
                        padding: 15px;
                        margin-bottom: 15px;
                        border-left: 4px solid #2dd2a3;
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>PharmaIntelligence Pro</h1>
                    <h2>Market Analysis Report</h2>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>Total Market Value</h3>
                        <div class="metric-value">${metrics.get('Total_Market_Value', 0)/1e6:.1f}M</div>
                    </div>
                    <div class="metric-card">
                        <h3>Average Growth</h3>
                        <div class="metric-value">{metrics.get('Avg_Growth_Rate', 0):.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>HHI Index</h3>
                        <div class="metric-value">{metrics.get('HHI_Index', 0):.0f}</div>
                    </div>
                </div>
                <div class="insights">
                    <h2>Strategic Insights</h2>
            """
        
            for insight in insights[:10]:
                html_content += f"""
                    <div class="insight">
                        <h3>{insight['title']}</h3>
                        <p>{insight['description']}</p>
                    </div>
                """
        
            html_content += """
                </div>
            </body>
            </html>
            """
        
            return html_content
        
        except Exception as e:
            st.error(f"HTML report generation error: {str(e)}")
            return ""

# ================================================
# 7. MAIN APPLICATION
# ================================================

def main():
    """Main application function."""
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="pharma-title">💊 PHARMAINTELLIGENCE PRO</h1>
        <p class="pharma-subtitle">
        Enterprise pharmaceutical market analytics with AI-powered forecasting, 
        anomaly detection, and comprehensive strategic insights.
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
    if 'international_analysis' not in st.session_state:
        st.session_state.international_analysis = None
    if 'anomaly_data' not in st.session_state:
        st.session_state.anomaly_data = None
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None

    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ CONTROL PANEL</h2>', unsafe_allow_html=True)
        
        with st.expander("📁 DATA UPLOAD", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload Excel/CSV File",
                type=['xlsx', 'xls', 'csv'],
                help="Supports 1M+ rows"
            )
            
            if uploaded_file:
                st.info("⚠️ Full dataset will be loaded")
                st.info(f"File: {uploaded_file.name}")
                
                if st.button("🚀 Load & Analyze Data", type="primary", use_container_width=True):
                    with st.spinner("Processing entire dataset..."):
                        processor = DataProcessor()
                        
                        data = processor.load_large_dataset(uploaded_file, sample_size=None)
                        
                        if data is not None and len(data) > 0:
                            data = processor.prepare_analytics_data(data)
                            
                            st.session_state.data = data
                            st.session_state.filtered_data = data.copy()
                            
                            analytics = AnalyticsEngine()
                            st.session_state.metrics = analytics.calculate_comprehensive_metrics(data)
                            st.session_state.insights = analytics.generate_strategic_insights(data)
                            st.session_state.international_analysis = analytics.international_product_analysis(data)
                            
                            st.success(f"✅ {len(data):,} rows loaded successfully!")
                            st.rerun()
        
        # Filters
        if st.session_state.data is not None:
            data = st.session_state.data
            
            filter_system = AdvancedFilterSystem()
            search_term, filter_config, apply_filters, clear_filters = filter_system.create_filter_sidebar(data)
            
            if apply_filters:
                with st.spinner("Applying filters..."):
                    filtered_data = filter_system.apply_filters(data, search_term, filter_config)
                    st.session_state.filtered_data = filtered_data
                    st.session_state.active_filters = filter_config
                    
                    analytics = AnalyticsEngine()
                    st.session_state.metrics = analytics.calculate_comprehensive_metrics(filtered_data)
                    st.session_state.insights = analytics.generate_strategic_insights(filtered_data)
                    st.session_state.international_analysis = analytics.international_product_analysis(filtered_data)
                    
                    st.success(f"✅ Filters applied: {len(filtered_data):,} rows")
                    st.rerun()
            
            if clear_filters:
                st.session_state.filtered_data = st.session_state.data.copy()
                st.session_state.active_filters = {}
                st.session_state.metrics = AnalyticsEngine().calculate_comprehensive_metrics(st.session_state.data)
                st.session_state.insights = AnalyticsEngine().generate_strategic_insights(st.session_state.data)
                st.session_state.international_analysis = AnalyticsEngine().international_product_analysis(st.session_state.data)
                st.success("✅ Filters cleared")
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
        <strong>PharmaIntelligence Pro</strong><br>
        v6.0 | AI-Powered Analytics<br>
        © 2024 All rights reserved
        </div>
        """, unsafe_allow_html=True)

    # Main content
    if st.session_state.data is None:
        show_welcome_screen()
        return

    data = st.session_state.filtered_data
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    intl_analysis = st.session_state.international_analysis

    # Filter status
    if st.session_state.active_filters:
        filter_info = f"🎯 **Active Filters:** "
        filter_items = []
        
        for key, value in st.session_state.active_filters.items():
            if key in ['Ulke', 'Sirket', 'Molekul']:
                if isinstance(value, list):
                    if len(value) > 3:
                        filter_items.append(f"{key}: {len(value)} options")
                    else:
                        filter_items.append(f"{key}: {', '.join(value[:3])}")
            elif key == 'sales_range':
                (min_val, max_val), col_name = value
                filter_items.append(f"Sales: ${min_val:,.0f}-${max_val:,.0f}")
            elif key == 'growth_range':
                (min_val, max_val), col_name = value
                filter_items.append(f"Growth: {min_val:.1f}%-{max_val:.1f}%")
            elif key == 'positive_growth':
                filter_items.append("Positive Growth")
            elif key == 'international_filter':
                filter_items.append(value)
        
        filter_info += " | ".join(filter_items)
        filter_info += f" | **Showing:** {len(data):,} / {len(st.session_state.data):,} rows"
        
        st.markdown(f'<div class="filter-status">{filter_info}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("❌ Clear All Filters", use_container_width=True):
                st.session_state.filtered_data = st.session_state.data.copy()
                st.session_state.active_filters = {}
                st.session_state.metrics = AnalyticsEngine().calculate_comprehensive_metrics(st.session_state.data)
                st.session_state.insights = AnalyticsEngine().generate_strategic_insights(st.session_state.data)
                st.session_state.international_analysis = AnalyticsEngine().international_product_analysis(st.session_state.data)
                st.success("✅ All filters cleared")
                st.rerun()
    else:
        st.info(f"🎯 No active filters | Showing: {len(data):,} rows")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 OVERVIEW",
        "📈 MARKET ANALYSIS",
        "💰 PRICE ANALYSIS",
        "🏆 COMPETITION",
        "🌍 INTERNATIONAL",
        "🔮 FORECASTING",
        "⚠️ ANOMALY DETECTION",
        "📑 REPORTING"
    ])

    with tab1:
        show_overview_tab(data, metrics, insights)

    with tab2:
        show_market_analysis_tab(data)

    with tab3:
        show_price_analysis_tab(data)

    with tab4:
        show_competition_tab(data, metrics)

    with tab5:
        show_international_tab(data, intl_analysis, metrics)

    with tab6:
        show_forecasting_tab(data)

    with tab7:
        show_anomaly_tab(data)

    with tab8:
        show_reporting_tab(data, metrics, insights, intl_analysis)

# ================================================
# 8. TAB FUNCTIONS
# ================================================

def show_welcome_screen():
    """Display welcome screen."""
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">💊</div>
        <h2 style="color: #f1f5f9; margin-bottom: 1rem;">Welcome to PharmaIntelligence Pro</h2>
        <p style="color: #cbd5e1; margin-bottom: 2rem; line-height: 1.6;">
        Upload your pharmaceutical data to unlock powerful analytics including
        AI-powered forecasting, anomaly detection, and strategic insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_overview_tab(df: pd.DataFrame, metrics: Dict, insights: List[Dict]):
    """Display overview tab."""
    st.markdown('<h2 class="section-title">Overview & Performance Indicators</h2>', unsafe_allow_html=True)

    visualizer = ProfessionalVisualizer()
    visualizer.create_dashboard_metrics(df, metrics)

    st.markdown('<h3 class="subsection-title">🔍 Strategic Insights</h3>', unsafe_allow_html=True)

    if insights:
        insight_cols = st.columns(2)
        
        for idx, insight in enumerate(insights[:6]):
            with insight_cols[idx % 2]:
                icon_map = {
                    'warning': '⚠️',
                    'success': '✅',
                    'info': 'ℹ️',
                    'geographic': '🌍',
                    'price': '💰',
                    'international': '🌐'
                }
                icon = icon_map.get(insight['type'], '💡')
                
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <div class="insight-icon">{icon}</div>
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-content">{insight['description']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<h3 class="subsection-title">📋 Data Preview</h3>', unsafe_allow_html=True)

    preview_col1, preview_col2 = st.columns([1, 3])

    with preview_col1:
        row_count = st.slider("Rows to Display", 10, 5000, 100, 10, key="row_preview")
        
        available_cols = df.columns.tolist()
        default_cols = []
        
        priority_cols = ['Molekul', 'Sirket', 'Ulke', 'Satış_2024', 'Buyume_2023_2024']
        for col in priority_cols:
            if col in available_cols:
                default_cols.append(col)
                if len(default_cols) >= 5:
                    break
        
        if len(default_cols) < 5:
            default_cols.extend([col for col in available_cols[:5] if col not in default_cols])
        
        display_cols = st.multiselect(
            "Columns to Display",
            options=available_cols,
            default=default_cols[:min(5, len(default_cols))],
            key="col_preview"
        )

    with preview_col2:
        if display_cols:
            st.dataframe(
                df[display_cols].head(row_count),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(
                df.head(row_count),
                use_container_width=True,
                height=400
            )

def show_market_analysis_tab(df: pd.DataFrame):
    """Display market analysis tab."""
    st.markdown('<h2 class="section-title">Market Analysis & Trends</h2>', unsafe_allow_html=True)

    visualizer = ProfessionalVisualizer()

    st.markdown('<h3 class="subsection-title">📈 Sales Trends</h3>', unsafe_allow_html=True)
    trend_chart = visualizer.sales_trend_chart(df)
    if trend_chart:
        st.plotly_chart(trend_chart, use_container_width=True, config={'displayModeBar': True})

    st.markdown('<h3 class="subsection-title">🏆 Market Share Analysis</h3>', unsafe_allow_html=True)
    share_chart = visualizer.market_share_analysis(df)
    if share_chart:
        st.plotly_chart(share_chart, use_container_width=True, config={'displayModeBar': True})

    st.markdown('<h3 class="subsection-title">🌐 Geographic Distribution</h3>', unsafe_allow_html=True)
    world_map = visualizer.world_map_visualization(df)
    if world_map:
        st.plotly_chart(world_map, use_container_width=True, config={'displayModeBar': True})

def show_price_analysis_tab(df: pd.DataFrame):
    """Display price analysis tab."""
    st.markdown('<h2 class="section-title">Price Analysis & Optimization</h2>', unsafe_allow_html=True)

    price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
    sales_cols = [col for col in df.columns if 'Satış_' in col]

    if not price_cols:
        st.info("Price analysis requires average price columns in the dataset.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Price Distribution")
        last_price_col = price_cols[-1]
        
        fig = px.histogram(
            df,
            x=last_price_col,
            nbins=50,
            title='Price Distribution',
            labels={last_price_col: 'Price (USD)'}
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Price Segmentation")
        
        price_data = df[last_price_col].dropna()
        if len(price_data) > 0:
            segments = pd.cut(
                price_data,
                bins=[0, 10, 50, 100, 500, float('inf')],
                labels=['Economy (<$10)', 'Standard ($10-$50)', 'Premium ($50-$100)',
                       'Super Premium ($100-$500)', 'Luxury (>$500)']
            )
            
            segment_counts = segments.value_counts()
            
            fig = px.bar(
                x=segment_counts.index,
                y=segment_counts.values,
                title='Products by Price Segment',
                labels={'x': 'Segment', 'y': 'Product Count'}
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            st.plotly_chart(fig, use_container_width=True)

def show_competition_tab(df: pd.DataFrame, metrics: Dict):
    """Display competition analysis tab."""
    st.markdown('<h2 class="section-title">Competition Analysis & Market Structure</h2>', unsafe_allow_html=True)

    st.markdown('<h3 class="subsection-title">📊 Competition Metrics</h3>', unsafe_allow_html=True)

    comp_cols = st.columns(4)

    with comp_cols[0]:
        hhi = metrics.get('HHI_Index', 0)
        hhi_status = "Monopolistic" if hhi > 2500 else "Oligopoly" if hhi > 1800 else "Competitive"
        st.metric("HHI Index", f"{hhi:.0f}", hhi_status)

    with comp_cols[1]:
        top3 = metrics.get('Top_3_Share', 0)
        concentration = "High" if top3 > 50 else "Medium" if top3 > 30 else "Low"
        st.metric("Top 3 Share", f"{top3:.1f}%", concentration)

    with comp_cols[2]:
        top5 = metrics.get('Top_5_Share', 0)
        st.metric("Top 5 Share", f"{top5:.1f}%")

    with comp_cols[3]:
        top10_mol = metrics.get('Top_10_Molecule_Share', 0)
        st.metric("Top 10 Molecule Share", f"{top10_mol:.1f}%")

    visualizer = ProfessionalVisualizer()

    st.markdown('<h3 class="subsection-title">🎯 Market Hierarchy</h3>', unsafe_allow_html=True)
    sunburst = visualizer.sunburst_hierarchy_chart(df)
    if sunburst:
        st.plotly_chart(sunburst, use_container_width=True, config={'displayModeBar': True})

    if 'Sirket' in df.columns:
        st.markdown('<h3 class="subsection-title">📊 Company Comparison</h3>', unsafe_allow_html=True)
        
        companies = df['Sirket'].value_counts().nlargest(10).index.tolist()
        selected_companies = st.multiselect(
            "Select companies to compare (max 5)",
            companies,
            default=companies[:min(3, len(companies))]
        )
        
        if len(selected_companies) > 0:
            radar = visualizer.radar_comparison_chart(df, selected_companies)
            if radar:
                st.plotly_chart(radar, use_container_width=True, config={'displayModeBar': True})

def show_international_tab(df: pd.DataFrame, analysis_df: Optional[pd.DataFrame], metrics: Dict):
    """Display international product tab."""
    st.markdown('<h2 class="section-title">🌍 International Product Analysis</h2>', unsafe_allow_html=True)

    if analysis_df is None:
        st.warning("International Product analysis data not available.")
        return

    st.markdown('<h3 class="subsection-title">📊 International Overview</h3>', unsafe_allow_html=True)

    intl_cols = st.columns(4)

    with intl_cols[0]:
        intl_count = metrics.get('International_Product_Count', 0)
        total_count = metrics.get('Total_Rows', 0)
        intl_pct = (intl_count / total_count * 100) if total_count > 0 else 0
        st.metric("International Products", f"{intl_count:,}", f"{intl_pct:.1f}%")

    with intl_cols[1]:
        intl_share = metrics.get('International_Product_Share', 0)
        st.metric("Market Share", f"{intl_share:.1f}%")

    with intl_cols[2]:
        if 'Country_Count' in analysis_df.columns:
            avg_countries = analysis_df['Country_Count'].mean()
            st.metric("Avg Countries", f"{avg_countries:.1f}")

    with intl_cols[3]:
        if 'Company_Count' in analysis_df.columns:
            avg_companies = analysis_df['Company_Count'].mean()
            st.metric("Avg Companies", f"{avg_companies:.1f}")

    st.markdown('<h3 class="subsection-title">📋 International Product Details</h3>', unsafe_allow_html=True)

    if len(analysis_df) > 0:
        display_cols = []
        
        for col in ['Molekul', 'Sirket', 'International', 'Total_Sales', 'Company_Count',
                    'Country_Count', 'Avg_Price', 'Avg_Growth', 'Segment']:
            if col in analysis_df.columns:
                display_cols.append(col)
        
        display_df = analysis_df[display_cols].copy()
        
        if 'Total_Sales' in display_df.columns:
            display_df['Total_Sales'] = display_df['Total_Sales'].apply(
                lambda x: f"${x/1e6:.2f}M" if pd.notnull(x) and x > 0 else "N/A"
            )
        
        if 'Avg_Growth' in display_df.columns:
            display_df['Avg_Growth'] = display_df['Avg_Growth'].apply(
                lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A"
            )
        
        if 'Avg_Price' in display_df.columns:
            display_df['Avg_Price'] = display_df['Avg_Price'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A"
            )
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )

def show_forecasting_tab(df: pd.DataFrame):
    """Display forecasting tab."""
    st.markdown('<h2 class="section-title">🔮 Market Forecasting</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card info">
        <div class="insight-title">📊 Forecasting Methodology</div>
        <div class="insight-content">
        Using Exponential Smoothing to predict future market values based on historical trends.
        Confidence intervals show the range of probable outcomes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        forecast_periods = st.slider("Forecast Periods (Years)", 1, 5, 2)
        
        if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating forecast..."):
                analytics = AnalyticsEngine()
                forecast_df = analytics.forecast_market(df, periods=forecast_periods)
                
                if forecast_df is not None:
                    st.session_state.forecast_data = forecast_df
                    st.success("✅ Forecast generated!")
                else:
                    st.error("Unable to generate forecast. Need at least 3 years of historical data.")

    with col2:
        if 'forecast_data' in st.session_state and st.session_state.forecast_data is not None:
            visualizer = ProfessionalVisualizer()
            forecast_chart = visualizer.forecast_visualization(df, st.session_state.forecast_data)
            
            if forecast_chart:
                st.plotly_chart(forecast_chart, use_container_width=True, config={'displayModeBar': True})

    if 'forecast_data' in st.session_state and st.session_state.forecast_data is not None:
        st.markdown('<h3 class="subsection-title">📊 Forecast Details</h3>', unsafe_allow_html=True)
        
        forecast_display = st.session_state.forecast_data.copy()
        forecast_display['Forecast'] = forecast_display['Forecast'].apply(lambda x: f"${x/1e6:.2f}M")
        forecast_display['Lower_Bound'] = forecast_display['Lower_Bound'].apply(lambda x: f"${x/1e6:.2f}M")
        forecast_display['Upper_Bound'] = forecast_display['Upper_Bound'].apply(lambda x: f"${x/1e6:.2f}M")
        
        if 'Growth_From_Last_Year' in forecast_display.columns:
            forecast_display['Growth_From_Last_Year'] = forecast_display['Growth_From_Last_Year'].apply(
                lambda x: f"{x:.1f}%"
            )
        
        st.dataframe(forecast_display, use_container_width=True)

def show_anomaly_tab(df: pd.DataFrame):
    """Display anomaly detection tab."""
    st.markdown('<h2 class="section-title">⚠️ Anomaly Detection & Monitoring</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card warning">
        <div class="insight-title">🔍 Anomaly Detection</div>
        <div class="insight-content">
        Using Isolation Forest algorithm to identify outliers and unusual patterns in the market.
        Products with high anomaly scores may require special attention.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
            with st.spinner("Analyzing market anomalies..."):
                analytics = AnalyticsEngine()
                anomaly_df = analytics.detect_anomalies(df)
                
                if anomaly_df is not None:
                    st.session_state.anomaly_data = anomaly_df
                    st.success("✅ Anomaly detection complete!")
                else:
                    st.error("Unable to perform anomaly detection.")

    with col2:
        if 'anomaly_data' in st.session_state and st.session_state.anomaly_data is not None:
            visualizer = ProfessionalVisualizer()
            anomaly_chart = visualizer.anomaly_scatter_plot(st.session_state.anomaly_data)
            
            if anomaly_chart:
                st.plotly_chart(anomaly_chart, use_container_width=True, config={'displayModeBar': True})

    if 'anomaly_data' in st.session_state and st.session_state.anomaly_data is not None:
        anomaly_df = st.session_state.anomaly_data
        
        st.markdown('<h3 class="subsection-title">⚠️ High-Risk Products</h3>', unsafe_allow_html=True)
        
        if 'Anomaly_Category' in anomaly_df.columns:
            high_risk = anomaly_df[anomaly_df['Anomaly_Category'] == 'High Risk']
            
            if len(high_risk) > 0:
                display_cols = ['Molekul', 'Sirket', 'Anomaly_Score'] if 'Molekul' in high_risk.columns and 'Sirket' in high_risk.columns else ['Anomaly_Score']
                
                sales_cols = [col for col in high_risk.columns if 'Satış_' in col]
                if sales_cols:
                    display_cols.append(sales_cols[-1])
                
                st.dataframe(
                    high_risk[display_cols].sort_values('Anomaly_Score').head(20),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No high-risk anomalies detected.")

def show_reporting_tab(df: pd.DataFrame, metrics: Dict, insights: List[Dict], analysis_df: Optional[pd.DataFrame]):
    """Display reporting tab."""
    st.markdown('<h2 class="section-title">📑 Reporting & Export</h2>', unsafe_allow_html=True)

    st.markdown('<h3 class="subsection-title">📊 Report Types</h3>', unsafe_allow_html=True)

    report_type = st.radio(
        "Select Report Type",
        ['Excel Comprehensive Report', 'HTML Interactive Report', 'CSV Raw Data', 'International Products CSV'],
        horizontal=True
    )

    st.markdown('<h3 class="subsection-title">🛠️ Generate Report</h3>', unsafe_allow_html=True)

    report_cols = st.columns(4)

    with report_cols[0]:
        if st.button("📈 Excel Report", use_container_width=True):
            with st.spinner("Generating Excel report..."):
                generator = ReportGenerator()
                excel_data = generator.generate_excel_report(df, metrics, insights)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_data,
                    file_name=f"pharma_report_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    with report_cols[1]:
        if st.button("🌐 HTML Report", use_container_width=True):
            with st.spinner("Generating HTML report..."):
                generator = ReportGenerator()
                html_data = generator.generate_html_report(df, metrics, insights)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="⬇️ Download HTML",
                    data=html_data,
                    file_name=f"pharma_report_{timestamp}.html",
                    mime="text/html",
                    use_container_width=True
                )

    with report_cols[2]:
        if st.button("💾 CSV Export", use_container_width=True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"pharma_data_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )

    with report_cols[3]:
        if st.button("🔄 Reset Analysis", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown('<h3 class="subsection-title">📈 Quick Statistics</h3>', unsafe_allow_html=True)

    stats_cols = st.columns(4)

    with stats_cols[0]:
        st.metric("Total Rows", f"{len(df):,}")

    with stats_cols[1]:
        st.metric("Total Columns", len(df.columns))

    with stats_cols[2]:
        memory_usage = df.memory_usage(deep=True).sum()/1024**2
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")

    with stats_cols[3]:
        intl_count = metrics.get('International_Product_Count', 0)
        st.metric("International Products", intl_count)

# ================================================
# 9. APPLICATION ENTRY POINT
# ================================================

if __name__ == "__main__":
    try:
        gc.enable()
        st.session_state.setdefault('app_started', True)
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Detailed error information:")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Reload Application", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
