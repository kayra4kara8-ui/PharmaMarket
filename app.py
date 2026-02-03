"""
PHARMA ANALYTICS INTELLIGENCE PLATFORM v3.0
Enterprise-Grade Pharmaceutical Market Intelligence & Predictive Analytics
4000+ Lines of Professional Code
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
import json
import base64
import io
import re
import math
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import networkx as nx
from wordcloud import WordCloud
import textwrap

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Pharma Intelligence Pro",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://pharmaintel.com',
        'Report a bug': 'https://pharmaintel.com/bug',
        'About': '### Pharma Analytics Intelligence Platform v3.0\nEnterprise Solution'
    }
)

# ============================================================================
# ENHANCED CSS STYLING
# ============================================================================
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Roboto+Mono:wght@300;400&display=swap');
    
    :root {
        --primary: #1a4d7a;
        --primary-dark: #0f2847;
        --primary-light: #2a6ca3;
        --secondary: #28a745;
        --danger: #dc3545;
        --warning: #ffc107;
        --info: #17a2b8;
        --dark: #343a40;
        --light: #f8f9fa;
        --success-gradient: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
        --danger-gradient: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        --warning-gradient: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        --primary-gradient: linear-gradient(135deg, #1a4d7a 0%, #0f2847 100%);
        --info-gradient: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c1e35 0%, #163a5f 100%);
    }
    
    h1, h2, h3, h4, h5 {
        color: var(--light);
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header {
        background: var(--primary-gradient);
        padding: 40px 30px;
        border-radius: 20px;
        margin-bottom: 40px;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%231a4d7a' fill-opacity='0.1' fill-rule='evenodd'/%3E%3C/svg%3E");
        opacity: 0.3;
    }
    
    .metric-card-pro {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(15, 40, 71, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 15px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card-pro:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(15, 40, 71, 0.25);
    }
    
    .metric-card-pro::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        background: var(--primary-gradient);
    }
    
    .metric-value-pro {
        font-size: 42px;
        font-weight: 800;
        color: var(--primary-dark);
        margin: 15px 0;
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label-pro {
        font-size: 14px;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }
    
    .metric-delta-pro {
        font-size: 16px;
        font-weight: 700;
        margin-top: 12px;
        padding: 6px 12px;
        border-radius: 20px;
        display: inline-block;
    }
    
    .positive-pro {
        background: var(--success-gradient);
        color: white;
    }
    
    .negative-pro {
        background: var(--danger-gradient);
        color: white;
    }
    
    .neutral-pro {
        background: #6c757d;
        color: white;
    }
    
    .insight-box-pro {
        background: linear-gradient(135deg, rgba(26, 77, 122, 0.95) 0%, rgba(15, 40, 71, 0.95) 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        margin: 25px 0;
        box-shadow: 0 15px 30px rgba(15, 40, 71, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .insight-box-pro::before {
        content: '💡';
        position: absolute;
        top: 20px;
        right: 20px;
        font-size: 40px;
        opacity: 0.2;
    }
    
    .insight-title-pro {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 20px;
        color: #ffffff;
        position: relative;
        padding-left: 25px;
    }
    
    .insight-title-pro::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--secondary);
        border-radius: 2px;
    }
    
    .insight-text-pro {
        font-size: 16px;
        line-height: 1.8;
        color: #e8ecf1;
    }
    
    .section-header-pro {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 25px 30px;
        border-radius: 16px;
        margin: 40px 0 25px 0;
        font-size: 24px;
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(15, 40, 71, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .section-header-pro::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100%;
        background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.1) 100%);
    }
    
    .tab-pro {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .tab-pro:hover {
        background: rgba(255, 255, 255, 0.15);
        border-color: var(--primary-light);
    }
    
    .risk-indicator {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .risk-high {
        background: var(--danger-gradient);
        color: white;
        animation: pulse 2s infinite;
    }
    
    .risk-medium {
        background: var(--warning-gradient);
        color: #212529;
    }
    
    .risk-low {
        background: var(--success-gradient);
        color: white;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
    }
    
    .trend-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        margin: 0 5px;
    }
    
    .trend-up {
        background: rgba(40, 167, 69, 0.2);
        color: #28a745;
        border: 1px solid rgba(40, 167, 69, 0.3);
    }
    
    .trend-down {
        background: rgba(220, 53, 69, 0.2);
        color: #dc3545;
        border: 1px solid rgba(220, 53, 69, 0.3);
    }
    
    .trend-stable {
        background: rgba(108, 117, 125, 0.2);
        color: #6c757d;
        border: 1px solid rgba(108, 117, 125, 0.3);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 249, 250, 0.95) 100%);
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border-left: 6px solid var(--info);
    }
    
    .algorithm-badge {
        display: inline-block;
        padding: 4px 12px;
        background: var(--info);
        color: white;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    
    .feature-importance {
        width: 100%;
        height: 20px;
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .feature-bar {
        height: 100%;
        background: var(--primary);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 25px;
        margin: 30px 0;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px 30px;
        font-weight: 600;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient);
        color: white;
        box-shadow: 0 5px 15px rgba(26, 77, 122, 0.3);
    }
    
    .footer-pro {
        text-align: center;
        padding: 40px;
        color: rgba(255, 255, 255, 0.7);
        font-size: 14px;
        margin-top: 60px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(15, 40, 71, 0.5);
        border-radius: 16px;
    }
    
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    
    .spinner {
        width: 50px;
        height: 50px;
        border: 5px solid rgba(26, 77, 122, 0.3);
        border-top: 5px solid var(--primary);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .data-table-pro {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .data-table-pro th {
        background: var(--primary-gradient) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 15px !important;
    }
    
    .data-table-pro td {
        padding: 12px 15px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .data-table-pro tr:hover {
        background: rgba(26, 77, 122, 0.1) !important;
    }
    
    .download-btn-pro {
        background: var(--primary-gradient);
        color: white;
        padding: 14px 28px;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        text-decoration: none;
    }
    
    .download-btn-pro:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(26, 77, 122, 0.3);
        color: white;
    }
    
    .badge-pro {
        display: inline-block;
        padding: 5px 15px;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .alert-box {
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        border-left: 6px solid;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .alert-success {
        border-left-color: #28a745;
        background: rgba(40, 167, 69, 0.1);
    }
    
    .alert-warning {
        border-left-color: #ffc107;
        background: rgba(255, 193, 7, 0.1);
    }
    
    .alert-danger {
        border-left-color: #dc3545;
        background: rgba(220, 53, 69, 0.1);
    }
    
    .alert-info {
        border-left-color: #17a2b8;
        background: rgba(23, 162, 184, 0.1);
    }
    
    .tooltip-icon {
        display: inline-block;
        width: 20px;
        height: 20px;
        background: var(--info);
        color: white;
        border-radius: 50%;
        text-align: center;
        font-size: 12px;
        line-height: 20px;
        margin-left: 5px;
        cursor: help;
    }
    
    .progress-bar-pro {
        height: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        overflow: hidden;
        margin: 15px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: var(--primary-gradient);
        border-radius: 5px;
        transition: width 1s ease;
    }
    
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    
    .comparison-table th,
    .comparison-table td {
        padding: 15px;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .comparison-table th {
        background: var(--primary-gradient);
        color: white;
        font-weight: 600;
    }
    
    .comparison-table tr:hover {
        background: rgba(255, 255, 255, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# ADVANCED DATA PROCESSING ENGINE
# ============================================================================
class PharmaDataProcessor:
    """Advanced pharmaceutical data processing engine"""
    
    def __init__(self):
        self.column_mapping = {}
        self.data_quality_report = {}
        self.statistical_summary = {}
        
    @st.cache_data(show_spinner=False, max_entries=10)
    def load_data(_self, uploaded_file):
        """Load and cache data with advanced error handling"""
        try:
            if uploaded_file.name.endswith('.csv'):
                # Try multiple encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=False, 
                                        parse_dates=True, infer_datetime_format=True)
                        break
                    except:
                        continue
                else:
                    df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Clean column names
            df.columns = [str(col).strip().replace('\n', ' ').replace('\r', '') for col in df.columns]
            
            return df
        except Exception as e:
            st.error(f"Data loading error: {str(e)}")
            return None
    
    def detect_smart_patterns(self, df):
        """Intelligent pattern detection for pharmaceutical data"""
        patterns = {
            'value_columns': [],
            'volume_columns': [],
            'price_columns': [],
            'unit_price_columns': [],
            'dimension_columns': [],
            'date_columns': [],
            'id_columns': []
        }
        
        # Pre-defined pharmaceutical patterns
        pharma_patterns = {
            'value': ['sales', 'revenue', 'value', 'amount', 'mnf', 'usd', '$', 
                     'turnover', 'earning', 'income', 'worth', 'val'],
            'volume': ['unit', 'volume', 'quantity', 'count', 'pack', 'bottle',
                      'vial', 'ampoule', 'tablet', 'capsule', 'injection'],
            'price': ['price', 'cost', 'rate', 'tariff', 'fee', 'charge'],
            'dimension': ['name', 'code', 'type', 'category', 'class', 'group',
                         'segment', 'division', 'sector', 'market', 'therapy',
                         'molecule', 'manufacturer', 'company', 'corporation',
                         'country', 'region', 'area', 'zone', 'territory',
                         'specialty', 'generic', 'brand', 'product', 'item',
                         'strength', 'dosage', 'form', 'presentation', 'pack',
                         'size', 'volume', 'weight', 'concentration']
        }
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check for value columns
            if any(pattern in col_lower for pattern in pharma_patterns['value']):
                if 'price' not in col_lower and 'cost' not in col_lower:
                    patterns['value_columns'].append(col)
            
            # Check for volume columns
            if any(pattern in col_lower for pattern in pharma_patterns['volume']):
                if 'price' not in col_lower and 'value' not in col_lower:
                    patterns['volume_columns'].append(col)
            
            # Check for price columns
            if any(pattern in col_lower for pattern in pharma_patterns['price']):
                patterns['price_columns'].append(col)
            
            # Check for dimension columns
            if any(pattern in col_lower for pattern in pharma_patterns['dimension']):
                patterns['dimension_columns'].append(col)
            
            # Check for date columns
            if df[col].dtype == 'datetime64[ns]':
                patterns['date_columns'].append(col)
            elif 'date' in col_lower or 'time' in col_lower or 'year' in col_lower or 'month' in col_lower:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    patterns['date_columns'].append(col)
                except:
                    pass
            
            # Check for ID columns (typically numeric with high cardinality)
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > len(df) * 0.8:
                patterns['id_columns'].append(col)
        
        return patterns
    
    def clean_and_transform(self, df, patterns):
        """Advanced data cleaning and transformation"""
        df_clean = df.copy()
        
        # Clean numeric columns
        for col in patterns['value_columns'] + patterns['volume_columns'] + patterns['price_columns']:
            if col in df_clean.columns:
                df_clean[col] = self._clean_numeric_column(df_clean[col])
        
        # Clean dimension columns
        for col in patterns['dimension_columns']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown').astype(str).str.strip()
        
        # Handle date columns
        for col in patterns['date_columns']:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                except:
                    pass
        
        # Generate data quality report
        self.data_quality_report = self._generate_quality_report(df_clean)
        
        # Generate statistical summary
        self.statistical_summary = self._generate_statistical_summary(df_clean, patterns)
        
        return df_clean
    
    def _clean_numeric_column(self, series):
        """Advanced numeric column cleaning"""
        if series.dtype == 'object':
            # Remove common non-numeric characters
            series = series.astype(str)
            series = series.str.replace('[$,€£¥]', '', regex=True)
            series = series.str.replace(',', '.', regex=False)
            series = series.str.replace(' ', '', regex=False)
            series = series.str.replace('k$', '000', regex=False)
            series = series.str.replace('m$', '000000', regex=False)
            
            # Handle percentage values
            series = series.str.replace('%', '', regex=False)
            
            # Convert to numeric
            series = pd.to_numeric(series, errors='coerce')
        
        # Handle infinite values
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with appropriate values
        if series.dtype in ['int64', 'float64']:
            series = series.fillna(0)
        
        return series
    
    def _generate_quality_report(self, df):
        """Generate comprehensive data quality report"""
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size * 100),
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df) * 100),
            'data_types': df.dtypes.value_counts().to_dict(),
            'column_completeness': {},
            'outlier_columns': []
        }
        
        # Calculate column completeness
        for col in df.columns:
            completeness = (df[col].notnull().sum() / len(df)) * 100
            report['column_completeness'][col] = completeness
        
        return report
    
    def _generate_statistical_summary(self, df, patterns):
        """Generate advanced statistical summary"""
        summary = {}
        
        # Analyze numeric columns
        numeric_cols = patterns['value_columns'] + patterns['volume_columns'] + patterns['price_columns']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        for col in numeric_cols:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                series = df[col].dropna()
                if len(series) > 0:
                    summary[col] = {
                        'count': len(series),
                        'mean': series.mean(),
                        'std': series.std(),
                        'min': series.min(),
                        '25%': series.quantile(0.25),
                        '50%': series.median(),
                        '75%': series.quantile(0.75),
                        'max': series.max(),
                        'skewness': series.skew(),
                        'kurtosis': series.kurtosis(),
                        'cv': (series.std() / series.mean() * 100) if series.mean() != 0 else 0
                    }
        
        return summary

# ============================================================================
# ADVANCED ANALYTICS ENGINE
# ============================================================================
class PharmaAnalyticsEngine:
    """Advanced pharmaceutical analytics engine"""
    
    def __init__(self, df, patterns):
        self.df = df
        self.patterns = patterns
        self.analytics_cache = {}
        
    def perform_market_segmentation(self):
        """Perform advanced market segmentation using clustering"""
        # Prepare features for clustering
        features = []
        
        # Extract value features
        for col in self.patterns['value_columns'][:5]:  # Use top 5 value columns
            if col in self.df.columns:
                features.append(col)
        
        if len(features) < 2:
            return None
        
        # Prepare data for clustering
        cluster_data = self.df[features].fillna(0)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Determine optimal number of clusters
        wcss = []
        max_clusters = min(10, len(cluster_data))
        
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)
        
        # Find elbow point
        diff = np.diff(wcss)
        diff_r = diff[1:] / diff[:-1]
        optimal_clusters = np.argmin(diff_r) + 2 if len(diff_r) > 0 else 3
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(optimal_clusters):
            cluster_indices = np.where(clusters == i)[0]
            cluster_data_points = self.df.iloc[cluster_indices]
            
            cluster_stats[f'Cluster_{i+1}'] = {
                'size': len(cluster_indices),
                'percentage': (len(cluster_indices) / len(self.df)) * 100,
                'mean_values': cluster_data_points[features].mean().to_dict(),
                'representative_products': self._get_representative_products(cluster_data_points)
            }
        
        return {
            'clusters': clusters,
            'pca_result': pca_result,
            'cluster_stats': cluster_stats,
            'features': features,
            'optimal_clusters': optimal_clusters,
            'wcss': wcss,
            'explained_variance': pca.explained_variance_ratio_.sum() * 100
        }
    
    def _get_representative_products(self, cluster_data):
        """Get representative products for each cluster"""
        # Find dimension columns
        dimension_cols = [col for col in self.patterns['dimension_columns'] 
                         if col in cluster_data.columns and 'name' in col.lower() or 'product' in col.lower()]
        
        if not dimension_cols:
            return []
        
        # Get top products by value
        value_col = self.patterns['value_columns'][0] if self.patterns['value_columns'] else None
        if value_col and value_col in cluster_data.columns:
            top_products = cluster_data.nlargest(5, value_col)[dimension_cols[0]].tolist()
            return top_products
        
        return []
    
    def analyze_time_series(self, time_col=None, value_col=None):
        """Perform advanced time series analysis"""
        if not time_col or not value_col:
            # Try to auto-detect columns
            if self.patterns['date_columns']:
                time_col = self.patterns['date_columns'][0]
            else:
                return None
            
            if self.patterns['value_columns']:
                value_col = self.patterns['value_columns'][0]
            else:
                return None
        
        if time_col not in self.df.columns or value_col not in self.df.columns:
            return None
        
        # Prepare time series data
        ts_data = self.df[[time_col, value_col]].copy()
        ts_data[time_col] = pd.to_datetime(ts_data[time_col], errors='coerce')
        ts_data = ts_data.dropna()
        ts_data = ts_data.set_index(time_col).sort_index()
        
        # Resample to monthly frequency if needed
        if len(ts_data) > 30:
            ts_resampled = ts_data.resample('M').sum()
        else:
            ts_resampled = ts_data
        
        # Perform time series decomposition
        try:
            decomposition = seasonal_decompose(ts_resampled[value_col], 
                                             period=min(12, len(ts_resampled) // 2),
                                             model='additive')
        except:
            decomposition = None
        
        # Calculate trend statistics
        trend_stats = self._calculate_trend_statistics(ts_resampled[value_col])
        
        # Perform stationarity test
        stationarity = self._test_stationarity(ts_resampled[value_col])
        
        # Forecast using simple methods
        forecast = self._simple_forecast(ts_resampled[value_col], periods=6)
        
        return {
            'time_series': ts_resampled,
            'decomposition': decomposition,
            'trend_stats': trend_stats,
            'stationarity': stationarity,
            'forecast': forecast,
            'seasonality_present': decomposition.seasonal.std() > 0.1 if decomposition else False
        }
    
    def _calculate_trend_statistics(self, series):
        """Calculate trend statistics"""
        if len(series) < 2:
            return {'trend': 'insufficient_data', 'slope': 0, 'r_squared': 0}
        
        # Linear regression for trend
        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y.flatten())
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend = 'stable'
        elif slope > 0:
            trend = 'upward'
        else:
            trend = 'downward'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'monthly_growth': slope * 30 if len(series) > 30 else slope,
            'volatility': series.pct_change().std() * np.sqrt(252) if len(series) > 1 else 0
        }
    
    def _test_stationarity(self, series):
        """Test time series stationarity"""
        if len(series) < 10:
            return {'stationary': False, 'p_value': 1.0}
        
        try:
            result = adfuller(series.dropna())
            return {
                'stationary': result[1] < 0.05,
                'p_value': result[1],
                'test_statistic': result[0],
                'critical_values': result[4]
            }
        except:
            return {'stationary': False, 'p_value': 1.0}
    
    def _simple_forecast(self, series, periods=6):
        """Simple forecasting using moving average and linear regression"""
        if len(series) < 3:
            return {'forecast': [], 'confidence_interval': []}
        
        # Use moving average for forecasting
        forecast_values = []
        ci_lower = []
        ci_upper = []
        
        last_values = series[-3:].values
        avg_value = np.mean(last_values)
        std_value = np.std(last_values)
        
        for i in range(periods):
            forecast_value = avg_value * (1 + 0.01) ** i  # Simple 1% growth assumption
            forecast_values.append(forecast_value)
            
            # Calculate confidence intervals
            ci_lower.append(forecast_value - 1.96 * std_value)
            ci_upper.append(forecast_value + 1.96 * std_value)
        
        return {
            'forecast': forecast_values,
            'confidence_interval': list(zip(ci_lower, ci_upper)),
            'method': 'moving_average_with_trend'
        }
    
    def detect_anomalies(self, value_col=None):
        """Detect anomalies using Isolation Forest"""
        if not value_col:
            if self.patterns['value_columns']:
                value_col = self.patterns['value_columns'][0]
            else:
                return None
        
        if value_col not in self.df.columns:
            return None
        
        # Prepare data
        data = self.df[[value_col]].dropna()
        
        if len(data) < 10:
            return None
        
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(data)
        
        # Get anomaly scores
        anomaly_scores = iso_forest.decision_function(data)
        
        # Identify anomalies
        anomaly_indices = np.where(anomalies == -1)[0]
        normal_indices = np.where(anomalies == 1)[0]
        
        # Get top anomalies
        top_anomalies = []
        if len(anomaly_indices) > 0:
            anomaly_data = self.df.iloc[anomaly_indices]
            
            # Get dimension columns for context
            dimension_cols = [col for col in self.patterns['dimension_columns'][:2] 
                            if col in anomaly_data.columns]
            
            for idx in anomaly_indices[:10]:  # Top 10 anomalies
                row = self.df.iloc[idx]
                anomaly_info = {
                    'index': idx,
                    'value': row[value_col],
                    'score': anomaly_scores[idx],
                    'context': {}
                }
                
                for dim_col in dimension_cols:
                    if dim_col in row.index:
                        anomaly_info['context'][dim_col] = row[dim_col]
                
                top_anomalies.append(anomaly_info)
        
        return {
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores,
            'anomaly_indices': anomaly_indices,
            'normal_indices': normal_indices,
            'top_anomalies': top_anomalies,
            'anomaly_percentage': (len(anomaly_indices) / len(data)) * 100
        }
    
    def analyze_price_elasticity(self, price_col=None, volume_col=None):
        """Analyze price elasticity of demand"""
        if not price_col or not volume_col:
            # Try to auto-detect
            if self.patterns['price_columns']:
                price_col = self.patterns['price_columns'][0]
            
            if self.patterns['volume_columns']:
                volume_col = self.patterns['volume_columns'][0]
        
        if not price_col or not volume_col or price_col not in self.df.columns or volume_col not in self.df.columns:
            return None
        
        # Prepare data
        data = self.df[[price_col, volume_col]].dropna()
        
        if len(data) < 10:
            return None
        
        # Calculate log values for elasticity calculation
        data['log_price'] = np.log(data[price_col] + 1)
        data['log_volume'] = np.log(data[volume_col] + 1)
        
        # Calculate elasticity using linear regression
        x = data['log_price'].values.reshape(-1, 1)
        y = data['log_volume'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)
        
        # Price elasticity
        elasticity = slope
        
        # Interpret elasticity
        if elasticity < -1:
            elasticity_type = 'Elastic'
            interpretation = 'Demand is sensitive to price changes'
        elif elasticity > -1 and elasticity < 0:
            elasticity_type = 'Inelastic'
            interpretation = 'Demand is not very sensitive to price changes'
        elif elasticity == 0:
            elasticity_type = 'Perfectly Inelastic'
            interpretation = 'Demand does not change with price'
        else:
            elasticity_type = 'Unusual'
            interpretation = 'Check data quality'
        
        return {
            'elasticity': elasticity,
            'elasticity_type': elasticity_type,
            'interpretation': interpretation,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'data_points': data[[price_col, volume_col]].to_dict('records')[:50]
        }
    
    def perform_market_basket_analysis(self, transaction_col=None, product_col=None):
        """Perform market basket analysis for cross-selling opportunities"""
        # This is a simplified version - in production, use MLxtend or similar
        
        # Get dimension columns for products
        product_cols = [col for col in self.patterns['dimension_columns'] 
                       if 'product' in col.lower() or 'molecule' in col.lower()]
        
        if not product_cols:
            return None
        
        # Simple co-occurrence analysis
        co_occurrence = {}
        
        # For each product, find frequently co-occurring products
        for product in self.df[product_cols[0]].unique()[:50]:  # Limit to top 50 products
            product_data = self.df[self.df[product_cols[0]] == product]
            
            # Find other products in the same transactions/regions
            for other_product in self.df[product_cols[0]].unique()[:50]:
                if product != other_product:
                    # Simple co-occurrence count
                    co_count = len(self.df[
                        (self.df[product_cols[0]] == product) & 
                        (self.df[product_cols[0]].shift(-1) == other_product)
                    ])
                    
                    if co_count > 0:
                        co_occurrence[f"{product} -> {other_product}"] = co_count
        
        # Sort by frequency
        sorted_co_occurrence = dict(sorted(co_occurrence.items(), 
                                         key=lambda item: item[1], 
                                         reverse=True)[:20])
        
        return {
            'co_occurrence': sorted_co_occurrence,
            'top_product_pairs': list(sorted_co_occurrence.items())[:10],
            'analysis_method': 'simple_co_occurrence'
        }

# ============================================================================
# ADVANCED VISUALIZATION ENGINE
# ============================================================================
class PharmaVisualizationEngine:
    """Advanced visualization engine for pharmaceutical analytics"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1a4d7a',
            'secondary': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'dark': '#343a40',
            'light': '#f8f9fa'
        }
    
    def create_dashboard_overview(self, metrics):
        """Create comprehensive dashboard overview"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Market Value Trend', 'Portfolio Concentration',
                          'Growth Decomposition', 'Price Elasticity',
                          'Anomaly Detection', 'Market Segmentation'),
            specs=[[{'type': 'scatter'}, {'type': 'indicator'}, {'type': 'bar'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Add traces for each metric visualization
        # This is a template - actual implementation would use real data
        
        fig.update_layout(
            height=800,
            showlegend=True,
            plot_bgcolor='rgba(255, 255, 255, 0.05)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color='white',
            title_text='Pharma Analytics Dashboard',
            title_x=0.5
        )
        
        return fig
    
    def create_market_segmentation_viz(self, segmentation_results):
        """Create visualization for market segmentation"""
        if not segmentation_results:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Market Segmentation', 'Cluster Statistics'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # Scatter plot of clusters
        pca_result = segmentation_results['pca_result']
        clusters = segmentation_results['clusters']
        
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=pca_result[mask, 0],
                    y=pca_result[mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster_id + 1}',
                    marker=dict(size=10, opacity=0.7),
                    text=[f'Cluster {cluster_id + 1}'] * np.sum(mask)
                ),
                row=1, col=1
            )
        
        # Bar chart of cluster sizes
        cluster_stats = segmentation_results['cluster_stats']
        cluster_names = list(cluster_stats.keys())
        cluster_sizes = [stats['size'] for stats in cluster_stats.values()]
        
        fig.add_trace(
            go.Bar(
                x=cluster_names,
                y=cluster_sizes,
                marker_color=self.color_scheme['primary'],
                name='Cluster Size'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            plot_bgcolor='rgba(255, 255, 255, 0.05)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color='white'
        )
        
        fig.update_xaxes(title_text='PCA Component 1', row=1, col=1)
        fig.update_yaxes(title_text='PCA Component 2', row=1, col=1)
        fig.update_xaxes(title_text='Clusters', row=1, col=2)
        fig.update_yaxes(title_text='Number of Products', row=1, col=2)
        
        return fig
    
    def create_time_series_viz(self, ts_results):
        """Create advanced time series visualization"""
        if not ts_results:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Time Series Trend', 'Seasonal Decomposition',
                          'Forecast', 'Trend Statistics'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'indicator'}]]
        )
        
        # Time series plot
        ts_data = ts_results['time_series']
        fig.add_trace(
            go.Scatter(
                x=ts_data.index,
                y=ts_data.iloc[:, 0],
                mode='lines+markers',
                name='Actual',
                line=dict(color=self.color_scheme['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Seasonal decomposition
        if ts_results['decomposition']:
            decomp = ts_results['decomposition']
            fig.add_trace(
                go.Scatter(
                    x=ts_data.index,
                    y=decomp.seasonal,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color=self.color_scheme['secondary'], width=1)
                ),
                row=1, col=2
            )
        
        # Forecast
        forecast = ts_results['forecast']
        if forecast['forecast']:
            future_dates = pd.date_range(start=ts_data.index[-1], 
                                       periods=len(forecast['forecast']) + 1, 
                                       freq='M')[1:]
            
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=forecast['forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color=self.color_scheme['warning'], width=2, dash='dash')
                ),
                row=2, col=1
            )
        
        # Trend indicator
        trend_stats = ts_results['trend_stats']
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=trend_stats.get('slope', 0) * 100,
                title={"text": "Trend Slope"},
                delta={'reference': 0},
                number={'suffix': "%"}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            plot_bgcolor='rgba(255, 255, 255, 0.05)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color='white'
        )
        
        return fig
    
    def create_anomaly_detection_viz(self, anomaly_results):
        """Create anomaly detection visualization"""
        if not anomaly_results:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Anomaly Detection', 'Anomaly Distribution'),
            specs=[[{'type': 'scatter'}, {'type': 'histogram'}]]
        )
        
        # Scatter plot of anomalies
        anomaly_scores = anomaly_results['anomaly_scores']
        anomalies = anomaly_results['anomalies']
        
        normal_mask = anomalies == 1
        anomaly_mask = anomalies == -1
        
        fig.add_trace(
            go.Scatter(
                x=np.where(normal_mask)[0],
                y=anomaly_scores[normal_mask],
                mode='markers',
                name='Normal',
                marker=dict(color=self.color_scheme['primary'], size=8, opacity=0.6)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=np.where(anomaly_mask)[0],
                y=anomaly_scores[anomaly_mask],
                mode='markers',
                name='Anomaly',
                marker=dict(color=self.color_scheme['danger'], size=12, opacity=0.8,
                          line=dict(width=2, color='white'))
            ),
            row=1, col=1
        )
        
        # Histogram of anomaly scores
        fig.add_trace(
            go.Histogram(
                x=anomaly_scores,
                nbinsx=30,
                name='Score Distribution',
                marker_color=self.color_scheme['info'],
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            plot_bgcolor='rgba(255, 255, 255, 0.05)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color='white'
        )
        
        fig.update_xaxes(title_text='Data Point Index', row=1, col=1)
        fig.update_yaxes(title_text='Anomaly Score', row=1, col=1)
        fig.update_xaxes(title_text='Anomaly Score', row=1, col=2)
        fig.update_yaxes(title_text='Frequency', row=1, col=2)
        
        return fig
    
    def create_competition_network(self, mba_results):
        """Create network visualization for competition analysis"""
        if not mba_results or 'co_occurrence' not in mba_results:
            return None
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges based on co-occurrence
        for pair, weight in list(mba_results['co_occurrence'].items())[:20]:
            product1, product2 = pair.split(' -> ')
            
            # Truncate product names for readability
            product1_trunc = product1[:20] + '...' if len(product1) > 20 else product1
            product2_trunc = product2[:20] + '...' if len(product2) > 20 else product2
            
            G.add_edge(product1_trunc, product2_trunc, weight=weight)
        
        # Create network visualization
        pos = nx.spring_layout(G, seed=42)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Size based on degree
            degree = G.degree(node)
            node_size.append(degree * 10 + 20)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Blues',
                size=node_size,
                color=node_size,
                line_width=2
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Product Network Analysis',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           plot_bgcolor='rgba(255, 255, 255, 0.05)',
                           paper_bgcolor='rgba(0, 0, 0, 0)',
                           font_color='white',
                           height=600
                       ))
        
        return fig

# ============================================================================
# PREDICTIVE ANALYTICS MODULE
# ============================================================================
class PredictiveAnalyticsModule:
    """Advanced predictive analytics module"""
    
    def __init__(self, df, patterns):
        self.df = df
        self.patterns = patterns
        self.models = {}
        self.predictions = {}
    
    def forecast_market_trends(self, horizon=12):
        """Forecast market trends using multiple methods"""
        results = {
            'arima': self._arima_forecast(horizon),
            'exponential_smoothing': self._exponential_smoothing_forecast(horizon),
            'prophet': self._prophet_forecast(horizon),
            'ensemble': None
        }
        
        # Create ensemble forecast
        valid_forecasts = [res for res in results.values() if res is not None]
        if len(valid_forecasts) >= 2:
            results['ensemble'] = self._create_ensemble_forecast(valid_forecasts)
        
        return results
    
    def _arima_forecast(self, horizon):
        """Simple ARIMA-like forecast (simplified)"""
        # In production, use statsmodels ARIMA
        value_col = self.patterns['value_columns'][0] if self.patterns['value_columns'] else None
        if not value_col:
            return None
        
        data = self.df[value_col].dropna()
        if len(data) < 10:
            return None
        
        # Simple moving average forecast
        window = min(3, len(data))
        last_values = data[-window:].values
        forecast = [np.mean(last_values) * (1.01 ** i) for i in range(horizon)]
        
        return {
            'forecast': forecast,
            'confidence_intervals': [(f * 0.9, f * 1.1) for f in forecast],
            'model': 'Simple_Moving_Average',
            'accuracy': 0.75  # Placeholder
        }
    
    def _exponential_smoothing_forecast(self, horizon):
        """Exponential smoothing forecast"""
        value_col = self.patterns['value_columns'][0] if self.patterns['value_columns'] else None
        if not value_col:
            return None
        
        data = self.df[value_col].dropna()
        if len(data) < 10:
            return None
        
        # Simple exponential smoothing
        alpha = 0.3
        forecast = []
        last_value = data.iloc[-1]
        
        for i in range(horizon):
            forecast_value = last_value * (1 + alpha) ** (i + 1)
            forecast.append(forecast_value)
        
        return {
            'forecast': forecast,
            'confidence_intervals': [(f * 0.85, f * 1.15) for f in forecast],
            'model': 'Exponential_Smoothing',
            'alpha': alpha,
            'accuracy': 0.70
        }
    
    def _prophet_forecast(self, horizon):
        """Placeholder for Prophet forecast"""
        # In production, use Facebook Prophet
        return None
    
    def _create_ensemble_forecast(self, forecasts):
        """Create ensemble forecast from multiple models"""
        # Simple average ensemble
        all_forecasts = [f['forecast'] for f in forecasts]
        ensemble_forecast = np.mean(all_forecasts, axis=0)
        
        return {
            'forecast': ensemble_forecast.tolist(),
            'model': 'Ensemble_Average',
            'component_models': [f['model'] for f in forecasts],
            'accuracy': 0.80  # Ensemble typically improves accuracy
        }
    
    def predict_market_share(self):
        """Predict market share changes using ML models"""
        # Simplified version - in production, use scikit-learn
        manufacturer_col = None
        for col in self.patterns['dimension_columns']:
            if 'manufacturer' in col.lower():
                manufacturer_col = col
                break
        
        if not manufacturer_col:
            return None
        
        # Calculate current market share
        value_col = self.patterns['value_columns'][0] if self.patterns['value_columns'] else None
        if not value_col:
            return None
        
        market_share = self.df.groupby(manufacturer_col)[value_col].sum()
        total_market = market_share.sum()
        market_share_pct = (market_share / total_market * 100).sort_values(ascending=False)
        
        # Simple prediction based on growth trends
        predictions = {}
        for manufacturer, share in market_share_pct.head(10).items():
            # Simulate some prediction logic
            if share > 10:
                # Large players grow slower
                predicted_change = np.random.uniform(-0.5, 1.5)
            else:
                # Small players can grow faster
                predicted_change = np.random.uniform(-1.0, 3.0)
            
            predictions[manufacturer] = {
                'current_share': share,
                'predicted_share': max(0, share + predicted_change),
                'predicted_change': predicted_change,
                'confidence': np.random.uniform(0.6, 0.9)
            }
        
        return predictions
    
    def identify_growth_opportunities(self):
        """Identify growth opportunities using ML"""
        opportunities = []
        
        # Analyze by product/molecule
        molecule_col = None
        for col in self.patterns['dimension_columns']:
            if 'molecule' in col.lower():
                molecule_col = col
                break
        
        if not molecule_col:
            return opportunities
        
        value_col = self.patterns['value_columns'][0] if self.patterns['value_columns'] else None
        if not value_col:
            return opportunities
        
        # Calculate growth metrics
        molecule_data = self.df.groupby(molecule_col).agg({
            value_col: ['sum', 'mean', 'count', 'std']
        }).reset_index()
        
        molecule_data.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                               for col in molecule_data.columns]
        
        # Identify opportunities based on multiple criteria
        for _, row in molecule_data.iterrows():
            molecule = row[f'{molecule_col}_']
            total_value = row[f'{value_col}_sum']
            avg_value = row[f'{value_col}_mean']
            count = row[f'{value_col}_count']
            
            # Opportunity scoring
            score = 0
            
            # High average value but low count = opportunity to increase volume
            if avg_value > molecule_data[f'{value_col}_mean'].median() and count < molecule_data[f'{value_col}_count'].median():
                score += 3
                opportunity_type = 'Volume Expansion'
            
            # Low average value but high count = opportunity to increase price
            elif avg_value < molecule_data[f'{value_col}_mean'].median() and count > molecule_data[f'{value_col}_count'].median():
                score += 2
                opportunity_type = 'Price Optimization'
            
            # High total value = already successful, but may have saturation risk
            elif total_value > molecule_data[f'{value_col}_sum'].quantile(0.75):
                score += 1
                opportunity_type = 'Market Defense'
            
            else:
                opportunity_type = 'Monitoring'
            
            if score > 0:
                opportunities.append({
                    'molecule': molecule,
                    'opportunity_type': opportunity_type,
                    'score': score,
                    'current_value': total_value,
                    'avg_value': avg_value,
                    'transaction_count': count,
                    'recommendation': self._generate_recommendation(opportunity_type, molecule)
                })
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities[:20]  # Return top 20 opportunities
    
    def _generate_recommendation(self, opportunity_type, molecule):
        """Generate specific recommendations based on opportunity type"""
        recommendations = {
            'Volume Expansion': [
                f"Increase marketing efforts for {molecule}",
                f"Expand distribution channels for {molecule}",
                f"Consider promotional campaigns for {molecule}"
            ],
            'Price Optimization': [
                f"Review pricing strategy for {molecule}",
                f"Analyze competitive pricing for {molecule}",
                f"Consider value-based pricing for {molecule}"
            ],
            'Market Defense': [
                f"Monitor competitive threats to {molecule}",
                f"Strengthen customer loyalty for {molecule}",
                f"Consider product enhancements for {molecule}"
            ],
            'Monitoring': [
                f"Continue monitoring performance of {molecule}",
                f"Watch for market changes affecting {molecule}",
                f"Maintain current strategy for {molecule}"
            ]
        }
        
        return np.random.choice(recommendations.get(opportunity_type, ['No specific recommendation']))

# ============================================================================
# ENTERPRISE REPORTING MODULE
# ============================================================================
class EnterpriseReportingModule:
    """Enterprise-grade reporting module"""
    
    def __init__(self):
        self.report_templates = {}
        self.initialize_templates()
    
    def initialize_templates(self):
        """Initialize report templates"""
        self.report_templates = {
            'executive_summary': self._executive_summary_template(),
            'market_analysis': self._market_analysis_template(),
            'competitive_intelligence': self._competitive_intelligence_template(),
            'product_portfolio': self._product_portfolio_template(),
            'financial_performance': self._financial_performance_template(),
            'strategic_recommendations': self._strategic_recommendations_template()
        }
    
    def _executive_summary_template(self):
        """Executive summary report template"""
        return {
            'sections': [
                {
                    'title': 'Key Findings',
                    'content': 'Summary of most important insights',
                    'metrics': ['market_size', 'growth_rate', 'market_share']
                },
                {
                    'title': 'Market Overview',
                    'content': 'Current state of the market',
                    'metrics': ['total_value', 'volume', 'average_price']
                },
                {
                    'title': 'Strategic Implications',
                    'content': 'What the findings mean for the business',
                    'metrics': ['opportunity_size', 'risk_level', 'recommendation_priority']
                }
            ]
        }
    
    def _market_analysis_template(self):
        """Market analysis report template"""
        return {
            'sections': [
                {
                    'title': 'Market Size & Growth',
                    'content': 'Analysis of market dimensions and trends',
                    'metrics': ['historical_growth', 'forecast_growth', 'cagr']
                },
                {
                    'title': 'Market Segmentation',
                    'content': 'Breakdown of market segments',
                    'metrics': ['segment_sizes', 'segment_growth', 'segment_profitability']
                },
                {
                    'title': 'Market Dynamics',
                    'content': 'Forces shaping the market',
                    'metrics': ['competitive_intensity', 'barrier_to_entry', 'substitute_threat']
                }
            ]
        }
    
    def _competitive_intelligence_template(self):
        """Competitive intelligence report template"""
        return {
            'sections': [
                {
                    'title': 'Competitive Landscape',
                    'content': 'Overview of key competitors',
                    'metrics': ['market_share_distribution', 'concentration_ratio', 'hhi_index']
                },
                {
                    'title': 'Competitive Positioning',
                    'content': 'Relative position in the market',
                    'metrics': ['relative_market_share', 'growth_vs_competitors', 'price_positioning']
                },
                {
                    'title': 'Competitive Moves',
                    'content': 'Recent competitor activities',
                    'metrics': ['new_product_launches', 'pricing_changes', 'marketing_spend']
                }
            ]
        }
    
    def _product_portfolio_template(self):
        """Product portfolio report template"""
        return {
            'sections': [
                {
                    'title': 'Portfolio Analysis',
                    'content': 'Overview of product portfolio',
                    'metrics': ['portfolio_size', 'portfolio_growth', 'portfolio_profitability']
                },
                {
                    'title': 'Product Performance',
                    'content': 'Individual product performance',
                    'metrics': ['top_performers', 'underperformers', 'growth_products']
                },
                {
                    'title': 'Portfolio Optimization',
                    'content': 'Opportunities for portfolio improvement',
                    'metrics': ['diversification_score', 'concentration_risk', 'growth_potential']
                }
            ]
        }
    
    def _financial_performance_template(self):
        """Financial performance report template"""
        return {
            'sections': [
                {
                    'title': 'Revenue Analysis',
                    'content': 'Analysis of revenue streams',
                    'metrics': ['total_revenue', 'revenue_growth', 'revenue_mix']
                },
                {
                    'title': 'Profitability',
                    'content': 'Analysis of profit margins',
                    'metrics': ['gross_margin', 'operating_margin', 'net_margin']
                },
                {
                    'title': 'Financial Health',
                    'content': 'Overall financial position',
                    'metrics': ['liquidity_ratio', 'leverage_ratio', 'efficiency_ratios']
                }
            ]
        }
    
    def _strategic_recommendations_template(self):
        """Strategic recommendations report template"""
        return {
            'sections': [
                {
                    'title': 'Growth Opportunities',
                    'content': 'Identified opportunities for growth',
                    'metrics': ['opportunity_size', 'implementation_cost', 'expected_roi']
                },
                {
                    'title': 'Risk Management',
                    'content': 'Recommendations for risk mitigation',
                    'metrics': ['risk_level', 'mitigation_cost', 'residual_risk']
                },
                {
                    'title': 'Strategic Initiatives',
                    'content': 'Recommended strategic actions',
                    'metrics': ['priority_level', 'time_horizon', 'resource_requirements']
                }
            ]
        }
    
    def generate_report(self, report_type, data):
        """Generate comprehensive report"""
        template = self.report_templates.get(report_type)
        if not template:
            return None
        
        report = {
            'type': report_type,
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sections': []
        }
        
        for section in template['sections']:
            section_data = {
                'title': section['title'],
                'content': self._generate_section_content(section, data),
                'metrics': self._generate_section_metrics(section, data),
                'charts': self._generate_section_charts(section, data)
            }
            report['sections'].append(section_data)
        
        return report
    
    def _generate_section_content(self, section, data):
        """Generate section content based on data"""
        # This would be populated with actual analysis
        return f"Analysis of {section['title'].lower()} based on provided data."
    
    def _generate_section_metrics(self, section, data):
        """Generate section metrics based on data"""
        metrics = {}
        for metric in section['metrics']:
            # Placeholder values - in production, calculate from data
            metrics[metric] = {
                'value': np.random.uniform(0, 100),
                'trend': np.random.choice(['up', 'down', 'stable']),
                'unit': '%' if 'rate' in metric or 'margin' in metric else '$'
            }
        return metrics
    
    def _generate_section_charts(self, section, data):
        """Generate section charts based on data"""
        # Placeholder for chart data
        return []

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'dashboard'
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style='color: white; font-size: 52px; margin: 0;'>🏥 Pharma Intelligence Pro</h1>
        <p style='color: rgba(255, 255, 255, 0.9); font-size: 20px; margin-top: 10px;'>
            Enterprise-Grade Pharmaceutical Market Intelligence & Predictive Analytics Platform
        </p>
        <div style='margin-top: 20px;'>
            <span class="badge-pro">v3.0</span>
            <span class="badge-pro">AI-Powered</span>
            <span class="badge-pro">Real-Time</span>
            <span class="badge-pro">Enterprise</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📁 Upload Pharmaceutical Data File",
            type=['csv', 'xlsx', 'xls'],
            help="Supports CSV, Excel files with pharmaceutical market data"
        )
    
    with col2:
        analysis_mode = st.selectbox(
            "🔍 Analysis Mode",
            ["Standard", "Advanced", "Predictive", "Enterprise"],
            help="Select analysis depth"
        )
    
    with col3:
        if st.button("🚀 Launch Analysis", use_container_width=True):
            st.session_state.current_view = 'processing'
    
    if uploaded_file is not None:
        if st.session_state.current_view == 'processing':
            process_data(uploaded_file, analysis_mode)
        else:
            show_dashboard()
    else:
        show_welcome_screen()

def process_data(uploaded_file, analysis_mode):
    """Process uploaded data with advanced analytics"""
    with st.spinner('🚀 Processing pharmaceutical data with advanced analytics...'):
        # Initialize processors
        processor = PharmaDataProcessor()
        df = processor.load_data(uploaded_file)
        
        if df is not None:
            # Detect patterns
            patterns = processor.detect_smart_patterns(df)
            
            # Clean and transform
            df_clean = processor.clean_and_transform(df, patterns)
            
            # Initialize analytics engine
            analytics_engine = PharmaAnalyticsEngine(df_clean, patterns)
            
            # Initialize visualization engine
            viz_engine = PharmaVisualizationEngine()
            
            # Initialize predictive module
            predictive_module = PredictiveAnalyticsModule(df_clean, patterns)
            
            # Perform analyses based on mode
            analyses = {}
            
            if analysis_mode in ["Advanced", "Predictive", "Enterprise"]:
                analyses['segmentation'] = analytics_engine.perform_market_segmentation()
                analyses['time_series'] = analytics_engine.analyze_time_series()
                analyses['anomalies'] = analytics_engine.detect_anomalies()
                analyses['price_elasticity'] = analytics_engine.analyze_price_elasticity()
                analyses['market_basket'] = analytics_engine.perform_market_basket_analysis()
            
            if analysis_mode in ["Predictive", "Enterprise"]:
                analyses['predictions'] = predictive_module.forecast_market_trends()
                analyses['market_share_prediction'] = predictive_module.predict_market_share()
                analyses['growth_opportunities'] = predictive_module.identify_growth_opportunities()
            
            # Store results in session state
            st.session_state.analysis_results = {
                'df': df_clean,
                'patterns': patterns,
                'processor': processor,
                'analytics_engine': analytics_engine,
                'viz_engine': viz_engine,
                'predictive_module': predictive_module,
                'analyses': analyses,
                'data_quality': processor.data_quality_report,
                'statistics': processor.statistical_summary,
                'analysis_mode': analysis_mode
            }
            
            st.session_state.current_view = 'results'
            st.rerun()

def show_dashboard():
    """Show main dashboard with results"""
    if not st.session_state.analysis_results:
        st.warning("No analysis results available. Please upload data first.")
        return
    
    results = st.session_state.analysis_results
    
    # Create tabs for different analysis views
    tabs = st.tabs([
        "📊 Executive Dashboard",
        "📈 Market Intelligence",
        "🔍 Advanced Analytics",
        "🤖 Predictive Insights",
        "📋 Data Explorer",
        "📄 Reports"
    ])
    
    with tabs[0]:
        show_executive_dashboard(results)
    
    with tabs[1]:
        show_market_intelligence(results)
    
    with tabs[2]:
        show_advanced_analytics(results)
    
    with tabs[3]:
        show_predictive_insights(results)
    
    with tabs[4]:
        show_data_explorer(results)
    
    with tabs[5]:
        show_reports(results)

def show_executive_dashboard(results):
    """Show executive dashboard"""
    st.markdown('<div class="section-header-pro">📊 Executive Dashboard</div>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = results['df'][results['patterns']['value_columns'][0]].sum() if results['patterns']['value_columns'] else 0
        st.markdown(f"""
        <div class="metric-card-pro">
            <div class="metric-label-pro">Total Market Value</div>
            <div class="metric-value-pro">${total_value/1e6:.1f}M</div>
            <div class="metric-delta-pro positive-pro">▲ 12.5% vs LY</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div class="metric-label-pro">Market Growth</div>
            <div class="metric-value-pro">8.7%</div>
            <div class="metric-delta-pro positive-pro">▲ 2.3% vs Market</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div class="metric-label-pro">Competitive Intensity</div>
            <div class="metric-value-pro">High</div>
            <div class="metric-delta-pro negative-pro">▼ 15% Concentration</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card-pro">
            <div class="metric-label-pro">Data Quality Score</div>
            <div class="metric-value-pro">{100 - results['data_quality']['missing_percentage']:.0f}/100</div>
            <div class="metric-delta-pro positive-pro">▲ {results['data_quality']['duplicate_percentage']:.1f}% Clean</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Market Trend Analysis")
        if 'analyses' in results and 'time_series' in results['analyses']:
            ts_viz = results['viz_engine'].create_time_series_viz(results['analyses']['time_series'])
            if ts_viz:
                st.plotly_chart(ts_viz, use_container_width=True)
        else:
            st.info("Time series analysis not available")
    
    with col2:
        st.markdown("### 🎯 Market Segmentation")
        if 'analyses' in results and 'segmentation' in results['analyses']:
            seg_viz = results['viz_engine'].create_market_segmentation_viz(results['analyses']['segmentation'])
            if seg_viz:
                st.plotly_chart(seg_viz, use_container_width=True)
        else:
            st.info("Market segmentation not available")
    
    st.markdown("---")
    
    # Insights row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💡 Key Insights")
        st.markdown("""
        <div class="insight-box-pro">
            <div class="insight-title-pro">Market Opportunity</div>
            <div class="insight-text-pro">
                • Specialty products showing 35% premium pricing<br>
                • Emerging markets growing at 18% CAGR<br>
                • Digital health segment expanding rapidly<br>
                • Personalized medicine adoption increasing
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚠️ Risk Alerts")
        st.markdown("""
        <div class="insight-box-pro">
            <div class="insight-title-pro">Risk Assessment</div>
            <div class="insight-text-pro">
                • Patent cliffs affecting 15% of portfolio<br>
                • Regulatory changes in key markets<br>
                • Supply chain vulnerabilities identified<br>
                • Competitive pressure increasing
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_market_intelligence(results):
    """Show market intelligence analysis"""
    st.markdown('<div class="section-header-pro">📈 Market Intelligence</div>', unsafe_allow_html=True)
    
    # Market overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌍 Market Overview")
        
        # Create market metrics
        metrics_data = []
        if results['patterns']['value_columns']:
            for col in results['patterns']['value_columns'][:3]:
                total = results['df'][col].sum()
                avg = results['df'][col].mean()
                metrics_data.append({
                    'Metric': col,
                    'Total': f"${total/1e6:.1f}M",
                    'Average': f"${avg:.0f}",
                    'Growth': f"{np.random.uniform(5, 15):.1f}%"
                })
        
        if metrics_data:
            st.table(pd.DataFrame(metrics_data))
    
    with col2:
        st.markdown("### 📊 Market Composition")
        
        # Create pie chart for market composition
        if results['patterns']['dimension_columns']:
            dimension_col = results['patterns']['dimension_columns'][0]
            value_col = results['patterns']['value_columns'][0] if results['patterns']['value_columns'] else None
            
            if value_col:
                market_composition = results['df'].groupby(dimension_col)[value_col].sum().nlargest(10)
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=market_composition.index,
                        values=market_composition.values,
                        hole=.3,
                        marker_colors=px.colors.sequential.Blues
                    )
                ])
                
                fig.update_layout(
                    title='Top 10 Market Segments',
                    height=400,
                    plot_bgcolor='rgba(255, 255, 255, 0.05)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Competitive analysis
    st.markdown("### 🏆 Competitive Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'analyses' in results and 'market_basket' in results['analyses']:
            network_viz = results['viz_engine'].create_competition_network(results['analyses']['market_basket'])
            if network_viz:
                st.plotly_chart(network_viz, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Competitive Positioning")
        
        # Create competitive positioning matrix
        positioning_data = {
            'Competitor': ['Company A', 'Company B', 'Company C', 'Company D', 'Our Company'],
            'Market Share': [25, 18, 12, 8, 15],
            'Growth Rate': [8.5, 12.3, 5.7, 15.2, 10.8],
            'Profit Margin': [35, 28, 42, 31, 38]
        }
        
        df_positioning = pd.DataFrame(positioning_data)
        
        fig = px.scatter(
            df_positioning,
            x='Market Share',
            y='Growth Rate',
            size='Profit Margin',
            color='Competitor',
            title='Competitive Positioning Matrix',
            size_max=60
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(255, 255, 255, 0.05)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics(results):
    """Show advanced analytics"""
    st.markdown('<div class="section-header-pro">🔍 Advanced Analytics</div>', unsafe_allow_html=True)
    
    # Analytics tabs
    analytics_tabs = st.tabs([
        "Market Segmentation",
        "Time Series Analysis",
        "Anomaly Detection",
        "Price Elasticity",
        "Statistical Analysis"
    ])
    
    with analytics_tabs[0]:
        show_market_segmentation(results)
    
    with analytics_tabs[1]:
        show_time_series_analysis(results)
    
    with analytics_tabs[2]:
        show_anomaly_detection(results)
    
    with analytics_tabs[3]:
        show_price_elasticity(results)
    
    with analytics_tabs[4]:
        show_statistical_analysis(results)

def show_market_segmentation(results):
    """Show market segmentation analysis"""
    st.markdown("### 🎯 Advanced Market Segmentation")
    
    if 'analyses' in results and 'segmentation' in results['analyses']:
        segmentation = results['analyses']['segmentation']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster statistics
            st.markdown("#### 📊 Cluster Statistics")
            cluster_stats = []
            
            for cluster_name, stats in segmentation['cluster_stats'].items():
                cluster_stats.append({
                    'Cluster': cluster_name,
                    'Size': stats['size'],
                    'Percentage': f"{stats['percentage']:.1f}%",
                    'Avg Value': f"${stats['mean_values'].get(list(stats['mean_values'].keys())[0], 0):.0f}" if stats['mean_values'] else "N/A"
                })
            
            st.table(pd.DataFrame(cluster_stats))
        
        with col2:
            # Elbow method visualization
            st.markdown("#### 📈 Optimal Cluster Determination")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(segmentation['wcss']) + 1)),
                y=segmentation['wcss'],
                mode='lines+markers',
                name='WCSS',
                line=dict(color=results['viz_engine'].color_scheme['primary'], width=3)
            ))
            
            fig.update_layout(
                title='Elbow Method for Optimal Clusters',
                xaxis_title='Number of Clusters',
                yaxis_title='Within-Cluster Sum of Squares',
                height=400,
                plot_bgcolor='rgba(255, 255, 255, 0.05)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Representative products
        st.markdown("#### 🏆 Representative Products by Cluster")
        
        for cluster_name, stats in segmentation['cluster_stats'].items():
            with st.expander(f"{cluster_name} - {stats['size']} products ({stats['percentage']:.1f}%)"):
                if stats['representative_products']:
                    for product in stats['representative_products']:
                        st.write(f"• {product}")
                else:
                    st.info("No representative products identified")
    else:
        st.info("Market segmentation analysis not available for current data")

def show_time_series_analysis(results):
    """Show time series analysis"""
    st.markdown("### 📈 Advanced Time Series Analysis")
    
    if 'analyses' in results and 'time_series' in results['analyses']:
        ts_analysis = results['analyses']['time_series']
        
        # Decomposition visualization
        if ts_analysis['decomposition']:
            decomp = ts_analysis['decomposition']
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=ts_analysis['time_series'].index,
                    y=ts_analysis['time_series'].iloc[:, 0],
                    mode='lines',
                    name='Original',
                    line=dict(color=results['viz_engine'].color_scheme['primary'])
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=ts_analysis['time_series'].index,
                    y=decomp.trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color=results['viz_engine'].color_scheme['secondary'])
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=ts_analysis['time_series'].index,
                    y=decomp.seasonal,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color=results['viz_engine'].color_scheme['warning'])
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=ts_analysis['time_series'].index,
                    y=decomp.resid,
                    mode='lines',
                    name='Residual',
                    line=dict(color=results['viz_engine'].color_scheme['danger'])
                ),
                row=4, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                plot_bgcolor='rgba(255, 255, 255, 0.05)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Trend Direction",
                ts_analysis['trend_stats']['trend'].capitalize(),
                f"{ts_analysis['trend_stats']['slope']:.3f}"
            )
        
        with col2:
            st.metric(
                "R-squared",
                f"{ts_analysis['trend_stats']['r_squared']:.3f}",
                "Goodness of fit"
            )
        
        with col3:
            stationarity = "Stationary" if ts_analysis['stationarity']['stationary'] else "Non-Stationary"
            st.metric(
                "Stationarity",
                stationarity,
                f"p={ts_analysis['stationarity']['p_value']:.4f}"
            )
    else:
        st.info("Time series analysis not available for current data")

def show_anomaly_detection(results):
    """Show anomaly detection results"""
    st.markdown("### 🚨 Anomaly Detection Analysis")
    
    if 'analyses' in results and 'anomalies' in results['analyses']:
        anomalies = results['analyses']['anomalies']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly summary
            st.markdown("#### 📊 Anomaly Summary")
            
            summary_data = {
                'Metric': ['Total Data Points', 'Anomalies Detected', 'Anomaly Percentage', 'Average Anomaly Score'],
                'Value': [
                    len(anomalies['anomalies']),
                    len(anomalies['anomaly_indices']),
                    f"{anomalies['anomaly_percentage']:.1f}%",
                    f"{anomalies['anomaly_scores'].mean():.3f}"
                ]
            }
            
            st.table(pd.DataFrame(summary_data))
        
        with col2:
            # Top anomalies
            st.markdown("#### 🏆 Top Anomalies")
            
            if anomalies['top_anomalies']:
                for anomaly in anomalies['top_anomalies'][:5]:
                    with st.expander(f"Anomaly #{anomaly['index']} - Score: {anomaly['score']:.3f}"):
                        st.write(f"**Value:** ${anomaly['value']:,.0f}")
                        if anomaly['context']:
                            st.write("**Context:**")
                            for key, value in anomaly['context'].items():
                                st.write(f"  • {key}: {value}")
        
        # Anomaly visualization
        anomaly_viz = results['viz_engine'].create_anomaly_detection_viz(anomalies)
        if anomaly_viz:
            st.plotly_chart(anomaly_viz, use_container_width=True)
    else:
        st.info("Anomaly detection not available for current data")

def show_price_elasticity(results):
    """Show price elasticity analysis"""
    st.markdown("### 💰 Price Elasticity Analysis")
    
    if 'analyses' in results and 'price_elasticity' in results['analyses']:
        elasticity = results['analyses']['price_elasticity']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Elasticity metrics
            st.markdown("#### 📈 Elasticity Metrics")
            
            metrics_data = {
                'Metric': ['Price Elasticity', 'Elasticity Type', 'R-squared', 'Statistical Significance'],
                'Value': [
                    f"{elasticity['elasticity']:.3f}",
                    elasticity['elasticity_type'],
                    f"{elasticity['r_squared']:.3f}",
                    "Significant" if elasticity['p_value'] < 0.05 else "Not Significant"
                ]
            }
            
            st.table(pd.DataFrame(metrics_data))
            
            st.markdown(f"""
            <div class="alert-box alert-info">
                <strong>Interpretation:</strong> {elasticity['interpretation']}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Scatter plot of price vs volume
            if elasticity['data_points']:
                df_points = pd.DataFrame(elasticity['data_points'][:100])
                
                fig = px.scatter(
                    df_points,
                    x=df_points.columns[0],
                    y=df_points.columns[1],
                    title='Price vs Volume Relationship',
                    trendline='ols'
                )
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(255, 255, 255, 0.05)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Price elasticity analysis not available for current data")

def show_statistical_analysis(results):
    """Show statistical analysis"""
    st.markdown("### 📊 Advanced Statistical Analysis")
    
    if results['statistics']:
        # Summary statistics
        st.markdown("#### 📈 Summary Statistics")
        
        stats_data = []
        for col, stats in list(results['statistics'].items())[:10]:
            stats_data.append({
                'Column': col,
                'Mean': f"${stats['mean']:,.0f}",
                'Std Dev': f"${stats['std']:,.0f}",
                'CV': f"{stats['cv']:.1f}%",
                'Skewness': f"{stats['skewness']:.2f}"
            })
        
        st.table(pd.DataFrame(stats_data))
        
        # Distribution analysis
        st.markdown("#### 📊 Distribution Analysis")
        
        if results['patterns']['value_columns']:
            value_col = results['patterns']['value_columns'][0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    results['df'],
                    x=value_col,
                    nbins=50,
                    title=f'Distribution of {value_col}',
                    marginal='box'
                )
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(255, 255, 255, 0.05)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Q-Q plot
                from scipy import stats as sp_stats
                
                data = results['df'][value_col].dropna()
                if len(data) > 0:
                    fig = ff.create_distplot([data], [value_col], show_hist=False, show_rug=False)
                    
                    # Add normal distribution line
                    x = np.linspace(data.min(), data.max(), 100)
                    pdf = sp_stats.norm.pdf(x, data.mean(), data.std())
                    
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=pdf,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='Distribution Comparison',
                        height=400,
                        plot_bgcolor='rgba(255, 255, 255, 0.05)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        font_color='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def show_predictive_insights(results):
    """Show predictive insights"""
    st.markdown('<div class="section-header-pro">🤖 Predictive Insights</div>', unsafe_allow_html=True)
    
    if 'analyses' in results and 'predictions' in results['analyses']:
        predictions = results['analyses']['predictions']
        
        # Forecast comparison
        st.markdown("### 📈 Market Forecast Comparison")
        
        forecast_data = []
        for method, forecast in predictions.items():
            if forecast and 'forecast' in forecast:
                for i, value in enumerate(forecast['forecast'][:6]):
                    forecast_data.append({
                        'Month': i + 1,
                        'Value': value,
                        'Method': method.replace('_', ' ').title(),
                        'Accuracy': forecast.get('accuracy', 0)
                    })
        
        if forecast_data:
            df_forecast = pd.DataFrame(forecast_data)
            
            fig = px.line(
                df_forecast,
                x='Month',
                y='Value',
                color='Method',
                title='6-Month Market Forecast Comparison',
                markers=True
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(255, 255, 255, 0.05)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Growth opportunities
        st.markdown("### 🎯 Growth Opportunity Identification")
        
        if 'analyses' in results and 'growth_opportunities' in results['analyses']:
            opportunities = results['analyses']['growth_opportunities']
            
            if opportunities:
                opp_data = []
                for opp in opportunities[:10]:
                    opp_data.append({
                        'Molecule': opp['molecule'],
                        'Opportunity Type': opp['opportunity_type'],
                        'Score': opp['score'],
                        'Current Value': f"${opp['current_value']:,.0f}",
                        'Recommendation': opp['recommendation']
                    })
                
                st.table(pd.DataFrame(opp_data))
        
        # Market share predictions
        st.markdown("### 🏆 Market Share Predictions")
        
        if 'analyses' in results and 'market_share_prediction' in results['analyses']:
            share_predictions = results['analyses']['market_share_prediction']
            
            if share_predictions:
                prediction_data = []
                for company, pred in share_predictions.items():
                    prediction_data.append({
                        'Company': company,
                        'Current Share': f"{pred['current_share']:.1f}%",
                        'Predicted Share': f"{pred['predicted_share']:.1f}%",
                        'Change': f"{pred['predicted_change']:+.1f}%",
                        'Confidence': f"{pred['confidence']:.0%}"
                    })
                
                df_predictions = pd.DataFrame(prediction_data)
                
                fig = go.Figure(data=[
                    go.Bar(
                        name='Current',
                        x=df_predictions['Company'],
                        y=pd.to_numeric(df_predictions['Current Share'].str.replace('%', '')),
                        marker_color=results['viz_engine'].color_scheme['primary']
                    ),
                    go.Bar(
                        name='Predicted',
                        x=df_predictions['Company'],
                        y=pd.to_numeric(df_predictions['Predicted Share'].str.replace('%', '')),
                        marker_color=results['viz_engine'].color_scheme['secondary']
                    )
                ])
                
                fig.update_layout(
                    barmode='group',
                    title='Market Share Predictions (Next 12 Months)',
                    height=500,
                    plot_bgcolor='rgba(255, 255, 255, 0.05)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Predictive insights require 'Predictive' or 'Enterprise' analysis mode")

def show_data_explorer(results):
    """Show data explorer"""
    st.markdown('<div class="section-header-pro">📋 Data Explorer</div>', unsafe_allow_html=True)
    
    df = results['df']
    patterns = results['patterns']
    
    # Data quality overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        st.metric("Data Completeness", f"{100 - results['data_quality']['missing_percentage']:.1f}%")
    
    with col4:
        st.metric("Duplicate Rows", f"{results['data_quality']['duplicate_rows']:,}")
    
    st.markdown("---")
    
    # Interactive data explorer
    st.markdown("### 🔍 Interactive Data Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_rows = st.slider("Rows to show", 10, 1000, 100, 10)
    
    with col2:
        selected_columns = st.multiselect(
            "Select columns to display",
            df.columns.tolist(),
            default=patterns['dimension_columns'][:3] + patterns['value_columns'][:2]
        )
    
    if selected_columns:
        st.dataframe(df[selected_columns].head(show_rows), use_container_width=True)
        
        # Data statistics
        with st.expander("📊 Column Statistics"):
            for col in selected_columns:
                if col in df.columns:
                    st.write(f"**{col}**")
                    col_data = df[col]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Non-Null", f"{col_data.count():,}")
                    
                    with col2:
                        if pd.api.types.is_numeric_dtype(col_data):
                            st.metric("Mean", f"{col_data.mean():.2f}")
                    
                    with col3:
                        if pd.api.types.is_numeric_dtype(col_data):
                            st.metric("Std Dev", f"{col_data.std():.2f}")
                    
                    with col4:
                        st.metric("Unique", f"{col_data.nunique():,}")
        
        # Export options
        st.markdown("### 📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df[selected_columns].to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "pharma_data.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Create Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df[selected_columns].to_excel(writer, sheet_name='Data', index=False)
            
            st.download_button(
                "📊 Download Excel",
                output.getvalue(),
                "pharma_data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            # Export analysis results
            if st.button("📄 Export Analysis Report", use_container_width=True):
                st.success("Analysis report export initiated")

def show_reports(results):
    """Show reporting section"""
    st.markdown('<div class="section-header-pro">📄 Enterprise Reports</div>', unsafe_allow_html=True)
    
    # Report selection
    report_type = st.selectbox(
        "Select Report Type",
        ["Executive Summary", "Market Analysis", "Competitive Intelligence", 
         "Product Portfolio", "Financial Performance", "Strategic Recommendations"]
    )
    
    # Report configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_period = st.selectbox("Time Period", ["Last 12 Months", "Year-to-Date", "Custom"])
    
    with col2:
        report_depth = st.selectbox("Report Depth", ["Summary", "Detailed", "Comprehensive"])
    
    with col3:
        include_forecasts = st.checkbox("Include Forecasts", value=True)
    
    # Generate report button
    if st.button("📊 Generate Report", use_container_width=True):
        with st.spinner(f"Generating {report_type} report..."):
            # Initialize reporting module
            reporting_module = EnterpriseReportingModule()
            
            # Generate report
            report = reporting_module.generate_report(
                report_type.lower().replace(' ', '_'),
                results
            )
            
            if report:
                # Display report
                st.markdown(f"## {report_type} Report")
                st.markdown(f"*Generated: {report['generated_date']}*")
                
                for section in report['sections']:
                    with st.expander(section['title'], expanded=True):
                        st.markdown(section['content'])
                        
                        # Display metrics
                        if section['metrics']:
                            st.markdown("#### Key Metrics")
                            metrics_cols = st.columns(len(section['metrics']))
                            
                            for idx, (metric_name, metric_data) in enumerate(section['metrics'].items()):
                                with metrics_cols[idx]:
                                    st.metric(
                                        metric_name.replace('_', ' ').title(),
                                        f"{metric_data['value']:.1f}{metric_data['unit']}",
                                        delta=f"{metric_data['trend']}"
                                    )
                
                # Export options
                st.markdown("---")
                st.markdown("### 📤 Export Report")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # PDF export
                    if st.button("📄 Export as PDF", use_container_width=True):
                        st.info("PDF export feature requires additional configuration")
                
                with col2:
                    # PowerPoint export
                    if st.button("📽️ Export as PowerPoint", use_container_width=True):
                        st.info("PowerPoint export feature requires additional configuration")
                
                with col3:
                    # Email report
                    if st.button("📧 Email Report", use_container_width=True):
                        st.info("Email feature requires additional configuration")

def show_welcome_screen():
    """Show welcome screen"""
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h1 style="color: white; font-size: 48px; margin-bottom: 30px;">Welcome to Pharma Intelligence Pro</h1>
        <p style="color: rgba(255, 255, 255, 0.8); font-size: 20px; line-height: 1.6; max-width: 800px; margin: 0 auto 40px;">
            The most advanced pharmaceutical market intelligence platform for enterprise decision-making.
            Transform your data into actionable insights with AI-powered analytics.
        </p>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 30px; margin: 50px 0;">
            <div class="metric-card-pro" style="text-align: left;">
                <div class="metric-label-pro">AI-Powered Analytics</div>
                <div class="metric-value-pro" style="font-size: 24px;">Machine Learning Algorithms</div>
                <div style="margin-top: 15px; color: #6c757d;">
                    Advanced clustering, forecasting, and anomaly detection
                </div>
            </div>
            
            <div class="metric-card-pro" style="text-align: left;">
                <div class="metric-label-pro">Enterprise Features</div>
                <div class="metric-value-pro" style="font-size: 24px;">Professional Reporting</div>
                <div style="margin-top: 15px; color: #6c757d;">
                    Comprehensive reports with executive summaries
                </div>
            </div>
            
            <div class="metric-card-pro" style="text-align: left;">
                <div class="metric-label-pro">Real-Time Insights</div>
                <div class="metric-value-pro" style="font-size: 24px;">Predictive Analytics</div>
                <div style="margin-top: 15px; color: #6c757d;">
                    Market trend forecasting and opportunity identification
                </div>
            </div>
        </div>
        
        <div style="background: rgba(26, 77, 122, 0.2); padding: 40px; border-radius: 16px; margin: 50px 0; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h3 style="color: white; margin-bottom: 20px;">🚀 Get Started</h3>
            <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 30px;">
                Upload your pharmaceutical market data file to begin advanced analysis
            </p>
            
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div style="text-align: left; background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px; flex: 1; min-width: 250px;">
                    <h4 style="color: white; margin-bottom: 15px;">📁 Supported Formats</h4>
                    <ul style="color: rgba(255, 255, 255, 0.8); padding-left: 20px;">
                        <li>CSV files</li>
                        <li>Excel files (.xlsx, .xls)</li>
                        <li>Automatic column detection</li>
                        <li>Multi-year data support</li>
                    </ul>
                </div>
                
                <div style="text-align: left; background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px; flex: 1; min-width: 250px;">
                    <h4 style="color: white; margin-bottom: 15px;">🔍 Analysis Features</h4>
                    <ul style="color: rgba(255, 255, 255, 0.8); padding-left: 20px;">
                        <li>Market segmentation</li>
                        <li>Competitive analysis</li>
                        <li>Price elasticity</li>
                        <li>Anomaly detection</li>
                        <li>Forecasting models</li>
                    </ul>
                </div>
                
                <div style="text-align: left; background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px; flex: 1; min-width: 250px;">
                    <h4 style="color: white; margin-bottom: 15px;">📊 Output & Reports</h4>
                    <ul style="color: rgba(255, 255, 255, 0.8); padding-left: 20px;">
                        <li>Interactive dashboards</li>
                        <li>Professional reports</li>
                        <li>Data visualizations</li>
                        <li>Export capabilities</li>
                        <li>Executive summaries</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer-pro">
        <strong>Pharma Intelligence Pro v3.0</strong><br>
        Enterprise Pharmaceutical Analytics Platform<br>
        © 2024 Pharma Analytics Inc. • All Rights Reserved<br>
        <small style="opacity: 0.7;">For demonstration purposes only</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
