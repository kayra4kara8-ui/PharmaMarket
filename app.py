import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
from datetime import datetime, timedelta
from io import BytesIO
import re
import json
import base64
import math
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
# app.py dosyasının başındaki import bölümünü şu şekilde değiştirin:
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available. Some features will be disabled.")from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Pharma Analytics Intelligence Platform - Enterprise",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.pharmaintelligence.com',
        'Report a bug': 'https://www.pharmaintelligence.com/bug',
        'About': '# Enterprise Pharma Analytics Platform v3.0'
    }
)

# ============================================================================
# CUSTOM CSS - PROFESSIONAL ENTERPRISE STYLING
# ============================================================================
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0c2d4d 0%, #1a5a8a 50%, #0c2d4d 100%);
        padding: 30px;
        border-radius: 0 0 20px 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00c9ff 0%, #92fe9d 100%);
    }
    
    .main-header h1 {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        background: linear-gradient(90deg, #ffffff 0%, #a8edea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: #e3f2fd;
        font-size: 16px;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    .metric-super-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(15, 40, 71, 0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    
    .metric-super-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(15, 40, 71, 0.15);
        border-color: #cbd5e1;
    }
    
    .metric-super-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(180deg, #1a4d7a 0%, #0f2847 100%);
    }
    
    .metric-value-xl {
        font-size: 42px;
        font-weight: 800;
        color: #1e3a8a;
        margin: 15px 0;
        line-height: 1;
    }
    
    .metric-label-xl {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }
    
    .metric-trend {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 15px;
        font-weight: 600;
        margin-top: 12px;
    }
    
    .trend-up {
        color: #10b981;
    }
    
    .trend-down {
        color: #ef4444;
    }
    
    .trend-neutral {
        color: #94a3b8;
    }
    
    .dashboard-section {
        background: white;
        border-radius: 18px;
        padding: 30px;
        margin: 25px 0;
        box-shadow: 0 10px 25px rgba(15, 40, 71, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 2px solid #e2e8f0;
        position: relative;
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 80px;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 15px 30px rgba(30, 58, 138, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .insight-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 150px;
        height: 150px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        transform: translate(50%, -50%);
    }
    
    .insight-card-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 15px;
        color: white;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .insight-card-content {
        font-size: 16px;
        line-height: 1.7;
        color: #e2e8f0;
    }
    
    .tab-styled {
        background: white !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 18px 35px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
        border-bottom: none !important;
        margin: 0 5px !important;
        transition: all 0.3s ease !important;
    }
    
    .tab-styled[aria-selected="true"] {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%) !important;
        color: white !important;
        border-color: #1e40af !important;
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.2) !important;
    }
    
    .sidebar-gradient {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-title {
        color: white;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .filter-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .stat-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 3px;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .badge-info {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
    }
    
    .progress-container {
        background: #f1f5f9;
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease;
    }
    
    .progress-success {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    }
    
    .progress-warning {
        background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%);
    }
    
    .progress-danger {
        background: linear-gradient(90deg, #ef4444 0%, #f87171 100%);
    }
    
    .data-table-container {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin: 20px 0;
    }
    
    .download-button {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.2);
    }
    
    .download-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(30, 64, 175, 0.3);
        background: linear-gradient(135deg, #1d4ed8 0%, #1e3a8a 100%);
    }
    
    .hover-card {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .hover-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
    }
    
    .animated-border {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
    }
    
    .animated-border::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .sparkline-container {
        padding: 15px;
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .confidence-interval {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0ea5e9;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .risk-matrix-cell {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .footer-enterprise {
        text-align: center;
        padding: 40px;
        color: #64748b;
        font-size: 14px;
        margin-top: 60px;
        border-top: 1px solid #e2e8f0;
        background: white;
        border-radius: 20px 20px 0 0;
        box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.05);
    }
    
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    
    .loading-spinner::after {
        content: '';
        width: 50px;
        height: 50px;
        border: 5px solid #e2e8f0;
        border-top-color: #3b82f6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Advanced Data Grid Styling */
    .ag-theme-alpine {
        --ag-border-radius: 8px;
        --ag-border-color: #e2e8f0;
        --ag-header-background-color: #f8fafc;
        --ag-odd-row-background-color: #f8fafc;
    }
    
    /* Alert Styling */
    .alert-box {
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left-color: #10b981;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left-color: #f59e0b;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left-color: #ef4444;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left-color: #3b82f6;
    }
    
    /* KPI Circle */
    .kpi-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin: 0 auto;
        position: relative;
    }
    
    .kpi-circle-inner {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        background: white;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    /* Waterfall Chart Styling */
    .waterfall-bar {
        transition: all 0.3s ease;
    }
    
    .waterfall-bar:hover {
        opacity: 0.8;
    }
    
    /* Network Graph Styling */
    .node-circle {
        stroke: white;
        stroke-width: 2;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .metric-value-xl {
            font-size: 32px;
        }
        
        .dashboard-section {
            padding: 20px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA ENGINE - ENHANCED DATA PROCESSING
# ============================================================================
class PharmaDataEngine:
    def __init__(self):
        self.df = None
        self.column_patterns = {}
        self.data_quality_metrics = {}
        self.cleaning_log = []
        
    def load_data(self, uploaded_file):
        """Load and validate pharmaceutical data"""
        try:
            file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
            st.info(f"📁 Loading file: {uploaded_file.name} ({file_size:.2f} MB)")
            
            if uploaded_file.name.endswith('.csv'):
                # Try multiple encodings for CSV
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=False)
                        st.success(f"✓ Data loaded with {encoding} encoding")
                        break
                    except:
                        continue
                else:
                    st.error("Could not read CSV with any encoding. Please check file format.")
                    return None
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                
            self.df = df
            self._analyze_data_quality()
            self._detect_column_patterns()
            self._clean_data()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Data loading failed: {str(e)}")
            return None
    
    def _analyze_data_quality(self):
        """Comprehensive data quality analysis"""
        self.data_quality_metrics = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.astype(str).to_dict(),
            'column_stats': {}
        }
        
        for col in self.df.columns:
            self.data_quality_metrics['column_stats'][col] = {
                'missing': self.df[col].isnull().sum(),
                'missing_pct': (self.df[col].isnull().sum() / len(self.df)) * 100,
                'unique': self.df[col].nunique(),
                'dtype': str(self.df[col].dtype)
            }
    
    def _detect_column_patterns(self):
        """Advanced column pattern detection with ML-based classification"""
        self.column_patterns = {
            'value_columns': [],
            'volume_columns': [],
            'unit_columns': [],
            'price_columns': [],
            'unit_price_columns': [],
            'dimension_columns': [],
            'date_columns': [],
            'categorical_columns': [],
            'geography_columns': [],
            'product_columns': [],
            'manufacturer_columns': [],
            'therapy_columns': [],
            'time_series_columns': []
        }
        
        col_classification_rules = {
            'value': ['usd', 'value', 'sales', 'revenue', 'mnf', '$', 'amount', 'income', 'turnover'],
            'volume': ['standard', 'unit', 'volume', 'su', 'quantity', 'qty', 'pack', 'count'],
            'price': ['price', 'avg', 'average', 'cost', 'rate'],
            'dimension': ['country', 'region', 'city', 'area', 'zone', 'territory'],
            'product': ['product', 'molecule', 'drug', 'brand', 'formula', 'compound'],
            'manufacturer': ['manufacturer', 'company', 'corp', 'inc', 'ltd', 'laboratory', 'pharma'],
            'therapy': ['therapy', 'specialty', 'indication', 'disease', 'therapy_area'],
            'date': ['date', 'year', 'month', 'quarter', 'period', 'week']
        }
        
        for col in self.df.columns:
            col_lower = str(col).lower()
            
            # Check each category
            for category, keywords in col_classification_rules.items():
                if any(keyword in col_lower for keyword in keywords):
                    if category == 'value' and 'price' not in col_lower:
                        self.column_patterns['value_columns'].append(col)
                    elif category == 'volume' and 'price' not in col_lower:
                        if 'standard' in col_lower or 'su' in col_lower:
                            self.column_patterns['volume_columns'].append(col)
                        else:
                            self.column_patterns['unit_columns'].append(col)
                    elif category == 'price':
                        if 'standard' in col_lower or 'su' in col_lower:
                            self.column_patterns['price_columns'].append(col)
                        else:
                            self.column_patterns['unit_price_columns'].append(col)
                    elif category == 'dimension':
                        self.column_patterns['dimension_columns'].append(col)
                        if any(geo in col_lower for geo in ['country', 'region', 'city']):
                            self.column_patterns['geography_columns'].append(col)
                    elif category == 'product':
                        self.column_patterns['product_columns'].append(col)
                    elif category == 'manufacturer':
                        self.column_patterns['manufacturer_columns'].append(col)
                    elif category == 'therapy':
                        self.column_patterns['therapy_columns'].append(col)
                    elif category == 'date':
                        self.column_patterns['date_columns'].append(col)
            
            # Detect categorical columns
            if self.df[col].nunique() <= 50 and self.df[col].dtype == 'object':
                self.column_patterns['categorical_columns'].append(col)
            
            # Detect time series columns (year patterns)
            if re.search(r'\b(20\d{2})\b', col):
                self.column_patterns['time_series_columns'].append(col)
    
    def _clean_data(self):
        """Advanced data cleaning with comprehensive error handling"""
        cleaning_operations = []
        
        # Clean numeric columns
        numeric_columns = (self.column_patterns['value_columns'] + 
                          self.column_patterns['volume_columns'] + 
                          self.column_patterns['unit_columns'] + 
                          self.column_patterns['price_columns'] + 
                          self.column_patterns['unit_price_columns'])
        
        for col in numeric_columns:
            if col in self.df.columns:
                original_dtype = self.df[col].dtype
                original_nulls = self.df[col].isnull().sum()
                
                # Comprehensive cleaning
                self.df[col] = self._clean_numeric_series(self.df[col])
                
                cleaning_operations.append({
                    'column': col,
                    'original_dtype': str(original_dtype),
                    'final_dtype': str(self.df[col].dtype),
                    'nulls_removed': original_nulls - self.df[col].isnull().sum(),
                    'cleaning_applied': True
                })
        
        # Clean categorical columns
        for col in self.column_patterns['dimension_columns']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown').astype(str).str.strip()
                # Standardize case
                self.df[col] = self.df[col].str.title()
        
        self.cleaning_log = cleaning_operations
    
    def _clean_numeric_series(self, series):
        """Advanced numeric series cleaning"""
        if series.dtype == 'object':
            series = series.astype(str)
            
            # Remove common non-numeric characters
            replacements = [
                (',', '.'),
                (' ', ''),
                ('$', ''),
                ('USD', ''),
                ('MNF', ''),
                ('€', ''),
                ('£', ''),
                ('¥', ''),
                ('K', '000'),
                ('M', '000000'),
                ('B', '000000000'),
                ('%', ''),
                ('(', '-'),
                (')', ''),
                ('[', ''),
                (']', '')
            ]
            
            for old, new in replacements:
                series = series.str.replace(old, new, regex=False)
            
            # Handle negative numbers in parentheses
            series = series.replace(r'\((\d+\.?\d*)\)', r'-\1', regex=True)
            
            # Convert to numeric
            series = pd.to_numeric(series, errors='coerce')
        
        # Handle outliers using IQR method
        if series.dtype in ['float64', 'int64']:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing
            series = series.clip(lower_bound, upper_bound)
        
        return series.fillna(0)
    
    def extract_years(self):
        """Extract years from column names"""
        years = set()
        for col in self.df.columns:
            matches = re.findall(r'\b(20\d{2})\b', str(col))
            years.update(matches)
        
        return sorted(list(years), reverse=True)
    
    def get_column_for_year(self, column_type, year):
        """Get specific column for a given year and type"""
        if column_type == 'value':
            candidates = self.column_patterns['value_columns']
        elif column_type == 'volume':
            candidates = self.column_patterns['volume_columns']
        elif column_type == 'price':
            candidates = self.column_patterns['price_columns']
        elif column_type == 'unit_price':
            candidates = self.column_patterns['unit_price_columns']
        elif column_type == 'units':
            candidates = self.column_patterns['unit_columns']
        else:
            return None
        
        # First try exact year match
        year_pattern = str(year)
        for col in candidates:
            if year_pattern in str(col):
                return col
        
        # Try fuzzy matching
        for col in candidates:
            if re.search(r'\b' + year_pattern[-2:] + r'\b', str(col)):
                return col
        
        # Return first candidate if no year match
        return candidates[0] if candidates else None
    
    def get_manufacturer_column(self):
        """Get manufacturer column"""
        if self.column_patterns['manufacturer_columns']:
            return self.column_patterns['manufacturer_columns'][0]
        
        # Fallback: look for manufacturer in dimension columns
        for col in self.column_patterns['dimension_columns']:
            if 'manufacturer' in str(col).lower() or 'company' in str(col).lower():
                return col
        
        return None
    
    def get_molecule_column(self):
        """Get molecule column"""
        if self.column_patterns['product_columns']:
            return self.column_patterns['product_columns'][0]
        
        for col in self.column_patterns['dimension_columns']:
            if 'molecule' in str(col).lower() or 'product' in str(col).lower():
                return col
        
        return None

# ============================================================================
# ADVANCED ANALYTICS ENGINE
# ============================================================================
class AdvancedAnalyticsEngine:
    def __init__(self, data_engine):
        self.data_engine = data_engine
        self.df = data_engine.df
        self.column_patterns = data_engine.column_patterns
        
    def calculate_market_metrics(self, year):
        """Calculate comprehensive market metrics"""
        value_col = self.data_engine.get_column_for_year('value', year)
        volume_col = self.data_engine.get_column_for_year('volume', year)
        
        if not value_col or not volume_col:
            return None
        
        total_value = self.df[value_col].sum()
        total_volume = self.df[volume_col].sum()
        
        metrics = {
            'total_value': total_value,
            'total_volume': total_volume,
            'average_price': total_value / total_volume if total_volume > 0 else 0,
            'value_millions': total_value / 1e6,
            'volume_thousands': total_volume / 1e3,
            'row_count': len(self.df),
            'non_zero_rows': (self.df[value_col] > 0).sum()
        }
        
        return metrics
    
    def calculate_growth_decomposition(self, current_year, previous_year):
        """Advanced growth decomposition analysis"""
        value_curr = self.data_engine.get_column_for_year('value', current_year)
        value_prev = self.data_engine.get_column_for_year('value', previous_year)
        volume_curr = self.data_engine.get_column_for_year('volume', current_year)
        volume_prev = self.data_engine.get_column_for_year('volume', previous_year)
        
        if not all([value_curr, value_prev, volume_curr, volume_prev]):
            return None
        
        # Calculate aggregate metrics
        total_value_curr = self.df[value_curr].sum()
        total_value_prev = self.df[value_prev].sum()
        total_volume_curr = self.df[volume_curr].sum()
        total_volume_prev = self.df[volume_prev].sum()
        
        # Price calculations
        avg_price_curr = total_value_curr / total_volume_curr if total_volume_curr > 0 else 0
        avg_price_prev = total_value_prev / total_volume_prev if total_volume_prev > 0 else 0
        
        # Growth rates
        value_growth = ((total_value_curr - total_value_prev) / total_value_prev * 100) if total_value_prev > 0 else 0
        volume_growth = ((total_volume_curr - total_volume_prev) / total_volume_prev * 100) if total_volume_prev > 0 else 0
        price_growth = ((avg_price_curr - avg_price_prev) / avg_price_prev * 100) if avg_price_prev > 0 else 0
        
        # Advanced decomposition using Laspeyres index
        volume_effect = (total_volume_curr - total_volume_prev) * avg_price_prev
        price_effect = (avg_price_curr - avg_price_prev) * total_volume_curr
        total_change = total_value_curr - total_value_prev
        
        # Residual effect (mix + other)
        residual_effect = total_change - volume_effect - price_effect
        
        # Contribution percentages
        if total_change != 0:
            volume_effect_pct = (volume_effect / total_change) * 100
            price_effect_pct = (price_effect / total_change) * 100
            residual_effect_pct = (residual_effect / total_change) * 100
        else:
            volume_effect_pct = price_effect_pct = residual_effect_pct = 0
        
        # Detailed product-level decomposition
        product_decomposition = self._calculate_product_level_decomposition(
            value_curr, value_prev, volume_curr, volume_prev
        )
        
        return {
            'value_growth': value_growth,
            'volume_growth': volume_growth,
            'price_growth': price_growth,
            'volume_effect': volume_effect,
            'price_effect': price_effect,
            'residual_effect': residual_effect,
            'volume_effect_pct': volume_effect_pct,
            'price_effect_pct': price_effect_pct,
            'residual_effect_pct': residual_effect_pct,
            'total_change': total_change,
            'avg_price_current': avg_price_curr,
            'avg_price_previous': avg_price_prev,
            'product_decomposition': product_decomposition
        }
    
    def _calculate_product_level_decomposition(self, value_curr, value_prev, volume_curr, volume_prev):
        """Product-level growth decomposition"""
        if not all([value_curr, value_prev, volume_curr, volume_prev]):
            return None
        
        # Calculate product-level contributions
        product_col = self.data_engine.get_molecule_column()
        if not product_col:
            return None
        
        decomposition_data = []
        
        for product in self.df[product_col].unique():
            product_mask = self.df[product_col] == product
            product_df = self.df[product_mask]
            
            if len(product_df) == 0:
                continue
            
            value_curr_product = product_df[value_curr].sum()
            value_prev_product = product_df[value_prev].sum()
            volume_curr_product = product_df[volume_curr].sum()
            volume_prev_product = product_df[volume_prev].sum()
            
            if value_prev_product == 0 or volume_prev_product == 0:
                continue
            
            avg_price_curr = value_curr_product / volume_curr_product if volume_curr_product > 0 else 0
            avg_price_prev = value_prev_product / volume_prev_product if volume_prev_product > 0 else 0
            
            value_growth = ((value_curr_product - value_prev_product) / value_prev_product * 100) if value_prev_product > 0 else 0
            
            volume_effect = (volume_curr_product - volume_prev_product) * avg_price_prev
            price_effect = (avg_price_curr - avg_price_prev) * volume_curr_product
            total_change = value_curr_product - value_prev_product
            residual_effect = total_change - volume_effect - price_effect
            
            decomposition_data.append({
                'product': product,
                'value_current': value_curr_product,
                'value_previous': value_prev_product,
                'value_growth': value_growth,
                'volume_effect': volume_effect,
                'price_effect': price_effect,
                'residual_effect': residual_effect,
                'contribution_to_total_growth': (value_curr_product - value_prev_product),
                'market_share_current': (value_curr_product / self.df[value_curr].sum() * 100) if self.df[value_curr].sum() > 0 else 0
            })
        
        return pd.DataFrame(decomposition_data)
    
    def calculate_market_concentration(self, year):
        """Calculate Herfindahl-Hirschman Index and concentration ratios"""
        value_col = self.data_engine.get_column_for_year('value', year)
        manufacturer_col = self.data_engine.get_manufacturer_column()
        
        if not value_col or not manufacturer_col:
            return None
        
        # Calculate market shares
        market_shares = self.df.groupby(manufacturer_col)[value_col].sum()
        total_market = market_shares.sum()
        
        if total_market == 0:
            return None
        
        market_shares_pct = (market_shares / total_market * 100).sort_values(ascending=False)
        
        # Calculate concentration ratios
        cr1 = market_shares_pct.iloc[0] if len(market_shares_pct) > 0 else 0
        cr3 = market_shares_pct.head(3).sum() if len(market_shares_pct) >= 3 else market_shares_pct.sum()
        cr5 = market_shares_pct.head(5).sum() if len(market_shares_pct) >= 5 else market_shares_pct.sum()
        cr10 = market_shares_pct.head(10).sum() if len(market_shares_pct) >= 10 else market_shares_pct.sum()
        
        # Calculate HHI
        shares_decimal = market_shares / total_market
        hhi = (shares_decimal ** 2).sum() * 10000
        
        # Calculate Lorenz curve and Gini coefficient
        lorenz_curve = self._calculate_lorenz_curve(market_shares)
        gini_coefficient = self._calculate_gini_coefficient(market_shares)
        
        # Market concentration classification
        if hhi < 1500:
            concentration_level = "Unconcentrated"
        elif hhi < 2500:
            concentration_level = "Moderately Concentrated"
        else:
            concentration_level = "Highly Concentrated"
        
        return {
            'hhi': hhi,
            'concentration_level': concentration_level,
            'cr1': cr1,
            'cr3': cr3,
            'cr5': cr5,
            'cr10': cr10,
            'total_manufacturers': len(market_shares),
            'market_shares': market_shares_pct,
            'lorenz_curve': lorenz_curve,
            'gini_coefficient': gini_coefficient
        }
    
    def _calculate_lorenz_curve(self, market_shares):
        """Calculate Lorenz curve data"""
        sorted_shares = np.sort(market_shares.values)
        cumulative_shares = np.cumsum(sorted_shares) / np.sum(sorted_shares)
        cumulative_population = np.arange(1, len(sorted_shares) + 1) / len(sorted_shares)
        
        return {
            'cumulative_population': cumulative_population,
            'cumulative_shares': cumulative_shares
        }
    
    def _calculate_gini_coefficient(self, market_shares):
        """Calculate Gini coefficient"""
        sorted_shares = np.sort(market_shares.values)
        n = len(sorted_shares)
        cumulative_sum = np.cumsum(sorted_shares)
        
        if n == 0 or np.sum(sorted_shares) == 0:
            return 0
        
        gini = (n + 1 - 2 * np.sum(cumulative_sum) / np.sum(sorted_shares)) / n
        return gini
    
    def calculate_price_elasticity(self, year_current, year_previous):
        """Calculate price elasticity of demand"""
        value_curr = self.data_engine.get_column_for_year('value', year_current)
        value_prev = self.data_engine.get_column_for_year('value', year_previous)
        volume_curr = self.data_engine.get_column_for_year('volume', year_current)
        volume_prev = self.data_engine.get_column_for_year('volume', year_previous)
        
        if not all([value_curr, value_prev, volume_curr, volume_prev]):
            return None
        
        # Calculate price and quantity changes at product level
        product_col = self.data_engine.get_molecule_column()
        if not product_col:
            return None
        
        elasticity_data = []
        
        for product in self.df[product_col].unique():
            product_mask = self.df[product_col] == product
            product_df = self.df[product_mask]
            
            if len(product_df) == 0:
                continue
            
            q1 = product_df[volume_prev].sum()  # Quantity previous
            q2 = product_df[volume_curr].sum()  # Quantity current
            rev1 = product_df[value_prev].sum()  # Revenue previous
            rev2 = product_df[value_curr].sum()  # Revenue current
            
            if q1 == 0 or q2 == 0 or rev1 == 0 or rev2 == 0:
                continue
            
            p1 = rev1 / q1  # Price previous
            p2 = rev2 / q2  # Price current
            
            # Calculate elasticity using midpoint formula
            quantity_change_pct = ((q2 - q1) / ((q1 + q2) / 2)) * 100
            price_change_pct = ((p2 - p1) / ((p1 + p2) / 2)) * 100
            
            if price_change_pct != 0:
                elasticity = quantity_change_pct / price_change_pct
            else:
                elasticity = 0
            
            elasticity_data.append({
                'product': product,
                'price_previous': p1,
                'price_current': p2,
                'quantity_previous': q1,
                'quantity_current': q2,
                'price_change_pct': price_change_pct,
                'quantity_change_pct': quantity_change_pct,
                'elasticity': elasticity,
                'elasticity_type': 'Elastic' if abs(elasticity) > 1 else 'Inelastic',
                'revenue_change': rev2 - rev1,
                'revenue_change_pct': ((rev2 - rev1) / rev1 * 100) if rev1 > 0 else 0
            })
        
        return pd.DataFrame(elasticity_data)
    
    def perform_cluster_analysis(self, year):
        """Perform K-means clustering on manufacturers"""
        value_col = self.data_engine.get_column_for_year('value', year)
        volume_col = self.data_engine.get_column_for_year('volume', year)
        manufacturer_col = self.data_engine.get_manufacturer_column()
        
        if not all([value_col, volume_col, manufacturer_col]):
            return None
        
        # Prepare data for clustering
        cluster_data = self.df.groupby(manufacturer_col).agg({
            value_col: 'sum',
            volume_col: 'sum'
        }).reset_index()
        
        cluster_data['avg_price'] = cluster_data[value_col] / cluster_data[volume_col].replace(0, np.nan)
        cluster_data = cluster_data.dropna(subset=['avg_price'])
        
        if len(cluster_data) < 3:
            return None
        
        # Features for clustering
        features = ['avg_price', volume_col]
        X = cluster_data[features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters using elbow method
        wcss = []
        for i in range(1, min(11, len(X_scaled))):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        # Choose number of clusters (simple elbow detection)
        diffs = np.diff(wcss)
        diffs_ratios = diffs[1:] / diffs[:-1]
        optimal_clusters = np.argmin(diffs_ratios) + 3 if len(diffs_ratios) > 0 else 3
        optimal_clusters = min(max(optimal_clusters, 2), 5)
        
        # Apply K-means with optimal clusters
        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        cluster_data['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in range(optimal_clusters):
            cluster_group = cluster_data[cluster_data['cluster'] == cluster_id]
            stats = {
                'cluster_id': cluster_id,
                'manufacturer_count': len(cluster_group),
                'avg_price_mean': cluster_group['avg_price'].mean(),
                'avg_price_std': cluster_group['avg_price'].std(),
                'volume_mean': cluster_group[volume_col].mean(),
                'value_mean': cluster_group[value_col].mean(),
                'manufacturers': cluster_group[manufacturer_col].tolist()
            }
            cluster_stats.append(stats)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        cluster_data['pca1'] = X_pca[:, 0]
        cluster_data['pca2'] = X_pca[:, 1]
        
        return {
            'cluster_data': cluster_data,
            'cluster_stats': cluster_stats,
            'optimal_clusters': optimal_clusters,
            'pca_variance_ratio': pca.explained_variance_ratio_,
            'features': features
        }
    
    def calculate_market_risk_score(self, years):
        """Calculate comprehensive market risk score"""
        if len(years) < 2:
            return None
        
        risk_factors = {}
        
        # 1. Market Concentration Risk
        latest_year = years[-1]
        concentration = self.calculate_market_concentration(latest_year)
        if concentration:
            hhi = concentration['hhi']
            if hhi > 2500:
                risk_factors['concentration_risk'] = {'score': 90, 'weight': 0.3}
            elif hhi > 1500:
                risk_factors['concentration_risk'] = {'score': 60, 'weight': 0.3}
            else:
                risk_factors['concentration_risk'] = {'score': 30, 'weight': 0.3}
        
        # 2. Growth Volatility Risk
        growth_rates = []
        for i in range(1, len(years)):
            decomp = self.calculate_growth_decomposition(years[i], years[i-1])
            if decomp:
                growth_rates.append(abs(decomp['value_growth']))
        
        if growth_rates:
            growth_volatility = np.std(growth_rates) if len(growth_rates) > 1 else 0
            if growth_volatility > 20:
                risk_factors['growth_volatility'] = {'score': 80, 'weight': 0.25}
            elif growth_volatility > 10:
                risk_factors['growth_volatility'] = {'score': 50, 'weight': 0.25}
            else:
                risk_factors['growth_volatility'] = {'score': 20, 'weight': 0.25}
        
        # 3. Price Sensitivity Risk
        if len(years) >= 2:
            elasticity_data = self.calculate_price_elasticity(years[-1], years[-2])
            if elasticity_data is not None and not elasticity_data.empty:
                avg_elasticity = elasticity_data['elasticity'].abs().mean()
                if avg_elasticity > 2:
                    risk_factors['price_sensitivity'] = {'score': 85, 'weight': 0.2}
                elif avg_elasticity > 1:
                    risk_factors['price_sensitivity'] = {'score': 55, 'weight': 0.2}
                else:
                    risk_factors['price_sensitivity'] = {'score': 25, 'weight': 0.2}
        
        # 4. Portfolio Dependency Risk
        value_col = self.data_engine.get_column_for_year('value', latest_year)
        product_col = self.data_engine.get_molecule_column()
        
        if value_col and product_col:
            product_values = self.df.groupby(product_col)[value_col].sum().sort_values(ascending=False)
            top3_share = product_values.head(3).sum() / product_values.sum() * 100 if product_values.sum() > 0 else 0
            
            if top3_share > 60:
                risk_factors['portfolio_dependency'] = {'score': 75, 'weight': 0.15}
            elif top3_share > 40:
                risk_factors['portfolio_dependency'] = {'score': 45, 'weight': 0.15}
            else:
                risk_factors['portfolio_dependency'] = {'score': 20, 'weight': 0.15}
        
        # 5. Market Size Risk
        market_metrics = self.calculate_market_metrics(latest_year)
        if market_metrics:
            market_size = market_metrics['total_value']
            if market_size < 1e6:
                risk_factors['market_size'] = {'score': 70, 'weight': 0.1}
            elif market_size < 10e6:
                risk_factors['market_size'] = {'score': 40, 'weight': 0.1}
            else:
                risk_factors['market_size'] = {'score': 15, 'weight': 0.1}
        
        # Calculate weighted risk score
        total_weight = sum(factor['weight'] for factor in risk_factors.values())
        if total_weight > 0:
            weighted_score = sum(factor['score'] * factor['weight'] for factor in risk_factors.values()) / total_weight
        else:
            weighted_score = 50
        
        # Risk classification
        if weighted_score >= 70:
            risk_level = "HIGH"
            risk_color = "#ef4444"
        elif weighted_score >= 40:
            risk_level = "MEDIUM"
            risk_color = "#f59e0b"
        else:
            risk_level = "LOW"
            risk_color = "#10b981"
        
        return {
            'overall_score': weighted_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_factors': risk_factors,
            'recommendations': self._generate_risk_recommendations(risk_factors, weighted_score)
        }
    
    def _generate_risk_recommendations(self, risk_factors, overall_score):
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        if overall_score >= 70:
            recommendations.append("🚨 **CRITICAL**: Implement immediate risk mitigation strategies")
            recommendations.append("📉 Diversify product portfolio to reduce dependency")
            recommendations.append("🏢 Consider strategic partnerships or M&A activities")
            recommendations.append("💰 Build financial reserves for market volatility")
        
        if 'concentration_risk' in risk_factors and risk_factors['concentration_risk']['score'] > 60:
            recommendations.append("🎯 Focus on niche markets with less competition")
            recommendations.append("🤝 Develop strategic alliances with smaller players")
        
        if 'growth_volatility' in risk_factors and risk_factors['growth_volatility']['score'] > 60:
            recommendations.append("📊 Implement advanced forecasting models")
            recommendations.append("🔄 Diversify across multiple therapy areas")
        
        if 'price_sensitivity' in risk_factors and risk_factors['price_sensitivity']['score'] > 60:
            recommendations.append("💎 Emphasize value-based pricing strategies")
            recommendations.append("🎁 Develop loyalty programs to reduce churn")
        
        if overall_score < 40:
            recommendations.append("✅ **LOW RISK**: Maintain current strategy with monitoring")
            recommendations.append("📈 Focus on growth opportunities in adjacent markets")
        
        return recommendations
    
    def forecast_market_trends(self, years, forecast_periods=3):
        """Forecast market trends using time series analysis"""
        if len(years) < 3:
            return None
        
        # Collect historical data
        historical_data = []
        for year in years:
            metrics = self.calculate_market_metrics(year)
            if metrics:
                historical_data.append({
                    'year': year,
                    'value': metrics['total_value'],
                    'volume': metrics['total_volume'],
                    'price': metrics['average_price']
                })
        
        if len(historical_data) < 3:
            return None
        
        historical_df = pd.DataFrame(historical_data)
        
        # Simple forecasting models
        forecasts = {}
        
        # Linear regression forecast for value
        X = np.array(historical_df['year']).reshape(-1, 1)
        y_value = historical_df['value'].values
        
        model_value = sm.OLS(y_value, sm.add_constant(X)).fit()
        
        # Generate forecasts
        future_years = list(range(years[-1] + 1, years[-1] + forecast_periods + 1))
        future_X = np.array(future_years).reshape(-1, 1)
        
        value_forecast = model_value.predict(sm.add_constant(future_X))
        
        # Calculate confidence intervals
        predictions = model_value.get_prediction(sm.add_constant(future_X))
        value_ci = predictions.conf_int(alpha=0.05)
        
        forecasts['value'] = {
            'years': future_years,
            'point_forecast': value_forecast,
            'lower_ci': value_ci[:, 0],
            'upper_ci': value_ci[:, 1],
            'growth_rates': [(value_forecast[i] - historical_df['value'].iloc[-1]) / historical_df['value'].iloc[-1] * 100 
                           for i in range(len(future_years))],
            'model_r2': model_value.rsquared,
            'model_pvalue': model_value.f_pvalue
        }
        
        # CAGR calculation
        cagr = ((value_forecast[-1] / historical_df['value'].iloc[0]) ** (1 / (future_years[-1] - years[0] + forecast_periods)) - 1) * 100
        
        return {
            'historical_data': historical_df,
            'forecasts': forecasts,
            'cagr': cagr,
            'forecast_periods': forecast_periods,
            'model_summary': {
                'value_model_r2': model_value.rsquared,
                'value_model_adj_r2': model_value.rsquared_adj,
                'value_model_fstat': model_value.fvalue
            }
        }

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================
class VisualizationEngine:
    @staticmethod
    def create_market_overview_dashboard(analytics_engine, year):
        """Create comprehensive market overview dashboard"""
        metrics = analytics_engine.calculate_market_metrics(year)
        
        if not metrics:
            return None
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Market Value Distribution', 'Price vs Volume Scatter',
                'Top 10 Products', 'Monthly Trend Analysis',
                'Geographic Distribution', 'Market Share Pyramid'
            ),
            specs=[
                [{'type': 'box'}, {'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'line'}, {'type': 'choropleth'}, {'type': 'funnel'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Add box plot for value distribution
        value_col = analytics_engine.data_engine.get_column_for_year('value', year)
        if value_col:
            fig.add_trace(
                go.Box(y=analytics_engine.df[value_col], name='Value Distribution', boxpoints='outliers'),
                row=1, col=1
            )
        
        # Add scatter plot for price vs volume
        volume_col = analytics_engine.data_engine.get_column_for_year('volume', year)
        if value_col and volume_col:
            fig.add_trace(
                go.Scatter(
                    x=analytics_engine.df[volume_col],
                    y=analytics_engine.df[value_col],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=analytics_engine.df[value_col] / analytics_engine.df[volume_col].replace(0, 1),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Price")
                    ),
                    name='Price-Volume'
                ),
                row=1, col=2
            )
        
        # Add top products bar chart
        product_col = analytics_engine.data_engine.get_molecule_column()
        if product_col and value_col:
            top_products = analytics_engine.df.groupby(product_col)[value_col].sum().nlargest(10)
            fig.add_trace(
                go.Bar(
                    x=top_products.values,
                    y=top_products.index,
                    orientation='h',
                    name='Top Products',
                    marker_color='#3b82f6'
                ),
                row=1, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=False,
            title_text=f"Market Overview Dashboard - {year}",
            title_font_size=24,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_growth_decomposition_chart(decomposition_data, current_year, previous_year):
        """Create waterfall chart for growth decomposition"""
        if not decomposition_data:
            return None
        
        fig = go.Figure(go.Waterfall(
            name="Growth Decomposition",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Previous Year", "Volume Effect", "Price Effect", "Mix Effect", "Current Year"],
            textposition="outside",
            text=[f"${decomposition_data['total_change']/2:,.0f}", 
                  f"{decomposition_data['volume_effect_pct']:.1f}%",
                  f"{decomposition_data['price_effect_pct']:.1f}%",
                  f"{decomposition_data['residual_effect_pct']:.1f}%",
                  f"${decomposition_data['total_change']/2:,.0f}"],
            y=[decomposition_data['total_change']/2,
               decomposition_data['volume_effect'],
               decomposition_data['price_effect'],
               decomposition_data['residual_effect'],
               decomposition_data['total_change']/2],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#3b82f6"}}
        ))
        
        fig.update_layout(
            title=f"Growth Decomposition Analysis: {previous_year} → {current_year}",
            xaxis_title="Components",
            yaxis_title="Value Contribution ($)",
            height=500,
            showlegend=False,
            waterfallgap=0.3
        )
        
        # Add annotations for percentages
        annotations = [
            dict(
                x="Volume Effect",
                y=decomposition_data['volume_effect'],
                text=f"{decomposition_data['volume_effect_pct']:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40
            ),
            dict(
                x="Price Effect",
                y=decomposition_data['price_effect'],
                text=f"{decomposition_data['price_effect_pct']:.1f}%",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40
            )
        ]
        
        fig.update_layout(annotations=annotations)
        
        return fig
    
    @staticmethod
    def create_market_concentration_chart(concentration_data):
        """Create market concentration visualization"""
        if not concentration_data:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Market Share Distribution', 'Lorenz Curve - Inequality'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Market share bar chart
        top_20 = concentration_data['market_shares'].head(20)
        fig.add_trace(
            go.Bar(
                x=top_20.index,
                y=top_20.values,
                name='Market Share',
                marker_color='#3b82f6',
                hovertemplate='%{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Lorenz curve
        lorenz = concentration_data['lorenz_curve']
        fig.add_trace(
            go.Scatter(
                x=lorenz['cumulative_population'],
                y=lorenz['cumulative_shares'],
                name='Lorenz Curve',
                line=dict(color='#ef4444', width=3),
                fill='tonexty',
                fillcolor='rgba(239, 68, 68, 0.2)'
            ),
            row=1, col=2
        )
        
        # Add equality line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name='Equality Line',
                line=dict(color='#10b981', dash='dash'),
                mode='lines'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            title_text=f"Market Concentration Analysis (HHI: {concentration_data['hhi']:.0f})",
            xaxis1_title="Manufacturers",
            yaxis1_title="Market Share (%)",
            xaxis2_title="Cumulative % of Manufacturers",
            yaxis2_title="Cumulative % of Market",
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_cluster_analysis_chart(cluster_analysis):
        """Create cluster analysis visualization"""
        if not cluster_analysis:
            return None
        
        cluster_data = cluster_analysis['cluster_data']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Manufacturer Clusters', 'Cluster Characteristics'),
            specs=[[{'type': 'scatter'}, {'type': 'box'}]]
        )
        
        # Scatter plot of clusters
        for cluster_id in sorted(cluster_data['cluster'].unique()):
            cluster_points = cluster_data[cluster_data['cluster'] == cluster_id]
            fig.add_trace(
                go.Scatter(
                    x=cluster_points['pca1'],
                    y=cluster_points['pca2'],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(size=12, opacity=0.7),
                    text=cluster_points[cluster_data.columns[0]],
                    hovertemplate='%{text}<br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Box plot of cluster characteristics
        for i, feature in enumerate(cluster_analysis['features']):
            fig.add_trace(
                go.Box(
                    y=cluster_data[feature],
                    x=cluster_data['cluster'].astype(str),
                    name=feature,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Manufacturer Segmentation Analysis",
            xaxis1_title="PCA Component 1",
            yaxis1_title="PCA Component 2",
            xaxis2_title="Cluster",
            yaxis2_title="Value",
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_forecast_chart(forecast_data):
        """Create forecast visualization"""
        if not forecast_data:
            return None
        
        historical = forecast_data['historical_data']
        forecasts = forecast_data['forecasts']['value']
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['year'],
            y=historical['value'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=10)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecasts['years'],
            y=forecasts['point_forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#10b981', width=3, dash='dash'),
            marker=dict(size=10)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecasts['years'] + forecasts['years'][::-1],
            y=list(forecasts['upper_ci']) + list(forecasts['lower_ci'][::-1]),
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            height=500,
            title="Market Value Forecast",
            xaxis_title="Year",
            yaxis_title="Market Value ($)",
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add CAGR annotation
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text=f"CAGR: {forecast_data['cagr']:.1f}%",
            showarrow=False,
            font=dict(size=14, color="#1e3a8a"),
            bgcolor="white",
            bordercolor="#1e3a8a",
            borderwidth=1,
            borderpad=4
        )
        
        return fig

# ============================================================================
# ENTERPRISE DASHBOARD COMPONENTS
# ============================================================================
class DashboardComponents:
    @staticmethod
    def create_metric_card(title, value, delta=None, delta_label="", prefix="", suffix="", icon="📊"):
        """Create advanced metric card"""
        delta_html = ""
        if delta is not None:
            delta_class = "trend-up" if delta > 0 else "trend-down"
            delta_symbol = "▲" if delta > 0 else "▼"
            delta_html = f'''
            <div class="metric-trend {delta_class}">
                {delta_symbol} {abs(delta):.1f}% {delta_label}
            </div>
            '''
        
        return f'''
        <div class="metric-super-card">
            <div class="metric-label-xl">{icon} {title}</div>
            <div class="metric-value-xl">{prefix}{value}{suffix}</div>
            {delta_html}
        </div>
        '''
    
    @staticmethod
    def create_alert_box(title, message, alert_type="info"):
        """Create alert box"""
        icons = {
            "success": "✅",
            "warning": "⚠️",
            "danger": "🚨",
            "info": "ℹ️"
        }
        
        return f'''
        <div class="alert-box alert-{alert_type}">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <span style="font-size: 20px;">{icons.get(alert_type, 'ℹ️')}</span>
                <strong style="font-size: 16px;">{title}</strong>
            </div>
            <div style="font-size: 14px; line-height: 1.6;">{message}</div>
        </div>
        '''
    
    @staticmethod
    def create_progress_indicator(label, value, max_value=100, color_type="success"):
        """Create progress indicator"""
        percentage = (value / max_value * 100) if max_value > 0 else 0
        color_class = f"progress-{color_type}"
        
        return f'''
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 600; color: #475569;">{label}</span>
                <span style="font-weight: 700; color: #1e3a8a;">{value:.1f}/{max_value}</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar {color_class}" style="width: {percentage}%;"></div>
            </div>
        </div>
        '''
    
    @staticmethod
    def create_risk_matrix():
        """Create risk matrix visualization"""
        risk_levels = ["Low", "Medium", "High", "Critical"]
        probabilities = ["Very Low", "Low", "Medium", "High", "Very High"]
        
        # Create risk matrix data
        risk_data = []
        for i, impact in enumerate(risk_levels):
            for j, prob in enumerate(probabilities):
                risk_score = (i + 1) * (j + 1)
                risk_class = "risk-low" if risk_score <= 4 else ("risk-medium" if risk_score <= 9 else ("risk-high" if risk_score <= 12 else "risk-critical"))
                risk_data.append({
                    "Probability": prob,
                    "Impact": impact,
                    "Risk Score": risk_score,
                    "Class": risk_class
                })
        
        risk_df = pd.DataFrame(risk_data)
        
        fig = go.Figure(data=go.Heatmap(
            z=risk_df["Risk Score"].values.reshape(len(risk_levels), len(probabilities)),
            x=probabilities,
            y=risk_levels,
            colorscale=[[0, '#10b981'], [0.3, '#f59e0b'], [0.6, '#ef4444'], [1, '#7c3aed']],
            text=risk_df["Risk Score"].values.reshape(len(risk_levels), len(probabilities)),
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            hoverinfo="text",
            hovertext=risk_df.apply(lambda x: f"Impact: {x['Impact']}<br>Probability: {x['Probability']}<br>Risk Score: {x['Risk Score']}", axis=1).values.reshape(len(risk_levels), len(probabilities))
        ))
        
        fig.update_layout(
            title="Risk Assessment Matrix",
            xaxis_title="Probability",
            yaxis_title="Impact",
            height=500,
            template="plotly_white",
            xaxis=dict(side="top")
        )
        
        return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Load custom CSS
    load_custom_css()
    
    # Enterprise Header
    st.markdown('''
    <div class="main-header">
        <h1>💊 Pharma Analytics Intelligence Platform - Enterprise Edition</h1>
        <p>Advanced Pharmaceutical Market Intelligence | Predictive Analytics | Risk Assessment | Strategic Insights</p>
        <div style="display: flex; gap: 15px; margin-top: 20px; flex-wrap: wrap;">
            <span class="stat-badge badge-success">Real-time Analytics</span>
            <span class="stat-badge badge-info">AI-powered Insights</span>
            <span class="stat-badge badge-warning">Risk Assessment</span>
            <span class="stat-badge badge-danger">Forecasting</span>
            <span class="stat-badge badge-info">Market Intelligence</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_engine' not in st.session_state:
        st.session_state.data_engine = None
    if 'analytics_engine' not in st.session_state:
        st.session_state.analytics_engine = None
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = None
    
    # File upload section
    st.markdown('''
    <div class="dashboard-section">
        <div class="section-title">📁 Data Upload & Configuration</div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Pharmaceutical Data File",
            type=['csv', 'xlsx', 'xls'],
            help="Supports CSV and Excel files with pharmaceutical market data"
        )
    
    with col2:
        st.markdown('<br>', unsafe_allow_html=True)
        sample_data = st.checkbox("Use Sample Data", help="Load sample pharmaceutical data for demonstration")
    
    if uploaded_file or sample_data:
        if sample_data:
            # Create sample data for demonstration
            np.random.seed(42)
            n_rows = 5000
            
            sample_df = pd.DataFrame({
                'Manufacturer': np.random.choice(['Pfizer', 'Novartis', 'Roche', 'Merck', 'GSK', 'Sanofi', 'AstraZeneca', 
                                                'Johnson & Johnson', 'Bayer', 'Eli Lilly'], n_rows),
                'Molecule': np.random.choice(['Atorvastatin', 'Levothyroxine', 'Lisinopril', 'Metformin', 'Amlodipine',
                                            'Omeprazole', 'Albuterol', 'Losartan', 'Gabapentin', 'Sertraline',
                                            'Simvastatin', 'Montelukast', 'Escitalopram', 'Rosuvastatin', 'Bupropion'], n_rows),
                'Country': np.random.choice(['USA', 'Germany', 'France', 'UK', 'Japan', 'Canada', 'Italy', 'Spain', 
                                           'Australia', 'Brazil'], n_rows),
                'Therapy_Area': np.random.choice(['Cardiology', 'Endocrinology', 'Psychiatry', 'Neurology', 'Oncology',
                                                'Respiratory', 'Gastroenterology', 'Rheumatology', 'Infectious Diseases'], n_rows),
                'Specialty_Flag': np.random.choice(['Specialty', 'Generic'], n_rows, p=[0.3, 0.7]),
                '2023_USD_MNF': np.random.exponential(10000, n_rows),
                '2023_Standard_Units': np.random.lognormal(8, 1.5, n_rows),
                '2024_USD_MNF': np.random.exponential(11000, n_rows),
                '2024_Standard_Units': np.random.lognormal(8.2, 1.5, n_rows),
                '2022_USD_MNF': np.random.exponential(9000, n_rows),
                '2022_Standard_Units': np.random.lognormal(7.8, 1.5, n_rows)
            })
            
            # Create a BytesIO object to simulate uploaded file
            import io
            buffer = io.BytesIO()
            sample_df.to_excel(buffer, index=False)
            buffer.seek(0)
            uploaded_file = buffer
        
        with st.spinner('🚀 Loading and analyzing pharmaceutical data...'):
            # Initialize data engine
            data_engine = PharmaDataEngine()
            
            if data_engine.load_data(uploaded_file):
                st.session_state.data_engine = data_engine
                st.session_state.analytics_engine = AdvancedAnalyticsEngine(data_engine)
                st.session_state.viz_engine = VisualizationEngine()
                
                st.success("✅ Data loaded successfully!")
                
                # Display data quality metrics
                with st.expander("🔍 Data Quality Report", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Rows", f"{data_engine.data_quality_metrics['total_rows']:,}")
                    with col2:
                        st.metric("Total Columns", data_engine.data_quality_metrics['total_columns'])
                    with col3:
                        missing_pct = (data_engine.data_quality_metrics['missing_values'] / 
                                     (data_engine.data_quality_metrics['total_rows'] * data_engine.data_quality_metrics['total_columns']) * 100)
                        st.metric("Missing Values", f"{missing_pct:.1f}%")
                    with col4:
                        st.metric("Duplicate Rows", data_engine.data_quality_metrics['duplicate_rows'])
                
                # Get available years
                years = data_engine.extract_years()
                
                if years:
                    # Sidebar Filters
                    st.sidebar.markdown('<div class="sidebar-gradient">', unsafe_allow_html=True)
                    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                    st.sidebar.markdown('<div class="sidebar-title">⚙️ Analysis Configuration</div>', unsafe_allow_html=True)
                    
                    selected_year = st.sidebar.selectbox(
                        "Focus Year",
                        options=years,
                        index=len(years)-1
                    )
                    
                    comparison_year = st.sidebar.selectbox(
                        "Comparison Year",
                        options=[y for y in years if y != selected_year],
                        index=0 if len(years) > 1 else 0
                    )
                    
                    # Manufacturer filter
                    manufacturer_col = data_engine.get_manufacturer_column()
                    if manufacturer_col:
                        manufacturers = sorted(data_engine.df[manufacturer_col].unique().tolist())
                        selected_manufacturers = st.sidebar.multiselect(
                            "Filter by Manufacturer",
                            options=manufacturers,
                            default=manufacturers[:5] if len(manufacturers) > 5 else manufacturers
                        )
                    
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)
                    st.sidebar.markdown('</div>', unsafe_allow_html=True)
                    
                    # Main Dashboard Tabs
                    tabs = st.tabs([
                        "📊 Executive Dashboard",
                        "📈 Market Intelligence",
                        "🎯 Competitive Analysis",
                        "⚗️ Product Analytics",
                        "📉 Financial Modeling",
                        "🚨 Risk Assessment",
                        "🔮 Forecasting",
                        "📋 Data Explorer"
                    ])
                    
                    with tabs[0]:
                        # Executive Dashboard
                        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">📊 Executive Dashboard</div>', unsafe_allow_html=True)
                        
                        # Top Metrics Row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            metrics = st.session_state.analytics_engine.calculate_market_metrics(selected_year)
                            if metrics:
                                st.markdown(
                                    DashboardComponents.create_metric_card(
                                        "Market Value",
                                        f"{metrics['value_millions']:.1f}",
                                        prefix="$",
                                        suffix="M",
                                        icon="💰"
                                    ),
                                    unsafe_allow_html=True
                                )
                        
                        with col2:
                            if metrics:
                                st.markdown(
                                    DashboardComponents.create_metric_card(
                                        "Market Volume",
                                        f"{metrics['volume_thousands']:.1f}",
                                        suffix="K Units",
                                        icon="📦"
                                    ),
                                    unsafe_allow_html=True
                                )
                        
                        with col3:
                            concentration = st.session_state.analytics_engine.calculate_market_concentration(selected_year)
                            if concentration:
                                st.markdown(
                                    DashboardComponents.create_metric_card(
                                        "Market Concentration",
                                        f"{concentration['hhi']:.0f}",
                                        suffix=" HHI",
                                        icon="🎯"
                                    ),
                                    unsafe_allow_html=True
                                )
                        
                        with col4:
                            risk_score = st.session_state.analytics_engine.calculate_market_risk_score(years)
                            if risk_score:
                                st.markdown(
                                    DashboardComponents.create_metric_card(
                                        "Risk Score",
                                        f"{risk_score['overall_score']:.0f}",
                                        suffix="/100",
                                        icon="🚨"
                                    ),
                                    unsafe_allow_html=True
                                )
                        
                        # Growth Analysis
                        st.markdown("---")
                        st.markdown("### 📈 Growth Analysis")
                        
                        if len(years) >= 2:
                            decomposition = st.session_state.analytics_engine.calculate_growth_decomposition(
                                selected_year, comparison_year
                            )
                            
                            if decomposition:
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    fig = st.session_state.viz_engine.create_growth_decomposition_chart(
                                        decomposition, selected_year, comparison_year
                                    )
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.markdown(DashboardComponents.create_alert_box(
                                        "Growth Insights",
                                        f"""
                                        • **Overall Growth:** {decomposition['value_growth']:.1f}%
                                        • **Price Contribution:** {decomposition['price_effect_pct']:.1f}%
                                        • **Volume Contribution:** {decomposition['volume_effect_pct']:.1f}%
                                        • **Mix Contribution:** {decomposition['residual_effect_pct']:.1f}%
                                        """,
                                        "info"
                                    ), unsafe_allow_html=True)
                        
                        # Market Overview Visualization
                        st.markdown("---")
                        st.markdown("### 🌐 Market Overview")
                        
                        fig = st.session_state.viz_engine.create_market_overview_dashboard(
                            st.session_state.analytics_engine, selected_year
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tabs[1]:
                        # Market Intelligence
                        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">📈 Market Intelligence</div>', unsafe_allow_html=True)
                        
                        # Market Concentration Analysis
                        st.markdown("### 🎯 Market Concentration Analysis")
                        
                        concentration = st.session_state.analytics_engine.calculate_market_concentration(selected_year)
                        if concentration:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                fig = st.session_state.viz_engine.create_market_concentration_chart(concentration)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown(DashboardComponents.create_alert_box(
                                    "Concentration Insights",
                                    f"""
                                    **Market Structure:** {concentration['concentration_level']}
                                    • **Top 3 Share:** {concentration['cr3']:.1f}%
                                    • **Top 5 Share:** {concentration['cr5']:.1f}%
                                    • **Gini Coefficient:** {concentration['gini_coefficient']:.3f}
                                    • **Total Manufacturers:** {concentration['total_manufacturers']}
                                    """,
                                    "warning" if concentration['hhi'] > 1500 else "success"
                                ), unsafe_allow_html=True)
                        
                        # Price Elasticity Analysis
                        st.markdown("---")
                        st.markdown("### 💰 Price Elasticity Analysis")
                        
                        if len(years) >= 2:
                            elasticity_data = st.session_state.analytics_engine.calculate_price_elasticity(
                                selected_year, comparison_year
                            )
                            
                            if elasticity_data is not None and not elasticity_data.empty:
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    # Create elasticity distribution plot
                                    fig = px.histogram(
                                        elasticity_data,
                                        x='elasticity',
                                        nbins=30,
                                        title='Price Elasticity Distribution',
                                        labels={'elasticity': 'Elasticity Coefficient'},
                                        color_discrete_sequence=['#3b82f6']
                                    )
                                    
                                    # Add vertical lines
                                    fig.add_vline(x=-1, line_dash="dash", line_color="red", 
                                                annotation_text="Elastic", annotation_position="top right")
                                    fig.add_vline(x=0, line_dash="dash", line_color="green", 
                                                annotation_text="Inelastic", annotation_position="top left")
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    avg_elasticity = elasticity_data['elasticity'].abs().mean()
                                    elastic_count = (elasticity_data['elasticity'].abs() > 1).sum()
                                    total_count = len(elasticity_data)
                                    
                                    st.markdown(DashboardComponents.create_alert_box(
                                        "Elasticity Summary",
                                        f"""
                                        **Average |Elasticity|:** {avg_elasticity:.2f}
                                        • **Elastic Products:** {elastic_count}/{total_count}
                                        • **Inelastic Products:** {total_count - elastic_count}/{total_count}
                                        • **Most Elastic:** {elasticity_data.loc[elasticity_data['elasticity'].abs().idxmax(), 'product']}
                                        """,
                                        "warning" if avg_elasticity > 1 else "info"
                                    ), unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tabs[2]:
                        # Competitive Analysis
                        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">🎯 Competitive Analysis</div>', unsafe_allow_html=True)
                        
                        # Cluster Analysis
                        st.markdown("### 🏢 Manufacturer Segmentation")
                        
                        cluster_analysis = st.session_state.analytics_engine.perform_cluster_analysis(selected_year)
                        if cluster_analysis:
                            fig = st.session_state.viz_engine.create_cluster_analysis_chart(cluster_analysis)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Display cluster statistics
                            st.markdown("#### Cluster Characteristics")
                            cols = st.columns(len(cluster_analysis['cluster_stats']))
                            
                            for idx, stats in enumerate(cluster_analysis['cluster_stats']):
                                with cols[idx]:
                                    st.markdown(f"""
                                    <div style="background: #f8fafc; padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6;">
                                        <h4>Cluster {stats['cluster_id']}</h4>
                                        <p>Manufacturers: {stats['manufacturer_count']}</p>
                                        <p>Avg Price: ${stats['avg_price_mean']:,.2f}</p>
                                        <p>Avg Volume: {stats['volume_mean']:,.0f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Competitive Positioning Matrix
                        st.markdown("---")
                        st.markdown("### 📊 Competitive Positioning")
                        
                        value_col = data_engine.get_column_for_year('value', selected_year)
                        volume_col = data_engine.get_column_for_year('volume', selected_year)
                        manufacturer_col = data_engine.get_manufacturer_column()
                        
                        if all([value_col, volume_col, manufacturer_col]):
                            comp_data = data_engine.df.groupby(manufacturer_col).agg({
                                value_col: 'sum',
                                volume_col: 'sum'
                            }).reset_index()
                            
                            comp_data['avg_price'] = comp_data[value_col] / comp_data[volume_col]
                            comp_data['market_share'] = (comp_data[value_col] / comp_data[value_col].sum() * 100)
                            
                            # Create bubble chart
                            fig = px.scatter(
                                comp_data,
                                x='avg_price',
                                y='market_share',
                                size=value_col,
                                color='avg_price',
                                hover_name=manufacturer_col,
                                title='Competitive Positioning Matrix',
                                labels={
                                    'avg_price': 'Average Price',
                                    'market_share': 'Market Share (%)',
                                    value_col: 'Total Value'
                                },
                                color_continuous_scale='Viridis',
                                size_max=60
                            )
                            
                            # Add quadrant lines
                            fig.add_hline(y=comp_data['market_share'].median(), line_dash="dash", line_color="gray")
                            fig.add_vline(x=comp_data['avg_price'].median(), line_dash="dash", line_color="gray")
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tabs[3]:
                        # Product Analytics
                        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">⚗️ Product Analytics</div>', unsafe_allow_html=True)
                        
                        # Product Portfolio Analysis
                        st.markdown("### 📈 Product Performance Matrix")
                        
                        value_col = data_engine.get_column_for_year('value', selected_year)
                        product_col = data_engine.get_molecule_column()
                        
                        if value_col and product_col:
                            product_data = data_engine.df.groupby(product_col).agg({
                                value_col: ['sum', 'count']
                            }).reset_index()
                            
                            product_data.columns = ['product', 'total_value', 'transaction_count']
                            product_data = product_data.sort_values('total_value', ascending=False)
                            
                            # Calculate cumulative share
                            product_data['cumulative_share'] = (product_data['total_value'].cumsum() / 
                                                              product_data['total_value'].sum() * 100)
                            
                            # Create Pareto chart
                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            # Bar chart for individual values
                            fig.add_trace(
                                go.Bar(
                                    x=product_data['product'].head(20),
                                    y=product_data['total_value'].head(20),
                                    name='Product Value',
                                    marker_color='#3b82f6'
                                ),
                                secondary_y=False
                            )
                            
                            # Line chart for cumulative share
                            fig.add_trace(
                                go.Scatter(
                                    x=product_data['product'].head(20),
                                    y=product_data['cumulative_share'].head(20),
                                    name='Cumulative Share',
                                    line=dict(color='#ef4444', width=3)
                                ),
                                secondary_y=True
                            )
                            
                            fig.update_layout(
                                title='Pareto Analysis - Top 20 Products',
                                xaxis_title='Products',
                                height=500,
                                showlegend=True
                            )
                            
                            fig.update_yaxes(title_text="Value ($)", secondary_y=False)
                            fig.update_yaxes(title_text="Cumulative Share (%)", secondary_y=True)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display key insights
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                top_product = product_data.iloc[0]
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
                                            color: white; padding: 20px; border-radius: 10px;">
                                    <h4>🏆 Top Product</h4>
                                    <p style="font-size: 24px; font-weight: bold;">{top_product['product']}</p>
                                    <p>Value: ${top_product['total_value']:,.0f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                pareto_80 = product_data[product_data['cumulative_share'] <= 80]
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                            color: white; padding: 20px; border-radius: 10px;">
                                    <h4>📊 Pareto Principle</h4>
                                    <p style="font-size: 24px; font-weight: bold;">{len(pareto_80)}</p>
                                    <p>Products make 80% of revenue</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                tail_products = product_data[product_data['cumulative_share'] > 95]
                                st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                                            color: white; padding: 20px; border-radius: 10px;">
                                    <h4>📉 Long Tail</h4>
                                    <p style="font-size: 24px; font-weight: bold;">{len(tail_products)}</p>
                                    <p>Products make 5% of revenue</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tabs[4]:
                        # Financial Modeling
                        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">📉 Financial Modeling</div>', unsafe_allow_html=True)
                        
                        # Time Series Analysis
                        st.markdown("### 📈 Time Series Analysis")
                        
                        if len(years) >= 3:
                            # Collect historical data
                            historical_metrics = []
                            for year in years:
                                metrics = st.session_state.analytics_engine.calculate_market_metrics(year)
                                if metrics:
                                    historical_metrics.append({
                                        'year': year,
                                        'value': metrics['total_value'],
                                        'volume': metrics['total_volume'],
                                        'price': metrics['average_price']
                                    })
                            
                            if historical_metrics:
                                hist_df = pd.DataFrame(historical_metrics)
                                
                                # Create time series plots
                                fig = make_subplots(
                                    rows=2, cols=2,
                                    subplot_titles=('Market Value Trend', 'Market Volume Trend',
                                                   'Average Price Trend', 'Growth Rates'),
                                    vertical_spacing=0.15
                                )
                                
                                # Market Value
                                fig.add_trace(
                                    go.Scatter(
                                        x=hist_df['year'],
                                        y=hist_df['value'],
                                        mode='lines+markers',
                                        name='Value',
                                        line=dict(color='#3b82f6', width=3)
                                    ),
                                    row=1, col=1
                                )
                                
                                # Market Volume
                                fig.add_trace(
                                    go.Scatter(
                                        x=hist_df['year'],
                                        y=hist_df['volume'],
                                        mode='lines+markers',
                                        name='Volume',
                                        line=dict(color='#10b981', width=3)
                                    ),
                                    row=1, col=2
                                )
                                
                                # Average Price
                                fig.add_trace(
                                    go.Scatter(
                                        x=hist_df['year'],
                                        y=hist_df['price'],
                                        mode='lines+markers',
                                        name='Price',
                                        line=dict(color='#ef4444', width=3)
                                    ),
                                    row=2, col=1
                                )
                                
                                # Growth Rates
                                hist_df['value_growth'] = hist_df['value'].pct_change() * 100
                                fig.add_trace(
                                    go.Bar(
                                        x=hist_df['year'][1:],
                                        y=hist_df['value_growth'][1:],
                                        name='Growth',
                                        marker_color=hist_df['value_growth'][1:].apply(
                                            lambda x: '#10b981' if x > 0 else '#ef4444'
                                        )
                                    ),
                                    row=2, col=2
                                )
                                
                                fig.update_layout(height=700, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Financial Metrics Table
                                st.markdown("### 📊 Financial Metrics")
                                
                                metrics_table = hist_df.copy()
                                metrics_table['value_millions'] = metrics_table['value'] / 1e6
                                metrics_table['volume_thousands'] = metrics_table['volume'] / 1e3
                                
                                # Calculate additional metrics
                                metrics_table['value_per_unit'] = metrics_table['value'] / metrics_table['volume']
                                metrics_table['value_growth_pct'] = metrics_table['value'].pct_change() * 100
                                metrics_table['volume_growth_pct'] = metrics_table['volume'].pct_change() * 100
                                
                                st.dataframe(
                                    metrics_table.style.format({
                                        'value_millions': '${:,.2f}M',
                                        'volume_thousands': '{:,.0f}K',
                                        'price': '${:,.2f}',
                                        'value_per_unit': '${:,.2f}',
                                        'value_growth_pct': '{:.1f}%',
                                        'volume_growth_pct': '{:.1f}%'
                                    }),
                                    use_container_width=True
                                )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tabs[5]:
                        # Risk Assessment
                        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">🚨 Risk Assessment</div>', unsafe_allow_html=True)
                        
                        # Risk Score Calculation
                        risk_score = st.session_state.analytics_engine.calculate_market_risk_score(years)
                        
                        if risk_score:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Create risk gauge
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=risk_score['overall_score'],
                                    title={'text': f"Market Risk Score: {risk_score['risk_level']}"},
                                    delta={'reference': 50},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': risk_score['risk_color']},
                                        'steps': [
                                            {'range': [0, 30], 'color': "#10b981"},
                                            {'range': [30, 70], 'color': "#f59e0b"},
                                            {'range': [70, 100], 'color': "#ef4444"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 70
                                        }
                                    }
                                ))
                                
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown(DashboardComponents.create_alert_box(
                                    "Risk Assessment",
                                    f"""
                                    **Overall Risk Level:** {risk_score['risk_level']}
                                    **Score:** {risk_score['overall_score']:.1f}/100
                                    
                                    **Key Risk Factors:**
                                    {'<br>'.join([f'• {factor}: {data["score"]:.0f}' for factor, data in risk_score['risk_factors'].items()])}
                                    """,
                                    "danger" if risk_score['overall_score'] >= 70 else 
                                    "warning" if risk_score['overall_score'] >= 40 else "success"
                                ), unsafe_allow_html=True)
                            
                            # Risk Matrix
                            st.markdown("### 📊 Risk Assessment Matrix")
                            risk_matrix_fig = DashboardComponents.create_risk_matrix()
                            st.plotly_chart(risk_matrix_fig, use_container_width=True)
                            
                            # Recommendations
                            st.markdown("### 💡 Risk Mitigation Recommendations")
                            
                            for i, recommendation in enumerate(risk_score['recommendations'], 1):
                                st.markdown(f"""
                                <div style="background: #f8fafc; padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 4px solid #3b82f6;">
                                    <strong>{i}.</strong> {recommendation}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tabs[6]:
                        # Forecasting
                        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">🔮 Forecasting & Predictive Analytics</div>', unsafe_allow_html=True)
                        
                        if len(years) >= 3:
                            forecast_data = st.session_state.analytics_engine.forecast_market_trends(years)
                            
                            if forecast_data:
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    fig = st.session_state.viz_engine.create_forecast_chart(forecast_data)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.markdown(DashboardComponents.create_alert_box(
                                        "Forecast Summary",
                                        f"""
                                        **Forecast Period:** {forecast_data['forecast_periods']} years
                                        **CAGR:** {forecast_data['cagr']:.1f}%
                                        **Model R²:** {forecast_data['model_summary']['value_model_r2']:.3f}
                                        **Confidence:** {'High' if forecast_data['model_summary']['value_model_r2'] > 0.8 else 'Medium'}
                                        """,
                                        "info"
                                    ), unsafe_allow_html=True)
                                
                                # Scenario Analysis
                                st.markdown("### 📊 Scenario Analysis")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                                color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                        <h4>Optimistic Scenario</h4>
                                        <p style="font-size: 28px; font-weight: bold;">+15%</p>
                                        <p>Growth Acceleration</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); 
                                                color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                        <h4>Base Scenario</h4>
                                        <p style="font-size: 28px; font-weight: bold;">{:.1f}%</p>
                                        <p>CAGR</p>
                                    </div>
                                    """.format(forecast_data['cagr']), unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown("""
                                    <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                                                color: white; padding: 20px; border-radius: 10px; text-align: center;">
                                        <h4>Pessimistic Scenario</h4>
                                        <p style="font-size: 28px; font-weight: bold;">-10%</p>
                                        <p>Market Contraction</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tabs[7]:
                        # Data Explorer
                        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">📋 Data Explorer</div>', unsafe_allow_html=True)
                        
                        # Data Preview
                        st.markdown("### 🔍 Data Preview")
                        
                        display_cols = st.multiselect(
                            "Select columns to display:",
                            options=data_engine.df.columns.tolist(),
                            default=data_engine.column_patterns['dimension_columns'][:3] + 
                                   data_engine.column_patterns['value_columns'][:2]
                        )
                        
                        if display_cols:
                            st.dataframe(
                                data_engine.df[display_cols].head(100),
                                use_container_width=True,
                                height=500
                            )
                            
                            # Data Statistics
                            st.markdown("### 📊 Column Statistics")
                            
                            stats_cols = st.columns(3)
                            for idx, col in enumerate(display_cols[:3]):
                                with stats_cols[idx]:
                                    if pd.api.types.is_numeric_dtype(data_engine.df[col]):
                                        st.metric(
                                            f"{col} Stats",
                                            f"Mean: {data_engine.df[col].mean():.2f}",
                                            delta=f"Std: {data_engine.df[col].std():.2f}"
                                        )
                            
                            # Export Options
                            st.markdown("### 📥 Export Data")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                csv_data = data_engine.df[display_cols].to_csv(index=False)
                                st.download_button(
                                    "Download CSV",
                                    data=csv_data,
                                    file_name="pharma_data.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                excel_buffer = BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                    data_engine.df[display_cols].to_excel(writer, index=False, sheet_name='Data')
                                
                                st.download_button(
                                    "Download Excel",
                                    data=excel_buffer.getvalue(),
                                    file_name="pharma_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            with col3:
                                json_data = data_engine.df[display_cols].head(50).to_json(orient='records')
                                st.download_button(
                                    "Download JSON",
                                    data=json_data,
                                    file_name="pharma_data.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Footer
                    st.markdown('''
                    <div class="footer-enterprise">
                        <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 20px;">
                            <div>
                                <h4 style="color: #1e3a8a; margin-bottom: 10px;">Pharma Analytics Intelligence Platform</h4>
                                <p style="color: #64748b; font-size: 12px;">Enterprise Edition v3.0</p>
                            </div>
                            <div>
                                <h4 style="color: #1e3a8a; margin-bottom: 10px;">Powered By</h4>
                                <p style="color: #64748b; font-size: 12px;">Streamlit • Plotly • Pandas • Scikit-learn</p>
                            </div>
                            <div>
                                <h4 style="color: #1e3a8a; margin-bottom: 10px;">Contact</h4>
                                <p style="color: #64748b; font-size: 12px;">analytics@pharmaintelligence.com</p>
                            </div>
                        </div>
                        <hr style="border: none; height: 1px; background: #e2e8f0; margin: 20px 0;">
                        <p style="color: #94a3b8; font-size: 12px;">
                            © 2024 Pharma Analytics Intelligence Platform. All rights reserved. | 
                            <a href="#" style="color: #3b82f6; text-decoration: none;">Privacy Policy</a> | 
                            <a href="#" style="color: #3b82f6; text-decoration: none;">Terms of Service</a>
                        </p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                else:
                    st.warning("No year columns detected in the dataset. Please check data format.")
            else:
                st.error("Failed to load data. Please check the file format and try again.")
    
    else:
        # Welcome screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('''
            <div class="insight-card">
                <div class="insight-card-title">
                    🚀 Welcome to Pharma Analytics Intelligence Platform
                </div>
                <div class="insight-card-content">
                    Transform your pharmaceutical market data into actionable intelligence with our 
                    enterprise-grade analytics platform. Upload your data to unlock powerful insights:
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div style="margin-top: 30px;">
                <h3>🎯 Key Features</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;">
                    <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.05);">
                        <h4>📊 Market Intelligence</h4>
                        <p>Advanced market analysis and competitive intelligence</p>
                    </div>
                    <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.05);">
                        <h4>🎯 Risk Assessment</h4>
                        <p>Comprehensive risk scoring and mitigation strategies</p>
                    </div>
                    <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.05);">
                        <h4>🔮 Forecasting</h4>
                        <p>Predictive analytics and scenario planning</p>
                    </div>
                    <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.05);">
                        <h4>📈 Financial Modeling</h4>
                        <p>Advanced financial metrics and KPI tracking</p>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div style="background: white; padding: 25px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.08);">
                <h3 style="color: #1e3a8a; margin-bottom: 20px;">📋 Quick Start Guide</h3>
                <div style="margin-bottom: 20px;">
                    <h4>1. Data Format</h4>
                    <p style="font-size: 14px; color: #64748b;">CSV or Excel files with sales data</p>
                </div>
                <div style="margin-bottom: 20px;">
                    <h4>2. Required Columns</h4>
                    <p style="font-size: 14px; color: #64748b;">• Value columns (USD, Sales)<br>
                    • Volume columns (Units)<br>
                    • Dimension columns (Product, Manufacturer)</p>
                </div>
                <div style="margin-bottom: 20px;">
                    <h4>3. Time Periods</h4>
                    <p style="font-size: 14px; color: #64748b;">Multiple years (2022, 2023, 2024)</p>
                </div>
                <div style="background: #f0f9ff; padding: 15px; border-radius: 10px; border-left: 4px solid #0ea5e9;">
                    <h4>💡 Pro Tip</h4>
                    <p style="font-size: 13px; color: #0369a1;">The platform automatically detects column patterns and cleans data</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div style="background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
                        color: white; padding: 25px; border-radius: 16px; margin-top: 20px;">
                <h3 style="margin-bottom: 15px;">📞 Need Help?</h3>
                <p style="font-size: 14px; margin-bottom: 15px;">Contact our analytics team for support:</p>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                    <p style="margin: 0; font-size: 13px;">📧 analytics@pharmaintelligence.com</p>
                    <p style="margin: 5px 0 0 0; font-size: 13px;">📞 +1 (555) 123-4567</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

