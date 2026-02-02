# app.py - PharmaIntelligence Pro Enterprise Dashboard v5.0
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Advanced analytics libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats, signal
import scipy.cluster.hierarchy as sch

# Time series analysis
# app.py'de Prophet import'unu try-except içine alın:
try:
    from prophet import Prophet
except ImportError:
    st.warning("Prophet paketi kurulu değil. Lütfen requirements.txt dosyanıza 'prophet>=1.1.0' ekleyin.")
    Prophet = None  # veya alternatif bir sınıf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Utilities
from datetime import datetime, timedelta
import json
from io import BytesIO, StringIO
import time
import gc
import traceback
import hashlib
import pickle
import base64
import zipfile
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import math
import itertools
import re
import unicodedata
import textwrap

# Visualization
import plotly.figure_factory as ff
from plotly.offline import iplot
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Database and caching
import sqlite3
from sqlite3 import Error
# app.py'de:
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
    st.warning("Redis paketi kurulu değil. Cache özellikleri devre dışı.")

# Kod içinde redis kullanımını kontrol edin:
# Redis import kontrolü
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

if REDIS_AVAILABLE:
    # redis işlemleri yap
    pass
else:
    # alternatif cache mekanizması kullan
    pass

# Joblib import kontrolü
try:
    import joblib
except ImportError:
    joblib = None
    print("Joblib kurulu değil")
# ================================================
# 1. ENTERPRISE KONFİGÜRASYON VE STİL AYARLARI
# ================================================

st.set_page_config(
    page_title="PharmaIntelligence Pro | Enterprise Pharma Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Yardım': 'https://pharmaintelligence.com/enterprise-support',
        'Hata Bildir': "https://pharmaintelligence.com/enterprise-bug-report",
        'Hakkında': """
        ### PharmaIntelligence Enterprise v5.0
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

# ENTERPRISE MAVİ TEMA CSS STYLES - 400+ SATIR CSS
ENTERPRISE_CSS = """
<style>
    /* === ROOT VARIABLES === */
    :root {
        --primary-gradient: linear-gradient(135deg, #0c1a32 0%, #14274e 50%, #1e3a5f 100%);
        --secondary-gradient: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 50%, #3b5a8a 100%);
        --accent-gradient: linear-gradient(135deg, #2d7dd2 0%, #4a9fe3 50%, #2acaea 100%);
        --success-gradient: linear-gradient(135deg, #2dd2a3 0%, #30c9c9 50%, #25b592 100%);
        --warning-gradient: linear-gradient(135deg, #f2c94c 0%, #f2b94c 50%, #e6b445 100%);
        --danger-gradient: linear-gradient(135deg, #eb5757 0%, #d64545 50%, #c53535 100%);
        --info-gradient: linear-gradient(135deg, #2acaea 0%, #2dd2a3 50%, #2d7dd2 100%);
        
        --primary-dark: #0c1a32;
        --primary-darker: #081224;
        --primary-light: #14274e;
        --secondary-dark: #1e3a5f;
        --secondary-light: #2d4a7a;
        --accent-blue: #2d7dd2;
        --accent-blue-light: #4a9fe3;
        --accent-blue-dark: #1a5fa0;
        --accent-cyan: #2acaea;
        --accent-teal: #30c9c9;
        --accent-turquoise: #2dd2a3;
        
        --success: #2dd2a3;
        --success-dark: #25b592;
        --warning: #f2c94c;
        --warning-dark: #e6b445;
        --danger: #eb5757;
        --danger-dark: #d64545;
        --info: #2acaea;
        --info-dark: #25b0d0;
        
        --text-primary: #ffffff;
        --text-secondary: #cbd5e1;
        --text-tertiary: #94a3b8;
        --text-muted: #64748b;
        --text-light: #e2e8f0;
        
        --bg-primary: #0c1a32;
        --bg-secondary: #14274e;
        --bg-tertiary: #1e3a5f;
        --bg-card: rgba(30, 58, 95, 0.8);
        --bg-card-solid: #1e3a5f;
        --bg-hover: rgba(45, 125, 210, 0.15);
        --bg-selected: rgba(45, 125, 210, 0.25);
        --bg-surface: rgba(20, 39, 78, 0.9);
        --bg-overlay: rgba(12, 26, 50, 0.95);
        
        --border-primary: #2d4a7a;
        --border-secondary: #3b5a8a;
        --border-accent: #2d7dd2;
        --border-success: #2dd2a3;
        --border-warning: #f2c94c;
        --border-danger: #eb5757;
        
        --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.3);
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.4);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.5);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.6);
        --shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.7);
        --shadow-2xl: 0 24px 64px rgba(0, 0, 0, 0.8);
        --shadow-inner: inset 0 2px 4px rgba(0, 0, 0, 0.2);
        --shadow-glow: 0 0 20px rgba(45, 125, 210, 0.3);
        
        --radius-xs: 4px;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --radius-2xl: 24px;
        --radius-full: 9999px;
        
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: 250ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-bounce: 400ms cubic-bezier(0.68, -0.55, 0.265, 1.55);
        
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        --font-mono: 'SF Mono', 'Roboto Mono', 'Courier New', monospace;
        --font-heading: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* === GLOBAL STYLES === */
    .stApp {
        background: var(--primary-gradient);
        background-attachment: fixed;
        font-family: var(--font-sans);
        color: var(--text-primary);
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 10% 20%, rgba(45, 125, 210, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(42, 202, 234, 0.1) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(30, 58, 95, 0.2) 0%, transparent 60%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Streamlit component overrides */
    .stDataFrame, .stTable {
        background: var(--bg-card) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-primary) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stDataFrame:hover, .stTable:hover {
        box-shadow: var(--shadow-md) !important;
        border-color: var(--border-accent) !important;
    }
    
    /* Metric cards styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 900 !important;
        color: var(--text-primary) !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.5px;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* Input fields styling */
    .stSelectbox > div > div,
    .stMultiselect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        background: var(--bg-tertiary) !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-primary) !important;
        color: var(--text-primary) !important;
        transition: all var(--transition-fast) !important;
    }
    
    .stSelectbox > div > div:hover,
    .stMultiselect > div > div:hover,
    .stTextInput > div > div > input:hover,
    .stNumberInput > div > div > input:hover {
        border-color: var(--border-accent) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stMultiselect > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(45, 125, 210, 0.1) !important;
    }
    
    /* Slider styling */
    .stSlider {
        background: var(--bg-tertiary) !important;
        padding: 1rem !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-primary) !important;
    }
    
    .stSlider > div > div > div {
        background: var(--accent-gradient) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all var(--transition-normal) !important;
        box-shadow: var(--shadow-sm) !important;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stButton > button:disabled {
        opacity: 0.5 !important;
        cursor: not-allowed !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        background: var(--bg-tertiary) !important;
        padding: 0.5rem !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-primary) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all var(--transition-fast) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Checkbox and radio */
    .stCheckbox > label,
    .stRadio > label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        transition: all var(--transition-fast) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-hover) !important;
        border-color: var(--border-accent) !important;
    }
    
    /* === TYPOGRAPHY === */
    .enterprise-title {
        font-size: 3.2rem;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        line-height: 1.1;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .enterprise-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 900px;
        line-height: 1.7;
        margin-bottom: 2.5rem;
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        border-left: 4px solid var(--accent-blue);
    }
    
    .section-title {
        font-size: 2rem;
        color: var(--text-primary);
        font-weight: 800;
        margin: 3rem 0 1.8rem 0;
        padding: 1.2rem 0 1.2rem 1.5rem;
        border-left: 6px solid var(--accent-blue);
        background: linear-gradient(90deg, rgba(45, 125, 210, 0.15), transparent);
        border-radius: var(--radius-sm);
        position: relative;
        overflow: hidden;
    }
    
    .section-title::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, var(--accent-blue), transparent);
    }
    
    .section-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-blue));
    }
    
    .subsection-title {
        font-size: 1.5rem;
        color: var(--text-primary);
        font-weight: 700;
        margin: 2.5rem 0 1.2rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid var(--border-primary);
        position: relative;
    }
    
    .subsection-title::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100px;
        height: 2px;
        background: var(--accent-gradient);
    }
    
    .card-title {
        font-size: 1.3rem;
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .card-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    /* === ENTERPRISE METRIC CARDS === */
    .enterprise-metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        padding: 1.8rem;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border-primary);
        transition: all var(--transition-normal);
        height: 100%;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .enterprise-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--accent-gradient);
        z-index: 2;
    }
    
    .enterprise-metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), transparent);
        opacity: 0;
        transition: opacity var(--transition-normal);
        z-index: -1;
    }
    
    .enterprise-metric-card:hover::after {
        opacity: 1;
    }
    
    .enterprise-metric-card:hover {
        transform: translateY(-6px);
        box-shadow: var(--shadow-xl);
        border-color: var(--border-accent);
    }
    
    .enterprise-metric-card.primary {
        background: linear-gradient(145deg, var(--secondary-dark), var(--bg-card));
        border: 1px solid var(--border-accent);
    }
    
    .enterprise-metric-card.success {
        background: linear-gradient(145deg, rgba(45, 210, 163, 0.15), var(--bg-card));
        border: 1px solid var(--border-success);
    }
    
    .enterprise-metric-card.warning {
        background: linear-gradient(145deg, rgba(242, 201, 76, 0.15), var(--bg-card));
        border: 1px solid var(--border-warning);
    }
    
    .enterprise-metric-card.danger {
        background: linear-gradient(145deg, rgba(235, 87, 87, 0.15), var(--bg-card));
        border: 1px solid var(--border-danger);
    }
    
    .enterprise-metric-card.info {
        background: linear-gradient(145deg, rgba(42, 202, 234, 0.15), var(--bg-card));
        border: 1px solid var(--info);
    }
    
    .enterprise-metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0.8rem 0;
        color: var(--text-primary);
        line-height: 1;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .enterprise-metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .enterprise-metric-trend {
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .enterprise-metric-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    
    /* === INSIGHT CARDS === */
    .enterprise-insight-card {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
        border-left: 6px solid;
        margin: 1rem 0;
        transition: all var(--transition-normal);
        position: relative;
        overflow: hidden;
    }
    
    .enterprise-insight-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), transparent);
        opacity: 0;
        transition: opacity var(--transition-normal);
    }
    
    .enterprise-insight-card:hover::before {
        opacity: 1;
    }
    
    .enterprise-insight-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .enterprise-insight-card.info { 
        border-left-color: var(--accent-blue);
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.1), var(--bg-card));
    }
    
    .enterprise-insight-card.success { 
        border-left-color: var(--success);
        background: linear-gradient(135deg, rgba(45, 210, 163, 0.1), var(--bg-card));
    }
    
    .enterprise-insight-card.warning { 
        border-left-color: var(--warning);
        background: linear-gradient(135deg, rgba(242, 201, 76, 0.1), var(--bg-card));
    }
    
    .enterprise-insight-card.danger { 
        border-left-color: var(--danger);
        background: linear-gradient(135deg, rgba(235, 87, 87, 0.1), var(--bg-card));
    }
    
    .enterprise-insight-card.premium { 
        border-left-color: var(--accent-cyan);
        background: linear-gradient(135deg, rgba(42, 202, 234, 0.1), var(--bg-card));
    }
    
    .enterprise-insight-icon {
        font-size: 1.8rem;
        margin-bottom: 0.8rem;
        display: inline-block;
        width: 50px;
        height: 50px;
        border-radius: var(--radius-full);
        background: rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .enterprise-insight-title {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
        line-height: 1.4;
    }
    
    .enterprise-insight-content {
        color: var(--text-secondary);
        line-height: 1.7;
        font-size: 0.98rem;
        margin-bottom: 0.5rem;
    }
    
    .enterprise-insight-footer {
        font-size: 0.85rem;
        color: var(--text-tertiary);
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* === FILTER SECTION === */
    .enterprise-filter-section {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-lg);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-primary);
        transition: all var(--transition-normal);
    }
    
    .enterprise-filter-section:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-xl);
    }
    
    .enterprise-filter-title {
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-size: 1.3rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid var(--border-primary);
    }
    
    .enterprise-filter-title::before {
        content: '';
        display: inline-block;
        width: 8px;
        height: 24px;
        background: var(--accent-gradient);
        border-radius: var(--radius-full);
    }
    
    /* === FILTER STATUS === */
    .enterprise-filter-status {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.25), rgba(42, 202, 234, 0.2));
        backdrop-filter: blur(10px);
        padding: 1.2rem 1.5rem;
        border-radius: var(--radius-lg);
        margin-bottom: 2rem;
        border-left: 6px solid var(--success);
        box-shadow: var(--shadow-lg);
        color: var(--text-primary);
        font-size: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .enterprise-filter-status::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), transparent);
        pointer-events: none;
    }
    
    .enterprise-filter-status-danger {
        background: linear-gradient(135deg, rgba(235, 87, 87, 0.25), rgba(214, 69, 69, 0.2));
        border-left: 6px solid var(--warning);
    }
    
    .enterprise-filter-status-warning {
        background: linear-gradient(135deg, rgba(242, 201, 76, 0.25), rgba(242, 185, 76, 0.2));
        border-left: 6px solid var(--accent-blue);
    }
    
    /* === DATA GRID === */
    .enterprise-data-grid {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        overflow: hidden;
        box-shadow: var(--shadow-xl);
        border: 1px solid var(--border-primary);
        transition: all var(--transition-normal);
    }
    
    .enterprise-data-grid:hover {
        border-color: var(--border-accent);
        box-shadow: var(--shadow-2xl);
    }
    
    .enterprise-data-grid-header {
        background: var(--secondary-dark);
        padding: 1.2rem 1.5rem;
        border-bottom: 1px solid var(--border-primary);
        font-weight: 700;
        color: var(--text-primary);
        font-size: 1.1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* === LOADING ANIMATIONS === */
    @keyframes enterprise-pulse {
        0%, 100% { 
            opacity: 1;
            transform: scale(1);
        }
        50% { 
            opacity: 0.5;
            transform: scale(0.98);
        }
    }
    
    @keyframes enterprise-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes enterprise-slide-in {
        from { 
            opacity: 0;
            transform: translateY(20px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes enterprise-fade-in {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes enterprise-gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .enterprise-loading-pulse {
        animation: enterprise-pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    .enterprise-loading-spin {
        animation: enterprise-spin 1s linear infinite;
    }
    
    .enterprise-animate-slide-in {
        animation: enterprise-slide-in 0.5s ease-out;
    }
    
    .enterprise-animate-fade-in {
        animation: enterprise-fade-in 0.3s ease-out;
    }
    
    .enterprise-animate-gradient {
        background-size: 200% 200%;
        animation: enterprise-gradient 3s ease infinite;
    }
    
    /* === STATUS INDICATORS === */
    .enterprise-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: var(--radius-full);
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .enterprise-status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: var(--radius-full);
        animation: enterprise-pulse 2s infinite;
    }
    
    .enterprise-status-online { 
        background: var(--success);
        box-shadow: 0 0 10px rgba(45, 210, 163, 0.5);
    }
    
    .enterprise-status-warning { 
        background: var(--warning);
        box-shadow: 0 0 10px rgba(242, 201, 76, 0.5);
    }
    
    .enterprise-status-error { 
        background: var(--danger);
        box-shadow: 0 0 10px rgba(235, 87, 87, 0.5);
    }
    
    .enterprise-status-processing { 
        background: var(--info);
        box-shadow: 0 0 10px rgba(42, 202, 234, 0.5);
    }
    
    /* === BADGES === */
    .enterprise-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        border-radius: var(--radius-full);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border: 1px solid;
        transition: all var(--transition-fast);
    }
    
    .enterprise-badge:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-sm);
    }
    
    .enterprise-badge-success {
        background: rgba(45, 210, 163, 0.2);
        color: var(--success);
        border-color: rgba(45, 210, 163, 0.3);
    }
    
    .enterprise-badge-warning {
        background: rgba(242, 201, 76, 0.2);
        color: var(--warning);
        border-color: rgba(242, 201, 76, 0.3);
    }
    
    .enterprise-badge-danger {
        background: rgba(235, 87, 87, 0.2);
        color: var(--danger);
        border-color: rgba(235, 87, 87, 0.3);
    }
    
    .enterprise-badge-info {
        background: rgba(45, 125, 210, 0.2);
        color: var(--accent-blue);
        border-color: rgba(45, 125, 210, 0.3);
    }
    
    .enterprise-badge-premium {
        background: rgba(42, 202, 234, 0.2);
        color: var(--accent-cyan);
        border-color: rgba(42, 202, 234, 0.3);
    }
    
    /* === SIDEBAR === */
    .enterprise-sidebar {
        background: var(--secondary-dark) !important;
        border-right: 1px solid var(--border-primary) !important;
    }
    
    .enterprise-sidebar-title {
        font-size: 1.5rem;
        color: var(--text-primary);
        font-weight: 800;
        margin-bottom: 1.8rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--accent-blue);
        position: relative;
        text-align: center;
    }
    
    .enterprise-sidebar-title::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 25%;
        width: 50%;
        height: 2px;
        background: var(--accent-gradient);
    }
    
    /* === FEATURE CARDS === */
    .enterprise-feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2.5rem 0;
    }
    
    .enterprise-feature-card {
        background: linear-gradient(145deg, var(--bg-card), var(--secondary-dark));
        padding: 2rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-primary);
        transition: all var(--transition-normal);
        height: 100%;
        position: relative;
        overflow: hidden;
        text-align: center;
    }
    
    .enterprise-feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--accent-gradient);
    }
    
    .enterprise-feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-2xl);
        border-color: var(--border-accent);
    }
    
    .enterprise-feature-icon {
        font-size: 3rem;
        margin-bottom: 1.5rem;
        display: inline-block;
        width: 80px;
        height: 80px;
        border-radius: var(--radius-full);
        background: rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.5rem;
    }
    
    .enterprise-feature-title {
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    .enterprise-feature-description {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    /* === WELCOME CONTAINER === */
    .enterprise-welcome-container {
        background: linear-gradient(145deg, var(--bg-card), var(--secondary-dark));
        padding: 4rem;
        border-radius: var(--radius-2xl);
        box-shadow: var(--shadow-2xl);
        text-align: center;
        margin: 3rem auto;
        max-width: 1000px;
        border: 1px solid var(--border-primary);
        position: relative;
        overflow: hidden;
    }
    
    .enterprise-welcome-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        right: -50%;
        bottom: -50%;
        background: 
            radial-gradient(circle at 30% 30%, rgba(45, 125, 210, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 70% 70%, rgba(42, 202, 234, 0.1) 0%, transparent 50%);
        pointer-events: none;
        animation: enterprise-gradient 10s ease infinite;
    }
    
    .enterprise-welcome-icon {
        font-size: 6rem;
        margin-bottom: 2rem;
        display: inline-block;
        width: 120px;
        height: 120px;
        border-radius: var(--radius-full);
        background: var(--accent-gradient);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 2rem;
        box-shadow: var(--shadow-lg);
    }
    
    .enterprise-welcome-title {
        font-size: 2.8rem;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        font-weight: 900;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* === TOOLTIPS === */
    .enterprise-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .enterprise-tooltip .enterprise-tooltip-text {
        visibility: hidden;
        width: 300px;
        background: var(--bg-tertiary);
        color: var(--text-primary);
        text-align: left;
        padding: 1rem;
        border-radius: var(--radius-md);
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity var(--transition-normal);
        box-shadow: var(--shadow-lg);
        border: 1px solid var(--border-primary);
        font-size: 0.9rem;
        line-height: 1.5;
        backdrop-filter: blur(10px);
    }
    
    .enterprise-tooltip:hover .enterprise-tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    /* === PROGRESS BARS === */
    .enterprise-progress-bar {
        width: 100%;
        height: 8px;
        background: var(--bg-tertiary);
        border-radius: var(--radius-full);
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .enterprise-progress-fill {
        height: 100%;
        background: var(--accent-gradient);
        border-radius: var(--radius-full);
        transition: width 1s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .enterprise-progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: enterprise-gradient 2s linear infinite;
    }
    
    /* === ALERTS === */
    .enterprise-alert {
        padding: 1.2rem 1.5rem;
        border-radius: var(--radius-md);
        margin: 1rem 0;
        border-left: 6px solid;
        background: var(--bg-card);
        box-shadow: var(--shadow-md);
        animation: enterprise-slide-in 0.3s ease-out;
    }
    
    .enterprise-alert-success {
        border-left-color: var(--success);
        background: linear-gradient(135deg, rgba(45, 210, 163, 0.1), var(--bg-card));
    }
    
    .enterprise-alert-warning {
        border-left-color: var(--warning);
        background: linear-gradient(135deg, rgba(242, 201, 76, 0.1), var(--bg-card));
    }
    
    .enterprise-alert-danger {
        border-left-color: var(--danger);
        background: linear-gradient(135deg, rgba(235, 87, 87, 0.1), var(--bg-card));
    }
    
    .enterprise-alert-info {
        border-left-color: var(--info);
        background: linear-gradient(135deg, rgba(42, 202, 234, 0.1), var(--bg-card));
    }
    
    /* === MODALS === */
    .enterprise-modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(12, 26, 50, 0.9);
        backdrop-filter: blur(5px);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: enterprise-fade-in 0.3s ease-out;
    }
    
    .enterprise-modal {
        background: var(--bg-card);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow-2xl);
        border: 1px solid var(--border-primary);
        max-width: 800px;
        width: 90%;
        max-height: 90vh;
        overflow-y: auto;
        animation: enterprise-slide-in 0.3s ease-out;
    }
    
    .enterprise-modal-header {
        padding: 1.5rem 2rem;
        border-bottom: 1px solid var(--border-primary);
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--secondary-dark);
        border-radius: var(--radius-xl) var(--radius-xl) 0 0;
    }
    
    .enterprise-modal-body {
        padding: 2rem;
    }
    
    /* === DASHBOARD LAYOUT === */
    .enterprise-dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .enterprise-dashboard-widget {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--border-primary);
        box-shadow: var(--shadow-md);
        transition: all var(--transition-normal);
        height: 100%;
    }
    
    .enterprise-dashboard-widget:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--border-accent);
    }
    
    /* === RESPONSIVE DESIGN === */
    @media (max-width: 768px) {
        .enterprise-title {
            font-size: 2.2rem;
        }
        
        .section-title {
            font-size: 1.6rem;
        }
        
        .enterprise-feature-grid {
            grid-template-columns: 1fr;
        }
        
        .enterprise-dashboard-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* === SCROLLBAR STYLING === */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
        border-radius: var(--radius-full);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-primary);
        border-radius: var(--radius-full);
        transition: background var(--transition-fast);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--border-accent);
    }
    
    /* === PRINT STYLES === */
    @media print {
        .stApp {
            background: white !important;
            color: black !important;
        }
        
        .enterprise-metric-card,
        .enterprise-insight-card {
            break-inside: avoid;
            box-shadow: none !important;
            border: 1px solid #ddd !important;
        }
    }
</style>
"""

st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)

# ================================================
# 2. ENTERPRISE VERİ İŞLEME SİSTEMİ - 500+ SATIR
# ================================================

class EnterpriseVeriSistemi:
    """Enterprise-level veri işleme ve yönetim sistemi"""
    
    def __init__(self):
        self.cache_dir = tempfile.mkdtemp(prefix="pharma_cache_")
        self.data_cache = {}
        self.metadata_cache = {}
        self.performance_stats = {
            'total_processed': 0,
            'avg_processing_time': 0,
            'memory_usage': 0
        }
        
    @staticmethod
    @st.cache_data(ttl=7200, show_spinner=False, max_entries=20, persist="disk")
    def buyuk_veri_yukle_optimize(_dosya, orneklem_boyutu=None, chunk_size=50000):
        """Büyük veri setlerini optimize edilmiş şekilde yükle"""
        try:
            baslangic = time.time()
            dosya_adi = _dosya.name if hasattr(_dosya, 'name') else str(_dosya)
            dosya_boyutu = len(_dosya.getvalue()) if hasattr(_dosya, 'getvalue') else 0
            
            with st.spinner(f"📥 **{dosya_adi}** yükleniyor...") as spinner:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Dosya türüne göre yükleme
                if dosya_adi.lower().endswith('.csv'):
                    # CSV için optimize edilmiş yükleme
                    if orneklem_boyutu:
                        df = pd.read_csv(_dosya, nrows=orneklem_boyutu)
                    else:
                        # Büyük CSV için chunk-based yükleme
                        chunks = []
                        chunk_reader = pd.read_csv(_dosya, chunksize=chunk_size)
                        
                        for i, chunk in enumerate(chunk_reader):
                            chunks.append(chunk)
                            progress = min((i + 1) * chunk_size / max(1, dosya_boyutu/100), 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"📊 Chunk {i+1} yüklendi: {len(chunk):,} satır")
                        
                        df = pd.concat(chunks, ignore_index=True)
                        
                elif dosya_adi.lower().endswith(('.xlsx', '.xls')):
                    # Excel için sheet detection ve optimize yükleme
                    xl = pd.ExcelFile(_dosya)
                    sheet_names = xl.sheet_names
                    
                    if len(sheet_names) == 1:
                        # Tek sheet varsa
                        if orneklem_boyutu:
                            df = pd.read_excel(_dosya, nrows=orneklem_boyutu)
                        else:
                            df = pd.read_excel(_dosya)
                    else:
                        # Multiple sheets varsa
                        st.info(f"📑 {len(sheet_names)} sheet bulundu: {', '.join(sheet_names)}")
                        selected_sheet = st.selectbox("Sheet seçin:", sheet_names)
                        df = pd.read_excel(_dosya, sheet_name=selected_sheet)
                
                elif dosya_adi.lower().endswith('.parquet'):
                    df = pd.read_parquet(_dosya)
                elif dosya_adi.lower().endswith('.json'):
                    df = pd.read_json(_dosya)
                else:
                    st.error(f"Desteklenmeyen dosya formatı: {dosya_adi}")
                    return None
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ {len(df):,} satır başarıyla yüklendi")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
            # Optimizasyon uygula
            df = EnterpriseVeriSistemi.dataframe_ileri_optimizasyon(df)
            
            bitis = time.time()
            yukleme_suresi = bitis - baslangic
            
            # Performans metrikleri
            bellek_kullanimi = df.memory_usage(deep=True).sum() / 1024**2
            st.success(f"""
            🚀 **Veri Yükleme Tamamlandı:**
            • **Satır:** {len(df):,}
            • **Sütun:** {len(df.columns)}
            • **Bellek:** {bellek_kullanimi:.1f} MB
            • **Süre:** {yukleme_suresi:.2f}s
            • **Satır/Saniye:** {len(df)/yukleme_suresi:,.0f}
            """)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.error(f"**Detay:** {traceback.format_exc()}")
            return None
    
    @staticmethod
    def dataframe_ileri_optimizasyon(df):
        """İleri seviye DataFrame optimizasyonu"""
        try:
            orijinal_bellek = df.memory_usage(deep=True).sum() / 1024**2
            orijinal_satir = len(df)
            
            with st.spinner("🔧 **Veri optimizasyonu uygulanıyor...**"):
                # 1. Sütun isimlerini standardize et
                df.columns = EnterpriseVeriSistemi.sutun_isimleri_standardizasyon(df.columns)
                
                # 2. Veri temizliği
                df = EnterpriseVeriSistemi.veri_temizleme_pipeline(df)
                
                # 3. Bellek optimizasyonu
                df = EnterpriseVeriSistemi.bellek_optimizasyonu(df)
                
                # 4. Veri tipleri optimizasyonu
                df = EnterpriseVeriSistemi.veri_tipi_optimizasyonu(df)
                
                # 5. Kategorik değişken optimizasyonu
                df = EnterpriseVeriSistemi.kategorik_optimizasyonu(df)
                
                # 6. Tarih/saat optimizasyonu
                df = EnterpriseVeriSistemi.tarih_optimizasyonu(df)
                
                # 7. NaN değer işleme
                df = EnterpriseVeriSistemi.nan_islemleri(df)
                
                # 8. Outlier tespiti ve işleme
                df = EnterpriseVeriSistemi.outlier_islemleri(df)
                
                # 9. Tekrar eden satırları kaldır
                orijinal_len = len(df)
                df = df.drop_duplicates()
                kaldirilan_satir = orijinal_len - len(df)
                if kaldirilan_satir > 0:
                    st.info(f"🗑️ {kaldirilan_satir:,} tekrar eden satır kaldırıldı")
                
                # 10. İndeks optimizasyonu
                df = df.reset_index(drop=True)
                
            optimize_bellek = df.memory_usage(deep=True).sum() / 1024**2
            bellek_tasarrufu = orijinal_bellek - optimize_bellek
            
            if bellek_tasarrufu > 0:
                st.success(f"""
                💾 **Optimizasyon Başarılı:**
                • **Orijinal:** {orijinal_bellek:.1f} MB
                • **Optimize:** {optimize_bellek:.1f} MB
                • **Tasarruf:** {bellek_tasarrufu:.1f} MB (%{bellek_tasarrufu/orijinal_bellek*100:.1f})
                """)
            
            return df
            
        except Exception as e:
            st.warning(f"Optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def sutun_isimleri_standardizasyon(sutunlar):
        """Sütun isimlerini enterprise standardına dönüştür"""
        temizlenen = []
        standard_map = {
            # İngilizce-Türkçe mapping
            'sales': 'Satış',
            'revenue': 'Gelir',
            'profit': 'Kar',
            'price': 'Fiyat',
            'cost': 'Maliyet',
            'quantity': 'Miktar',
            'volume': 'Hacim',
            'unit': 'Birim',
            'product': 'Ürün',
            'company': 'Şirket',
            'corporation': 'Şirket',
            'country': 'Ülke',
            'region': 'Bölge',
            'city': 'Şehir',
            'date': 'Tarih',
            'time': 'Zaman',
            'year': 'Yıl',
            'month': 'Ay',
            'quarter': 'Çeyrek',
            'week': 'Hafta',
            'day': 'Gün',
            'molecule': 'Molekül',
            'drug': 'İlaç',
            'pharma': 'Farma',
            'market': 'Pazar',
            'share': 'Pay',
            'growth': 'Büyüme',
            'trend': 'Trend',
            'forecast': 'Tahmin',
            'actual': 'Gerçek',
            'target': 'Hedef',
            'budget': 'Bütçe',
            'customer': 'Müşteri',
            'client': 'Müşteri',
            'patient': 'Hasta',
            'doctor': 'Doktor',
            'hospital': 'Hastane',
            'clinic': 'Klinik',
            'prescription': 'Reçete',
            'dosage': 'Doz',
            'mg': 'mg',
            'ml': 'ml',
            'strength': 'Kuvvet',
            'form': 'Form',
            'tablet': 'Tablet',
            'capsule': 'Kapsül',
            'injection': 'Enjeksiyon',
            'cream': 'Krem',
            'ointment': 'Merhem',
            'syrup': 'Şurup',
            'drops': 'Damla',
            
            # Türkçe karakter düzeltme
            'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
            'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
            'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
        }
        
        for sutun in sutunlar:
            if not isinstance(sutun, str):
                sutun = str(sutun)
            
            # Önce standart mapping uygula
            orijinal_sutun = sutun.lower()
            for eng, tr in standard_map.items():
                if eng.lower() in orijinal_sutun:
                    sutun = sutun.lower().replace(eng.lower(), tr)
            
            # Türkçe karakter düzeltme
            for tr_char, en_char in list(standard_map.items())[-6:]:
                sutun = sutun.replace(tr_char, en_char)
            
            # Özel karakterleri temizle
            sutun = re.sub(r'[^\w\s]', '_', sutun)
            sutun = re.sub(r'\s+', '_', sutun.strip())
            sutun = sutun.replace('\n', '_').replace('\r', '_').replace('\t', '_')
            
            # Baştaki ve sondaki underscore'ları temizle
            sutun = sutun.strip('_')
            
            # Büyük harf standardizasyonu
            sutun = sutun.title()
            
            # Özel durumlar
            if sutun == '':
                sutun = f'Unnamed_{len(temizlenen)}'
            
            temizlenen.append(sutun)
        
        return temizlenen
    
    @staticmethod
    def veri_temizleme_pipeline(df):
        """Veri temizleme pipeline'ı"""
        try:
            # Boş stringleri NaN'a çevir
            df = df.replace(['', ' ', '  ', '   ', 'null', 'NULL', 'None', 'none', 'NaN', 'nan'], np.nan)
            
            # String sütunları temizle
            string_sutunlar = df.select_dtypes(include=['object']).columns
            for sutun in string_sutunlar:
                try:
                    df[sutun] = df[sutun].astype(str).str.strip()
                    # Fazla boşlukları kaldır
                    df[sutun] = df[sutun].str.replace(r'\s+', ' ', regex=True)
                except:
                    pass
            
            return df
        except Exception as e:
            st.warning(f"Veri temizleme hatası: {str(e)}")
            return df
    
    @staticmethod
    def bellek_optimizasyonu(df):
        """Bellek kullanımını optimize et"""
        try:
            # Sayısal sütunlar için optimize et
            for sutun in df.select_dtypes(include=[np.number]).columns:
                try:
                    sutun_min = df[sutun].min()
                    sutun_max = df[sutun].max()
                    sutun_tipi = df[sutun].dtype
                    
                    # Integer optimizasyonu
                    if np.issubdtype(sutun_tipi, np.integer):
                        if sutun_min >= 0:
                            if sutun_max <= 255:
                                df[sutun] = df[sutun].astype(np.uint8)
                            elif sutun_max <= 65535:
                                df[sutun] = df[sutun].astype(np.uint16)
                            elif sutun_max <= 4294967295:
                                df[sutun] = df[sutun].astype(np.uint32)
                        else:
                            if sutun_min >= -128 and sutun_max <= 127:
                                df[sutun] = df[sutun].astype(np.int8)
                            elif sutun_min >= -32768 and sutun_max <= 32767:
                                df[sutun] = df[sutun].astype(np.int16)
                            elif sutun_min >= -2147483648 and sutun_max <= 2147483647:
                                df[sutun] = df[sutun].astype(np.int32)
                    
                    # Float optimizasyonu
                    elif np.issubdtype(sutun_tipi, np.floating):
                        df[sutun] = df[sutun].astype(np.float32)
                        
                except Exception as e:
                    continue
            
            return df
        except Exception as e:
            st.warning(f"Bellek optimizasyonu hatası: {str(e)}")
            return df
    
    @staticmethod
    def veri_tipi_optimizasyonu(df):
        """Veri tiplerini optimize et"""
        try:
            # Boolean sütunları tespit et
            for sutun in df.select_dtypes(include=['object']).columns:
                try:
                    unique_vals = df[sutun].dropna().unique()
                    if len(unique_vals) == 2:
                        if set(unique_vals).issubset({'True', 'False', 'true', 'false', '1', '0', 'yes', 'no'}):
                            df[sutun] = df[sutun].map({'True': True, 'true': True, '1': True, 'yes': True,
                                                      'False': False, 'false': False, '0': False, 'no': False})
                except:
                    pass
            
            return df
        except Exception as e:
            st.warning(f"Veri tipi optimizasyonu hatası: {str(e)}")
            return df
    
    @staticmethod
    def kategorik_optimizasyonu(df):
        """Kategorik değişkenleri optimize et"""
        try:
            for sutun in df.select_dtypes(include=['object']).columns:
                try:
                    unique_count = df[sutun].nunique()
                    total_count = len(df)
                    
                    # Eşsiz değer oranına göre kategori optimizasyonu
                    unique_ratio = unique_count / total_count
                    
                    if unique_ratio < 0.5:  # %50'den az eşsiz değer
                        df[sutun] = df[sutun].astype('category')
                    
                    # Çok fazla kategori varsa, küçük kategorileri 'Diğer' olarak birleştir
                    elif unique_count > 100:
                        value_counts = df[sutun].value_counts()
                        small_categories = value_counts[value_counts < total_count * 0.01].index
                        df[sutun] = df[sutun].replace(small_categories, 'Diğer')
                        df[sutun] = df[sutun].astype('category')
                        
                except:
                    pass
            
            return df
        except Exception as e:
            st.warning(f"Kategorik optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def tarih_optimizasyonu(df):
        """Tarih/saat sütunlarını optimize et"""
        try:
            tarih_deseni = ['tarih', 'date', 'time', 'zaman', 'year', 'yıl', 'month', 'ay', 
                           'day', 'gün', 'hour', 'saat', 'minute', 'dakika', 'second', 'saniye',
                           'datetime', 'timestamp']
            
            for sutun in df.columns:
                sutun_str = str(sutun).lower()
                if any(desen in sutun_str for desen in tarih_deseni):
                    try:
                        df[sutun] = pd.to_datetime(df[sutun], errors='coerce', infer_datetime_format=True)
                        
                        # Tarih parçalarını ayır
                        if 'date' in sutun_str or 'tarih' in sutun_str:
                            df[f'{sutun}_Yıl'] = df[sutun].dt.year
                            df[f'{sutun}_Ay'] = df[sutun].dt.month
                            df[f'{sutun}_Gün'] = df[sutun].dt.day
                            df[f'{sutun}_Hafta'] = df[sutun].dt.isocalendar().week
                            df[f'{sutun}_Çeyrek'] = df[sutun].dt.quarter
                            
                    except:
                        pass
            
            return df
        except Exception as e:
            st.warning(f"Tarih optimizasyonu hatası: {str(e)}")
            return df
    
    @staticmethod
    def nan_islemleri(df):
        """NaN değerleri işle"""
        try:
            nan_raporu = {
                'toplam_nan': df.isna().sum().sum(),
                'nan_oran': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
                'sutun_bazli_nan': df.isna().sum(),
                'sutun_nan_oran': (df.isna().sum() / len(df)) * 100
            }
            
            if nan_raporu['toplam_nan'] > 0:
                st.warning(f"⚠️ **{nan_raporu['toplam_nan']:,} NaN değer bulundu** (%{nan_raporu['nan_oran']:.1f})")
                
                # NaN stratejisi seçimi
                nan_strategisi = st.selectbox(
                    "NaN değerleri nasıl işlemek istersiniz?",
                    ['Görmezden gel', 'Sütun bazlı doldur', 'İleri doldurma', 'KNN ile doldur', 'Sil'],
                    key='nan_strategisi'
                )
                
                if nan_strategisi == 'Sütun bazlı doldur':
                    for sutun in df.columns:
                        if df[sutun].isna().any():
                            if df[sutun].dtype == 'object' or df[sutun].dtype.name == 'category':
                                df[sutun] = df[sutun].fillna(df[sutun].mode()[0] if not df[sutun].mode().empty else 'Bilinmiyor')
                            elif np.issubdtype(df[sutun].dtype, np.number):
                                df[sutun] = df[sutun].fillna(df[sutun].median())
                
                elif nan_strategisi == 'İleri doldurma':
                    df = df.ffill().bfill()
                
                elif nan_strategisi == 'Sil':
                    orijinal_len = len(df)
                    df = df.dropna()
                    silinen = orijinal_len - len(df)
                    st.info(f"🗑️ {silinen:,} satır silindi (NaN içeren)")
            
            return df
        except Exception as e:
            st.warning(f"NaN işleme hatası: {str(e)}")
            return df
    
    @staticmethod
    def outlier_islemleri(df):
        """Outlier tespiti ve işleme"""
        try:
            sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns
            
            if len(sayisal_sutunlar) > 0:
                outlier_raporu = {}
                
                for sutun in sayisal_sutunlar[:10]:  # İlk 10 sayısal sütun
                    try:
                        Q1 = df[sutun].quantile(0.25)
                        Q3 = df[sutun].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        alt_sinir = Q1 - 1.5 * IQR
                        ust_sinir = Q3 + 1.5 * IQR
                        
                        outlier_count = ((df[sutun] < alt_sinir) | (df[sutun] > ust_sinir)).sum()
                        outlier_oran = (outlier_count / len(df)) * 100
                        
                        if outlier_count > 0:
                            outlier_raporu[sutun] = {
                                'count': outlier_count,
                                'ratio': outlier_oran,
                                'lower': alt_sinir,
                                'upper': ust_sinir
                            }
                            
                    except:
                        pass
                
                if outlier_raporu:
                    st.warning(f"⚠️ **Outlier tespit edildi**")
                    
                    for sutun, degerler in list(outlier_raporu.items())[:5]:
                        st.write(f"• **{sutun}:** {degerler['count']:,} outlier (%{degerler['ratio']:.1f})")
            
            return df
        except Exception as e:
            st.warning(f"Outlier işleme hatası: {str(e)}")
            return df
    
    def analiz_verisi_hazirla(self, df):
        """Analiz için gerekli verileri hazırla"""
        try:
            with st.spinner("📊 **Analiz verileri hazırlanıyor...**"):
                # 1. Satış trend analizi
                df = self.satis_trend_analizi_hazirla(df)
                
                # 2. Büyüme oranları
                df = self.buyume_oranlari_hazirla(df)
                
                # 3. Pazar payı hesaplama
                df = self.pazar_payi_hesapla(df)
                
                # 4. Fiyat analizi
                df = self.fiyat_analizi_hazirla(df)
                
                # 5. Performans metrikleri
                df = self.performans_metrikleri_hazirla(df)
                
                # 6. International Product analizi
                df = self.international_product_analizi_hazirla(df)
                
                # 7. Rekabet analizi
                df = self.rekabet_analizi_hazirla(df)
                
                # 8. Risk skorları
                df = self.risk_skorlari_hesapla(df)
                
                # 9. Tahmin edici değişkenler
                df = self.tahmin_edici_degiskenler_hazirla(df)
                
                # 10. Segmentasyon özellikleri
                df = self.segmentasyon_ozellikleri_hazirla(df)
            
            st.success(f"✅ **Analiz verileri hazırlandı:** {len(df.columns)} sütun")
            return df
            
        except Exception as e:
            st.error(f"Analiz verisi hazırlama hatası: {str(e)}")
            return df
    
    def satis_trend_analizi_hazirla(self, df):
        """Satış trend analizi için verileri hazırla"""
        try:
            satis_sutunlari = [s for s in df.columns if 'Satış' in s or 'Sales' in s]
            
            if len(satis_sutunlari) >= 2:
                satis_sutunlari = sorted(satis_sutunlari)
                
                for i in range(1, len(satis_sutunlari)):
                    onceki_sutun = satis_sutunlari[i-1]
                    simdiki_sutun = satis_sutunlari[i]
                    
                    # Yıllık büyüme
                    buyume_sutun_adi = f'Yıllık_Büyüme_{onceki_sutun.split("_")[-1]}_{simdiki_sutun.split("_")[-1]}'
                    df[buyume_sutun_adi] = ((df[simdiki_sutun] - df[onceki_sutun]) / 
                                           df[onceki_sutun].replace(0, np.nan)) * 100
                    
                    # Mutlak büyüme
                    mutlak_buyume_adi = f'Mutlak_Büyüme_{onceki_sutun.split("_")[-1]}_{simdiki_sutun.split("_")[-1]}'
                    df[mutlak_buyume_adi] = df[simdiki_sutun] - df[onceki_sutun]
                
                # CAGR hesapla
                if len(satis_sutunlari) >= 2:
                    ilk_sutun = satis_sutunlari[0]
                    son_sutun = satis_sutunlari[-1]
                    yil_farki = int(son_sutun.split('_')[-1]) - int(ilk_sutun.split('_')[-1])
                    
                    if yil_farki > 0:
                        df['CAGR'] = ((df[son_sutun] / df[ilk_sutun].replace(0, np.nan)) ** 
                                     (1/yil_farki) - 1) * 100
            
            return df
        except Exception as e:
            st.warning(f"Satış trend hazırlama hatası: {str(e)}")
            return df
    
    def buyume_oranlari_hazirla(self, df):
        """Büyüme oranlarını hazırla"""
        try:
            buyume_sutunlari = [s for s in df.columns if 'Büyüme' in s or 'Growth' in s]
            
            if buyume_sutunlari:
                # Ortalama büyüme
                df['Ortalama_Büyüme'] = df[buyume_sutunlari].mean(axis=1, skipna=True)
                
                # Büyüme volatilitesi
                df['Büyüme_Volatilitesi'] = df[buyume_sutunlari].std(axis=1, skipna=True)
                
                # Büyüme trendi (regresyon eğimi)
                for idx in df.index:
                    try:
                        buyume_degerleri = df.loc[idx, buyume_sutunlari].values
                        if len(buyume_degerleri) >= 3:
                            x = np.arange(len(buyume_degerleri))
                            slope, intercept = np.polyfit(x, buyume_degerleri, 1)
                            df.at[idx, 'Büyüme_Trend_Eğimi'] = slope
                    except:
                        df.at[idx, 'Büyüme_Trend_Eğimi'] = np.nan
            
            return df
        except Exception as e:
            st.warning(f"Büyüme oranı hazırlama hatası: {str(e)}")
            return df
    
    def pazar_payi_hesapla(self, df):
        """Pazar payı hesaplamaları"""
        try:
            satis_sutunlari = [s for s in df.columns if 'Satış' in s or 'Sales' in s]
            
            if satis_sutunlari:
                son_satis_sutun = satis_sutunlari[-1]
                toplam_pazar = df[son_satis_sutun].sum()
                
                if toplam_pazar > 0:
                    # Global pazar payı
                    df['Global_Pazar_Payı'] = (df[son_satis_sutun] / toplam_pazar) * 100
                    
                    # Segment bazlı pazar payı
                    if 'Segment' in df.columns:
                        segment_paylari = df.groupby('Segment')[son_satis_sutun].transform('sum')
                        df['Segment_Pazar_Payı'] = (df[son_satis_sutun] / segment_paylari) * 100
                    
                    # Ülke bazlı pazar payı
                    if 'Ülke' in df.columns or 'Country' in df.columns:
                        ulke_sutun = 'Ülke' if 'Ülke' in df.columns else 'Country'
                        ulke_paylari = df.groupby(ulke_sutun)[son_satis_sutun].transform('sum')
                        df['Ülke_Pazar_Payı'] = (df[son_satis_sutun] / ulke_paylari) * 100
            
            return df
        except Exception as e:
            st.warning(f"Pazar payı hesaplama hatası: {str(e)}")
            return df
    
    def fiyat_analizi_hazirla(self, df):
        """Fiyat analizi için verileri hazırla"""
        try:
            fiyat_sutunlari = [s for s in df.columns if 'Fiyat' in s or 'Price' in s]
            hacim_sutunlari = [s for s in df.columns if 'Hacim' in s or 'Volume' in s or 'Miktar' in s or 'Quantity' in s]
            
            if fiyat_sutunlari and hacim_sutunlari:
                son_fiyat_sutun = fiyat_sutunlari[-1]
                son_hacim_sutun = hacim_sutunlari[-1]
                
                # Fiyat-Hacim çarpımı
                df['Fiyat_Hacim_Çarpımı'] = df[son_fiyat_sutun] * df[son_hacim_sutun]
                
                # Fiyat segmenti
                fiyat_quantile = df[son_fiyat_sutun].quantile([0.33, 0.67])
                df['Fiyat_Segmenti'] = pd.cut(df[son_fiyat_sutun],
                                            bins=[-np.inf, fiyat_quantile[0.33], fiyat_quantile[0.67], np.inf],
                                            labels=['Düşük', 'Orta', 'Yüksek'])
                
                # Fiyat esnekliği (basit korelasyon)
                if len(df) > 10:
                    try:
                        correlation = df[[son_fiyat_sutun, son_hacim_sutun]].corr().iloc[0,1]
                        df['Fiyat_Esneklik_Korelasyonu'] = correlation
                    except:
                        pass
            
            return df
        except Exception as e:
            st.warning(f"Fiyat analizi hazırlama hatası: {str(e)}")
            return df
    
    def performans_metrikleri_hazirla(self, df):
        """Performans metriklerini hesapla"""
        try:
            # Performans skoru (çok boyutlu)
            skor_bilesenleri = []
            
            # 1. Satış performansı
            satis_sutunlari = [s for s in df.columns if 'Satış' in s or 'Sales' in s]
            if satis_sutunlari:
                son_satis = satis_sutunlari[-1]
                satis_skor = (df[son_satis] - df[son_satis].mean()) / df[son_satis].std()
                skor_bilesenleri.append(satis_skor.fillna(0))
            
            # 2. Büyüme performansı
            buyume_sutunlari = [s for s in df.columns if 'Büyüme' in s and 'Ortalama' not in s]
            if buyume_sutunlari:
                son_buyume = buyume_sutunlari[-1]
                buyume_skor = (df[son_buyume] - df[son_buyume].mean()) / df[son_buyume].std()
                skor_bilesenleri.append(buyume_skor.fillna(0))
            
            # 3. Pazar payı performansı
            if 'Global_Pazar_Payı' in df.columns:
                pazar_payi_skor = (df['Global_Pazar_Payı'] - df['Global_Pazar_Payı'].mean()) / df['Global_Pazar_Payı'].std()
                skor_bilesenleri.append(pazar_payi_skor.fillna(0))
            
            if skor_bilesenleri:
                # Ağırlıklı ortalama (satış: 0.4, büyüme: 0.4, pazar payı: 0.2)
                agirliklar = [0.4, 0.4, 0.2][:len(skor_bilesenleri)]
                toplam_agirlik = sum(agirliklar)
                normalized_agirliklar = [w/toplam_agirlik for w in agirliklar]
                
                performans_skoru = pd.Series(0, index=df.index)
                for skor, agirlik in zip(skor_bilesenleri, normalized_agirliklar):
                    performans_skoru += skor * agirlik
                
                # Min-max scaling (0-100)
                df['Performans_Skoru'] = ((performans_skoru - performans_skoru.min()) / 
                                         (performans_skoru.max() - performans_skoru.min() + 1e-10)) * 100
            
            # Performans segmenti
            if 'Performans_Skoru' in df.columns:
                df['Performans_Segmenti'] = pd.qcut(df['Performans_Skoru'], 
                                                   q=5, 
                                                   labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek', 'Çok Yüksek'])
            
            return df
        except Exception as e:
            st.warning(f"Performans metrikleri hazırlama hatası: {str(e)}")
            return df
    
    def international_product_analizi_hazirla(self, df):
        """International Product analizi için verileri hazırla"""
        try:
            # Molekül ve coğrafi dağılım kontrolü
            molekul_sutun = None
            ulke_sutun = None
            sirket_sutun = None
            
            for sutun in df.columns:
                sutun_lower = str(sutun).lower()
                if 'molekül' in sutun_lower or 'molecule' in sutun_lower:
                    molekul_sutun = sutun
                elif 'ülke' in sutun_lower or 'country' in sutun_lower:
                    ulke_sutun = sutun
                elif 'şirket' in sutun_lower or 'corporation' in sutun_lower or 'company' in sutun_lower:
                    sirket_sutun = sutun
            
            if molekul_sutun and (ulke_sutun or sirket_sutun):
                # International Product tespiti
                international_data = []
                
                for molekul in df[molekul_sutun].unique():
                    molekul_df = df[df[molekul_sutun] == molekul]
                    
                    ulke_sayisi = molekul_df[ulke_sutun].nunique() if ulke_sutun else 0
                    sirket_sayisi = molekul_df[sirket_sutun].nunique() if sirket_sutun else 0
                    
                    # International kriteri
                    international_mi = (ulke_sayisi > 1 or sirket_sayisi > 1)
                    
                    # Satış verisi
                    satis_sutunlari = [s for s in df.columns if 'Satış' in s or 'Sales' in s]
                    toplam_satis = molekul_df[satis_sutunlari[-1]].sum() if satis_sutunlari else 0
                    
                    international_data.append({
                        'Molekül': molekul,
                        'International_Product': international_mi,
                        'Ülke_Sayısı': ulke_sayisi,
                        'Şirket_Sayısı': sirket_sayisi,
                        'Toplam_Satış': toplam_satis,
                        'Ürün_Çeşitliliği': len(molekul_df)
                    })
                
                international_df = pd.DataFrame(international_data)
                
                # International skoru
                international_df['International_Skoru'] = (
                    international_df['Ülke_Sayısı'] * 0.6 + 
                    international_df['Şirket_Sayısı'] * 0.4
                )
                
                # International segmenti
                international_df['International_Segmenti'] = pd.qcut(
                    international_df['International_Skoru'],
                    q=4,
                    labels=['Yerel', 'Bölgesel', 'Ulusal', 'Global']
                )
                
                # Ana dataframe'e merge et
                df = df.merge(international_df[[molekul_sutun, 'International_Product', 'International_Skoru', 'International_Segmenti']],
                             on=molekul_sutun, how='left')
            
            return df
        except Exception as e:
            st.warning(f"International Product analizi hazırlama hatası: {str(e)}")
            return df
    
    def rekabet_analizi_hazirla(self, df):
        """Rekabet analizi için verileri hazırla"""
        try:
            # Rekabet yoğunluğu
            if 'Şirket' in df.columns or 'Corporation' in df.columns:
                sirket_sutun = 'Şirket' if 'Şirket' in df.columns else 'Corporation'
                satis_sutunlari = [s for s in df.columns if 'Satış' in s or 'Sales' in s]
                
                if satis_sutunlari:
                    son_satis = satis_sutunlari[-1]
                    
                    # Her şirket için rekabet metrikleri
                    sirket_metrikleri = df.groupby(sirket_sutun).agg({
                        son_satis: ['sum', 'count', 'mean']
                    }).round(2)
                    
                    sirket_metrikleri.columns = ['Toplam_Satış', 'Ürün_Sayısı', 'Ortalama_Satış']
                    
                    # Pazar payı
                    toplam_pazar = sirket_metrikleri['Toplam_Satış'].sum()
                    sirket_metrikleri['Pazar_Payı'] = (sirket_metrikleri['Toplam_Satış'] / toplam_pazar) * 100
                    
                    # Rekabet skoru (düşük pazar payı = yüksek rekabet)
                    sirket_metrikleri['Rekabet_Skoru'] = 100 - sirket_metrikleri['Pazar_Payı']
                    
                    # Ana dataframe'e merge et
                    df = df.merge(sirket_metrikleri[['Pazar_Payı', 'Rekabet_Skoru']], 
                                 left_on=sirket_sutun, right_index=True, how='left')
            
            return df
        except Exception as e:
            st.warning(f"Rekabet analizi hazırlama hatası: {str(e)}")
            return df
    
    def risk_skorlari_hesapla(self, df):
        """Risk skorlarını hesapla"""
        try:
            risk_bilesenleri = []
            
            # 1. Büyüme volatilitesi riski
            if 'Büyüme_Volatilitesi' in df.columns:
                buyume_risk = (df['Büyüme_Volatilitesi'] - df['Büyüme_Volatilitesi'].min()) / \
                             (df['Büyüme_Volatilitesi'].max() - df['Büyüme_Volatilitesi'].min() + 1e-10)
                risk_bilesenleri.append(buyume_risk)
            
            # 2. Pazar payı değişkenliği riski
            if 'Global_Pazar_Payı' in df.columns:
                pazar_payi_degiskenlik = df['Global_Pazar_Payı'].rolling(window=3, min_periods=1).std()
                pazar_risk = (pazar_payi_degiskenlik - pazar_payi_degiskenlik.min()) / \
                            (pazar_payi_degiskenlik.max() - pazar_payi_degiskenlik.min() + 1e-10)
                risk_bilesenleri.append(pazar_risk.fillna(0))
            
            # 3. Rekabet riski
            if 'Rekabet_Skoru' in df.columns:
                rekabet_risk = df['Rekabet_Skoru'] / 100
                risk_bilesenleri.append(rekabet_risk)
            
            if risk_bilesenleri:
                risk_skoru = pd.Series(0, index=df.index)
                for risk in risk_bilesenleri:
                    risk_skoru += risk
                
                risk_skoru = risk_skoru / len(risk_bilesenleri)
                df['Risk_Skoru'] = risk_skoru * 100
                
                # Risk segmenti
                df['Risk_Segmenti'] = pd.qcut(df['Risk_Skoru'], 
                                             q=4, 
                                             labels=['Düşük Risk', 'Orta Risk', 'Yüksek Risk', 'Çok Yüksek Risk'])
            
            return df
        except Exception as e:
            st.warning(f"Risk skoru hesaplama hatası: {str(e)}")
            return df
    
    def tahmin_edici_degiskenler_hazirla(self, df):
        """Tahmin modelleri için değişkenleri hazırla"""
        try:
            # Zaman serisi değişkenleri
            if 'Tarih' in df.columns:
                df['Ay'] = df['Tarih'].dt.month
                df['Yıl'] = df['Tarih'].dt.year
                df['Çeyrek'] = df['Tarih'].dt.quarter
                df['Hafta_Günü'] = df['Tarih'].dt.dayofweek
                df['Ay_Günü'] = df['Tarih'].dt.day
            
            # Lag değişkenleri (gecikmeli değişkenler)
            sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns
            for sutun in sayisal_sutunlar[:5]:  # İlk 5 sayısal sütun
                for lag in [1, 2, 3]:
                    df[f'{sutun}_Lag_{lag}'] = df[sutun].shift(lag)
            
            # Hareketli ortalamalar
            for sutun in sayisal_sutunlar[:3]:
                df[f'{sutun}_MA_7'] = df[sutun].rolling(window=7, min_periods=1).mean()
                df[f'{sutun}_MA_30'] = df[sutun].rolling(window=30, min_periods=1).mean()
            
            # Değişim oranları
            for sutun in sayisal_sutunlar[:3]:
                df[f'{sutun}_Değişim'] = df[sutun].pct_change() * 100
            
            return df
        except Exception as e:
            st.warning(f"Tahmin değişkenleri hazırlama hatası: {str(e)}")
            return df
    
    def segmentasyon_ozellikleri_hazirla(self, df):
        """Segmentasyon için özellikleri hazırla"""
        try:
            # Segmentasyon özellik matrisi
            segmentasyon_ozellikleri = []
            
            # 1. Satış özellikleri
            satis_sutunlari = [s for s in df.columns if 'Satış' in s or 'Sales' in s]
            if satis_sutunlari:
                segmentasyon_ozellikleri.extend(satis_sutunlari[-2:])  # Son 2 yıl
            
            # 2. Büyüme özellikleri
            buyume_sutunlari = [s for s in df.columns if 'Büyüme' in s and 'Ortalama' not in s]
            if buyume_sutunlari:
                segmentasyon_ozellikleri.extend(buyume_sutunlari[-1:])  # Son büyüme
            
            # 3. Fiyat özellikleri
            fiyat_sutunlari = [s for s in df.columns if 'Fiyat' in s or 'Price' in s]
            if fiyat_sutunlari:
                segmentasyon_ozellikleri.extend(fiyat_sutunlari[-1:])
            
            # 4. Pazar payı
            if 'Global_Pazar_Payı' in df.columns:
                segmentasyon_ozellikleri.append('Global_Pazar_Payı')
            
            # 5. International skoru
            if 'International_Skoru' in df.columns:
                segmentasyon_ozellikleri.append('International_Skoru')
            
            # 6. Performans skoru
            if 'Performans_Skoru' in df.columns:
                segmentasyon_ozellikleri.append('Performans_Skoru')
            
            # 7. Risk skoru
            if 'Risk_Skoru' in df.columns:
                segmentasyon_ozellikleri.append('Risk_Skoru')
            
            # Özellik matrisini kaydet
            if segmentasyon_ozellikleri:
                st.session_state.segmentasyon_ozellikleri = segmentasyon_ozellikleri
                st.info(f"📊 **{len(segmentasyon_ozellikleri)} segmentasyon özelliği hazırlandı**")
            
            return df
        except Exception as e:
            st.warning(f"Segmentasyon özellikleri hazırlama hatası: {str(e)}")
            return df
    
    def veri_kalitesi_raporu(self, df):
        """Veri kalitesi raporu oluştur"""
        try:
            rapor = {
                'Genel_İstatistikler': {
                    'Toplam_Satır': len(df),
                    'Toplam_Sütun': len(df.columns),
                    'Bellek_Kullanımı_MB': df.memory_usage(deep=True).sum() / 1024**2,
                    'Tekil_Satır': df.drop_duplicates().shape[0],
                    'Tekrar_Oranı': ((len(df) - df.drop_duplicates().shape[0]) / len(df)) * 100 if len(df) > 0 else 0
                },
                'Veri_Tipleri': df.dtypes.value_counts().to_dict(),
                'NaN_Analizi': {
                    'Toplam_NaN': df.isna().sum().sum(),
                    'NaN_Oranı': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'En_Fazla_NaN_Sütun': df.isna().sum().idxmax() if df.isna().sum().max() > 0 else None,
                    'En_Fazla_NaN_Sayısı': df.isna().sum().max()
                },
                'Sütun_Analizi': {}
            }
            
            # Her sütun için detaylı analiz
            for sutun in df.columns:
                sutun_raporu = {
                    'Veri_Tipi': str(df[sutun].dtype),
                    'Benzersiz_Değer': df[sutun].nunique(),
                    'NaN_Sayısı': df[sutun].isna().sum(),
                    'NaN_Oranı': (df[sutun].isna().sum() / len(df)) * 100,
                    'Örnek_Değerler': df[sutun].dropna().unique()[:5].tolist() if df[sutun].nunique() <= 10 else 'Çok Fazla'
                }
                
                if np.issubdtype(df[sutun].dtype, np.number):
                    sutun_raporu.update({
                        'Min': float(df[sutun].min()),
                        'Max': float(df[sutun].max()),
                        'Ortalama': float(df[sutun].mean()),
                        'Medyan': float(df[sutun].median()),
                        'Standart_Sapma': float(df[sutun].std()),
                        'Çeyrekler': {
                            'Q1': float(df[sutun].quantile(0.25)),
                            'Q2': float(df[sutun].quantile(0.50)),
                            'Q3': float(df[sutun].quantile(0.75))
                        }
                    })
                
                rapor['Sütun_Analizi'][sutun] = sutun_raporu
            
            return rapor
            
        except Exception as e:
            st.error(f"Veri kalitesi raporu hatası: {str(e)}")
            return {}

# ================================================
# 3. ENTERPRISE FİLTRELEME SİSTEMİ - 400+ SATIR
# ================================================

class EnterpriseFiltreSistemi:
    """Enterprise-level gelişmiş filtreleme sistemi"""
    
    def __init__(self):
        self.filtre_gecmisi = []
        self.kayitli_filtreler = {}
        self.filtre_templates = {
            'yuksek_buyume': {'min_buyume': 20, 'min_satis': 100000},
            'riskli_urunler': {'max_buyume': -10, 'min_risk': 70},
            'international_urunler': {'min_ulke': 2, 'min_sirket': 2},
            'premium_urunler': {'min_fiyat': 100, 'min_performans': 70}
        }
    
    def filtre_paneli_olustur(self, df):
        """Enterprise filtreleme paneli oluştur"""
        with st.sidebar:
            st.markdown('<div class="enterprise-sidebar-title">🎛️ ENTERPRISE FİLTRELEME</div>', unsafe_allow_html=True)
            
            # Filtre geçmişi
            if self.filtre_gecmisi:
                with st.expander("📜 **Filtre Geçmişi**", expanded=False):
                    for i, filtre in enumerate(reversed(self.filtre_gecmisi[-5:])):
                        st.caption(f"{i+1}. {filtre['isim']} - {filtre['tarih']}")
            
            # Ana filtre paneli
            with st.expander("🎯 **TEMEL FİLTRELER**", expanded=True):
                filtre_config = self.temel_filtreler(df)
            
            with st.expander("📊 **İLERİ FİLTRELER**", expanded=False):
                filtre_config.update(self.ileri_filtreler(df))
            
            with st.expander("🔍 **ÖZEL FİLTRELER**", expanded=False):
                filtre_config.update(self.ozel_filtreler(df))
            
            with st.expander("💾 **FİLTRE YÖNETİMİ**", expanded=False):
                self.filtre_yonetimi(df, filtre_config)
            
            return filtre_config
    
    def temel_filtreler(self, df):
        """Temel filtreler"""
        filtre_config = {}
        
        # Global arama
        st.markdown('<div class="enterprise-filter-title">🔍 Global Arama</div>', unsafe_allow_html=True)
        arama_terimi = st.text_input(
            "Anahtar kelime ara",
            placeholder="Molekül, Şirket, Ülke...",
            help="Tüm sütunlarda arama yapın",
            key="global_arama_enterprise"
        )
        if arama_terimi:
            filtre_config['arama_terimi'] = arama_terimi
        
        # Kategorik filtreler
        st.markdown('<div class="enterprise-filter-title">🏷️ Kategorik Filtreler</div>', unsafe_allow_html=True)
        
        # Molekül filtreleme
        molekul_sutun = next((s for s in df.columns if 'molekül' in str(s).lower() or 'molecule' in str(s).lower()), None)
        if molekul_sutun:
            secilen_molekuller = self.akilli_coklu_secim(
                "🧬 Moleküller",
                df[molekul_sutun].dropna().unique(),
                key="molekuller_filtre"
            )
            if secilen_molekuller:
                filtre_config['molekuller'] = (molekul_sutun, secilen_molekuller)
        
        # Şirket filtreleme
        sirket_sutun = next((s for s in df.columns if 'şirket' in str(s).lower() or 'corporation' in str(s).lower() or 'company' in str(s).lower()), None)
        if sirket_sutun:
            secilen_sirketler = self.akilli_coklu_secim(
                "🏢 Şirketler",
                df[sirket_sutun].dropna().unique(),
                key="sirketler_filtre"
            )
            if secilen_sirketler:
                filtre_config['sirketler'] = (sirket_sutun, secilen_sirketler)
        
        # Ülke filtreleme
        ulke_sutun = next((s for s in df.columns if 'ülke' in str(s).lower() or 'country' in str(s).lower()), None)
        if ulke_sutun:
            secilen_ulkeler = self.akilli_coklu_secim(
                "🌍 Ülkeler",
                df[ulke_sutun].dropna().unique(),
                key="ulkeler_filtre"
            )
            if secilen_ulkeler:
                filtre_config['ulkeler'] = (ulke_sutun, secilen_ulkeler)
        
        return filtre_config
    
    def ileri_filtreler(self, df):
        """İleri seviye filtreler"""
        filtre_config = {}
        
        st.markdown('<div class="enterprise-filter-title">📈 Sayısal Filtreler</div>', unsafe_allow_html=True)
        
        # Satış filtreleri
        satis_sutunlari = [s for s in df.columns if 'satış' in str(s).lower() or 'sales' in str(s).lower()]
        if satis_sutunlari:
            son_satis = satis_sutunlari[-1]
            min_satis = float(df[son_satis].min())
            max_satis = float(df[son_satis].max())
            
            col1, col2 = st.columns(2)
            with col1:
                satis_min = st.number_input(
                    "Min Satış",
                    min_value=min_satis,
                    max_value=max_satis,
                    value=min_satis,
                    step=(max_satis - min_satis) / 100,
                    key="satis_min"
                )
            with col2:
                satis_max = st.number_input(
                    "Max Satış",
                    min_value=min_satis,
                    max_value=max_satis,
                    value=max_satis,
                    step=(max_satis - min_satis) / 100,
                    key="satis_max"
                )
            
            if satis_min != min_satis or satis_max != max_satis:
                filtre_config['satis_araligi'] = (son_satis, satis_min, satis_max)
        
        # Büyüme filtreleri
        buyume_sutunlari = [s for s in df.columns if 'büyüme' in str(s).lower() or 'growth' in str(s).lower()]
        if buyume_sutunlari:
            son_buyume = buyume_sutunlari[-1]
            min_buyume = float(df[son_buyume].min())
            max_buyume = float(df[son_buyume].max())
            
            col1, col2 = st.columns(2)
            with col1:
                buyume_min = st.number_input(
                    "Min Büyüme (%)",
                    min_value=min_buyume,
                    max_value=max_buyume,
                    value=min(min_buyume, -50),
                    step=5.0,
                    key="buyume_min"
                )
            with col2:
                buyume_max = st.number_input(
                    "Max Büyüme (%)",
                    min_value=min_buyume,
                    max_value=max_buyume,
                    value=max(max_buyume, 150),
                    step=5.0,
                    key="buyume_max"
                )
            
            if buyume_min != min_buyume or buyume_max != max_buyume:
                filtre_config['buyume_araligi'] = (son_buyume, buyume_min, buyume_max)
        
        # Fiyat filtreleri
        fiyat_sutunlari = [s for s in df.columns if 'fiyat' in str(s).lower() or 'price' in str(s).lower()]
        if fiyat_sutunlari:
            son_fiyat = fiyat_sutunlari[-1]
            min_fiyat = float(df[son_fiyat].min())
            max_fiyat = float(df[son_fiyat].max())
            
            fiyat_araligi = st.slider(
                "Fiyat Aralığı",
                min_value=min_fiyat,
                max_value=max_fiyat,
                value=(min_fiyat, max_fiyat),
                key="fiyat_araligi"
            )
            
            if fiyat_araligi[0] != min_fiyat or fiyat_araligi[1] != max_fiyat:
                filtre_config['fiyat_araligi'] = (son_fiyat, fiyat_araligi[0], fiyat_araligi[1])
        
        return filtre_config
    
    def ozel_filtreler(self, df):
        """Özel filtreler"""
        filtre_config = {}
        
        st.markdown('<div class="enterprise-filter-title">🎯 Özel Filtreler</div>', unsafe_allow_html=True)
        
        # International Product filtreleri
        if 'International_Product' in df.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.checkbox("🌍 Sadece International", key="sadece_international"):
                    filtre_config['sadece_international'] = True
            with col2:
                if st.checkbox("🏠 Sadece Yerel", key="sadece_yerel"):
                    filtre_config['sadece_yerel'] = True
        
        # Performans filtreleri
        if 'Performans_Skoru' in df.columns:
            performans_seviye = st.select_slider(
                "Performans Seviyesi",
                options=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek', 'Çok Yüksek'],
                value=('Çok Düşük', 'Çok Yüksek'),
                key="performans_seviye"
            )
            
            if performans_seviye != ('Çok Düşük', 'Çok Yüksek'):
                filtre_config['performans_seviye'] = performans_seviye
        
        # Risk filtreleri
        if 'Risk_Skoru' in df.columns:
            risk_seviye = st.select_slider(
                "Risk Seviyesi",
                options=['Düşük Risk', 'Orta Risk', 'Yüksek Risk', 'Çok Yüksek Risk'],
                value=('Düşük Risk', 'Çok Yüksek Risk'),
                key="risk_seviye"
            )
            
            if risk_seviye != ('Düşük Risk', 'Çok Yüksek Risk'):
                filtre_config['risk_seviye'] = risk_seviye
        
        # Trend filtreleri
        if 'Büyüme_Trend_Eğimi' in df.columns:
            trend_yonu = st.radio(
                "Büyüme Trendi",
                ['Tümü', 'Yükselen', 'Düşen', 'Stabil'],
                key="trend_yonu"
            )
            
            if trend_yonu != 'Tümü':
                filtre_config['trend_yonu'] = trend_yonu
        
        return filtre_config
    
    def filtre_yonetimi(self, df, filtre_config):
        """Filtre yönetimi"""
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ **Filtre Uygula**", type="primary", use_container_width=True):
                self.filtre_gecmisi.append({
                    'isim': 'Manuel Filtre',
                    'tarih': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'config': filtre_config
                })
                st.session_state.filtre_config = filtre_config
                st.rerun()
        
        with col2:
            if st.button("🗑️ **Filtreleri Temizle**", use_container_width=True):
                st.session_state.filtre_config = {}
                st.rerun()
        
        # Filtre şablonları
        st.markdown("---")
        st.markdown("**💾 Filtre Şablonları**")
        
        selected_template = st.selectbox(
            "Şablon Seçin",
            list(self.filtre_templates.keys()),
            key="filtre_template"
        )
        
        if st.button("📋 **Şablonu Uygula**", use_container_width=True):
            template_config = self.filtre_templates[selected_template]
            # Template'e özel filtreleri uygula
            st.session_state.filtre_config = template_config
            st.rerun()
        
        # Filtreyi kaydet
        st.markdown("---")
        filtre_adi = st.text_input("Filtre Adı", placeholder="Özel filtre adı")
        
        if st.button("💾 **Filtreyi Kaydet**", use_container_width=True) and filtre_adi:
            self.kayitli_filtreler[filtre_adi] = {
                'config': filtre_config,
                'tarih': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.success(f"✅ '{filtre_adi}' filtresi kaydedildi!")
    
    def akilli_coklu_secim(self, baslik, secenekler, key):
        """Akıllı çoklu seçim bileşeni"""
        if not len(secenekler):
            return []
        
        # Arama kutusu
        arama = st.text_input(f"{baslik} Ara", key=f"{key}_arama")
        
        if arama:
            filtrelenmis = [opt for opt in secenekler if arama.lower() in str(opt).lower()]
        else:
            filtrelenmis = list(secenekler)
        
        # Gruplama (ilk harfe göre)
        if len(filtrelenmis) > 50:
            gruplar = {}
            for opt in filtrelenmis:
                ilk_harf = str(opt)[0].upper() if str(opt) else '#'
                if ilk_harf not in gruplar:
                    gruplar[ilk_harf] = []
                gruplar[ilk_harf].append(opt)
            
            secilenler = []
            for harf, grup in sorted(gruplar.items()):
                with st.expander(f"**{harf}** ({len(grup)})", expanded=False):
                    grup_secilen = st.multiselect(
                        f"{baslik} - {harf}",
                        options=grup,
                        default=[],
                        key=f"{key}_{harf}"
                    )
                    secilenler.extend(grup_secilen)
        else:
            secilenler = st.multiselect(
                baslik,
                options=filtrelenmis,
                default=[],
                key=key
            )
        
        if secilenler:
            st.caption(f"✅ {len(secilenler)} / {len(secenekler)} seçildi")
        
        return secilenler
    
    def filtreleri_uygula(self, df, filtre_config):
        """Filtreleri dataframe'e uygula"""
        if not filtre_config:
            return df
        
        filtrelenmis_df = df.copy()
        filtre_sayaci = 0
        
        # Arama terimi
        if 'arama_terimi' in filtre_config:
            arama_mask = pd.Series(False, index=filtrelenmis_df.index)
            for sutun in filtrelenmis_df.columns:
                try:
                    arama_mask = arama_mask | filtrelenmis_df[sutun].astype(str).str.contains(
                        filtre_config['arama_terimi'], case=False, na=False
                    )
                except:
                    continue
            filtrelenmis_df = filtrelenmis_df[arama_mask]
            filtre_sayaci += 1
        
        # Kategorik filtreler
        for filtre_anahtar in ['molekuller', 'sirketler', 'ulkeler']:
            if filtre_anahtar in filtre_config:
                sutun, degerler = filtre_config[filtre_anahtar]
                if degerler:
                    filtrelenmis_df = filtrelenmis_df[filtrelenmis_df[sutun].isin(degerler)]
                    filtre_sayaci += 1
        
        # Sayısal aralık filtreleri
        for filtre_anahtar in ['satis_araligi', 'buyume_araligi', 'fiyat_araligi']:
            if filtre_anahtar in filtre_config:
                sutun, min_val, max_val = filtre_config[filtre_anahtar]
                filtrelenmis_df = filtrelenmis_df[
                    (filtrelenmis_df[sutun] >= min_val) & 
                    (filtrelenmis_df[sutun] <= max_val)
                ]
                filtre_sayaci += 1
        
        # Özel filtreler
        if 'sadece_international' in filtre_config:
            filtrelenmis_df = filtrelenmis_df[filtrelenmis_df['International_Product'] == True]
            filtre_sayaci += 1
        
        if 'sadece_yerel' in filtre_config:
            filtrelenmis_df = filtrelenmis_df[filtrelenmis_df['International_Product'] == False]
            filtre_sayaci += 1
        
        if 'performans_seviye' in filtre_config:
            min_seviye, max_seviye = filtre_config['performans_seviye']
            seviye_sirasi = {'Çok Düşük': 0, 'Düşük': 1, 'Orta': 2, 'Yüksek': 3, 'Çok Yüksek': 4}
            min_idx = seviye_sirasi[min_seviye]
            max_idx = seviye_sirasi[max_seviye]
            
            filtrelenmis_df = filtrelenmis_df[
                filtrelenmis_df['Performans_Segmenti'].apply(
                    lambda x: min_idx <= seviye_sirasi.get(x, 0) <= max_idx
                )
            ]
            filtre_sayaci += 1
        
        if 'trend_yonu' in filtre_config:
            if filtre_config['trend_yonu'] == 'Yükselen':
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df['Büyüme_Trend_Eğimi'] > 0]
            elif filtre_config['trend_yonu'] == 'Düşen':
                filtrelenmis_df = filtrelenmis_df[filtrelenmis_df['Büyüme_Trend_Eğimi'] < 0]
            elif filtre_config['trend_yonu'] == 'Stabil':
                filtrelenmis_df = filtrelenmis_df[
                    (filtrelenmis_df['Büyüme_Trend_Eğimi'] >= -0.5) & 
                    (filtrelenmis_df['Büyüme_Trend_Eğimi'] <= 0.5)
                ]
            filtre_sayaci += 1
        
        # Filtre durumunu göster
        if filtre_sayaci > 0:
            st.markdown(f"""
            <div class="enterprise-filter-status">
                🎯 **{filtre_sayaci} aktif filtre** | 
                **{len(filtrelenmis_df):,} / {len(df):,}** satır gösteriliyor
            </div>
            """, unsafe_allow_html=True)
        
        return filtrelenmis_df

# ================================================
# 4. ENTERPRISE ANALİTİK MOTORU - 800+ SATIR
# ================================================

class EnterpriseAnalitikMotoru:
    """Enterprise-level analitik motoru"""
    
    def __init__(self):
        self.ml_models = {}
        self.analiz_cache = {}
        self.performance_metrics = {}
        
    def kapsamli_metrik_analizi(self, df):
        """Kapsamlı metrik analizi yap"""
        try:
            metrikler = {
                'genel_istatistikler': self.genel_istatistikler(df),
                'pazar_analizi': self.pazar_analizi_metrikleri(df),
                'buyume_analizi': self.buyume_analizi_metrikleri(df),
                'fiyat_analizi': self.fiyat_analizi_metrikleri(df),
                'rekabet_analizi': self.rekabet_analizi_metrikleri(df),
                'international_analiz': self.international_analiz_metrikleri(df),
                'risk_analizi': self.risk_analizi_metrikleri(df),
                'segment_analizi': self.segment_analizi_metrikleri(df)
            }
            
            # Toplam performans skoru
            metrikler['toplam_performans'] = self.toplam_performans_skoru(metrikler)
            
            return metrikler
            
        except Exception as e:
            st.error(f"Metrik analizi hatası: {str(e)}")
            return {}
    
    def genel_istatistikler(self, df):
        """Genel istatistikler"""
        try:
            return {
                'toplam_satir': len(df),
                'toplam_sutun': len(df.columns),
                'bellek_kullanimi_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'unique_molekul': df['Molekül'].nunique() if 'Molekül' in df.columns else 0,
                'unique_sirket': df['Şirket'].nunique() if 'Şirket' in df.columns else 0,
                'unique_ulke': df['Ülke'].nunique() if 'Ülke' in df.columns else 0,
                'ortalama_satis': float(df['Satış_2024'].mean()) if 'Satış_2024' in df.columns else 0,
                'toplam_satis': float(df['Satış_2024'].sum()) if 'Satış_2024' in df.columns else 0,
                'median_satis': float(df['Satış_2024'].median()) if 'Satış_2024' in df.columns else 0
            }
        except:
            return {}
    
    def pazar_analizi_metrikleri(self, df):
        """Pazar analizi metrikleri"""
        try:
            metrikler = {}
            
            if 'Satış_2024' in df.columns:
                toplam_pazar = df['Satış_2024'].sum()
                
                # Pazar konsantrasyonu
                if 'Şirket' in df.columns:
                    sirket_paylari = df.groupby('Şirket')['Satış_2024'].sum() / toplam_pazar * 100
                    metrikler['hhi_index'] = (sirket_paylari ** 2).sum() / 10000
                    
                    # CR4, CR8
                    for n in [4, 8]:
                        metrikler[f'cr{n}'] = sirket_paylari.nlargest(n).sum()
                
                # Segment konsantrasyonu
                if 'Segment' in df.columns:
                    segment_paylari = df.groupby('Segment')['Satış_2024'].sum() / toplam_pazar * 100
                    metrikler['segment_consantrasyon'] = segment_paylari.max()
                
                # Coğrafi konsantrasyon
                if 'Ülke' in df.columns:
                    ulke_paylari = df.groupby('Ülke')['Satış_2024'].sum() / toplam_pazar * 100
                    metrikler['top5_ulke_payi'] = ulke_paylari.nlargest(5).sum()
                    metrikler['en_buyuk_ulke_payi'] = ulke_paylari.max()
            
            return metrikler
        except:
            return {}
    
    def buyume_analizi_metrikleri(self, df):
        """Büyüme analizi metrikleri"""
        try:
            metrikler = {}
            
            buyume_sutunlari = [s for s in df.columns if 'Büyüme' in s and 'Ortalama' not in s]
            
            if buyume_sutunlari:
                son_buyume = buyume_sutunlari[-1]
                
                metrikler.update({
                    'ortalama_buyume': float(df[son_buyume].mean()),
                    'median_buyume': float(df[son_buyume].median()),
                    'buyume_std': float(df[son_buyume].std()),
                    'pozitif_buyume_orani': float((df[son_buyume] > 0).mean() * 100),
                    'yuksek_buyume_orani': float((df[son_buyume] > 20).mean() * 100),
                    'negatif_buyume_orani': float((df[son_buyume] < 0).mean() * 100)
                })
                
                # Büyüme korelasyonları
                if 'Satış_2024' in df.columns:
                    correlation = df[['Satış_2024', son_buyume]].corr().iloc[0,1]
                    metrikler['satis_buyume_korelasyon'] = float(correlation)
            
            return metrikler
        except:
            return {}
    
    def fiyat_analizi_metrikleri(self, df):
        """Fiyat analizi metrikleri"""
        try:
            metrikler = {}
            
            fiyat_sutunlari = [s for s in df.columns if 'Fiyat' in s or 'Price' in s]
            
            if fiyat_sutunlari:
                son_fiyat = fiyat_sutunlari[-1]
                
                metrikler.update({
                    'ortalama_fiyat': float(df[son_fiyat].mean()),
                    'median_fiyat': float(df[son_fiyat].median()),
                    'fiyat_std': float(df[son_fiyat].std()),
                    'fiyat_cv': float((df[son_fiyat].std() / df[son_fiyat].mean()) * 100) if df[son_fiyat].mean() > 0 else 0,
                    'min_fiyat': float(df[son_fiyat].min()),
                    'max_fiyat': float(df[son_fiyat].max()),
                    'fiyat_araligi': float(df[son_fiyat].max() - df[son_fiyat].min())
                })
                
                # Fiyat segment dağılımı
                if 'Fiyat_Segmenti' in df.columns:
                    segment_dagilim = df['Fiyat_Segmenti'].value_counts(normalize=True) * 100
                    for segment, oran in segment_dagilim.items():
                        metrikler[f'fiyat_segment_{segment}_orani'] = float(oran)
            
            return metrikler
        except:
            return {}
    
    def rekabet_analizi_metrikleri(self, df):
        """Rekabet analizi metrikleri"""
        try:
            metrikler = {}
            
            if 'Şirket' in df.columns and 'Satış_2024' in df.columns:
                sirket_analizi = df.groupby('Şirket').agg({
                    'Satış_2024': ['sum', 'count', 'mean', 'std']
                }).round(2)
                
                sirket_analizi.columns = ['toplam_satis', 'urun_sayisi', 'ortalama_satis', 'satis_std']
                
                # Rekabet metrikleri
                metrikler.update({
                    'sirket_sayisi': len(sirket_analizi),
                    'ortalama_urun_sirket': float(sirket_analizi['urun_sayisi'].mean()),
                    'max_pazar_payi': float((sirket_analizi['toplam_satis'] / sirket_analizi['toplam_satis'].sum()).max() * 100),
                    'min_pazar_payi': float((sirket_analizi['toplam_satis'] / sirket_analizi['toplam_satis'].sum()).min() * 100),
                    'pazar_payi_gini': float(self.gini_katsayisi(sirket_analizi['toplam_satis']))
                })
            
            return metrikler
        except:
            return {}
    
    def international_analiz_metrikleri(self, df):
        """International analiz metrikleri"""
        try:
            metrikler = {}
            
            if 'International_Product' in df.columns:
                intl_df = df[df['International_Product'] == True]
                local_df = df[df['International_Product'] == False]
                
                metrikler.update({
                    'international_urun_sayisi': len(intl_df),
                    'local_urun_sayisi': len(local_df),
                    'international_orani': float(len(intl_df) / len(df) * 100) if len(df) > 0 else 0,
                    
                    # Satış karşılaştırması
                    'international_satis_payi': float(intl_df['Satış_2024'].sum() / df['Satış_2024'].sum() * 100) if 'Satış_2024' in df.columns else 0,
                    'international_ortalama_satis': float(intl_df['Satış_2024'].mean()) if 'Satış_2024' in intl_df.columns else 0,
                    'local_ortalama_satis': float(local_df['Satış_2024'].mean()) if 'Satış_2024' in local_df.columns else 0,
                    
                    # Büyüme karşılaştırması
                    'international_ortalama_buyume': float(intl_df['Yıllık_Büyüme'].mean()) if 'Yıllık_Büyüme' in intl_df.columns else 0,
                    'local_ortalama_buyume': float(local_df['Yıllık_Büyüme'].mean()) if 'Yıllık_Büyüme' in local_df.columns else 0,
                    
                    # Fiyat karşılaştırması
                    'international_ortalama_fiyat': float(intl_df['Fiyat'].mean()) if 'Fiyat' in intl_df.columns else 0,
                    'local_ortalama_fiyat': float(local_df['Fiyat'].mean()) if 'Fiyat' in local_df.columns else 0
                })
                
                # International segment dağılımı
                if 'International_Segmenti' in df.columns:
                    segment_dagilim = df['International_Segmenti'].value_counts(normalize=True) * 100
                    for segment, oran in segment_dagilim.items():
                        metrikler[f'intl_segment_{segment}_orani'] = float(oran)
            
            return metrikler
        except:
            return {}
    
    def risk_analizi_metrikleri(self, df):
        """Risk analizi metrikleri"""
        try:
            metrikler = {}
            
            if 'Risk_Skoru' in df.columns:
                metrikler.update({
                    'ortalama_risk': float(df['Risk_Skoru'].mean()),
                    'max_risk': float(df['Risk_Skoru'].max()),
                    'min_risk': float(df['Risk_Skoru'].min()),
                    'risk_std': float(df['Risk_Skoru'].std()),
                    'yuksek_risk_orani': float((df['Risk_Skoru'] > 70).mean() * 100),
                    'dusuk_risk_orani': float((df['Risk_Skoru'] < 30).mean() * 100)
                })
            
            # Volatilite analizi
            buyume_sutunlari = [s for s in df.columns if 'Büyüme' in s and 'Ortalama' not in s]
            if len(buyume_sutunlari) >= 2:
                volatilite = df[buyume_sutunlari].std(axis=1).mean()
                metrikler['ortalama_volatilite'] = float(volatilite)
            
            return metrikler
        except:
            return {}
    
    def segment_analizi_metrikleri(self, df):
        """Segment analizi metrikleri"""
        try:
            metrikler = {}
            
            segment_sutunlari = [s for s in df.columns if 'Segment' in s]
            
            for segment_sutun in segment_sutunlari:
                if segment_sutun in df.columns:
                    segment_dagilim = df[segment_sutun].value_counts(normalize=True) * 100
                    
                    for segment, oran in segment_dagilim.items():
                        metrikler[f'{segment_sutun}_{segment}_orani'] = float(oran)
                    
                    # Segment çeşitliliği (entropy)
                    entropy = -sum((p/100) * math.log(p/100 + 1e-10) for p in segment_dagilim.values())
                    metrikler[f'{segment_sutun}_entropy'] = float(entropy)
            
            return metrikler
        except:
            return {}
    
    def toplam_performans_skoru(self, metrikler):
        """Toplam performans skoru hesapla"""
        try:
            skor = 0
            agirliklar = []
            degerler = []
            
            # Pazar büyüklüğü (25%)
            if 'genel_istatistikler' in metrikler:
                pazar_buyuklugu = metrikler['genel_istatistikler'].get('toplam_satis', 0)
                # Normalize et (0-100)
                pazar_skor = min(pazar_buyuklugu / 1e9 * 10, 100)
                skor += pazar_skor * 0.25
                agirliklar.append(0.25)
                degerler.append(pazar_skor)
            
            # Büyüme performansı (20%)
            if 'buyume_analizi' in metrikler:
                buyume = metrikler['buyume_analizi'].get('ortalama_buyume', 0)
                buyume_skor = max(0, min(buyume, 100))  # %0-100 arası
                skor += buyume_skor * 0.20
                agirliklar.append(0.20)
                degerler.append(buyume_skor)
            
            # Rekabet avantajı (15%)
            if 'rekabet_analizi' in metrikler:
                hhi = metrikler['pazar_analizi'].get('hhi_index', 2500)
                # HHI düşükse rekabet yüksek, puan düşük
                rekabet_skor = max(0, 100 - (hhi / 25))
                skor += rekabet_skor * 0.15
                agirliklar.append(0.15)
                degerler.append(rekabet_skor)
            
            # International yayılım (15%)
            if 'international_analiz' in metrikler:
                intl_oran = metrikler['international_analiz'].get('international_orani', 0)
                intl_skor = min(intl_oran * 2, 100)  # %50 international = 100 puan
                skor += intl_skor * 0.15
                agirliklar.append(0.15)
                degerler.append(intl_skor)
            
            # Risk yönetimi (15%)
            if 'risk_analizi' in metrikler:
                risk = metrikler['risk_analizi'].get('ortalama_risk', 50)
                risk_skor = max(0, 100 - risk)  # Risk düşükse puan yüksek
                skor += risk_skor * 0.15
                agirliklar.append(0.15)
                degerler.append(risk_skor)
            
            # Fiyat stabilitesi (10%)
            if 'fiyat_analizi' in metrikler:
                fiyat_cv = metrikler['fiyat_analizi'].get('fiyat_cv', 50)
                fiyat_skor = max(0, 100 - fiyat_cv)  # CV düşükse stabil, puan yüksek
                skor += fiyat_skor * 0.10
                agirliklar.append(0.10)
                degerler.append(fiyat_skor)
            
            # Toplam skoru normalize et
            toplam_agirlik = sum(agirliklar)
            if toplam_agirlik > 0:
                skor = skor / toplam_agirlik
            
            return {
                'toplam_skor': float(skor),
                'bilesenler': dict(zip(['Pazar', 'Büyüme', 'Rekabet', 'International', 'Risk', 'Fiyat'], degerler)),
                'agirliklar': agirliklar
            }
            
        except:
            return {'toplam_skor': 0, 'bilesenler': {}, 'agirliklar': []}
    
    def gini_katsayisi(self, values):
        """Gini katsayısı hesapla"""
        try:
            values = np.sort(values)
            n = len(values)
            index = np.arange(1, n + 1)
            return (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
        except:
            return 0
    
    def gelismis_pazar_segmentasyonu(self, df, yontem='kmeans', kume_sayisi=4):
        """Gelişmiş pazar segmentasyonu"""
        try:
            with st.spinner("🔬 **Pazar segmentasyonu yapılıyor...**"):
                # Segmentasyon özelliklerini al
                ozellikler = st.session_state.get('segmentasyon_ozellikleri', [])
                
                if not ozellikler:
                    # Varsayılan özellikler
                    ozellikler = []
                    for sutun in ['Satış_2024', 'Yıllık_Büyüme', 'Fiyat', 'Global_Pazar_Payı']:
                        if sutun in df.columns:
                            ozellikler.append(sutun)
                
                if len(ozellikler) < 2:
                    st.warning("Segmentasyon için yeterli özellik bulunamadı")
                    return None
                
                # Veriyi hazırla
                X = df[ozellikler].fillna(0)
                
                # Ölçeklendir
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Segmentasyon modeli
                if yontem == 'kmeans':
                    model = KMeans(
                        n_clusters=kume_sayisi,
                        random_state=42,
                        n_init=20,
                        max_iter=500,
                        tol=1e-4,
                        algorithm='elkan'
                    )
                elif yontem == 'hierarchical':
                    model = AgglomerativeClustering(
                        n_clusters=kume_sayisi,
                        linkage='ward',
                        metric='euclidean'
                    )
                elif yontem == 'dbscan':
                    model = DBSCAN(
                        eps=0.5,
                        min_samples=10,
                        metric='euclidean'
                    )
                else:
                    model = KMeans(n_clusters=kume_sayisi, random_state=42)
                
                # Küme tahmini
                kumeler = model.fit_predict(X_scaled)
                
                # Sonuçları dataframe'e ekle
                sonuc_df = df.copy()
                sonuc_df['Küme'] = kumeler
                
                # Küme isimlendirme
                kume_istatistikleri = []
                for kume in np.unique(kumeler):
                    kume_df = sonuc_df[sonuc_df['Küme'] == kume]
                    
                    # Küme özellikleri
                    kume_ozellikleri = {
                        'kume': kume,
                        'urun_sayisi': len(kume_df),
                        'ortalama_satis': kume_df['Satış_2024'].mean() if 'Satış_2024' in kume_df.columns else 0,
                        'ortalama_buyume': kume_df['Yıllık_Büyüme'].mean() if 'Yıllık_Büyüme' in kume_df.columns else 0,
                        'ortalama_fiyat': kume_df['Fiyat'].mean() if 'Fiyat' in kume_df.columns else 0,
                        'ortalama_pazar_payi': kume_df['Global_Pazar_Payı'].mean() if 'Global_Pazar_Payı' in kume_df.columns else 0
                    }
                    
                    # Küme ismi belirle
                    if kume_ozellikleri['ortalama_buyume'] > 20 and kume_ozellikleri['ortalama_satis'] > kume_ozellikleri['ortalama_satis']:
                        kume_ismi = 'Yıldız Ürünler'
                    elif kume_ozellikleri['ortalama_buyume'] > 10:
                        kume_ismi = 'Gelişen Ürünler'
                    elif kume_ozellikleri['ortalama_pazar_payi'] > 5:
                        kume_ismi = 'Olgun Ürünler'
                    elif kume_ozellikleri['ortalama_buyume'] < 0:
                        kume_ismi = 'Düşen Ürünler'
                    else:
                        kume_ismi = f'Küme {kume}'
                    
                    kume_ozellikleri['isim'] = kume_ismi
                    kume_istatistikleri.append(kume_ozellikleri)
                
                # Küme isimlerini dataframe'e ekle
                kume_isim_map = {k['kume']: k['isim'] for k in kume_istatistikleri}
                sonuc_df['Küme_İsmi'] = sonuc_df['Küme'].map(kume_isim_map)
                
                # Segmentasyon kalitesi metrikleri
                if hasattr(model, 'inertia_'):
                    inertia = model.inertia_
                else:
                    inertia = None
                
                if len(np.unique(kumeler)) > 1:
                    try:
                        silhouette = silhouette_score(X_scaled, kumeler)
                        calinski = calinski_harabasz_score(X_scaled, kumeler)
                        davies = davies_bouldin_score(X_scaled, kumeler)
                    except:
                        silhouette = None
                        calinski = None
                        davies = None
                else:
                    silhouette = None
                    calinski = None
                    davies = None
                
                # Sonuçları döndür
                return {
                    'data': sonuc_df,
                    'kume_istatistikleri': kume_istatistikleri,
                    'segmentasyon_metrikleri': {
                        'inertia': inertia,
                        'silhouette_score': silhouette,
                        'calinski_score': calinski,
                        'davies_score': davies,
                        'kume_sayisi': len(np.unique(kumeler))
                    },
                    'ozellikler': ozellikler,
                    'model': model
                }
            
        except Exception as e:
            st.error(f"Segmentasyon hatası: {str(e)}")
            return None
    
    def satis_tahmini_modeli(self, df, tahmin_horizonu=12):
        """Satış tahmini modeli oluştur"""
        try:
            with st.spinner("🔮 **Satış tahmini modeli oluşturuluyor...**"):
                # Zaman serisi verisini hazırla
                if 'Tarih' not in df.columns or 'Satış' not in ''.join(df.columns):
                    st.warning("Tahmin için tarih ve satış verisi gereklidir")
                    return None
                
                # Satış sütununu bul
                satis_sutun = next((s for s in df.columns if 'satış' in str(s).lower()), None)
                
                if not satis_sutun:
                    return None
                
                # Zaman serisi verisi
                ts_data = df[['Tarih', satis_sutun]].copy()
                ts_data = ts_data.set_index('Tarih')
                ts_data = ts_data.resample('M').sum()  # Aylık toplam
                
                # Prophet modeli
                prophet_data = ts_data.reset_index()
                prophet_data.columns = ['ds', 'y']
                
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.05
                )
                
                model.fit(prophet_data)
                
                # Gelecek tahmini
                future = model.make_future_dataframe(periods=tahmin_horizonu, freq='M')
                forecast = model.predict(future)
                
                # Model metrikleri
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                
                train_size = int(len(prophet_data) * 0.8)
                train_data = prophet_data.iloc[:train_size]
                test_data = prophet_data.iloc[train_size:]
                
                if len(test_data) > 0:
                    test_future = model.make_future_dataframe(periods=len(test_data))
                    test_forecast = model.predict(test_future)
                    
                    y_true = test_data['y'].values
                    y_pred = test_forecast.iloc[-len(test_data):]['yhat'].values
                    
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                else:
                    mae = rmse = mape = 0
                
                return {
                    'model': model,
                    'forecast': forecast,
                    'original_data': prophet_data,
                    'metrics': {
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape
                    },
                    'tahmin_horizonu': tahmin_horizonu
                }
            
        except Exception as e:
            st.error(f"Tahmin modeli hatası: {str(e)}")
            return None
    
    def risk_analizi_modeli(self, df):
        """Risk analizi modeli oluştur"""
        try:
            with st.spinner("⚠️ **Risk analizi yapılıyor...**"):
                risk_faktörleri = []
                
                # 1. Büyüme volatilitesi riski
                if 'Büyüme_Volatilitesi' in df.columns:
                    buyume_risk = df['Büyüme_Volatilitesi'] / df['Büyüme_Volatilitesi'].max() * 100
                    risk_faktörleri.append(buyume_risk)
                
                # 2. Pazar payı değişkenliği
                if 'Global_Pazar_Payı' in df.columns:
                    pazar_degiskenlik = df['Global_Pazar_Payı'].rolling(3, min_periods=1).std()
                    pazar_risk = pazar_degiskenlik / pazar_degiskenlik.max() * 100
                    risk_faktörleri.append(pazar_risk.fillna(0))
                
                # 3. Fiyat volatilitesi
                fiyat_sutunlari = [s for s in df.columns if 'fiyat' in str(s).lower()]
                if fiyat_sutunlari:
                    fiyat_vol = df[fiyat_sutunlari[-1]].rolling(3, min_periods=1).std()
                    fiyat_risk = fiyat_vol / fiyat_vol.max() * 100
                    risk_faktörleri.append(fiyat_risk.fillna(0))
                
                # 4. Rekabet riski
                if 'Rekabet_Skoru' in df.columns:
                    rekabet_risk = df['Rekabet_Skoru']
                    risk_faktörleri.append(rekabet_risk)
                
                if risk_faktörleri:
                    # Ağırlıklı risk skoru
                    agirliklar = [0.3, 0.3, 0.2, 0.2][:len(risk_faktörleri)]
                    risk_skoru = pd.Series(0, index=df.index)
                    
                    for risk, agirlik in zip(risk_faktörleri, agirliklar):
                        risk_skoru += risk * agirlik
                    
                    # Risk segmentasyonu
                    risk_segmentleri = pd.qcut(risk_skoru, q=4, labels=[
                        'Düşük Risk', 'Orta Risk', 'Yüksek Risk', 'Çok Yüksek Risk'
                    ])
                    
                    return {
                        'risk_skoru': risk_skoru,
                        'risk_segmenti': risk_segmentleri,
                        'risk_faktorleri': risk_faktörleri,
                        'agirliklar': agirliklar
                    }
                else:
                    return None
            
        except Exception as e:
            st.error(f"Risk analizi hatası: {str(e)}")
            return None
    
    def stratejik_oneriler(self, df, metrikler):
        """Stratejik öneriler oluştur"""
        try:
            oneriler = []
            
            # 1. Pazar liderliği analizi
            if 'Şirket' in df.columns and 'Satış_2024' in df.columns:
                sirket_paylari = df.groupby('Şirket')['Satış_2024'].sum().sort_values(ascending=False)
                pazar_lideri = sirket_paylari.index[0]
                lider_payi = sirket_paylari.iloc[0] / sirket_paylari.sum() * 100
                
                if lider_payi > 30:
                    oneriler.append({
                        'tip': 'warning',
                        'baslik': '🏆 Pazar Konsantrasyonu Yüksek',
                        'aciklama': f'{pazar_lideri} %{lider_payi:.1f} pazar payı ile dominant konumda. Rekabet analizi önerilir.',
                        'oneri': 'Yeni pazar giriş stratejileri geliştirin.'
                    })
            
            # 2. Büyüme fırsatları
            buyume_sutun = next((s for s in df.columns if 'büyüme' in str(s).lower() and 'ortalama' not in s), None)
            if buyume_sutun:
                yuksek_buyume = df[df[buyume_sutun] > 20]
                if len(yuksek_buyume) > 0:
                    oneriler.append({
                        'tip': 'success',
                        'baslik': '🚀 Yüksek Büyüme Fırsatları',
                        'aciklama': f'{len(yuksek_buyume)} ürün %20\'den fazla büyüme gösteriyor.',
                        'oneri': 'Bu ürünlere yatırımı artırın.'
                    })
            
            # 3. Riskli ürünler
            if 'Risk_Skoru' in df.columns:
                yuksek_risk = df[df['Risk_Skoru'] > 70]
                if len(yuksek_risk) > 0:
                    oneriler.append({
                        'tip': 'danger',
                        'baslik': '⚠️ Yüksek Riskli Ürünler',
                        'aciklama': f'{len(yuksek_risk)} ürün yüksek risk skoruna sahip.',
                        'oneri': 'Risk yönetimi stratejileri geliştirin.'
                    })
            
            # 4. International fırsatlar
            if 'International_Product' in df.columns:
                international_urunler = df[df['International_Product'] == True]
                local_urunler = df[df['International_Product'] == False]
                
                if len(international_urunler) > 0 and len(local_urunler) > 0:
                    intl_buyume = international_urunler[buyume_sutun].mean() if buyume_sutun else 0
                    local_buyume = local_urunler[buyume_sutun].mean() if buyume_sutun else 0
                    
                    if intl_buyume > local_buyume:
                        oneriler.append({
                            'tip': 'info',
                            'baslik': '🌍 International Ürünler Daha Hızlı Büyüyor',
                            'aciklama': f'International ürünler yerel ürünlerden %{intl_buyume-local_buyume:.1f} daha hızlı büyüyor.',
                            'oneri': 'International ürün portföyünüzü genişletin.'
                        })
            
            # 5. Fiyat optimizasyonu
            fiyat_sutun = next((s for s in df.columns if 'fiyat' in str(s).lower()), None)
            if fiyat_sutun and buyume_sutun:
                correlation = df[[fiyat_sutun, buyume_sutun]].corr().iloc[0,1]
                if correlation < -0.3:
                    oneriler.append({
                        'tip': 'warning',
                        'baslik': '💰 Fiyat Esnekliği Yüksek',
                        'aciklama': 'Fiyat artışları satışları önemli ölçüde etkiliyor.',
                        'oneri': 'Fiyatlandırma stratejinizi gözden geçirin.'
                    })
            
            return oneriler[:10]  # İlk 10 öneri
            
        except Exception as e:
            st.error(f"Stratejik öneri hatası: {str(e)}")
            return []

# ================================================
# 5. ENTERPRISE GÖRSELLEŞTİRME MOTORU - 600+ SATIR
# ================================================

class EnterpriseGorsellestirme:
    """Enterprise-level görselleştirme motoru"""
    
    def __init__(self):
        self.theme = {
            'background': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font_color': '#ffffff',
            'grid_color': 'rgba(255,255,255,0.1)',
            'colorscale': 'Viridis',
            'color_continuous': px.colors.sequential.Viridis,
            'color_discrete': px.colors.qualitative.Bold
        }
    
    def metrik_paneli_olustur(self, metrikler):
        """Metrik paneli oluştur"""
        try:
            # Performans skoru
            if 'toplam_performans' in metrikler:
                performans = metrikler['toplam_performans']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    self.metrik_karti(
                        "🏆 Toplam Performans",
                        f"{performans['toplam_skor']:.1f}",
                        "Puan",
                        "primary"
                    )
                
                with col2:
                    if 'genel_istatistikler' in metrikler:
                        toplam_satis = metrikler['genel_istatistikler'].get('toplam_satis', 0)
                        self.metrik_karti(
                            "💰 Toplam Pazar",
                            f"${toplam_satis/1e9:.2f}B",
                            "2024 Satışları",
                            "info"
                        )
                
                with col3:
                    if 'buyume_analizi' in metrikler:
                        ortalama_buyume = metrikler['buyume_analizi'].get('ortalama_buyume', 0)
                        self.metrik_karti(
                            "📈 Ortalama Büyüme",
                            f"%{ortalama_buyume:.1f}",
                            "Yıllık Büyüme",
                            "success" if ortalama_buyume > 0 else "danger"
                        )
                
                with col4:
                    if 'international_analiz' in metrikler:
                        intl_orani = metrikler['international_analiz'].get('international_orani', 0)
                        self.metrik_karti(
                            "🌍 International Oranı",
                            f"%{intl_orani:.1f}",
                            "Çoklu Pazar Ürünleri",
                            "info"
                        )
            
            # İkinci satır
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                if 'pazar_analizi' in metrikler:
                    hhi = metrikler['pazar_analizi'].get('hhi_index', 0)
                    durum = "Monopol" if hhi > 2500 else "Oligopol" if hhi > 1500 else "Rekabetçi"
                    self.metrik_karti(
                        "🏢 Rekabet Yoğunluğu",
                        f"{hhi:.0f}",
                        durum,
                        "warning" if hhi > 1500 else "success"
                    )
            
            with col6:
                if 'risk_analizi' in metrikler:
                    ortalama_risk = metrikler['risk_analizi'].get('ortalama_risk', 0)
                    self.metrik_karti(
                        "⚠️ Ortalama Risk",
                        f"%{ortalama_risk:.1f}",
                        "Risk Skoru",
                        "danger" if ortalama_risk > 50 else "success"
                    )
            
            with col7:
                if 'fiyat_analizi' in metrikler:
                    fiyat_cv = metrikler['fiyat_analizi'].get('fiyat_cv', 0)
                    self.metrik_karti(
                        "💰 Fiyat Stabilitesi",
                        f"%{fiyat_cv:.1f}",
                        "CV Değeri",
                        "warning" if fiyat_cv > 30 else "success"
                    )
            
            with col8:
                if 'rekabet_analizi' in metrikler:
                    sirket_sayisi = metrikler['rekabet_analizi'].get('sirket_sayisi', 0)
                    self.metrik_karti(
                        "🏭 Aktif Şirket",
                        f"{sirket_sayisi}",
                        "Pazardaki Şirket",
                        "info"
                    )
            
        except Exception as e:
            st.error(f"Metrik paneli hatası: {str(e)}")
    
    def metrik_karti(self, baslik, deger, alt_bilgi, tip="primary"):
        """Metrik kartı oluştur"""
        st.markdown(f"""
        <div class="enterprise-metric-card {tip}">
            <div class="enterprise-metric-label">{baslik}</div>
            <div class="enterprise-metric-value">{deger}</div>
            <div class="enterprise-metric-trend">
                <span class="badge badge-{tip}">{alt_bilgi}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def satis_trend_grafigi(self, df):
        """Satış trend grafiği"""
        try:
            satis_sutunlari = [s for s in df.columns if 'satış' in str(s).lower() or 'sales' in str(s).lower()]
            
            if len(satis_sutunlari) >= 2:
                # Yıllık verileri topla
                yillik_veri = []
                for sutun in sorted(satis_sutunlari):
                    yil = sutun.split('_')[-1]
                    yillik_veri.append({
                        'Yıl': yil,
                        'Toplam Satış': df[sutun].sum(),
                        'Ortalama Satış': df[sutun].mean(),
                        'Ürün Sayısı': (df[sutun] > 0).sum()
                    })
                
                yillik_df = pd.DataFrame(yillik_veri)
                
                # Ana grafik
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Yıllık Satış Trendi', 'Ortalama Satış',
                                   'Aktif Ürün Sayısı', 'Büyüme Oranları'),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )
                
                # 1. Toplam satış
                fig.add_trace(
                    go.Bar(
                        x=yillik_df['Yıl'],
                        y=yillik_df['Toplam Satış'],
                        name='Toplam Satış',
                        marker_color='#2d7dd2',
                        text=[f'${x/1e6:.0f}M' for x in yillik_df['Toplam Satış']],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
                
                # 2. Ortalama satış
                fig.add_trace(
                    go.Scatter(
                        x=yillik_df['Yıl'],
                        y=yillik_df['Ortalama Satış'],
                        mode='lines+markers',
                        name='Ortalama Satış',
                        line=dict(color='#2acaea', width=3),
                        marker=dict(size=10)
                    ),
                    row=1, col=2
                )
                
                # 3. Ürün sayısı
                fig.add_trace(
                    go.Bar(
                        x=yillik_df['Yıl'],
                        y=yillik_df['Ürün Sayısı'],
                        name='Aktif Ürün',
                        marker_color='#2dd2a3'
                    ),
                    row=2, col=1
                )
                
                # 4. Büyüme oranları
                if len(yillik_df) > 1:
                    buyume_oranlari = []
                    for i in range(1, len(yillik_df)):
                        buyume = ((yillik_df['Toplam Satış'].iloc[i] - yillik_df['Toplam Satış'].iloc[i-1]) / 
                                 yillik_df['Toplam Satış'].iloc[i-1] * 100) if yillik_df['Toplam Satış'].iloc[i-1] > 0 else 0
                        buyume_oranlari.append(buyume)
                    
                    fig.add_trace(
                        go.Bar(
                            x=yillik_df['Yıl'].iloc[1:],
                            y=buyume_oranlari,
                            name='Büyüme (%)',
                            marker_color=['#2dd2a3' if g > 0 else '#eb5757' for g in buyume_oranlari],
                            text=[f'{g:.1f}%' for g in buyume_oranlari],
                            textposition='auto'
                        ),
                        row=2, col=2
                    )
                
                fig.update_layout(
                    height=700,
                    plot_bgcolor=self.theme['background'],
                    paper_bgcolor=self.theme['paper_bgcolor'],
                    font_color=self.theme['font_color'],
                    showlegend=False,
                    title_text='📈 Satış Trendleri Analizi',
                    title_x=0.5,
                    title_font=dict(size=24)
                )
                
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor=self.theme['grid_color'])
                
                return fig
            
            return None
            
        except Exception as e:
            st.error(f"Satış trend grafiği hatası: {str(e)}")
            return None
    
    def pazar_payi_dagilimi(self, df):
        """Pazar payı dağılımı"""
        try:
            if 'Şirket' in df.columns and 'Satış_2024' in df.columns:
                sirket_satis = df.groupby('Şirket')['Satış_2024'].sum().sort_values(ascending=False)
                top_sirketler = sirket_satis.head(10)
                
                # Diğer şirketler
                diger_satis = sirket_satis.iloc[10:].sum() if len(sirket_satis) > 10 else 0
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Pazar Payı Dağılımı', 'Top 10 Şirket Performansı'),
                    specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                    column_widths=[0.4, 0.6]
                )
                
                # Pasta grafiği
                if diger_satis > 0:
                    pasta_verisi = pd.concat([top_sirketler, pd.Series({'Diğer': diger_satis})])
                else:
                    pasta_verisi = top_sirketler
                
                fig.add_trace(
                    go.Pie(
                        labels=pasta_verisi.index,
                        values=pasta_verisi.values,
                        hole=0.4,
                        marker_colors=px.colors.qualitative.Bold,
                        textinfo='percent+label',
                        textposition='outside',
                        insidetextorientation='radial'
                    ),
                    row=1, col=1
                )
                
                # Bar grafiği
                fig.add_trace(
                    go.Bar(
                        x=top_sirketler.values,
                        y=top_sirketler.index,
                        orientation='h',
                        marker_color='#2d7dd2',
                        text=[f'${x/1e6:.1f}M' for x in top_sirketler.values],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor=self.theme['background'],
                    paper_bgcolor=self.theme['paper_bgcolor'],
                    font_color=self.theme['font_color'],
                    showlegend=False,
                    title_text='🏢 Pazar Konsantrasyonu Analizi',
                    title_x=0.5
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.error(f"Pazar payı grafiği hatası: {str(e)}")
            return None
    
    def fiyat_hacim_analizi(self, df):
        """Fiyat-hacim analizi"""
        try:
            fiyat_sutun = next((s for s in df.columns if 'fiyat' in str(s).lower()), None)
            hacim_sutun = next((s for s in df.columns if 'hacim' in str(s).lower() or 'miktar' in str(s).lower()), None)
            
            if fiyat_sutun and hacim_sutun:
                # Veri hazırlama
                plot_df = df[[fiyat_sutun, hacim_sutun]].dropna()
                
                if len(plot_df) > 10000:
                    plot_df = plot_df.sample(10000, random_state=42)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Fiyat-Hacim İlişkisi', 'Fiyat Dağılımı',
                                   'Hacim Dağılımı', 'Fiyat Segmentleri'),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )
                
                # 1. Scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=plot_df[fiyat_sutun],
                        y=plot_df[hacim_sutun],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=plot_df[hacim_sutun],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Hacim")
                        ),
                        name='Ürünler'
                    ),
                    row=1, col=1
                )
                
                # 2. Fiyat histogram
                fig.add_trace(
                    go.Histogram(
                        x=df[fiyat_sutun],
                        nbinsx=50,
                        marker_color='#2d7dd2',
                        name='Fiyat Dağılımı'
                    ),
                    row=1, col=2
                )
                
                # 3. Hacim histogram
                fig.add_trace(
                    go.Histogram(
                        x=df[hacim_sutun],
                        nbinsx=50,
                        marker_color='#2acaea',
                        name='Hacim Dağılımı'
                    ),
                    row=2, col=1
                )
                
                # 4. Fiyat segmentleri
                fiyat_q1 = df[fiyat_sutun].quantile(0.33)
                fiyat_q2 = df[fiyat_sutun].quantile(0.67)
                
                segmentler = pd.cut(df[fiyat_sutun], 
                                   bins=[-np.inf, fiyat_q1, fiyat_q2, np.inf],
                                   labels=['Düşük', 'Orta', 'Yüksek'])
                
                segment_sayilari = segmentler.value_counts()
                
                fig.add_trace(
                    go.Bar(
                        x=segment_sayilari.index,
                        y=segment_sayilari.values,
                        marker_color='#2dd2a3',
                        text=segment_sayilari.values,
                        textposition='auto'
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(
                    height=700,
                    plot_bgcolor=self.theme['background'],
                    paper_bgcolor=self.theme['paper_bgcolor'],
                    font_color=self.theme['font_color'],
                    showlegend=False,
                    title_text='💰 Fiyat-Hacim Analizi',
                    title_x=0.5
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.error(f"Fiyat-hacim grafiği hatası: {str(e)}")
            return None
    
    def international_product_analizi(self, df):
        """International Product analizi"""
        try:
            if 'International_Product' not in df.columns:
                return None
            
            intl_df = df[df['International_Product'] == True]
            local_df = df[df['International_Product'] == False]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('International vs Local', 'Satış Karşılaştırması',
                               'Coğrafi Yayılım', 'Büyüme Performansı'),
                specs=[[{'type': 'pie'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'bar'}]],
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # 1. International vs Local dağılımı
            intl_count = len(intl_df)
            local_count = len(local_df)
            
            fig.add_trace(
                go.Pie(
                    labels=['International', 'Local'],
                    values=[intl_count, local_count],
                    hole=0.4,
                    marker_colors=['#2d7dd2', '#64748b']
                ),
                row=1, col=1
            )
            
            # 2. Satış karşılaştırması
            if 'Satış_2024' in df.columns:
                intl_satis = intl_df['Satış_2024'].sum()
                local_satis = local_df['Satış_2024'].sum()
                
                fig.add_trace(
                    go.Bar(
                        x=['International', 'Local'],
                        y=[intl_satis, local_satis],
                        marker_color=['#2d7dd2', '#64748b'],
                        text=[f'${intl_satis/1e6:.1f}M', f'${local_satis/1e6:.1f}M'],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
            
            # 3. Coğrafi yayılım
            if 'Ülke_Sayısı' in df.columns:
                ulke_dagilim = df['Ülke_Sayısı'].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(
                        x=ulke_dagilim.index.astype(str),
                        y=ulke_dagilim.values,
                        marker_color='#2acaea'
                    ),
                    row=2, col=1
                )
            
            # 4. Büyüme karşılaştırması
            buyume_sutun = next((s for s in df.columns if 'büyüme' in str(s).lower()), None)
            if buyume_sutun:
                intl_buyume = intl_df[buyume_sutun].mean()
                local_buyume = local_df[buyume_sutun].mean()
                
                fig.add_trace(
                    go.Bar(
                        x=['International', 'Local'],
                        y=[intl_buyume, local_buyume],
                        marker_color=['#2d7dd2', '#64748b'],
                        text=[f'{intl_buyume:.1f}%', f'{local_buyume:.1f}%'],
                        textposition='auto'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                plot_bgcolor=self.theme['background'],
                paper_bgcolor=self.theme['paper_bgcolor'],
                font_color=self.theme['font_color'],
                showlegend=False,
                title_text='🌍 International Product Analizi',
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.error(f"International Product grafiği hatası: {str(e)}")
            return None
    
    def segmentasyon_analizi(self, segmentasyon_sonuclari):
        """Segmentasyon analizi grafikleri"""
        try:
            if not segmentasyon_sonuclari:
                return None
            
            df = segmentasyon_sonuclari['data']
            kume_istatistikleri = segmentasyon_sonuclari['kume_istatistikleri']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Küme Dağılımı', 'Küme Özellikleri',
                               'Satış vs Büyüme', 'Fiyat vs Pazar Payı'),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # 1. Küme dağılımı
            kume_dagilim = df['Küme_İsmi'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=kume_dagilim.index,
                    values=kume_dagilim.values,
                    hole=0.3,
                    marker_colors=px.colors.qualitative.Bold
                ),
                row=1, col=1
            )
            
            # 2. Küme özellikleri (radar chart için bar)
            kume_df = pd.DataFrame(kume_istatistikleri)
            
            fig.add_trace(
                go.Bar(
                    x=kume_df['isim'],
                    y=kume_df['ortalama_satis'],
                    name='Ort. Satış',
                    marker_color='#2d7dd2'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=kume_df['isim'],
                    y=kume_df['ortalama_buyume'],
                    name='Ort. Büyüme',
                    marker_color='#2acaea'
                ),
                row=1, col=2
            )
            
            # 3. Satış vs Büyüme scatter
            fig.add_trace(
                go.Scatter(
                    x=df['Satış_2024'] if 'Satış_2024' in df.columns else df.iloc[:, 0],
                    y=df['Yıllık_Büyüme'] if 'Yıllık_Büyüme' in df.columns else df.iloc[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=df['Küme'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=df['Küme_İsmi']
                ),
                row=2, col=1
            )
            
            # 4. Fiyat vs Pazar Payı
            if 'Fiyat' in df.columns and 'Global_Pazar_Payı' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Fiyat'],
                        y=df['Global_Pazar_Payı'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=df['Küme'],
                            colorscale='Viridis',
                            showscale=False
                        ),
                        text=df['Küme_İsmi']
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                plot_bgcolor=self.theme['background'],
                paper_bgcolor=self.theme['paper_bgcolor'],
                font_color=self.theme['font_color'],
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                title_text='🔬 Pazar Segmentasyonu Analizi',
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Segmentasyon grafiği hatası: {str(e)}")
            return None
    
    def tahmin_grafikleri(self, tahmin_sonuclari):
        """Tahmin grafikleri"""
        try:
            if not tahmin_sonuclari:
                return None
            
            forecast = tahmin_sonuclari['forecast']
            original_data = tahmin_sonuclari['original_data']
            metrics = tahmin_sonuclari['metrics']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Satış Tahmini', 'Tahmin Bileşenleri',
                               'Hata Analizi', 'Trend Analizi'),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # 1. Satış tahmini
            fig.add_trace(
                go.Scatter(
                    x=original_data['ds'],
                    y=original_data['y'],
                    mode='lines+markers',
                    name='Gerçek Satış',
                    line=dict(color='#2d7dd2', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Tahmin',
                    line=dict(color='#2acaea', width=3, dash='dash')
                ),
                row=1, col=1
            )
            
            # Güven aralığı
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(42, 202, 234, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Güven Aralığı'
                ),
                row=1, col=1
            )
            
            # 2. Tahmin bileşenleri
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['trend'],
                    mode='lines',
                    name='Trend',
                    line=dict(color='#2dd2a3', width=2)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yearly'],
                    mode='lines',
                    name='Yıllık Mevsimsellik',
                    line=dict(color='#f2c94c', width=2)
                ),
                row=1, col=2
            )
            
            # 3. Hata analizi
            if len(original_data) > 0:
                errors = original_data['y'] - forecast.loc[:len(original_data)-1, 'yhat']
                
                fig.add_trace(
                    go.Histogram(
                        x=errors,
                        nbinsx=50,
                        marker_color='#eb5757',
                        name='Hata Dağılımı'
                    ),
                    row=2, col=1
                )
                
                # Hata metrikleri
                fig.add_annotation(
                    xref="x domain",
                    yref="y domain",
                    x=0.5, y=0.9,
                    text=f"MAE: {metrics['mae']:.2f}<br>RMSE: {metrics['rmse']:.2f}<br>MAPE: {metrics['mape']:.1f}%",
                    showarrow=False,
                    font=dict(size=12, color='white'),
                    align='center',
                    bordercolor='white',
                    borderwidth=1,
                    borderpad=4,
                    bgcolor='rgba(0,0,0,0.5)',
                    row=2, col=1
                )
            
            # 4. Trend analizi
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'].pct_change() * 100,
                    mode='lines',
                    name='Büyüme Oranı',
                    line=dict(color='#8b5cf6', width=2)
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=700,
                plot_bgcolor=self.theme['background'],
                paper_bgcolor=self.theme['paper_bgcolor'],
                font_color=self.theme['font_color'],
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                title_text='🔮 Satış Tahmini ve Analizi',
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Tahmin grafiği hatası: {str(e)}")
            return None
    
    def risk_analizi_grafigi(self, risk_analizi):
        """Risk analizi grafiği"""
        try:
            if not risk_analizi:
                return None
            
            risk_skoru = risk_analizi['risk_skoru']
            risk_segmenti = risk_analizi['risk_segmenti']
            risk_faktorleri = risk_analizi['risk_faktorleri']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Risk Skoru Dağılımı', 'Risk Segmentleri',
                               'Risk Faktörleri', 'Risk Korelasyonu'),
                vertical_spacing=0.15,
                horizontal_spacing=0.15
            )
            
            # 1. Risk skoru dağılımı
            fig.add_trace(
                go.Histogram(
                    x=risk_skoru,
                    nbinsx=50,
                    marker_color='#eb5757',
                    name='Risk Skoru'
                ),
                row=1, col=1
            )
            
            # 2. Risk segmentleri
            segment_dagilim = risk_segmenti.value_counts()
            fig.add_trace(
                go.Bar(
                    x=segment_dagilim.index,
                    y=segment_dagilim.values,
                    marker_color=['#2dd2a3', '#f2c94c', '#eb5757', '#8b0000'],
                    text=segment_dagilim.values,
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 3. Risk faktörleri
            if risk_faktorleri:
                faktor_isimleri = ['Büyüme Vol.', 'Pazar Değiş.', 'Fiyat Vol.', 'Rekabet'][:len(risk_faktorleri)]
                
                for i, (isim, faktor) in enumerate(zip(faktor_isimleri, risk_faktorleri)):
                    fig.add_trace(
                        go.Box(
                            y=faktor,
                            name=isim,
                            marker_color=px.colors.qualitative.Bold[i]
                        ),
                        row=2, col=1
                    )
            
            # 4. Risk korelasyonu heatmap (basitleştirilmiş)
            if len(risk_faktorleri) > 1:
                faktor_df = pd.DataFrame({f'Faktor_{i}': f for i, f in enumerate(risk_faktorleri)})
                corr_matrix = faktor_df.corr()
                
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=faktor_isimleri,
                        y=faktor_isimleri,
                        colorscale='RdBu',
                        zmid=0
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                plot_bgcolor=self.theme['background'],
                paper_bgcolor=self.theme['paper_bgcolor'],
                font_color=self.theme['font_color'],
                showlegend=False,
                title_text='⚠️ Risk Analizi',
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Risk analizi grafiği hatası: {str(e)}")
            return None

# ================================================
# 6. ENTERPRISE RAPORLAMA SİSTEMİ - 300+ SATIR
# ================================================

class EnterpriseRaporlama:
    """Enterprise-level raporlama sistemi"""
    
    def __init__(self):
        self.rapor_templates = {
            'genel_bakis': 'Genel Bakış Raporu',
            'pazar_analizi': 'Pazar Analizi Raporu',
            'international': 'International Product Raporu',
            'risk': 'Risk Analizi Raporu',
            'tahmin': 'Satış Tahmini Raporu',
            'tam': 'Tam Kapsamlı Rapor'
        }
    
    def rapor_paneli_olustur(self):
        """Raporlama paneli oluştur"""
        with st.sidebar.expander("📑 **RAPORLAMA**", expanded=False):
            st.markdown('<div class="enterprise-filter-title">📊 Rapor Türleri</div>', unsafe_allow_html=True)
            
            rapor_turu = st.selectbox(
                "Rapor Türü",
                list(self.rapor_templates.values()),
                key="rapor_turu"
            )
            
            rapor_format = st.radio(
                "Çıktı Formatı",
                ['Excel', 'PDF', 'HTML', 'JSON'],
                horizontal=True,
                key="rapor_format"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 **Rapor Oluştur**", use_container_width=True, type="primary"):
                    return True
            
            with col2:
                if st.button("🔄 **Sıfırla**", use_container_width=True):
                    for key in ['veri', 'filtrelenmis_veri', 'metrikler', 'icgoruler', 'segmentasyon', 'tahmin', 'risk']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            
            st.markdown("---")
            st.markdown("**💾 Dışa Aktarma**")
            
            col3, col4 = st.columns(2)
            
            with col3:
                if st.button("📊 **Veriyi İndir**", use_container_width=True):
                    return 'veri_indir'
            
            with col4:
                if st.button("📈 **Grafikleri İndir**", use_container_width=True):
                    return 'grafik_indir'
            
            return False
    
    def excel_raporu_olustur(self, df, metrikler, segmentasyon=None, tahmin=None, risk=None):
        """Excel raporu oluştur"""
        try:
            with st.spinner("📊 **Excel raporu oluşturuluyor...**"):
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # 1. Ham veri
                    df.to_excel(writer, sheet_name='HAM_VERİ', index=False)
                    
                    # 2. Metrikler
                    metrik_df = self.metrikleri_dataframe(metrikler)
                    metrik_df.to_excel(writer, sheet_name='METRİKLER', index=False)
                    
                    # 3. Segment analizi
                    if segmentasyon:
                        segment_df = segmentasyon['data']
                        segment_df.to_excel(writer, sheet_name='SEGMENT_ANALİZİ', index=False)
                        
                        kume_istatistik_df = pd.DataFrame(segmentasyon['kume_istatistikleri'])
                        kume_istatistik_df.to_excel(writer, sheet_name='KÜME_İSTATİSTİKLERİ', index=False)
                    
                    # 4. Tahmin analizi
                    if tahmin:
                        tahmin_df = tahmin['forecast']
                        tahmin_df.to_excel(writer, sheet_name='TAHMİN_ANALİZİ', index=True)
                    
                    # 5. Risk analizi
                    if risk:
                        risk_df = pd.DataFrame({
                            'Risk_Skoru': risk['risk_skoru'],
                            'Risk_Segmenti': risk['risk_segmenti']
                        })
                        risk_df.to_excel(writer, sheet_name='RİSK_ANALİZİ', index=False)
                    
                    # 6. Özet tablolar
                    self.ozet_tablolari_olustur(df, writer)
                
                output.seek(0)
                
                # İndirme butonu
                zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="⬇️ **Excel Raporunu İndir**",
                    data=output,
                    file_name=f"pharma_rapor_{zaman_damgasi}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"Excel raporu oluşturma hatası: {str(e)}")
    
    def metrikleri_dataframe(self, metrikler):
        """Metrikleri dataframe'e dönüştür"""
        try:
            rows = []
            
            for kategori, alt_metrikler in metrikler.items():
                if isinstance(alt_metrikler, dict):
                    for anahtar, deger in alt_metrikler.items():
                        if isinstance(deger, dict):
                            for sub_key, sub_val in deger.items():
                                rows.append({
                                    'Kategori': kategori,
                                    'Metrik': f"{anahtar} - {sub_key}",
                                    'Değer': sub_val
                                })
                        else:
                            rows.append({
                                'Kategori': kategori,
                                'Metrik': anahtar,
                                'Değer': deger
                            })
            
            return pd.DataFrame(rows)
        except:
            return pd.DataFrame()
    
    def ozet_tablolari_olustur(self, df, writer):
        """Özet tabloları oluştur"""
        try:
            # Şirket bazlı özet
            if 'Şirket' in df.columns and 'Satış_2024' in df.columns:
                sirket_ozet = df.groupby('Şirket').agg({
                    'Satış_2024': ['sum', 'mean', 'count', 'std'],
                    'Yıllık_Büyüme': 'mean',
                    'Fiyat': 'mean',
                    'Global_Pazar_Payı': 'mean'
                }).round(2)
                
                sirket_ozet.columns = ['_'.join(col).strip() for col in sirket_ozet.columns.values]
                sirket_ozet.to_excel(writer, sheet_name='ŞİRKET_ÖZET')
            
            # Ülke bazlı özet
            if 'Ülke' in df.columns:
                ulke_ozet = df.groupby('Ülke').agg({
                    'Satış_2024': ['sum', 'mean', 'count'],
                    'Yıllık_Büyüme': 'mean',
                    'Fiyat': 'mean'
                }).round(2)
                
                ulke_ozet.columns = ['_'.join(col).strip() for col in ulke_ozet.columns.values]
                ulke_ozet.to_excel(writer, sheet_name='ÜLKE_ÖZET')
            
            # Molekül bazlı özet
            if 'Molekül' in df.columns:
                molekul_ozet = df.groupby('Molekül').agg({
                    'Satış_2024': ['sum', 'mean', 'count'],
                    'Yıllık_Büyüme': 'mean',
                    'Fiyat': 'mean',
                    'International_Product': lambda x: (x == True).sum()
                }).round(2)
                
                molekul_ozet.columns = ['_'.join(col).strip() for col in molekul_ozet.columns.values]
                molekul_ozet.to_excel(writer, sheet_name='MOLEKÜL_ÖZET')
            
        except Exception as e:
            st.warning(f"Özet tablo oluşturma hatası: {str(e)}")
    
    def html_raporu_olustur(self, df, metrikler, segmentasyon=None, tahmin=None, risk=None):
        """HTML raporu oluştur"""
        try:
            with st.spinner("📄 **HTML raporu oluşturuluyor...**"):
                html_content = """
                <!DOCTYPE html>
                <html lang="tr">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>PharmaIntelligence Pro Raporu</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        .header { background: linear-gradient(135deg, #0c1a32, #14274e); color: white; padding: 30px; border-radius: 10px; }
                        .metric-card { background: #f5f5f5; padding: 20px; margin: 10px; border-radius: 8px; border-left: 4px solid #2d7dd2; }
                        .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                        .table th, .table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                        .table th { background-color: #14274e; color: white; }
                        .section { margin: 40px 0; }
                    </style>
                </head>
                <body>
                """
                
                # Başlık
                html_content += f"""
                <div class="header">
                    <h1>PharmaIntelligence Pro Enterprise Raporu</h1>
                    <p>Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Toplam Kayıt: {len(df):,} | Toplam Sütun: {len(df.columns)}</p>
                </div>
                """
                
                # Metrikler bölümü
                html_content += "<div class='section'><h2>📊 Temel Metrikler</h2>"
                
                if 'genel_istatistikler' in metrikler:
                    genel = metrikler['genel_istatistikler']
                    html_content += f"""
                    <div style="display: flex; flex-wrap: wrap;">
                        <div class="metric-card" style="flex: 1; min-width: 200px;">
                            <h3>💰 Toplam Pazar</h3>
                            <h2>${genel.get('toplam_satis', 0)/1e9:.2f}B</h2>
                        </div>
                        <div class="metric-card" style="flex: 1; min-width: 200px;">
                            <h3>📈 Ortalama Büyüme</h3>
                            <h2>%{metrikler.get('buyume_analizi', {}).get('ortalama_buyume', 0):.1f}</h2>
                        </div>
                        <div class="metric-card" style="flex: 1; min-width: 200px;">
                            <h3>🌍 International Oranı</h3>
                            <h2>%{metrikler.get('international_analiz', {}).get('international_orani', 0):.1f}</h2>
                        </div>
                    </div>
                    """
                
                # Performans skoru
                if 'toplam_performans' in metrikler:
                    performans = metrikler['toplam_performans']
                    html_content += f"""
                    <div class="metric-card">
                        <h3>🏆 Toplam Performans Skoru</h3>
                        <h1 style="color: #2d7dd2;">{performans['toplam_skor']:.1f}/100</h1>
                    </div>
                    """
                
                html_content += "</div>"
                
                # Veri önizleme
                html_content += "<div class='section'><h2>📋 Veri Önizleme</h2>"
                html_content += df.head(20).to_html(classes='table', index=False)
                html_content += "</div>"
                
                # Kapanış
                html_content += """
                <div class="section">
                    <hr>
                    <p style="text-align: center; color: #666;">
                        © 2024 PharmaIntelligence Pro Enterprise | Tüm hakları saklıdır.<br>
                        Bu rapor otomatik olarak oluşturulmuştur.
                    </p>
                </div>
                </body>
                </html>
                """
                
                # İndirme butonu
                zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="⬇️ **HTML Raporunu İndir**",
                    data=html_content,
                    file_name=f"pharma_rapor_{zaman_damgasi}.html",
                    mime="text/html",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"HTML raporu oluşturma hatası: {str(e)}")

# ================================================
# 7. ANA UYGULAMA - 500+ SATIR
# ================================================

def main():
    """Ana uygulama fonksiyonu"""
    
    # Başlık
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="enterprise-title">🏥 PHARMAINTELLIGENCE PRO</h1>
        <p class="enterprise-subtitle">
        Enterprise-level pharmaceutical market intelligence platform with advanced analytics, 
        predictive modeling, and strategic insights. Powered by machine learning and real-time data processing.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Session state başlatma
    if 'enterprise_veri' not in st.session_state:
        st.session_state.enterprise_veri = None
    if 'filtrelenmis_veri' not in st.session_state:
        st.session_state.filtrelenmis_veri = None
    if 'metrikler' not in st.session_state:
        st.session_state.metrikler = None
    if 'segmentasyon' not in st.session_state:
        st.session_state.segmentasyon = None
    if 'tahmin' not in st.session_state:
        st.session_state.tahmin = None
    if 'risk_analizi' not in st.session_state:
        st.session_state.risk_analizi = None
    if 'stratjik_oneriler' not in st.session_state:
        st.session_state.stratejik_oneriler = []
    if 'filtre_config' not in st.session_state:
        st.session_state.filtre_config = {}
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="enterprise-sidebar-title">🚀 ENTERPRISE PANEL</div>', unsafe_allow_html=True)
        
        # Veri yükleme
        with st.expander("📁 **VERİ YÜKLEME**", expanded=True):
            yuklenen_dosya = st.file_uploader(
                "Excel/CSV/Parquet Dosyası Yükleyin",
                type=['xlsx', 'xls', 'csv', 'parquet', 'json'],
                help="10M+ satır desteklenir. Enterprise seviyesinde optimizasyon uygulanır."
            )
            
            if yuklenen_dosya:
                st.info(f"**Dosya:** {yuklenen_dosya.name}")
                
                col1, col2 = st.columns(2)
                with col1:
                    orneklem = st.checkbox("Örneklem Kullan", value=False, help="Büyük dosyalar için örneklem kullanın")
                
                if orneklem:
                    orneklem_boyutu = st.number_input("Örneklem Boyutu", min_value=1000, max_value=1000000, value=10000, step=1000)
                else:
                    orneklem_boyutu = None
                
                if st.button("🚀 **VERİYİ YÜKLE & ANALİZ ET**", type="primary", use_container_width=True):
                    with st.spinner("**Enterprise veri işleme başlatıldı...**"):
                        # Veri yükleme
                        veri_sistemi = EnterpriseVeriSistemi()
                        df = veri_sistemi.buyuk_veri_yukle_optimize(yuklenen_dosya, orneklem_boyutu)
                        
                        if df is not None:
                            # Analiz verilerini hazırla
                            df = veri_sistemi.analiz_verisi_hazirla(df)
                            
                            # Session state'e kaydet
                            st.session_state.enterprise_veri = df
                            st.session_state.filtrelenmis_veri = df.copy()
                            
                            # Analiz motoru
                            analitik = EnterpriseAnalitikMotoru()
                            st.session_state.metrikler = analitik.kapsamli_metrik_analizi(df)
                            st.session_state.stratejik_oneriler = analitik.stratejik_oneriler(df, st.session_state.metrikler)
                            
                            st.success(f"✅ **{len(df):,} satır başarıyla yüklendi ve analiz edildi!**")
                            st.rerun()
        
        # Filtreleme
        if st.session_state.enterprise_veri is not None:
            with st.expander("🎯 **FİLTRELEME**", expanded=True):
                filtre_sistemi = EnterpriseFiltreSistemi()
                filtre_config = filtre_sistemi.filtre_paneli_olustur(st.session_state.enterprise_veri)
                
                if st.button("✅ **FİLTRELERİ UYGULA**", type="primary", use_container_width=True):
                    # Filtreleri uygula
                    filtrelenmis_df = filtre_sistemi.filtreleri_uygula(st.session_state.enterprise_veri, filtre_config)
                    st.session_state.filtrelenmis_veri = filtrelenmis_df
                    st.session_state.filtre_config = filtre_config
                    
                    # Metrikleri güncelle
                    analitik = EnterpriseAnalitikMotoru()
                    st.session_state.metrikler = analitik.kapsamli_metrik_analizi(filtrelenmis_df)
                    
                    st.success(f"✅ **Filtreler uygulandı:** {len(filtrelenmis_df):,} satır")
                    st.rerun()
        
        # Raporlama
        if st.session_state.enterprise_veri is not None:
            with st.expander("📑 **RAPORLAMA**", expanded=False):
                raporlama = EnterpriseRaporlama()
                rapor_istegi = raporlama.rapor_paneli_olustur()
                
                if rapor_istegi == 'veri_indir':
                    # Veriyi indir
                    csv = st.session_state.filtrelenmis_veri.to_csv(index=False)
                    zaman_damgasi = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="📥 **CSV İndir**",
                        data=csv,
                        file_name=f"pharma_veri_{zaman_damgasi}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
        <strong>PharmaIntelligence Pro Enterprise</strong><br>
        v5.0 | 4000+ Satır Kod<br>
        © 2024 Tüm hakları saklıdır.
        </div>
        """, unsafe_allow_html=True)
    
    # Ana içerik
    if st.session_state.enterprise_veri is None:
        # Hoşgeldiniz ekranı
        hosgeldiniz_ekrani()
    else:
        # Dashboard
        dashboard_goster()

def hosgeldiniz_ekrani():
    """Hoşgeldiniz ekranı"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="enterprise-welcome-container">
            <div class="enterprise-welcome-icon">🏥</div>
            <h2 class="enterprise-welcome-title">PharmaIntelligence Pro Enterprise</h2>
            <p style="color: #cbd5e1; margin-bottom: 2.5rem; line-height: 1.8; font-size: 1.1rem;">
            Dünyanın en gelişmiş farma pazarı analiz platformu. 
            International Product analizi, makine öğrenmesi, tahmin modellemesi 
            ve gerçek zamanlı stratejik içgörüler ile işinizi bir üst seviyeye taşıyın.
            </p>
            
            <div class="enterprise-feature-grid">
                <div class="enterprise-feature-card">
                    <div class="enterprise-feature-icon">🌍</div>
                    <div class="enterprise-feature-title">International Product Analytics</div>
                    <div class="enterprise-feature-description">Çoklu pazar ürün analizi ve global strateji geliştirme</div>
                </div>
                <div class="enterprise-feature-card">
                    <div class="enterprise-feature-icon">🤖</div>
                    <div class="enterprise-feature-title">AI & Machine Learning</div>
                    <div class="enterprise-feature-description">Yapay zeka destekli tahmin modelleri ve segmentasyon</div>
                </div>
                <div class="enterprise-feature-card">
                    <div class="enterprise-feature-icon">📊</div>
                    <div class="enterprise-feature-title">Real-time Analytics</div>
                    <div class="enterprise-feature-description">Gerçek zamanlı veri işleme ve görselleştirme</div>
                </div>
                <div class="enterprise-feature-card">
                    <div class="enterprise-feature-icon">🎯</div>
                    <div class="enterprise-feature-title">Strategic Insights</div>
                    <div class="enterprise-feature-description">Stratejik öneriler ve aksiyon planları</div>
                </div>
            </div>
            
            <div class="get-started-box" style="margin-top: 3rem;">
                <div class="get-started-title">🚀 Başlamak İçin</div>
                <div class="get-started-steps">
                1. Sol taraftaki panelden veri dosyanızı yükleyin<br>
                2. "Veriyi Yükle & Analiz Et" butonuna tıklayın<br>
                3. Enterprise dashboard ile analizlerinizi keşfedin<br>
                4. Gelişmiş filtreler ve raporlarla stratejilerinizi belirleyin
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def dashboard_goster():
    """Dashboard gösterimi"""
    # Veri kontrolü
    if st.session_state.filtrelenmis_veri is None or st.session_state.metrikler is None:
        st.error("Veri yüklenmedi veya analiz edilmedi!")
        return
    
    df = st.session_state.filtrelenmis_veri
    metrikler = st.session_state.metrikler
    
    # Filtre durumu
    if st.session_state.filtre_config:
        st.markdown(f"""
        <div class="enterprise-filter-status">
            🎯 **Aktif Filtreler** | 
            **{len(df):,} / {len(st.session_state.enterprise_veri):,}** satır gösteriliyor
        </div>
        """, unsafe_allow_html=True)
    
    # Tablar
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ",
        "💰 FİYAT ANALİZİ",
        "🌍 INTERNATIONAL",
        "🤖 MAKİNE ÖĞRENMESİ",
        "⚠️ RİSK ANALİZİ",
        "🔮 TAHMİN MODELLERİ",
        "📑 RAPORLAMA"
    ])
    
    with tab1:
        genel_bakis_tab(df, metrikler)
    
    with tab2:
        pazar_analizi_tab(df, metrikler)
    
    with tab3:
        fiyat_analizi_tab(df)
    
    with tab4:
        international_tab(df, metrikler)
    
    with tab5:
        makine_ogrenmesi_tab(df)
    
    with tab6:
        risk_analizi_tab(df)
    
    with tab7:
        tahmin_modelleri_tab(df)
    
    with tab8:
        raporlama_tab(df, metrikler)

def genel_bakis_tab(df, metrikler):
    """Genel Bakış tab'ı"""
    st.markdown('<h2 class="section-title">📊 GENEL BAKIŞ VE PERFORMANS</h2>', unsafe_allow_html=True)
    
    # Metrik paneli
    gorsellestirme = EnterpriseGorsellestirme()
    gorsellestirme.metrik_paneli_olustur(metrikler)
    
    # Stratejik öneriler
    st.markdown('<h3 class="subsection-title">💡 STRATEJİK ÖNERİLER</h3>', unsafe_allow_html=True)
    
    if st.session_state.stratejik_oneriler:
        for oneri in st.session_state.stratejik_oneriler[:5]:
            st.markdown(f"""
            <div class="enterprise-insight-card {oneri['tip']}">
                <div class="enterprise-insight-title">{oneri['baslik']}</div>
                <div class="enterprise-insight-content">{oneri['aciklama']}</div>
                <div class="enterprise-insight-footer">
                    <span>💡 Öneri:</span>
                    <span>{oneri['oneri']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Stratejik öneriler yükleniyor...")
    
    # Veri önizleme
    st.markdown('<h3 class="subsection-title">📋 VERİ ÖNİZLEME</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        satir_sayisi = st.slider("Satır Sayısı", 10, 1000, 100, 10)
        
        mevcut_sutunlar = df.columns.tolist()
        oncelikli_sutunlar = ['Molekül', 'Şirket', 'Ülke', 'Satış_2024', 'Yıllık_Büyüme', 'Fiyat']
        varsayilan_sutunlar = [s for s in oncelikli_sutunlar if s in mevcut_sutunlar][:6]
        
        secilen_sutunlar = st.multiselect(
            "Sütunlar",
            options=mevcut_sutunlar,
            default=varsayilan_sutunlar,
            key="genel_sutunlar"
        )
    
    with col2:
        if secilen_sutunlar:
            st.dataframe(df[secilen_sutunlar].head(satir_sayisi), use_container_width=True, height=400)
        else:
            st.dataframe(df.head(satir_sayisi), use_container_width=True, height=400)
    
    # Veri kalitesi
    st.markdown('<h3 class="subsection-title">🔍 VERİ KALİTESİ</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        nan_orani = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("NaN Oranı", f"%{nan_orani:.1f}")
    
    with col2:
        unique_molekul = df['Molekül'].nunique() if 'Molekül' in df.columns else 0
        st.metric("Benzersiz Molekül", f"{unique_molekul:,}")
    
    with col3:
        bellek = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Bellek Kullanımı", f"{bellek:.1f} MB")
    
    with col4:
        duplicate = len(df) - len(df.drop_duplicates())
        st.metric("Kopya Satır", f"{duplicate:,}")

def pazar_analizi_tab(df, metrikler):
    """Pazar Analizi tab'ı"""
    st.markdown('<h2 class="section-title">📈 PAZAR ANALİZİ VE TRENDLER</h2>', unsafe_allow_html=True)
    
    gorsellestirme = EnterpriseGorsellestirme()
    
    # Satış trendleri
    st.markdown('<h3 class="subsection-title">📊 SATIŞ TRENDLERİ</h3>', unsafe_allow_html=True)
    trend_grafik = gorsellestirme.satis_trend_grafigi(df)
    if trend_grafik:
        st.plotly_chart(trend_grafik, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Satış trend analizi için yeterli veri bulunamadı.")
    
    # Pazar payı dağılımı
    st.markdown('<h3 class="subsection-title">🏢 PAZAR PAYI DAĞILIMI</h3>', unsafe_allow_html=True)
    pazar_grafik = gorsellestirme.pazar_payi_dagilimi(df)
    if pazar_grafik:
        st.plotly_chart(pazar_grafik, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Pazar payı analizi için yeterli veri bulunamadı.")
    
    # Rekabet analizi
    st.markdown('<h3 class="subsection-title">🏆 REKABET ANALİZİ</h3>', unsafe_allow_html=True)
    
    if 'rekabet_analizi' in metrikler:
        rekabet = metrikler['rekabet_analizi']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("HHI İndeksi", f"{metrikler.get('pazar_analizi', {}).get('hhi_index', 0):.0f}")
        
        with col2:
            st.metric("CR4", f"%{metrikler.get('pazar_analizi', {}).get('cr4', 0):.1f}")
        
        with col3:
            st.metric("Şirket Sayısı", rekabet.get('sirket_sayisi', 0))
        
        with col4:
            st.metric("Gini Katsayısı", f"{rekabet.get('pazar_payi_gini', 0):.3f}")
    else:
        st.info("Rekabet analizi metrikleri mevcut değil.")

def fiyat_analizi_tab(df):
    """Fiyat Analizi tab'ı"""
    st.markdown('<h2 class="section-title">💰 FİYAT ANALİZİ VE OPTİMİZASYON</h2>', unsafe_allow_html=True)
    
    gorsellestirme = EnterpriseGorsellestirme()
    
    # Fiyat-hacim analizi
    st.markdown('<h3 class="subsection-title">📊 FİYAT-HACİM İLİŞKİSİ</h3>', unsafe_allow_html=True)
    fiyat_grafik = gorsellestirme.fiyat_hacim_analizi(df)
    if fiyat_grafik:
        st.plotly_chart(fiyat_grafik, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Fiyat-hacim analizi için yeterli veri bulunamadı.")
    
    # Fiyat segmentasyonu
    st.markdown('<h3 class="subsection-title">🎯 FİYAT SEGMENTASYONU</h3>', unsafe_allow_html=True)
    
    if 'Fiyat_Segmenti' in df.columns:
        segment_dagilim = df['Fiyat_Segmenti'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(
                values=segment_dagilim.values,
                names=segment_dagilim.index,
                title='Fiyat Segmentleri Dağılımı',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            for segment, sayi in segment_dagilim.items():
                st.metric(f"{segment}", f"{sayi:,}")
    else:
        st.info("Fiyat segmentasyonu verisi bulunamadı.")

def international_tab(df, metrikler):
    """International Product tab'ı"""
    st.markdown('<h2 class="section-title">🌍 INTERNATIONAL PRODUCT ANALİZİ</h2>', unsafe_allow_html=True)
    
    gorsellestirme = EnterpriseGorsellestirme()
    
    # International analiz grafiği
    st.markdown('<h3 class="subsection-title">📈 INTERNATIONAL VS LOCAL ANALİZİ</h3>', unsafe_allow_html=True)
    intl_grafik = gorsellestirme.international_product_analizi(df)
    if intl_grafik:
        st.plotly_chart(intl_grafik, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("International Product analizi için yeterli veri bulunamadı.")
    
    # International metrikler
    st.markdown('<h3 class="subsection-title">📊 INTERNATIONAL METRİKLER</h3>', unsafe_allow_html=True)
    
    if 'international_analiz' in metrikler:
        intl = metrikler['international_analiz']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("International Ürün", f"{intl.get('international_urun_sayisi', 0):,}")
        
        with col2:
            st.metric("International Oranı", f"%{intl.get('international_orani', 0):.1f}")
        
        with col3:
            st.metric("Satış Payı", f"%{intl.get('international_satis_payi', 0):.1f}")
        
        with col4:
            buyume_fark = intl.get('international_ortalama_buyume', 0) - intl.get('local_ortalama_buyume', 0)
            st.metric("Büyüme Farkı", f"%{buyume_fark:.1f}")
    else:
        st.info("International analiz metrikleri mevcut değil.")
    
    # International segment dağılımı
    if 'International_Segmenti' in df.columns:
        st.markdown('<h3 class="subsection-title">🏷️ INTERNATIONAL SEGMENT DAĞILIMI</h3>', unsafe_allow_html=True)
        
        segment_dagilim = df['International_Segmenti'].value_counts()
        
        fig = px.bar(
            x=segment_dagilim.index,
            y=segment_dagilim.values,
            title='International Segment Dağılımı',
            color=segment_dagilim.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            xaxis_title='Segment',
            yaxis_title='Ürün Sayısı'
        )
        st.plotly_chart(fig, use_container_width=True)

def makine_ogrenmesi_tab(df):
    """Makine Öğrenmesi tab'ı"""
    st.markdown('<h2 class="section-title">🤖 MAKİNE ÖĞRENMESİ ANALİZLERİ</h2>', unsafe_allow_html=True)
    
    # Segmentasyon analizi
    st.markdown('<h3 class="subsection-title">🔬 PAZAR SEGMENTASYONU</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        yontem = st.selectbox("Yöntem", ['kmeans', 'hierarchical', 'dbscan'], key="segmentasyon_yontem")
    
    with col2:
        kume_sayisi = st.slider("Küme Sayısı", 2, 10, 4, key="kume_sayisi")
    
    with col3:
        if st.button("🚀 **SEGMENTASYON ANALİZİ YAP**", use_container_width=True):
            with st.spinner("Segmentasyon analizi yapılıyor..."):
                analitik = EnterpriseAnalitikMotoru()
                segmentasyon = analitik.gelismis_pazar_segmentasyonu(df, yontem, kume_sayisi)
                
                if segmentasyon:
                    st.session_state.segmentasyon = segmentasyon
                    st.success(f"✅ {segmentasyon['segmentasyon_metrikleri']['kume_sayisi']} küme tespit edildi!")
                    st.rerun()
    
    # Segmentasyon sonuçları
    if st.session_state.segmentasyon:
        st.markdown('<h4 class="subsection-title">📊 SEGMENTASYON SONUÇLARI</h4>', unsafe_allow_html=True)
        
        gorsellestirme = EnterpriseGorsellestirme()
        segment_grafik = gorsellestirme.segmentasyon_analizi(st.session_state.segmentasyon)
        
        if segment_grafik:
            st.plotly_chart(segment_grafik, use_container_width=True, config={'displayModeBar': True})
        
        # Küme istatistikleri
        st.markdown('<h4 class="subsection-title">📈 KÜME İSTATİSTİKLERİ</h4>', unsafe_allow_html=True)
        
        kume_df = pd.DataFrame(st.session_state.segmentasyon['kume_istatistikleri'])
        st.dataframe(kume_df, use_container_width=True)
        
        # Segmentasyon kalitesi
        st.markdown('<h4 class="subsection-title">🎯 SEGMENTASYON KALİTESİ</h4>', unsafe_allow_html=True)
        
        metrikler = st.session_state.segmentasyon['segmentasyon_metrikleri']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if metrikler['inertia']:
                st.metric("Inertia", f"{metrikler['inertia']:,.0f}")
        
        with col2:
            if metrikler['silhouette_score']:
                st.metric("Silhouette", f"{metrikler['silhouette_score']:.3f}")
        
        with col3:
            if metrikler['calinski_score']:
                st.metric("Calinski", f"{metrikler['calinski_score']:,.0f}")
        
        with col4:
            st.metric("Küme Sayısı", metrikler['kume_sayisi'])

def risk_analizi_tab(df):
    """Risk Analizi tab'ı"""
    st.markdown('<h2 class="section-title">⚠️ RİSK ANALİZİ VE YÖNETİMİ</h2>', unsafe_allow_html=True)
    
    # Risk analizi
    if st.button("🚀 **RİSK ANALİZİ YAP**", use_container_width=True):
        with st.spinner("Risk analizi yapılıyor..."):
            analitik = EnterpriseAnalitikMotoru()
            risk_analizi = analitik.risk_analizi_modeli(df)
            
            if risk_analizi:
                st.session_state.risk_analizi = risk_analizi
                st.success("✅ Risk analizi tamamlandı!")
                st.rerun()
    
    # Risk analizi sonuçları
    if st.session_state.risk_analizi:
        gorsellestirme = EnterpriseGorsellestirme()
        risk_grafik = gorsellestirme.risk_analizi_grafigi(st.session_state.risk_analizi)
        
        if risk_grafik:
            st.plotly_chart(risk_grafik, use_container_width=True, config={'displayModeBar': True})
        
        # Risk segmentleri
        st.markdown('<h3 class="subsection-title">📊 RİSK SEGMENT DAĞILIMI</h3>', unsafe_allow_html=True)
        
        if 'risk_segmenti' in st.session_state.risk_analizi:
            risk_df = pd.DataFrame({
                'Risk_Skoru': st.session_state.risk_analizi['risk_skoru'],
                'Risk_Segmenti': st.session_state.risk_analizi['risk_segmenti']
            })
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                segment_dagilim = risk_df['Risk_Segmenti'].value_counts()
                
                fig = px.bar(
                    x=segment_dagilim.index,
                    y=segment_dagilim.values,
                    title='Risk Segmentleri Dağılımı',
                    color=segment_dagilim.values,
                    color_continuous_scale='Reds'
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    xaxis_title='Risk Segmenti',
                    yaxis_title='Ürün Sayısı'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                for segment, sayi in segment_dagilim.items():
                    st.metric(f"{segment}", f"{sayi:,}")
        
        # Yüksek riskli ürünler
        st.markdown('<h3 class="subsection-title">🚨 YÜKSEK RİSKLİ ÜRÜNLER</h3>', unsafe_allow_html=True)
        
        if 'risk_skoru' in st.session_state.risk_analizi:
            df['Risk_Skoru'] = st.session_state.risk_analizi['risk_skoru']
            yuksek_risk = df[df['Risk_Skoru'] > 70]
            
            if len(yuksek_risk) > 0:
                st.dataframe(
                    yuksek_risk[['Molekül', 'Şirket', 'Risk_Skoru', 'Satış_2024', 'Yıllık_Büyüme']].head(20),
                    use_container_width=True,
                    height=300
                )
            else:
                st.success("✅ Yüksek riskli ürün bulunamadı.")
    else:
        st.info("Risk analizi yapmak için butona tıklayın.")

def tahmin_modelleri_tab(df):
    """Tahmin Modelleri tab'ı"""
    st.markdown('<h2 class="section-title">🔮 SATIŞ TAHMİN MODELLERİ</h2>', unsafe_allow_html=True)
    
    # Tahmin modeli
    col1, col2 = st.columns([1, 2])
    
    with col1:
        tahmin_periyodu = st.selectbox("Tahmin Periyodu", [3, 6, 12, 24], key="tahmin_periyodu")
    
    with col2:
        if st.button("🚀 **TAHMİN MODELİ OLUŞTUR**", use_container_width=True):
            with st.spinner("Tahmin modeli oluşturuluyor..."):
                analitik = EnterpriseAnalitikMotoru()
                tahmin = analitik.satis_tahmini_modeli(df, tahmin_periyodu)
                
                if tahmin:
                    st.session_state.tahmin = tahmin
                    st.success(f"✅ {tahmin_periyodu} aylık tahmin modeli oluşturuldu!")
                    st.rerun()
    
    # Tahmin sonuçları
    if st.session_state.tahmin:
        gorsellestirme = EnterpriseGorsellestirme()
        tahmin_grafik = gorsellestirme.tahmin_grafikleri(st.session_state.tahmin)
        
        if tahmin_grafik:
            st.plotly_chart(tahmin_grafik, use_container_width=True, config={'displayModeBar': True})
        
        # Tahmin metrikleri
        st.markdown('<h3 class="subsection-title">📈 TAHMİN MODELİ PERFORMANSI</h3>', unsafe_allow_html=True)
        
        metrikler = st.session_state.tahmin['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", f"{metrikler['mae']:.2f}")
        
        with col2:
            st.metric("RMSE", f"{metrikler['rmse']:.2f}")
        
        with col3:
            st.metric("MAPE", f"%{metrikler['mape']:.1f}")
        
        with col4:
            st.metric("Tahmin Periyodu", f"{st.session_state.tahmin['tahmin_horizonu']} ay")
        
        # Gelecek tahminleri
        st.markdown('<h3 class="subsection-title">🔮 GELECEK TAHMİNLERİ</h3>', unsafe_allow_html=True)
        
        forecast = st.session_state.tahmin['forecast']
        son_tahminler = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(st.session_state.tahmin['tahmin_horizonu'])
        
        st.dataframe(son_tahminler, use_container_width=True)
    else:
        st.info("Tahmin modeli oluşturmak için butona tıklayın.")

def raporlama_tab(df, metrikler):
    """Raporlama tab'ı"""
    st.markdown('<h2 class="section-title">📑 RAPORLAMA VE DIŞA AKTARMA</h2>', unsafe_allow_html=True)
    
    # Rapor seçenekleri
    st.markdown('<h3 class="subsection-title">📊 RAPOR TÜRLERİ</h3>', unsafe_allow_html=True)
    
    rapor_turu = st.selectbox(
        "Rapor Türü Seçin",
        ['Excel Detaylı Rapor', 'HTML Özet Rapor', 'JSON Veri Paketi', 'CSV Ham Veri'],
        key="rapor_turu_secim"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 **EXCEL RAPORU**", use_container_width=True):
            raporlama = EnterpriseRaporlama()
            raporlama.excel_raporu_olustur(
                df, 
                metrikler, 
                st.session_state.segmentasyon,
                st.session_state.tahmin,
                st.session_state.risk_analizi
            )
    
    with col2:
        if st.button("🌐 **HTML RAPORU**", use_container_width=True):
            raporlama = EnterpriseRaporlama()
            raporlama.html_raporu_olustur(
                df,
                metrikler,
                st.session_state.segmentasyon,
                st.session_state.tahmin,
                st.session_state.risk_analizi
            )
    
    with col3:
        if st.button("🔄 **ANALİZİ SIFIRLA**", use_container_width=True):
            for key in ['enterprise_veri', 'filtrelenmis_veri', 'metrikler', 'stratejik_oneriler',
                       'segmentasyon', 'tahmin', 'risk_analizi', 'filtre_config']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Veri istatistikleri
    st.markdown('<h3 class="subsection-title">📈 VERİ İSTATİSTİKLERİ</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with col2:
        st.metric("Toplam Sütun", len(df.columns))
    
    with col3:
        bellek = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Bellek Kullanımı", f"{bellek:.1f} MB")
    
    with col4:
        st.metric("Analiz Metrik", len(metrikler) if metrikler else 0)
    
    # Performans özeti
    if 'toplam_performans' in metrikler:
        st.markdown('<h3 class="subsection-title">🏆 PERFORMANS ÖZETİ</h3>', unsafe_allow_html=True)
        
        performans = metrikler['toplam_performans']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(performans['bilesenler'].keys()),
                    y=list(performans['bilesenler'].values()),
                    marker_color='#2d7dd2'
                )
            ])
            
            fig.update_layout(
                title='Performans Bileşenleri',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                xaxis_title='Bileşen',
                yaxis_title='Skor (0-100)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="enterprise-metric-card primary">
                <div class="enterprise-metric-label">TOPLAM PERFORMANS</div>
                <div class="enterprise-metric-value">{performans['toplam_skor']:.1f}</div>
                <div class="enterprise-metric-trend">
                    <span class="badge badge-primary">/100</span>
                    <span>Genel Skor</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ================================================
# 8. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    try:
        # Garbage collection'ı etkinleştir
        gc.enable()
        
        # Ana uygulamayı başlat
        main()
        
    except Exception as e:
        # Hata yönetimi
        st.error(f"🚨 **ENTERPRISE UYGULAMA HATASI**")
        st.error(f"**Hata Mesajı:** {str(e)}")
        st.error("**Detaylı Hata Bilgisi:**")
        st.code(traceback.format_exc())
        
        # Yenileme butonu
        if st.button("🔄 **SAYFAYI YENİLE**", type="primary", use_container_width=True):
            st.rerun()




