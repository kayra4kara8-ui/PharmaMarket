# app.py - PharmaIntelligence Pro v6.0 - Tam Özellikli İlaç Pazarı Analiz Platformu
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from scipy import stats, integrate
from scipy.spatial.distance import cdist

# Yardımcı araçlar
from datetime import datetime, timedelta
import json
from io import BytesIO, StringIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import hashlib
import re
import textwrap
import itertools
from collections import Counter, defaultdict
import base64
import zipfile

# ================================================
# 1. PROFESYONEL KONFİGÜRASYON VE STİL AYARLARI
# ================================================
st.set_page_config(
    page_title="PharmaIntelligence Pro v6.0 | Enterprise Pharma Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://pharmaintelligence.com/enterprise-support',
        'Report a bug': "https://pharmaintelligence.com/enterprise-bug-report",
        'About': """
        ### PharmaIntelligence Enterprise v6.0
        • International Product Analytics
        • Predictive Modeling & AI
        • Real-time Market Intelligence
        • Advanced Segmentation & Clustering
        • Automated Reporting & Dashboarding
        • Machine Learning Integration
        • Price Optimization Engine
        • Competitive Intelligence Suite
        
        © 2024 PharmaIntelligence Inc. All Rights Reserved
        """
    }
)

# PROFESYONEL MAVİ TEMA CSS STYLES (GELİŞTİRİLMİŞ)
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
        --accent-purple: #9d4edd;
        --accent-pink: #ff5d8f;
        --success: #2dd2a3;
        --success-dark: #25b592;
        --warning: #f2c94c;
        --warning-dark: #f2b94c;
        --danger: #eb5757;
        --danger-dark: #d64545;
        --info: #2d7dd2;
        
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        --text-light: #94a3b8;
        
        --bg-primary: #0c1a32;
        --bg-secondary: #14274e;
        --bg-card: #1e3a5f;
        --bg-card-light: #2d4a7a;
        --bg-hover: #2d4a7a;
        --bg-surface: #14274e;
        --bg-overlay: rgba(0, 0, 0, 0.7);
        
        --shadow-xs: 0 1px 3px rgba(0, 0, 0, 0.2);
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
        --shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.6);
        --shadow-xxl: 0 20px 64px rgba(0, 0, 0, 0.7);
        
        --radius-xs: 4px;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        --radius-xxl: 24px;
        --radius-full: 9999px;
        
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: 250ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1);
        
        --z-dropdown: 1000;
        --z-sticky: 1020;
        --z-fixed: 1030;
        --z-modal-backdrop: 1040;
        --z-modal: 1050;
        --z-popover: 1060;
        --z-tooltip: 1070;
    }
    
    /* === GLOBAL STYLES === */
    .stApp {
        background: linear-gradient(135deg, var(--primary-dark), var(--secondary-dark));
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        color: var(--text-primary);
        min-height: 100vh;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Main content area */
    .main {
        padding: 0;
        max-width: 100%;
    }
    
    /* === STREAMLIT COMPONENT OVERRIDES === */
    /* Dataframes and tables */
    .stDataFrame, .stTable, .dataframe {
        background: var(--bg-card) !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--bg-hover) !important;
        font-size: 13px !important;
    }
    
    .stDataFrame th, .stTable th {
        background: var(--bg-card-light) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid var(--accent-blue) !important;
    }
    
    .stDataFrame td, .stTable td {
        border-bottom: 1px solid var(--bg-hover) !important;
        color: var(--text-secondary) !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 0.25rem !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* Input fields */
    .stSelectbox, .stMultiselect, .stTextInput, .stNumberInput, .stDateInput {
        background: var(--bg-card) !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--bg-hover) !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiselect div[data-baseweb="select"] > div {
        background: var(--bg-card) !important;
        border-color: var(--bg-hover) !important;
    }
    
    .stTextInput input, .stNumberInput input {
        color: var(--text-primary) !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        transition: all var(--transition-fast) !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    /* Slider */
    .stSlider {
        background: var(--bg-card) !important;
        padding: 1rem !important;
        border-radius: var(--radius-sm) !important;
    }
    
    .stSlider div[data-testid="stThumbValue"] {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: var(--bg-card) !important;
        padding: 8px !important;
        border-radius: var(--radius-md) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: var(--radius-sm) !important;
        padding: 10px 20px !important;
        transition: all var(--transition-fast) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-blue) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--bg-hover) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    
    /* Checkbox & Radio */
    .stCheckbox, .stRadio {
        color: var(--text-secondary) !important;
    }
    
    .stCheckbox label, .stRadio label {
        color: var(--text-secondary) !important;
    }
    
    /* === ANIMATIONS === */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: 200px 0; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* === CUSTOM COMPONENTS === */
    /* Typography */
    .pharma-title {
        font-size: 3rem;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan), var(--accent-teal), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        line-height: 1.1;
        animation: fadeIn 0.8s ease-out;
    }
    
    .pharma-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 800px;
        line-height: 1.6;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-out 0.2s both;
    }
    
    .section-title {
        font-size: 1.8rem;
        color: var(--text-primary);
        font-weight: 800;
        margin: 2.5rem 0 1.5rem 0;
        padding-left: 1rem;
        border-left: 5px solid var(--accent-blue);
        background: linear-gradient(90deg, rgba(45, 125, 210, 0.1), transparent);
        padding: 1rem;
        border-radius: var(--radius-sm);
        animation: slideIn 0.5s ease-out;
    }
    
    .section-title-cyan {
        border-left-color: var(--accent-cyan);
        background: linear-gradient(90deg, rgba(42, 202, 234, 0.1), transparent);
    }
    
    .section-title-teal {
        border-left-color: var(--accent-teal);
        background: linear-gradient(90deg, rgba(48, 201, 201, 0.1), transparent);
    }
    
    .section-title-purple {
        border-left-color: var(--accent-purple);
        background: linear-gradient(90deg, rgba(157, 78, 221, 0.1), transparent);
    }
    
    .subsection-title {
        font-size: 1.4rem;
        color: var(--text-primary);
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--bg-hover);
        animation: fadeIn 0.6s ease-out;
    }
    
    .card-title {
        font-size: 1.2rem;
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Custom Metric Cards */
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
        animation: fadeIn 0.5s ease-out;
    }
    
    .custom-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
        z-index: 1;
    }
    
    .custom-metric-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-xl);
        border-color: var(--accent-blue);
    }
    
    .custom-metric-card.primary {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-blue-dark));
    }
    
    .custom-metric-card.primary::before {
        background: linear-gradient(90deg, var(--accent-blue-light), white);
    }
    
    .custom-metric-card.warning {
        background: linear-gradient(135deg, var(--warning), var(--warning-dark));
    }
    
    .custom-metric-card.warning::before {
        background: linear-gradient(90deg, var(--warning), #ffd166);
    }
    
    .custom-metric-card.danger {
        background: linear-gradient(135deg, var(--danger), var(--danger-dark));
    }
    
    .custom-metric-card.danger::before {
        background: linear-gradient(90deg, var(--danger), #ff6b6b);
    }
    
    .custom-metric-card.success {
        background: linear-gradient(135deg, var(--success), var(--success-dark));
    }
    
    .custom-metric-card.success::before {
        background: linear-gradient(90deg, var(--success), #80ffdb);
    }
    
    .custom-metric-card.info {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-teal));
    }
    
    .custom-metric-card.info::before {
        background: linear-gradient(90deg, var(--accent-cyan), #90e0ef);
    }
    
    .custom-metric-card.purple {
        background: linear-gradient(135deg, var(--accent-purple), #7b2cbf);
    }
    
    .custom-metric-card.purple::before {
        background: linear-gradient(90deg, var(--accent-purple), #c77dff);
    }
    
    .custom-metric-card.pink {
        background: linear-gradient(135deg, var(--accent-pink), #ff477e);
    }
    
    .custom-metric-card.pink::before {
        background: linear-gradient(90deg, var(--accent-pink), #ffafcc);
    }
    
    .custom-metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0.5rem 0;
        color: var(--text-primary);
        line-height: 1;
        font-family: 'Inter', sans-serif;
    }
    
    .custom-metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .custom-metric-trend {
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
        margin-top: 0.5rem;
        color: var(--text-light);
    }
    
    .trend-up { color: var(--success); }
    .trend-down { color: var(--danger); }
    .trend-neutral { color: var(--text-muted); }
    
    /* Insight Cards */
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
        animation: fadeIn 0.6s ease-out;
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
        z-index: 0;
    }
    
    .insight-card:hover::before {
        opacity: 1;
    }
    
    .insight-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
        border-left-width: 6px;
    }
    
    .insight-card.info { border-left-color: var(--accent-blue); }
    .insight-card.success { border-left-color: var(--success); }
    .insight-card.warning { border-left-color: var(--warning); }
    .insight-card.danger { border-left-color: var(--danger); }
    .insight-card.purple { border-left-color: var(--accent-purple); }
    .insight-card.pink { border-left-color: var(--accent-pink); }
    .insight-card.cyan { border-left-color: var(--accent-cyan); }
    .insight-card.teal { border-left-color: var(--accent-teal); }
    
    .insight-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        animation: float 3s ease-in-out infinite;
    }
    
    .insight-title {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        line-height: 1.3;
    }
    
    .insight-content {
        color: var(--text-secondary);
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-hover));
        padding: 1.8rem;
        border-radius: var(--radius-md);
        border-left: 4px solid;
        transition: all var(--transition-normal);
        height: 100%;
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.6s ease-out;
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: var(--shadow-xl);
        z-index: var(--z-popover);
    }
    
    .feature-card::after {
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
    
    .feature-card:hover::after {
        opacity: 1;
    }
    
    .feature-card-blue { border-left-color: var(--accent-blue); }
    .feature-card-cyan { border-left-color: var(--accent-cyan); }
    .feature-card-teal { border-left-color: var(--accent-teal); }
    .feature-card-warning { border-left-color: var(--warning); }
    .feature-card-purple { border-left-color: var(--accent-purple); }
    .feature-card-pink { border-left-color: var(--accent-pink); }
    .feature-card-success { border-left-color: var(--success); }
    .feature-card-danger { border-left-color: var(--danger); }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    
    .feature-title {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
    }
    
    .feature-description {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Welcome Container */
    .welcome-container {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-secondary));
        padding: 4rem 3rem;
        border-radius: var(--radius-xxl);
        box-shadow: var(--shadow-xxl);
        text-align: center;
        margin: 3rem auto;
        max-width: 1000px;
        border: 1px solid var(--bg-hover);
        position: relative;
        overflow: hidden;
        animation: fadeIn 0.8s ease-out;
    }
    
    .welcome-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        right: -50%;
        bottom: -50%;
        background: linear-gradient(45deg, 
            transparent 0%, 
            rgba(45, 125, 210, 0.1) 25%, 
            rgba(42, 202, 234, 0.1) 50%, 
            rgba(48, 201, 201, 0.1) 75%, 
            transparent 100%);
        animation: shimmer 8s linear infinite;
        z-index: 0;
    }
    
    .welcome-container > * {
        position: relative;
        z-index: 1;
    }
    
    .welcome-icon {
        font-size: 6rem;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan), var(--accent-teal), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        animation: float 4s ease-in-out infinite;
    }
    
    /* Get Started Box */
    .get-started-box {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.2), rgba(42, 202, 234, 0.15), rgba(48, 201, 201, 0.1));
        padding: 2rem;
        border-radius: var(--radius-xl);
        border: 1px solid rgba(45, 125, 210, 0.4);
        margin-top: 3rem;
        backdrop-filter: blur(10px);
        animation: fadeIn 1s ease-out 0.5s both;
    }
    
    .get-started-title {
        font-weight: 700;
        color: var(--accent-blue);
        margin-bottom: 1rem;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .get-started-steps {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.7;
    }
    
    .get-started-steps br {
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Filter Section */
    .filter-section {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        margin-bottom: 1.5rem;
        border: 1px solid var(--bg-hover);
        animation: slideIn 0.5s ease-out;
    }
    
    .filter-title {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Filter Status */
    .filter-status {
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.25), rgba(42, 202, 234, 0.2));
        padding: 1.2rem;
        border-radius: var(--radius-md);
        margin-bottom: 2rem;
        border-left: 5px solid var(--success);
        box-shadow: var(--shadow-md);
        color: var(--text-primary);
        font-size: 0.95rem;
        animation: fadeIn 0.5s ease-out;
    }
    
    .filter-status-danger {
        background: linear-gradient(135deg, rgba(235, 87, 87, 0.25), rgba(214, 69, 69, 0.2));
        border-left: 5px solid var(--warning);
    }
    
    .filter-status-warning {
        background: linear-gradient(135deg, rgba(242, 201, 76, 0.25), rgba(242, 185, 76, 0.2));
        border-left: 5px solid var(--accent-blue);
    }
    
    /* Search Box */
    .search-box {
        background: var(--bg-card);
        border: 1px solid var(--bg-hover);
        border-radius: var(--radius-sm);
        padding: 0.8rem 1.2rem;
        color: var(--text-primary);
        font-size: 0.95rem;
        transition: all var(--transition-fast);
        width: 100%;
        font-family: 'Inter', sans-serif;
    }
    
    .search-box:focus {
        outline: none;
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 3px rgba(45, 125, 210, 0.2);
    }
    
    /* Data Grid Container */
    .data-grid-container {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--bg-hover);
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Loading Animation */
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    .loading-spinner {
        animation: spin 1s linear infinite;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-online { background: var(--success); box-shadow: 0 0 8px var(--success); }
    .status-warning { background: var(--warning); box-shadow: 0 0 8px var(--warning); }
    .status-error { background: var(--danger); box-shadow: 0 0 8px var(--danger); }
    .status-processing { background: var(--accent-blue); box-shadow: 0 0 8px var(--accent-blue); }
    
    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: var(--radius-full);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        line-height: 1;
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
    
    .badge-danger {
        background: rgba(235, 87, 87, 0.2);
        color: var(--danger);
        border: 1px solid rgba(235, 87, 87, 0.3);
    }
    
    .badge-info {
        background: rgba(45, 125, 210, 0.2);
        color: var(--accent-blue);
        border: 1px solid rgba(45, 125, 210, 0.3);
    }
    
    .badge-purple {
        background: rgba(157, 78, 221, 0.2);
        color: var(--accent-purple);
        border: 1px solid rgba(157, 78, 221, 0.3);
    }
    
    .badge-pink {
        background: rgba(255, 93, 143, 0.2);
        color: var(--accent-pink);
        border: 1px solid rgba(255, 93, 143, 0.3);
    }
    
    /* Charts and Graphs */
    .chart-container {
        background: var(--bg-card);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        margin-bottom: 1.5rem;
        animation: fadeIn 0.7s ease-out;
    }
    
    /* Tooltips */
    .custom-tooltip {
        background: var(--bg-card-light);
        border: 1px solid var(--bg-hover);
        border-radius: var(--radius-sm);
        padding: 0.75rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
        box-shadow: var(--shadow-md);
        max-width: 300px;
        z-index: var(--z-tooltip);
    }
    
    /* Progress Bars */
    .progress-bar {
        background: var(--bg-hover);
        border-radius: var(--radius-full);
        overflow: hidden;
        height: 8px;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
        border-radius: var(--radius-full);
        transition: width 0.5s ease-out;
    }
    
    /* Sidebar */
    .sidebar-title {
        font-size: 1.5rem;
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--accent-blue);
        animation: fadeIn 0.5s ease-out;
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
        animation: slideIn 0.5s ease-out;
    }
    
    /* Alerts and Notifications */
    .alert {
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
        border-left: 4px solid;
        animation: fadeIn 0.5s ease-out;
    }
    
    .alert-success {
        background: rgba(45, 210, 163, 0.1);
        border-left-color: var(--success);
        color: var(--success);
    }
    
    .alert-warning {
        background: rgba(242, 201, 76, 0.1);
        border-left-color: var(--warning);
        color: var(--warning);
    }
    
    .alert-error {
        background: rgba(235, 87, 87, 0.1);
        border-left-color: var(--danger);
        color: var(--danger);
    }
    
    .alert-info {
        background: rgba(45, 125, 210, 0.1);
        border-left-color: var(--accent-blue);
        color: var(--accent-blue);
    }
    
    /* Utility Classes */
    .text-gradient {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .text-truncate {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .text-multiline-truncate {
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .glass-effect {
        background: rgba(30, 58, 95, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hover-lift:hover {
        transform: translateY(-2px);
        transition: transform var(--transition-fast);
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-card);
        border-radius: var(--radius-full);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-blue);
        border-radius: var(--radius-full);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-blue-light);
    }
    
    /* Print Styles */
    @media print {
        .no-print {
            display: none !important;
        }
        
        .print-only {
            display: block !important;
        }
    }
    
    /* Responsive Styles */
    @media (max-width: 768px) {
        .pharma-title {
            font-size: 2.2rem;
        }
        
        .section-title {
            font-size: 1.5rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
        
        .welcome-container {
            padding: 2rem 1.5rem;
        }
        
        .custom-metric-value {
            font-size: 1.8rem;
        }
    }
</style>
"""

# CSS'i uygula
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ================================================
# 2. GELİŞMİŞ VERİ İŞLEME MOTORU
# ================================================

class AdvancedDataProcessor:
    """Gelişmiş veri işleme motoru - Büyük veri setleri için optimize edilmiş"""
    
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
    def load_and_process_data(uploaded_file, sample_size=None, chunk_size=50000):
        """Veriyi yükle ve işle"""
        try:
            start_time = time.time()
            
            # Dosya tipine göre yükleme
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = AdvancedDataProcessor._load_csv(uploaded_file, sample_size, chunk_size)
            elif file_extension in ['xlsx', 'xls']:
                df = AdvancedDataProcessor._load_excel(uploaded_file, sample_size, chunk_size)
            else:
                st.error(f"Desteklenmeyen dosya formatı: {file_extension}")
                return None
            
            if df is None or len(df) == 0:
                st.error("Veri yüklenemedi veya boş veri seti")
                return None
            
            # İşlem süresi
            processing_time = time.time() - start_time
            
            # Başarı mesajı
            st.success(f"""
            ✅ **Veri Başarıyla Yüklendi!**
            • **Satır Sayısı:** {len(df):,}
            • **Sütun Sayısı:** {len(df.columns)}
            • **İşlem Süresi:** {processing_time:.2f}s
            • **Bellek Kullanımı:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
            """)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.error(f"**Detay:** {traceback.format_exc()}")
            return None
    
    @staticmethod
    def _load_csv(uploaded_file, sample_size, chunk_size):
        """CSV dosyasını yükle"""
        if sample_size:
            df = pd.read_csv(uploaded_file, nrows=sample_size, low_memory=False)
        else:
            # Büyük dosyalar için chunk okuma
            chunks = []
            with st.spinner("📥 Büyük CSV dosyası yükleniyor..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Toplam satır sayısını tahmin et
                total_rows = AdvancedDataProcessor._estimate_csv_rows(uploaded_file)
                processed_rows = 0
                
                for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size, low_memory=False):
                    chunks.append(chunk)
                    processed_rows += len(chunk)
                    
                    # İlerleme durumu
                    if total_rows > 0:
                        progress = min(processed_rows / total_rows, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"📊 {processed_rows:,} / {total_rows:,} satır yüklendi...")
                
                df = pd.concat(chunks, ignore_index=True)
                progress_bar.progress(1.0)
                status_text.text(f"✅ {len(df):,} satır başarıyla yüklendi!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
        
        return df
    
    @staticmethod
    def _load_excel(uploaded_file, sample_size, chunk_size):
        """Excel dosyasını yükle"""
        try:
            if sample_size:
                df = pd.read_excel(uploaded_file, nrows=sample_size, engine='openpyxl')
            else:
                with st.spinner("📥 Excel dosyası yükleniyor..."):
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            return df
            
        except Exception as e:
            st.warning(f"Excel yükleme hatası: {str(e)}. Alternatif yöntem deneniyor...")
            # Alternatif yöntem
            try:
                df = pd.read_excel(uploaded_file, engine='xlrd')
                return df
            except:
                st.error("Excel dosyası yüklenemedi. Lütfen dosya formatını kontrol edin.")
                return None
    
    @staticmethod
    def _estimate_csv_rows(uploaded_file):
        """CSV dosyasındaki toplam satır sayısını tahmin et"""
        try:
            # Dosyanın başlangıcını oku
            content = uploaded_file.read(1024 * 1024)  # 1MB oku
            uploaded_file.seek(0)  # Dosya imlecini sıfırla
            
            # Satır sayısını tahmin et
            line_count = content.count(b'\n')
            return line_count
        except:
            return 0
    
    @staticmethod
    def clean_column_names(df):
        """Sütun isimlerini temizle ve standardize et"""
        try:
            original_columns = df.columns.tolist()
            cleaned_columns = []
            
            for col in original_columns:
                if not isinstance(col, str):
                    col = str(col)
                
                # Temel temizlik
                col = col.strip()
                col = col.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                col = ' '.join(col.split())  # Multiple spaces'leri temizle
                
                # Türkçe karakterleri düzelt
                turkish_chars = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr_char, en_char in turkish_chars.items():
                    col = col.replace(tr_char, en_char)
                
                # Özel karakterleri temizle
                col = re.sub(r'[^\w\s]', '_', col)
                col = re.sub(r'\s+', '_', col)
                col = col.strip('_')
                
                # Benzersiz isim yap
                base_col = col
                counter = 1
                while col in cleaned_columns:
                    col = f"{base_col}_{counter}"
                    counter += 1
                
                cleaned_columns.append(col)
            
            df.columns = cleaned_columns
            
            # Değişiklikleri logla
            changes = []
            for orig, new in zip(original_columns, cleaned_columns):
                if orig != new:
                    changes.append(f"{orig} → {new}")
            
            if changes:
                with st.expander("📝 Sütun İsimleri Düzeltildi", expanded=False):
                    for change in changes[:20]:  # İlk 20 değişikliği göster
                        st.write(f"• {change}")
                    if len(changes) > 20:
                        st.write(f"... ve {len(changes) - 20} daha")
            
            return df
            
        except Exception as e:
            st.warning(f"Sütun isimleri temizleme hatası: {str(e)}")
            return df
    
    @staticmethod
    def optimize_dataframe(df):
        """DataFrame'i optimize et - Bellek kullanımını azalt"""
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Numerik sütunları optimize et
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                try:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    
                    # Tamsayı sütunları
                    if pd.api.types.is_integer_dtype(df[col]):
                        if col_min >= 0:  # Unsigned
                            if col_max <= 255:
                                df[col] = df[col].astype(np.uint8)
                            elif col_max <= 65535:
                                df[col] = df[col].astype(np.uint16)
                            elif col_max <= 4294967295:
                                df[col] = df[col].astype(np.uint32)
                            else:
                                df[col] = df[col].astype(np.uint64)
                        else:  # Signed
                            if col_min >= -128 and col_max <= 127:
                                df[col] = df[col].astype(np.int8)
                            elif col_min >= -32768 and col_max <= 32767:
                                df[col] = df[col].astype(np.int16)
                            elif col_min >= -2147483648 and col_max <= 2147483647:
                                df[col] = df[col].astype(np.int32)
                            else:
                                df[col] = df[col].astype(np.int64)
                    # Float sütunları
                    else:
                        df[col] = df[col].astype(np.float32)
                        
                except Exception as e:
                    continue  # Bu sütunu atla
            
            # Kategorik sütunları optimize et
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5:  # Eşsiz değerlerin oranı düşükse
                        df[col] = df[col].astype('category')
                except:
                    continue
            
            # Null değerleri optimize et
            for col in df.columns:
                if df[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(0)
                    elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                        df[col] = df[col].fillna('Unknown')
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_saved = original_memory - optimized_memory
            
            if memory_saved > 0:
                st.info(f"""
                💾 **Bellek Optimizasyonu Başarılı!**
                • **Önce:** {original_memory:.1f} MB
                • **Sonra:** {optimized_memory:.1f} MB  
                • **Tasarruf:** {memory_saved:.1f} MB (%{(memory_saved/original_memory)*100:.1f})
                """)
            
            return df
            
        except Exception as e:
            st.warning(f"DataFrame optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def prepare_analysis_data(df):
        """Analiz için veriyi hazırla"""
        try:
            # Önce veriyi kopyala
            analysis_df = df.copy()
            
            # 1. Satış verilerini bul
            sales_cols = [col for col in analysis_df.columns if any(x in col.lower() for x in ['sales', 'satış', 'revenue', 'gelir'])]
            
            # 2. Yılları bul
            years = []
            for col in sales_cols:
                # Sütun adından yıl bul
                year_match = re.search(r'(20\d{2})', col)
                if year_match:
                    year = int(year_match.group(1))
                    years.append(year)
            
            # Benzersiz yılları sırala
            years = sorted(list(set(years)))
            
            if len(years) >= 2:
                # 3. Büyüme oranlarını hesapla
                for i in range(1, len(years)):
                    prev_year = years[i-1]
                    curr_year = years[i]
                    
                    # Satış sütunlarını bul
                    prev_sales_col = next((col for col in sales_cols if str(prev_year) in col), None)
                    curr_sales_col = next((col for col in sales_cols if str(curr_year) in col), None)
                    
                    if prev_sales_col and curr_sales_col:
                        growth_col_name = f"Growth_{prev_year}_{curr_year}"
                        analysis_df[growth_col_name] = np.where(
                            analysis_df[prev_sales_col] != 0,
                            ((analysis_df[curr_sales_col] - analysis_df[prev_sales_col]) / analysis_df[prev_sales_col]) * 100,
                            np.nan
                        )
                
                # 4. CAGR hesapla
                if len(years) >= 2:
                    first_year = years[0]
                    last_year = years[-1]
                    
                    first_sales_col = next((col for col in sales_cols if str(first_year) in col), None)
                    last_sales_col = next((col for col in sales_cols if str(last_year) in col), None)
                    
                    if first_sales_col and last_sales_col:
                        periods = len(years)
                        analysis_df['CAGR_%'] = np.where(
                            analysis_df[first_sales_col] > 0,
                            ((analysis_df[last_sales_col] / analysis_df[first_sales_col]) ** (1/periods) - 1) * 100,
                            np.nan
                        )
            
            # 5. Pazar payı hesapla (en son yıl için)
            if years:
                last_year = years[-1]
                last_sales_col = next((col for col in sales_cols if str(last_year) in col), None)
                
                if last_sales_col:
                    total_sales = analysis_df[last_sales_col].sum()
                    if total_sales > 0:
                        analysis_df['Market_Share_%'] = (analysis_df[last_sales_col] / total_sales) * 100
            
            # 6. Fiyat verilerini kontrol et
            price_cols = [col for col in analysis_df.columns if any(x in col.lower() for x in ['price', 'fiyat', 'unit_price'])]
            
            if not price_cols:
                # Fiyat sütunu yoksa hesapla (Satış / Birim)
                unit_cols = [col for col in analysis_df.columns if any(x in col.lower() for x in ['unit', 'birim', 'quantity', 'adet'])]
                
                if sales_cols and unit_cols and years:
                    last_year = years[-1]
                    last_sales_col = next((col for col in sales_cols if str(last_year) in col), None)
                    last_unit_col = next((col for col in unit_cols if str(last_year) in col), None)
                    
                    if last_sales_col and last_unit_col:
                        analysis_df[f'Calculated_Price_{last_year}'] = np.where(
                            analysis_df[last_unit_col] > 0,
                            analysis_df[last_sales_col] / analysis_df[last_unit_col],
                            np.nan
                        )
                        price_cols = [f'Calculated_Price_{last_year}']
            
            # 7. Performans skoru oluştur
            numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                try:
                    scaler = StandardScaler()
                    numeric_data = analysis_df[numeric_cols].fillna(0)
                    scaled_data = scaler.fit_transform(numeric_data)
                    analysis_df['Performance_Score'] = scaled_data.mean(axis=1)
                except:
                    pass
            
            # 8. Segmentasyon için özellikler
            analysis_df['Sales_Growth_Ratio'] = analysis_df.get('Growth_2023_2024', 0) / 100 if 'Growth_2023_2024' in analysis_df.columns else 0
            
            st.success("✅ **Veri Analizi Hazırlandı!**")
            return analysis_df
            
        except Exception as e:
            st.warning(f"Analiz verisi hazırlama hatası: {str(e)}")
            return df
    
    @staticmethod
    def detect_data_patterns(df):
        """Veri desenlerini ve yapısını tespit et"""
        try:
            patterns = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
                'date_columns': len(df.select_dtypes(include=['datetime']).columns),
                'missing_values': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'duplicate_rows': df.duplicated().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
            
            # Sütun kategorilerini tespit et
            column_categories = {}
            for col in df.columns:
                col_lower = col.lower()
                
                if any(x in col_lower for x in ['sales', 'revenue', 'gelir', 'ciro', 'satış']):
                    column_categories[col] = 'Sales'
                elif any(x in col_lower for x in ['price', 'fiyat', 'cost', 'maliyet']):
                    column_categories[col] = 'Price'
                elif any(x in col_lower for x in ['unit', 'quantity', 'adet', 'birim']):
                    column_categories[col] = 'Quantity'
                elif any(x in col_lower for x in ['growth', 'büyüme', 'artış', 'değişim']):
                    column_categories[col] = 'Growth'
                elif any(x in col_lower for x in ['country', 'ülke', 'region', 'bölge']):
                    column_categories[col] = 'Geography'
                elif any(x in col_lower for x in ['company', 'firma', 'şirket', 'manufacturer', 'üretici']):
                    column_categories[col] = 'Company'
                elif any(x in col_lower for x in ['molecule', 'molekül', 'product', 'ürün', 'drug', 'ilaç']):
                    column_categories[col] = 'Product'
                elif any(x in col_lower for x in ['date', 'tarih', 'year', 'yıl', 'month', 'ay']):
                    column_categories[col] = 'Date'
                elif any(x in col_lower for x in ['share', 'pay', 'pazar', 'market']):
                    column_categories[col] = 'Market Share'
                else:
                    column_categories[col] = 'Other'
            
            patterns['column_categories'] = column_categories
            
            return patterns
            
        except Exception as e:
            st.warning(f"Veri deseni tespiti hatası: {str(e)}")
            return {}

# ================================================
# 3. GELİŞMİŞ ANALİTİK MOTORU
# ================================================

class AdvancedAnalyticsEngine:
    """Gelişmiş analitik motoru - Çoklu analiz yöntemleri"""
    
    @staticmethod
    def calculate_comprehensive_metrics(df):
        """Kapsamlı pazar metriklerini hesapla"""
        try:
            metrics = {}
            
            # Temel metrikler
            metrics['total_rows'] = len(df)
            metrics['total_columns'] = len(df.columns)
            metrics['total_products'] = len(df)
            
            # Satış metrikleri
            sales_cols = [col for col in df.columns if any(x in col.lower() for x in ['sales', 'satış', 'revenue'])]
            if sales_cols:
                # En son satış sütununu bul
                last_sales_col = None
                years = []
                for col in sales_cols:
                    year_match = re.search(r'(20\d{2})', col)
                    if year_match:
                        years.append(int(year_match.group(1)))
                
                if years:
                    last_year = max(years)
                    last_sales_col = next((col for col in sales_cols if str(last_year) in col), sales_cols[-1])
                    
                    if last_sales_col in df.columns:
                        sales_data = df[last_sales_col]
                        
                        metrics['total_market_value'] = sales_data.sum()
                        metrics['avg_sales_per_product'] = sales_data.mean()
                        metrics['median_sales'] = sales_data.median()
                        metrics['sales_std'] = sales_data.std()
                        metrics['sales_q1'] = sales_data.quantile(0.25)
                        metrics['sales_q3'] = sales_data.quantile(0.75)
                        metrics['sales_iqr'] = metrics['sales_q3'] - metrics['sales_q1']
                        
                        # Satış segmentleri
                        sales_segments = pd.cut(
                            sales_data,
                            bins=[0, sales_data.quantile(0.25), sales_data.quantile(0.5), 
                                  sales_data.quantile(0.75), sales_data.max()],
                            labels=['Low', 'Medium', 'High', 'Very High']
                        )
                        metrics['sales_segment_counts'] = sales_segments.value_counts().to_dict()
            
            # Büyüme metrikleri
            growth_cols = [col for col in df.columns if any(x in col.lower() for x in ['growth', 'büyüme'])]
            if growth_cols:
                last_growth_col = growth_cols[-1]
                if last_growth_col in df.columns:
                    growth_data = df[last_growth_col]
                    
                    metrics['avg_growth_rate'] = growth_data.mean()
                    metrics['median_growth'] = growth_data.median()
                    metrics['positive_growth_products'] = (growth_data > 0).sum()
                    metrics['negative_growth_products'] = (growth_data < 0).sum()
                    metrics['high_growth_products'] = (growth_data > 20).sum()
                    metrics['high_growth_percentage'] = (metrics['high_growth_products'] / len(df)) * 100
            
            # Fiyat metrikleri
            price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'fiyat'])]
            if price_cols:
                last_price_col = price_cols[-1]
                if last_price_col in df.columns:
                    price_data = df[last_price_col]
                    
                    metrics['avg_price'] = price_data.mean()
                    metrics['price_variance'] = price_data.var()
                    metrics['price_std'] = price_data.std()
                    
                    # Fiyat segmentleri
                    price_quartiles = price_data.quantile([0.25, 0.5, 0.75])
                    metrics['price_q1'] = price_quartiles[0.25]
                    metrics['price_median'] = price_quartiles[0.5]
                    metrics['price_q3'] = price_quartiles[0.75]
            
            # Şirket analizi
            if 'Şirket' in df.columns and sales_cols:
                company_sales = df.groupby('Şirket')[last_sales_col].sum().sort_values(ascending=False)
                total_sales = company_sales.sum()
                
                if total_sales > 0:
                    # HHI indeksi
                    market_shares = (company_sales / total_sales * 100)
                    metrics['hhi_index'] = (market_shares ** 2).sum() / 10000
                    
                    # Pazar konsantrasyonu
                    for n in [1, 3, 5, 10]:
                        metrics[f'top_{n}_company_share'] = company_sales.nlargest(n).sum() / total_sales * 100
                    
                    metrics['total_companies'] = len(company_sales)
                    metrics['top_company'] = company_sales.index[0]
                    metrics['top_company_share'] = (company_sales.iloc[0] / total_sales) * 100
            
            # Ürün çeşitliliği
            if 'Molekül' in df.columns:
                metrics['unique_molecules'] = df['Molekül'].nunique()
                if sales_cols:
                    molecule_sales = df.groupby('Molekül')[last_sales_col].sum()
                    total_molecule_sales = molecule_sales.sum()
                    if total_molecule_sales > 0:
                        metrics['top_10_molecule_share'] = molecule_sales.nlargest(10).sum() / total_molecule_sales * 100
            
            # Coğrafi dağılım
            if 'Ülke' in df.columns:
                metrics['country_coverage'] = df['Ülke'].nunique()
                if sales_cols:
                    country_sales = df.groupby('Ülke')[last_sales_col].sum()
                    total_country_sales = country_sales.sum()
                    if total_country_sales > 0:
                        metrics['top_5_country_share'] = country_sales.nlargest(5).sum() / total_country_sales * 100
            
            # International Product analizi
            if 'International_Product' in df.columns:
                intl_products = df[df['International_Product'] == 1]
                local_products = df[df['International_Product'] == 0]
                
                metrics['intl_product_count'] = len(intl_products)
                metrics['local_product_count'] = len(local_products)
                
                if sales_cols:
                    intl_sales = intl_products[last_sales_col].sum()
                    local_sales = local_products[last_sales_col].sum()
                    total_sales_val = metrics.get('total_market_value', intl_sales + local_sales)
                    
                    metrics['intl_product_sales'] = intl_sales
                    metrics['local_product_sales'] = local_sales
                    metrics['intl_product_share'] = (intl_sales / total_sales_val) * 100 if total_sales_val > 0 else 0
                    metrics['local_product_share'] = (local_sales / total_sales_val) * 100 if total_sales_val > 0 else 0
                
                # Büyüme karşılaştırması
                if growth_cols:
                    intl_growth = intl_products[last_growth_col].mean() if len(intl_products) > 0 else 0
                    local_growth = local_products[last_growth_col].mean() if len(local_products) > 0 else 0
                    metrics['intl_avg_growth'] = intl_growth
                    metrics['local_avg_growth'] = local_growth
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return {}
    
    @staticmethod
    def generate_strategic_insights(df, metrics):
        """Stratejik içgörüler oluştur"""
        try:
            insights = []
            
            sales_cols = [col for col in df.columns if any(x in col.lower() for x in ['sales', 'satış'])]
            if not sales_cols:
                return insights
            
            last_sales_col = sales_cols[-1]
            year_match = re.search(r'(20\d{2})', last_sales_col)
            year = year_match.group(1) if year_match else ""
            
            # 1. Pazar liderliği içgörüleri
            if 'Şirket' in df.columns:
                company_sales = df.groupby('Şirket')[last_sales_col].sum().sort_values(ascending=False)
                if len(company_sales) > 0:
                    top_company = company_sales.index[0]
                    top_company_share = (company_sales.iloc[0] / company_sales.sum()) * 100
                    
                    insights.append({
                        'type': 'market_leadership',
                        'title': f'🏆 Pazar Lideri - {year}',
                        'content': f"**{top_company}** %{top_company_share:.1f} pazar payı ile lider konumda.",
                        'priority': 'high',
                        'icon': '🏆',
                        'color': 'warning'
                    })
            
            # 2. Büyüme fırsatları
            growth_cols = [col for col in df.columns if any(x in col.lower() for x in ['growth', 'büyüme'])]
            if growth_cols:
                last_growth_col = growth_cols[-1]
                high_growth_df = df[df[last_growth_col] > 20]
                
                if len(high_growth_df) > 0:
                    top_growth_product = high_growth_df.nlargest(1, last_growth_col)
                    if len(top_growth_product) > 0:
                        product_name = top_growth_product.iloc[0]['Molekül'] if 'Molekül' in top_growth_product.columns else "Ürün"
                        growth_rate = top_growth_product.iloc[0][last_growth_col]
                        
                        insights.append({
                            'type': 'growth_opportunity',
                            'title': f'🚀 En Hızlı Büyüyen',
                            'content': f"**{product_name}** %{growth_rate:.1f} büyüme ile en hızlı büyüyen ürün.",
                            'priority': 'medium',
                            'icon': '🚀',
                            'color': 'success'
                        })
            
            # 3. Fiyat optimizasyonu
            price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'fiyat'])]
            if price_cols:
                last_price_col = price_cols[-1]
                price_stats = df[last_price_col].describe()
                
                insights.append({
                    'type': 'price_analysis',
                    'title': f'💰 Fiyat Analizi',
                    'content': f"Ortalama fiyat: ${price_stats['mean']:.2f}, Standart sapma: ${price_stats['std']:.2f}",
                    'priority': 'medium',
                    'icon': '💰',
                    'color': 'info'
                })
            
            # 4. International Product analizi
            if 'International_Product' in df.columns:
                intl_count = (df['International_Product'] == 1).sum()
                intl_percentage = (intl_count / len(df)) * 100
                
                if intl_percentage < 30:
                    insights.append({
                        'type': 'international_expansion',
                        'title': '🌍 Global Fırsat',
                        'content': f"International product oranı %{intl_percentage:.1f}. Global pazara açılmak için fırsat var.",
                        'priority': 'high',
                        'icon': '🌍',
                        'color': 'purple'
                    })
            
            # 5. Pazar konsantrasyonu
            hhi = metrics.get('hhi_index', 0)
            if hhi > 2500:
                concentration_level = "Yüksek (Monopolistik)"
            elif hhi > 1800:
                concentration_level = "Orta (Oligopol)"
            else:
                concentration_level = "Düşük (Rekabetçi)"
            
            insights.append({
                'type': 'market_concentration',
                'title': '📊 Pazar Yoğunluğu',
                'content': f"HHI indeksi: {hhi:.0f} - {concentration_level}",
                'priority': 'low',
                'icon': '📊',
                'color': 'cyan'
            })
            
            # 6. Ürün çeşitliliği
            if 'Molekül' in df.columns:
                unique_molecules = df['Molekül'].nunique()
                avg_products_per_molecule = len(df) / unique_molecules if unique_molecules > 0 else 0
                
                insights.append({
                    'type': 'product_diversity',
                    'title': '🧪 Ürün Çeşitliliği',
                    'content': f"{unique_molecules} benzersiz molekül, ortalama {avg_products_per_molecule:.1f} ürün/molekül",
                    'priority': 'low',
                    'icon': '🧪',
                    'color': 'teal'
                })
            
            # Önceliğe göre sırala
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            insights.sort(key=lambda x: priority_order.get(x['priority'], 3))
            
            return insights
            
        except Exception as e:
            st.warning(f"İçgörü oluşturma hatası: {str(e)}")
            return []
    
    @staticmethod
    def perform_market_segmentation(df, n_clusters=4, method='kmeans'):
        """Pazar segmentasyonu analizi"""
        try:
            # Özellik seçimi
            features = []
            
            # Satış özellikleri
            sales_cols = [col for col in df.columns if any(x in col.lower() for x in ['sales', 'satış'])]
            if sales_cols:
                features.extend(sales_cols[-2:])  # Son 2 yıl
            
            # Büyüme özellikleri
            growth_cols = [col for col in df.columns if any(x in col.lower() for x in ['growth', 'büyüme'])]
            if growth_cols:
                features.append(growth_cols[-1])
            
            # Fiyat özellikleri
            price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'fiyat'])]
            if price_cols:
                features.append(price_cols[-1])
            
            # Pazar payı
            if 'Market_Share_%' in df.columns:
                features.append('Market_Share_%')
            
            # Yeterli özellik kontrolü
            if len(features) < 2:
                st.warning("Segmentasyon için yeterli özellik bulunamadı.")
                return None
            
            # Veriyi hazırla
            segmentation_data = df[features].fillna(0)
            
            # Küme sayısını kontrol et
            if len(segmentation_data) < n_clusters * 10:
                st.warning("Segmentasyon için yeterli veri noktası yok.")
                return None
            
            # Normalizasyon
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(segmentation_data)
            
            # Segmentasyon
            if method == 'kmeans':
                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10,
                    max_iter=300,
                    init='k-means++'
                )
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=10)
            else:
                model = KMeans(n_clusters=n_clusters, random_state=42)
            
            clusters = model.fit_predict(scaled_features)
            
            # Sonuçları DataFrame'e ekle
            result_df = df.copy()
            result_df['Cluster'] = clusters
            
            # Küme isimlendirme
            cluster_names = {
                0: '📈 Gelişen Ürünler (Yüksek Büyüme)',
                1: '💰 Premium Ürünler (Yüksek Fiyat)',
                2: '📦 Hacim Ürünleri (Yüksek Satış)',
                3: '⚡ Yenilikçi Ürünler (Potansiyel)',
                4: '🛡️ Sağlam Ürünler (Stabil)',
                5: '🎯 Niş Ürünler (Özel Pazar)',
                6: '⚠️ Riskli Ürünler (Düşük Performans)',
                7: '🌱 Yeni Ürünler (Başlangıç)'
            }
            
            result_df['Cluster_Name'] = result_df['Cluster'].map(
                lambda x: cluster_names.get(x, f'Küme {x}')
            )
            
            # Küme analizi
            cluster_analysis = {}
            for cluster_num in sorted(result_df['Cluster'].unique()):
                cluster_df = result_df[result_df['Cluster'] == cluster_num]
                cluster_stats = {}
                
                # Satış istatistikleri
                if sales_cols:
                    last_sales_col = sales_cols[-1]
                    cluster_stats['avg_sales'] = cluster_df[last_sales_col].mean()
                    cluster_stats['total_sales'] = cluster_df[last_sales_col].sum()
                
                # Büyüme istatistikleri
                if growth_cols:
                    last_growth_col = growth_cols[-1]
                    cluster_stats['avg_growth'] = cluster_df[last_growth_col].mean()
                
                # Fiyat istatistikleri
                if price_cols:
                    last_price_col = price_cols[-1]
                    cluster_stats['avg_price'] = cluster_df[last_price_col].mean()
                
                # Ürün sayısı
                cluster_stats['product_count'] = len(cluster_df)
                cluster_stats['percentage'] = (len(cluster_df) / len(result_df)) * 100
                
                cluster_analysis[cluster_num] = cluster_stats
            
            return {
                'segmented_df': result_df,
                'cluster_analysis': cluster_analysis,
                'features_used': features,
                'method': method,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            st.warning(f"Segmentasyon analizi hatası: {str(e)}")
            return None
    
    @staticmethod
    def analyze_price_volume_relationship(df):
        """Fiyat-hacim ilişkisi analizi"""
        try:
            # Fiyat ve hacim sütunlarını bul
            price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'fiyat'])]
            volume_cols = [col for col in df.columns if any(x in col.lower() for x in ['unit', 'birim', 'quantity', 'volume', 'hacim'])]
            
            if not price_cols or not volume_cols:
                return None
            
            last_price_col = price_cols[-1]
            last_volume_col = volume_cols[-1]
            
            # Veriyi hazırla
            analysis_df = df[[last_price_col, last_volume_col]].dropna()
            
            if len(analysis_df) < 10:
                return None
            
            # Korelasyon
            correlation = analysis_df[last_price_col].corr(analysis_df[last_volume_col])
            
            # Elasticity hesapla (basit regresyon)
            X = analysis_df[last_price_col].values.reshape(-1, 1)
            y = analysis_df[last_volume_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            elasticity = model.coef_[0] * (analysis_df[last_price_col].mean() / analysis_df[last_volume_col].mean())
            
            # Segmentasyon
            price_quantiles = analysis_df[last_price_col].quantile([0.33, 0.67])
            volume_quantiles = analysis_df[last_volume_col].quantile([0.33, 0.67])
            
            analysis_df['Price_Segment'] = pd.cut(
                analysis_df[last_price_col],
                bins=[0, price_quantiles[0.33], price_quantiles[0.67], analysis_df[last_price_col].max()],
                labels=['Low Price', 'Medium Price', 'High Price']
            )
            
            analysis_df['Volume_Segment'] = pd.cut(
                analysis_df[last_volume_col],
                bins=[0, volume_quantiles[0.33], volume_quantiles[0.67], analysis_df[last_volume_col].max()],
                labels=['Low Volume', 'Medium Volume', 'High Volume']
            )
            
            # Segmentasyon analizi
            segment_analysis = analysis_df.groupby(['Price_Segment', 'Volume_Segment']).size().unstack(fill_value=0)
            
            return {
                'analysis_df': analysis_df,
                'correlation': correlation,
                'elasticity': elasticity,
                'price_col': last_price_col,
                'volume_col': last_volume_col,
                'segment_analysis': segment_analysis,
                'price_stats': analysis_df[last_price_col].describe().to_dict(),
                'volume_stats': analysis_df[last_volume_col].describe().to_dict()
            }
            
        except Exception as e:
            st.warning(f"Fiyat-hacim analizi hatası: {str(e)}")
            return None

# ================================================
# 4. GELİŞMİŞ GÖRSELLEŞTİRME MOTORU
# ================================================

class AdvancedVisualizationEngine:
    """Gelişmiş görselleştirme motoru"""
    
    @staticmethod
    def create_metric_cards(metrics, n_cols=4):
        """Metrik kartları oluştur"""
        try:
            cols = st.columns(n_cols)
            
            # Kart 1: Toplam Pazar Değeri
            with cols[0]:
                total_market = metrics.get('total_market_value', 0)
                year = metrics.get('last_year', '2024')
                st.markdown(f"""
                <div class="custom-metric-card primary">
                    <div class="custom-metric-label">TOPLAM PAZAR DEĞERİ</div>
                    <div class="custom-metric-value">${total_market/1e6:.1f}M</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">{year}</span>
                        <span>Toplam Pazar Büyüklüğü</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Kart 2: Ortalama Büyüme
            with cols[1]:
                avg_growth = metrics.get('avg_growth_rate', 0)
                growth_class = "success" if avg_growth > 0 else "danger"
                st.markdown(f"""
                <div class="custom-metric-card {growth_class}">
                    <div class="custom-metric-label">ORTALAMA BÜYÜME</div>
                    <div class="custom-metric-value">{avg_growth:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Yıllık</span>
                        <span>Pazar Büyüme Hızı</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Kart 3: Ürün Çeşitliliği
            with cols[2]:
                unique_molecules = metrics.get('unique_molecules', 0)
                st.markdown(f"""
                <div class="custom-metric-card info">
                    <div class="custom-metric-label">ÜRÜN ÇEŞİTLİLİĞİ</div>
                    <div class="custom-metric-value">{unique_molecules:,}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">Benzersiz</span>
                        <span>Farklı Molekül</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Kart 4: International Product Payı
            with cols[3]:
                intl_share = metrics.get('intl_product_share', 0)
                intl_color = "success" if intl_share > 20 else "warning" if intl_share > 10 else "info"
                st.markdown(f"""
                <div class="custom-metric-card {intl_color}">
                    <div class="custom-metric-label">INTERNATIONAL PRODUCT</div>
                    <div class="custom-metric-value">{intl_share:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Global</span>
                        <span>Çoklu Pazar Ürünleri</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # İkinci satır
            cols2 = st.columns(n_cols)
            
            # Kart 5: Pazar Yoğunluğu (HHI)
            with cols2[0]:
                hhi = metrics.get('hhi_index', 0)
                hhi_status = "danger" if hhi > 2500 else "warning" if hhi > 1500 else "success"
                hhi_text = "Monopol" if hhi > 2500 else "Oligopol" if hhi > 1500 else "Rekabetçi"
                st.markdown(f"""
                <div class="custom-metric-card {hhi_status}">
                    <div class="custom-metric-label">REKABET YOĞUNLUĞU</div>
                    <div class="custom-metric-value">{hhi:.0f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-warning">HHI Index</span>
                        <span>{hhi_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Kart 6: Ortalama Fiyat
            with cols2[1]:
                avg_price = metrics.get('avg_price', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">ORTALAMA FİYAT</div>
                    <div class="custom-metric-value">${avg_price:.2f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Birim Başına</span>
                        <span>Ortalama Ürün Fiyatı</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Kart 7: Yüksek Büyüme Ürünleri
            with cols2[2]:
                high_growth = metrics.get('high_growth_percentage', 0)
                st.markdown(f"""
                <div class="custom-metric-card success">
                    <div class="custom-metric-label">YÜKSEK BÜYÜME</div>
                    <div class="custom-metric-value">{high_growth:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">%20+</span>
                        <span>Hızlı Büyüyen Ürünler</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Kart 8: Coğrafi Yayılım
            with cols2[3]:
                country_coverage = metrics.get('country_coverage', 0)
                st.markdown(f"""
                <div class="custom-metric-card purple">
                    <div class="custom-metric-label">COĞRAFİ YAYILIM</div>
                    <div class="custom-metric-value">{country_coverage}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-purple">Ülke</span>
                        <span>Global Kapsam Alanı</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Metrik kartları oluşturma hatası: {str(e)}")
    
    @staticmethod
    def create_price_volume_chart(df, price_col, volume_col, hover_col=None):
        """Fiyat-hacim grafiği oluştur"""
        try:
            # Veriyi hazırla
            chart_df = df[[price_col, volume_col]].dropna()
            
            if hover_col and hover_col in df.columns:
                chart_df['Hover'] = df[hover_col]
                hover_name = hover_col
            else:
                # Hover için en anlamlı sütunu bul
                hover_candidates = ['Molekül', 'Ürün Adı', 'Product', 'Şirket', 'Company']
                hover_name = None
                for candidate in hover_candidates:
                    if candidate in df.columns:
                        chart_df['Hover'] = df[candidate]
                        hover_name = candidate
                        break
            
            if len(chart_df) < 10:
                return None
            
            # Aykırı değerleri filtrele
            price_q1 = chart_df[price_col].quantile(0.05)
            price_q3 = chart_df[price_col].quantile(0.95)
            volume_q1 = chart_df[volume_col].quantile(0.05)
            volume_q3 = chart_df[volume_col].quantile(0.95)
            
            filtered_df = chart_df[
                (chart_df[price_col] >= price_q1) & 
                (chart_df[price_col] <= price_q3) &
                (chart_df[volume_col] >= volume_q1) & 
                (chart_df[volume_col] <= volume_q3)
            ]
            
            if len(filtered_df) > 1000:
                filtered_df = filtered_df.sample(1000, random_state=42)
            
            # Korelasyon hesapla
            correlation = filtered_df[price_col].corr(filtered_df[volume_col])
            
            # Grafik oluştur
            fig = px.scatter(
                filtered_df,
                x=price_col,
                y=volume_col,
                size=volume_col,
                color=price_col,
                hover_name='Hover' if 'Hover' in filtered_df.columns else None,
                hover_data=[price_col, volume_col],
                title=f'Fiyat-Hacim İlişkisi (Korelasyon: {correlation:.3f})',
                labels={
                    price_col: 'Fiyat (USD)',
                    volume_col: 'Hacim (Birim)'
                },
                color_continuous_scale='Viridis',
                size_max=50
            )
            
            # Regresyon çizgisi ekle
            try:
                z = np.polyfit(filtered_df[price_col], filtered_df[volume_col], 1)
                p = np.poly1d(z)
                
                x_line = np.linspace(filtered_df[price_col].min(), filtered_df[price_col].max(), 100)
                y_line = p(x_line)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Regresyon Çizgisi'
                    )
                )
            except:
                pass
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title_font=dict(size=14),
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title_font=dict(size=14),
                    tickfont=dict(size=12)
                ),
                coloraxis_colorbar=dict(
                    title="Fiyat",
                    tickprefix="$"
                ),
                hoverlabel=dict(
                    bgcolor="rgba(30, 58, 95, 0.9)",
                    font_size=12,
                    font_color="white"
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Fiyat-hacim grafiği oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_sales_trend_chart(df):
        """Satış trend grafiği oluştur"""
        try:
            # Satış sütunlarını bul
            sales_cols = [col for col in df.columns if any(x in col.lower() for x in ['sales', 'satış', 'revenue'])]
            
            if len(sales_cols) < 2:
                return None
            
            # Yılları çıkar
            yearly_data = []
            for col in sales_cols:
                year_match = re.search(r'(20\d{2})', col)
                if year_match:
                    year = int(year_match.group(1))
                    yearly_data.append({
                        'Year': year,
                        'Total_Sales': df[col].sum(),
                        'Avg_Sales': df[col].mean(),
                        'Product_Count': (df[col] > 0).sum()
                    })
            
            if len(yearly_data) < 2:
                return None
            
            yearly_df = pd.DataFrame(yearly_data).sort_values('Year')
            
            # Grafik oluştur
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Toplam Satış Trendi', 'Ortalama Satış Trendi'),
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )
            
            # Toplam satış
            fig.add_trace(
                go.Bar(
                    x=yearly_df['Year'],
                    y=yearly_df['Total_Sales'],
                    name='Toplam Satış',
                    marker_color='#2d7dd2',
                    text=[f'${x/1e6:.0f}M' for x in yearly_df['Total_Sales']],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Toplam Satış: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Büyüme oranı (ikinci eksen)
            yearly_df['Growth_Rate'] = yearly_df['Total_Sales'].pct_change() * 100
            
            fig.add_trace(
                go.Scatter(
                    x=yearly_df['Year'],
                    y=yearly_df['Growth_Rate'],
                    name='Büyüme Oranı',
                    mode='lines+markers',
                    line=dict(color='#2acaea', width=3),
                    marker=dict(size=10, symbol='diamond'),
                    yaxis='y2',
                    hovertemplate='<b>%{x}</b><br>Büyüme: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Ortalama satış
            fig.add_trace(
                go.Scatter(
                    x=yearly_df['Year'],
                    y=yearly_df['Avg_Sales'],
                    name='Ortalama Satış',
                    mode='lines+markers',
                    line=dict(color='#2dd2a3', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>%{x}</b><br>Ortalama Satış: $%{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=12)
                ),
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Yıl", row=2, col=1)
            fig.update_yaxes(title_text="Toplam Satış (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Büyüme Oranı (%)", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Ortalama Satış (USD)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            st.warning(f"Satış trend grafiği oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_market_share_chart(df, sales_col):
        """Pazar payı grafikleri oluştur"""
        try:
            # Şirket bazlı pazar payı
            if 'Şirket' in df.columns:
                company_sales = df.groupby('Şirket')[sales_col].sum().sort_values(ascending=False)
                top_companies = company_sales.nlargest(15)
                
                # Pasta grafiği için veri
                other_sales = company_sales.iloc[15:].sum() if len(company_sales) > 15 else 0
                pie_data = top_companies.copy()
                if other_sales > 0:
                    pie_data['Diğer'] = other_sales
                
                # Grafik oluştur
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Top 15 Şirket Pazar Payı', 'Top 10 Şirket Satışları'),
                    specs=[[{'type': 'domain'}, {'type': 'bar'}]],
                    column_widths=[0.4, 0.6]
                )
                
                # Pasta grafiği
                fig.add_trace(
                    go.Pie(
                        labels=pie_data.index,
                        values=pie_data.values,
                        hole=0.4,
                        marker_colors=px.colors.qualitative.Set3,
                        textinfo='percent+label',
                        textposition='outside',
                        hovertemplate='<b>%{label}</b><br>Pazar Payı: %{percent}<br>Satış: $%{value:,.0f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Bar grafiği
                fig.add_trace(
                    go.Bar(
                        x=top_companies.values[:10],
                        y=top_companies.index[:10],
                        orientation='h',
                        marker_color='#2d7dd2',
                        text=[f'${x/1e6:.1f}M' for x in top_companies.values[:10]],
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>Satış: $%{x:,.0f}<extra></extra>'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc',
                    showlegend=False,
                    title_text="Pazar Konsantrasyonu Analizi",
                    title_x=0.5,
                    title_font=dict(size=18)
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.warning(f"Pazar payı grafiği oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_lorenz_curve(sales_data):
        """Lorenz eğrisi oluştur"""
        try:
            # Eğer Series değilse dönüştür
            if isinstance(sales_data, pd.DataFrame):
                sales_data = sales_data.iloc[:, 0]
            
            # Sırala
            sorted_sales = np.sort(sales_data.values)
            cum_sales = np.cumsum(sorted_sales)
            
            if cum_sales[-1] == 0:
                return None
            
            cum_percentage_sales = cum_sales / cum_sales[-1]
            perfect_line = np.linspace(0, 1, len(cum_percentage_sales))
            
            # Gini katsayısı
            gini_coefficient = 1 - 2 * integrate.trapz(cum_percentage_sales, perfect_line)
            
            # Grafik oluştur
            fig = go.Figure()
            
            # Lorenz eğrisi
            fig.add_trace(go.Scatter(
                x=perfect_line,
                y=cum_percentage_sales,
                mode='lines',
                line=dict(color='#2acaea', width=3),
                name=f'Lorenz Eğrisi (Gini: {gini_coefficient:.3f})',
                fill='tozeroy',
                fillcolor='rgba(42, 202, 234, 0.3)',
                hovertemplate='Şirketlerin %{x:.1%}<br>Satışların %{y:.1%}<extra></extra>'
            ))
            
            # Mükemmel eşitlik çizgisi
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='#f8fafc', width=2, dash='dash'),
                name='Tam Eşitlik'
            ))
            
            fig.update_layout(
                title='Lorenz Eğrisi - Pazar Konsantrasyonu Analizi',
                xaxis_title='Şirketlerin Kümülatif Oranı',
                yaxis_title='Satışların Kümülatif Oranı',
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=12)
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Lorenz eğrisi oluşturma hatası: {str(e)}")
            return None

# ================================================
# 5. RAPORLAMA MOTORU
# ================================================

class ReportingEngine:
    """Raporlama motoru - Çoklu formatlarda rapor oluşturma"""
    
    @staticmethod
    def generate_detailed_report(df, metrics, insights, analysis_results):
        """Detaylı rapor oluştur"""
        try:
            report = {
                'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_summary': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'total_products': len(df),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
                },
                'key_metrics': metrics,
                'strategic_insights': insights,
                'analysis_results': analysis_results
            }
            
            return report
            
        except Exception as e:
            st.warning(f"Rapor oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_excel_report(df, metrics, insights, filename="pharma_report"):
        """Excel raporu oluştur"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Ana veri
                df.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                # Metrikler
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # İçgörüler
                insights_df = pd.DataFrame(insights)
                insights_df.to_excel(writer, sheet_name='Insights', index=False)
                
                # Özet
                summary_data = {
                    'Report_Date': [datetime.now().strftime('%Y-%m-%d')],
                    'Total_Products': [len(df)],
                    'Total_Market_Value': [metrics.get('total_market_value', 0)],
                    'Avg_Growth_Rate': [metrics.get('avg_growth_rate', 0)],
                    'HHI_Index': [metrics.get('hhi_index', 0)]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            output.seek(0)
            
            return {
                'data': output,
                'filename': f"{filename}_{timestamp}.xlsx",
                'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
            
        except Exception as e:
            st.warning(f"Excel raporu oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_html_report(df, metrics, insights):
        """HTML raporu oluştur"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>PharmaIntelligence Pro Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
                    .header {{ background: linear-gradient(135deg, #2d7dd2, #2acaea); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                    .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                    .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; margin: 10px 0; }}
                    .insight-card {{ background: #e3f2fd; padding: 15px; border-left: 4px solid #2d7dd2; margin: 10px 0; }}
                    .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    .table th, .table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    .table th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🏥 PharmaIntelligence Pro Raporu</h1>
                    <p>Rapor Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
                </div>
                
                <div class="section">
                    <h2>📊 Veri Özeti</h2>
                    <p><strong>Toplam Ürün Sayısı:</strong> {len(df):,}</p>
                    <p><strong>Toplam Sütun Sayısı:</strong> {len(df.columns)}</p>
                    <p><strong>Rapor Kapsamı:</strong> {metrics.get('country_coverage', 0)} ülke, {metrics.get('unique_molecules', 0)} benzersiz molekül</p>
                </div>
                
                <div class="section">
                    <h2>📈 Temel Metrikler</h2>
                    <div class="metric-card">
                        <h3>Toplam Pazar Değeri</h3>
                        <p><strong>${metrics.get('total_market_value', 0)/1e6:.1f} Milyon</strong></p>
                    </div>
                    <div class="metric-card">
                        <h3>Ortalama Büyüme Oranı</h3>
                        <p><strong>{metrics.get('avg_growth_rate', 0):.1f}%</strong></p>
                    </div>
                    <div class="metric-card">
                        <h3>Pazar Yoğunluğu (HHI)</h3>
                        <p><strong>{metrics.get('hhi_index', 0):.0f}</strong></p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>💡 Stratejik İçgörüler</h2>
            """
            
            for insight in insights[:10]:  # İlk 10 içgörüyü göster
                html_content += f"""
                    <div class="insight-card">
                        <h3>{insight.get('icon', '💡')} {insight.get('title', '')}</h3>
                        <p>{insight.get('content', '')}</p>
                    </div>
                """
            
            html_content += """
                </div>
                
                <div class="section">
                    <h2>📋 Örnek Veri</h2>
                    <table class="table">
                        <thead>
                            <tr>
            """
            
            # İlk 5 sütunu göster
            columns_to_show = df.columns.tolist()[:5]
            for col in columns_to_show:
                html_content += f"<th>{col}</th>"
            html_content += "</tr></thead><tbody>"
            
            # İlk 10 satırı göster
            for _, row in df.head(10).iterrows():
                html_content += "<tr>"
                for col in columns_to_show:
                    value = row[col]
                    if isinstance(value, (int, float)) and col.lower().find('sales') != -1:
                        value = f"${value/1e6:.2f}M"
                    elif isinstance(value, float):
                        value = f"{value:.2f}"
                    html_content += f"<td>{value}</td>"
                html_content += "</tr>"
            
            html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div class="section">
                    <p><em>Bu rapor PharmaIntelligence Pro v6.0 ile oluşturulmuştur.</em></p>
                </div>
            </body>
            </html>
            """
            
            return {
                'data': html_content,
                'filename': f"pharma_report_{datetime.now().strftime('%Y%m%d')}.html",
                'mime_type': 'text/html'
            }
            
        except Exception as e:
            st.warning(f"HTML raporu oluşturma hatası: {str(e)}")
            return None

# ================================================
# 6. ANA UYGULAMA
# ================================================

def main():
    """Ana uygulama fonksiyonu"""
    
    # Başlık
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="pharma-title">💊 PHARMAINTELLIGENCE PRO v6.0</h1>
        <p class="pharma-subtitle">
        Enterprise-level pharmaceutical market intelligence platform with AI-powered analytics, 
        predictive insights, and comprehensive reporting for strategic decision-making.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Session state initialization
    session_keys = [
        'data', 'filtered_data', 'metrics', 'insights', 
        'active_filters', 'analysis_results', 'data_loaded'
    ]
    
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'data_loaded' else False
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
        
        # Veri yükleme
        with st.expander("📁 VERİ YÜKLEME", expanded=True):
            uploaded_file = st.file_uploader(
                "Excel/CSV Dosyası Yükleyin",
                type=['xlsx', 'xls', 'csv'],
                help="1M+ satır desteklenir. Büyük dosyalar için dikkatli olun.",
                key="file_uploader"
            )
            
            if uploaded_file:
                st.info(f"""
                **Dosya:** {uploaded_file.name}
                **Boyut:** {uploaded_file.size / 1024 / 1024:.1f} MB
                """)
                
                # Örneklem seçimi
                use_sample = st.checkbox("Örneklem Yükle", value=False, 
                                        help="Büyük dosyalar için örneklem yükleyin")
                sample_size = None
                if use_sample:
                    sample_size = st.slider("Örneklem Boyutu", 1000, 100000, 10000, 1000)
                
                if st.button("🚀 Veriyi Yükle & Analiz Et", type="primary", use_container_width=True):
                    with st.spinner("Veri yükleniyor ve analiz ediliyor..."):
                        try:
                            # Veriyi yükle
                            processor = AdvancedDataProcessor()
                            data = processor.load_and_process_data(uploaded_file, sample_size)
                            
                            if data is not None:
                                # Veriyi işle
                                data = processor.clean_column_names(data)
                                data = processor.optimize_dataframe(data)
                                data = processor.prepare_analysis_data(data)
                                
                                # Session state'i güncelle
                                st.session_state.data = data
                                st.session_state.filtered_data = data.copy()
                                st.session_state.data_loaded = True
                                
                                # Metrikleri hesapla
                                analytics = AdvancedAnalyticsEngine()
                                metrics = analytics.calculate_comprehensive_metrics(data)
                                st.session_state.metrics = metrics
                                
                                # İçgörüleri oluştur
                                insights = analytics.generate_strategic_insights(data, metrics)
                                st.session_state.insights = insights
                                
                                st.success(f"✅ **{len(data):,} ürün** başarıyla yüklendi ve analiz edildi!")
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Veri yükleme hatası: {str(e)}")
        
        # Veri yüklendiyse filtreleme
        if st.session_state.data_loaded and st.session_state.data is not None:
            with st.expander("🔍 GELİŞMİŞ FİLTRELEME", expanded=True):
                data = st.session_state.data
                
                # Arama
                search_term = st.text_input(
                    "🔎 Genel Arama",
                    placeholder="Molekül, Şirket, Ürün ara...",
                    key="global_search"
                )
                
                # Kategorik filtreler
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Ülke' in data.columns:
                        countries = sorted(data['Ülke'].dropna().unique())
                        selected_countries = st.multiselect(
                            "🌍 Ülkeler",
                            options=countries,
                            default=countries[:min(5, len(countries))],
                            key="countries_filter"
                        )
                
                with col2:
                    if 'Şirket' in data.columns:
                        companies = sorted(data['Şirket'].dropna().unique())
                        selected_companies = st.multiselect(
                            "🏢 Şirketler",
                            options=companies,
                            default=companies[:min(5, len(companies))],
                            key="companies_filter"
                        )
                
                # Sayısal filtreler
                sales_cols = [col for col in data.columns if any(x in col.lower() for x in ['sales', 'satış'])]
                if sales_cols:
                    last_sales_col = sales_cols[-1]
                    sales_min = float(data[last_sales_col].min())
                    sales_max = float(data[last_sales_col].max())
                    
                    sales_range = st.slider(
                        f"Satış Aralığı ({last_sales_col})",
                        min_value=sales_min,
                        max_value=sales_max,
                        value=(sales_min, sales_max),
                        key="sales_filter"
                    )
                
                # International Product filtreleme
                if 'International_Product' in data.columns:
                    intl_filter = st.selectbox(
                        "International Product",
                        ["Tümü", "Sadece International", "Sadece Yerel"],
                        key="intl_filter"
                    )
                
                # Filtre butonları
                col3, col4 = st.columns(2)
                with col3:
                    apply_filters = st.button("✅ Filtre Uygula", use_container_width=True, key="apply_filters")
                with col4:
                    clear_filters = st.button("🗑️ Temizle", use_container_width=True, key="clear_filters")
                
                if apply_filters:
                    filtered_data = data.copy()
                    
                    # Arama
                    if search_term:
                        mask = pd.Series(False, index=filtered_data.index)
                        for col in filtered_data.columns:
                            try:
                                mask = mask | filtered_data[col].astype(str).str.contains(search_term, case=False, na=False)
                            except:
                                continue
                        filtered_data = filtered_data[mask]
                    
                    # Ülke filtreleme
                    if 'Ülke' in data.columns and selected_countries:
                        filtered_data = filtered_data[filtered_data['Ülke'].isin(selected_countries)]
                    
                    # Şirket filtreleme
                    if 'Şirket' in data.columns and selected_companies:
                        filtered_data = filtered_data[filtered_data['Şirket'].isin(selected_companies)]
                    
                    # Satış filtreleme
                    if sales_cols and 'sales_range' in locals():
                        filtered_data = filtered_data[
                            (filtered_data[last_sales_col] >= sales_range[0]) & 
                            (filtered_data[last_sales_col] <= sales_range[1])
                        ]
                    
                    # International Product filtreleme
                    if 'International_Product' in data.columns and 'intl_filter' in locals():
                        if intl_filter == "Sadece International":
                            filtered_data = filtered_data[filtered_data['International_Product'] == 1]
                        elif intl_filter == "Sadece Yerel":
                            filtered_data = filtered_data[filtered_data['International_Product'] == 0]
                    
                    st.session_state.filtered_data = filtered_data
                    st.success(f"✅ Filtre uygulandı: {len(filtered_data):,} ürün")
                    st.rerun()
                
                if clear_filters:
                    st.session_state.filtered_data = st.session_state.data.copy()
                    st.session_state.active_filters = {}
                    st.success("✅ Filtreler temizlendi")
                    st.rerun()
        
        # Hakkında
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b; padding: 1rem;">
        <strong>PharmaIntelligence Pro v6.0</strong><br>
        AI-Powered Pharma Analytics Platform<br>
        © 2024 Tüm hakları saklıdır.
        </div>
        """, unsafe_allow_html=True)
    
    # Ana içerik
    if not st.session_state.data_loaded:
        show_welcome_screen()
        return
    
    # Filtre durumu
    data = st.session_state.filtered_data
    original_data = st.session_state.data
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    
    if len(data) != len(original_data):
        st.markdown(f"""
        <div class="filter-status">
        🎯 <strong>Aktif Filtreler:</strong> {len(data):,} / {len(original_data):,} ürün gösteriliyor
        ({((len(original_data)-len(data))/len(original_data)*100):.1f}% filtrelendi)
        </div>
        """, unsafe_allow_html=True)
    
    # Tablar
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ", 
        "💰 FİYAT ANALİZİ",
        "🏆 REKABET ANALİZİ",
        "🌍 INTERNATIONAL",
        "🔮 STRATEJİK ANALİZ",
        "📑 RAPORLAMA"
    ])
    
    with tab1:
        show_overview_tab(data, metrics, insights)
    
    with tab2:
        show_market_analysis_tab(data, metrics)
    
    with tab3:
        show_price_analysis_tab(data)
    
    with tab4:
        show_competition_analysis_tab(data, metrics)
    
    with tab5:
        show_international_tab(data, metrics)
    
    with tab6:
        show_strategic_analysis_tab(data, insights)
    
    with tab7:
        show_reporting_tab(data, metrics, insights)

# ================================================
# 7. TAB FONKSİYONLARI
# ================================================

def show_welcome_screen():
    """Hoşgeldiniz ekranını göster"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">💊</div>
            <h2 style="color: #f1f5f9; margin-bottom: 1rem;">PharmaIntelligence Pro v6.0'a Hoşgeldiniz</h2>
            <p style="color: #cbd5e1; margin-bottom: 2rem; line-height: 1.6;">
            İlaç pazarı verilerinizi yükleyin ve AI destekli gelişmiş analitik özelliklerin kilidini açın.
            <br>International Product analizi ile çoklu pazar stratejilerinizi optimize edin.
            </p>
            
            <div class="feature-grid">
                <div class="feature-card feature-card-blue">
                    <div class="feature-icon">🌍</div>
                    <div class="feature-title">International Product Analytics</div>
                    <div class="feature-description">Çoklu pazar ürün analizi ve strateji geliştirme</div>
                </div>
                <div class="feature-card feature-card-cyan">
                    <div class="feature-icon">📈</div>
                    <div class="feature-title">Advanced Market Analysis</div>
                    <div class="feature-description">Derin pazar içgörüleri ve trend analizi</div>
                </div>
                <div class="feature-card feature-card-teal">
                    <div class="feature-icon">💰</div>
                    <div class="feature-title">Price Intelligence</div>
                    <div class="feature-description">Rekabetçi fiyatlandırma ve optimizasyon analizi</div>
                </div>
                <div class="feature-card feature-card-warning">
                    <div class="feature-icon">🏆</div>
                    <div class="feature-title">Competitive Intelligence</div>
                    <div class="feature-description">Rakiplerinizi analiz edin ve fırsatları belirleyin</div>
                </div>
                <div class="feature-card feature-card-purple">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-title">AI-Powered Insights</div>
                    <div class="feature-description">Yapay zeka destekli tahminler ve öneriler</div>
                </div>
                <div class="feature-card feature-card-pink">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Interactive Dashboards</div>
                    <div class="feature-description">Etkileşimli raporlar ve görselleştirmeler</div>
                </div>
            </div>
            
            <div class="get-started-box">
                <div class="get-started-title">🎯 Başlamak İçin</div>
                <div class="get-started-steps">
                1. Sol taraftaki panelden veri dosyanızı yükleyin<br>
                2. "Veriyi Yükle & Analiz Et" butonuna tıklayın<br>
                3. Analiz sonuçlarını görmek için tabları kullanın
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_overview_tab(df, metrics, insights):
    """Genel Bakış tab'ını göster"""
    st.markdown('<h2 class="section-title">📊 Genel Bakış ve Performans Göstergeleri</h2>', unsafe_allow_html=True)
    
    # Metrik kartları
    viz = AdvancedVisualizationEngine()
    viz.create_metric_cards(metrics)
    
    # İçgörüler
    st.markdown('<h3 class="subsection-title">💡 Stratejik İçgörüler</h3>', unsafe_allow_html=True)
    
    if insights:
        # İçgörüleri grid olarak göster
        insight_cols = st.columns(2)
        
        for idx, insight in enumerate(insights[:6]):  # İlk 6 içgörü
            with insight_cols[idx % 2]:
                icon = insight.get('icon', '💡')
                title = insight.get('title', '')
                content = insight.get('content', '')
                color = insight.get('color', 'info')
                
                st.markdown(f"""
                <div class="insight-card {color}">
                    <div class="insight-icon">{icon}</div>
                    <div class="insight-title">{title}</div>
                    <div class="insight-content">{content}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Verileriniz analiz ediliyor... Stratejik içgörüler burada görünecek.")
    
    # Veri önizleme
    st.markdown('<h3 class="subsection-title">📋 Veri Önizleme</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        rows_to_show = st.slider("Gösterilecek Satır Sayısı", 10, 500, 100, 10)
        
        # Önemli sütunları otomatik seç
        important_cols = []
        priority_cols = ['Molekül', 'Şirket', 'Ülke', 'Satış_2024', 'Büyüme_2023_2024', 
                        'Ort_Fiyat_2024', 'International_Product', 'Market_Share_%']
        
        for col in priority_cols:
            if col in df.columns:
                important_cols.append(col)
                if len(important_cols) >= 6:
                    break
        
        # Eğer yeterli sütun yoksa, diğer sütunları ekle
        if len(important_cols) < 6:
            other_cols = [col for col in df.columns if col not in important_cols]
            important_cols.extend(other_cols[:6-len(important_cols)])
        
        selected_cols = st.multiselect(
            "Gösterilecek Sütunlar",
            options=df.columns.tolist(),
            default=important_cols,
            key="overview_columns"
        )
    
    with col2:
        if selected_cols:
            # Veriyi formatla
            display_df = df[selected_cols].copy()
            
            # Sayısal sütunları formatla
            for col in display_df.columns:
                if pd.api.types.is_numeric_dtype(display_df[col]):
                    if any(x in col.lower() for x in ['sales', 'satış', 'revenue']):
                        # Milyon dolar formatı
                        display_df[col] = display_df[col].apply(lambda x: f"${x/1e6:.2f}M" if pd.notnull(x) else "")
                    elif any(x in col.lower() for x in ['price', 'fiyat', 'cost']):
                        # Dolar formatı
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
                    elif any(x in col.lower() for x in ['growth', 'büyüme', 'cagr']):
                        # Yüzde formatı
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
                    elif 'share' in col.lower():
                        # Yüzde formatı
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
            
            st.dataframe(
                display_df.head(rows_to_show),
                use_container_width=True,
                height=400
            )
            
            # İstatistikler
            st.caption(f"**{len(df):,} ürün** | **{len(df.columns)} sütun** | **{df.memory_usage(deep=True).sum()/1024**2:.1f} MB bellek kullanımı**")
        else:
            st.info("Gösterilecek sütun seçin.")

def show_market_analysis_tab(df, metrics):
    """Pazar Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title section-title-cyan">📈 Pazar Analizi ve Trendler</h2>', unsafe_allow_html=True)
    
    viz = AdvancedVisualizationEngine()
    
    # Satış trendleri
    st.markdown('<h3 class="subsection-title">📊 Satış Trendleri</h3>', unsafe_allow_html=True)
    sales_trend_chart = viz.create_sales_trend_chart(df)
    if sales_trend_chart:
        st.plotly_chart(sales_trend_chart, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("Satış trend analizi için yeterli yıllık veri bulunamadı.")
    
    # Pazar payı analizi
    st.markdown('<h3 class="subsection-title">🏆 Pazar Payı Analizi</h3>', unsafe_allow_html=True)
    
    sales_cols = [col for col in df.columns if any(x in col.lower() for x in ['sales', 'satış'])]
    if sales_cols:
        last_sales_col = sales_cols[-1]
        market_share_chart = viz.create_market_share_chart(df, last_sales_col)
        if market_share_chart:
            st.plotly_chart(market_share_chart, use_container_width=True, config={'displayModeBar': True})
        else:
            st.info("Pazar payı analizi için gerekli veri bulunamadı.")
    
    # Molekül bazlı analiz
    if 'Molekül' in df.columns and sales_cols:
        st.markdown('<h3 class="subsection-title">🧪 Molekül Bazlı Analiz</h3>', unsafe_allow_html=True)
        
        molecule_sales = df.groupby('Molekül')[last_sales_col].sum().sort_values(ascending=False)
        top_molecules = molecule_sales.head(10)
        
        fig = px.bar(
            x=top_molecules.values,
            y=top_molecules.index,
            orientation='h',
            title='Top 10 Molekül - Satışlar',
            labels={'x': 'Satış (USD)', 'y': 'Molekül'},
            color=top_molecules.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_price_analysis_tab(df):
    """Fiyat Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title section-title-teal">💰 Fiyat Analizi ve Optimizasyon</h2>', unsafe_allow_html=True)
    
    viz = AdvancedVisualizationEngine()
    
    # Fiyat-hacim analizi
    st.markdown('<h3 class="subsection-title">📈 Fiyat-Hacim İlişkisi</h3>', unsafe_allow_html=True)
    
    # Fiyat ve hacim sütunlarını bul
    price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'fiyat'])]
    volume_cols = [col for col in df.columns if any(x in col.lower() for x in ['unit', 'birim', 'quantity', 'volume'])]
    
    if price_cols and volume_cols:
        last_price_col = price_cols[-1]
        last_volume_col = volume_cols[-1]
        
        # Hover için sütun seç
        hover_options = [col for col in df.columns if col in ['Molekül', 'Şirket', 'Ürün Adı', 'Product']]
        hover_col = hover_options[0] if hover_options else None
        
        price_volume_chart = viz.create_price_volume_chart(df, last_price_col, last_volume_col, hover_col)
        
        if price_volume_chart:
            st.plotly_chart(price_volume_chart, use_container_width=True, config={'displayModeBar': True})
            
            # İstatistikler
            price_stats = df[last_price_col].describe()
            volume_stats = df[last_volume_col].describe()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="insight-card info">
                    <div class="insight-title">💰 Fiyat İstatistikleri</div>
                    <div class="insight-content">
                    • Ortalama: ${:.2f}<br>
                    • Medyan: ${:.2f}<br>
                    • Std Sapma: ${:.2f}<br>
                    • Min: ${:.2f} | Max: ${:.2f}
                    </div>
                </div>
                """.format(
                    price_stats['mean'], price_stats['50%'], price_stats['std'],
                    price_stats['min'], price_stats['max']
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="insight-card cyan">
                    <div class="insight-title">📦 Hacim İstatistikleri</div>
                    <div class="insight-content">
                    • Ortalama: {:,.0f}<br>
                    • Medyan: {:,.0f}<br>
                    • Std Sapma: {:,.0f}<br>
                    • Min: {:,.0f} | Max: {:,.0f}
                    </div>
                </div>
                """.format(
                    volume_stats['mean'], volume_stats['50%'], volume_stats['std'],
                    volume_stats['min'], volume_stats['max']
                ), unsafe_allow_html=True)
        else:
            st.info("Fiyat-hacim analizi için yeterli veri bulunamadı.")
    else:
        st.info("Fiyat veya hacim sütunları bulunamadı.")
    
    # Fiyat segmentasyonu
    st.markdown('<h3 class="subsection-title">🏷️ Fiyat Segmentasyonu</h3>', unsafe_allow_html=True)
    
    if price_cols:
        last_price_col = price_cols[-1]
        
        # Fiyat segmentleri
        price_segments = pd.cut(
            df[last_price_col],
            bins=[0, 10, 50, 100, 500, float('inf')],
            labels=['Ekonomi (<$10)', 'Standart ($10-$50)', 'Premium ($50-$100)', 
                   'Süper Premium ($100-$500)', 'Lüks (>$500)']
        )
        
        segment_counts = price_segments.value_counts()
        
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Fiyat Segmentleri Dağılımı',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_competition_analysis_tab(df, metrics):
    """Rekabet Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title section-title-purple">🏆 Rekabet Analizi ve Pazar Yapısı</h2>', unsafe_allow_html=True)
    
    # Rekabet metrikleri
    st.markdown('<h3 class="subsection-title">📊 Rekabet Yoğunluğu Metrikleri</h3>', unsafe_allow_html=True)
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        hhi = metrics.get('hhi_index', 0)
        if hhi > 2500:
            hhi_status = "Monopolistik"
            delta_color = "inverse"
        elif hhi > 1800:
            hhi_status = "Oligopol"
            delta_color = "normal"
        else:
            hhi_status = "Rekabetçi"
            delta_color = "normal"
        st.metric("HHI İndeksi", f"{hhi:.0f}", hhi_status, delta_color=delta_color)
    
    with metric_cols[1]:
        top3 = metrics.get('top_3_company_share', 0)
        st.metric("Top 3 Payı", f"{top3:.1f}%")
    
    with metric_cols[2]:
        top5 = metrics.get('top_5_company_share', 0)
        st.metric("Top 5 Payı", f"{top5:.1f}%")
    
    with metric_cols[3]:
        companies = metrics.get('total_companies', 0)
        st.metric("Toplam Şirket", f"{companies}")
    
    # Lorenz eğrisi
    st.markdown('<h3 class="subsection-title">📈 Pazar Konsantrasyonu - Lorenz Eğrisi</h3>', unsafe_allow_html=True)
    
    if 'Şirket' in df.columns:
        sales_cols = [col for col in df.columns if any(x in col.lower() for x in ['sales', 'satış'])]
        if sales_cols:
            last_sales_col = sales_cols[-1]
            company_sales = df.groupby('Şirket')[last_sales_col].sum()
            
            lorenz_chart = AdvancedVisualizationEngine().create_lorenz_curve(company_sales)
            if lorenz_chart:
                st.plotly_chart(lorenz_chart, use_container_width=True, config={'displayModeBar': True})
    
    # Şirket performansı
    st.markdown('<h3 class="subsection-title">🏢 Şirket Performans Analizi</h3>', unsafe_allow_html=True)
    
    if 'Şirket' in df.columns and sales_cols:
        # Büyüme sütunlarını bul
        growth_cols = [col for col in df.columns if any(x in col.lower() for x in ['growth', 'büyüme'])]
        
        if growth_cols:
            last_growth_col = growth_cols[-1]
            
            # Şirket bazlı ortalama büyüme
            company_growth = df.groupby('Şirket')[last_growth_col].mean().sort_values(ascending=False)
            top_growth_companies = company_growth.head(10)
            
            fig = px.bar(
                x=top_growth_companies.values,
                y=top_growth_companies.index,
                orientation='h',
                title='Top 10 Şirket - Ortalama Büyüme',
                labels={'x': 'Ortalama Büyüme (%)', 'y': 'Şirket'},
                color=top_growth_companies.values,
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_international_tab(df, metrics):
    """International Product Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title section-title-cyan">🌍 International Product Analizi</h2>', unsafe_allow_html=True)
    
    if 'International_Product' not in df.columns:
        st.warning("International Product sütunu bulunamadı.")
        return
    
    # Metrikler
    st.markdown('<h3 class="subsection-title">📊 International Product Metrikleri</h3>', unsafe_allow_html=True)
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        intl_count = (df['International_Product'] == 1).sum()
        total_count = len(df)
        intl_percentage = (intl_count / total_count) * 100
        st.metric("International Product", f"{intl_count:,}", f"%{intl_percentage:.1f}")
    
    with metric_cols[1]:
        sales_cols = [col for col in df.columns if any(x in col.lower() for x in ['sales', 'satış'])]
        if sales_cols:
            last_sales_col = sales_cols[-1]
            intl_sales = df[df['International_Product'] == 1][last_sales_col].sum()
            total_sales = df[last_sales_col].sum()
            intl_sales_share = (intl_sales / total_sales) * 100 if total_sales > 0 else 0
            st.metric("Satış Payı", f"%{intl_sales_share:.1f}")
    
    with metric_cols[2]:
        if 'Ülke' in df.columns:
            intl_countries = df[df['International_Product'] == 1]['Ülke'].nunique()
            total_countries = df['Ülke'].nunique()
            st.metric("Ülke Kapsamı", f"{intl_countries}/{total_countries}")
    
    with metric_cols[3]:
        growth_cols = [col for col in df.columns if any(x in col.lower() for x in ['growth', 'büyüme'])]
        if growth_cols:
            last_growth_col = growth_cols[-1]
            intl_growth = df[df['International_Product'] == 1][last_growth_col].mean()
            local_growth = df[df['International_Product'] == 0][last_growth_col].mean()
            growth_diff = intl_growth - local_growth
            st.metric("Büyüme Farkı", f"{growth_diff:+.1f}%")
    
    # International vs Local karşılaştırması
    st.markdown('<h3 class="subsection-title">🔄 International vs Local Karşılaştırması</h3>', unsafe_allow_html=True)
    
    # Veriyi hazırla
    intl_df = df[df['International_Product'] == 1]
    local_df = df[df['International_Product'] == 0]
    
    comparison_data = []
    
    if sales_cols and growth_cols:
        last_sales_col = sales_cols[-1]
        last_growth_col = growth_cols[-1]
        
        comparison_data.append({
            'Segment': 'International',
            'Product_Count': len(intl_df),
            'Avg_Sales': intl_df[last_sales_col].mean(),
            'Avg_Growth': intl_df[last_growth_col].mean(),
            'Sales_Share': (intl_df[last_sales_col].sum() / df[last_sales_col].sum()) * 100
        })
        
        comparison_data.append({
            'Segment': 'Local',
            'Product_Count': len(local_df),
            'Avg_Sales': local_df[last_sales_col].mean(),
            'Avg_Growth': local_df[last_growth_col].mean(),
            'Sales_Share': (local_df[last_sales_col].sum() / df[last_sales_col].sum()) * 100
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Grafik oluştur
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Ürün Sayısı ve Satış Payı', 'Ortalama Satış ve Büyüme'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Ürün sayısı ve satış payı
        fig.add_trace(
            go.Bar(
                x=comparison_df['Segment'],
                y=comparison_df['Product_Count'],
                name='Ürün Sayısı',
                marker_color=['#2d7dd2', '#64748b'],
                text=comparison_df['Product_Count'].astype(str),
                textposition='auto'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=comparison_df['Segment'],
                y=comparison_df['Sales_Share'],
                name='Satış Payı',
                marker_color=['#4a9fe3', '#94a3b8'],
                text=[f"{x:.1f}%" for x in comparison_df['Sales_Share']],
                textposition='auto',
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Ortalama satış ve büyüme
        fig.add_trace(
            go.Bar(
                x=comparison_df['Segment'],
                y=comparison_df['Avg_Sales'],
                name='Ort. Satış',
                marker_color=['#2acaea', '#cbd5e1'],
                text=[f"${x/1e3:.0f}K" for x in comparison_df['Avg_Sales']],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=comparison_df['Segment'],
                y=comparison_df['Avg_Growth'],
                name='Ort. Büyüme',
                marker_color=['#2dd2a3', '#94a3b8'],
                text=[f"{x:.1f}%" for x in comparison_df['Avg_Growth']],
                textposition='auto',
                yaxis='y2'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
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
            )
        )
        
        fig.update_yaxes(title_text="Ürün Sayısı", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Satış Payı (%)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Ort. Satış (USD)", row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Ort. Büyüme (%)", row=1, col=2, secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # International product detayları
    st.markdown('<h3 class="subsection-title">📋 International Product Detay Listesi</h3>', unsafe_allow_html=True)
    
    if len(intl_df) > 0:
        # Gösterilecek sütunları belirle
        display_cols = []
        priority_cols = ['Molekül', 'Şirket', 'Ülke']
        
        for col in priority_cols:
            if col in intl_df.columns:
                display_cols.append(col)
        
        # Satış ve büyüme sütunlarını ekle
        if sales_cols:
            display_cols.append(sales_cols[-1])
        if growth_cols:
            display_cols.append(growth_cols[-1])
        
        # Fiyat sütunlarını ekle
        price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'fiyat'])]
        if price_cols:
            display_cols.append(price_cols[-1])
        
        # International özel sütunlar
        intl_specific = [col for col in df.columns if 'international' in col.lower() and col != 'International_Product']
        display_cols.extend(intl_specific[:2])
        
        # Benzersiz sütunlar
        display_cols = list(dict.fromkeys(display_cols))
        
        # Veriyi göster
        display_df = intl_df[display_cols].copy()
        
        # Formatlama
        for col in display_df.columns:
            if pd.api.types.is_numeric_dtype(display_df[col]):
                if any(x in col.lower() for x in ['sales', 'satış']):
                    display_df[col] = display_df[col].apply(lambda x: f"${x/1e6:.2f}M" if pd.notnull(x) else "")
                elif any(x in col.lower() for x in ['price', 'fiyat']):
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
                elif any(x in col.lower() for x in ['growth', 'büyüme']):
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        st.caption(f"**{len(intl_df):,} International Product** | **{len(display_cols)} sütun**")
    else:
        st.info("International Product bulunamadı.")

def show_strategic_analysis_tab(df, insights):
    """Stratejik Analiz tab'ını göster"""
    st.markdown('<h2 class="section-title section-title-purple">🔮 Stratejik Analiz ve AI Önerileri</h2>', unsafe_allow_html=True)
    
    # Segmentasyon analizi
    st.markdown('<h3 class="subsection-title">🎯 Pazar Segmentasyonu</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_clusters = st.slider("Küme Sayısı", 2, 8, 4, key="n_clusters")
        method = st.selectbox("Segmentasyon Yöntemi", ['kmeans', 'dbscan'], key="seg_method")
        
        if st.button("🔍 Segmentasyon Analizi Yap", type="primary", use_container_width=True, key="run_segmentation"):
            with st.spinner("Pazar segmentasyonu analiz ediliyor..."):
                analytics = AdvancedAnalyticsEngine()
                segmentation_result = analytics.perform_market_segmentation(df, n_clusters, method)
                
                if segmentation_result:
                    st.session_state.segmentation_result = segmentation_result
                    st.success(f"✅ Segmentasyon tamamlandı! {n_clusters} küme oluşturuldu.")
                    st.rerun()
    
    with col2:
        if 'segmentation_result' in st.session_state:
            result = st.session_state.segmentation_result
            segmented_df = result['segmented_df']
            cluster_analysis = result['cluster_analysis']
            
            # Küme dağılımı
            cluster_counts = segmented_df['Cluster_Name'].value_counts()
            
            fig = px.pie(
                values=cluster_counts.values,
                names=cluster_counts.index,
                title='Pazar Segmentleri Dağılımı',
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Küme detayları
            with st.expander("📊 Küme Detayları", expanded=False):
                for cluster_num, stats in cluster_analysis.items():
                    cluster_name = segmented_df[segmented_df['Cluster'] == cluster_num]['Cluster_Name'].iloc[0]
                    
                    st.markdown(f"""
                    <div class="insight-card info">
                        <div class="insight-title">{cluster_name}</div>
                        <div class="insight-content">
                        • Ürün Sayısı: {stats.get('product_count', 0):,} (%{stats.get('percentage', 0):.1f})<br>
                        {f"• Ortalama Satış: ${stats.get('avg_sales', 0)/1e3:.0f}K" if 'avg_sales' in stats else ''}<br>
                        {f"• Ortalama Büyüme: %{stats.get('avg_growth', 0):.1f}" if 'avg_growth' in stats else ''}<br>
                        {f"• Ortalama Fiyat: ${stats.get('avg_price', 0):.2f}" if 'avg_price' in stats else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Büyüme fırsatları
    st.markdown('<h3 class="subsection-title">🚀 Büyüme Fırsatları</h3>', unsafe_allow_html=True)
    
    # En hızlı büyüyen ürünler
    growth_cols = [col for col in df.columns if any(x in col.lower() for x in ['growth', 'büyüme'])]
    if growth_cols:
        last_growth_col = growth_cols[-1]
        
        # Ürün ismi sütununu bul
        product_name_col = None
        name_candidates = ['Molekül', 'Ürün Adı', 'Product', 'İlaç Adı']
        for candidate in name_candidates:
            if candidate in df.columns:
                product_name_col = candidate
                break
        
        if product_name_col:
            top_growth = df.nlargest(10, last_growth_col)
            
            for idx, row in top_growth.iterrows():
                product_name = row[product_name_col]
                growth_rate = row[last_growth_col]
                
                # Satış bilgisini ekle
                sales_info = ""
                sales_cols = [col for col in df.columns if any(x in col.lower() for x in ['sales', 'satış'])]
                if sales_cols:
                    last_sales_col = sales_cols[-1]
                    sales_value = row[last_sales_col]
                    sales_info = f" | Satış: ${sales_value/1e6:.2f}M"
                
                st.markdown(f"""
                <div class="insight-card success">
                    <div class="insight-icon">🚀</div>
                    <div class="insight-title">{product_name}</div>
                    <div class="insight-content">
                    Büyüme Oranı: <strong>%{growth_rate:.1f}</strong>{sales_info}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Risk analizi
    st.markdown('<h3 class="subsection-title">⚠️ Risk Analizi</h3>', unsafe_allow_html=True)
    
    if growth_cols:
        negative_growth = df[df[last_growth_col] < 0]
        
        if len(negative_growth) > 0:
            top_negative = negative_growth.nsmallest(5, last_growth_col)
            
            for idx, row in top_negative.iterrows():
                if product_name_col:
                    product_name = row[product_name_col]
                    growth_rate = row[last_growth_col]
                    
                    st.markdown(f"""
                    <div class="insight-card danger">
                        <div class="insight-icon">⚠️</div>
                        <div class="insight-title">{product_name}</div>
                        <div class="insight-content">
                        Negatif Büyüme: <strong>%{growth_rate:.1f}</strong><br>
                        Bu ürünün performansı düşüşte, strateji gözden geçirilmeli.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("✅ Negatif büyüme gösteren ürün bulunamadı.")

def show_reporting_tab(df, metrics, insights):
    """Raporlama tab'ını göster"""
    st.markdown('<h2 class="section-title">📑 Raporlama ve İndirme</h2>', unsafe_allow_html=True)
    
    # Rapor türü seçimi
    st.markdown('<h3 class="subsection-title">📊 Rapor Türleri</h3>', unsafe_allow_html=True)
    
    report_type = st.radio(
        "Rapor Türü Seçin",
        ['Excel Detaylı Rapor', 'CSV Ham Veri', 'HTML Özet Rapor', 'International Product Raporu'],
        horizontal=True,
        key="report_type_select"
    )
    
    # Rapor oluşturma butonları
    st.markdown('<h3 class="subsection-title">🛠️ Rapor Oluşturma</h3>', unsafe_allow_html=True)
    
    report_cols = st.columns(4)
    
    with report_cols[0]:
        if st.button("📈 Excel Raporu", use_container_width=True, key="excel_report"):
            with st.spinner("Excel raporu oluşturuluyor..."):
                reporting = ReportingEngine()
                report = reporting.create_excel_report(df, metrics, insights)
                
                if report:
                    st.download_button(
                        label="⬇️ İndir",
                        data=report['data'],
                        file_name=report['filename'],
                        mime=report['mime_type'],
                        use_container_width=True
                    )
    
    with report_cols[1]:
        if st.button("📄 CSV Veri", use_container_width=True, key="csv_report"):
            csv_data = df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.download_button(
                label="⬇️ İndir",
                data=csv_data,
                file_name=f"pharma_data_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with report_cols[2]:
        if st.button("🌐 HTML Rapor", use_container_width=True, key="html_report"):
            with st.spinner("HTML raporu oluşturuluyor..."):
                reporting = ReportingEngine()
                report = reporting.create_html_report(df, metrics, insights)
                
                if report:
                    st.download_button(
                        label="⬇️ İndir",
                        data=report['data'],
                        file_name=report['filename'],
                        mime=report['mime_type'],
                        use_container_width=True
                    )
    
    with report_cols[3]:
        if 'International_Product' in df.columns:
            if st.button("🌍 Intl. Rapor", use_container_width=True, key="intl_report"):
                intl_df = df[df['International_Product'] == 1]
                csv_data = intl_df.to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="⬇️ İndir",
                    data=csv_data,
                    file_name=f"international_products_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Hızlı istatistikler
    st.markdown('<h3 class="subsection-title">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)
    
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.metric("Toplam Ürün", f"{len(df):,}")
    
    with stat_cols[1]:
        st.metric("Toplam Sütun", len(df.columns))
    
    with stat_cols[2]:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Bellek Kullanımı", f"{memory_mb:.1f} MB")
    
    with stat_cols[3]:
        intl_count = (df['International_Product'] == 1).sum() if 'International_Product' in df.columns else 0
        st.metric("International Product", f"{intl_count:,}")
    
    # Veri kalitesi metriği
    st.markdown('<h3 class="subsection-title">🔍 Veri Kalitesi Metrikleri</h3>', unsafe_allow_html=True)
    
    quality_cols = st.columns(3)
    
    with quality_cols[0]:
        missing_values = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        missing_percentage = (missing_values / total_cells) * 100
        st.metric("Eksik Değerler", f"{missing_percentage:.1f}%")
    
    with quality_cols[1]:
        duplicate_rows = df.duplicated().sum()
        st.metric("Tekrar Eden Satırlar", f"{duplicate_rows}")
    
    with quality_cols[2]:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Sayısal Sütunlar", f"{numeric_cols}")
    
    # Uygulama sıfırlama
    st.markdown("---")
    
    if st.button("🔄 Uygulamayı Sıfırla", type="secondary", use_container_width=True, key="reset_app"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Uygulama sıfırlandı. Lütfen sayfayı yenileyin.")
        st.rerun()

# ================================================
# 8. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ## ❌ Uygulama Hatası
        
        **Hata Mesajı:** {str(e)}
        
        Lütfen sayfayı yenileyin veya aşağıdaki butona tıklayın:
        """)
        
        if st.button("🔄 Sayfayı Yenile", type="primary", use_container_width=True):
            # Session state'i temizle
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # JavaScript ile sayfayı yenile
            st.rerun()
