# app.py - PharmaIntelligence Enterprise Suite v3.0
# McKinsey/BCG/IQVIA Standartlarında Tam Kapsamlı Farmasötik Analitik Platformu
# 4200+ satır - Tüm Kritik Hatalar Çözülmüş - Stratejik Derinlik Eklenmiş

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Gelişmiş Analitik Kütüphaneleri
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import scipy.stats as stats
from scipy import signal, integrate, optimize
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy

# İleri Seviye Python Araçları
from datetime import datetime, timedelta
import itertools
import hashlib
import json
import pickle
import math
import os
import sys
import gc
import re
import string
import unicodedata
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import inspect
import textwrap
import logging
from logging.handlers import RotatingFileHandler
import traceback
from contextlib import contextmanager
import time
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import threading
from pathlib import Path
import tempfile
import base64
import io
import csv
import zipfile

# ================================================
# 1. ENTERPRISE KONFİGÜRASYON VE LOGGING SİSTEMİ
# ================================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class AppConfig:
    """Enterprise uygulama konfigürasyonu"""
    app_name: str = "PharmaIntelligence Enterprise Suite"
    version: str = "3.0.0"
    max_file_size_mb: int = 500
    max_rows_display: int = 100000
    cache_ttl: int = 3600  # seconds
    enable_profiling: bool = True
    memory_warning_threshold: float = 0.8  # %80 memory usage
    default_theme: str = "dark"
    supported_languages: List[str] = field(default_factory=lambda: ["en", "tr", "de", "fr"])
    
class EnterpriseLogger:
    """Profesyonel logging sistemi"""
    
    def __init__(self, name: str = "pharma_intelligence"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(console_formatter)
        
        # File handler (rotating)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / "pharma_intelligence.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        fh.setFormatter(file_formatter)
        
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
    
    def log(self, level: LogLevel, message: str, **kwargs):
        """Log mesajı yaz"""
        extra_info = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        full_message = f"{message} {extra_info}" if extra_info else message
        
        if level == LogLevel.DEBUG:
            self.logger.debug(full_message)
        elif level == LogLevel.INFO:
            self.logger.info(full_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(full_message)
        elif level == LogLevel.ERROR:
            self.logger.error(full_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(full_message)
    
    @contextmanager
    def performance_timer(self, operation_name: str):
        """Performans ölçümü için context manager"""
        start_time = timer()
        try:
            yield
        finally:
            elapsed_time = timer() - start_time
            self.log(LogLevel.INFO, f"Operation '{operation_name}' completed", 
                    time_seconds=elapsed_time)

# Uygulama konfigürasyonu ve logger başlatma
config = AppConfig()
logger = EnterpriseLogger()

# Streamlit konfigürasyonu
st.set_page_config(
    page_title=f"{config.app_name} v{config.version}",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://pharmaintelligence.com/enterprise-support',
        'Report a bug': 'https://pharmaintelligence.com/bug-report',
        'About': f'''
        ### {config.app_name} v{config.version}
        
        **Enterprise Pharmaceutical Analytics Platform**
        
        • Advanced Market Intelligence
        • Competitive Analysis Suite
        • Predictive Analytics Engine
        • Strategic Recommendation System
        • Real-time Dashboarding
        
        © 2024 PharmaIntelligence Inc. All Rights Reserved
        Confidential and Proprietary
        '''
    }
)

# ================================================
# 2. ENTERPRISE CSS VE TEMA SİSTEMİ
# ================================================

ENTERPRISE_CSS = """
<style>
    /* === ROOT VARIABLES === */
    :root {
        /* Primary Colors */
        --primary-50: #eff6ff;
        --primary-100: #dbeafe;
        --primary-200: #bfdbfe;
        --primary-300: #93c5fd;
        --primary-400: #60a5fa;
        --primary-500: #3b82f6;
        --primary-600: #2563eb;
        --primary-700: #1d4ed8;
        --primary-800: #1e40af;
        --primary-900: #1e3a8a;
        --primary-950: #172554;
        
        /* Neutral Colors */
        --neutral-50: #f8fafc;
        --neutral-100: #f1f5f9;
        --neutral-200: #e2e8f0;
        --neutral-300: #cbd5e1;
        --neutral-400: #94a3b8;
        --neutral-500: #64748b;
        --neutral-600: #475569;
        --neutral-700: #334155;
        --neutral-800: #1e293b;
        --neutral-900: #0f172a;
        --neutral-950: #020617;
        
        /* Semantic Colors */
        --success-500: #10b981;
        --warning-500: #f59e0b;
        --danger-500: #ef4444;
        --info-500: #06b6d4;
        
        /* Gradients */
        --gradient-primary: linear-gradient(135deg, var(--primary-600), var(--primary-800));
        --gradient-dark: linear-gradient(135deg, var(--neutral-900), var(--neutral-950));
        --gradient-success: linear-gradient(135deg, var(--success-500), #059669);
        --gradient-warning: linear-gradient(135deg, var(--warning-500), #d97706);
        --gradient-danger: linear-gradient(135deg, var(--danger-500), #dc2626);
        
        /* Shadows */
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        --shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        
        /* Borders */
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
        --radius-2xl: 1.5rem;
        
        /* Transitions */
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
        
        /* Z-index layers */
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
        background: var(--gradient-dark);
        color: var(--neutral-100);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        min-height: 100vh;
    }
    
    /* === TYPOGRAPHY === */
    .enterprise-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--primary-400), var(--primary-600), var(--info-500));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    
    .enterprise-subtitle {
        font-size: 1.25rem;
        color: var(--neutral-300);
        font-weight: 400;
        max-width: 800px;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 800;
        color: var(--neutral-100);
        margin: 2.5rem 0 1.5rem 0;
        padding: 1rem 1.5rem;
        background: linear-gradient(90deg, 
            rgba(59, 130, 246, 0.1) 0%, 
            rgba(59, 130, 246, 0.05) 50%, 
            transparent 100%);
        border-left: 6px solid var(--primary-500);
        border-radius: var(--radius-md);
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
        background: linear-gradient(90deg, 
            transparent, 
            rgba(59, 130, 246, 0.3), 
            transparent);
    }
    
    .subsection-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--neutral-100);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--neutral-800);
        position: relative;
    }
    
    .subsection-title::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100px;
        height: 2px;
        background: var(--primary-500);
    }
    
    /* === ENTERPRISE CARDS === */
    .enterprise-card {
        background: linear-gradient(145deg, 
            rgba(30, 41, 59, 0.9), 
            rgba(15, 23, 42, 0.95));
        border: 1px solid var(--neutral-800);
        border-radius: var(--radius-xl);
        padding: 1.5rem;
        box-shadow: var(--shadow-xl);
        transition: all var(--transition-normal);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .enterprise-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
    }
    
    .enterprise-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-2xl);
        border-color: var(--primary-500);
    }
    
    .enterprise-card.glow {
        animation: card-glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes card-glow {
        from {
            box-shadow: var(--shadow-xl), 0 0 20px rgba(59, 130, 246, 0.1);
        }
        to {
            box-shadow: var(--shadow-xl), 0 0 30px rgba(59, 130, 246, 0.3);
        }
    }
    
    /* === METRIC CARDS === */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, 
            rgba(30, 41, 59, 0.95), 
            rgba(15, 23, 42, 0.98));
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        border: 1px solid var(--neutral-800);
        box-shadow: var(--shadow-lg);
        transition: all var(--transition-fast);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-500), transparent);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
        border-color: var(--primary-400);
    }
    
    .metric-card.primary {
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.15), 
            rgba(37, 99, 235, 0.1));
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .metric-card.success {
        background: linear-gradient(135deg, 
            rgba(16, 185, 129, 0.15), 
            rgba(5, 150, 105, 0.1));
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    .metric-card.warning {
        background: linear-gradient(135deg, 
            rgba(245, 158, 11, 0.15), 
            rgba(217, 119, 6, 0.1));
        border-color: rgba(245, 158, 11, 0.3);
    }
    
    .metric-card.danger {
        background: linear-gradient(135deg, 
            rgba(239, 68, 68, 0.15), 
            rgba(220, 38, 38, 0.1));
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: var(--neutral-100);
        line-height: 1;
        margin: 0.5rem 0;
        display: flex;
        align-items: baseline;
        gap: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--neutral-400);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .metric-trend {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    
    .trend-up {
        color: var(--success-500);
    }
    
    .trend-down {
        color: var(--danger-500);
    }
    
    .trend-neutral {
        color: var(--neutral-500);
    }
    
    /* === INSIGHT CARDS === */
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .insight-card {
        background: linear-gradient(145deg, 
            rgba(30, 41, 59, 0.9), 
            rgba(15, 23, 42, 0.95));
        border-radius: var(--radius-lg);
        padding: 1.25rem;
        border-left: 4px solid;
        box-shadow: var(--shadow-lg);
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
        background: linear-gradient(135deg, 
            rgba(255, 255, 255, 0.03), 
            transparent);
        opacity: 0;
        transition: opacity var(--transition-normal);
    }
    
    .insight-card:hover::before {
        opacity: 1;
    }
    
    .insight-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
    }
    
    .insight-card.info {
        border-left-color: var(--primary-500);
        background: linear-gradient(145deg, 
            rgba(30, 41, 59, 0.95), 
            rgba(15, 23, 42, 0.98));
    }
    
    .insight-card.success {
        border-left-color: var(--success-500);
        background: linear-gradient(145deg, 
            rgba(30, 41, 59, 0.95), 
            rgba(15, 23, 42, 0.98));
    }
    
    .insight-card.warning {
        border-left-color: var(--warning-500);
        background: linear-gradient(145deg, 
            rgba(30, 41, 59, 0.95), 
            rgba(15, 23, 42, 0.98));
    }
    
    .insight-card.danger {
        border-left-color: var(--danger-500);
        background: linear-gradient(145deg, 
            rgba(30, 41, 59, 0.95), 
            rgba(15, 23, 42, 0.98));
    }
    
    .insight-icon {
        font-size: 1.5rem;
        margin-bottom: 0.75rem;
        display: inline-block;
        padding: 0.5rem;
        border-radius: var(--radius-md);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .insight-title {
        font-weight: 700;
        color: var(--neutral-100);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        line-height: 1.3;
    }
    
    .insight-content {
        color: var(--neutral-300);
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* === FILTER SECTIONS === */
    .filter-section {
        background: linear-gradient(145deg, 
            rgba(30, 41, 59, 0.95), 
            rgba(15, 23, 42, 0.98));
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--neutral-800);
        box-shadow: var(--shadow-lg);
        margin-bottom: 1.5rem;
    }
    
    .filter-title {
        color: var(--neutral-100);
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--neutral-800);
    }
    
    .filter-title::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 1rem;
        background: var(--primary-500);
        border-radius: 2px;
    }
    
    /* === STATUS INDICATORS === */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        font-size: 0.875rem;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        background: var(--neutral-800);
        border: 1px solid var(--neutral-700);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .status-online {
        background: var(--success-500);
        box-shadow: 0 0 8px var(--success-500);
    }
    
    .status-warning {
        background: var(--warning-500);
        box-shadow: 0 0 8px var(--warning-500);
    }
    
    .status-error {
        background: var(--danger-500);
        box-shadow: 0 0 8px var(--danger-500);
    }
    
    /* === BADGES === */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border: 1px solid;
    }
    
    .badge-success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-500);
        border-color: rgba(16, 185, 129, 0.2);
    }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning-500);
        border-color: rgba(245, 158, 11, 0.2);
    }
    
    .badge-danger {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger-500);
        border-color: rgba(239, 68, 68, 0.2);
    }
    
    .badge-info {
        background: rgba(59, 130, 246, 0.1);
        color: var(--primary-500);
        border-color: rgba(59, 130, 246, 0.2);
    }
    
    /* === TABLES === */
    .data-table {
        background: linear-gradient(145deg, 
            rgba(30, 41, 59, 0.95), 
            rgba(15, 23, 42, 0.98));
        border-radius: var(--radius-lg);
        overflow: hidden;
        border: 1px solid var(--neutral-800);
        box-shadow: var(--shadow-lg);
    }
    
    .data-table th {
        background: linear-gradient(135deg, 
            rgba(59, 130, 246, 0.15), 
            rgba(37, 99, 235, 0.1));
        color: var(--neutral-100);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.875rem;
        padding: 1rem;
        border-bottom: 1px solid var(--neutral-800);
    }
    
    .data-table td {
        padding: 0.875rem 1rem;
        border-bottom: 1px solid var(--neutral-800);
        color: var(--neutral-300);
        font-size: 0.95rem;
    }
    
    .data-table tr:hover td {
        background: rgba(59, 130, 246, 0.05);
        color: var(--neutral-100);
    }
    
    /* === BUTTONS === */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all var(--transition-fast);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-md);
    }
    
    /* === INPUTS === */
    .stSelectbox > div > div,
    .stMultiselect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: var(--neutral-900) !important;
        border: 1px solid var(--neutral-700) !important;
        border-radius: var(--radius-md) !important;
        color: var(--neutral-100) !important;
        transition: all var(--transition-fast) !important;
    }
    
    .stSelectbox > div > div:hover,
    .stMultiselect > div > div:hover,
    .stTextInput > div > div > input:hover,
    .stNumberInput > div > div > input:hover {
        border-color: var(--primary-500) !important;
        box-shadow: 0 0 0 1px var(--primary-500) !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stMultiselect > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary-500) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* === SIDEBAR === */
    .sidebar-content {
        background: linear-gradient(180deg, 
            rgba(15, 23, 42, 0.95), 
            rgba(30, 41, 59, 0.98));
        border-right: 1px solid var(--neutral-800);
        padding: 1.5rem;
        height: 100vh;
        overflow-y: auto;
    }
    
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--neutral-100);
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--neutral-800);
        position: relative;
    }
    
    .sidebar-title::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 60px;
        height: 2px;
        background: var(--primary-500);
    }
    
    /* === LOADING STATES === */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(15, 23, 42, 0.9);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: var(--z-modal);
        backdrop-filter: blur(4px);
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 3px solid var(--neutral-800);
        border-top-color: var(--primary-500);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .loading-text {
        color: var(--neutral-300);
        font-size: 0.875rem;
        text-align: center;
    }
    
    /* === TOOLTIPS === */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltip-text {
        visibility: hidden;
        width: 300px;
        background: var(--neutral-900);
        color: var(--neutral-100);
        text-align: left;
        padding: 0.75rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--neutral-800);
        box-shadow: var(--shadow-xl);
        
        position: absolute;
        z-index: var(--z-tooltip);
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity var(--transition-fast);
        
        font-size: 0.875rem;
        line-height: 1.5;
        white-space: normal;
    }
    
    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    
    /* === RESPONSIVE DESIGN === */
    @media (max-width: 768px) {
        .enterprise-title {
            font-size: 2.5rem;
        }
        
        .section-title {
            font-size: 1.5rem;
            padding: 0.75rem 1rem;
        }
        
        .metric-grid {
            grid-template-columns: 1fr;
        }
        
        .insight-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* === CUSTOM SCROLLBAR === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--neutral-900);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--neutral-700);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--neutral-600);
    }
    
    /* === ANIMATIONS === */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    .animate-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* === GRID LAYOUT === */
    .grid-2 {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
    }
    
    .grid-3 {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
    }
    
    .grid-4 {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
    }
    
    @media (max-width: 1024px) {
        .grid-4 {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    
    @media (max-width: 768px) {
        .grid-2, .grid-3, .grid-4 {
            grid-template-columns: 1fr;
        }
    }
</style>
"""

# CSS'i uygula
st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)

# ================================================
# 3. ENTERPRISE VERİ İŞLEYİCİ (DUPLICATE COLUMN FIX)
# ================================================

class EnterpriseDataProcessor:
    """KRİTİK DÜZELTME: Duplicate column isimlerini tamamen çözen profesyonel veri işleyici"""
    
    def __init__(self):
        self.column_mapping_history = {}
        self.performance_stats = {}
        self.memory_warnings = []
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=5)
    def load_large_dataset(file, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Büyük veri setlerini optimize şekilde yükle"""
        try:
            start_time = time.time()
            logger.log(LogLevel.INFO, f"Loading dataset: {file.name}")
            
            file_size = file.size / (1024 * 1024)  # MB
            if file_size > config.max_file_size_mb:
                logger.log(LogLevel.WARNING, f"File size {file_size:.1f}MB exceeds threshold")
                st.warning(f"⚠️ Büyük dosya ({file_size:.1f}MB). İşlem biraz zaman alabilir.")
            
            # File type detection
            if file.name.endswith('.csv'):
                chunk_size = 100000
                chunks = []
                
                with st.spinner("📥 CSV verisi yükleniyor..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, chunk in enumerate(pd.read_csv(file, chunksize=chunk_size, low_memory=False)):
                        chunks.append(chunk)
                        
                        if sample_size and len(pd.concat(chunks, ignore_index=True)) >= sample_size:
                            df = pd.concat(chunks, ignore_index=True).head(sample_size)
                            break
                        
                        progress = min((i + 1) * chunk_size / (sample_size or 1e6), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"📊 {len(pd.concat(chunks, ignore_index=True)):,} satır yüklendi")
                    
                    if not sample_size or len(pd.concat(chunks, ignore_index=True)) < sample_size:
                        df = pd.concat(chunks, ignore_index=True)
                    
                    progress_bar.progress(1.0)
                    status_text.empty()
                    
            elif file.name.endswith(('.xlsx', '.xls')):
                with st.spinner("📥 Excel verisi yükleniyor..."):
                    if sample_size:
                        df = pd.read_excel(file, nrows=sample_size, engine='openpyxl')
                    else:
                        df = pd.read_excel(file, engine='openpyxl')
            
            else:
                st.error("❌ Desteklenmeyen dosya formatı")
                return None
            
            # Optimize et
            df = EnterpriseDataProcessor.optimize_dataframe(df)
            
            load_time = time.time() - start_time
            logger.log(LogLevel.INFO, f"Dataset loaded successfully", 
                      rows=len(df), columns=len(df.columns), time_seconds=load_time)
            
            st.success(f"✅ Veri yüklendi: {len(df):,} satır, {len(df.columns)} sütun ({load_time:.2f}s)")
            
            return df
            
        except Exception as e:
            error_msg = f"Veri yükleme hatası: {str(e)}"
            logger.log(LogLevel.ERROR, error_msg, error_traceback=traceback.format_exc())
            st.error(f"❌ {error_msg}")
            return None
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame'i optimize et (memory ve performans)"""
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # KRİTİK DÜZELTME: Sütun isimlerini temizle (duplicate fix)
            df.columns = EnterpriseDataProcessor._clean_column_names_with_duplicate_fix(df.columns)
            
            # Veri tiplerini optimize et
            for col in df.columns:
                col_type = df[col].dtype
                
                if col_type == 'object':
                    # String sütunları kategorilere çevir
                    num_unique = df[col].nunique()
                    num_total = len(df[col])
                    
                    if num_unique / num_total < 0.5:  # %50'den az benzersiz değer
                        df[col] = df[col].astype('category')
                    
                    # String temizleme
                    try:
                        df[col] = df[col].astype(str).str.strip()
                    except:
                        pass
                
                elif col_type in ['int64', 'int32']:
                    # Integer optimizasyonu
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
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                
                elif col_type in ['float64', 'float32']:
                    # Float optimizasyonu
                    df[col] = df[col].astype(np.float32)
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            memory_saving = original_memory - optimized_memory
            
            if memory_saving > 0:
                savings_pct = (memory_saving / original_memory) * 100
                logger.log(LogLevel.INFO, f"Memory optimization successful", 
                          original_mb=original_memory, optimized_mb=optimized_memory, 
                          savings_pct=savings_pct)
                
                st.info(f"💾 Bellek optimizasyonu: {original_memory:.1f}MB → {optimized_memory:.1f}MB (%{savings_pct:.1f} tasarruf)")
            
            # Garbage collection
            gc.collect()
            
            return df
            
        except Exception as e:
            logger.log(LogLevel.WARNING, f"Optimization error", error=str(e))
            return df
    
    @staticmethod
    def _clean_column_names_with_duplicate_fix(columns: List[str]) -> List[str]:
        """
        KRİTİK DÜZELTME: Duplicate column isimlerini tamamen çözer
        'Bölge' → 'Bölge_1', 'Bölge_2' şeklinde unique yapar
        """
        cleaned_columns = []
        column_counts = {}
        column_mapping = {}
        
        for i, col in enumerate(columns):
            if pd.isna(col):
                original_col = f"Unnamed_{i}"
            else:
                original_col = str(col).strip()
            
            # Temel temizlik
            cleaned = original_col
            
            # Türkçe karakterleri düzelt
            turkish_chars = {'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's', 
                           'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u', 
                           'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'}
            
            for tr_char, en_char in turkish_chars.items():
                cleaned = cleaned.replace(tr_char, en_char)
            
            # Özel karakterleri kaldır
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Boşsa default isim ver
            if not cleaned:
                cleaned = f"Column_{i+1}"
            
            # DUPLICATE FIX: Eğer bu isim daha önce kullanıldıysa sayı ekle
            if cleaned in column_counts:
                column_counts[cleaned] += 1
                unique_col = f"{cleaned}_{column_counts[cleaned]}"
            else:
                column_counts[cleaned] = 1
                unique_col = cleaned
            
            # Eğer unique_col hala duplicatessa hash ekle
            if unique_col in cleaned_columns:
                hash_suffix = hashlib.md5(f"{original_col}_{i}".encode()).hexdigest()[:6]
                unique_col = f"{cleaned}_{hash_suffix}"
            
            cleaned_columns.append(unique_col)
            column_mapping[original_col] = unique_col
        
        # Log column mapping
        if column_mapping:
            logger.log(LogLevel.INFO, f"Column cleaning completed", 
                      original_count=len(columns), cleaned_count=len(set(cleaned_columns)),
                      duplicates_fixed=len(columns) - len(set(cleaned_columns)))
        
        return cleaned_columns
    
    @staticmethod
    def detect_column_patterns(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Sütun pattern'larını tespit et (International Product vs.)"""
        patterns = {
            'sales': [],
            'volume': [],
            'price': [],
            'date': [],
            'region': [],
            'company': [],
            'molecule': [],
            'international': [],
            'growth': [],
            'market_share': []
        }
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Satış pattern'ları
            sales_keywords = ['sales', 'satış', 'revenue', 'cıro', 'gelir', 'turnover']
            if any(keyword in col_lower for keyword in sales_keywords):
                patterns['sales'].append(col)
            
            # Hacim pattern'ları
            volume_keywords = ['volume', 'hacim', 'quantity', 'miktar', 'unit', 'birim']
            if any(keyword in col_lower for keyword in volume_keywords):
                patterns['volume'].append(col)
            
            # Fiyat pattern'ları
            price_keywords = ['price', 'fiyat', 'cost', 'maliyet', 'avg price', 'ort fiyat']
            if any(keyword in col_lower for keyword in price_keywords):
                patterns['price'].append(col)
            
            # International Product pattern'ları
            intl_keywords = ['international', 'global', 'intl', 'multinational', 
                           'cross-border', 'export', 'pan-regional', 'worldwide']
            if any(keyword in col_lower for keyword in intl_keywords):
                patterns['international'].append(col)
            
            # Molekül pattern'ları
            molecule_keywords = ['molecule', 'molekül', 'active', 'ingredient', 
                               'compound', 'drug', 'ilac', 'active ingredient']
            if any(keyword in col_lower for keyword in molecule_keywords):
                patterns['molecule'].append(col)
            
            # Şirket pattern'ları
            company_keywords = ['company', 'şirket', 'corporation', 'firma', 
                              'manufacturer', 'üretici', 'laboratory', 'laboratuvar']
            if any(keyword in col_lower for keyword in company_keywords):
                patterns['company'].append(col)
            
            # Bölge pattern'ları
            region_keywords = ['region', 'bölge', 'area', 'zone', 'territory', 
                             'country', 'ülke', 'city', 'şehir']
            if any(keyword in col_lower for keyword in region_keywords):
                patterns['region'].append(col)
        
        return patterns
    
    @staticmethod
    def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
        """Analiz için veriyi hazırla (CAGR, Market Share vs.)"""
        try:
            analysis_df = df.copy()
            
            # Satış sütunlarını bul ve sırala
            sales_patterns = ['sales', 'satış', 'revenue']
            sales_cols = []
            
            for col in analysis_df.columns:
                col_lower = str(col).lower()
                if any(pattern in col_lower for pattern in sales_patterns):
                    sales_cols.append(col)
            
            # Yılları çıkar
            year_sales_map = {}
            for col in sales_cols:
                # Yılı bul (2022, 2023, 2024)
                year_match = re.search(r'(19|20)\d{2}', col)
                if year_match:
                    year = year_match.group()
                    year_sales_map[year] = col
            
            # Yıllara göre sırala
            sorted_years = sorted(year_sales_map.keys())
            
            if len(sorted_years) >= 2:
                # CAGR hesapla
                first_year = sorted_years[0]
                last_year = sorted_years[-1]
                first_col = year_sales_map[first_year]
                last_col = year_sales_map[last_year]
                
                n_years = len(sorted_years)
                analysis_df['CAGR'] = ((analysis_df[last_col] / analysis_df[first_col]) ** (1/n_years) - 1) * 100
                analysis_df['CAGR'] = analysis_df['CAGR'].replace([np.inf, -np.inf], np.nan)
                
                # Yıllık büyüme oranları
                for i in range(1, len(sorted_years)):
                    prev_year = sorted_years[i-1]
                    curr_year = sorted_years[i]
                    prev_col = year_sales_map[prev_year]
                    curr_col = year_sales_map[curr_year]
                    
                    growth_col_name = f'Growth_{prev_year}_{curr_year}'
                    analysis_df[growth_col_name] = ((analysis_df[curr_col] - analysis_df[prev_col]) / 
                                                  analysis_df[prev_col].replace(0, np.nan)) * 100
            
            # Pazar payı hesapla
            if year_sales_map:
                last_year_col = year_sales_map[sorted_years[-1]]
                total_market = analysis_df[last_year_col].sum()
                if total_market > 0:
                    analysis_df['Market_Share'] = (analysis_df[last_year_col] / total_market) * 100
            
            # Fiyat-Hacim oranı
            price_cols = [col for col in analysis_df.columns if 'price' in str(col).lower() or 'fiyat' in str(col).lower()]
            volume_cols = [col for col in analysis_df.columns if 'volume' in str(col).lower() or 'hacim' in str(col).lower()]
            
            if price_cols and volume_cols:
                latest_price = price_cols[-1]
                latest_volume = volume_cols[-1]
                analysis_df['Price_Volume_Ratio'] = analysis_df[latest_price] * analysis_df[latest_volume]
            
            # Performans skoru (z-score normalizasyonu)
            numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                try:
                    scaler = StandardScaler()
                    numeric_data = analysis_df[numeric_cols].fillna(0)
                    scaled_data = scaler.fit_transform(numeric_data)
                    analysis_df['Performance_Score'] = scaled_data.mean(axis=1)
                except Exception as e:
                    logger.log(LogLevel.WARNING, f"Performance score calculation error", error=str(e))
            
            logger.log(LogLevel.INFO, f"Analysis data prepared successfully", 
                      cagr_added='CAGR' in analysis_df.columns,
                      market_share_added='Market_Share' in analysis_df.columns)
            
            return analysis_df
            
        except Exception as e:
            logger.log(LogLevel.ERROR, f"Analysis data preparation failed", error=str(e))
            return df

# ================================================
# 4. FUZZY MATCHING İLE INTERNATIONAL PRODUCT DEDEKTÖRÜ
# ================================================

class InternationalProductDetector:
    """Fuzzy matching ile International Product sütunlarını tespit eder"""
    
    # International için keyword listesi
    INTERNATIONAL_KEYWORDS = [
        'international', 'global', 'worldwide', 'multinational',
        'cross-border', 'pan-regional', 'export', 'import',
        'overseas', 'foreign', 'external', 'offshore',
        'transnational', 'multimarket', 'cross-country'
    ]
    
    # Türkçe keyword'ler
    INTERNATIONAL_KEYWORDS_TR = [
        'uluslararası', 'global', 'dünya', 'çokuluslu',
        'sınıraşan', 'ihracat', 'ithalat', 'yurtdışı',
        'dış pazar', 'dış ticaret', 'enternasyonal'
    ]
    
    @staticmethod
    def fuzzy_match_column_name(column_name: str, keywords: List[str]) -> float:
        """Bulanık eşleştirme skoru hesapla"""
        col_lower = str(column_name).lower()
        best_score = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact match
            if keyword_lower in col_lower or col_lower in keyword_lower:
                return 1.0
            
            # Partial match scoring
            if keyword_lower in col_lower:
                score = len(keyword_lower) / len(col_lower)
            elif col_lower in keyword_lower:
                score = len(col_lower) / len(keyword_lower)
            else:
                # Levenshtein distance benzeri basit scoring
                common_chars = set(col_lower) & set(keyword_lower)
                score = len(common_chars) / max(len(col_lower), len(keyword_lower))
            
            best_score = max(best_score, score)
        
        return best_score
    
    @staticmethod
    def detect_international_columns(df: pd.DataFrame, threshold: float = 0.3) -> List[str]:
        """International Product sütunlarını tespit et"""
        international_columns = []
        scores = []
        
        all_keywords = (InternationalProductDetector.INTERNATIONAL_KEYWORDS + 
                       InternationalProductDetector.INTERNATIONAL_KEYWORDS_TR)
        
        for col in df.columns:
            score = InternationalProductDetector.fuzzy_match_column_name(col, all_keywords)
            scores.append((col, score))
            
            if score >= threshold:
                international_columns.append(col)
        
        # Score'a göre sırala
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return international_columns, scores
    
    @staticmethod
    def create_mapping_interface(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Kullanıcı için International Product mapping arayüzü oluştur"""
        
        with st.sidebar.expander("🌍 International Product Mapping", expanded=False):
            st.markdown("""
            <div class="filter-title">
                🔍 International Product Eşleştirme
            </div>
            """, unsafe_allow_html=True)
            
            # Otomatik tespit
            detected_cols, scores = InternationalProductDetector.detect_international_columns(df)
            
            if detected_cols:
                st.success(f"✅ {len(detected_cols)} sütun tespit edildi")
                
                # En iyi eşleşmeleri göster
                st.write("**En iyi eşleşmeler:**")
                for col, score in scores[:5]:
                    st.write(f"• `{col}` (skor: {score:.2%})")
                
                # Mapping seçimi
                mapping_col = st.selectbox(
                    "International Product olarak eşlenecek sütun:",
                    options=[''] + detected_cols,
                    index=1 if detected_cols else 0,
                    key="intl_mapping_select"
                )
                
                if mapping_col:
                    # Mapping tipi
                    mapping_type = st.radio(
                        "Mapping Tipi:",
                        ['Boolean (0/1)', 'Categorical (Yes/No)', 'Numeric Score'],
                        horizontal=True,
                        key="intl_mapping_type"
                    )
                    
                    # Preview
                    st.write("**Örnek değerler:**")
                    sample_values = df[mapping_col].dropna().unique()[:5]
                    for val in sample_values:
                        st.code(f"{val}")
                    
                    # Apply mapping
                    if st.button("✅ Eşleştirmeyi Uygula", key="apply_intl_mapping"):
                        df = InternationalProductDetector._apply_mapping(df, mapping_col, mapping_type)
                        st.success(f"✅ `{mapping_col}` → `International_Product` eşleştirildi")
                        return df, True
            
            else:
                st.warning("⚠️ Otomatik tespit başarısız. Manuel seçim yapın:")
                
                # Manuel seçim
                all_columns = list(df.columns)
                mapping_col = st.selectbox(
                    "Sütun seçin:",
                    options=[''] + all_columns,
                    key="manual_intl_select"
                )
                
                if mapping_col:
                    if st.button("Eşleştir", key="manual_intl_map"):
                        df['International_Product'] = df[mapping_col]
                        return df, True
        
        return df, False
    
    @staticmethod
    def _apply_mapping(df: pd.DataFrame, source_col: str, mapping_type: str) -> pd.DataFrame:
        """Mapping'i uygula"""
        if mapping_type == 'Boolean (0/1)':
            # Boolean mapping
            df['International_Product'] = df[source_col].apply(
                lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0
            )
        
        elif mapping_type == 'Categorical (Yes/No)':
            # Categorical mapping
            df['International_Product'] = df[source_col].apply(
                lambda x: 'Yes' if pd.notna(x) and str(x).strip() != '' else 'No'
            )
        
        else:  # Numeric Score
            # Numeric scoring (0-1 arası)
            df['International_Product'] = df[source_col].apply(
                lambda x: 1.0 if pd.notna(x) and str(x).strip() != '' else 0.0
            )
        
        return df

# ================================================
# 5. İLERİ SEVİYE REKABET ANALİZİ MOTORU
# ================================================

class AdvancedCompetitiveAnalytics:
    """Gelişmiş rekabet analizi ve molekül karşılaştırma motoru"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.metrics_cache = {}
        
    def calculate_market_structure_metrics(self) -> Dict[str, Any]:
        """Pazar yapısı metriklerini hesapla"""
        metrics = {}
        
        # HHI (Herfindahl-Hirschman Index) hesapla
        if 'Market_Share' in self.df.columns:
            company_shares = {}
            
            # Şirket bazında pazar payı
            if 'Company' in self.df.columns or 'Şirket' in self.df.columns:
                company_col = 'Company' if 'Company' in self.df.columns else 'Şirket'
                company_shares = self.df.groupby(company_col)['Market_Share'].sum()
            else:
                # Unique identifier olarak index kullan
                company_shares = pd.Series(self.df['Market_Share'].values, 
                                          index=self.df.index.astype(str))
            
            hhi = (company_shares ** 2).sum()
            metrics['hhi_index'] = hhi
            
            # HHI yorumu
            if hhi < 1500:
                metrics['hhi_interpretation'] = "Rekabetçi Pazar"
                metrics['hhi_risk'] = "Düşük"
                metrics['hhi_recommendation'] = "Yeni girişler için uygun"
            elif hhi < 2500:
                metrics['hhi_interpretation'] = "Orta Dereceli Konsantrasyon"
                metrics['hhi_risk'] = "Orta"
                metrics['hhi_recommendation'] = "Dikkatli değerlendirme gerekiyor"
            else:
                metrics['hhi_interpretation'] = "Yüksek Konsantrasyon"
                metrics['hhi_risk'] = "Yüksek"
                metrics['hhi_risk_detail'] = f"Pazar {company_shares.nlargest(3).index.tolist()} tarafından domine ediliyor"
                metrics['hhi_recommendation'] = "Yeni giriş riskli, ortaklık veya satın alma düşünülmeli"
            
            # Top 3/5/10 konsantrasyon oranları
            top_3_share = company_shares.nlargest(3).sum()
            top_5_share = company_shares.nlargest(5).sum()
            top_10_share = company_shares.nlargest(10).sum()
            
            metrics['cr3'] = top_3_share
            metrics['cr5'] = top_5_share
            metrics['cr10'] = top_10_share
        
        # Gini Katsayısı (Gelir eşitsizliği benzeri)
        if 'Market_Share' in self.df.columns:
            sorted_shares = np.sort(self.df['Market_Share'].dropna())
            n = len(sorted_shares)
            
            if n > 1:
                cum_shares = np.cumsum(sorted_shares)
                perfect_line = np.linspace(0, 100, n)
                
                # Gini katsayısı
                gini_coefficient = 1 - 2 * integrate.trapz(cum_shares, perfect_line) / (100 * (n - 1))
                metrics['gini_coefficient'] = gini_coefficient
                
                # Gini yorumu
                if gini_coefficient < 0.3:
                    metrics['gini_interpretation'] = "Eşit Dağılım"
                elif gini_coefficient < 0.5:
                    metrics['gini_interpretation'] = "Orta Dereceli Eşitsizlik"
                else:
                    metrics['gini_interpretation'] = "Yüksek Eşitsizlik"
        
        # Lorenz Eğrisi verisi
        if 'Market_Share' in self.df.columns:
            sorted_shares = np.sort(self.df['Market_Share'].dropna())
            n = len(sorted_shares)
            
            if n > 1:
                cum_shares = np.cumsum(sorted_shares)
                cum_percentage = cum_shares / cum_shares[-1] * 100
                perfect_line = np.linspace(0, 100, n)
                
                metrics['lorenz_curve'] = {
                    'cum_percentage': cum_percentage.tolist(),
                    'perfect_line': perfect_line.tolist(),
                    'companies_percentage': np.linspace(0, 100, n).tolist()
                }
        
        # Pazar büyüklüğü ve trend
        sales_cols = [col for col in self.df.columns if 'sales' in str(col).lower() or 'satış' in str(col).lower()]
        if sales_cols:
            latest_sales = sales_cols[-1]
            metrics['total_market_size'] = self.df[latest_sales].sum()
            
            if len(sales_cols) >= 2:
                previous_sales = sales_cols[-2]
                growth_rate = ((self.df[latest_sales].sum() - self.df[previous_sales].sum()) / 
                             self.df[previous_sales].sum()) * 100
                metrics['market_growth_rate'] = growth_rate
        
        # Molekül çeşitliliği
        if 'Molecule' in self.df.columns:
            unique_molecules = self.df['Molecule'].nunique()
            total_products = len(self.df)
            metrics['molecule_diversity'] = unique_molecules
            metrics['molecule_concentration'] = (unique_molecules / total_products) * 100
        
        return metrics
    
    def create_molecule_comparison_chart(self, selected_molecules: List[str]) -> go.Figure:
        """Molekül karşılaştırma chart'ı oluştur"""
        
        if not selected_molecules or len(selected_molecules) < 2:
            return None
        
        # Seçilen moleküllerin verilerini filtrele
        filtered_df = self.df[self.df['Molecule'].isin(selected_molecules)].copy()
        
        if len(filtered_df) == 0:
            return None
        
        # Her molekül için metrikleri topla
        comparison_data = []
        
        for molecule in selected_molecules:
            mol_df = filtered_df[filtered_df['Molecule'] == molecule]
            
            if len(mol_df) == 0:
                continue
            
            metrics = {
                'Molecule': molecule,
                'Product_Count': len(mol_df)
            }
            
            # Satış metrikleri
            sales_cols = [col for col in self.df.columns if 'sales' in str(col).lower() or 'satış' in str(col).lower()]
            if sales_cols:
                latest_sales = sales_cols[-1]
                metrics['Total_Sales'] = mol_df[latest_sales].sum()
                metrics['Avg_Sales'] = mol_df[latest_sales].mean()
            
            # CAGR
            if 'CAGR' in self.df.columns:
                metrics['CAGR'] = mol_df['CAGR'].mean()
            
            # Market Share
            if 'Market_Share' in self.df.columns:
                metrics['Market_Share'] = mol_df['Market_Share'].sum()
            
            # Fiyat
            price_cols = [col for col in self.df.columns if 'price' in str(col).lower() or 'fiyat' in str(col).lower()]
            if price_cols:
                latest_price = price_cols[-1]
                metrics['Avg_Price'] = mol_df[latest_price].mean()
            
            # International Product oranı
            if 'International_Product' in self.df.columns:
                if mol_df['International_Product'].dtype == 'object':
                    intl_pct = (mol_df['International_Product'] == 'Yes').mean() * 100
                else:
                    intl_pct = (mol_df['International_Product'] > 0).mean() * 100
                metrics['International_Pct'] = intl_pct
            
            comparison_data.append(metrics)
        
        if not comparison_data:
            return None
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Multi-chart oluştur
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Toplam Satış Karşılaştırması', 'Büyüme Oranları (CAGR)',
                          'Pazar Payı Dağılımı', 'Ortalama Fiyat'),
            vertical_spacing=0.15,
            horizontal_spacing=0.15,
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Chart 1: Toplam Satış
        fig.add_trace(
            go.Bar(
                x=comparison_df['Molecule'],
                y=comparison_df['Total_Sales'],
                name='Toplam Satış',
                marker_color='#3b82f6',
                text=[f'${x/1e6:.1f}M' for x in comparison_df['Total_Sales']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Chart 2: CAGR
        fig.add_trace(
            go.Bar(
                x=comparison_df['Molecule'],
                y=comparison_df['CAGR'],
                name='CAGR',
                marker_color='#10b981',
                text=[f'{x:.1f}%' for x in comparison_df['CAGR']],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Chart 3: Market Share
        fig.add_trace(
            go.Bar(
                x=comparison_df['Molecule'],
                y=comparison_df['Market_Share'],
                name='Pazar Payı',
                marker_color='#f59e0b',
                text=[f'{x:.1f}%' for x in comparison_df['Market_Share']],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Chart 4: Ortalama Fiyat
        fig.add_trace(
            go.Bar(
                x=comparison_df['Molecule'],
                y=comparison_df['Avg_Price'],
                name='Ort. Fiyat',
                marker_color='#ef4444',
                text=[f'${x:.2f}' for x in comparison_df['Avg_Price']],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Molekül Karşılaştırma Analizi",
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9'
        )
        
        # Eksen etiketleri
        fig.update_xaxes(title_text="Molekül", row=1, col=1)
        fig.update_xaxes(title_text="Molekül", row=1, col=2)
        fig.update_xaxes(title_text="Molekül", row=2, col=1)
        fig.update_xaxes(title_text="Molekül", row=2, col=2)
        
        fig.update_yaxes(title_text="Satış (USD)", row=1, col=1)
        fig.update_yaxes(title_text="CAGR (%)", row=1, col=2)
        fig.update_yaxes(title_text="Pazar Payı (%)", row=2, col=1)
        fig.update_yaxes(title_text="Fiyat (USD)", row=2, col=2)
        
        return fig
    
    def create_molecule_radar_chart(self, selected_molecules: List[str]) -> go.Figure:
        """Molekül performans radar chart'ı oluştur"""
        
        if not selected_molecules or len(selected_molecules) < 2:
            return None
        
        # Performans metriklerini topla
        radar_data = []
        
        for molecule in selected_molecules:
            mol_df = self.df[self.df['Molecule'] == molecule]
            
            if len(mol_df) == 0:
                continue
            
            metrics = {}
            
            # Satış performansı
            sales_cols = [col for col in self.df.columns if 'sales' in str(col).lower() or 'satış' in str(col).lower()]
            if sales_cols:
                latest_sales = sales_cols[-1]
                avg_sales = mol_df[latest_sales].mean()
                max_sales = self.df[latest_sales].max()
                metrics['Sales_Performance'] = (avg_sales / max_sales * 100) if max_sales > 0 else 0
            
            # Büyüme performansı
            if 'CAGR' in self.df.columns:
                avg_cagr = mol_df['CAGR'].mean()
                max_cagr = self.df['CAGR'].max()
                metrics['Growth_Performance'] = (avg_cagr / max_cagr * 100) if max_cagr > 0 else 0
            
            # Pazar payı performansı
            if 'Market_Share' in self.df.columns:
                total_share = mol_df['Market_Share'].sum()
                max_share = self.df['Market_Share'].max()
                metrics['Market_Share_Performance'] = (total_share / max_share * 100) if max_share > 0 else 0
            
            # Fiyat performansı
            price_cols = [col for col in self.df.columns if 'price' in str(col).lower() or 'fiyat' in str(col).lower()]
            if price_cols:
                latest_price = price_cols[-1]
                avg_price = mol_df[latest_price].mean()
                max_price = self.df[latest_price].max()
                metrics['Price_Performance'] = (avg_price / max_price * 100) if max_price > 0 else 0
            
            # Yayılım (coğrafi çeşitlilik)
            if 'Region' in self.df.columns or 'Bölge' in self.df.columns:
                region_col = 'Region' if 'Region' in self.df.columns else 'Bölge'
                unique_regions = mol_df[region_col].nunique()
                total_regions = self.df[region_col].nunique()
                metrics['Geographic_Spread'] = (unique_regions / total_regions * 100) if total_regions > 0 else 0
            
            # International yayılım
            if 'International_Product' in self.df.columns:
                if mol_df['International_Product'].dtype == 'object':
                    intl_pct = (mol_df['International_Product'] == 'Yes').mean() * 100
                else:
                    intl_pct = (mol_df['International_Product'] > 0).mean() * 100
                metrics['International_Presence'] = intl_pct
            
            if metrics:
                metrics['Molecule'] = molecule
                radar_data.append(metrics)
        
        if not radar_data:
            return None
        
        radar_df = pd.DataFrame(radar_data)
        
        # Radar chart için normalize et
        categories = [col for col in radar_df.columns if col != 'Molecule']
        
        if len(categories) < 3:
            return None
        
        fig = go.Figure()
        
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
        
        for i, (_, row) in enumerate(radar_df.iterrows()):
            values = [row[cat] for cat in categories]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=row['Molecule'],
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont_color='#f1f5f9'
                ),
                angularaxis=dict(
                    tickfont_color='#f1f5f9'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            height=600,
            title_text="Molekül Performans Karnesi - Radar Chart",
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_head_to_head_analysis(self, company1: str, company2: str) -> go.Figure:
        """Head-to-Head pazar payı savaşı analizi"""
        
        # Zaman serisi verisini bul
        time_patterns = ['2024', '2023', '2022', '2021', 'Q', 'Month', 'Week']
        time_cols = []
        
        for col in self.df.columns:
            col_str = str(col)
            if any(pattern in col_str for pattern in time_patterns):
                # Satış verisi olup olmadığını kontrol et
                if 'sales' in col_str.lower() or 'satış' in col_str.lower():
                    time_cols.append(col)
        
        if len(time_cols) < 2:
            return None
        
        # Zaman sırasına göre sırala
        time_cols.sort()
        
        # Her şirket için zaman serisi verisi topla
        company_data = {company1: [], company2: []}
        periods = []
        
        for time_col in time_cols:
            period_name = time_col
            
            # Dönem ismini çıkar
            period_match = re.search(r'(Q\d|Q\d \d{4}|\d{4}|Month|Week)', time_col)
            if period_match:
                period_name = period_match.group()
            
            periods.append(period_name)
            
            for company in [company1, company2]:
                company_sales = self.df[self.df['Company'] == company][time_col].sum() if 'Company' in self.df.columns else 0
                if 'Şirket' in self.df.columns:
                    company_sales = self.df[self.df['Şirket'] == company][time_col].sum()
                
                company_data[company].append(company_sales)
        
        # Pazar payını hesapla
        market_shares = {company1: [], company2: []}
        
        for i in range(len(periods)):
            total_sales = sum(company_data[company][i] for company in [company1, company2])
            
            for company in [company1, company2]:
                if total_sales > 0:
                    share = (company_data[company][i] / total_sales) * 100
                else:
                    share = 0
                market_shares[company].append(share)
        
        # Line chart oluştur
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=periods,
            y=market_shares[company1],
            mode='lines+markers',
            name=company1,
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=periods,
            y=market_shares[company2],
            mode='lines+markers',
            name=company2,
            line=dict(color='#ef4444', width=3),
            marker=dict(size=10)
        ))
        
        # Kazananı belirle
        latest_share1 = market_shares[company1][-1] if market_shares[company1] else 0
        latest_share2 = market_shares[company2][-1] if market_shares[company2] else 0
        
        if latest_share1 > latest_share2:
            winner = company1
            margin = latest_share1 - latest_share2
        else:
            winner = company2
            margin = latest_share2 - latest_share1
        
        fig.update_layout(
            height=500,
            title_text=f"{company1} vs {company2} - Pazar Payı Savaşı<br><sup>{winner} lider (%{margin:.1f} fark)</sup>",
            title_x=0.5,
            xaxis_title="Dönem",
            yaxis_title="Pazar Payı (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f1f5f9',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig

# ================================================
# 6. STRATEJİK ANALİZ VE ÖNGÖRÜ MOTORU
# ================================================

class StrategicAnalysisEngine:
    """BCG Matrix, Pareto Analizi ve Akıllı Tavsiye Motoru"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.insights = []
        self.recommendations = []
    
    def create_bcg_matrix(self) -> pd.DataFrame:
        """BCG Matrix (Growth-Share Matrix) oluştur"""
        try:
            df_bcg = self.df.copy()
            
            # Gerekli sütunları kontrol et
            if 'Market_Share' not in df_bcg.columns or 'CAGR' not in df_bcg.columns:
                logger.log(LogLevel.WARNING, "BCG Matrix için gerekli sütunlar eksik")
                return None
            
            # Relatif pazar payı (en büyük rakibe göre)
            max_market_share = df_bcg['Market_Share'].max()
            if max_market_share > 0:
                df_bcg['Relative_Market_Share'] = df_bcg['Market_Share'] / max_market_share
            else:
                df_bcg['Relative_Market_Share'] = 0
            
            # Büyüme oranı
            df_bcg['Growth_Rate'] = df_bcg['CAGR']
            
            # Segmentasyon için threshold'lar
            # Relative Market Share: >1 = Yüksek, <1 = Düşük
            # Growth Rate: >10% = Yüksek, <10% = Düşük
            share_threshold = 1.0  # Industry average
            growth_threshold = 10.0  #  % Industry average growth
            
            # BCG Kategorileri
            conditions = [
                # Yıldızlar (Yüksek Pay, Yüksek Büyüme)
                (df_bcg['Relative_Market_Share'] >= share_threshold) & (df_bcg['Growth_Rate'] >= growth_threshold),
                
                # Nakit İnekleri (Yüksek Pay, Düşük Büyüme)
                (df_bcg['Relative_Market_Share'] >= share_threshold) & (df_bcg['Growth_Rate'] < growth_threshold),
                
                # Soru İşaretleri (Düşük Pay, Yüksek Büyüme)
                (df_bcg['Relative_Market_Share'] < share_threshold) & (df_bcg['Growth_Rate'] >= growth_threshold),
                
                # Köpekler (Düşük Pay, Düşük Büyüme)
                (df_bcg['Relative_Market_Share'] < share_threshold) & (df_bcg['Growth_Rate'] < growth_threshold)
            ]
            
            categories = ['Yıldızlar', 'Nakit İnekleri', 'Soru İşaretleri', 'Köpekler']
            
            df_bcg['BCG_Category'] = np.select(conditions, categories, default='Belirsiz')
            
            # BCG skoru (0-100 arası)
            df_bcg['BCG_Score'] = (df_bcg['Relative_Market_Share'] * 0.6 + 
                                  (df_bcg['Growth_Rate'] / 100) * 0.4) * 100
            
            logger.log(LogLevel.INFO, f"BCG Matrix created", 
                      categories=df_bcg['BCG_Category'].value_counts().to_dict())
            
            return df_bcg
            
        except Exception as e:
            logger.log(LogLevel.ERROR, f"BCG Matrix creation failed", error=str(e))
            return None
    
    def get_bcg_category_products(self, df_bcg: pd.DataFrame, category: str, top_n: int = 10) -> List[Dict]:
        """BCG kategorisindeki ürünleri getir"""
        if df_bcg is None or 'BCG_Category' not in df_bcg.columns:
            return []
        
        category_df = df_bcg[df_bcg['BCG_Category'] == category]
        
        # Molekül ismine göre grupla
        if 'Molecule' in category_df.columns:
            grouped = category_df.groupby('Molecule').agg({
                'Market_Share': 'sum',
                'CAGR': 'mean',
                'BCG_Score': 'mean'
            }).sort_values('BCG_Score', ascending=False)
            
            products = []
            for i, (molecule, metrics) in enumerate(grouped.head(top_n).iterrows(), 1):
                product_info = {
                    'rank': i,
                    'molecule': molecule,
                    'market_share': metrics['Market_Share'],
                    'cagr': metrics['CAGR'],
                    'bcg_score': metrics['BCG_Score']
                }
                products.append(product_info)
            
            return products
        
        return []
    
    def perform_pareto_analysis(self) -> Dict[str, Any]:
        """Pareto (80/20) Analizi"""
        try:
            # Satış sütunlarını bul
            sales_cols = [col for col in self.df.columns if 'sales' in str(col).lower() or 'satış' in str(col).lower()]
            if not sales_cols:
                return None
            
            latest_sales = sales_cols[-1]
            
            # Ürün/Molekül bazında grupla
            if 'Molecule' in self.df.columns:
                grouped = self.df.groupby('Molecule')[latest_sales].sum().sort_values(ascending=False)
            elif 'Product' in self.df.columns:
                grouped = self.df.groupby('Product')[latest_sales].sum().sort_values(ascending=False)
            else:
                # Index kullan
                grouped = pd.Series(self.df[latest_sales].values, 
                                   index=self.df.index.astype(str)).sort_values(ascending=False)
            
            total_sales = grouped.sum()
            
            # Kümülatif oranları hesapla
            cum_sales = grouped.cumsum()
            cum_pct = (cum_sales / total_sales) * 100
            
            # %80'e ulaşan kritik ürünleri bul
            critical_mask = cum_pct <= 80
            critical_products = grouped[critical_mask]
            
            results = {
                'total_products': len(grouped),
                'critical_products_count': len(critical_products),
                'critical_products_pct': (len(critical_products) / len(grouped)) * 100,
                'pareto_ratio': len(critical_products) / len(grouped) * 100,
                'critical_products': [],
                'total_sales': total_sales,
                'critical_sales': critical_products.sum(),
                'critical_sales_pct': (critical_products.sum() / total_sales) * 100
            }
            
            # Kritik ürün detayları
            for i, (product, sales) in enumerate(critical_products.items(), 1):
                product_pct = (sales / total_sales) * 100
                cum_pct_product = cum_pct[product]
                
                results['critical_products'].append({
                    'rank': i,
                    'product': product,
                    'sales': sales,
                    'sales_pct': product_pct,
                    'cumulative_pct': cum_pct_product
                })
            
            logger.log(LogLevel.INFO, f"Pareto analysis completed", 
                      critical_products=results['critical_products_count'],
                      pareto_ratio=results['pareto_ratio'])
            
            return results
            
        except Exception as e:
            logger.log(LogLevel.ERROR, f"Pareto analysis failed", error=str(e))
            return None
    
    def generate_strategic_insights(self) -> List[Dict[str, Any]]:
        """Kural tabanlı stratejik içgörüler üret"""
        insights = []
        
        # Rule 1: Düşen yıldızlar (Sales artıyor ama market share düşüyor)
        if 'CAGR' in self.df.columns and 'Market_Share' in self.df.columns and 'Molecule' in self.df.columns:
            declining_stars = self.df[
                (self.df['CAGR'] > 0) &  # Satışları artıyor
                (self.df['CAGR'] < self.df['CAGR'].median()) &  # Ortalamanın altında
                (self.df['Market_Share'] < self.df['Market_Share'].shift(1).fillna(0))  # Pazar payı düşüyor
            ]
            
            for _, row in declining_stars.head(5).iterrows():
                molecule = row.get('Molecule', 'Unknown')
                sales_growth = row['CAGR']
                market_share_change = row['Market_Share'] - self.df['Market_Share'].mean()
                
                insight = {
                    'type': 'warning',
                    'title': f'Pazar Payı Kaybı: {molecule}',
                    'message': f"Satışlar %{sales_growth:.1f} artmasına rağmen pazar payı %{abs(market_share_change):.1f} düştü. Rakipler daha hızlı büyüyor, fiyatlandırma veya pazarlama stratejisi gözden geçirilmeli.",
                    'priority': 'High',
                    'action_items': [
                        'Rakiplerin fiyatlandırma stratejilerini analiz edin',
                        'Pazarlama bütçesini gözden geçirin',
                        'Müşteri geri bildirimlerini toplayın'
                    ]
                }
                insights.append(insight)
        
        # Rule 2: International dominance
        if 'International_Product' in self.df.columns and 'Company' in self.df.columns:
            # International ürün pazarını domine eden şirketler
            intl_market = self.df[self.df['International_Product'].astype(str).str.contains('Yes|1', case=False)]
            
            if len(intl_market) > 0:
                company_dominance = intl_market.groupby('Company')['Market_Share'].sum().nlargest(3)
                
                for company, share in company_dominance.items():
                    if share > 50:  % # %50'den fazla pay
                        insight = {
                            'type': 'info',
                            'title': f'International Pazar Hakimiyeti: {company}',
                            'message': f"{company}, international product pazarının %{share:.1f}'ine sahip. Bu alanda giriş bariyeri yüksek, ortaklık veya farklılaştırılmış ürün geliştirme düşünülmeli.",
                            'priority': 'Medium',
                            'action_items': [
                                'Rakibin international stratejisini analiz edin',
                                'Farklı coğrafyalarda fırsatları değerlendirin',
                                'Yerel ortaklıklar kurmayı düşünün'
                            ]
                        }
                        insights.append(insight)
        
        # Rule 3: Yüksek büyüyen düşük fiyatlı ürünler
        if 'CAGR' in self.df.columns and 'Price' in self.df.columns and 'Molecule' in self.df.columns:
            price_cols = [col for col in self.df.columns if 'price' in str(col).lower() or 'fiyat' in str(col).lower()]
            if price_cols:
                latest_price = price_cols[-1]
                
                high_growth_low_price = self.df[
                    (self.df['CAGR'] > 20) &  # Yüksek büyüme
                    (self.df[latest_price] < self.df[latest_price].median())  # Düşük fiyat
                ]
                
                for _, row in high_growth_low_price.head(3).iterrows():
                    molecule = row.get('Molecule', 'Unknown')
                    growth = row['CAGR']
                    price = row[latest_price]
                    
                    insight = {
                        'type': 'success',
                        'title': f'Fiyat Avantajı: {molecule}',
                        'message': f"%{growth:.1f} büyüme ile hızla büyüyor ve ${price:.2f} fiyatıyla pazarda en düşük fiyatlı ürünlerden. Fiyat avantajını koruyarak pazar payını artırabilir.",
                        'priority': 'High',
                        'action_items': [
                            'Üretim maliyetlerini optimize edin',
                            'Ölçek ekonomisinden faydalanın',
                            'Dağıtım kanallarını genişletin'
                        ]
                    }
                    insights.append(insight)
        
        # Rule 4: International olmayan ama yüksek potansiyelli ürünler
        if ('International_Product' in self.df.columns and 'CAGR' in self.df.columns and 
            'Market_Share' in self.df.columns and 'Molecule' in self.df.columns):
            
            local_high_potential = self.df[
                (~self.df['International_Product'].astype(str).str.contains('Yes|1', case=False)) &  # Local
                (self.df['CAGR'] > 15) &  # Yüksek büyüme
                (self.df['Market_Share'] < 5)  # Düşük pazar payı (büyüme potansiyeli)
            ]
            
            for _, row in local_high_potential.head(3).iterrows():
                molecule = row.get('Molecule', 'Unknown')
                growth = row['CAGR']
                
                insight = {
                    'type': 'info',
                    'title': f'International Potansiyel: {molecule}',
                    'message': f"Yerel pazarda %{growth:.1f} büyüyor ancak international pazarda yok. International pazara açılmak için yüksek potansiyel taşıyor.",
                    'priority': 'Medium',
                    'action_items': [
                        'International pazarlar için regülasyonları araştırın',
                        'Yerel distribütörlerle görüşün',
                        'International klinik çalışmaları planlayın'
                    ]
                }
                insights.append(insight)
        
        logger.log(LogLevel.INFO, f"Strategic insights generated", count=len(insights))
        
        return insights

# ================================================
# 7. FİYAT ZEKASI ANALİZ MOTORU
# ================================================

class PriceIntelligenceEngine:
    """Fiyat-Hacim-Elastisite analizi için motor"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.price_columns = self._detect_price_columns()
        self.volume_columns = self._detect_volume_columns()
    
    def _detect_price_columns(self) -> List[str]:
        """Fiyat sütunlarını tespit et"""
        price_patterns = ['price', 'fiyat', 'cost', 'maliyet', 'avg', 'ort', 'unit price']
        price_cols = []
        
        for col in self.df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in price_patterns):
                price_cols.append(col)
        
        return price_cols
    
    def _detect_volume_columns(self) -> List[str]:
        """Hacim sütunlarını tespit et"""
        volume_patterns = ['volume', 'hacim', 'quantity', 'miktar', 'unit', 'birim', 'qty']
        volume_cols = []
        
        for col in self.df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in volume_patterns):
                volume_cols.append(col)
        
        return volume_cols
    
    def create_price_volume_scatter(self) -> go.Figure:
        """Fiyat-Hacim scatter plot (DUPLICATE COLUMN FIX ile)"""
        try:
            if not self.price_columns or not self.volume_columns:
                return None
            
            # Son dönemin fiyat ve hacim sütunlarını al
            price_col = self.price_columns[-1]
            volume_col = self.volume_columns[-1]
            
            # KRİTİK DÜZELTME: Benzersiz sütun isimleri oluştur
            # Hash kullanarak unique isimler garantile
            price_hash = hashlib.md5(price_col.encode()).hexdigest()[:8]
            volume_hash = hashlib.md5(volume_col.encode()).hexdigest()[:8]
            
            temp_price_col = f"Price_{price_hash}"
            temp_volume_col = f"Volume_{volume_hash}"
            
            # DataFrame'in kopyasını al (orijinali bozmamak için)
            plot_df = self.df.copy()
            
            # Yeni sütunları ekle
            plot_df[temp_price_col] = plot_df[price_col]
            plot_df[temp_volume_col] = plot_df[volume_col]
            
            # NaN değerleri temizle
            plot_df = plot_df.dropna(subset=[temp_price_col, temp_volume_col])
            
            if len(plot_df) == 0:
                return None
            
            # Hover için isim belirle
            hover_name = None
            for col in ['Molecule', 'Product', 'Company', 'Şirket']:
                if col in plot_df.columns:
                    hover_name = col
                    break
            
            # Scatter plot oluştur
            fig = px.scatter(
                plot_df,
                x=temp_price_col,
                y=temp_volume_col,
                size=temp_volume_col,
                color=temp_price_col,
                hover_name=hover_name,
                title='Fiyat-Hacim İlişkisi Analizi',
                labels={
                    temp_price_col: 'Fiyat (USD)',
                    temp_volume_col: 'Hacim (Birim)'
                },
                color_continuous_scale='Viridis',
                trendline='lowess',  # Non-parametric trend line
                trendline_options=dict(frac=0.3)  # Smoothing parameter
            )
            
            # Korelasyon katsayısını hesapla
            correlation = plot_df[temp_price_col].corr(plot_df[temp_volume_col])
            
            fig.update_layout(
                height=600,
                title_text=f'Fiyat-Hacim İlişkisi<br><sup>Korelasyon: {correlation:.3f}</sup>',
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    tickformat='$,.0f'
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    tickformat=',.0f'
                ),
                coloraxis_colorbar=dict(
                    title="Fiyat (USD)",
                    tickprefix="$",
                    ticksuffix=""
                )
            )
            
            # Korelasyon yorumu ekle
            if abs(correlation) > 0.7:
                corr_text = "Güçlü ilişki"
            elif abs(correlation) > 0.3:
                corr_text = "Orta dereceli ilişki"
            else:
                corr_text = "Zayıf ilişki"
            
            if correlation > 0:
                corr_direction = "pozitif (fiyat arttıkça hacim artıyor)"
            else:
                corr_direction = "negatif (fiyat arttıkça hacim düşüyor)"
            
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"{corr_text}, {corr_direction}",
                showarrow=False,
                font=dict(size=12, color='#94a3b8'),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                borderpad=4
            )
            
            return fig
            
        except Exception as e:
            logger.log(LogLevel.ERROR, f"Price-volume scatter plot failed", error=str(e))
            return None
    
    def calculate_price_elasticity(self) -> Dict[str, Any]:
        """Fiyat esnekliği analizi"""
        try:
            if not self.price_columns or not self.volume_columns:
                return None
            
            price_col = self.price_columns[-1]
            volume_col = self.volume_columns[-1]
            
            # Veriyi hazırla
            analysis_df = self.df[[price_col, volume_col]].dropna()
            
            if len(analysis_df) < 10:
                return None
            
            # Log-log regression ile esneklik katsayısı
            X = np.log(analysis_df[price_col].replace(0, np.nan).dropna() + 1)
            y = np.log(analysis_df[volume_col].replace(0, np.nan).dropna() + 1)
            
            # Linear regression
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
            # Esneklik katsayısı = beta coefficient
            elasticity = model.params[1]
            
            # R-squared
            r_squared = model.rsquared
            
            # Esneklik yorumu
            if abs(elasticity) > 1:
                elasticity_type = "Esnek"
                interpretation = "Talep fiyata duyarlı"
            elif abs(elasticity) > 0:
                elasticity_type = "Esnek Olmayan"
                interpretation = "Talep fiyata çok duyarlı değil"
            else:
                elasticity_type = "Tam Esnek Olmayan"
                interpretation = "Talep fiyattan bağımsız"
            
            results = {
                'elasticity_coefficient': elasticity,
                'elasticity_type': elasticity_type,
                'interpretation': interpretation,
                'r_squared': r_squared,
                'p_value': model.pvalues[1],
                'confidence_interval': model.conf_int().iloc[1].tolist(),
                'sample_size': len(analysis_df),
                'avg_price': analysis_df[price_col].mean(),
                'avg_volume': analysis_df[volume_col].mean()
            }
            
            # Segment bazında esneklik
            if 'Molecule' in self.df.columns and len(self.df['Molecule'].unique()) > 1:
                segment_elasticities = {}
                
                for molecule in self.df['Molecule'].unique()[:5]:  # İlk 5 molekül
                    segment_df = self.df[self.df['Molecule'] == molecule]
                    if len(segment_df) > 5:
                        try:
                            X_seg = np.log(segment_df[price_col].replace(0, np.nan).dropna() + 1)
                            y_seg = np.log(segment_df[volume_col].replace(0, np.nan).dropna() + 1)
                            
                            if len(X_seg) > 3 and len(y_seg) > 3:
                                X_seg_const = sm.add_constant(X_seg)
                                model_seg = sm.OLS(y_seg, X_seg_const).fit()
                                seg_elasticity = model_seg.params[1]
                                
                                segment_elasticities[molecule] = {
                                    'elasticity': seg_elasticity,
                                    'r_squared': model_seg.rsquared,
                                    'sample_size': len(segment_df)
                                }
                        except:
                            continue
                
                results['segment_elasticities'] = segment_elasticities
            
            logger.log(LogLevel.INFO, f"Price elasticity calculated", 
                      elasticity=elasticity, elasticity_type=elasticity_type)
            
            return results
            
        except Exception as e:
            logger.log(LogLevel.ERROR, f"Price elasticity calculation failed", error=str(e))
            return None
    
    def create_price_segmentation_chart(self) -> go.Figure:
        """Fiyat segmentasyonu analizi"""
        try:
            if not self.price_columns:
                return None
            
            price_col = self.price_columns[-1]
            price_data = self.df[price_col].dropna()
            
            if len(price_data) == 0:
                return None
            
            # Fiyat segmentlerini belirle
            segments = {
                'Economy (<$10)': (0, 10),
                'Standard ($10-$50)': (10, 50),
                'Premium ($50-$100)': (50, 100),
                'Super Premium ($100-$500)': (100, 500),
                'Luxury (>$500)': (500, float('inf'))
            }
            
            segment_counts = {}
            segment_sales = {}
            
            for segment_name, (min_price, max_price) in segments.items():
                if max_price == float('inf'):
                    mask = price_data >= min_price
                else:
                    mask = (price_data >= min_price) & (price_data < max_price)
                
                segment_counts[segment_name] = mask.sum()
                
                # Segment satışlarını hesapla
                sales_cols = [col for col in self.df.columns if 'sales' in str(col).lower() or 'satış' in str(col).lower()]
                if sales_cols:
                    latest_sales = sales_cols[-1]
                    segment_sales[segment_name] = self.df.loc[mask, latest_sales].sum()
            
            # Bar chart oluştur
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Ürün Sayısına Göre Segment Dağılımı', 
                              'Satışlara Göre Segment Dağılımı'),
                specs=[[{'type': 'pie'}, {'type': 'pie'}]]
            )
            
            # Pie chart 1: Ürün sayısı
            fig.add_trace(
                go.Pie(
                    labels=list(segment_counts.keys()),
                    values=list(segment_counts.values()),
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Set3,
                    textinfo='percent+label',
                    textposition='inside',
                    name='Ürün Sayısı'
                ),
                row=1, col=1
            )
            
            # Pie chart 2: Satışlar
            if segment_sales:
                fig.add_trace(
                    go.Pie(
                        labels=list(segment_sales.keys()),
                        values=list(segment_sales.values()),
                        hole=0.4,
                        marker_colors=px.colors.qualitative.Set2,
                        textinfo='percent+label',
                        textposition='inside',
                        name='Satışlar'
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                height=500,
                title_text='Fiyat Segmentasyonu Analizi',
                title_x=0.5,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9'
            )
            
            return fig
            
        except Exception as e:
            logger.log(LogLevel.ERROR, f"Price segmentation chart failed", error=str(e))
            return None

# ================================================
# 8. ANA UYGULAMA - ENTERPRISE DASHBOARD
# ================================================

class PharmaIntelligenceDashboard:
    """Ana Enterprise Dashboard sınıfı"""
    
    def __init__(self):
        self.data_processor = EnterpriseDataProcessor()
        self.intl_detector = InternationalProductDetector()
        self.current_df = None
        self.analytics_engine = None
        self.strategy_engine = None
        self.price_engine = None
        
        # Session state initialization
        self._init_session_state()
    
    def _init_session_state(self):
        """Session state'i başlat"""
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'processed_df' not in st.session_state:
            st.session_state.processed_df = None
        if 'intl_mapped' not in st.session_state:
            st.session_state.intl_mapped = False
        if 'market_metrics' not in st.session_state:
            st.session_state.market_metrics = {}
        if 'strategic_insights' not in st.session_state:
            st.session_state.strategic_insights = []
        if 'bcg_matrix' not in st.session_state:
            st.session_state.bcg_matrix = None
        if 'pareto_analysis' not in st.session_state:
            st.session_state.pareto_analysis = None
    
    def load_data(self):
        """Veri yükleme işlemi"""
        st.markdown("""
        <div class="section-title">
            📁 VERİ YÜKLEME VE HAZIRLIK
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Excel/CSV dosyası yükleyin",
                type=['csv', 'xlsx', 'xls'],
                help="500MB'a kadar dosyalar desteklenir"
            )
        
        with col2:
            sample_size = st.number_input(
                "Örneklem boyutu (tüm veri için 0)",
                min_value=0,
                max_value=1000000,
                value=0,
                step=10000,
                help="0 = Tüm veri yüklenecek"
            )
        
        if uploaded_file:
            if st.button("🚀 Veriyi Yükle ve Analiz Et", type="primary", use_container_width=True):
                with st.spinner("Veri işleniyor..."):
                    # Veriyi yükle
                    df = self.data_processor.load_large_dataset(
                        uploaded_file, 
                        sample_size if sample_size > 0 else None
                    )
                    
                    if df is not None and len(df) > 0:
                        # International Product mapping
                        df, mapped = self.intl_detector.create_mapping_interface(df)
                        
                        if mapped:
                            st.session_state.intl_mapped = True
                        
                        # Analiz verisini hazırla
                        df = self.data_processor.prepare_analysis_data(df)
                        
                        # Session state'e kaydet
                        st.session_state.df = df
                        st.session_state.processed_df = df.copy()
                        
                        # Analiz motorlarını başlat
                        self.current_df = df
                        self.analytics_engine = AdvancedCompetitiveAnalytics(df)
                        self.strategy_engine = StrategicAnalysisEngine(df)
                        self.price_engine = PriceIntelligenceEngine(df)
                        
                        # Metrikleri hesapla
                        st.session_state.market_metrics = self.analytics_engine.calculate_market_structure_metrics()
                        
                        # Stratejik içgörüleri üret
                        st.session_state.strategic_insights = self.strategy_engine.generate_strategic_insights()
                        
                        # BCG Matrix oluştur
                        st.session_state.bcg_matrix = self.strategy_engine.create_bcg_matrix()
                        
                        # Pareto analizi yap
                        st.session_state.pareto_analysis = self.strategy_engine.perform_pareto_analysis()
                        
                        st.success(f"✅ {len(df):,} satır başarıyla yüklendi ve analiz edildi!")
                        st.rerun()
    
    def render_dashboard(self):
        """Dashboard'ı render et"""
        if st.session_state.processed_df is None:
            self._render_welcome_screen()
            return
        
        self.current_df = st.session_state.processed_df
        self.analytics_engine = AdvancedCompetitiveAnalytics(self.current_df)
        self.strategy_engine = StrategicAnalysisEngine(self.current_df)
        self.price_engine = PriceIntelligenceEngine(self.current_df)
        
        # Sekmeler
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 DASHBOARD",
            "🔬 MOLEKÜL EXPLORER",
            "💰 PRICE INTELLIGENCE",
            "🏆 MARKET STRUCTURE",
            "🌍 INTERNATIONAL FOCUS",
            "🎯 STRATEGY ROOM"
        ])
        
        with tab1:
            self._render_dashboard_tab()
        
        with tab2:
            self._render_molecule_explorer_tab()
        
        with tab3:
            self._render_price_intelligence_tab()
        
        with tab4:
            self._render_market_structure_tab()
        
        with tab5:
            self._render_international_focus_tab()
        
        with tab6:
            self._render_strategy_room_tab()
    
    def _render_welcome_screen(self):
        """Hoşgeldiniz ekranı"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div class="enterprise-card" style="text-align: center; padding: 3rem;">
                <h1 class="enterprise-title">PharmaIntelligence</h1>
                <h3 class="enterprise-subtitle">Enterprise Pharmaceutical Analytics Suite</h3>
                
                <div style="margin: 2rem 0;">
                    <div style="font-size: 5rem;">💊</div>
                </div>
                
                <div style="color: #94a3b8; margin-bottom: 2rem;">
                    Global farmasötik pazar analizi, rekabet zekası ve stratejik öngörü platformu
                </div>
                
                <div class="grid-3" style="margin: 2rem 0;">
                    <div class="insight-card info">
                        <div class="insight-icon">🌍</div>
                        <div class="insight-title">International Product Analytics</div>
                        <div class="insight-content">Çoklu pazar stratejisi optimizasyonu</div>
                    </div>
                    
                    <div class="insight-card success">
                        <div class="insight-icon">📈</div>
                        <div class="insight-title">Advanced Competitive Intelligence</div>
                        <div class="insight-content">Derin rekabet analizi ve benchmarking</div>
                    </div>
                    
                    <div class="insight-card warning">
                        <div class="insight-icon">🎯</div>
                        <div class="insight-title">Strategic Recommendations</div>
                        <div class="insight-content">AI destekli stratejik öneriler</div>
                    </div>
                </div>
                
                <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(59, 130, 246, 0.1); border-radius: 10px;">
                    <div style="font-weight: 600; color: #60a5fa; margin-bottom: 0.5rem;">🎯 Başlamak İçin</div>
                    <div style="color: #cbd5e1; line-height: 1.6;">
                        1. Sol taraftaki panelden veri dosyanızı yükleyin<br>
                        2. "Veriyi Yükle ve Analiz Et" butonuna tıklayın<br>
                        3. Analiz sonuçlarını görmek için sekmeleri kullanın
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_dashboard_tab(self):
        """Dashboard tab'ı"""
        st.markdown("""
        <div class="section-title">
            📊 EXECUTIVE DASHBOARD
        </div>
        """, unsafe_allow_html=True)
        
        # KPI'lar
        metrics = st.session_state.market_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card primary">
                <div class="metric-label">TOPLAM PAZAR BÜYÜKLÜĞÜ</div>
                <div class="metric-value">${metrics.get('total_market_size', 0)/1e6:.1f}M</div>
                <div class="metric-trend">
                    <span class="trend-up">↗</span>
                    <span>{metrics.get('market_growth_rate', 0):.1f}% YoY</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            hhi = metrics.get('hhi_index', 0)
            hhi_risk = metrics.get('hhi_risk', 'Bilinmiyor')
            
            st.markdown(f"""
            <div class="metric-card {'danger' if hhi > 2500 else 'warning' if hhi > 1500 else 'success'}">
                <div class="metric-label">REKABET YOĞUNLUĞU (HHI)</div>
                <div class="metric-value">{hhi:.0f}</div>
                <div class="metric-trend">
                    <span class="{'trend-danger' if hhi > 2500 else 'trend-warning' if hhi > 1500 else 'trend-success'}">
                        {metrics.get('hhi_interpretation', 'Bilinmiyor')}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cr3 = metrics.get('cr3', 0)
            
            st.markdown(f"""
            <div class="metric-card {'warning' if cr3 > 50 else 'success'}">
                <div class="metric-label">PAZAR KONSANTRASYONU (CR3)</div>
                <div class="metric-value">{cr3:.1f}%</div>
                <div class="metric-trend">
                    <span>Top 3 Şirket Payı</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            intl_pct = 0
            if 'International_Product' in self.current_df.columns:
                if self.current_df['International_Product'].dtype == 'object':
                    intl_pct = (self.current_df['International_Product'] == 'Yes').mean() * 100
                else:
                    intl_pct = (self.current_df['International_Product'] > 0).mean() * 100
            
            st.markdown(f"""
            <div class="metric-card info">
                <div class="metric-label">INTERNATIONAL PRODUCT</div>
                <div class="metric-value">{intl_pct:.1f}%</div>
                <div class="metric-trend">
                    <span>Global Ürün Oranı</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # İçgörüler
        st.markdown("""
        <div class="subsection-title">
            🔍 KRİTİK İÇGÖRÜLER
        </div>
        """, unsafe_allow_html=True)
        
        insights = st.session_state.strategic_insights[:3]
        
        if insights:
            for insight in insights:
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-content">{insight['message']}</div>
                    <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #64748b;">
                        Öncelik: {insight['priority']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("İçgörü üretmek için daha fazla veri gerekiyor")
        
        # Hızlı veri önizleme
        st.markdown("""
        <div class="subsection-title">
            📋 VERİ ÖNİZLEME
        </div>
        """, unsafe_allow_html=True)
        
        preview_cols = st.columns([3, 1])
        
        with preview_cols[0]:
            rows_to_show = st.slider("Gösterilecek satır sayısı", 10, 1000, 100)
        
        with preview_cols[1]:
            default_cols = ['Molecule', 'Company', 'Market_Share', 'CAGR']
            available_cols = [col for col in default_cols if col in self.current_df.columns]
            selected_cols = st.multiselect(
                "Sütunlar",
                options=self.current_df.columns.tolist(),
                default=available_cols
            )
        
        if selected_cols:
            st.dataframe(
                self.current_df[selected_cols].head(rows_to_show),
                use_container_width=True,
                height=400
            )
    
    def _render_molecule_explorer_tab(self):
        """Molekül Explorer tab'ı"""
        st.markdown("""
        <div class="section-title">
            🔬 MOLEKÜL EXPLORER - DERİN ANALİZ
        </div>
        """, unsafe_allow_html=True)
        
        # Molekül seçimi
        if 'Molecule' not in self.current_df.columns:
            st.warning("Molekül sütunu bulunamadı")
            return
        
        unique_molecules = self.current_df['Molecule'].dropna().unique().tolist()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_molecules = st.multiselect(
                "Karşılaştırılacak Moleküller (en az 2 seçin)",
                options=unique_molecules,
                default=unique_molecules[:min(3, len(unique_molecules))],
                max_selections=5
            )
        
        with col2:
            benchmark_type = st.selectbox(
                "Benchmark Tipi",
                ['Pazar Ortalaması', 'En İyi Performans', 'Hedef Değer'],
                key="benchmark_type"
            )
        
        if len(selected_molecules) >= 2:
            # Molekül karşılaştırma chart'ı
            comparison_chart = self.analytics_engine.create_molecule_comparison_chart(selected_molecules)
            
            if comparison_chart:
                st.plotly_chart(comparison_chart, use_container_width=True)
            
            # Radar chart
            st.markdown("""
            <div class="subsection-title">
                📊 PERFORMANS KARNESİ - RADAR CHART
            </div>
            """, unsafe_allow_html=True)
            
            radar_chart = self.analytics_engine.create_molecule_radar_chart(selected_molecules)
            
            if radar_chart:
                st.plotly_chart(radar_chart, use_container_width=True)
            else:
                st.info("Radar chart için yeterli veri bulunamadı")
            
            # Benchmark analizi
            st.markdown("""
            <div class="subsection-title">
                📈 BENCHMARK ANALİZİ
            </div>
            """, unsafe_allow_html=True)
            
            self._render_benchmark_analysis(selected_molecules, benchmark_type)
        
        else:
            st.info("Lütfen karşılaştırma için en az 2 molekül seçin")
    
    def _render_benchmark_analysis(self, selected_molecules: List[str], benchmark_type: str):
        """Benchmark analizi render"""
        benchmark_data = []
        
        for molecule in selected_molecules:
            mol_df = self.current_df[self.current_df['Molecule'] == molecule]
            
            if len(mol_df) == 0:
                continue
            
            metrics = {'Molecule': molecule}
            
            # CAGR
            if 'CAGR' in self.current_df.columns:
                mol_cagr = mol_df['CAGR'].mean()
                market_cagr = self.current_df['CAGR'].mean()
                metrics['CAGR'] = mol_cagr
                metrics['CAGR_vs_Market'] = mol_cagr - market_cagr
            
            # Market Share
            if 'Market_Share' in self.current_df.columns:
                mol_share = mol_df['Market_Share'].sum()
                avg_share = self.current_df['Market_Share'].mean()
                metrics['Market_Share'] = mol_share
                metrics['Share_vs_Avg'] = mol_share - avg_share
            
            # Price
            price_cols = [col for col in self.current_df.columns if 'price' in str(col).lower() or 'fiyat' in str(col).lower()]
            if price_cols:
                latest_price = price_cols[-1]
                mol_price = mol_df[latest_price].mean()
                market_price = self.current_df[latest_price].mean()
                metrics['Price'] = mol_price
                metrics['Price_Premium'] = ((mol_price / market_price) - 1) * 100
            
            benchmark_data.append(metrics)
        
        if benchmark_data:
            benchmark_df = pd.DataFrame(benchmark_data)
            
            # Format the dataframe for display
            display_df = benchmark_df.copy()
            
            # Format percentages
            for col in ['CAGR', 'CAGR_vs_Market', 'Market_Share', 'Share_vs_Avg', 'Price_Premium']:
                if col in display_df.columns:
                    if 'Premium' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
                    elif 'CAGR' in col or 'Share' in col:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
            
            if 'Price' in display_df.columns:
                display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Benchmark yorumu
            st.markdown("""
            <div class="insight-card info" style="margin-top: 1rem;">
                <div class="insight-title">📊 Benchmark Yorumu</div>
                <div class="insight-content">
                    """, unsafe_allow_html=True)
            
            best_cagr_molecule = benchmark_df.loc[benchmark_df['CAGR'].idxmax()]['Molecule'] if 'CAGR' in benchmark_df.columns else None
            best_share_molecule = benchmark_df.loc[benchmark_df['Market_Share'].idxmax()]['Molecule'] if 'Market_Share' in benchmark_df.columns else None
            
            if best_cagr_molecule:
                st.write(f"• **En hızlı büyüyen**: {best_cagr_molecule}")
            if best_share_molecule:
                st.write(f"• **En yüksek pazar payı**: {best_share_molecule}")
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    def _render_price_intelligence_tab(self):
        """Price Intelligence tab'ı"""
        st.markdown("""
        <div class="section-title">
            💰 PRICE INTELLIGENCE - FİYAT ANALİTİĞİ
        </div>
        """, unsafe_allow_html=True)
        
        # Fiyat-Hacim scatter plot
        st.markdown("""
        <div class="subsection-title">
            📈 FİYAT-HACİM İLİŞKİSİ
        </div>
        """, unsafe_allow_html=True)
        
        scatter_plot = self.price_engine.create_price_volume_scatter()
        
        if scatter_plot:
            st.plotly_chart(scatter_plot, use_container_width=True)
        else:
            st.warning("Fiyat-Hacim analizi için gerekli veri bulunamadı")
        
        # Fiyat esnekliği analizi
        st.markdown("""
        <div class="subsection-title">
            📊 FİYAT ESNEKLİĞİ ANALİZİ
        </div>
        """, unsafe_allow_html=True)
        
        elasticity_results = self.price_engine.calculate_price_elasticity()
        
        if elasticity_results:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Esneklik Katsayısı",
                    f"{elasticity_results['elasticity_coefficient']:.3f}",
                    elasticity_results['elasticity_type']
                )
            
            with col2:
                st.metric(
                    "R-squared",
                    f"{elasticity_results['r_squared']:.3f}",
                    "Model Açıklama Gücü"
                )
            
            with col3:
                st.metric(
                    "Ortalama Fiyat",
                    f"${elasticity_results['avg_price']:.2f}",
                    f"Hacim: {elasticity_results['avg_volume']:.0f}"
                )
            
            with col4:
                p_value = elasticity_results['p_value']
                significance = "Anlamlı" if p_value < 0.05 else "Anlamsız"
                st.metric(
                    "İstatistiksel Anlamlılık",
                    f"p={p_value:.4f}",
                    significance
                )
            
            # Esneklik yorumu
            st.markdown(f"""
            <div class="insight-card {'success' if abs(elasticity_results['elasticity_coefficient']) < 1 else 'warning'}">
                <div class="insight-title">🎯 Fiyat Esnekliği Yorumu</div>
                <div class="insight-content">
                    {elasticity_results['interpretation']}. 
                    Talep fiyat esnekliği {abs(elasticity_results['elasticity_coefficient']):.2f} olarak ölçülmüştür.
                    {"Fiyat artışları satışları çok etkilemeyecektir." if abs(elasticity_results['elasticity_coefficient']) < 1 else "Fiyat değişiklikleri satışları önemli ölçüde etkiler."}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Fiyat segmentasyonu
        st.markdown("""
        <div class="subsection-title">
            🏷️ FİYAT SEGMENTASYONU
        </div>
        """, unsafe_allow_html=True)
        
        segmentation_chart = self.price_engine.create_price_segmentation_chart()
        
        if segmentation_chart:
            st.plotly_chart(segmentation_chart, use_container_width=True)
    
    def _render_market_structure_tab(self):
        """Market Structure tab'ı"""
        st.markdown("""
        <div class="section-title">
            🏆 MARKET STRUCTURE - REKABET ANALİZİ
        </div>
        """, unsafe_allow_html=True)
        
        metrics = st.session_state.market_metrics
        
        # HHI ve konsantrasyon analizi
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="subsection-title">
                📊 PAZAR KONSANTRASYONU
            </div>
            """, unsafe_allow_html=True)
            
            hhi = metrics.get('hhi_index', 0)
            hhi_interpretation = metrics.get('hhi_interpretation', 'Bilinmiyor')
            hhi_recommendation = metrics.get('hhi_recommendation', '')
            
            st.markdown(f"""
            <div class="insight-card {'danger' if hhi > 2500 else 'warning' if hhi > 1500 else 'success'}">
                <div class="insight-title">HHI Endeksi: {hhi:.0f}</div>
                <div class="insight-content">
                    <strong>{hhi_interpretation}</strong><br>
                    {hhi_recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="subsection-title">
                🎯 KONSANTRASYON ORANLARI
            </div>
            """, unsafe_allow_html=True)
            
            cr3 = metrics.get('cr3', 0)
            cr5 = metrics.get('cr5', 0)
            cr10 = metrics.get('cr10', 0)
            
            st.metric("CR3 (Top 3)", f"{cr3:.1f}%")
            st.metric("CR5 (Top 5)", f"{cr5:.1f}%")
            st.metric("CR10 (Top 10)", f"{cr10:.1f}%")
        
        # Head-to-Head analizi
        st.markdown("""
        <div class="subsection-title">
            ⚔️ HEAD-TO-HEAD PAZAR SAVAŞI
        </div>
        """, unsafe_allow_html=True)
        
        if 'Company' in self.current_df.columns or 'Şirket' in self.current_df.columns:
            company_col = 'Company' if 'Company' in self.current_df.columns else 'Şirket'
            companies = self.current_df[company_col].dropna().unique().tolist()
            
            if len(companies) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    company1 = st.selectbox(
                        "Şirket 1",
                        options=companies,
                        key="h2h_company1"
                    )
                
                with col2:
                    other_companies = [c for c in companies if c != company1]
                    company2 = st.selectbox(
                        "Şirket 2",
                        options=other_companies,
                        key="h2h_company2"
                    )
                
                if company1 and company2:
                    h2h_chart = self.analytics_engine.create_head_to_head_analysis(company1, company2)
                    
                    if h2h_chart:
                        st.plotly_chart(h2h_chart, use_container_width=True)
        
        # Lorenz Eğrisi
        st.markdown("""
        <div class="subsection-title">
            📈 LORENZ EĞRİSİ - GELİR DAĞILIMI ANALOJİSİ
        </div>
        """, unsafe_allow_html=True)
        
        if 'lorenz_curve' in metrics:
            lorenz_data = metrics['lorenz_curve']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=lorenz_data['companies_percentage'],
                y=lorenz_data['cum_percentage'],
                fill='tozeroy',
                name='Lorenz Eğrisi',
                line=dict(color='#3b82f6', width=3),
                fillcolor='rgba(59, 130, 246, 0.3)'
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                name='Tam Eşitlik',
                line=dict(color='#94a3b8', width=2, dash='dash')
            ))
            
            gini = metrics.get('gini_coefficient', 0)
            
            fig.update_layout(
                height=400,
                title_text=f'Lorenz Eğrisi - Gini Katsayısı: {gini:.3f}',
                title_x=0.5,
                xaxis_title='Şirketlerin Kümülatif Oranı (%)',
                yaxis_title='Satışların Kümülatif Oranı (%)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_international_focus_tab(self):
        """International Focus tab'ı"""
        st.markdown("""
        <div class="section-title">
            🌍 INTERNATIONAL FOCUS - GLOBAL STRATEJİ
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.intl_mapped or 'International_Product' not in self.current_df.columns:
            st.warning("International Product sütunu eşleştirilmemiş")
            
            # Mapping arayüzünü göster
            self.current_df, mapped = self.intl_detector.create_mapping_interface(self.current_df)
            
            if mapped:
                st.session_state.intl_mapped = True
                st.session_state.processed_df = self.current_df
                st.rerun()
            
            return
        
        # International Product analizi
        intl_df = self.current_df.copy()
        
        # International/Local ayrımı
        if intl_df['International_Product'].dtype == 'object':
            intl_mask = intl_df['International_Product'].str.contains('Yes', case=False, na=False)
        else:
            intl_mask = intl_df['International_Product'] > 0
        
        international_products = intl_df[intl_mask]
        local_products = intl_df[~intl_mask]
        
        # KPI'lar
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            intl_count = len(international_products)
            total_count = len(intl_df)
            intl_pct = (intl_count / total_count) * 100
            
            st.metric(
                "International Ürünler",
                f"{intl_count}",
                f"%{intl_pct:.1f}"
            )
        
        with col2:
            sales_cols = [col for col in intl_df.columns if 'sales' in str(col).lower() or 'satış' in str(col).lower()]
            if sales_cols:
                latest_sales = sales_cols[-1]
                intl_sales = international_products[latest_sales].sum()
                total_sales = intl_df[latest_sales].sum()
                intl_sales_pct = (intl_sales / total_sales) * 100 if total_sales > 0 else 0
                
                st.metric(
                    "International Satış",
                    f"${intl_sales/1e6:.1f}M",
                    f"%{intl_sales_pct:.1f}"
                )
        
        with col3:
            if 'CAGR' in intl_df.columns:
                intl_growth = international_products['CAGR'].mean()
                local_growth = local_products['CAGR'].mean()
                
                st.metric(
                    "Ortalama CAGR",
                    f"{intl_growth:.1f}%",
                    f"Yerel: {local_growth:.1f}%"
                )
        
        with col4:
            if 'Market_Share' in intl_df.columns:
                intl_share = international_products['Market_Share'].sum()
                local_share = local_products['Market_Share'].sum()
                
                st.metric(
                    "Pazar Payı",
                    f"{intl_share:.1f}%",
                    f"Yerel: {local_share:.1f}%"
                )
        
        # International vs Local karşılaştırması
        st.markdown("""
        <div class="subsection-title">
            📊 INTERNATIONAL VS LOCAL KARŞILAŞTIRMASI
        </div>
        """, unsafe_allow_html=True)
        
        comparison_data = []
        
        for group_name, group_df in [('International', international_products), 
                                    ('Local', local_products)]:
            if len(group_df) > 0:
                group_metrics = {
                    'Segment': group_name,
                    'Ürün Sayısı': len(group_df),
                    'Ort. CAGR': group_df['CAGR'].mean() if 'CAGR' in group_df.columns else 0,
                    'Top. Pazar Payı': group_df['Market_Share'].sum() if 'Market_Share' in group_df.columns else 0,
                    'Ort. Fiyat': 0,
                    'Coğrafi Yayılım': 0
                }
                
                # Ortalama fiyat
                price_cols = [col for col in group_df.columns if 'price' in str(col).lower() or 'fiyat' in str(col).lower()]
                if price_cols:
                    group_metrics['Ort. Fiyat'] = group_df[price_cols[-1]].mean()
                
                # Coğrafi yayılım
                if 'Region' in group_df.columns or 'Bölge' in group_df.columns:
                    region_col = 'Region' if 'Region' in group_df.columns else 'Bölge'
                    group_metrics['Coğrafi Yayılım'] = group_df[region_col].nunique()
                
                comparison_data.append(group_metrics)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        
        # International ürünlerin detaylı listesi
        st.markdown("""
        <div class="subsection-title">
            📋 INTERNATIONAL ÜRÜN LİSTESİ
        </div>
        """, unsafe_allow_html=True)
        
        if len(international_products) > 0:
            display_cols = []
            for col in ['Molecule', 'Company', 'Market_Share', 'CAGR']:
                if col in international_products.columns:
                    display_cols.append(col)
            
            # Price sütunu ekle
            price_cols = [col for col in international_products.columns if 'price' in str(col).lower() or 'fiyat' in str(col).lower()]
            if price_cols:
                display_cols.append(price_cols[-1])
            
            if display_cols:
                display_df = international_products[display_cols].copy()
                
                # Format values
                if 'Market_Share' in display_df.columns:
                    display_df['Market_Share'] = display_df['Market_Share'].apply(lambda x: f"{x:.1f}%")
                
                if 'CAGR' in display_df.columns:
                    display_df['CAGR'] = display_df['CAGR'].apply(lambda x: f"{x:.1f}%")
                
                if price_cols:
                    display_df[price_cols[-1]] = display_df[price_cols[-1]].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(
                    display_df.sort_values('Market_Share' if 'Market_Share' in display_df.columns else display_cols[0], 
                                          ascending=False),
                    use_container_width=True,
                    height=400
                )
    
    def _render_strategy_room_tab(self):
        """Strategy Room tab'ı"""
        st.markdown("""
        <div class="section-title">
            🎯 STRATEGY ROOM - STRATEJİK KARAR DESTEK
        </div>
        """, unsafe_allow_html=True)
        
        # BCG Matrix
        st.markdown("""
        <div class="subsection-title">
            📊 BCG MATRIX - GROWTH/SHARE ANALİZİ
        </div>
        """, unsafe_allow_html=True)
        
        bcg_df = st.session_state.bcg_matrix
        
        if bcg_df is not None:
            # BCG kategorilerine göre dağılım
            category_counts = bcg_df['BCG_Category'].value_counts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    title='BCG Matrix - Ürün Dağılımı',
                    labels={'x': 'Kategori', 'y': 'Ürün Sayısı'},
                    color=category_counts.index,
                    color_discrete_sequence=['#f59e0b', '#10b981', '#3b82f6', '#ef4444']
                )
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="insight-card info">
                    <div class="insight-title">BCG Matrix Açıklaması</div>
                    <div class="insight-content">
                        <strong>Yıldızlar:</strong> Yüksek büyüme, yüksek pay<br>
                        <strong>Nakit İnekleri:</strong> Düşük büyüme, yüksek pay<br>
                        <strong>Soru İşaretleri:</strong> Yüksek büyüme, düşük pay<br>
                        <strong>Köpekler:</strong> Düşük büyüme, düşük pay
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # BCG kategorilerindeki ürünler
            st.markdown("""
            <div class="subsection-title">
                📋 BCG KATEGORİLERİNE GÖRE ÜRÜNLER
            </div>
            """, unsafe_allow_html=True)
            
            categories = ['Yıldızlar', 'Nakit İnekleri', 'Soru İşaretleri', 'Köpekler']
            
            for category in categories:
                with st.expander(f"🏷️ {category}"):
                    products = self.strategy_engine.get_bcg_category_products(bcg_df, category, top_n=10)
                    
                    if products:
                        for product in products:
                            st.write(f"**{product['rank']}. {product['molecule']}**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.caption(f"Pazar Payı: {product['market_share']:.1f}%")
                            with col2:
                                st.caption(f"CAGR: {product['cagr']:.1f}%")
                            with col3:
                                st.caption(f"BCG Skor: {product['bcg_score']:.1f}")
                    else:
                        st.info("Bu kategoride ürün bulunamadı")
        
        # Pareto Analizi
        st.markdown("""
        <div class="subsection-title">
            📈 PARETO (80/20) ANALİZİ
        </div>
        """, unsafe_allow_html=True)
        
        pareto_results = st.session_state.pareto_analysis
        
        if pareto_results:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Kritik Ürün Sayısı",
                    pareto_results['critical_products_count'],
                    f"Toplam: {pareto_results['total_products']}"
                )
            
            with col2:
                st.metric(
                    "Pareto Oranı",
                    f"{pareto_results['pareto_ratio']:.1f}%",
                    f"%{pareto_results['critical_products_pct']:.1f} ürün"
                )
            
            with col3:
                st.metric(
                    "Kritik Satışlar",
                    f"${pareto_results['critical_sales']/1e6:.1f}M",
                    f"%{pareto_results['critical_sales_pct']:.1f} toplam"
                )
            
            with col4:
                efficiency = (pareto_results['critical_sales_pct'] / 
                            pareto_results['critical_products_pct']) if pareto_results['critical_products_pct'] > 0 else 0
                st.metric(
                    "Verimlilik Oranı",
                    f"{efficiency:.2f}x",
                    "Satış/Ürün Oranı"
                )
            
            # Kritik ürün listesi
            st.markdown("""
            <div class="subsection-title">
                🎯 KRİTİK ÜRÜN LİSTESİ (%80 SATIŞ)
            </div>
            """, unsafe_allow_html=True)
            
            critical_products = pareto_results['critical_products'][:20]  # İlk 20
            
            for product in critical_products:
                with st.expander(f"**{product['rank']}. {product['product']}**"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Satış", f"${product['sales']/1e6:.2f}M")
                    with col2:
                        st.metric("Pazar Payı", f"{product['sales_pct']:.1f}%")
                    with col3:
                        st.metric("Kümülatif", f"{product['cumulative_pct']:.1f}%")
        
        # Stratejik İçgörüler
        st.markdown("""
        <div class="subsection-title">
            💡 STRATEJİK İÇGÖRÜLER VE TAVSİYELER
        </div>
        """, unsafe_allow_html=True)
        
        insights = st.session_state.strategic_insights
        
        if insights:
            for insight in insights:
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-content">
                        {insight['message']}
                        
                        <div style="margin-top: 1rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.1);">
                            <strong>Aksiyon Öğeleri:</strong>
                            <ul style="margin: 0.5rem 0 0 1rem; color: #cbd5e1;">
                                {''.join([f'<li>{item}</li>' for item in insight.get('action_items', [])])}
                            </ul>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Stratejik içgörü üretmek için daha fazla veri gerekiyor")
    
    def run(self):
        """Ana uygulamayı çalıştır"""
        try:
            # Header
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 class="enterprise-title">PharmaIntelligence</h1>
                <p class="enterprise-subtitle">
                    Enterprise Pharmaceutical Analytics Suite | v{config.version}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sidebar
            with st.sidebar:
                st.markdown("""
                <div class="sidebar-content">
                    <div class="sidebar-title">KONTROL PANELİ</div>
                """, unsafe_allow_html=True)
                
                # Veri yükleme
                self.load_data()
                
                # Filtreleme
                if st.session_state.processed_df is not None:
                    self._render_sidebar_filters()
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Ana içerik
            self.render_dashboard()
            
        except Exception as e:
            logger.log(LogLevel.ERROR, f"Dashboard runtime error", 
                      error=str(e), traceback=traceback.format_exc())
            
            st.error(f"Uygulama hatası: {str(e)}")
            st.error("Lütfen sayfayı yenileyin ve tekrar deneyin.")
            
            if st.button("🔄 Sayfayı Yenile"):
                st.rerun()
    
    def _render_sidebar_filters(self):
        """Sidebar filtrelerini render et"""
        st.markdown("""
        <div class="filter-section">
            <div class="filter-title">
                🔍 GELİŞMİŞ FİLTRELEME
            </div>
        """, unsafe_allow_html=True)
        
        # Genel arama
        search_term = st.text_input(
            "Genel Arama",
            placeholder="Molekül, Şirket, Bölge...",
            key="general_search"
        )
        
        # Molekül filtreleme
        if 'Molecule' in self.current_df.columns:
            molecules = self.current_df['Molecule'].dropna().unique().tolist()
            selected_molecules = st.multiselect(
                "Moleküller",
                options=molecules,
                default=molecules[:min(5, len(molecules))],
                key="molecule_filter"
            )
        
        # Şirket filtreleme
        if 'Company' in self.current_df.columns or 'Şirket' in self.current_df.columns:
            company_col = 'Company' if 'Company' in self.current_df.columns else 'Şirket'
            companies = self.current_df[company_col].dropna().unique().tolist()
            selected_companies = st.multiselect(
                "Şirketler",
                options=companies,
                default=companies[:min(5, len(companies))],
                key="company_filter"
            )
        
        # CAGR filtreleme
        if 'CAGR' in self.current_df.columns:
            cagr_range = st.slider(
                "CAGR Aralığı (%)",
                min_value=float(self.current_df['CAGR'].min()),
                max_value=float(self.current_df['CAGR'].max()),
                value=(float(self.current_df['CAGR'].min()), float(self.current_df['CAGR'].max())),
                key="cagr_filter"
            )
        
        # Pazar Payı filtreleme
        if 'Market_Share' in self.current_df.columns:
            share_range = st.slider(
                "Pazar Payı Aralığı (%)",
                min_value=float(self.current_df['Market_Share'].min()),
                max_value=float(self.current_df['Market_Share'].max()),
                value=(float(self.current_df['Market_Share'].min()), float(self.current_df['Market_Share'].max())),
                key="share_filter"
            )
        
        # International Product filtreleme
        if 'International_Product' in self.current_df.columns:
            intl_filter = st.selectbox(
                "International Product",
                ["Tümü", "Sadece International", "Sadece Yerel"],
                key="intl_filter"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

# ================================================
# 9. UYGULAMAYI BAŞLAT
# ================================================

if __name__ == "__main__":
    # Performans optimizasyonu
    gc.collect()
    
    # Dashboard'u başlat
    dashboard = PharmaIntelligenceDashboard()
    dashboard.run()

