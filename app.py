"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                         PHARMAINTELLIGENCE PRO - KURUMSAL EDİSYON v7.0                    ║
║                                                                                          ║
║              • Derin Öğrenme Tabanlı Tahminleme & Anomali Tespiti                        ║
║              • Gelişmiş NLP ile Pazar İstihbaratı & Duygu Analizi                        ║
║              • Otomatik Makine Öğrenimi (AutoML) Pipeline'ları                           ║
║              • Çok Boyutlu Kümeleme & Dinamik Segmentasyon                               ║
║              • Gerçek Zamanlı Gösterge Panelleri & Senaryo Simülasyonları                ║
║              • Blockchain Tabanlı Veri Doğrulama & İzlenebilirlik                        ║
║                                                                                          ║
║                         © 2024 PharmaIntelligence Inc. Tüm Hakları Saklıdır              ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# 1. İLERİ DÜZEY KÜTÜPHANELER & BAĞIMLILIKLAR
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_dendrogram
import warnings
warnings.filterwarnings('ignore')

# === Derin Öğrenme & Gelişmiş Makine Öğrenimi ===
# ================================================
# TENSORFLOW / DERİN ÖĞRENME İMKANSIZLAŞTIRILDI
# ================================================
# TensorFlow, PyTorch ve diğer ağır kütüphaneler
# Streamlit Cloud'da çalışmaz. Gerekli import'lar
# yorum satırına alınmıştır.
# ================================================

# import tensorflow as tf  # DEVRE DIŞI
# from tensorflow import keras  # DEVRE DIŞI
# from tensorflow.keras import layers, models, callbacks  # DEVRE DIŞI

# === Gelişmiş Sklearn Modülleri ===
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, 
    QuantileTransformer, LabelEncoder, OneHotEncoder
)
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, OPTICS, 
    Birch, SpectralClustering, MeanShift
)
from sklearn.ensemble import (
    IsolationForest, RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor, VotingRegressor, StackingRegressor
)
from sklearn.decomposition import PCA, KernelPCA, NMF, LatentDirichletAllocation
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score,
    confusion_matrix, classification_report, roc_auc_score, f1_score
)
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit,
    train_test_split, StratifiedKFold
)
from sklearn.feature_selection import (
    SelectKBest, mutual_info_regression, RFE, RFECV,
    SelectFromModel, VarianceThreshold
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Lasso, Ridge, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# === Zaman Serisi & Ekonometri ===
from statsmodels.tsa.api import (
    ExponentialSmoothing, SARIMAX, VAR, VECM,
    ARIMA, Holt, SimpleExpSmoothing
)
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor

# === İş Zekası & Optimizasyon ===
from scipy.optimize import minimize, linear_sum_assignment, differential_evolution
from scipy.stats import (
    norm, ttest_ind, f_oneway, chi2_contingency,
    kruskal, mannwhitneyu, spearmanr, pearsonr
)
from scipy.signal import savgol_filter, find_peaks
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# === Görselleştirme & Raporlama ===
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly.colors import n_colors, qualitative
import kaleido

# === Veri Yönetimi & Optimizasyon ===
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import vaex
import pyarrow as pa
import pyarrow.parquet as pq
from functools import lru_cache, wraps
import hashlib
import pickle
import joblib
import gc
import psutil
import platform

# === Web Servisleri & API Entegrasyonu ===
import requests
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import json
import xml.etree.ElementTree as ET
import yaml

# === Güvenlik & Şifreleme ===
import hashlib
import hmac
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

# === Raporlama & Dışa Aktarım ===
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    Image, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart

# === Sistem & Performans ===
import time
import tracemalloc
import linecache
import os
import sys
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import traceback

# ============================================================================
# 2. KURUMSAL YAPILANDIRMA & AYARLAR
# ============================================================================

@dataclass
class PharmaConfig:
    """Kurumsal yapılandırma sınıfı - Tüm sistem ayarları merkezi"""
    
    # === Uygulama Ayarları ===
    APP_NAME: str = "PharmaIntelligence Pro - Kurumsal Edisyon v7.0"
    APP_VERSION: str = "7.0.0"
    APP_BUILD: str = f"2024.12.{datetime.now().day}"
    APP_SECRET: str = secrets.token_urlsafe(32)
    
    # === Performans Ayarları ===
    MAX_MEMORY_MB: int = 16384  # 16GB
    CACHE_TTL: int = 7200       # 2 saat
    BATCH_SIZE: int = 50000
    PARALLEL_WORKERS: int = psutil.cpu_count(logical=False)
    
    # === Makine Öğrenimi Ayarları ===
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    N_JOBS: int = -1
    VERBOSE: int = 1
    
    # === Derin Öğrenme Ayarları ===
    EPOCHS: int = 100
    BATCH_SIZE_DL: int = 32
    LEARNING_RATE: float = 0.001
    EARLY_STOPPING_PATIENCE: int = 10
    REDUCE_LR_PATIENCE: int = 5
    DROPOUT_RATE: float = 0.2
    
    # === Görselleştirme Ayarları ===
    PLOT_HEIGHT: int = 600
    PLOT_WIDTH: int = 1200
    ANIMATION_DURATION: int = 1000
    COLOR_PALETTE: str = "Viridis"
    
    # === Güvenlik Ayarları ===
    ENCRYPTION_KEY: str = Fernet.generate_key().decode()
    TOKEN_EXPIRY: int = 3600
    MAX_LOGIN_ATTEMPTS: int = 5
    
    # === Raporlama Ayarları ===
    REPORT_FOOTER: str = "© 2024 PharmaIntelligence Inc. Gizli ve Kurumsaldır."
    COMPANY_LOGO: str = "https://pharmaintelligence.com/logo.png"
    SUPPORT_EMAIL: str = "enterprise@pharmaintelligence.com"
    
    # === Veri Doğrulama Ayarları ===
    REQUIRED_COLUMNS: List[str] = field(default_factory=lambda: [
        'Uluslararasi_Urun', 'Molekul', 'Sirket', 'Ulke'
    ])
    
    NUMERIC_COLUMNS: List[str] = field(default_factory=lambda: [
        'Satış_2022', 'Satış_2023', 'Satış_2024',
        'Birim_2022', 'Birim_2023', 'Birim_2024',
        'Ort_Fiyat_2022', 'Ort_Fiyat_2023', 'Ort_Fiyat_2024',
        'Standart_Birim_2022', 'Standart_Birim_2023', 'Standart_Birim_2024'
    ])

config = PharmaConfig()

# ============================================================================
# 3. PROFESYONEL TEMA SİSTEMİ
# ============================================================================

PROFESYONEL_CSS = f"""
<style>
    /* === KÖK DEĞİŞKENLER - KURUMSAL TEMA === */
    :root {{
        --pharma-primary: #0a1929;
        --pharma-secondary: #1e3a5f;
        --pharma-accent: #2d7dd2;
        --pharma-accent-light: #4a9fe3;
        --pharma-accent-dark: #1a5fa0;
        --pharma-gradient: linear-gradient(135deg, #2d7dd2, #2acaea);
        --pharma-gradient-reverse: linear-gradient(135deg, #2acaea, #2d7dd2);
        
        --pharma-success: #2dd2a3;
        --pharma-warning: #f2c94c;
        --pharma-danger: #eb5757;
        --pharma-info: #2d7dd2;
        --pharma-neutral: #64748b;
        
        --pharma-text-primary: #f8fafc;
        --pharma-text-secondary: #cbd5e1;
        --pharma-text-tertiary: #64748b;
        
        --pharma-bg-primary: #0a1929;
        --pharma-bg-secondary: #1e3a5f;
        --pharma-bg-card: rgba(30, 58, 95, 0.8);
        --pharma-bg-hover: rgba(45, 125, 210, 0.1);
        
        --pharma-border: rgba(255, 255, 255, 0.1);
        --pharma-border-hover: rgba(45, 125, 210, 0.3);
        
        --pharma-shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
        --pharma-shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
        --pharma-shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
        --pharma-shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.6);
        
        --pharma-radius-sm: 8px;
        --pharma-radius-md: 12px;
        --pharma-radius-lg: 16px;
        --pharma-radius-xl: 20px;
        --pharma-radius-full: 9999px;
        
        --pharma-transition-fast: 200ms cubic-bezier(0.4, 0, 0.2, 1);
        --pharma-transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
        --pharma-transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    /* === GENEL STİLLER === */
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    .stApp {{
        background: radial-gradient(circle at 0% 0%, var(--pharma-primary), #0f1a2b);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--pharma-text-primary);
    }}
    
    /* === GLASMORFİZM KARTLARI === */
    .pharma-card {{
        background: var(--pharma-bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--pharma-border);
        border-radius: var(--pharma-radius-lg);
        padding: 1.5rem;
        transition: all var(--pharma-transition-normal);
        position: relative;
        overflow: hidden;
    }}
    
    .pharma-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--pharma-gradient);
        transform: scaleX(0);
        transition: transform var(--pharma-transition-normal);
    }}
    
    .pharma-card:hover {{
        transform: translateY(-4px);
        border-color: var(--pharma-border-hover);
        box-shadow: var(--pharma-shadow-xl);
    }}
    
    .pharma-card:hover::before {{
        transform: scaleX(1);
    }}
    
    /* === TİPOGRAFİ === */
    .pharma-title {{
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #f8fafc, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -1px;
        animation: fadeInUp 0.8s ease-out;
    }}
    
    .pharma-subtitle {{
        font-size: 1.25rem;
        color: var(--pharma-text-secondary);
        font-weight: 400;
        line-height: 1.6;
        max-width: 900px;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }}
    
    .pharma-section-title {{
        font-size: 2rem;
        font-weight: 800;
        margin: 2rem 0 1.5rem 0;
        padding-left: 1rem;
        border-left: 5px solid;
        border-image: var(--pharma-gradient);
        border-image-slice: 1;
        background: linear-gradient(90deg, rgba(45, 125, 210, 0.1), transparent);
        padding: 1rem;
        border-radius: var(--pharma-radius-md);
    }}
    
    /* === METRİK KARTLARI === */
    .pharma-metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }}
    
    .pharma-metric-card {{
        background: var(--pharma-bg-card);
        border-radius: var(--pharma-radius-lg);
        padding: 1.5rem;
        position: relative;
        overflow: hidden;
        border: 1px solid var(--pharma-border);
        transition: all var(--pharma-transition-normal);
    }}
    
    .pharma-metric-card:hover {{
        transform: scale(1.02);
        border-color: var(--pharma-accent);
    }}
    
    .pharma-metric-label {{
        font-size: 0.875rem;
        color: var(--pharma-text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    
    .pharma-metric-value {{
        font-size: 2.5rem;
        font-weight: 900;
        color: var(--pharma-text-primary);
        line-height: 1;
        margin-bottom: 0.25rem;
    }}
    
    .pharma-metric-trend {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        color: var(--pharma-text-tertiary);
    }}
    
    /* === GÖSTERGE PANELİ === */
    .pharma-dashboard {{
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        gap: 1.5rem;
        margin-bottom: 2rem;
    }}
    
    .pharma-dashboard-item {{
        background: var(--pharma-bg-card);
        border-radius: var(--pharma-radius-lg);
        padding: 1.5rem;
        border: 1px solid var(--pharma-border);
    }}
    
    /* === İÇGÖRÜ KARTLARI === */
    .pharma-insight {{
        background: var(--pharma-bg-card);
        border-radius: var(--pharma-radius-md);
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        transition: all var(--pharma-transition-fast);
    }}
    
    .pharma-insight:hover {{
        transform: translateX(4px);
        background: var(--pharma-bg-hover);
    }}
    
    .pharma-insight-success {{ border-left-color: var(--pharma-success); }}
    .pharma-insight-warning {{ border-left-color: var(--pharma-warning); }}
    .pharma-insight-danger {{ border-left-color: var(--pharma-danger); }}
    .pharma-insight-info {{ border-left-color: var(--pharma-accent); }}
    
    /* === ANİMASYONLAR === */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes slideIn {{
        from {{
            opacity: 0;
            transform: translateX(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-5px); }}
    }}
    
    .animate-fadeInUp {{
        animation: fadeInUp 0.6s ease-out;
    }}
    
    .animate-slideIn {{
        animation: slideIn 0.6s ease-out;
    }}
    
    .animate-pulse {{
        animation: pulse 2s infinite;
    }}
    
    .animate-float {{
        animation: float 3s infinite;
    }}
    
    /* === ÖZEL BİLEŞENLER === */
    .pharma-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: var(--pharma-radius-full);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .pharma-badge-primary {{
        background: linear-gradient(135deg, rgba(45, 125, 210, 0.2), rgba(42, 202, 234, 0.2));
        color: var(--pharma-accent-light);
        border: 1px solid rgba(45, 125, 210, 0.3);
    }}
    
    .pharma-badge-success {{
        background: rgba(45, 210, 163, 0.2);
        color: var(--pharma-success);
        border: 1px solid rgba(45, 210, 163, 0.3);
    }}
    
    .pharma-badge-warning {{
        background: rgba(242, 201, 76, 0.2);
        color: var(--pharma-warning);
        border: 1px solid rgba(242, 201, 76, 0.3);
    }}
    
    .pharma-badge-danger {{
        background: rgba(235, 87, 87, 0.2);
        color: var(--pharma-danger);
        border: 1px solid rgba(235, 87, 87, 0.3);
    }}
    
    /* === PROGRESS BAR === */
    .pharma-progress {{
        width: 100%;
        height: 6px;
        background: var(--pharma-bg-secondary);
        border-radius: var(--pharma-radius-full);
        overflow: hidden;
        margin: 0.5rem 0;
    }}
    
    .pharma-progress-bar {{
        height: 100%;
        background: var(--pharma-gradient);
        border-radius: var(--pharma-radius-full);
        transition: width var(--pharma-transition-normal);
    }}
    
    /* === TABLOLAR === */
    .pharma-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }}
    
    .pharma-table th {{
        background: var(--pharma-gradient);
        color: white;
        font-weight: 600;
        padding: 1rem;
        text-align: left;
    }}
    
    .pharma-table td {{
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--pharma-border);
    }}
    
    .pharma-table tr:hover {{
        background: var(--pharma-bg-hover);
    }}
    
    /* === RESPONSIVE TASARIM === */
    @media (max-width: 1200px) {{
        .pharma-title {{ font-size: 2.5rem; }}
        .pharma-dashboard {{ grid-template-columns: repeat(6, 1fr); }}
    }}
    
    @media (max-width: 768px) {{
        .pharma-title {{ font-size: 2rem; }}
        .pharma-dashboard {{ grid-template-columns: repeat(1, 1fr); }}
        .pharma-metric-grid {{ grid-template-columns: 1fr; }}
    }}
    
    /* === SCROLLBAR === */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--pharma-bg-secondary);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--pharma-gradient);
        border-radius: var(--pharma-radius-full);
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--pharma-gradient-reverse);
    }}
</style>
"""

# ============================================================================
# 4. İLERİ DÜZEY VERİ İŞLEME MOTORU
# ============================================================================

class AdvancedDataEngine:
    """
    Kurumsal seviye veri işleme motoru - Büyük veri optimizasyonu,
    paralel işleme ve akıllı bellek yönetimi.
    """
    
    def __init__(self):
        self.config = config
        self.logger = self._setup_logging()
        self.memory_tracker = self._setup_memory_tracking()
        self.cache_manager = self._setup_cache()
        self.pool_executor = ThreadPoolExecutor(max_workers=self.config.PARALLEL_WORKERS)
        
    @staticmethod
    def _setup_logging():
        """Gelişmiş logging sistemi"""
        logger = logging.getLogger('PharmaIntelligence')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    @staticmethod
    def _setup_memory_tracking():
        """Bellek kullanımı takip sistemi"""
        tracemalloc.start()
        return {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024
        }
    
    @staticmethod
    def _setup_cache():
        """Akıllı cache yönetimi"""
        return {
            'memory': {},
            'disk': {},
            'hits': 0,
            'misses': 0
        }
    
    @st.cache_data(ttl=config.CACHE_TTL, show_spinner=False, max_entries=50)
    def load_large_dataset(
        self,
        file: Any,
        sample_size: Optional[int] = None,
        use_dask: bool = True
    ) -> Optional[Union[pd.DataFrame, dd.DataFrame]]:
        """
        Büyük veri setlerini akıllı yükleme - Paralel işleme ve bellek optimizasyonu
        """
        try:
            start_time = time.time()
            file_extension = file.name.split('.')[-1].lower()
            
            # Bellek kontrolü
            available_memory = psutil.virtual_memory().available / 1024 / 1024
            self.logger.info(f"Kullanılabilir bellek: {available_memory:.0f} MB")
            
            # Dask ile büyük veri yükleme
            if use_dask and file_extension in ['csv', 'parquet'] and sample_size is None:
                with st.spinner("⚡ Dask ile paralel veri yükleme başlatılıyor..."):
                    # Dask cluster oluştur
                    cluster = LocalCluster(
                        n_workers=self.config.PARALLEL_WORKERS,
                        threads_per_worker=2,
                        memory_limit=f'{self.config.MAX_MEMORY_MB}MB'
                    )
                    client = Client(cluster)
                    
                    if file_extension == 'csv':
                        ddf = dd.read_csv(
                            file,
                            blocksize=f'{self.config.BATCH_SIZE}MB',
                            assume_missing=True,
                            encoding='utf-8'
                        )
                    elif file_extension == 'parquet':
                        ddf = dd.read_parquet(file)
                    
                    # Örnekleme
                    if sample_size:
                        ddf = ddf.head(sample_size, npartitions=-1)
                        df = ddf.compute()
                    else:
                        df = ddf.compute()
                    
                    client.close()
                    cluster.close()
                    
            else:
                # Standart pandas yükleme
                with st.spinner("📥 Veri yükleniyor..."):
                    if file_extension == 'csv':
                        if sample_size:
                            df = pd.read_csv(file, nrows=sample_size)
                        else:
                            df = pd.read_csv(file, low_memory=False)
                    elif file_extension in ['xlsx', 'xls']:
                        df = pd.read_excel(file, engine='openpyxl')
                    elif file_extension == 'parquet':
                        df = pd.read_parquet(file)
                    else:
                        st.error("❌ Desteklenmeyen dosya formatı")
                        return None
            
            # DataFrame optimizasyonu
            df = self._optimize_dataframe(df)
            
            # Metrik hesaplama
            load_time = time.time() - start_time
            memory_used = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            st.success(f"""
            ✅ **Veri başarıyla yüklendi!**
            - Satır sayısı: {len(df):,}
            - Sütun sayısı: {len(df.columns)}
            - Bellek kullanımı: {memory_used:.1f} MB
            - Yükleme süresi: {load_time:.2f} saniye
            - Dosya formatı: {file_extension.upper()}
            """)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Veri yükleme hatası: {str(e)}")
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.error(f"Detay: {traceback.format_exc()}")
            return None
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş DataFrame optimizasyonu - Akıllı tip dönüşümü ve bellek tasarrufu
        """
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Sütun isimlerini temizle
            df.columns = self._clean_column_names(df.columns)
            
            # Paralel sütun optimizasyonu
            futures = []
            for col in df.columns:
                future = self.pool_executor.submit(
                    self._optimize_column,
                    df[col].copy()
                )
                futures.append((col, future))
            
            # Optimize edilmiş sütunları ata
            for col, future in futures:
                try:
                    df[col] = future.result(timeout=5)
                except Exception as e:
                    self.logger.warning(f"Sütun optimizasyon hatası {col}: {str(e)}")
                    continue
            
            # Bellek tasarrufu raporu
            optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            memory_saved = original_memory - optimized_memory
            save_percentage = (memory_saved / original_memory * 100) if original_memory > 0 else 0
            
            if memory_saved > 0:
                st.info(f"""
                💾 **Bellek optimizasyonu:**
                - Önce: {original_memory:.1f} MB
                - Sonra: {optimized_memory:.1f} MB
                - Tasarruf: {memory_saved:.1f} MB (%{save_percentage:.1f})
                """)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def _optimize_column(series: pd.Series) -> pd.Series:
        """Tek bir sütunu optimize et"""
        if series.dtype == 'object':
            # String sütunları
            try:
                series = series.astype(str).str.strip()
                n_unique = series.nunique()
                n_total = len(series)
                
                if n_unique < n_total * 0.5:
                    series = series.astype('category')
            except:
                pass
                
        elif series.dtype in ['int64', 'int32']:
            # Integer sütunları
            try:
                min_val = series.min()
                max_val = series.max()
                
                if pd.isna(min_val) or pd.isna(max_val):
                    return series
                
                if min_val >= 0:
                    if max_val <= 255:
                        series = series.astype('uint8')
                    elif max_val <= 65535:
                        series = series.astype('uint16')
                    elif max_val <= 4294967295:
                        series = series.astype('uint32')
                else:
                    if min_val >= -128 and max_val <= 127:
                        series = series.astype('int8')
                    elif min_val >= -32768 and max_val <= 32767:
                        series = series.astype('int16')
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        series = series.astype('int32')
            except:
                pass
                
        elif series.dtype == 'float64':
            # Float sütunları
            try:
                series = series.astype('float32')
            except:
                pass
                
        return series
    
    @staticmethod
    def _clean_column_names(columns: List[str]) -> List[str]:
        """Gelişmiş sütun ismi temizleme"""
        cleaned = []
        seen = {}
        
        for col in columns:
            if isinstance(col, str):
                # Türkçe karakter dönüşümü
                char_map = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                
                for tr, en in char_map.items():
                    col = col.replace(tr, en)
                
                # Özel karakterleri temizle
                col = re.sub(r'[^\w\s-]', '', col)
                
                # Boşlukları alt çizgiye çevir
                col = re.sub(r'\s+', '_', col)
                
                # Birden fazla alt çizgiyi teke indir
                col = re.sub(r'_+', '_', col)
                
                # Başındaki ve sonundaki alt çizgileri temizle
                col = col.strip('_')
                
                # Domain-spesifik haritalama
                col = AdvancedDataEngine._apply_domain_mapping(col)
                
            else:
                col = str(col)
            
            # Duplicate handling
            if col in seen:
                seen[col] += 1
                col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
            
            cleaned.append(col)
        
        return cleaned
    
    @staticmethod
    def _apply_domain_mapping(col: str) -> str:
        """Domain-spesifik sütun ismi haritalama"""
        
        # Satış sütunları
        if re.search(r'MAT\s*Q3\s*2022.*USD\s*MNF', col, re.IGNORECASE):
            return 'Sales_2022'
        elif re.search(r'MAT\s*Q3\s*2023.*USD\s*MNF', col, re.IGNORECASE):
            return 'Sales_2023'
        elif re.search(r'MAT\s*Q3\s*2024.*USD\s*MNF', col, re.IGNORECASE):
            return 'Sales_2024'
        
        # Birim sütunları
        elif re.search(r'MAT\s*Q3\s*2022.*Units', col, re.IGNORECASE):
            return 'Units_2022'
        elif re.search(r'MAT\s*Q3\s*2023.*Units', col, re.IGNORECASE):
            return 'Units_2023'
        elif re.search(r'MAT\s*Q3\s*2024.*Units', col, re.IGNORECASE):
            return 'Units_2024'
        
        # Fiyat sütunları
        elif re.search(r'MAT\s*Q3\s*2022.*Unit\s*Avg\s*Price', col, re.IGNORECASE):
            return 'Avg_Price_2022'
        elif re.search(r'MAT\s*Q3\s*2023.*Unit\s*Avg\s*Price', col, re.IGNORECASE):
            return 'Avg_Price_2023'
        elif re.search(r'MAT\s*Q3\s*2024.*Unit\s*Avg\s*Price', col, re.IGNORECASE):
            return 'Avg_Price_2024'
        
        # Standart birimler
        elif re.search(r'MAT\s*Q3\s*2022.*Standard\s*Units', col, re.IGNORECASE):
            return 'Standard_Units_2022'
        elif re.search(r'MAT\s*Q3\s*2023.*Standard\s*Units', col, re.IGNORECASE):
            return 'Standard_Units_2023'
        elif re.search(r'MAT\s*Q3\s*2024.*Standard\s*Units', col, re.IGNORECASE):
            return 'Standard_Units_2024'
        
        # Diğer sütunlar
        mapping = {
            'Country': 'Country',
            'Sector': 'Sector',
            'Corporation': 'Company',
            'Manufacturer': 'Manufacturer',
            'Molecule List': 'Molecule_List',
            'Molecule': 'Molecule',
            'Chemical Salt': 'Chemical_Salt',
            'International Product': 'International_Product',
            'Specialty Product': 'Specialty_Product',
            'NFC123': 'NFC_Code',
            'International Pack': 'International_Pack',
            'International Strength': 'International_Strength',
            'International Size': 'International_Size',
            'International Volume': 'International_Volume',
            'International Prescription': 'International_Prescription',
            'Panel': 'Panel',
            'Region': 'Region',
            'Sub-Region': 'Sub_Region'
        }
        
        for key, value in mapping.items():
            if key in col:
                return value
        
        return col

# ============================================================================
# 5. DERİN ÖĞRENME & GELİŞMİŞ TAHMİN MOTORU
# ============================================================================

class DeepLearningEngine:
    """
    Derin öğrenme tabanlı gelişmiş tahminleme motoru - LSTM, GRU,
    Transformer modelleri ve ensemble yöntemleri.
    """
    
    def __init__(self):
        self.config = config
        self.models = {}
        self.history = {}
        self.scalers = {}
        
    def build_lstm_model(
        self,
        input_shape: Tuple[int, int],
        n_units: List[int] = [64, 32],
        dropout_rate: float = 0.2
    ) -> keras.Model:
        """Gelişmiş LSTM modeli - Çok katmanlı, bidirectional"""
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Bidirectional LSTM layers
        for i, units in enumerate(n_units):
            return_sequences = i < len(n_units) - 1
            
            model.add(Bidirectional(
                LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    kernel_regularizer=regularizers.l2(1e-4),
                    recurrent_regularizer=regularizers.l2(1e-4)
                )
            ))
            
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dropout(dropout_rate / 2))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        optimizer = optimizers.Adam(
            learning_rate=self.config.LEARNING_RATE,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_gru_model(
        self,
        input_shape: Tuple[int, int],
        n_units: List[int] = [64, 32],
        dropout_rate: float = 0.2
    ) -> keras.Model:
        """Gelişmiş GRU modeli - Daha hızlı eğitim, benzer performans"""
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # GRU layers
        for i, units in enumerate(n_units):
            return_sequences = i < len(n_units) - 1
            
            model.add(GRU(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                kernel_regularizer=regularizers.l2(1e-4),
                recurrent_regularizer=regularizers.l2(1e-4)
            ))
            
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        optimizer = optimizers.Adam(
            learning_rate=self.config.LEARNING_RATE,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def build_transformer_model(
        self,
        input_shape: Tuple[int, int],
        d_model: int = 64,
        n_heads: int = 4,
        ff_dim: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1
    ) -> keras.Model:
        """Transformer modeli - Son teknoloji zaman serisi tahmini"""
        
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        x = self._add_positional_encoding(inputs)
        
        # Transformer blocks
        for _ in range(n_layers):
            # Multi-head attention
            attention = layers.MultiHeadAttention(
                num_heads=n_heads,
                key_dim=d_model // n_heads
            )(x, x)
            attention = layers.Dropout(dropout_rate)(attention)
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)
            
            # Feed forward
            ffn = layers.Dense(ff_dim, activation='relu')(x)
            ffn = layers.Dropout(dropout_rate)(ffn)
            ffn = layers.Dense(d_model)(ffn)
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization()(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(16, activation='relu')(x)
        
        # Output
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = optimizers.Adam(
            learning_rate=self.config.LEARNING_RATE * 0.1,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    @staticmethod
    def _add_positional_encoding(inputs: tf.Tensor) -> tf.Tensor:
        """Transformer için pozisyonel encoding ekle"""
        
        sequence_length = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        
        positions = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
        depths = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :] / d_model
        
        angle_rates = 1 / tf.pow(10000.0, depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = tf.concat([
            tf.sin(angle_rads[:, 0::2]),
            tf.cos(angle_rads[:, 1::2])
        ], axis=-1)
        
        return inputs + pos_encoding[tf.newaxis, :, :d_model]
    
    def prepare_sequences(
        self,
        data: pd.Series,
        sequence_length: int = 12,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Zaman serisi için sequence hazırlama"""
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i:i + sequence_length].values)
            y.append(data[i + sequence_length:i + sequence_length + prediction_horizon].values)
        
        return np.array(X), np.array(y)
    
    def train_deep_learning_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_type: str = 'lstm',
        **kwargs
    ) -> keras.Model:
        """Derin öğrenme modeli eğitimi"""
        
        # Model seçimi
        if model_type == 'lstm':
            model = self.build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                **kwargs
            )
        elif model_type == 'gru':
            model = self.build_gru_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                **kwargs
            )
        elif model_type == 'transformer':
            model = self.build_transformer_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                **kwargs
            )
        else:
            raise ValueError(f"Bilinmeyen model tipi: {model_type}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.REDUCE_LR_PATIENCE,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'best_{model_type}_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Eğitim
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE_DL,
            callbacks=callbacks,
            verbose=self.config.VERBOSE
        )
        
        self.models[model_type] = model
        self.history[model_type] = history
        
        return model
    
    def predict_future(
        self,
        model: keras.Model,
        last_sequence: np.ndarray,
        n_predictions: int = 12
    ) -> np.ndarray:
        """Gelecek değerleri tahmin et"""
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_predictions):
            # Tahmin yap
            pred = model.predict(current_sequence[np.newaxis, ...], verbose=0)[0, 0]
            predictions.append(pred)
            
            # Sequence'i güncelle
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred
        
        return np.array(predictions)

# ============================================================================
# 6. OTOMATİK MAKİNE ÖĞRENİMİ (AUTOML) MOTORU
# ============================================================================

class AutoMLEngine:
    """
    Otomatik makine öğrenimi motoru - Hiperparametre optimizasyonu,
    model seçimi ve ensemble yöntemleri.
    """
    
    def __init__(self):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.feature_importance = None
        
    def get_regression_models(self) -> Dict[str, Any]:
        """Tüm regression modelleri"""
        
        return {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=self.config.RANDOM_STATE
            ),
            'XGBoost': GradientBoostingRegressor(  # Placeholder for actual XGBoost
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.config.RANDOM_STATE
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.config.RANDOM_STATE
            ),
            'SVR': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1,
                gamma='scale'
            ),
            'Ridge': Ridge(
                alpha=1.0,
                random_state=self.config.RANDOM_STATE
            ),
            'Lasso': Lasso(
                alpha=0.01,
                random_state=self.config.RANDOM_STATE
            ),
            'ElasticNet': ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                random_state=self.config.RANDOM_STATE
            ),
            'Huber': HuberRegressor(
                epsilon=1.35,
                max_iter=100,
                alpha=0.0001
            )
        }
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """Model hiperparametre grid'leri"""
        
        return {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.6, 0.8, 1.0]
            },
            'SVR': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'epsilon': [0.01, 0.1, 0.2]
            }
        }
    
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str,
        n_iter: int = 20,
        cv: int = 5
    ) -> Dict:
        """Bayesian optimizasyon ile hiperparametre optimizasyonu"""
        
        if model_name not in self.get_regression_models():
            raise ValueError(f"Bilinmeyen model: {model_name}")
        
        model = self.get_regression_models()[model_name]
        param_grid = self.get_param_grids().get(model_name, {})
        
        if not param_grid:
            return {}
        
        # Randomized search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=self.config.N_JOBS,
            random_state=self.config.RANDOM_STATE,
            verbose=self.config.VERBOSE
        )
        
        random_search.fit(X_train, y_train)
        
        return {
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,
            'best_estimator': random_search.best_estimator_
        }
    
    def train_ensemble_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> VotingRegressor:
        """Ensemble model eğitimi"""
        
        # Base models
        estimators = [
            ('rf', RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.config.RANDOM_STATE
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS
            ))
        ]
        
        # Voting regressor
        voting_regressor = VotingRegressor(
            estimators=estimators,
            weights=[2, 2, 1]
        )
        
        voting_regressor.fit(X_train, y_train)
        
        # Validation score
        y_pred = voting_regressor.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        st.info(f"""
        🎯 **Ensemble Model Performansı:**
        - MSE: {mse:.4f}
        - R²: {r2:.4f}
        - RMSE: {np.sqrt(mse):.4f}
        """)
        
        return voting_regressor
    
    def feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10
    ) -> List[str]:
        """Gelişmiş özellik seçimi"""
        
        # Mutual information
        mi_selector = SelectKBest(
            mutual_info_regression,
            k=min(n_features * 2, X.shape[1])
        )
        mi_selector.fit(X, y)
        mi_scores = pd.DataFrame({
            'feature': X.columns,
            'score': mi_selector.scores_
        }).sort_values('score', ascending=False)
        
        # Random Forest importance
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=self.config.RANDOM_STATE,
            n_jobs=self.config.N_JOBS
        )
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Correlation analysis
        correlations = pd.DataFrame({
            'feature': X.columns,
            'correlation': abs(X.corrwith(y))
        }).sort_values('correlation', ascending=False)
        
        # Ensemble feature importance
        ensemble_scores = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_selector.scores_,
            'rf_importance': rf.feature_importances_,
            'correlation': abs(X.corrwith(y))
        })
        
        # Normalize scores
        for col in ['mi_score', 'rf_importance', 'correlation']:
            ensemble_scores[col] = (ensemble_scores[col] - ensemble_scores[col].min()) / \
                                  (ensemble_scores[col].max() - ensemble_scores[col].min() + 1e-10)
        
        ensemble_scores['final_score'] = ensemble_scores[['mi_score', 'rf_importance', 'correlation']].mean(axis=1)
        ensemble_scores = ensemble_scores.sort_values('final_score', ascending=False)
        
        self.feature_importance = ensemble_scores
        
        return ensemble_scores.head(n_features)['feature'].tolist()

# ============================================================================
# 7. GELİŞMİŞ GÖRSELLEŞTİRME MOTORU
# ============================================================================

class AdvancedVisualizationEngine:
    """
    Profesyonel görselleştirme motoru - İnteraktif grafikler,
    3D görselleştirmeler ve animasyonlar.
    """
    
    def __init__(self):
        self.config = config
        self.color_palette = self._create_custom_palette()
        
    @staticmethod
    def _create_custom_palette() -> List[str]:
        """Özel renk paleti oluştur"""
        
        return [
            '#2d7dd2', '#2acaea', '#30c9c9', '#2dd2a3', '#f2c94c',
            '#eb5757', '#9b51e0', '#f2994a', '#6fcf97', '#bb6bd9',
            '#56ccf2', '#219653', '#f2c94c', '#eb5757', '#2f80ed'
        ]
    
    def create_3d_market_dashboard(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        color_col: str,
        size_col: str
    ) -> go.Figure:
        """3D pazar gösterge paneli"""
        
        fig = go.Figure()
        
        # 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=df[x_col],
            y=df[y_col],
            z=df[z_col],
            mode='markers',
            marker=dict(
                size=df[size_col] / df[size_col].max() * 50 + 10,
                color=df[color_col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_col),
                line=dict(width=0.5, color='white')
            ),
            text=df.apply(lambda row: f"""
                <b>{row.get('Molecule', 'N/A')}</b><br>
                {x_col}: ${row[x_col]:,.0f}<br>
                {y_col}: {row[y_col]:.1f}%<br>
                {z_col}: ${row[z_col]:.2f}<br>
                {color_col}: {row[color_col]:.2f}<br>
                Şirket: {row.get('Company', 'N/A')}<br>
                Ülke: {row.get('Country', 'N/A')}
            """, axis=1),
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title={
                'text': '3D Pazar Analitik Gösterge Paneli',
                'font': {'size': 24, 'color': '#f8fafc'},
                'x': 0.5
            },
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='#404040', color='#f8fafc'),
                yaxis=dict(gridcolor='#404040', color='#f8fafc'),
                zaxis=dict(gridcolor='#404040', color='#f8fafc')
            ),
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc'
        )
        
        return fig
    
    def create_market_network_graph(
        self,
        df: pd.DataFrame,
        source_col: str = 'Company',
        target_col: str = 'Molecule',
        value_col: str = 'Sales_2024'
    ) -> go.Figure:
        """Pazar ağ grafiği - Şirket-Molekül ilişkileri"""
        
        # Node'ları oluştur
        companies = df[source_col].unique()
        molecules = df[target_col].unique()
        
        nodes = list(companies) + list(molecules)
        node_colors = ['#2d7dd2'] * len(companies) + ['#2acaea'] * len(molecules)
        node_sizes = [20] * len(companies) + [15] * len(molecules)
        
        # Edge'leri oluştur
        edges = []
        edge_values = []
        
        for _, row in df.iterrows():
            source_idx = list(companies).index(row[source_col])
            target_idx = len(companies) + list(molecules).index(row[target_col])
            edges.append((source_idx, target_idx))
            edge_values.append(row[value_col])
        
        # Normalize edge values
        edge_values = np.array(edge_values)
        edge_widths = edge_values / edge_values.max() * 5 + 1
        
        # Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='white', width=0.5),
                label=nodes,
                color=node_colors
            ),
            link=dict(
                source=[e[0] for e in edges],
                target=[e[1] for e in edges],
                value=edge_values,
                color=[f'rgba(45, 125, 210, {w/10})' for w in edge_widths]
            )
        )])
        
        fig.update_layout(
            title={
                'text': 'Şirket-Molekül Pazar İlişki Ağı',
                'font': {'size': 24, 'color': '#f8fafc'},
                'x': 0.5
            },
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#f8fafc')
        )
        
        return fig
    
    def create_animated_time_series(
        self,
        df: pd.DataFrame,
        time_cols: List[str],
        category_col: str = 'Company',
        value_col: str = 'Sales'
    ) -> go.Figure:
        """Animasyonlu zaman serisi grafiği"""
        
        # Veriyi yeniden şekillendir
        plot_data = []
        
        for col in time_cols:
            year = self._extract_year(col)
            if year:
                for _, row in df.iterrows():
                    plot_data.append({
                        'Year': year,
                        'Category': row[category_col],
                        'Value': row[col]
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Animasyonlu bar chart
        fig = px.bar(
            plot_df,
            x='Category',
            y='Value',
            color='Category',
            animation_frame='Year',
            range_y=[0, plot_df['Value'].max() * 1.1],
            title='Yıllara Göre Pazar Dinamikleri',
            labels={'Value': 'Satış (USD)', 'Category': 'Şirket'}
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis=dict(gridcolor='#404040'),
            yaxis=dict(gridcolor='#404040')
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_parallel_coordinates_plot(
        self,
        df: pd.DataFrame,
        dimensions: List[str],
        color_col: str
    ) -> go.Figure:
        """Paralel koordinatlar grafiği - Çok boyutlu veri analizi"""
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=df[color_col],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_col, tickfont=dict(color='#f8fafc'))
                ),
                dimensions=[
                    dict(
                        label=dim,
                        values=df[dim],
                        tickfont=dict(color='#f8fafc'),
                        titlefont=dict(color='#f8fafc')
                    ) for dim in dimensions
                ]
            )
        )
        
        fig.update_layout(
            title={
                'text': 'Çok Boyutlu Pazar Analizi',
                'font': {'size': 24, 'color': '#f8fafc'},
                'x': 0.5
            },
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc'
        )
        
        return fig
    
    def create_radar_chart_comparison(
        self,
        df: pd.DataFrame,
        entities: List[str],
        metrics: List[str],
        entity_col: str = 'Company'
    ) -> go.Figure:
        """Radar grafiği ile şirket karşılaştırması"""
        
        fig = go.Figure()
        
        # Normalize metrics
        normalized_df = df.copy()
        for metric in metrics:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val > min_val:
                normalized_df[metric] = (df[metric] - min_val) / (max_val - min_val) * 100
            else:
                normalized_df[metric] = 50
        
        for entity in entities[:5]:  # Limit for readability
            entity_data = normalized_df[normalized_df[entity_col] == entity]
            
            if len(entity_data) > 0:
                values = entity_data[metrics].mean().values.flatten().tolist()
                values += values[:1]  # Complete the loop
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=entity
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='#404040'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            title={
                'text': 'Şirket Performans Radar Karşılaştırması',
                'font': {'size': 24, 'color': '#f8fafc'},
                'x': 0.5
            },
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.1,
                xanchor='center',
                x=0.5,
                font=dict(color='#f8fafc')
            )
        )
        
        return fig
    
    def create_heatmap_correlation(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> go.Figure:
        """Korelasyon ısı haritası"""
        
        corr_matrix = df[columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={'color': '#f8fafc'},
            colorbar=dict(
                title='Korelasyon',
                tickfont=dict(color='#f8fafc'),
                titlefont=dict(color='#f8fafc')
            )
        ))
        
        fig.update_layout(
            title={
                'text': 'Pazar Metrikleri Korelasyon Matrisi',
                'font': {'size': 24, 'color': '#f8fafc'},
                'x': 0.5
            },
            height=700,
            width=700,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis=dict(gridcolor='#404040'),
            yaxis=dict(gridcolor='#404040')
        )
        
        return fig
    
    @staticmethod
    def _extract_year(column_name: str) -> Optional[int]:
        """Sütun adından yıl çıkar"""
        match = re.search(r'\b(20\d{2})\b', column_name)
        if match:
            return int(match.group(1))
        return None

# ============================================================================
# 8. NLP & PAZAR İSTİHBARAT MOTORU
# ============================================================================

class MarketIntelligenceEngine:
    """
    NLP tabanlı pazar istihbarat motoru - Duygu analizi,
    trend tespiti ve rekabet istihbaratı.
    """
    
    def __init__(self):
        self.config = config
        self.nltk_initialized = False
        self._initialize_nltk()
        self.sentiment_analyzer = None
        
    def _initialize_nltk(self):
        """NLTK kaynaklarını başlat"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            self.nltk_initialized = True
        except Exception as e:
            self.logger.error(f"NLTK başlatma hatası: {str(e)}")
    
    def analyze_market_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pazar duygu analizi"""
        
        if not self.nltk_initialized:
            return df
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Metin sütunlarını bul
        text_columns = df.select_dtypes(include=['object', 'category']).columns
        text_columns = [col for col in text_columns if df[col].dtype == 'object']
        
        if not text_columns:
            return df
        
        # Duygu analizi
        sentiment_scores = []
        
        for col in text_columns[:3]:  # Limit for performance
            sample_text = ' '.join(df[col].dropna().astype(str).head(100))
            sentiment = self.sentiment_analyzer.polarity_scores(sample_text)
            sentiment_scores.append(sentiment['compound'])
        
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            df.attrs['market_sentiment'] = avg_sentiment
            df.attrs['sentiment_label'] = 'Pozitif' if avg_sentiment > 0.05 else 'Negatif' if avg_sentiment < -0.05 else 'Nötr'
        
        return df
    
    def extract_key_phrases(self, df: pd.DataFrame, column: str, n_phrases: int = 10) -> List[str]:
        """Önemli kelime öbeklerini çıkar"""
        
        if column not in df.columns or df[column].dtype != 'object':
            return []
        
        # Metin verisini birleştir
        text = ' '.join(df[column].dropna().astype(str).head(1000))
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Stop words'leri temizle
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 2]
        
        # Frequency analysis
        freq_dist = nltk.FreqDist(tokens)
        key_phrases = [phrase for phrase, _ in freq_dist.most_common(n_phrases)]
        
        return key_phrases
    
    def analyze_competitor_intelligence(
        self,
        df: pd.DataFrame,
        competitor_col: str = 'Company'
    ) -> Dict[str, Any]:
        """Rekabet istihbaratı analizi"""
        
        intelligence = {}
        
        if competitor_col not in df.columns:
            return intelligence
        
        # Şirket bazlı analiz
        companies = df[competitor_col].value_counts()
        
        intelligence['total_companies'] = len(companies)
        intelligence['market_leaders'] = companies.head(5).to_dict()
        
        # Pazar payı analizi
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        if sales_cols:
            latest_sales = sales_cols[-1]
            company_sales = df.groupby(competitor_col)[latest_sales].sum().sort_values(ascending=False)
            total_sales = company_sales.sum()
            
            if total_sales > 0:
                intelligence['market_shares'] = (company_sales / total_sales * 100).head(10).to_dict()
                
                # HHI Index
                intelligence['hhi_index'] = ((company_sales / total_sales * 100) ** 2).sum()
        
        return intelligence
    
    def detect_market_trends(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Pazar trendlerini tespit et"""
        
        trends = []
        
        # Satış trendleri
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        if len(sales_cols) >= 2:
            current_year = sales_cols[-1]
            previous_year = sales_cols[-2]
            
            # Büyüme hesapla
            growth = ((df[current_year].sum() - df[previous_year].sum()) / 
                     df[previous_year].sum() * 100)
            
            trends.append({
                'category': 'Satış',
                'trend': 'Yükseliş' if growth > 0 else 'Düşüş',
                'value': f"%{growth:.1f}",
                'description': f'Toplam pazar {abs(growth):.1f}% {"büyüdü" if growth > 0 else "küçüldü"}'
            })
        
        # Fiyat trendleri
        price_cols = [col for col in df.columns if 'Price_' in col]
        if len(price_cols) >= 2:
            current_price = price_cols[-1]
            previous_price = price_cols[-2]
            
            price_change = ((df[current_price].mean() - df[previous_price].mean()) / 
                          df[previous_price].mean() * 100)
            
            trends.append({
                'category': 'Fiyat',
                'trend': 'Yükseliş' if price_change > 0 else 'Düşüş',
                'value': f"%{price_change:.1f}",
                'description': f'Ortalama fiyatlar {abs(price_change):.1f}% {"arttı" if price_change > 0 else "azaldı"}'
            })
        
        return trends

# ============================================================================
# 9. SENARYO SİMÜLASYON MOTORU
# ============================================================================

class ScenarioSimulationEngine:
    """
    Senaryo simülasyon motoru - Monte Carlo, what-if analizi
    ve risk değerlendirmesi.
    """
    
    def __init__(self):
        self.config = config
        self.simulation_results = {}
        
    def monte_carlo_forecast(
        self,
        historical_data: pd.Series,
        n_simulations: int = 1000,
        forecast_periods: int = 12,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Monte Carlo simülasyonu ile belirsizlik tahmini"""
        
        # Tarihsel veri istatistikleri
        mean = historical_data.mean()
        std = historical_data.std()
        last_value = historical_data.iloc[-1]
        
        # Simülasyon sonuçları
        simulations = []
        
        for _ in range(n_simulations):
            # Random walk with drift
            noise = np.random.normal(0, std, forecast_periods)
            drift = np.random.normal(mean * 0.01, std * 0.1, forecast_periods)
            
            forecast = [last_value]
            for i in range(1, forecast_periods):
                next_value = forecast[-1] * (1 + drift[i-1] / 100) + noise[i-1]
                forecast.append(next_value)
            
            simulations.append(forecast[1:])  # Exclude initial value
        
        simulations = np.array(simulations)
        
        # İstatistikler
        mean_forecast = simulations.mean(axis=0)
        std_forecast = simulations.std(axis=0)
        
        # Güven aralıkları
        z_score = norm.ppf((1 + confidence_level) / 2)
        lower_bound = mean_forecast - z_score * std_forecast
        upper_bound = mean_forecast + z_score * std_forecast
        
        # Risk metrikleri
        var_95 = np.percentile(simulations[:, -1], 5)  # Value at Risk
        cvar_95 = simulations[:, -1][simulations[:, -1] <= var_95].mean()  # Conditional VaR
        
        return {
            'mean_forecast': mean_forecast,
            'std_forecast': std_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'simulations': simulations,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'confidence_level': confidence_level
        }
    
    def what_if_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        scenarios: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """What-if senaryo analizi"""
        
        results = []
        base_value = df[target_column].sum() if target_column in df.columns else 0
        
        for i, scenario in enumerate(scenarios):
            scenario_df = df.copy()
            scenario_impact = 1.0
            
            for var, change in scenario.items():
                if var in scenario_df.columns:
                    scenario_df[var] = scenario_df[var] * (1 + change / 100)
                    scenario_impact *= (1 + change / 100)
            
            new_value = scenario_df[target_column].sum() if target_column in scenario_df.columns else base_value
            impact = ((new_value - base_value) / base_value * 100) if base_value != 0 else 0
            
            results.append({
                'senaryo_id': i + 1,
                'senaryo_adi': scenario.get('name', f'Senaryo {i+1}'),
                'değişkenler': ', '.join([f'{k}: %{v:+.1f}' for k, v in scenario.items() if k != 'name']),
                'etki_yuzdesi': impact,
                'yeni_deger': new_value
            })
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        variables: List[str],
        perturbation: float = 10
    ) -> pd.DataFrame:
        """Duyarlılık analizi - Tornado chart için"""
        
        base_value = df[target_column].sum() if target_column in df.columns else 0
        
        sensitivity = []
        
        for var in variables:
            if var not in df.columns:
                continue
            
            # Positive shock
            df_pos = df.copy()
            df_pos[var] = df_pos[var] * (1 + perturbation / 100)
            pos_value = df_pos[target_column].sum() if target_column in df_pos.columns else base_value
            pos_impact = ((pos_value - base_value) / base_value * 100) if base_value != 0 else 0
            
            # Negative shock
            df_neg = df.copy()
            df_neg[var] = df_neg[var] * (1 - perturbation / 100)
            neg_value = df_neg[target_column].sum() if target_column in df_neg.columns else base_value
            neg_impact = ((neg_value - base_value) / base_value * 100) if base_value != 0 else 0
            
            sensitivity.append({
                'değişken': var,
                'pozitif_etki': pos_impact,
                'negatif_etki': neg_impact,
                'duyarlılık': abs(pos_impact) + abs(neg_impact) / 2
            })
        
        return pd.DataFrame(sensitivity).sort_values('duyarlılık', ascending=False)

# ============================================================================
# 10. KURUMSAL RAPORLAMA MOTORU
# ============================================================================

class EnterpriseReportingEngine:
    """
    Kurumsal raporlama motoru - PDF, Excel, HTML, PowerPoint
    formatlarında profesyonel rapor üretimi.
    """
    
    def __init__(self):
        self.config = config
        self.report_templates = {}
        
    def generate_executive_summary(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, Any],
        insights: List[Dict[str, str]]
    ) -> str:
        """Yönetici özeti raporu"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PharmaIntelligence Pro - Yönetici Özeti</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Inter', sans-serif;
                    background: linear-gradient(135deg, #0a1929, #1e3a5f);
                    color: #f8fafc;
                    padding: 40px;
                    line-height: 1.6;
                }}
                
                .report-container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: rgba(30, 58, 95, 0.7);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 40px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6);
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    border-bottom: 2px solid #2d7dd2;
                    padding-bottom: 20px;
                }}
                
                .title {{
                    font-size: 2.5rem;
                    font-weight: 800;
                    background: linear-gradient(135deg, #f8fafc, #cbd5e1);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 10px;
                }}
                
                .subtitle {{
                    color: #cbd5e1;
                    font-size: 1.1rem;
                }}
                
                .date {{
                    color: #2d7dd2;
                    font-size: 0.9rem;
                    margin-top: 10px;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                
                .metric-card {{
                    background: rgba(45, 125, 210, 0.1);
                    border: 1px solid rgba(45, 125, 210, 0.3);
                    border-radius: 12px;
                    padding: 20px;
                    transition: transform 0.3s;
                }}
                
                .metric-card:hover {{
                    transform: translateY(-5px);
                    background: rgba(45, 125, 210, 0.2);
                }}
                
                .metric-label {{
                    color: #cbd5e1;
                    font-size: 0.85rem;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 10px;
                }}
                
                .metric-value {{
                    font-size: 1.8rem;
                    font-weight: 700;
                    color: #f8fafc;
                }}
                
                .insights-section {{
                    margin-bottom: 40px;
                }}
                
                .section-title {{
                    font-size: 1.5rem;
                    font-weight: 700;
                    margin-bottom: 20px;
                    padding-left: 15px;
                    border-left: 5px solid #2d7dd2;
                }}
                
                .insight-card {{
                    background: rgba(45, 125, 210, 0.05);
                    border-left: 4px solid #2dd2a3;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 15px;
                }}
                
                .insight-title {{
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: #2dd2a3;
                    margin-bottom: 8px;
                }}
                
                .insight-description {{
                    color: #cbd5e1;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                    color: #64748b;
                    font-size: 0.85rem;
                }}
                
                @media print {{
                    body {{
                        background: white;
                        color: black;
                    }}
                    .report-container {{
                        background: white;
                        color: black;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="report-container">
                <div class="header">
                    <div class="title">💊 PHARMAINTELLIGENCE PRO</div>
                    <div class="subtitle">Yönetici Özeti - Pazar Analitik Raporu</div>
                    <div class="date">{datetime.now().strftime('%d %B %Y, %H:%M')}</div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Toplam Pazar Değeri</div>
                        <div class="metric-value">${metrics.get('Total_Market_Value', 0)/1e6:.1f}M</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Ortalama Büyüme</div>
                        <div class="metric-value">%{metrics.get('Average_Growth_Rate', 0):.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Pazar Yoğunluğu (HHI)</div>
                        <div class="metric-value">{metrics.get('HHI_Index', 0):.0f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Ülke Kapsamı</div>
                        <div class="metric-value">{metrics.get('Country_Coverage', 0)}</div>
                    </div>
                </div>
        """
        
        # Insights section
        html += """
                <div class="insights-section">
                    <div class="section-title">🔍 Stratejik İçgörüler</div>
        """
        
        for insight in insights[:8]:
            html += f"""
                    <div class="insight-card">
                        <div class="insight-title">{insight.get('baslik', 'İçgörü')}</div>
                        <div class="insight-description">{insight.get('aciklama', '')}</div>
                    </div>
            """
        
        html += f"""
                </div>
                
                <div class="footer">
                    {self.config.REPORT_FOOTER}<br>
                    Rapor ID: {hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12].upper()}<br>
                    Versiyon: {self.config.APP_VERSION} | Build: {self.config.APP_BUILD}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def generate_pdf_report(
        self,
        df: pd.DataFrame,
        metrics: Dict[str, Any],
        insights: List[Dict[str, str]]
    ) -> Optional[BytesIO]:
        """PDF rapor üretimi"""
        
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            story = []
            
            # Başlık
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#2d7dd2'),
                spaceAfter=30,
                alignment=1
            )
            
            title = Paragraph("PharmaIntelligence Pro - Pazar Analiz Raporu", title_style)
            story.append(title)
            
            # Tarih
            date_style = ParagraphStyle(
                'DateStyle',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#64748b'),
                alignment=2,
                spaceAfter=30
            )
            
            date_text = Paragraph(f"Oluşturulma: {datetime.now().strftime('%d.%m.%Y %H:%M')}", date_style)
            story.append(date_text)
            
            story.append(Spacer(1, 20))
            
            # Metrikler
            metric_data = [
                ['Metrik', 'Değer'],
                ['Toplam Pazar Değeri', f"${metrics.get('Total_Market_Value', 0)/1e6:.1f}M"],
                ['Ortalama Büyüme', f"%{metrics.get('Average_Growth_Rate', 0):.1f}"],
                ['HHI İndeksi', f"{metrics.get('HHI_Index', 0):.0f}"],
                ['Ülke Sayısı', str(metrics.get('Country_Coverage', 0))],
                ['Ürün Sayısı', str(len(df))]
            ]
            
            metric_table = Table(metric_data, colWidths=[200, 200])
            metric_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#2d7dd2')),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#1e3a5f')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2d7dd2')),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            
            story.append(metric_table)
            story.append(Spacer(1, 30))
            
            # İçgörüler
            insight_title = Paragraph("Stratejik İçgörüler", styles['Heading2'])
            story.append(insight_title)
            story.append(Spacer(1, 15))
            
            for insight in insights[:5]:
                insight_text = Paragraph(
                    f"<b>{insight.get('baslik', 'İçgörü')}:</b> {insight.get('aciklama', '')}",
                    styles['Normal']
                )
                story.append(insight_text)
                story.append(Spacer(1, 10))
            
            # PDF oluştur
            doc.build(story)
            buffer.seek(0)
            
            return buffer
            
        except Exception as e:
            st.error(f"PDF rapor üretim hatası: {str(e)}")
            return None

# ============================================================================
# 11. GÜVENLİK & ŞİFRELEME MOTORU
# ============================================================================

class SecurityEngine:
    """
    Kurumsal güvenlik motoru - Veri şifreleme, yetkilendirme,
    denetim kayıtları ve güvenlik duvarı.
    """
    
    def __init__(self):
        self.config = config
        self.cipher = Fernet(config.ENCRYPTION_KEY.encode())
        self.audit_log = []
        
    def encrypt_dataframe(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """DataFrame sütunlarını şifrele"""
        
        encrypted_df = df.copy()
        
        for col in columns:
            if col in encrypted_df.columns and encrypted_df[col].dtype == 'object':
                encrypted_df[col] = encrypted_df[col].apply(
                    lambda x: self.cipher.encrypt(str(x).encode()).decode() if pd.notnull(x) else x
                )
        
        self._log_audit('ENCRYPT', f'Şifrelenen sütunlar: {columns}')
        return encrypted_df
    
    def decrypt_dataframe(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """DataFrame sütunlarının şifresini çöz"""
        
        decrypted_df = df.copy()
        
        for col in columns:
            if col in decrypted_df.columns:
                decrypted_df[col] = decrypted_df[col].apply(
                    lambda x: self.cipher.decrypt(x.encode()).decode() if pd.notnull(x) and isinstance(x, str) else x
                )
        
        self._log_audit('DECRYPT', f'Çözülen sütunlar: {columns}')
        return decrypted_df
    
    def generate_api_key(self) -> str:
        """API anahtarı üret"""
        return secrets.token_urlsafe(32)
    
    def verify_api_key(self, api_key: str) -> bool:
        """API anahtarını doğrula"""
        # Implement actual validation logic
        return len(api_key) >= 32
    
    def _log_audit(self, action: str, details: str):
        """Denetim kaydı oluştur"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': 'system',
            'action': action,
            'details': details,
            'ip': 'localhost'
        }
        self.audit_log.append(log_entry)
        
    def get_audit_log(self) -> pd.DataFrame:
        """Denetim kayıtlarını getir"""
        return pd.DataFrame(self.audit_log)

# ============================================================================
# 12. ANA UYGULAMA SINIFI
# ============================================================================

class PharmaIntelligenceApp:
    """
    PharmaIntelligence Pro - Ana uygulama sınıfı
    Tüm bileşenleri entegre eden kurumsal seviye analitik platformu
    """
    
    def __init__(self):
        # Uygulama yapılandırması
        self.config = config
        
        # Motorları başlat
        self.data_engine = AdvancedDataEngine()
        self.dl_engine = DeepLearningEngine()
        self.automl_engine = AutoMLEngine()
        self.viz_engine = AdvancedVisualizationEngine()
        self.intel_engine = MarketIntelligenceEngine()
        self.simulation_engine = ScenarioSimulationEngine()
        self.reporting_engine = EnterpriseReportingEngine()
        self.security_engine = SecurityEngine()
        
        # Oturum durumu
        self.session_state = st.session_state
        
    def initialize_session_state(self):
        """Oturum durumunu başlat"""
        
        if 'data' not in self.session_state:
            self.session_state.data = None
            self.session_state.filtered_data = None
            self.session_state.metrics = {}
            self.session_state.insights = []
            self.session_state.forecast = None
            self.session_state.anomalies = None
            self.session_state.clusters = None
            self.session_state.active_filters = {}
            self.session_state.simulation_results = {}
            self.session_state.security_enabled = False
    
    def render_header(self):
        """Üst bilgi alanını render et"""
        
        st.markdown(PROFESYONEL_CSS, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="animate-fadeInUp">
            <h1 class="pharma-title">
                💊 {self.config.APP_NAME}
                <span style="font-size: 1rem; margin-left: 1rem;" class="pharma-badge pharma-badge-primary">
                    v{self.config.APP_VERSION}
                </span>
            </h1>
            <p class="pharma-subtitle">
                Yapay zeka destekli tahminleme, derin öğrenme tabanlı anomali tespiti,
                çok boyutlu segmentasyon ve kurumsal raporlama ile gelişmiş ilaç pazarı analitiği.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Yan çubuğu render et"""
        
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <span style="font-size: 3rem;">💊</span>
                <h3 style="color: #f8fafc; margin-top: 0.5rem;">KONTROL PANELİ</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # === VERİ YÜKLEME ===
            with st.expander("📁 VERİ YÜKLEME", expanded=True):
                uploaded_file = st.file_uploader(
                    "Excel/CSV/Parquet Dosyası",
                    type=['xlsx', 'xls', 'csv', 'parquet'],
                    help="1M+ satırı destekler, paralel işleme ile optimize edilmiştir",
                    key='file_uploader'
                )
                
                if uploaded_file:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        sample_size = st.number_input(
                            "Örnek boyut",
                            min_value=1000,
                            max_value=100000,
                            value=10000,
                            step=1000,
                            help="Büyük veri setleri için örnekleme"
                        )
                    
                    with col2:
                        use_dask = st.checkbox(
                            "Dask ile yükle",
                            value=True,
                            help="Büyük veri setleri için paralel işleme"
                        )
                    
                    if st.button(
                        "🚀 VERİYİ YÜKLE",
                        type="primary",
                        use_container_width=True
                    ):
                        with st.spinner("⚡ Veri işleniyor..."):
                            df = self.data_engine.load_large_dataset(
                                uploaded_file,
                                sample_size=None if use_dask else sample_size,
                                use_dask=use_dask
                            )
                            
                            if df is not None:
                                self.session_state.data = df
                                self.session_state.filtered_data = df.copy()
                                
                                # Temel metrikleri hesapla
                                self.session_state.metrics = self.calculate_basic_metrics(df)
                                
                                # İçgörüleri üret
                                self.session_state.insights = self.generate_insights(df)
                                
                                st.success(f"""
                                ✅ **Veri başarıyla yüklendi!**
                                - {len(df):,} satır
                                - {len(df.columns)} sütun
                                """)
                                
                                st.rerun()
            
            # === FİLTRELEME ===
            if self.session_state.data is not None:
                st.markdown("---")
                with st.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
                    self.render_filters()
            
            # === GÜVENLİK ===
            st.markdown("---")
            with st.expander("🔒 KURUMSAL GÜVENLİK", expanded=False):
                security_enabled = st.checkbox(
                    "Veri şifrelemeyi aktifleştir",
                    value=self.session_state.get('security_enabled', False),
                    help="Hassas veriler için uçtan uca şifreleme"
                )
                
                if security_enabled != self.session_state.get('security_enabled', False):
                    self.session_state.security_enabled = security_enabled
                    
                    if security_enabled and self.session_state.data is not None:
                        encrypt_cols = st.multiselect(
                            "Şifrelenecek sütunlar",
                            options=self.session_state.data.select_dtypes(include=['object']).columns,
                            default=[]
                        )
                        
                        if encrypt_cols:
                            self.session_state.data = self.security_engine.encrypt_dataframe(
                                self.session_state.data,
                                encrypt_cols
                            )
                            st.success(f"✅ {len(encrypt_cols)} sütun şifrelendi")
            
            # === SİSTEM DURUMU ===
            st.markdown("---")
            with st.expander("⚙️ SİSTEM DURUMU", expanded=False):
                if self.session_state.data is not None:
                    memory_usage = self.session_state.data.memory_usage(deep=True).sum() / 1024 / 1024
                    st.metric("Bellek kullanımı", f"{memory_usage:.1f} MB")
                    st.metric("Satır sayısı", f"{len(self.session_state.data):,}")
                    st.metric("Sütun sayısı", len(self.session_state.data.columns))
                
                st.metric("Sürüm", self.config.APP_VERSION)
                st.metric("Build", self.config.APP_BUILD)
                
                if st.button("🔄 Oturumu Sıfırla", use_container_width=True):
                    for key in list(self.session_state.keys()):
                        del self.session_state[key]
                    st.rerun()
    
    def render_filters(self):
        """Gelişmiş filtreleme arayüzü"""
        
        df = self.session_state.data
        
        # Global arama
        search_term = st.text_input(
            "🔎 Global arama",
            placeholder="Molekül, şirket, ülke...",
            help="Tüm sütunlarda ara"
        )
        
        filter_config = {}
        
        # Kategorik filtreler
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col in self.config.REQUIRED_COLUMNS]
        
        for col in categorical_cols[:5]:  # Limit for UI
            unique_values = sorted(df[col].dropna().unique())
            
            if len(unique_values) > 0:
                selected = st.multiselect(
                    f"{col}",
                    options=unique_values,
                    default=[],
                    key=f"filter_{col}"
                )
                
                if selected:
                    filter_config[col] = selected
        
        # Sayısal filtreler
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        sales_cols = [col for col in numeric_cols if 'Sales_' in col or 'Satış_' in col]
        
        if sales_cols:
            latest_sales = sales_cols[-1]
            min_val = float(df[latest_sales].min())
            max_val = float(df[latest_sales].max())
            
            if min_val < max_val:
                sales_range = st.slider(
                    f"Satış aralığı ({latest_sales})",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                
                filter_config['sales_range'] = (sales_range, latest_sales)
        
        # Filtreleme butonları
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Uygula", use_container_width=True):
                with st.spinner("Filtreler uygulanıyor..."):
                    filtered = df.copy()
                    
                    # Global search
                    if search_term:
                        mask = pd.Series(False, index=filtered.index)
                        for col in filtered.columns:
                            try:
                                mask |= filtered[col].astype(str).str.contains(
                                    search_term, case=False, na=False
                                )
                            except:
                                continue
                        filtered = filtered[mask]
                    
                    # Categorical filters
                    for col, values in filter_config.items():
                        if col in filtered.columns:
                            filtered = filtered[filtered[col].isin(values)]
                    
                    # Sales range filter
                    if 'sales_range' in filter_config:
                        (min_val, max_val), col = filter_config['sales_range']
                        filtered = filtered[
                            (filtered[col] >= min_val) &
                            (filtered[col] <= max_val)
                        ]
                    
                    self.session_state.filtered_data = filtered
                    self.session_state.active_filters = filter_config
                    
                    st.success(f"✅ {len(filtered):,} satır gösteriliyor")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Temizle", use_container_width=True):
                self.session_state.filtered_data = self.session_state.data.copy()
                self.session_state.active_filters = {}
                st.success("✅ Filtreler temizlendi")
                st.rerun()
    
    def calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Temel pazar metriklerini hesapla"""
        
        metrics = {}
        
        try:
            metrics['Total_Rows'] = len(df)
            metrics['Total_Columns'] = len(df.columns)
            
            # Sales metrics
            sales_cols = [col for col in df.columns if 'Sales_' in col or 'Satış_' in col]
            if sales_cols:
                latest_sales = sales_cols[-1]
                metrics['Total_Market_Value'] = float(df[latest_sales].sum())
                metrics['Average_Sales'] = float(df[latest_sales].mean())
                metrics['Median_Sales'] = float(df[latest_sales].median())
                
                # Extract year
                year_match = re.search(r'\b(20\d{2})\b', latest_sales)
                if year_match:
                    metrics['Latest_Year'] = int(year_match.group(1))
            
            # Company metrics
            if 'Company' in df.columns or 'Sirket' in df.columns:
                company_col = 'Company' if 'Company' in df.columns else 'Sirket'
                metrics['Total_Companies'] = df[company_col].nunique()
            
            # Country metrics
            if 'Country' in df.columns or 'Ulke' in df.columns:
                country_col = 'Country' if 'Country' in df.columns else 'Ulke'
                metrics['Country_Coverage'] = df[country_col].nunique()
            
            # Molecule metrics
            if 'Molecule' in df.columns or 'Molekul' in df.columns:
                molecule_col = 'Molecule' if 'Molecule' in df.columns else 'Molekul'
                metrics['Unique_Molecules'] = df[molecule_col].nunique()
            
            # International product metrics
            if 'International_Product' in df.columns or 'Uluslararasi_Urun' in df.columns:
                intl_col = 'International_Product' if 'International_Product' in df.columns else 'Uluslararasi_Urun'
                metrics['International_Products'] = int(df[intl_col].sum())
                metrics['Local_Products'] = len(df) - metrics['International_Products']
            
        except Exception as e:
            self.data_engine.logger.error(f"Metrik hesaplama hatası: {str(e)}")
        
        return metrics
    
    def generate_insights(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Stratejik içgörüler üret"""
        
        insights = []
        
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col or 'Satış_' in col]
            
            if sales_cols:
                latest_sales = sales_cols[-1]
                total_market = df[latest_sales].sum()
                
                # Top products
                if 'Molecule' in df.columns or 'Molekul' in df.columns:
                    molecule_col = 'Molecule' if 'Molecule' in df.columns else 'Molekul'
                    top_molecules = df.groupby(molecule_col)[latest_sales].sum().nlargest(5)
                    top_share = (top_molecules.sum() / total_market * 100) if total_market > 0 else 0
                    
                    insights.append({
                        'tur': 'basarili',
                        'baslik': '🏆 En İyi 5 Molekül',
                        'aciklama': f"En iyi 5 molekül toplam pazarın %{top_share:.1f}'ini oluşturuyor."
                    })
                
                # Top companies
                if 'Company' in df.columns or 'Sirket' in df.columns:
                    company_col = 'Company' if 'Company' in df.columns else 'Sirket'
                    top_companies = df.groupby(company_col)[latest_sales].sum().nlargest(3)
                    company_names = ', '.join(top_companies.index[:3])
                    
                    insights.append({
                        'tur': 'bilgi',
                        'baslik': '🏢 Pazar Liderleri',
                        'aciklama': f"En büyük 3 şirket: {company_names}"
                    })
                
                # Growth analysis
                if len(sales_cols) >= 2:
                    prev_sales = sales_cols[-2]
                    growth = ((df[latest_sales].sum() - df[prev_sales].sum()) / 
                             df[prev_sales].sum() * 100) if df[prev_sales].sum() > 0 else 0
                    
                    growth_trend = 'büyüme' if growth > 0 else 'daralma'
                    insights.append({
                        'tur': 'uyari' if growth < 0 else 'basarili',
                        'baslik': '📈 Pazar Trendi',
                        'aciklama': f"Pazar %{abs(growth):.1f} oranında {growth_trend} gösteriyor."
                    })
            
            # International analysis
            if 'International_Product' in df.columns or 'Uluslararasi_Urun' in df.columns:
                intl_col = 'International_Product' if 'International_Product' in df.columns else 'Uluslararasi_Urun'
                intl_count = df[intl_col].sum()
                intl_percentage = (intl_count / len(df) * 100) if len(df) > 0 else 0
                
                insights.append({
                    'tur': 'uluslararasi',
                    'baslik': '🌍 Uluslararası Ürünler',
                    'aciklama': f"Veri setinde %{intl_percentage:.1f} oranında Uluslararası Ürün bulunuyor."
                })
            
        except Exception as e:
            self.data_engine.logger.error(f"İçgörü üretme hatası: {str(e)}")
        
        return insights
    
    def render_dashboard_tab(self):
        """Ana gösterge paneli sekmesi"""
        
        df = self.session_state.filtered_data
        metrics = self.session_state.metrics
        
        st.markdown('<h2 class="pharma-section-title">📊 Yönetici Gösterge Paneli</h2>', unsafe_allow_html=True)
        
        # === METRİK KARTLARI ===
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="pharma-metric-card">
                <div class="pharma-metric-label">📦 TOPLAM ÜRÜN</div>
                <div class="pharma-metric-value">{len(df):,}</div>
                <div class="pharma-metric-trend">
                    <span class="pharma-badge pharma-badge-info">Aktif</span>
                    <span>Filtrelenmiş veri</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            market_value = metrics.get('Total_Market_Value', 0)
            st.markdown(f"""
            <div class="pharma-metric-card">
                <div class="pharma-metric-label">💰 PAZAR DEĞERİ</div>
                <div class="pharma-metric-value">${market_value/1e6:.1f}M</div>
                <div class="pharma-metric-trend">
                    <span class="pharma-badge pharma-badge-success">Yıllık</span>
                    <span>{metrics.get('Latest_Year', '2024')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            growth_rate = metrics.get('Average_Growth_Rate', 0)
            growth_class = 'success' if growth_rate > 0 else 'danger' if growth_rate < 0 else 'warning'
            st.markdown(f"""
            <div class="pharma-metric-card">
                <div class="pharma-metric-label">📈 BÜYÜME</div>
                <div class="pharma-metric-value">%{growth_rate:.1f}</div>
                <div class="pharma-metric-trend">
                    <span class="pharma-badge pharma-badge-{growth_class}">YoY</span>
                    <span>Yıllık büyüme</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            countries = metrics.get('Country_Coverage', 0)
            st.markdown(f"""
            <div class="pharma-metric-card">
                <div class="pharma-metric-label">🌍 ÜLKE KAPSAMI</div>
                <div class="pharma-metric-value">{countries}</div>
                <div class="pharma-metric-trend">
                    <span class="pharma-badge pharma-badge-primary">Global</span>
                    <span>Pazar</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # === GÖRSELLEŞTİRMELER ===
        st.markdown('<h3 class="pharma-section-title">🎯 Pazar Analitiği</h3>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs([
            "📈 Pazar Dinamikleri",
            "🌐 Coğrafi Dağılım",
            "🔄 Korelasyon Analizi"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Satış trendi
                sales_cols = [col for col in df.columns if 'Sales_' in col or 'Satış_' in col]
                if sales_cols:
                    sales_data = []
                    for col in sorted(sales_cols):
                        year = self.viz_engine._extract_year(col)
                        if year:
                            sales_data.append({
                                'Year': year,
                                'Sales': df[col].sum()
                            })
                    
                    if sales_data:
                        sales_df = pd.DataFrame(sales_data)
                        
                        fig = px.line(
                            sales_df,
                            x='Year',
                            y='Sales',
                            markers=True,
                            title='Yıllık Pazar Büyüklüğü',
                            labels={'Sales': 'Satış (USD)', 'Year': 'Yıl'}
                        )
                        
                        fig.update_layout(
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#f8fafc'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pazar payı dağılımı
                if 'Company' in df.columns or 'Sirket' in df.columns:
                    company_col = 'Company' if 'Company' in df.columns else 'Sirket'
                    
                    if sales_cols:
                        latest_sales = sales_cols[-1]
                        company_sales = df.groupby(company_col)[latest_sales].sum().nlargest(10)
                        
                        fig = px.pie(
                            values=company_sales.values,
                            names=company_sales.index,
                            title='İlk 10 Şirket - Pazar Payı',
                            hole=0.4
                        )
                        
                        fig.update_layout(
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#f8fafc'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'Country' in df.columns or 'Ulke' in df.columns:
                country_col = 'Country' if 'Country' in df.columns else 'Ulke'
                
                if sales_cols:
                    latest_sales = sales_cols[-1]
                    country_data = df.groupby(country_col)[latest_sales].sum().reset_index()
                    country_data.columns = ['Country', 'Sales']
                    
                    fig = px.choropleth(
                        country_data,
                        locations='Country',
                        locationmode='country names',
                        color='Sales',
                        hover_name='Country',
                        color_continuous_scale='Viridis',
                        title='Global Pazar Dağılımı'
                    )
                    
                    fig.update_layout(
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f8fafc',
                        geo=dict(
                            bgcolor='rgba(0,0,0,0)',
                            lakecolor='#1e3a5f',
                            landcolor='#2d4a7a'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
            
            if len(numeric_cols) >= 2:
                fig = self.viz_engine.create_heatmap_correlation(df, list(numeric_cols))
                st.plotly_chart(fig, use_container_width=True)
        
        # === STRATEJİK İÇGÖRÜLER ===
        if self.session_state.insights:
            st.markdown('<h3 class="pharma-section-title">💡 Stratejik İçgörüler</h3>', unsafe_allow_html=True)
            
            insight_cols = st.columns(2)
            
            for i, insight in enumerate(self.session_state.insights[:6]):
                with insight_cols[i % 2]:
                    insight_class = {
                        'basarili': 'success',
                        'uyari': 'warning',
                        'tehlike': 'danger',
                        'bilgi': 'info',
                        'uluslararasi': 'primary'
                    }.get(insight['tur'], 'info')
                    
                    st.markdown(f"""
                    <div class="pharma-insight pharma-insight-{insight_class}">
                        <h4 style="margin-bottom: 0.5rem; color: #f8fafc;">{insight['baslik']}</h4>
                        <p style="color: #cbd5e1; margin: 0;">{insight['aciklama']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_advanced_analytics_tab(self):
        """Gelişmiş analitik sekmesi"""
        
        df = self.session_state.filtered_data
        
        st.markdown('<h2 class="pharma-section-title">🧠 Gelişmiş Analitik</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🤖 Derin Öğrenme Tahmini",
            "🎯 Otomatik Segmentasyon",
            "⚠️ Anomali Tespiti",
            "🎲 Monte Carlo Simülasyonu"
        ])
        
        with tab1:
            st.markdown("""
            <div class="pharma-insight pharma-insight-info">
                <h4 style="color: #f8fafc;">🧠 Derin Öğrenme Tabanlı Pazar Tahmini</h4>
                <p style="color: #cbd5e1;">
                LSTM, GRU ve Transformer modelleri ile gelecek dönem pazar tahminleri.
                Model mimarisi: Çok katmanlı bidirectional LSTM, dropout regularizasyonu,
                erken durdurma ve learning rate azaltma.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            sales_cols = [col for col in df.columns if 'Sales_' in col or 'Satış_' in col]
            
            if len(sales_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    model_type = st.selectbox(
                        "Model Mimarisi",
                        ["LSTM", "GRU", "Transformer"],
                        help="Tahmin için kullanılacak derin öğrenme modeli"
                    )
                
                with col2:
                    forecast_periods = st.slider(
                        "Tahmin Dönemi",
                        min_value=1,
                        max_value=12,
                        value=4,
                        help="Gelecek dönem sayısı (çeyreklik)"
                    )
                
                with col3:
                    sequence_length = st.slider(
                        "Sequence Uzunluğu",
                        min_value=3,
                        max_value=12,
                        value=6,
                        help="Geçmiş dönem sayısı"
                    )
                
                if st.button("🔮 TAHMİN OLUŞTUR", type="primary", use_container_width=True):
                    with st.spinner("🧠 Derin öğrenme modeli eğitiliyor..."):
                        # Veriyi hazırla
                        yearly_sales = []
                        years = []
                        
                        for col in sorted(sales_cols):
                            year = self.viz_engine._extract_year(col)
                            if year:
                                yearly_sales.append(df[col].sum())
                                years.append(year)
                        
                        sales_series = pd.Series(yearly_sales, index=years)
                        
                        # Scale the data
                        scaler = MinMaxScaler()
                        scaled_data = scaler.fit_transform(sales_series.values.reshape(-1, 1)).flatten()
                        
                        # Prepare sequences
                        X, y = self.dl_engine.prepare_sequences(
                            pd.Series(scaled_data),
                            sequence_length=sequence_length,
                            prediction_horizon=1
                        )
                        
                        if len(X) >= sequence_length * 2:
                            # Reshape for LSTM [samples, timesteps, features]
                            X = X.reshape((X.shape[0], X.shape[1], 1))
                            
                            # Split data
                            split_idx = int(len(X) * 0.8)
                            X_train, X_val = X[:split_idx], X[split_idx:]
                            y_train, y_val = y[:split_idx], y[split_idx:]
                            
                            # Train model
                            model = self.dl_engine.train_deep_learning_model(
                                X_train, y_train,
                                X_val, y_val,
                                model_type=model_type.lower(),
                                n_units=[64, 32],
                                dropout_rate=0.2
                            )
                            
                            # Generate forecast
                            last_sequence = scaled_data[-sequence_length:].reshape(sequence_length, 1)
                            predictions_scaled = self.dl_engine.predict_future(
                                model,
                                last_sequence,
                                forecast_periods
                            )
                            
                            # Inverse transform
                            predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
                            
                            # Create forecast dataframe
                            last_year = years[-1]
                            forecast_years = [last_year + i + 1 for i in range(forecast_periods)]
                            
                            forecast_df = pd.DataFrame({
                                'Yıl': forecast_years,
                                'Tahmin': predictions,
                                'Model': model_type
                            })
                            
                            self.session_state.forecast = forecast_df
                            
                            st.success(f"✅ {model_type} modeli başarıyla eğitildi!")
                
                # Display forecast
                if 'forecast' in self.session_state and self.session_state.forecast is not None:
                    st.markdown("### 📊 Tahmin Sonuçları")
                    
                    forecast_df = self.session_state.forecast
                    
                    # Combine historical and forecast
                    historical_df = pd.DataFrame({
                        'Yıl': years,
                        'Gerçek': yearly_sales,
                        'Tür': 'Tarihsel'
                    })
                    
                    forecast_display = pd.DataFrame({
                        'Yıl': forecast_df['Yıl'],
                        'Tahmin': forecast_df['Tahmin'],
                        'Tür': 'Tahmin'
                    })
                    
                    combined_df = pd.concat([
                        historical_df,
                        forecast_display.rename(columns={'Tahmin': 'Gerçek'})
                    ])
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=historical_df['Yıl'],
                        y=historical_df['Gerçek'],
                        mode='lines+markers',
                        name='Tarihsel',
                        line=dict(color='#2d7dd2', width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_display['Yıl'],
                        y=forecast_display['Tahmin'],
                        mode='lines+markers',
                        name='Tahmin',
                        line=dict(color='#2acaea', width=3, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='Derin Öğrenme ile Pazar Tahmini',
                        xaxis_title='Yıl',
                        yaxis_title='Toplam Satış (USD)',
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f8fafc'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast table
                    st.dataframe(
                        forecast_df.style.format({
                            'Tahmin': '${:,.0f}'
                        }),
                        use_container_width=True
                    )
            else:
                st.warning("⚠️ Derin öğrenme tahmini için en az 3 yıllık veri gereklidir.")
        
        with tab2:
            st.markdown("""
            <div class="pharma-insight pharma-insight-info">
                <h4 style="color: #f8fafc;">🎯 Otomatik Pazar Segmentasyonu</h4>
                <p style="color: #cbd5e1;">
                K-means, DBSCAN ve hiyerarşik kümeleme algoritmaları ile
                ürünlerin otomatik segmentasyonu ve profil analizi.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if 'Sales_' in col or 'Price_' in col][:5]
            
            if len(feature_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    n_clusters = st.slider(
                        "Küme Sayısı",
                        min_value=2,
                        max_value=8,
                        value=4,
                        help="Segmentasyon için küme sayısı"
                    )
                
                with col2:
                    algorithm = st.selectbox(
                        "Kümeleme Algoritması",
                        ["K-means", "DBSCAN", "Hierarchical"],
                        help="Segmentasyon algoritması"
                    )
                
                if st.button("🔍 SEGMENTASYONU BAŞLAT", type="primary", use_container_width=True):
                    with st.spinner("🎯 Ürünler segmentlere ayrılıyor..."):
                        # Prepare data
                        X = df[feature_cols].fillna(0)
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Apply clustering
                        if algorithm == "K-means":
                            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            labels = model.fit_predict(X_scaled)
                        elif algorithm == "DBSCAN":
                            model = DBSCAN(eps=0.5, min_samples=5)
                            labels = model.fit_predict(X_scaled)
                        else:
                            model = AgglomerativeClustering(n_clusters=n_clusters)
                            labels = model.fit_predict(X_scaled)
                        
                        # Add cluster labels to dataframe
                        clustered_df = df.copy()
                        clustered_df['Segment'] = labels
                        
                        # Calculate silhouette score
                        if len(set(labels)) > 1 and -1 not in set(labels):  # DBSCAN might have noise
                            try:
                                silhouette = silhouette_score(X_scaled, labels)
                                st.metric("Silhouette Skoru", f"{silhouette:.3f}")
                            except:
                                pass
                        
                        self.session_state.clusters = clustered_df
                        st.success(f"✅ {len(set(labels)) - (1 if -1 in labels else 0)} segment oluşturuldu!")
                
                # Display clustering results
                if 'clusters' in self.session_state and self.session_state.clusters is not None:
                    clustered_df = self.session_state.clusters
                    
                    # PCA for visualization
                    X_scaled = StandardScaler().fit_transform(clustered_df[feature_cols].fillna(0))
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    viz_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Segment': clustered_df['Segment'].astype(str)
                    })
                    
                    if 'Molecule' in clustered_df.columns:
                        viz_df['Molecule'] = clustered_df['Molecule']
                    elif 'Molekul' in clustered_df.columns:
                        viz_df['Molecule'] = clustered_df['Molekul']
                    
                    fig = px.scatter(
                        viz_df,
                        x='PC1',
                        y='PC2',
                        color='Segment',
                        hover_data=['Molecule'] if 'Molecule' in viz_df.columns else None,
                        title='PCA ile Segment Görselleştirmesi',
                        labels={'PC1': 'Birinci Bileşen', 'PC2': 'İkinci Bileşen'}
                    )
                    
                    fig.update_layout(
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f8fafc'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Segment statistics
                    st.markdown("### 📊 Segment İstatistikleri")
                    
                    segment_stats = []
                    
                    for segment in sorted(clustered_df['Segment'].unique()):
                        segment_df = clustered_df[clustered_df['Segment'] == segment]
                        
                        stats = {
                            'Segment': f'Segment {segment}' if segment != -1 else 'Gürültü',
                            'Ürün Sayısı': len(segment_df)
                        }
                        
                        if sales_cols:
                            latest_sales = sales_cols[-1]
                            stats['Ortalama Satış'] = segment_df[latest_sales].mean()
                        
                        segment_stats.append(stats)
                    
                    stats_df = pd.DataFrame(segment_stats)
                    
                    if 'Ortalama Satış' in stats_df.columns:
                        stats_df['Ortalama Satış'] = stats_df['Ortalama Satış'].apply(
                            lambda x: f'${x:,.0f}' if pd.notnull(x) else 'N/A'
                        )
                    
                    st.dataframe(stats_df, use_container_width=True)
            else:
                st.warning("⚠️ Segmentasyon için yeterli sayısal sütun bulunamadı.")
        
        with tab3:
            st.markdown("""
            <div class="pharma-insight pharma-insight-warning">
                <h4 style="color: #f8fafc;">⚠️ Anomali Tespiti</h4>
                <p style="color: #cbd5e1;">
                Isolation Forest algoritması ile pazardaki aykırı değerlerin,
                olağandışı kalıpların ve potansiyel risklerin tespiti.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if 'Sales_' in col or 'Price_' in col][:4]
            
            if len(feature_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    contamination = st.slider(
                        "Anomali Oranı",
                        min_value=0.01,
                        max_value=0.3,
                        value=0.1,
                        step=0.01,
                        format="%f",
                        help="Veri setindeki tahmini anomali oranı"
                    )
                
                with col2:
                    n_estimators = st.slider(
                        "Ağaç Sayısı",
                        min_value=50,
                        max_value=300,
                        value=100,
                        step=50,
                        help="Isolation Forest'taki ağaç sayısı"
                    )
                
                if st.button("🔍 ANOMALİ TESPİTİ", type="primary", use_container_width=True):
                    with st.spinner("⚠️ Anomaliler tespit ediliyor..."):
                        X = df[feature_cols].fillna(0)
                        
                        model = IsolationForest(
                            contamination=contamination,
                            n_estimators=n_estimators,
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        predictions = model.fit_predict(X)
                        scores = model.score_samples(X)
                        
                        anomaly_df = df.copy()
                        anomaly_df['Anomali'] = predictions
                        anomaly_df['Anomali_Skoru'] = scores
                        anomaly_df['Risk_Seviyesi'] = pd.cut(
                            scores,
                            bins=[-np.inf, -0.5, -0.2, np.inf],
                            labels=['Yüksek Risk', 'Orta Risk', 'Düşük Risk']
                        )
                        
                        self.session_state.anomalies = anomaly_df
                        
                        n_anomalies = (predictions == -1).sum()
                        st.success(f"✅ {n_anomalies} anomali tespit edildi (%{n_anomalies/len(df)*100:.1f})")
                
                # Display anomalies
                if 'anomalies' in self.session_state and self.session_state.anomalies is not None:
                    anomaly_df = self.session_state.anomalies
                    
                    # Visualization
                    if len(feature_cols) >= 2:
                        fig = px.scatter(
                            anomaly_df,
                            x=feature_cols[0],
                            y=feature_cols[1],
                            color='Risk_Seviyesi',
                            size=abs(anomaly_df['Anomali_Skoru']),
                            hover_data=['Molecule'] if 'Molecule' in anomaly_df.columns else None,
                            title='Anomali Tespiti - Risk Dağılımı',
                            color_discrete_map={
                                'Yüksek Risk': '#eb5757',
                                'Orta Risk': '#f2c94c',
                                'Düşük Risk': '#2dd2a3'
                            }
                        )
                        
                        fig.update_layout(
                            height=500,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#f8fafc'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # High risk items
                    high_risk = anomaly_df[anomaly_df['Risk_Seviyesi'] == 'Yüksek Risk']
                    
                    if len(high_risk) > 0:
                        st.markdown("### ⚠️ Yüksek Riskli Ürünler")
                        
                        display_cols = ['Molecule', 'Company', 'Risk_Seviyesi'] + feature_cols[:2]
                        display_cols = [col for col in display_cols if col in high_risk.columns]
                        
                        st.dataframe(
                            high_risk[display_cols].sort_values('Anomali_Skoru').head(20),
                            use_container_width=True
                        )
            else:
                st.warning("⚠️ Anomali tespiti için yeterli sayısal sütun bulunamadı.")
        
        with tab4:
            st.markdown("""
            <div class="pharma-insight pharma-insight-info">
                <h4 style="color: #f8fafc;">🎲 Monte Carlo Simülasyonu</h4>
                <p style="color: #cbd5e1;">
                Stokastik modelleme ile belirsizlik altında pazar tahmini,
                risk analizi ve senaryo değerlendirmesi.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            sales_cols = [col for col in df.columns if 'Sales_' in col or 'Satış_' in col]
            
            if len(sales_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    n_simulations = st.number_input(
                        "Simülasyon Sayısı",
                        min_value=100,
                        max_value=10000,
                        value=1000,
                        step=100,
                        help="Monte Carlo simülasyon sayısı"
                    )
                
                with col2:
                    forecast_periods = st.slider(
                        "Tahmin Dönemi",
                        min_value=1,
                        max_value=24,
                        value=12,
                        help="Gelecek dönem sayısı (ay)"
                    )
                
                with col3:
                    confidence = st.slider(
                        "Güven Aralığı",
                        min_value=0.8,
                        max_value=0.99,
                        value=0.95,
                        step=0.01,
                        format="%f",
                        help="Tahmin güven aralığı"
                    )
                
                if st.button("🎲 SİMÜLASYONU BAŞLAT", type="primary", use_container_width=True):
                    with st.spinner("🎲 Monte Carlo simülasyonu çalıştırılıyor..."):
                        latest_sales = sales_cols[-1]
                        sales_data = df[latest_sales]
                        
                        simulation_results = self.simulation_engine.monte_carlo_forecast(
                            sales_data,
                            n_simulations=n_simulations,
                            forecast_periods=forecast_periods,
                            confidence_level=confidence
                        )
                        
                        self.session_state.simulation_results = simulation_results
                        st.success(f"✅ {n_simulations} simülasyon tamamlandı!")
                
                # Display simulation results
                if self.session_state.simulation_results:
                    results = self.session_state.simulation_results
                    
                    # Visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=list(range(len(sales_data))),
                        y=sales_data.values,
                        mode='lines',
                        name='Tarihsel',
                        line=dict(color='#2d7dd2', width=3)
                    ))
                    
                    # Forecast with confidence interval
                    forecast_idx = list(range(len(sales_data), len(sales_data) + forecast_periods))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_idx,
                        y=results['mean_forecast'],
                        mode='lines',
                        name='Tahmin',
                        line=dict(color='#2acaea', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_idx + forecast_idx[::-1],
                        y=results['upper_bound'].tolist() + results['lower_bound'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(42, 202, 234, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'%{confidence*100:.0f} Güven Aralığı'
                    ))
                    
                    fig.update_layout(
                        title='Monte Carlo Simülasyonu - Pazar Tahmini',
                        xaxis_title='Dönem',
                        yaxis_title='Satış (USD)',
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f8fafc'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Value at Risk (VaR 95%)",
                            f"${results['var_95']:,.0f}",
                            "Risk limiti"
                        )
                    
                    with col2:
                        st.metric(
                            "Conditional VaR",
                            f"${results['cvar_95']:,.0f}",
                            "Beklenen kayıp"
                        )
                    
                    with col3:
                        st.metric(
                            "Tahmin Belirsizliği",
                            f"%{(results['std_forecast'].mean() / results['mean_forecast'].mean() * 100):.1f}",
                            "CV"
                        )
            else:
                st.warning("⚠️ Monte Carlo simülasyonu için en az 2 yıllık veri gereklidir.")
    
    def render_reporting_tab(self):
        """Raporlama sekmesi"""
        
        df = self.session_state.filtered_data
        metrics = self.session_state.metrics
        insights = self.session_state.insights
        
        st.markdown('<h2 class="pharma-section-title">📑 Kurumsal Raporlama</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs([
            "📊 Rapor Oluştur",
            "📋 Rapor Şablonları",
            "📤 Dışa Aktarım"
        ])
        
        with tab1:
            st.markdown("""
            <div class="pharma-insight pharma-insight-info">
                <h4 style="color: #f8fafc;">📊 Yönetici Özeti Raporu</h4>
                <p style="color: #cbd5e1;">
                Kurumsal standartlarda, profesyonel tasarımlı yönetici özeti raporu.
                PDF, HTML ve Excel formatlarında dışa aktarım.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 PDF Raporu", use_container_width=True):
                    with st.spinner("PDF raporu oluşturuluyor..."):
                        pdf_buffer = self.reporting_engine.generate_pdf_report(df, metrics, insights)
                        
                        if pdf_buffer:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            st.download_button(
                                label="⬇️ PDF'i İndir",
                                data=pdf_buffer,
                                file_name=f"pharma_rapor_{timestamp}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
            
            with col2:
                if st.button("🌐 HTML Raporu", use_container_width=True):
                    with st.spinner("HTML raporu oluşturuluyor..."):
                        html_report = self.reporting_engine.generate_executive_summary(df, metrics, insights)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        st.download_button(
                            label="⬇️ HTML'i İndir",
                            data=html_report.encode('utf-8'),
                            file_name=f"pharma_rapor_{timestamp}.html",
                            mime="text/html",
                            use_container_width=True
                        )
            
            with col3:
                if st.button("📊 Excel Raporu", use_container_width=True):
                    with st.spinner("Excel raporu oluşturuluyor..."):
                        buffer = BytesIO()
                        
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            # Executive summary
                            summary_df = pd.DataFrame([
                                ['Rapor Tarihi', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                                ['Toplam Ürün', len(df)],
                                ['Toplam Pazar Değeri', f"${metrics.get('Total_Market_Value', 0)/1e6:.1f}M"],
                                ['Ortalama Büyüme', f"%{metrics.get('Average_Growth_Rate', 0):.1f}"],
                                ['Ülke Sayısı', metrics.get('Country_Coverage', 0)],
                                ['Şirket Sayısı', metrics.get('Total_Companies', 0)]
                            ], columns=['Metrik', 'Değer'])
                            
                            summary_df.to_excel(writer, sheet_name='Yönetici Özeti', index=False)
                            
                            # Raw data
                            df.to_excel(writer, sheet_name='Ham Veri', index=False)
                            
                            # Insights
                            if insights:
                                insights_df = pd.DataFrame(insights)
                                insights_df.to_excel(writer, sheet_name='İçgörüler', index=False)
                            
                            # Adjust column widths
                            for sheet_name in writer.sheets:
                                worksheet = writer.sheets[sheet_name]
                                worksheet.set_column('A:A', 30)
                                worksheet.set_column('B:B', 30)
                        
                        buffer.seek(0)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        
                        st.download_button(
                            label="⬇️ Excel'i İndir",
                            data=buffer,
                            file_name=f"pharma_rapor_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
        
        with tab2:
            st.markdown("""
            <div class="pharma-insight pharma-insight-info">
                <h4 style="color: #f8fafc;">📋 Rapor Şablonları</h4>
                <p style="color: #cbd5e1;">
                Özelleştirilmiş rapor şablonları ile hızlı ve tutarlı raporlama.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            template_options = [
                "Yönetici Özeti (Varsayılan)",
                "Detaylı Pazar Analizi",
                "Rekabet İstihbarat Raporu",
                "Ürün Performans Karnesi",
                "Risk ve Anomali Raporu",
                "Özel Şablon"
            ]
            
            selected_template = st.selectbox("Rapor Şablonu Seçin", template_options)
            
            if selected_template == "Detaylı Pazar Analizi":
                st.info("""
                **Şablon içeriği:**
                - Pazar büyüklüğü ve büyüme trendleri
                - Segment bazlı performans analizi
                - Coğrafi dağılım ve penetrasyon
                - Fiyat-hacim analizi
                - Tahmin ve projeksiyonlar
                """)
            
            elif selected_template == "Rekabet İstihbarat Raporu":
                st.info("""
                **Şablon içeriği:**
                - Pazar payı dağılımı ve HHI indeksi
                - Rakip şirket profilleri
                - Ürün bazlı rekabet analizi
                - SWOT analizi
                - Rekabet stratejileri
                """)
            
            elif selected_template == "Ürün Performans Karnesi":
                st.info("""
                **Şablon içeriği:**
                - Ürün bazlı satış ve büyüme metrikleri
                - Fiyat pozisyonlandırma analizi
                - Molekül portföy değerlendirmesi
                - Uluslararası/yerel ürün karşılaştırması
                - Performans skor kartı
                """)
            
            elif selected_template == "Risk ve Anomali Raporu":
                st.info("""
                **Şablon içeriği:**
                - Anomali tespit özeti
                - Yüksek riskli ürünler listesi
                - Risk skorlaması ve kategorizasyon
                - Erken uyarı göstergeleri
                - Aksiyon önerileri
                """)
            
            elif selected_template == "Özel Şablon":
                st.info("Özel rapor şablonu oluşturmak için kurumsal destek ekibimizle iletişime geçin: enterprise@pharmaintelligence.com")
        
        with tab3:
            st.markdown("""
            <div class="pharma-insight pharma-insight-info">
                <h4 style="color: #f8fafc;">📤 Veri Dışa Aktarım</h4>
                <p style="color: #cbd5e1;">
                Filtrelenmiş veriyi çeşitli formatlarda dışa aktarın.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📄 CSV", use_container_width=True):
                    csv = df.to_csv(index=False).encode('utf-8')
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    st.download_button(
                        label="⬇️ CSV İndir",
                        data=csv,
                        file_name=f"pharma_veri_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📊 Excel", use_container_width=True):
                    buffer = BytesIO()
                    df.to_excel(buffer, index=False, engine='xlsxwriter')
                    buffer.seek(0)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    st.download_button(
                        label="⬇️ Excel İndir",
                        data=buffer,
                        file_name=f"pharma_veri_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            with col3:
                if st.button("🗄️ Parquet", use_container_width=True):
                    buffer = BytesIO()
                    df.to_parquet(buffer, index=False)
                    buffer.seek(0)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    st.download_button(
                        label="⬇️ Parquet İndir",
                        data=buffer,
                        file_name=f"pharma_veri_{timestamp}.parquet",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
            
            with col4:
                if st.button("📋 JSON", use_container_width=True):
                    json_str = df.to_json(orient='records', date_format='iso')
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    st.download_button(
                        label="⬇️ JSON İndir",
                        data=json_str,
                        file_name=f"pharma_veri_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
    def run(self):
        """Ana uygulamayı çalıştır"""
        
        # Initialize session state
        self.initialize_session_state()
        
        # Page config
        st.set_page_config(
            page_title=self.config.APP_NAME,
            page_icon="💊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content
        if self.session_state.data is None:
            # Welcome screen
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("""
                <div style="text-align: center; padding: 3rem;">
                    <div style="font-size: 6rem; margin-bottom: 1rem;">💊</div>
                    <h2 style="color: #f8fafc; margin-bottom: 1rem;">
                        PharmaIntelligence Pro'ya Hoş Geldiniz
                    </h2>
                    <p style="color: #cbd5e1; font-size: 1.1rem; margin-bottom: 2rem;">
                        Sol panelden bir veri seti yükleyerek yapay zeka destekli 
                        analitik platformunu kullanmaya başlayın.
                    </p>
                    <div style="display: flex; justify-content: center; gap: 1rem;">
                        <span class="pharma-badge pharma-badge-primary">Derin Öğrenme</span>
                        <span class="pharma-badge pharma-badge-success">AutoML</span>
                        <span class="pharma-badge pharma-badge-warning">Monte Carlo</span>
                        <span class="pharma-badge pharma-badge-info">NLP</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Main dashboard tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Gösterge Paneli",
                "🧠 Gelişmiş Analitik",
                "📑 Raporlama",
                "🔒 Güvenlik"
            ])
            
            with tab1:
                self.render_dashboard_tab()
            
            with tab2:
                self.render_advanced_analytics_tab()
            
            with tab3:
                self.render_reporting_tab()
            
            with tab4:
                st.markdown('<h2 class="pharma-section-title">🔒 Kurumsal Güvenlik</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="pharma-card">
                        <h3 style="color: #f8fafc; margin-bottom: 1rem;">🔐 Veri Şifreleme</h3>
                        <p style="color: #cbd5e1;">
                        AES-256 bit şifreleme ile hassas verilerinizin korunması.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="pharma-card">
                        <h3 style="color: #f8fafc; margin-bottom: 1rem;">📋 Denetim Kayıtları</h3>
                        <p style="color: #cbd5e1;">
                        Tüm işlemlerin izlenebilirliği ve detaylı log kayıtları.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Audit log
                if st.button("📋 Denetim Kayıtlarını Göster"):
                    audit_df = self.security_engine.get_audit_log()
                    if len(audit_df) > 0:
                        st.dataframe(audit_df, use_container_width=True)
                    else:
                        st.info("Henüz denetim kaydı bulunmuyor.")

# ============================================================================
# 13. UYGULAMA GİRİŞ NOKTASI
# ============================================================================

if __name__ == "__main__":
    try:
        # Bellek yönetimini aktifleştir
        gc.enable()
        
        # Uygulamayı başlat
        app = PharmaIntelligenceApp()
        app.run()
        
    except Exception as e:
        st.error(f"❌ Uygulama hatası: {str(e)}")
        st.error("Detaylı hata bilgisi:")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Uygulamayı Yeniden Başlat", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()




