"""
PharmaIntelligence Pro v8.0 - ProdPack Derinlik Analizi
Enterprise Karar Destek Platformu
Versiyon: 8.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
import re
import json
import gc
import sys
import os
import traceback
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Generator
from collections import defaultdict, Counter, OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import hashlib
import pickle
import base64
import math
import random
from pathlib import Path

# ================================================
# 1. BİLİMSEL HESAPLAMA VE İSTATİSTİK
# ================================================
from scipy import stats
from scipy.stats import zscore, pearsonr, spearmanr, kendalltau, norm, chi2
from scipy.stats import shapiro, kstest, anderson, jarque_bera
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.signal import savgol_filter, detrend
from scipy.optimize import curve_fit, minimize

# ================================================
# 2. MAKİNE ÖĞRENMESİ VE ÖN İŞLEME
# ================================================
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer,
    QuantileTransformer, LabelEncoder, OneHotEncoder, OrdinalEncoder,
    KBinsDiscretizer, PolynomialFeatures
)
from sklearn.decomposition import (
    PCA, KernelPCA, TruncatedSVD, FactorAnalysis, NMF, FastICA,
    LatentDirichletAllocation
)
from sklearn.manifold import (
    TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
)
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, Birch, OPTICS,
    SpectralClustering, MeanShift, estimate_bandwidth
)
from sklearn.ensemble import (
    IsolationForest, RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor, RANSACRegressor, TheilSenRegressor
)
from sklearn.svm import SVR, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors, KNeighborsRegressor
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.feature_selection import (
    RFE, RFECV, SelectKBest, SelectPercentile, mutual_info_regression,
    f_regression, VarianceThreshold, SelectFromModel
)
from sklearn.model_selection import (
    TimeSeriesSplit, KFold, cross_val_score, GridSearchCV,
    RandomizedSearchCV, train_test_split, learning_curve
)
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.utils import resample, shuffle
import umap

# ================================================
# 3. ZAMAN SERİSİ VE TAHMİNLEME
# ================================================
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import WLS

# ================================================
# 4. DERİN ÖĞRENME VE AI (OPSİYONEL)
# ================================================
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False
    st.warning("pmdarima kurulu değil...")

# ================================================
# 5. RAPORLAMA VE GÖRSELLEŞTİRME
# ================================================
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Wedge
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

try:
    from reportlab.lib.pagesizes import letter, A4, landscape
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
        Image as RLImage, PageBreak, KeepTogether
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ================================================
# 6. WEB SCRAPING VE API (İLERİ)
# ================================================
import requests
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote
import xml.etree.ElementTree as ET
import json
import csv

# ================================================
# 7. PERFORMANS VE PARALEL İŞLEME
# ================================================
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial, lru_cache
from itertools import product, combinations, permutations
from joblib import Parallel, delayed, dump, load
import psutil
import platform

# ================================================
# 8. GÜVENLİK VE ŞİFRELEME
# ================================================
import secrets
import hashlib
import hmac
import binascii
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

# ================================================
# UYARILARI KAPAT
# ================================================
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
sns.set_style('darkgrid')
plt.rcParams['figure.facecolor'] = '#0A1A2F'
plt.rcParams['axes.facecolor'] = '#1E3A5F'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# ================================================
# 9. ENUM VE VERİ SINIFLARI (GENİŞLETİLMİŞ)
# ================================================

class RiskLevel(Enum):
    """Risk seviyeleri"""
    KRITIK = "🔴 Kritik Risk"
    YUKSEK = "🟠 Yüksek Risk"
    ORTA = "🟡 Orta Risk"
    DUSUK = "🟢 Düşük Risk"
    NORMAL = "✅ Normal"
    GUVENLI = "🛡️ Güvenli"

class GrowthCategory(Enum):
    """Büyüme kategorileri"""
    HIPER = "🚀 Hiper Büyüme (>%100)"
    COK_YUKSEK = "📈 Çok Yüksek (%50-100)"
    YUKSEK = "📊 Yüksek (%20-50)"
    ORTA = "📉 Orta (%5-20)"
    DURGUN = "⚖️ Durgun (-%5 - %5)"
    DARALAN = "⚠️ Daralan (<-5%)"
    KRITIK_DARALMA = "🔥 Kritik Daralma (<-20%)"

class ProductSegment(Enum):
    """Gelişmiş BCG Matrisi segmentleri"""
    YILDIZ = "⭐ Yıldız Ürünler"
    NAKIT_INEK = "🐄 Nakit İnekleri"
    SORU_ISARETI = "❓ Soru İşaretleri"
    ZAYIF = "💤 Zayıf Ürünler"
    YUKSELEN_YILDIZ = "🌟 Yükselen Yıldızlar"
    POTANSIYEL = "🎯 Potansiyel Vaat Edenler"
    OLGUN = "🏆 Olgun Ürünler"
    GERILEYEN = "📉 Gerileyen Ürünler"
    DISRUPTIVE = "💎 Disruptif İnovasyonlar"
    NIŞ = "🎯 Niş Ürünler"

class MarketConcentration(Enum):
    """Pazar yoğunluğu sınıflandırması (HHI bazlı)"""
    MONOPOL = "👑 Monopol (HHI > 2500)"
    YUKSEK_OLIGOPOL = "🏢 Yüksek Oligopol (2000-2500)"
    OLIGOPOL = "🏛️ Oligopol (1500-2000)"
    REKABETCI = "⚔️ Rekabetçi (1000-1500)"
    PARCALI = "🧩 Parçalı (500-1000)"
    ATOMISTIK = "✨ Atomistik (HHI < 500)"

class PortfolioStrategy(Enum):
    """Portföy stratejileri"""
    AGGRESIF_BUYUME = "🚀 Agresif Büyüme"
    KORUMA = "🛡️ Pazar Koruma"
    HASAT = "🌾 Nakit Hasadı"
    ELDEN_CIKAR = "💰 Elden Çıkarma"
    ARGE = "🔬 Ar-Ge Yatırımı"
    BEKLE_GOR = "👁️ Bekle-Gör"
    DIVERSIFIKASYON = "🎲 Diversifikasyon"
    POZISYON_GUNCELLE = "🔄 Pozisyon Güncelleme"

class DataQuality(Enum):
    """Veri kalitesi seviyeleri"""
    MUKEMMEL = "💎 Mükemmel"
    IYI = "✅ İyi"
    ORTA = "⚠️ Orta"
    DUSUK = "🔻 Düşük"
    KRITIK = "❌ Kritik"

# ================================================
# 10. GELİŞMİŞ DATA CLASSES
# ================================================

@dataclass
class ProdPackNode:
    """ProdPack Hiyerarşi Düğümü - Genişletilmiş"""
    id: str
    name: str
    level: str  # molecule, company, brand, pack
    parent_id: Optional[str] = None
    sales_2024: float = 0.0
    sales_2023: float = 0.0
    sales_2022: float = 0.0
    growth_rate_2023_2024: float = 0.0
    growth_rate_2022_2023: float = 0.0
    cagr_3y: float = 0.0
    market_share: float = 0.0
    market_share_change: float = 0.0
    price_2024: float = 0.0
    price_change: float = 0.0
    volume_2024: float = 0.0
    volume_change: float = 0.0
    profitability: float = 0.0  # Tahmini kar marjı
    risk_score: float = 0.0
    anomaly_score: float = 0.0
    segment: str = "Belirlenmemiş"
    children: List['ProdPackNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Sözlük dönüşümü"""
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level,
            'parent_id': self.parent_id,
            'sales_2024': self.sales_2024,
            'growth_rate': self.growth_rate_2023_2024,
            'market_share': self.market_share,
            'segment': self.segment,
            'risk_score': self.risk_score
        }

@dataclass
class MarketMetrics:
    """Pazar metrikleri - Genişletilmiş"""
    total_market_value_2024: float = 0.0
    total_market_value_2023: float = 0.0
    total_market_value_2022: float = 0.0
    yoy_growth_2024: float = 0.0
    yoy_growth_2023: float = 0.0
    cagr_3y: float = 0.0
    cagr_5y: float = 0.0
    hhi_index: float = 0.0
    hhi_trend: float = 0.0
    concentration_ratio_4: float = 0.0
    concentration_ratio_8: float = 0.0
    gini_coefficient: float = 0.0
    market_volatility: float = 0.0
    price_index: float = 0.0
    price_elasticity: float = 0.0
    volume_index: float = 0.0
    international_penetration: float = 0.0
    innovation_index: float = 0.0
    generic_penetration: float = 0.0
    brand_concentration: float = 0.0
    molecule_concentration: float = 0.0
    market_maturity: str = "Gelişmekte"
    growth_stage: str = "Büyüme"
    seasonality_index: float = 0.0
    forecast_2025: float = 0.0
    forecast_2026: float = 0.0
    forecast_confidence: float = 0.95
    forecast_volatility: float = 0.0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Metrikleri DataFrame'e dönüştür"""
        df = pd.DataFrame([asdict(self)])
        return df

@dataclass
class StrategicInsight:
    """Stratejik içgörü - Genişletilmiş"""
    id: str
    title: str
    description: str
    insight_type: str  # growth, risk, opportunity, threat, trend
    priority: str  # critical, high, medium, low
    impact: str  # strategic, operational, financial, reputational
    confidence: float  # 0-1
    recommendation: str
    action_items: List[str] = field(default_factory=list)
    kpis: Dict[str, float] = field(default_factory=dict)
    affected_products: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    source: str = "AI Analytics"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ForecastResult:
    """Tahmin sonucu - Genişletilmiş"""
    periods: List[str]
    predictions: List[float]
    lower_bound_80: List[float]
    upper_bound_80: List[float]
    lower_bound_95: List[float]
    upper_bound_95: List[float]
    model_type: str
    mape: float
    rmse: float
    mae: float
    r2: float
    growth_rate: float
    cagr_forecast: float
    seasonality_strength: float
    trend_strength: float
    residual_std: float
    confidence_level: float = 0.95
    
    def get_forecast_df(self) -> pd.DataFrame:
        """Tahmin DataFrame'i oluştur"""
        return pd.DataFrame({
            'Dönem': self.periods,
            'Tahmin': self.predictions,
            'Alt_Sınır_80': self.lower_bound_80,
            'Üst_Sınır_80': self.upper_bound_80,
            'Alt_Sınır_95': self.lower_bound_95,
            'Üst_Sınır_95': self.upper_bound_95
        })

@dataclass
class CompanyProfile:
    """Şirket profili - Genişletilmiş"""
    company_name: str
    total_sales_2024: float
    market_share: float
    market_share_change: float
    product_count: int
    molecule_count: int
    growth_rate: float
    cagr_3y: float
    profitability: float
    geographic_presence: int
    innovation_score: float
    risk_score: float
    competitive_position: str
    swot: Dict[str, List[str]] = field(default_factory=dict)
    top_products: List[str] = field(default_factory=list)
    key_molecules: List[str] = field(default_factory=list)
    strategic_initiatives: List[str] = field(default_factory=list)

# ================================================
# 11. SABİTLER VE YAPILANDIRMA
# ================================================

class ExecutiveColors:
    """Executive Dark Mode renk paleti - Profesyonel"""
    PRIMARY = "#0A1A2F"  # Lacivert
    SECONDARY = "#1E3A5F"  # Koyu Lacivert
    TERTIARY = "#2C3E50"  # Gri-Lacivert
    ACCENT_GOLD = "#D4AF37"  # Altın
    ACCENT_SILVER = "#C0C0C0"  # Gümüş
    ACCENT_BRONZE = "#CD7F32"  # Bronz
    ACCENT_BLUE = "#3498DB"  # Parlak Mavi
    ACCENT_GREEN = "#2ECC71"  # Zümrüt Yeşili
    ACCENT_RED = "#E74C3C"  # Kırmızı
    ACCENT_ORANGE = "#F39C12"  # Turuncu
    ACCENT_PURPLE = "#9B59B6"  # Mor
    BACKGROUND = "#0F2A3F"  # Arkaplan
    SURFACE = "#1A2C3E"  # Yüzey
    SURFACE_LIGHT = "#2C3E50"  # Açık Yüzey
    TEXT_PRIMARY = "#FFFFFF"  # Beyaz
    TEXT_SECONDARY = "#BDC3C7"  # Açık Gri
    TEXT_MUTED = "#95A5A6"  # Soluk Gri
    SUCCESS = "#2ECC71"  # Başarı
    WARNING = "#F39C12"  # Uyarı
    DANGER = "#E74C3C"  # Tehlike
    INFO = "#3498DB"  # Bilgi
    GRID = "#34495E"  # Izgara
    CHART_1 = "#D4AF37"
    CHART_2 = "#3498DB"
    CHART_3 = "#2ECC71"
    CHART_4 = "#E74C3C"
    CHART_5 = "#9B59B6"
    CHART_6 = "#F39C12"
    CHART_7 = "#1ABC9C"
    CHART_8 = "#E67E22"

class ChartTemplates:
    """Plotly şablonları - Executive tema"""
    
    @staticmethod
    def executive_template() -> go.layout.Template:
        """Executive dark mode template"""
        template = go.layout.Template()
        
        # Layout ayarları
        template.layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=ExecutiveColors.TEXT_PRIMARY, family='Inter, Arial, sans-serif'),
            title=dict(font=dict(size=20, color=ExecutiveColors.ACCENT_GOLD)),
            xaxis=dict(
                gridcolor=ExecutiveColors.GRID,
                linecolor=ExecutiveColors.ACCENT_SILVER,
                tickcolor=ExecutiveColors.ACCENT_SILVER,
                title_font=dict(color=ExecutiveColors.TEXT_SECONDARY)
            ),
            yaxis=dict(
                gridcolor=ExecutiveColors.GRID,
                linecolor=ExecutiveColors.ACCENT_SILVER,
                tickcolor=ExecutiveColors.ACCENT_SILVER,
                title_font=dict(color=ExecutiveColors.TEXT_SECONDARY)
            ),
            legend=dict(
                font=dict(color=ExecutiveColors.TEXT_PRIMARY),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor=ExecutiveColors.ACCENT_SILVER,
                borderwidth=1
            ),
            hoverlabel=dict(
                bgcolor=ExecutiveColors.SURFACE,
                font_color=ExecutiveColors.TEXT_PRIMARY,
                bordercolor=ExecutiveColors.ACCENT_GOLD
            ),
            colorway=[
                ExecutiveColors.CHART_1,
                ExecutiveColors.CHART_2,
                ExecutiveColors.CHART_3,
                ExecutiveColors.CHART_4,
                ExecutiveColors.CHART_5,
                ExecutiveColors.CHART_6,
                ExecutiveColors.CHART_7,
                ExecutiveColors.CHART_8
            ]
        )
        return template

# ================================================
# 12. GELİŞMİŞ VERİ İŞLEME MOTORU
# ================================================

class AdvancedDataEngine:
    """
    Gelişmiş veri işleme motoru - 100+ metod
    Büyük veri optimizasyonu, paralel işleme, akıllı dönüşümler
    """
    
    def __init__(self):
        self.cache = {}
        self.column_metadata = {}
        self.data_quality_score = 0.0
        self.processing_stats = defaultdict(int)
        
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
    def load_and_clean_data(uploaded_file) -> pd.DataFrame:
        """
        Veri yükleme ve temizleme - Cache ile optimize edilmiş
        1M+ satır için optimize
        """
        try:
            # Dosya tipine göre yükleme
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8')
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file, engine='xlrd')
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith('.feather'):
                df = pd.read_feather(uploaded_file)
            elif uploaded_file.name.endswith('.pkl'):
                df = pd.read_pickle(uploaded_file)
            else:
                st.error("Desteklenmeyen dosya formatı")
                return pd.DataFrame()
            
            # Büyük veri optimizasyonu
            if len(df) > 100000:
                df = AdvancedDataEngine._optimize_large_dataframe(df)
            
            # Sütun isimlerini temizle ve standardize et
            df.columns = AdvancedDataEngine._clean_column_names_pro(df.columns.tolist())
            
            # Regex ile yıl ayıklama ve sütunları yeniden adlandır
            df = AdvancedDataEngine._extract_years_advanced(df)
            
            # Tip dönüşümleri - Güvenli
            df = AdvancedDataEngine._safe_type_conversion(df)
            
            # Eksik veri işleme
            df = AdvancedDataEngine._handle_missing_values(df)
            
            # Aykırı değer tespiti ve işleme
            df = AdvancedDataEngine._detect_and_handle_outliers(df)
            
            # ProdPack hiyerarşisi için sütunları oluştur/güçlendir
            df = AdvancedDataEngine._ensure_prodpack_hierarchy(df)
            
            # Analitik özellikler ekle
            df = AdvancedDataEngine._create_analytical_features(df)
            
            return df
            
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")
            st.exception(e)
            return pd.DataFrame()
    
    @staticmethod
    def _optimize_large_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Büyük DataFrame'ler için bellek optimizasyonu"""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                # Kategorik dönüşüm
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
        
        return df
    
    @staticmethod
    def _clean_column_names_pro(columns: List[str]) -> List[str]:
        """Profesyonel sütun isimlendirme - Benzersiz, standardize"""
        cleaned = []
        seen = defaultdict(int)
        
        # Kapsamlı Türkçe-İngilizce terim sözlüğü
        term_mapping = {
            'molecule': 'Molekul',
            'molekül': 'Molekul',
            'etken': 'Molekul',
            'active': 'Molekul',
            'ingredient': 'Molekul',
            
            'brand': 'Marka',
            'marka': 'Marka',
            'urun': 'Urun',
            'product': 'Urun',
            
            'company': 'Sirket',
            'firma': 'Sirket',
            'manufacturer': 'Uretici',
            'uretici': 'Uretici',
            'corp': 'Sirket',
            'inc': 'Sirket',
            
            'pack': 'Paket',
            'package': 'Paket',
            'prodpack': 'Paket',
            'sku': 'Paket',
            'form': 'Form',
            'doz': 'Dozaj',
            'strength': 'Dozaj',
            'size': 'Boyut',
            
            'sales': 'Satis',
            'satış': 'Satis',
            'revenue': 'Gelir',
            'gelir': 'Gelir',
            'turnover': 'Ciro',
            'ciro': 'Ciro',
            
            'volume': 'Hacim',
            'hacim': 'Hacim',
            'unit': 'Birim',
            'birim': 'Birim',
            'quantity': 'Miktar',
            'miktar': 'Miktar',
            
            'price': 'Fiyat',
            'fiyat': 'Fiyat',
            'cost': 'Maliyet',
            'maliyet': 'Maliyet',
            
            'growth': 'Buyume',
            'b growth': 'Buyume',
            'buyume': 'Buyume',
            'cagr': 'CAGR',
            
            'share': 'Pay',
            'market share': 'Pazar_Payi',
            'pazar payı': 'Pazar_Payi',
            'pazar payi': 'Pazar_Payi',
            
            'region': 'Bolge',
            'bölge': 'Bolge',
            'sub region': 'Alt_Bolge',
            'alt bölge': 'Alt_Bolge',
            
            'country': 'Ulke',
            'ülke': 'Ulke',
            'city': 'Sehir',
            'şehir': 'Sehir',
            
            'year': 'Yil',
            'yıl': 'Yil',
            'month': 'Ay',
            'quarter': 'Ceyrek',
            
            'profit': 'Kar',
            'kar': 'Kar',
            'margin': 'Marj',
            'marj': 'Marj',
            
            'risk': 'Risk',
            'anomaly': 'Anomali',
            'outlier': 'Aykiri',
            
            'segment': 'Segment',
            'category': 'Kategori',
            'kategori': 'Kategori',
            'class': 'Sinif'
        }
        
        for col in columns:
            original = str(col)
            col_clean = original
            
            # Türkçe karakter düzeltme
            turkish_chars = {
                'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
            }
            for tr, en in turkish_chars.items():
                col_clean = col_clean.replace(tr, en)
            
            # Özel karakterleri temizle
            col_clean = re.sub(r'[^\w\s\-]', ' ', col_clean)
            col_clean = re.sub(r'\s+', '_', col_clean.strip())
            
            # Terimleri dönüştür
            col_lower = col_clean.lower()
            for eng, tr in term_mapping.items():
                if eng in col_lower or eng.replace(' ', '_') in col_lower:
                    col_clean = tr
                    break
            
            # Benzersiz isimlendirme
            base_name = col_clean[:30]  # Maksimum 30 karakter
            counter = 1
            unique_name = base_name
            
            while unique_name in seen:
                if len(f"{base_name}_{counter}") <= 30:
                    unique_name = f"{base_name}_{counter}"
                else:
                    unique_name = f"{base_name[:25]}_{counter}"
                counter += 1
            
            seen[unique_name] = True
            cleaned.append(unique_name)
        
        return cleaned
    
    @staticmethod
    def _extract_years_advanced(df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş yıl ayıklama - Regex, hata yönetimi, çoklu format
        """
        year_pattern = re.compile(r'20\d{2}|\d{2}/\d{4}|\d{4}/\d{2}|\d{2}-\d{4}')
        new_columns = {}
        
        for col in df.columns:
            col_str = str(col)
            matches = year_pattern.findall(col_str)
            
            if matches:
                year = matches[0]
                # Sadece 4 haneli yılı al
                year_match = re.search(r'20\d{2}', year)
                if year_match:
                    year = year_match.group()
                    
                    # Kategori belirle
                    cat_lower = col_str.lower()
                    if any(x in cat_lower for x in ['satis', 'sales', 'gelir', 'revenue']):
                        new_name = f'Satis_{year}'
                    elif any(x in cat_lower for x in ['hacim', 'volume', 'birim', 'unit', 'miktar']):
                        new_name = f'Hacim_{year}'
                    elif any(x in cat_lower for x in ['fiyat', 'price']):
                        new_name = f'Fiyat_{year}'
                    elif any(x in cat_lower for x in ['maliyet', 'cost']):
                        new_name = f'Maliyet_{year}'
                    elif any(x in cat_lower for x in ['kar', 'profit']):
                        new_name = f'Kar_{year}'
                    elif any(x in cat_lower for x in ['marj', 'margin']):
                        new_name = f'Marj_{year}'
                    else:
                        new_name = f'Deger_{year}'
                    
                    new_columns[col] = new_name
                    continue
            
            new_columns[col] = col
        
        df.rename(columns=new_columns, inplace=True)
        return df
    
    @staticmethod
    def _safe_type_conversion(df: pd.DataFrame) -> pd.DataFrame:
        """
        Güvenli tip dönüşümü - pd.api.types kullanımı
        """
        for col in df.columns:
            # Sayısal dönüşüm
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # Yıl içeren sütunlar
                if any(x in col for x in ['Satis', 'Hacim', 'Fiyat', 'Maliyet']):
                    try:
                        # Önce virgülleri kaldır, sonra sayısal yap
                        df[col] = df[col].astype(str).str.replace(',', '.').str.replace('[^0-9\.\-]', '', regex=True)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
                
                # Tarih sütunları
                elif any(x in col.lower() for x in ['date', 'tarih', 'time']):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
                
                # Kategorik sütunlar
                else:
                    unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
                    if unique_ratio < 0.05:  # %5'ten az benzersiz değer
                        df[col] = df[col].astype('category')
            
            # Boolean tipi - Ambiguous truth value hatası için çözüm
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)
            
            # Zaman serisi indeksi
            elif 'tarih' in col.lower() or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        return df
    
    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş eksik veri işleme
        """
        missing_threshold = 0.7  # %70'ten fazla eksik varsa sütunu sil
        
        for col in df.columns:
            missing_ratio = df[col].isnull().mean()
            
            if missing_ratio > missing_threshold:
                df = df.drop(columns=[col])
                continue
            
            if missing_ratio > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Sayısal: medyan ile doldur
                    df[col] = df[col].fillna(df[col].median())
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Tarih: ileri/geri taşıma
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    # Kategorik: mod ile doldur
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna('Bilinmiyor')
        
        return df
    
    @staticmethod
    def _detect_and_handle_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Aykırı değer tespiti ve işleme
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].nunique() < 10:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Winsorization
                df[col] = df[col].clip(lower_bound, upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs(zscore(df[col].fillna(df[col].median()), nan_policy='omit'))
                threshold = 3
                
                outlier_mask = z_scores > threshold
                if outlier_mask.any():
                    median_val = df[col].median()
                    df.loc[outlier_mask, col] = median_val
        
        return df
    
    @staticmethod
    def _ensure_prodpack_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
        """
        ProdPack hiyerarşisi için zorunlu sütunları oluştur
        """
        # Molekül sütunu
        if 'Molekul' not in df.columns:
            molecule_candidates = ['Molecule', 'Active', 'Ingredient', 'Etken', 'Madde']
            for col in df.columns:
                if any(c.lower() in col.lower() for c in molecule_candidates):
                    df.rename(columns={col: 'Molekul'}, inplace=True)
                    break
            if 'Molekul' not in df.columns:
                df['Molekul'] = 'Genel'
        
        # Şirket sütunu
        if 'Sirket' not in df.columns and 'Uretici' not in df.columns:
            company_candidates = ['Company', 'Firma', 'Manufacturer', 'Marka_Sahibi']
            for col in df.columns:
                if any(c.lower() in col.lower() for c in company_candidates):
                    df.rename(columns={col: 'Sirket'}, inplace=True)
                    break
            if 'Sirket' not in df.columns:
                df['Sirket'] = 'Bilinmeyen'
        
        # Marka sütunu
        if 'Marka' not in df.columns:
            brand_candidates = ['Brand', 'Urun_Adi', 'Product', 'Trade_Name']
            for col in df.columns:
                if any(c.lower() in col.lower() for c in brand_candidates):
                    df.rename(columns={col: 'Marka'}, inplace=True)
                    break
            if 'Marka' not in df.columns:
                df['Marka'] = df.get('Paket', 'Standart')
        
        # Paket sütunu (ProdPack)
        if 'Paket' not in df.columns:
            pack_candidates = ['Pack', 'Package', 'SKU', 'Form', 'Doz', 'Size', 'ProdPack']
            for col in df.columns:
                if any(c.lower() in col.lower() for c in pack_candidates):
                    df.rename(columns={col: 'Paket'}, inplace=True)
                    break
            if 'Paket' not in df.columns:
                df['Paket'] = 'Standart Paket'
        
        return df
    
    @staticmethod
    def _create_analytical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analitik özellik mühendisliği
        """
        # Satış sütunlarını bul
        sales_cols = [col for col in df.columns if re.search(r'Satis_20\d{2}', col)]
        sales_cols.sort()
        
        if len(sales_cols) >= 1:
            latest_sales = sales_cols[-1]
            
            # Toplam pazar
            df['Toplam_Pazar'] = df[latest_sales].sum()
            
            # Pazar payı
            total_sales = df[latest_sales].sum()
            if total_sales > 0:
                df['Pazar_Payi'] = (df[latest_sales] / total_sales) * 100
            
            # Birikimli satış
            df['Kumulatif_Satis'] = df[latest_sales].cumsum()
        
        if len(sales_cols) >= 2:
            prev_sales = sales_cols[-2]
            
            # Büyüme oranı
            mask = df[prev_sales] > 0
            df.loc[mask, 'Buyume_Orani'] = ((df.loc[mask, latest_sales] - df.loc[mask, prev_sales]) 
                                           / df.loc[mask, prev_sales]) * 100
            df['Buyume_Orani'] = df['Buyume_Orani'].fillna(0).clip(-100, 1000)
            
            # Mutlak değişim
            df['Mutlak_Degisim'] = df[latest_sales] - df[prev_sales]
        
        if len(sales_cols) >= 3:
            # 3 yıllık CAGR
            first_sales = sales_cols[0]
            df['CAGR_3Y'] = ((df[latest_sales] / df[first_sales].replace(0, np.nan)) ** (1/3) - 1) * 100
            df['CAGR_3Y'] = df['CAGR_3Y'].fillna(0)
        
        # Fiyat sütunlarını bul
        price_cols = [col for col in df.columns if re.search(r'Fiyat_20\d{2}', col)]
        if price_cols:
            latest_price = price_cols[-1]
            if 'Fiyat' not in df.columns:
                df['Fiyat'] = df[latest_price]
        
        # Performans indeksi
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 3:
            # Normalize edilmiş ortalama
            scaler = StandardScaler()
            try:
                scaled_data = scaler.fit_transform(df[numeric_cols].fillna(0))
                df['Performans_Indeksi'] = scaled_data.mean(axis=1)
            except:
                pass
        
        return df
    
    @staticmethod
    def parallel_process(df: pd.DataFrame, func: Callable, n_jobs: int = -1) -> pd.DataFrame:
        """
        Paralel veri işleme - Büyük veri için
        """
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        
        if len(df) < 10000 or n_jobs == 1:
            return func(df)
        
        # Parçalara böl
        chunks = np.array_split(df, n_jobs)
        
        # Paralel işleme
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(func, chunk) for chunk in chunks]
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        return pd.concat(results, ignore_index=True)

# ================================================
# 13. PRODPACK DERİNLİK ANALİZİ (YENİ MODÜL - GENİŞLETİLMİŞ)
# ================================================

class ProdPackDeepDive:
    """
    Gelişmiş ProdPack Derinlik Analizi
    Molekül → Şirket → Marka → Paket hiyerarşisi
    Sunburst, Sankey, Treemap, İcicle görselleştirmeleri
    Kanibalizasyon, büyüme matrisi, pazar payı trendi
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.hierarchy_root = None
        self.sales_cols = []
        self.growth_cols = []
        self.latest_year = None
        self.latest_sales_col = None
        self.prev_sales_col = None
        self.latest_growth_col = None
        
        self._initialize_columns()
        self._preprocess_data()
    
    def _initialize_columns(self):
        """Sütunları initialize et"""
        # Satış sütunları
        self.sales_cols = [col for col in self.df.columns if re.search(r'Satis_20\d{2}', col)]
        self.sales_cols.sort(key=lambda x: int(re.search(r'20\d{2}', x).group()))
        
        if self.sales_cols:
            self.latest_sales_col = self.sales_cols[-1]
            self.latest_year = int(re.search(r'20\d{2}', self.latest_sales_col).group())
            
            if len(self.sales_cols) >= 2:
                self.prev_sales_col = self.sales_cols[-2]
        
        # Büyüme sütunları
        self.growth_cols = [col for col in self.df.columns if 'Buyume_' in col]
        self.growth_cols.sort()
        if self.growth_cols:
            self.latest_growth_col = self.growth_cols[-1]
    
    def _preprocess_data(self):
        """Veriyi analiz için hazırla"""
        # Toplam pazar payı hesapla
        if self.latest_sales_col:
            total_market = self.df[self.latest_sales_col].sum()
            if total_market > 0:
                self.df['Pazar_Payi_2024'] = (self.df[self.latest_sales_col] / total_market) * 100
        
        # Büyüme oranı hesapla
        if self.prev_sales_col and self.latest_sales_col:
            mask = self.df[self.prev_sales_col] > 0
            self.df['Buyume_2023_2024'] = 0.0
            self.df.loc[mask, 'Buyume_2023_2024'] = (
                (self.df.loc[mask, self.latest_sales_col] - self.df.loc[mask, self.prev_sales_col]) 
                / self.df.loc[mask, self.prev_sales_col] * 100
            )
            self.latest_growth_col = 'Buyume_2023_2024'
        
        # 3 yıllık CAGR hesapla
        if len(self.sales_cols) >= 3:
            first_sales = self.sales_cols[0]
            mask = self.df[first_sales] > 0
            self.df['CAGR_3Y'] = 0.0
            n_years = self.latest_year - int(re.search(r'20\d{2}', first_sales).group())
            if n_years > 0:
                self.df.loc[mask, 'CAGR_3Y'] = (
                    (self.df.loc[mask, self.latest_sales_col] / self.df.loc[mask, first_sales]) ** (1/n_years) - 1
                ) * 100
    
    def build_hierarchy(self, selected_molecule: Optional[str] = None) -> ProdPackNode:
        """
        Hiyerarşik ağacı oluştur - Molekül -> Şirket -> Marka -> Paket
        """
        # Filtreleme
        df_filtered = self.df
        if selected_molecule and selected_molecule != 'Tüm Moleküller':
            if 'Molekul' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['Molekul'] == selected_molecule]
        
        # Kök düğüm
        root_name = selected_molecule if selected_molecule and selected_molecule != 'Tüm Moleküller' else 'Tüm Moleküller'
        root = ProdPackNode(
            id='root',
            name=root_name,
            level='molecule',
            sales_2024=df_filtered[self.latest_sales_col].sum() if self.latest_sales_col else 0,
            sales_2023=df_filtered[self.prev_sales_col].sum() if self.prev_sales_col else 0,
            growth_rate_2023_2024=df_filtered[self.latest_growth_col].mean() if self.latest_growth_col else 0
        )
        
        # Şirket seviyesi
        if 'Sirket' in df_filtered.columns:
            for sirket, sirket_df in df_filtered.groupby('Sirket'):
                if pd.isna(sirket) or sirket == '':
                    continue
                
                sirket_node = ProdPackNode(
                    id=f"sirket_{hashlib.md5(str(sirket).encode()).hexdigest()[:8]}",
                    name=str(sirket)[:30],
                    level='company',
                    parent_id='root',
                    sales_2024=sirket_df[self.latest_sales_col].sum() if self.latest_sales_col else 0,
                    sales_2023=sirket_df[self.prev_sales_col].sum() if self.prev_sales_col else 0,
                    growth_rate_2023_2024=sirket_df[self.latest_growth_col].mean() if self.latest_growth_col else 0,
                    market_share=(sirket_df[self.latest_sales_col].sum() / df_filtered[self.latest_sales_col].sum() * 100) if self.latest_sales_col else 0
                )
                root.children.append(sirket_node)
                
                # Marka seviyesi
                if 'Marka' in sirket_df.columns:
                    for marka, marka_df in sirket_df.groupby('Marka'):
                        if pd.isna(marka) or marka == '':
                            continue
                        
                        marka_node = ProdPackNode(
                            id=f"marka_{hashlib.md5(f'{sirket}_{marka}'.encode()).hexdigest()[:8]}",
                            name=str(marka)[:30],
                            level='brand',
                            parent_id=sirket_node.id,
                            sales_2024=marka_df[self.latest_sales_col].sum() if self.latest_sales_col else 0,
                            sales_2023=marka_df[self.prev_sales_col].sum() if self.prev_sales_col else 0,
                            growth_rate_2023_2024=marka_df[self.latest_growth_col].mean() if self.latest_growth_col else 0,
                            market_share=(marka_df[self.latest_sales_col].sum() / df_filtered[self.latest_sales_col].sum() * 100) if self.latest_sales_col else 0
                        )
                        sirket_node.children.append(marka_node)
                        
                        # Paket seviyesi (ProdPack)
                        if 'Paket' in marka_df.columns:
                            for paket, paket_df in marka_df.groupby('Paket'):
                                if pd.isna(paket) or paket == '':
                                    continue
                                
                                paket_node = ProdPackNode(
                                    id=f"paket_{hashlib.md5(f'{sirket}_{marka}_{paket}'.encode()).hexdigest()[:8]}",
                                    name=str(paket)[:30],
                                    level='pack',
                                    parent_id=marka_node.id,
                                    sales_2024=paket_df[self.latest_sales_col].sum() if self.latest_sales_col else 0,
                                    sales_2023=paket_df[self.prev_sales_col].sum() if self.prev_sales_col else 0,
                                    growth_rate_2023_2024=paket_df[self.latest_growth_col].mean() if self.latest_growth_col else 0,
                                    market_share=(paket_df[self.latest_sales_col].sum() / df_filtered[self.latest_sales_col].sum() * 100) if self.latest_sales_col else 0
                                )
                                marka_node.children.append(paket_node)
        
        self.hierarchy_root = root
        return root
    
    def create_sunburst_diagram(self, root: ProdPackNode) -> go.Figure:
        """
        İnteraktif Sunburst diyagramı
        """
        labels = []
        parents = []
        values = []
        colors = []
        customdata = []
        
        def traverse(node: ProdPackNode, level: int = 0):
            # Etiket
            display_name = f"{node.name}"
            if node.level == 'pack':
                display_name = f"📦 {node.name}"
            elif node.level == 'brand':
                display_name = f"🏷️ {node.name}"
            elif node.level == 'company':
                display_name = f"🏢 {node.name}"
            elif node.level == 'molecule':
                display_name = f"💊 {node.name}"
            
            labels.append(display_name)
            
            # Parent
            if node.parent_id:
                parent_node = self._find_node(root, node.parent_id)
                if parent_node:
                    parent_display = f"{parent_node.name}"
                    if parent_node.level == 'pack':
                        parent_display = f"📦 {parent_node.name}"
                    elif parent_node.level == 'brand':
                        parent_display = f"🏷️ {parent_node.name}"
                    elif parent_node.level == 'company':
                        parent_display = f"🏢 {parent_node.name}"
                    parents.append(parent_display)
                else:
                    parents.append('')
            else:
                parents.append('')
            
            # Değer (satış)
            values.append(node.sales_2024 if node.sales_2024 > 0 else 0.01)
            
            # Renk (büyüme oranına göre)
            if node.growth_rate_2023_2024 > 20:
                colors.append(ExecutiveColors.SUCCESS)
            elif node.growth_rate_2023_2024 > 5:
                colors.append(ExecutiveColors.ACCENT_GOLD)
            elif node.growth_rate_2023_2024 > -5:
                colors.append(ExecutiveColors.INFO)
            else:
                colors.append(ExecutiveColors.DANGER)
            
            # Custom data
            customdata.append([
                f"{node.sales_2024:,.0f}",
                f"%{node.growth_rate_2023_2024:.1f}",
                f"%{node.market_share:.1f}",
                node.level
            ])
            
            for child in node.children:
                traverse(child, level + 1)
        
        traverse(root)
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(
                colors=colors,
                line=dict(width=1, color=ExecutiveColors.SURFACE)
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         'Satış: %{customdata[0]}₺<br>' +
                         'Büyüme: %{customdata[1]}<br>' +
                         'Pazar Payı: %{customdata[2]}<br>' +
                         'Seviye: %{customdata[3]}<br>' +
                         '<extra></extra>',
            customdata=customdata,
            textinfo='label+percent entry',
            insidetextorientation='radial'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'<b>ProdPack Hiyerarşisi: {root.name}</b>',
                font=dict(size=24, color=ExecutiveColors.ACCENT_GOLD),
                x=0.5
            ),
            margin=dict(t=80, l=20, r=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=ExecutiveColors.TEXT_PRIMARY, size=12),
            height=700,
            template=ChartTemplates.executive_template()
        )
        
        return fig
    
    def create_sankey_diagram(self, root: ProdPackNode) -> go.Figure:
        """
        Sankey akış diyagramı
        """
        labels = []
        sources = []
        targets = []
        values = []
        
        def collect_nodes(node: ProdPackNode):
            node_label = f"{node.name} ({node.sales_2024:,.0f}₺)"
            if node_label not in labels:
                labels.append(node_label)
            
            node_index = labels.index(node_label)
            
            for child in node.children:
                child_label = f"{child.name} ({child.sales_2024:,.0f}₺)"
                if child_label not in labels:
                    labels.append(child_label)
                
                child_index = labels.index(child_label)
                sources.append(node_index)
                targets.append(child_index)
                values.append(child.sales_2024)
                
                collect_nodes(child)
        
        collect_nodes(root)
        
        # Renkler
        node_colors = []
        for i, label in enumerate(labels):
            if '📦' in label:
                node_colors.append(ExecutiveColors.CHART_3)
            elif '🏷️' in label:
                node_colors.append(ExecutiveColors.CHART_2)
            elif '🏢' in label:
                node_colors.append(ExecutiveColors.CHART_1)
            elif '💊' in label:
                node_colors.append(ExecutiveColors.ACCENT_GOLD)
            else:
                node_colors.append(ExecutiveColors.CHART_5)
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color=ExecutiveColors.SURFACE, width=0.5),
                label=labels,
                color=node_colors,
                hovertemplate='<b>%{label}</b><br>Toplam: %{value:,.0f}₺<extra></extra>'
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=[ExecutiveColors.ACCENT_SILVER] * len(sources),
                hovertemplate='Akış: %{value:,.0f}₺<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title=dict(
                text=f'<b>ProdPack Akış Diyagramı: {root.name}</b>',
                font=dict(size=24, color=ExecutiveColors.ACCENT_GOLD),
                x=0.5
            ),
            font=dict(size=12, color=ExecutiveColors.TEXT_PRIMARY),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=700,
            template=ChartTemplates.executive_template()
        )
        
        return fig
    
    def create_treemap(self, root: ProdPackNode) -> go.Figure:
        """
        Hiyerarşik Treemap görselleştirmesi
        """
        ids = []
        labels = []
        parents = []
        values = []
        colors = []
        
        def traverse(node: ProdPackNode):
            ids.append(node.id)
            labels.append(f"{node.name}<br>%{node.growth_rate_2023_2024:.1f}")
            parents.append(node.parent_id if node.parent_id else '')
            values.append(node.sales_2024 if node.sales_2024 > 0 else 0.01)
            
            # Renk
            if node.growth_rate_2023_2024 > 20:
                colors.append(ExecutiveColors.SUCCESS)
            elif node.growth_rate_2023_2024 > 5:
                colors.append(ExecutiveColors.ACCENT_GOLD)
            elif node.growth_rate_2023_2024 > -5:
                colors.append(ExecutiveColors.INFO)
            else:
                colors.append(ExecutiveColors.DANGER)
            
            for child in node.children:
                traverse(child)
        
        traverse(root)
        
        fig = go.Figure(go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=colors,
                line=dict(width=1, color=ExecutiveColors.SURFACE)
            ),
            textinfo='label+value+percent root',
            hovertemplate='<b>%{label}</b><br>Satış: %{value:,.0f}₺<br>Büyüme: %{color:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'<b>ProdPack Treemap: {root.name}</b>',
                font=dict(size=24, color=ExecutiveColors.ACCENT_GOLD),
                x=0.5
            ),
            margin=dict(t=80, l=10, r=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=ExecutiveColors.TEXT_PRIMARY),
            height=700,
            template=ChartTemplates.executive_template()
        )
        
        return fig
    
    def analyze_cannibalization_matrix(self, selected_molecule: str) -> pd.DataFrame:
        """
        Pazar Kanibalizasyonu Analizi - Büyüme/Hacim Matrisi
        Aynı şirket içindeki farklı paketlerin birbirinin payından çalıp çalmadığını analiz et
        """
        if selected_molecule == 'Tüm Moleküller' or not self.latest_sales_col:
            return pd.DataFrame()
        
        df_mol = self.df[self.df['Molekul'] == selected_molecule].copy()
        
        if 'Sirket' not in df_mol.columns or 'Paket' not in df_mol.columns:
            return pd.DataFrame()
        
        results = []
        
        for sirket, sirket_df in df_mol.groupby('Sirket'):
            if len(sirket_df) < 2:
                continue
            
            total_sales = sirket_df[self.latest_sales_col].sum()
            
            # Şirket bazlı büyüme
            if self.prev_sales_col:
                prev_total = sirket_df[self.prev_sales_col].sum()
                company_growth = ((total_sales - prev_total) / prev_total * 100) if prev_total > 0 else 0
            else:
                company_growth = 0
            
            for idx, row in sirket_df.iterrows():
                paket = row['Paket']
                sales = row[self.latest_sales_col]
                prev_sales = row[self.prev_sales_col] if self.prev_sales_col else 0
                
                # Büyüme oranı
                if prev_sales > 0:
                    growth = ((sales - prev_sales) / prev_sales) * 100
                else:
                    growth = 0
                
                # Pazar payı
                share = (sales / total_sales * 100) if total_sales > 0 else 0
                
                # Pazar payı değişimi
                if self.prev_sales_col:
                    prev_share = (prev_sales / prev_total * 100) if prev_total > 0 else 0
                    share_change = share - prev_share
                else:
                    share_change = 0
                
                # Kanibalizasyon skoru
                # Yüksek pazar payı + Düşük büyüme = Kanibalizasyon riski
                # Negatif pazar payı değişimi de kanibalizasyon göstergesi
                cannibal_score = (share * 0.4) + (abs(share_change) * 0.3) + (max(0, 20 - growth) * 0.3)
                
                # Kanibalizasyon tipi
                if share_change < -5 and growth < company_growth - 10:
                    cannibal_type = "🔴 Yüksek Kanibalizasyon"
                elif share_change < -2 or growth < company_growth - 5:
                    cannibal_type = "🟠 Orta Kanibalizasyon"
                elif share_change < 0:
                    cannibal_type = "🟡 Düşük Kanibalizasyon"
                else:
                    cannibal_type = "🟢 Kanibalizasyon Yok"
                
                results.append({
                    'Şirket': sirket,
                    'Paket': paket,
                    'Satış_2024': sales,
                    'Satış_2023': prev_sales,
                    'Büyüme_Oranı': growth,
                    'Şirket_Büyümesi': company_growth,
                    'Pazar_Payı': share,
                    'Pazar_Payı_Değişimi': share_change,
                    'Kanibalizasyon_Skoru': cannibal_score,
                    'Kanibalizasyon_Tipi': cannibal_type,
                    'Risk_Seviyesi': 'Yüksek' if cannibal_score > 50 else 'Orta' if cannibal_score > 30 else 'Düşük'
                })
        
        df_result = pd.DataFrame(results)
        
        if not df_result.empty:
            df_result = df_result.sort_values('Kanibalizasyon_Skoru', ascending=False)
            
            # Büyüme/Hacim matrisi kategorisi
            conditions = [
                (df_result['Büyüme_Oranı'] > 20) & (df_result['Pazar_Payı'] > 10),
                (df_result['Büyüme_Oranı'] > 20) & (df_result['Pazar_Payı'] <= 10),
                (df_result['Büyüme_Oranı'].between(0, 20)) & (df_result['Pazar_Payı'] > 10),
                (df_result['Büyüme_Oranı'] < 0) & (df_result['Pazar_Payı'] < 5)
            ]
            choices = ['Yıldız', 'Soru İşareti', 'Nakit İneği', 'Zayıf']
            df_result['Matris_Kategorisi'] = np.select(conditions, choices, default='Orta')
        
        return df_result
    
    def create_cannibalization_heatmap(self, cannibal_df: pd.DataFrame) -> go.Figure:
        """
        Kanibalizasyon ısı haritası
        """
        if cannibal_df.empty:
            return go.Figure()
        
        # Pivot tablo oluştur
        pivot_df = cannibal_df.pivot_table(
            values='Kanibalizasyon_Skoru',
            index='Paket',
            columns='Şirket',
            aggfunc='first'
        ).fillna(0)
        
        fig = px.imshow(
            pivot_df,
            text_auto='.0f',
            color_continuous_scale=['green', 'yellow', 'red'],
            title='Kanibalizasyon Isı Haritası',
            labels=dict(x='Şirket', y='Paket', color='Skor')
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=ExecutiveColors.TEXT_PRIMARY),
            height=500,
            template=ChartTemplates.executive_template()
        )
        
        return fig
    
    def create_growth_share_matrix(self, df: pd.DataFrame) -> go.Figure:
        """
        Büyüme/Pazar Payı Matrisi (BCG Matrix)
        """
        if self.latest_sales_col not in df.columns or self.latest_growth_col not in df.columns:
            return go.Figure()
        
        fig = go.Figure()
        
        # Şirket bazlı gruplama
        if 'Sirket' in df.columns:
            for sirket, sirket_df in df.groupby('Sirket'):
                avg_growth = sirket_df[self.latest_growth_col].mean() if self.latest_growth_col in sirket_df.columns else 0
                avg_share = sirket_df['Pazar_Payi_2024'].mean() if 'Pazar_Payi_2024' in sirket_df.columns else 0
                total_sales = sirket_df[self.latest_sales_col].sum()
                
                fig.add_trace(go.Scatter(
                    x=[avg_share],
                    y=[avg_growth],
                    mode='markers+text',
                    name=sirket,
                    marker=dict(
                        size=np.log1p(total_sales) * 2,
                        color=ExecutiveColors.CHART_1,
                        line=dict(width=2, color=ExecutiveColors.ACCENT_GOLD)
                    ),
                    text=[sirket[:15]],
                    textposition='top center',
                    hovertemplate=(
                        f'<b>{sirket}</b><br>' +
                        f'Pazar Payı: %{{x:.1f}}%<br>' +
                        f'Büyüme: %{{y:.1f}}%<br>' +
                        f'Satış: {total_sales:,.0f}₺<br>' +
                        '<extra></extra>'
                    )
                ))
        
        # BCG Matrix bölgeleri
        fig.add_shape(
            type='line',
            x0=10, y0=0, x1=10, y1=100,
            line=dict(color=ExecutiveColors.ACCENT_SILVER, width=1, dash='dash')
        )
        fig.add_shape(
            type='line',
            x0=0, y0=10, x1=100, y1=10,
            line=dict(color=ExecutiveColors.ACCENT_SILVER, width=1, dash='dash')
        )
        
        # Bölge etiketleri
        fig.add_annotation(x=25, y=80, text="⭐ Yıldızlar", showarrow=False,
                          font=dict(size=14, color=ExecutiveColors.ACCENT_GOLD))
        fig.add_annotation(x=5, y=80, text="❓ Soru İşaretleri", showarrow=False,
                          font=dict(size=14, color=ExecutiveColors.INFO))
        fig.add_annotation(x=25, y=5, text="🐄 Nakit İnekleri", showarrow=False,
                          font=dict(size=14, color=ExecutiveColors.SUCCESS))
        fig.add_annotation(x=5, y=5, text="⚠️ Zayıf Ürünler", showarrow=False,
                          font=dict(size=14, color=ExecutiveColors.DANGER))
        
        fig.update_layout(
            title=dict(
                text='<b>Büyüme-Pazar Payı Matrisi (BCG)</b>',
                font=dict(size=20, color=ExecutiveColors.ACCENT_GOLD),
                x=0.5
            ),
            xaxis=dict(
                title='Pazar Payı (%)',
                gridcolor=ExecutiveColors.GRID,
                range=[0, max(df['Pazar_Payi_2024'].max() + 5, 30)]
            ),
            yaxis=dict(
                title='Büyüme Oranı (%)',
                gridcolor=ExecutiveColors.GRID,
                range=[min(df[self.latest_growth_col].min() - 5, -10), 
                       max(df[self.latest_growth_col].max() + 5, 50)]
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=ExecutiveColors.TEXT_PRIMARY),
            height=600,
            template=ChartTemplates.executive_template()
        )
        
        return fig
    
    def _find_node(self, root: ProdPackNode, node_id: str) -> Optional[ProdPackNode]:
        """ID'ye göre düğüm bul"""
        if root.id == node_id:
            return root
        for child in root.children:
            found = self._find_node(child, node_id)
            if found:
                return found
        return None
    
    def get_pack_performance_table(self, selected_molecule: str, n_top: int = 20) -> pd.DataFrame:
        """
        Paket performans tablosu
        """
        if selected_molecule == 'Tüm Moleküller':
            df_filtered = self.df
        else:
            df_filtered = self.df[self.df['Molekul'] == selected_molecule]
        
        cols = ['Paket', 'Sirket', 'Marka']
        if self.latest_sales_col:
            cols.append(self.latest_sales_col)
        if self.prev_sales_col:
            cols.append(self.prev_sales_col)
        if self.latest_growth_col:
            cols.append(self.latest_growth_col)
        if 'Pazar_Payi_2024' in df_filtered.columns:
            cols.append('Pazar_Payi_2024')
        if 'CAGR_3Y' in df_filtered.columns:
            cols.append('CAGR_3Y')
        
        available_cols = [col for col in cols if col in df_filtered.columns]
        
        result_df = df_filtered[available_cols].copy()
        
        if self.latest_sales_col in result_df.columns:
            result_df = result_df.sort_values(self.latest_sales_col, ascending=False).head(n_top)
        
        return result_df

# ================================================
# 14. STRATEJİK TAHMİN VE ÖNGÖRÜ MOTORU
# ================================================

class StrategicForecastEngine:
    """
    Gelişmiş tahminleme motoru
    Holt-Winters, Prophet, ARIMA, Ensemble
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.sales_cols = [col for col in df.columns if re.search(r'Satis_20\d{2}', col)]
        self.sales_cols.sort(key=lambda x: int(re.search(r'20\d{2}', x).group()))
        
        self.yearly_sales = {}
        self.years = []
        self.sales_values = []
        
        self._prepare_time_series()
    
    def _prepare_time_series(self):
        """Zaman serisi hazırlığı"""
        for col in self.sales_cols:
            year_match = re.search(r'20\d{2}', col)
            if year_match:
                year = int(year_match.group())
                sales_sum = self.df[col].sum()
                self.yearly_sales[year] = sales_sum
                self.years.append(year)
                self.sales_values.append(sales_sum)
    
    @st.cache_data(ttl=3600)
    def forecast_holt_winters(_self, periods: int = 8) -> Optional[ForecastResult]:
        """
        Holt-Winters üstel düzeltme ile tahmin
        """
        if len(_self.sales_values) < 4:
            return None
        
        series = pd.Series(
            _self.sales_values,
            index=pd.date_range(start=f'{_self.years[0]}-01-01', periods=len(_self.years), freq='Y')
        )
        
        try:
            # Model seçimi - trend ve mevsimsellik kontrolü
            has_seasonality = len(_self.sales_values) >= 8
            
            if has_seasonality:
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=4,
                    initialization_method='estimated'
                )
            else:
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal=None,
                    initialization_method='estimated'
                )
            
            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)
            
            # Hata metrikleri
            fitted_values = fitted_model.fittedvalues
            mape = np.mean(np.abs((series - fitted_values) / series)) * 100
            rmse = np.sqrt(np.mean((series - fitted_values) ** 2))
            mae = np.mean(np.abs(series - fitted_values))
            r2 = 1 - (np.sum((series - fitted_values) ** 2) / np.sum((series - series.mean()) ** 2))
            
            # Güven aralıkları
            residuals = series - fitted_values
            std_resid = residuals.std()
            
            future_years = [_self.years[-1] + i + 1 for i in range(periods)]
            periods_str = [f'{y}' for y in future_years]
            
            # Büyüme oranı
            growth_rate = ((forecast.iloc[-1] - series.iloc[-1]) / series.iloc[-1]) * 100
            
            # CAGR
            cagr = ((forecast.iloc[-1] / series.iloc[0]) ** (1/(len(future_years) + len(_self.years) - 1)) - 1) * 100
            
            return ForecastResult(
                periods=periods_str,
                predictions=forecast.tolist(),
                lower_bound_80=(forecast - 1.28 * std_resid).tolist(),
                upper_bound_80=(forecast + 1.28 * std_resid).tolist(),
                lower_bound_95=(forecast - 1.96 * std_resid).tolist(),
                upper_bound_95=(forecast + 1.96 * std_resid).tolist(),
                model_type='Holt-Winters',
                mape=mape,
                rmse=rmse,
                mae=mae,
                r2=r2,
                growth_rate=growth_rate,
                cagr_forecast=cagr,
                seasonality_strength=0.7 if has_seasonality else 0.0,
                trend_strength=0.8,
                residual_std=std_resid
            )
            
        except Exception as e:
            st.warning(f"Holt-Winters tahmin hatası: {str(e)}")
            return None
    
    @st.cache_data(ttl=3600)
    def forecast_prophet(_self, periods: int = 8) -> Optional[ForecastResult]:
        """
        Facebook Prophet ile tahmin
        """
        if not PROPHET_AVAILABLE or len(_self.sales_values) < 4:
            return None
        
        try:
            df_prophet = pd.DataFrame({
                'ds': pd.date_range(start=f'{_self.years[0]}-01-01', periods=len(_self.years), freq='Y'),
                'y': _self.sales_values
            })
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            model.fit(df_prophet)
            
            future = model.make_future_dataframe(periods=periods, freq='Y')
            forecast = model.predict(future)
            
            # Son 'periods' kadar tahmini al
            forecast_tail = forecast.tail(periods)
            
            # Hata metrikleri
            y_true = df_prophet['y'].values
            y_pred = forecast['yhat'].values[:len(y_true)]
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
            
            future_years = [_self.years[-1] + i + 1 for i in range(periods)]
            periods_str = [f'{y}' for y in future_years]
            
            growth_rate = ((forecast_tail['yhat'].iloc[-1] - _self.sales_values[-1]) / _self.sales_values[-1]) * 100
            cagr = ((forecast_tail['yhat'].iloc[-1] / _self.sales_values[0]) ** (1/(len(future_years) + len(_self.years) - 1)) - 1) * 100
            
            return ForecastResult(
                periods=periods_str,
                predictions=forecast_tail['yhat'].tolist(),
                lower_bound_80=forecast_tail['yhat_lower'].tolist(),
                upper_bound_80=forecast_tail['yhat_upper'].tolist(),
                lower_bound_95=forecast_tail['yhat_lower'].tolist(),
                upper_bound_95=forecast_tail['yhat_upper'].tolist(),
                model_type='Prophet',
                mape=mape,
                rmse=rmse,
                mae=mae,
                r2=r2,
                growth_rate=growth_rate,
                cagr_forecast=cagr,
                seasonality_strength=0.8,
                trend_strength=0.9,
                residual_std=0.0
            )
            
        except Exception as e:
            st.warning(f"Prophet tahmin hatası: {str(e)}")
            return None
    
    @st.cache_data(ttl=3600)
    def forecast_arima(_self, periods: int = 8) -> Optional[ForecastResult]:
        """
        ARIMA/SARIMA ile tahmin
        """
        if not ARIMA_AVAILABLE or len(_self.sales_values) < 4:
            return None
        
        try:
            series = pd.Series(
                _self.sales_values,
                index=pd.date_range(start=f'{_self.years[0]}-01-01', periods=len(_self.years), freq='Y')
            )
            
            # Otomatik model seçimi
            model = auto_arima(
                series,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True)
            
            # Hata metrikleri
            y_true = series.values
            y_pred = model.predict_in_sample()
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
            
            future_years = [_self.years[-1] + i + 1 for i in range(periods)]
            periods_str = [f'{y}' for y in future_years]
            
            growth_rate = ((forecast[-1] - _self.sales_values[-1]) / _self.sales_values[-1]) * 100
            cagr = ((forecast[-1] / _self.sales_values[0]) ** (1/(len(future_years) + len(_self.years) - 1)) - 1) * 100
            
            return ForecastResult(
                periods=periods_str,
                predictions=forecast.tolist(),
                lower_bound_80=conf_int[:, 0].tolist(),
                upper_bound_80=conf_int[:, 1].tolist(),
                lower_bound_95=conf_int[:, 0].tolist(),
                upper_bound_95=conf_int[:, 1].tolist(),
                model_type='ARIMA',
                mape=mape,
                rmse=rmse,
                mae=mae,
                r2=r2,
                growth_rate=growth_rate,
                cagr_forecast=cagr,
                seasonality_strength=0.6,
                trend_strength=0.7,
                residual_std=0.0
            )
            
        except Exception as e:
            st.warning(f"ARIMA tahmin hatası: {str(e)}")
            return None
    
    @st.cache_data(ttl=3600)
    def forecast_ensemble(_self, periods: int = 8) -> Optional[ForecastResult]:
        """
        Ensemble tahmin (Holt-Winters + Prophet + ARIMA)
        """
        forecasts = []
        
        hw = _self.forecast_holt_winters(periods)
        if hw:
            forecasts.append(hw)
        
        prophet = _self.forecast_prophet(periods)
        if prophet:
            forecasts.append(prophet)
        
        arima = _self.forecast_arima(periods)
        if arima:
            forecasts.append(arima)
        
        if len(forecasts) < 2:
            return forecasts[0] if forecasts else None
        
        # Ensemble ağırlıkları (MAPE'e göre ters orantılı)
        weights = []
        for f in forecasts:
            weight = 1 / (f.mape + 1)  # MAPE küçükse ağırlık büyük
            weights.append(weight)
        
        weights = np.array(weights) / np.sum(weights)
        
        # Weighted average
        ensemble_predictions = np.zeros(periods)
        ensemble_lower_80 = np.zeros(periods)
        ensemble_upper_80 = np.zeros(periods)
        ensemble_lower_95 = np.zeros(periods)
        ensemble_upper_95 = np.zeros(periods)
        
        for i, f in enumerate(forecasts):
            ensemble_predictions += np.array(f.predictions) * weights[i]
            ensemble_lower_80 += np.array(f.lower_bound_80) * weights[i]
            ensemble_upper_80 += np.array(f.upper_bound_80) * weights[i]
            ensemble_lower_95 += np.array(f.lower_bound_95) * weights[i]
            ensemble_upper_95 += np.array(f.upper_bound_95) * weights[i]
        
        # Ortalama metrikler
        avg_mape = np.mean([f.mape for f in forecasts])
        avg_rmse = np.mean([f.rmse for f in forecasts])
        avg_mae = np.mean([f.mae for f in forecasts])
        avg_r2 = np.mean([f.r2 for f in forecasts])
        
        growth_rate = ((ensemble_predictions[-1] - _self.sales_values[-1]) / _self.sales_values[-1]) * 100
        cagr = ((ensemble_predictions[-1] / _self.sales_values[0]) ** (1/(periods + len(_self.years) - 1)) - 1) * 100
        
        return ForecastResult(
            periods=forecasts[0].periods,
            predictions=ensemble_predictions.tolist(),
            lower_bound_80=ensemble_lower_80.tolist(),
            upper_bound_80=ensemble_upper_80.tolist(),
            lower_bound_95=ensemble_lower_95.tolist(),
            upper_bound_95=ensemble_upper_95.tolist(),
            model_type='Ensemble',
            mape=avg_mape,
            rmse=avg_rmse,
            mae=avg_mae,
            r2=avg_r2,
            growth_rate=growth_rate,
            cagr_forecast=cagr,
            seasonality_strength=np.mean([f.seasonality_strength for f in forecasts]),
            trend_strength=np.mean([f.trend_strength for f in forecasts]),
            residual_std=np.mean([f.residual_std for f in forecasts])
        )
    
    def plot_forecast_comparison(self, forecast_results: Dict[str, ForecastResult]) -> go.Figure:
        """
        Tahmin modelleri karşılaştırma grafiği
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Tahmin Karşılaştırması', 'MAPE', 'RMSE', 'Büyüme Oranı'),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'bar'}]
            ]
        )
        
        # Ana tahmin grafiği
        colors = [ExecutiveColors.CHART_1, ExecutiveColors.CHART_2, 
                 ExecutiveColors.CHART_3, ExecutiveColors.CHART_4]
        
        for i, (name, forecast) in enumerate(forecast_results.items()):
            # Tarihsel veri
            if i == 0:
                fig.add_trace(
                    go.Scatter(
                        x=self.years,
                        y=self.sales_values,
                        mode='lines+markers',
                        name='Tarihsel',
                        line=dict(color=ExecutiveColors.ACCENT_GOLD, width=4),
                        marker=dict(size=10)
                    ),
                    row=1, col=1
                )
            
            # Tahmin
            fig.add_trace(
                go.Scatter(
                    x=forecast.periods,
                    y=forecast.predictions,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=3, dash='dash'),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
        
        # MAPE karşılaştırması
        for i, (name, forecast) in enumerate(forecast_results.items()):
            fig.add_trace(
                go.Bar(
                    x=[name],
                    y=[forecast.mape],
                    name=name,
                    marker_color=colors[i % len(colors)],
                    text=[f'{forecast.mape:.1f}%'],
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        # RMSE karşılaştırması
        for i, (name, forecast) in enumerate(forecast_results.items()):
            fig.add_trace(
                go.Bar(
                    x=[name],
                    y=[forecast.rmse],
                    name=name,
                    marker_color=colors[i % len(colors)],
                    text=[f'{forecast.rmse:,.0f}'],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # Büyüme oranı karşılaştırması
        for i, (name, forecast) in enumerate(forecast_results.items()):
            fig.add_trace(
                go.Bar(
                    x=[name],
                    y=[forecast.growth_rate],
                    name=name,
                    marker_color=colors[i % len(colors)],
                    text=[f'{forecast.growth_rate:.1f}%'],
                    textposition='auto'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(
                text='<b>Tahmin Modelleri Karşılaştırması</b>',
                font=dict(size=20, color=ExecutiveColors.ACCENT_GOLD),
                x=0.5
            ),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=ExecutiveColors.TEXT_PRIMARY),
            height=700,
            template=ChartTemplates.executive_template()
        )
        
        return fig

# ================================================
# 15. RİSK VE ANOMALİ TESPİT MOTORU
# ================================================

class RiskAnomalyDetector:
    """
    Gelişmiş risk ve anomali tespit motoru
    Isolation Forest, LOF, One-Class SVM, Elliptic Envelope
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.sales_cols = [col for col in df.columns if re.search(r'Satis_20\d{2}', col)]
        self.feature_cols = []
        self.anomaly_scores = {}
        self.risk_scores = {}
        
    def prepare_features(self) -> np.ndarray:
        """
        Anomali tespiti için özellik mühendisliği
        """
        features = []
        feature_names = []
        
        # 1. Satış özellikleri (son 3 yıl)
        if len(self.sales_cols) >= 1:
            for col in self.sales_cols[-3:]:
                self.df[f'{col}_norm'] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
                features.append(f'{col}_norm')
                feature_names.append(f'{col}_norm')
        
        # 2. Büyüme oranları
        growth_cols = [col for col in self.df.columns if 'Buyume_' in col]
        for col in growth_cols[-3:]:
            if col in self.df.columns:
                self.df[f'{col}_norm'] = self.df[col].fillna(0)
                features.append(f'{col}_norm')
                feature_names.append(f'{col}_norm')
        
        # 3. Pazar payı
        if 'Pazar_Payi_2024' in self.df.columns:
            self.df['Pazar_Payi_norm'] = (self.df['Pazar_Payi_2024'] - self.df['Pazar_Payi_2024'].mean()) / self.df['Pazar_Payi_2024'].std()
            features.append('Pazar_Payi_norm')
            feature_names.append('Pazar_Payi_norm')
        
        # 4. CAGR
        if 'CAGR_3Y' in self.df.columns:
            self.df['CAGR_3Y_norm'] = self.df['CAGR_3Y'].fillna(0)
            features.append('CAGR_3Y_norm')
            feature_names.append('CAGR_3Y_norm')
        
        # 5. Performans indeksi
        if 'Performans_Indeksi' in self.df.columns:
            features.append('Performans_Indeksi')
            feature_names.append('Performans_Indeksi')
        
        self.feature_cols = features
        
        if not features:
            return np.array([])
        
        X = self.df[features].fillna(0).values
        return X
    
    def detect_isolation_forest(self, contamination: float = 0.1) -> pd.DataFrame:
        """
        Isolation Forest ile anomali tespiti
        """
        X = self.prepare_features()
        
        if X.shape[0] < 10 or X.shape[1] < 2:
            self.df['Anomali_IF'] = 1
            self.df['Anomali_Skoru_IF'] = 0
            self.df['Risk_IF'] = 'Düşük'
            return self.df
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples='auto',
            bootstrap=False,
            n_jobs=-1
        )
        
        predictions = iso_forest.fit_predict(X)
        scores = iso_forest.score_samples(X)
        
        # Normalize skorlar (0-1 arası, 1 = normal, 0 = anormal)
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        self.df['Anomali_IF'] = predictions
        self.df['Anomali_Skoru_IF'] = normalized_scores
        self.df['Risk_IF'] = np.where(
            predictions == -1,
            np.where(normalized_scores < 0.3, 'Kritik', 'Yüksek'),
            np.where(normalized_scores > 0.7, 'Düşük', 'Orta')
        )
        
        return self.df
    
    def detect_lof(self, contamination: float = 0.1) -> pd.DataFrame:
        """
        Local Outlier Factor ile anomali tespiti
        """
        X = self.prepare_features()
        
        if X.shape[0] < 20 or X.shape[1] < 2:
            self.df['Anomali_LOF'] = 1
            self.df['Anomali_Skoru_LOF'] = 0
            return self.df
        
        lof = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=20,
            novelty=False,
            n_jobs=-1
        )
        
        predictions = lof.fit_predict(X)
        scores = -lof.negative_outlier_factor_  # Negatif değerlerden pozitife çevir
        
        # Normalize
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        self.df['Anomali_LOF'] = predictions
        self.df['Anomali_Skoru_LOF'] = normalized_scores
        
        return self.df
    
    def detect_one_class_svm(self, nu: float = 0.1) -> pd.DataFrame:
        """
        One-Class SVM ile anomali tespiti
        """
        X = self.prepare_features()
        
        if X.shape[0] < 10 or X.shape[1] < 2:
            self.df['Anomali_SVM'] = 1
            self.df['Anomali_Skoru_SVM'] = 0
            return self.df
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        svm = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='scale'
        )
        
        predictions = svm.fit_predict(X_scaled)
        
        # Karar fonksiyonu skorları
        scores = svm.decision_function(X_scaled)
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        self.df['Anomali_SVM'] = predictions
        self.df['Anomali_Skoru_SVM'] = normalized_scores
        
        return self.df
    
    def detect_elliptic_envelope(self, contamination: float = 0.1) -> pd.DataFrame:
        """
        Elliptic Envelope ile anomali tespiti
        """
        X = self.prepare_features()
        
        if X.shape[0] < 20 or X.shape[1] < 2:
            self.df['Anomali_EE'] = 1
            self.df['Anomali_Skoru_EE'] = 0
            return self.df
        
        try:
            ee = EllipticEnvelope(
                contamination=contamination,
                random_state=42,
                support_fraction=0.7
            )
            
            predictions = ee.fit_predict(X)
            scores = ee.decision_function(X)
            
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            
            self.df['Anomali_EE'] = predictions
            self.df['Anomali_Skoru_EE'] = normalized_scores
            
        except:
            self.df['Anomali_EE'] = 1
            self.df['Anomali_Skoru_EE'] = 0
        
        return self.df
    
    def ensemble_anomaly_detection(self) -> pd.DataFrame:
        """
        Ensemble anomali tespiti (tüm algoritmaların ortalaması)
        """
        # Tüm algoritmaları çalıştır
        self.detect_isolation_forest()
        self.detect_lof()
        self.detect_one_class_svm()
        self.detect_elliptic_envelope()
        
        # Ensemble skor
        score_cols = ['Anomali_Skoru_IF', 'Anomali_Skoru_LOF', 
                     'Anomali_Skoru_SVM', 'Anomali_Skoru_EE']
        
        available_cols = [col for col in score_cols if col in self.df.columns]
        
        if available_cols:
            self.df['Anomali_Skoru_Ensemble'] = self.df[available_cols].mean(axis=1)
            
            # Anomali kararı (3/4 algoritma anormal derse)
            anomaly_cols = ['Anomali_IF', 'Anomali_LOF', 'Anomali_SVM', 'Anomali_EE']
            avail_anom = [col for col in anomaly_cols if col in self.df.columns]
            
            if avail_anom:
                self.df['Anomali_Sayisi'] = (self.df[avail_anom] == -1).sum(axis=1)
                self.df['Anomali_Ensemble'] = np.where(self.df['Anomali_Sayisi'] >= len(avail_anom) * 0.5, -1, 1)
                
                # Risk seviyesi
                conditions = [
                    (self.df['Anomali_Ensemble'] == -1) & (self.df['Anomali_Skoru_Ensemble'] < 0.3),
                    (self.df['Anomali_Ensemble'] == -1) & (self.df['Anomali_Skoru_Ensemble'] < 0.5),
                    (self.df['Anomali_Ensemble'] == 1) & (self.df['Anomali_Skoru_Ensemble'] < 0.7),
                    (self.df['Anomali_Ensemble'] == 1) & (self.df['Anomali_Skoru_Ensemble'] >= 0.7)
                ]
                choices = ['🔴 Kritik Risk', '🟠 Yüksek Risk', '🟡 Orta Risk', '🟢 Düşük Risk']
                
                self.df['Risk_Seviyesi'] = np.select(conditions, choices, default='⚪ Belirlenemedi')
        
        return self.df
    
    def calculate_financial_risk(self) -> pd.DataFrame:
        """
        Finansal risk skorlaması
        """
        if 'Risk_Seviyesi' not in self.df.columns:
            self.ensemble_anomaly_detection()
        
        # Satış oynaklığı
        if len(self.sales_cols) >= 3:
            sales_data = self.df[self.sales_cols[-3:]].values
            self.df['Satis_Oynakligi'] = np.std(sales_data, axis=1) / (np.mean(sales_data, axis=1) + 1)
        
        # Büyüme istikrarsızlığı
        growth_cols = [col for col in self.df.columns if 'Buyume_' in col]
        if len(growth_cols) >= 2:
            growth_data = self.df[growth_cols[-2:]].values
            self.df['Buyume_Degisimi'] = np.abs(np.diff(growth_data, axis=1)).flatten()
        
        # Pazar payı riski
        if 'Pazar_Payi_2024' in self.df.columns:
            avg_share = self.df['Pazar_Payi_2024'].mean()
            self.df['Pazar_Payi_Riski'] = np.where(
                self.df['Pazar_Payi_2024'] < avg_share * 0.5,
                1.0,
                self.df['Pazar_Payi_2024'] / avg_share
            )
        
        # Kompozit risk skoru
        risk_components = []
        
        if 'Anomali_Skoru_Ensemble' in self.df.columns:
            risk_components.append((1 - self.df['Anomali_Skoru_Ensemble']) * 0.4)
        
        if 'Satis_Oynakligi' in self.df.columns:
            risk_components.append(self.df['Satis_Oynakligi'].fillna(0.5) * 0.3)
        
        if 'Buyume_Degisimi' in self.df.columns:
            risk_components.append(self.df['Buyume_Degisimi'].fillna(0.5) * 0.2)
        
        if 'Pazar_Payi_Riski' in self.df.columns:
            risk_components.append((1 - self.df['Pazar_Payi_Riski'].clip(0, 1)) * 0.1)
        
        if risk_components:
            self.df['Finansal_Risk_Skoru'] = np.sum(risk_components, axis=0)
            self.df['Finansal_Risk'] = pd.cut(
                self.df['Finansal_Risk_Skoru'],
                bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
                labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek', 'Kritik']
            )
        
        return self.df
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Risk özet istatistikleri
        """
        summary = {}
        
        if 'Risk_Seviyesi' in self.df.columns:
            risk_counts = self.df['Risk_Seviyesi'].value_counts()
            summary['risk_distribution'] = risk_counts.to_dict()
            summary['critical_count'] = int(risk_counts.get('🔴 Kritik Risk', 0))
            summary['high_count'] = int(risk_counts.get('🟠 Yüksek Risk', 0))
            summary['medium_count'] = int(risk_counts.get('🟡 Orta Risk', 0))
            summary['low_count'] = int(risk_counts.get('🟢 Düşük Risk', 0))
        
        if 'Finansal_Risk' in self.df.columns:
            fin_risk_counts = self.df['Finansal_Risk'].value_counts()
            summary['financial_risk'] = fin_risk_counts.to_dict()
        
        if self.sales_cols:
            latest_sales = self.sales_cols[-1]
            risk_by_sales = self.df.groupby('Risk_Seviyesi')[latest_sales].sum().to_dict() if 'Risk_Seviyesi' in self.df.columns else {}
            summary['sales_at_risk'] = risk_by_sales
        
        return summary
    
    def plot_risk_dashboard(self) -> go.Figure:
        """
        Risk dashboard görselleştirmesi
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Risk Dağılımı', 'Finansal Risk', 'Risk-Satış İlişkisi',
                          'Anomali Skor Dağılımı', 'Risk Matrisi', 'Zaman Bazlı Risk'),
            specs=[
                [{'type': 'pie'}, {'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'heatmap'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. Risk dağılımı (Pie)
        if 'Risk_Seviyesi' in self.df.columns:
            risk_counts = self.df['Risk_Seviyesi'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.4,
                    marker_colors=[ExecutiveColors.DANGER, ExecutiveColors.WARNING,
                                  ExecutiveColors.INFO, ExecutiveColors.SUCCESS]
                ),
                row=1, col=1
            )
        
        # 2. Finansal risk (Bar)
        if 'Finansal_Risk' in self.df.columns:
            fin_risk_counts = self.df['Finansal_Risk'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=fin_risk_counts.index,
                    y=fin_risk_counts.values,
                    marker_color=ExecutiveColors.CHART_2
                ),
                row=1, col=2
            )
        
        # 3. Risk-Satış ilişkisi
        if 'Risk_Seviyesi' in self.df.columns and self.sales_cols:
            latest_sales = self.sales_cols[-1]
            risk_sales = self.df.groupby('Risk_Seviyesi')[latest_sales].sum().reset_index()
            fig.add_trace(
                go.Bar(
                    x=risk_sales['Risk_Seviyesi'],
                    y=risk_sales[latest_sales],
                    marker_color=ExecutiveColors.CHART_3,
                    name='Risk Bazlı Satış'
                ),
                row=1, col=3
            )
        
        # 4. Anomali skor dağılımı
        if 'Anomali_Skoru_Ensemble' in self.df.columns:
            fig.add_trace(
                go.Histogram(
                    x=self.df['Anomali_Skoru_Ensemble'],
                    nbinsx=30,
                    marker_color=ExecutiveColors.CHART_4
                ),
                row=2, col=1
            )
        
        # 5. Risk matrisi (Büyüme vs Pazar Payı)
        if self.sales_cols and 'Buyume_Orani' in self.df.columns and 'Pazar_Payi_2024' in self.df.columns:
            risk_matrix = pd.crosstab(
                pd.cut(self.df['Buyume_Orani'], bins=5),
                pd.cut(self.df['Pazar_Payi_2024'], bins=5)
            )
            fig.add_trace(
                go.Heatmap(
                    z=risk_matrix.values,
                    x=[f'{i:.1f}' for i in risk_matrix.columns],
                    y=[f'{i:.1f}' for i in risk_matrix.index],
                    colorscale='RdYlGn_r'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(
                text='<b>Risk ve Anomali Dashboard</b>',
                font=dict(size=20, color=ExecutiveColors.ACCENT_GOLD),
                x=0.5
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=ExecutiveColors.TEXT_PRIMARY),
            height=800,
            template=ChartTemplates.executive_template()
        )
        
        return fig

# ================================================
# 16. SEGMENTASYON VE KÜMELEME MOTORU
# ================================================

class SegmentationEngine:
    """
    Gelişmiş segmentasyon ve kümeleme motoru
    PCA, K-Means, Hierarchical, DBSCAN, Gaussian Mixture
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.sales_cols = [col for col in df.columns if re.search(r'Satis_20\d{2}', col)]
        self.features = []
        self.segmentation_results = {}
        
    def prepare_segmentation_features(self) -> np.ndarray:
        """
        Segmentasyon için özellik mühendisliği
        """
        features = []
        
        # 1. Satış (son yıl)
        if self.sales_cols:
            latest_sales = self.sales_cols[-1]
            self.df['feature_sales'] = np.log1p(self.df[latest_sales].fillna(0))
            features.append('feature_sales')
        
        # 2. Büyüme oranı
        growth_cols = [col for col in self.df.columns if 'Buyume_' in col]
        if growth_cols:
            self.df['feature_growth'] = self.df[growth_cols[-1]].fillna(0).clip(-100, 500)
            features.append('feature_growth')
        
        # 3. Pazar payı
        if 'Pazar_Payi_2024' in self.df.columns:
            self.df['feature_share'] = self.df['Pazar_Payi_2024'].fillna(0)
            features.append('feature_share')
        
        # 4. CAGR
        if 'CAGR_3Y' in self.df.columns:
            self.df['feature_cagr'] = self.df['CAGR_3Y'].fillna(0).clip(-50, 100)
            features.append('feature_cagr')
        
        # 5. Fiyat (varsa)
        price_cols = [col for col in self.df.columns if re.search(r'Fiyat_20\d{2}', col)]
        if price_cols:
            self.df['feature_price'] = np.log1p(self.df[price_cols[-1]].fillna(self.df[price_cols[-1]].median()))
            features.append('feature_price')
        
        # 6. Risk skoru
        if 'Anomali_Skoru_Ensemble' in self.df.columns:
            self.df['feature_risk'] = self.df['Anomali_Skoru_Ensemble']
            features.append('feature_risk')
        
        self.features = features
        
        if len(features) < 2:
            return np.array([])
        
        X = self.df[features].fillna(0).values
        return X
    
    def pca_analysis(self, n_components: int = 2) -> np.ndarray:
        """
        PCA ile boyut indirgeme
        """
        X = self.prepare_segmentation_features()
        
        if X.shape[0] < 5 or X.shape[1] < 2:
            return np.array([])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=min(n_components, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        self.df['PCA1'] = X_pca[:, 0]
        if X_pca.shape[1] > 1:
            self.df['PCA2'] = X_pca[:, 1]
        
        # Varyans açıklama oranları
        self.pca_explained_variance = pca.explained_variance_ratio_
        self.pca_components = pca.components_
        
        return X_pca
    
    def kmeans_segmentation(self, n_clusters: int = 4) -> pd.DataFrame:
        """
        K-Means ile segmentasyon
        """
        X = self.pca_analysis()
        
        if X.size == 0:
            return self.df
        
        # Optimal küme sayısını bul (Elbow metodu)
        if n_clusters == 'auto':
            inertias = []
            sil_scores = []
            K_range = range(2, min(10, len(self.df) // 5))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                inertias.append(kmeans.inertia_)
                if len(set(labels)) > 1:
                    sil_scores.append(silhouette_score(X, labels))
                else:
                    sil_scores.append(0)
            
            # En iyi silhouette skoruna göre
            if sil_scores:
                n_clusters = K_range[np.argmax(sil_scores)]
            else:
                n_clusters = 4
        
        # K-Means uygula
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X)
        
        self.df['Segment_KMeans'] = labels
        self.df['Segment_Uzaklik'] = kmeans.transform(X).min(axis=1)
        
        # Silhouette skoru
        if len(set(labels)) > 1:
            self.silhouette_score_kmeans = silhouette_score(X, labels)
        else:
            self.silhouette_score_kmeans = 0
        
        # Segment isimlendirme
        self._name_segments('Segment_KMeans', X)
        
        return self.df
    
    def hierarchical_clustering(self, n_clusters: int = 4) -> pd.DataFrame:
        """
        Hiyerarşik kümeleme
        """
        X = self.pca_analysis(n_components=min(5, len(self.features)))
        
        if X.size == 0:
            return self.df
        
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(X)
        
        self.df['Segment_Hierarchical'] = labels
        
        return self.df
    
    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
        """
        DBSCAN ile yoğunluk bazlı kümeleme
        """
        X = self.pca_analysis()
        
        if X.size == 0:
            return self.df
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        self.df['Segment_DBSCAN'] = labels
        
        return self.df
    
    def gaussian_mixture_segmentation(self, n_components: int = 4) -> pd.DataFrame:
        """
        Gaussian Mixture Model ile segmentasyon
        """
        X = self.pca_analysis()
        
        if X.size == 0:
            return self.df
        
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(X)
        probs = gmm.predict_proba(X)
        
        self.df['Segment_GMM'] = labels
        self.df['Segment_GMM_Confidence'] = probs.max(axis=1)
        
        return self.df
    
    def ensemble_segmentation(self) -> pd.DataFrame:
        """
        Ensemble segmentasyon (tüm algoritmaların oylaması)
        """
        self.kmeans_segmentation()
        self.hierarchical_clustering()
        self.dbscan_clustering()
        self.gaussian_mixture_segmentation()
        
        # Oylama
        segment_cols = ['Segment_KMeans', 'Segment_Hierarchical', 'Segment_GMM']
        available_cols = [col for col in segment_cols if col in self.df.columns]
        
        if available_cols:
            # Mode (en sık görülen değer)
            self.df['Segment'] = self.df[available_cols].mode(axis=1)[0]
            
            # Güven skoru
            self.df['Segment_Guven'] = self.df[available_cols].apply(
                lambda x: x.value_counts().iloc[0] / len(x), axis=1
            )
            
            # Segment isimlendirme
            self._name_segments('Segment')
        
        return self.df
    
    def _name_segments(self, segment_col: str, X: np.ndarray = None):
        """
        Segmentleri özelliklerine göre isimlendir
        """
        if segment_col not in self.df.columns:
            return
        
        segment_names = {}
        
        for segment_id in self.df[segment_col].unique():
            if pd.isna(segment_id):
                continue
            
            mask = self.df[segment_col] == segment_id
            segment_df = self.df[mask]
            
            # Ortalama değerler
            avg_sales = segment_df['feature_sales'].mean() if 'feature_sales' in segment_df.columns else 0
            avg_growth = segment_df['feature_growth'].mean() if 'feature_growth' in segment_df.columns else 0
            avg_share = segment_df['feature_share'].mean() if 'feature_share' in segment_df.columns else 0
            avg_risk = segment_df['feature_risk'].mean() if 'feature_risk' in segment_df.columns else 0.5
            
            # Segment isimlendirme mantığı
            if avg_share > 70:  # Çok yüksek pazar payı
                if avg_growth > 20:
                    name = '👑 Pazar Liderleri'
                else:
                    name = '🐄 Nakit İnekleri'
            elif avg_share > 30:  # Yüksek pazar payı
                if avg_growth > 15:
                    name = '⭐ Yıldız Ürünler'
                else:
                    name = '🏆 Olgun Ürünler'
            elif avg_share > 10:  # Orta pazar payı
                if avg_growth > 25:
                    name = '🚀 Yükselen Yıldızlar'
                elif avg_growth > 10:
                    name = '📈 Büyüyen Ürünler'
                else:
                    name = '⚖️ İstikrarlı Ürünler'
            else:  # Düşük pazar payı
                if avg_growth > 30:
                    name = '🎯 Potansiyel Vaat Edenler'
                elif avg_growth > 10:
                    name = '❓ Soru İşaretleri'
                elif avg_growth < -10:
                    name = '⚠️ Gerileyen Ürünler'
                else:
                    name = '📦 Niş Ürünler'
            
            # Risk faktörü
            if avg_risk < 0.3:
                name = '🔴 ' + name + ' (Yüksek Risk)'
            elif avg_risk < 0.6:
                name = '🟡 ' + name + ' (Orta Risk)'
            
            segment_names[segment_id] = name
        
        self.df[f'{segment_col}_Adi'] = self.df[segment_col].map(segment_names)
    
    def get_segmentation_summary(self) -> pd.DataFrame:
        """
        Segmentasyon özet tablosu
        """
        if 'Segment_Adi' not in self.df.columns:
            return pd.DataFrame()
        
        summary = []
        
        for segment in self.df['Segment_Adi'].unique():
            seg_df = self.df[self.df['Segment_Adi'] == segment]
            
            # Temel metrikler
            row = {
                'Segment': segment,
                'Ürün Sayısı': len(seg_df),
                'Ürün Oranı': f"{len(seg_df) / len(self.df) * 100:.1f}%",
                'Toplam Satış': seg_df[self.sales_cols[-1]].sum() if self.sales_cols else 0,
                'Satış Oranı': f"{seg_df[self.sales_cols[-1]].sum() / self.df[self.sales_cols[-1]].sum() * 100:.1f}%" if self.sales_cols else "0%",
                'Ort. Büyüme': seg_df['feature_growth'].mean() if 'feature_growth' in seg_df.columns else 0,
                'Ort. Pazar Payı': seg_df['feature_share'].mean() if 'feature_share' in seg_df.columns else 0,
                'Ort. Risk Skoru': seg_df['feature_risk'].mean() if 'feature_risk' in seg_df.columns else 0
            }
            
            summary.append(row)
        
        df_summary = pd.DataFrame(summary)
        
        if self.sales_cols:
            df_summary = df_summary.sort_values('Toplam Satış', ascending=False)
        
        return df_summary
    
    def plot_segmentation_dashboard(self) -> go.Figure:
        """
        Segmentasyon dashboard görselleştirmesi
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('PCA Segmentasyon Haritası', 'Segment Dağılımı', 'Segment Performansı',
                          'Segment-Satış İlişkisi', 'Segment-Büyüme Matrisi', 'Segment Risk Profili'),
            specs=[
                [{'type': 'scatter'}, {'type': 'pie'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'bar'}]
            ]
        )
        
        # 1. PCA Segmentasyon Haritası
        if 'PCA1' in self.df.columns and 'PCA2' in self.df.columns and 'Segment_Adi' in self.df.columns:
            for segment in self.df['Segment_Adi'].unique():
                seg_df = self.df[self.df['Segment_Adi'] == segment]
                fig.add_trace(
                    go.Scatter(
                        x=seg_df['PCA1'],
                        y=seg_df['PCA2'],
                        mode='markers',
                        name=segment[:20],
                        marker=dict(
                            size=8,
                            opacity=0.7
                        ),
                        text=seg_df['Paket'] if 'Paket' in seg_df.columns else None
                    ),
                    row=1, col=1
                )
        
        # 2. Segment Dağılımı
        if 'Segment_Adi' in self.df.columns:
            seg_counts = self.df['Segment_Adi'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=seg_counts.index,
                    values=seg_counts.values,
                    hole=0.4
                ),
                row=1, col=2
            )
        
        # 3. Segment Performansı (Satış)
        if 'Segment_Adi' in self.df.columns and self.sales_cols:
            seg_sales = self.df.groupby('Segment_Adi')[self.sales_cols[-1]].sum().reset_index()
            fig.add_trace(
                go.Bar(
                    x=seg_sales['Segment_Adi'],
                    y=seg_sales[self.sales_cols[-1]],
                    marker_color=ExecutiveColors.CHART_2
                ),
                row=1, col=3
            )
        
        # 4. Segment-Satış İlişkisi
        if 'Segment_Adi' in self.df.columns and 'feature_sales' in self.df.columns:
            fig.add_trace(
                go.Box(
                    x=self.df['Segment_Adi'],
                    y=self.df['feature_sales'],
                    marker_color=ExecutiveColors.CHART_3
                ),
                row=2, col=1
            )
        
        # 5. Segment-Büyüme Matrisi
        if 'feature_growth' in self.df.columns and 'feature_share' in self.df.columns and 'Segment_Adi' in self.df.columns:
            for segment in self.df['Segment_Adi'].unique():
                seg_df = self.df[self.df['Segment_Adi'] == segment]
                fig.add_trace(
                    go.Scatter(
                        x=seg_df['feature_share'],
                        y=seg_df['feature_growth'],
                        mode='markers',
                        name=segment[:15],
                        marker=dict(size=6),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # 6. Segment Risk Profili
        if 'Segment_Adi' in self.df.columns and 'feature_risk' in self.df.columns:
            seg_risk = self.df.groupby('Segment_Adi')['feature_risk'].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=seg_risk['Segment_Adi'],
                    y=seg_risk['feature_risk'],
                    marker_color=ExecutiveColors.CHART_4
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            title=dict(
                text='<b>Segmentasyon Analiz Dashboard</b>',
                font=dict(size=20, color=ExecutiveColors.ACCENT_GOLD),
                x=0.5
            ),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=ExecutiveColors.TEXT_PRIMARY),
            height=900,
            template=ChartTemplates.executive_template()
        )
        
        return fig

# ================================================
# 17. EXECUTIVE UI VE GÖSTERGE PANELİ
# ================================================

class ExecutiveUI:
    """
    Executive Dark Mode UI Bileşenleri
    Lacivert, Gümüş, Altın teması
    Insight Box, Metrik Kartları, Dashboard
    """
    
    @staticmethod
    def apply_theme():
        """Executive Dark Mode temasını uygula"""
        st.markdown(f"""
        <style>
            /* Ana arkaplan - Lacivert gradyan */
            .stApp {{
                background: linear-gradient(135deg, {ExecutiveColors.PRIMARY}, {ExecutiveColors.SECONDARY});
                background-attachment: fixed;
            }}
            
            /* Ana container */
            .main > div {{
                background-color: transparent;
            }}
            
            /* Executive Kartlar */
            .executive-card {{
                background: rgba(30, 58, 95, 0.7);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 1.5rem;
                border: 1px solid {ExecutiveColors.ACCENT_SILVER};
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
                margin-bottom: 1rem;
                transition: transform 0.3s ease;
            }}
            
            .executive-card:hover {{
                transform: translateY(-5px);
                border-color: {ExecutiveColors.ACCENT_GOLD};
                box-shadow: 0 12px 48px rgba(212, 175, 55, 0.2);
            }}
            
            /* Insight Box - Yönetici Özeti */
            .insight-box {{
                background: linear-gradient(145deg, rgba(212, 175, 55, 0.1), rgba(192, 192, 192, 0.05));
                border-left: 6px solid {ExecutiveColors.ACCENT_GOLD};
                border-radius: 10px;
                padding: 1.25rem;
                margin: 1rem 0;
                font-size: 1rem;
                color: {ExecutiveColors.TEXT_PRIMARY};
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            
            .insight-box strong {{
                color: {ExecutiveColors.ACCENT_GOLD};
                font-size: 1.1rem;
            }}
            
            /* Metrik Kartları - Altın Çerçeveli */
            .metric-card {{
                background: linear-gradient(145deg, {ExecutiveColors.SURFACE}, {ExecutiveColors.PRIMARY});
                border: 1px solid {ExecutiveColors.ACCENT_GOLD};
                border-radius: 12px;
                padding: 1.2rem;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            
            .metric-value {{
                font-size: 2.2rem;
                font-weight: 800;
                color: {ExecutiveColors.ACCENT_GOLD};
                margin: 0.5rem 0;
            }}
            
            .metric-label {{
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: {ExecutiveColors.TEXT_SECONDARY};
            }}
            
            /* Başlıklar */
            .executive-title {{
                font-size: 3rem;
                background: linear-gradient(135deg, {ExecutiveColors.ACCENT_GOLD}, {ExecutiveColors.ACCENT_SILVER});
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 900;
                margin-bottom: 0.5rem;
            }}
            
            .section-title {{
                font-size: 1.8rem;
                color: {ExecutiveColors.ACCENT_GOLD};
                font-weight: 700;
                margin: 2rem 0 1rem 0;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid {ExecutiveColors.ACCENT_GOLD};
            }}
            
            /* Butonlar */
            .stButton > button {{
                background: linear-gradient(145deg, {ExecutiveColors.ACCENT_GOLD}, #B8860B);
                color: {ExecutiveColors.PRIMARY};
                font-weight: 700;
                border: none;
                border-radius: 8px;
                padding: 0.6rem 1.2rem;
                transition: all 0.3s ease;
            }}
            
            .stButton > button:hover {{
                background: linear-gradient(145deg, #FFD700, {ExecutiveColors.ACCENT_GOLD});
                transform: scale(1.05);
                box-shadow: 0 4px 12px rgba(212, 175, 55, 0.4);
            }}
            
            /* Sekmeler */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 2px;
                background-color: {ExecutiveColors.SURFACE};
                padding: 5px;
                border-radius: 12px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                height: 50px;
                border-radius: 8px;
                color: {ExecutiveColors.TEXT_SECONDARY};
                font-weight: 600;
            }}
            
            .stTabs [aria-selected="true"] {{
                background-color: {ExecutiveColors.ACCENT_GOLD}20;
                color: {ExecutiveColors.ACCENT_GOLD};
                border-bottom: 2px solid {ExecutiveColors.ACCENT_GOLD};
            }}
            
            /* Dataframe */
            .stDataFrame {{
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid {ExecutiveColors.ACCENT_SILVER};
            }}
            
            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {ExecutiveColors.PRIMARY}, {ExecutiveColors.SECONDARY});
                border-right: 1px solid {ExecutiveColors.ACCENT_GOLD};
            }}
            
            /* Progress bar */
            .stProgress > div > div > div {{
                background: linear-gradient(90deg, {ExecutiveColors.ACCENT_GOLD}, {ExecutiveColors.ACCENT_SILVER});
            }}
            
            /* Animasyonlar */
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .fade-in {{
                animation: fadeIn 0.6s ease-out;
            }}
        </style>
        
        <!-- Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
        <style>
            * {{
                font-family: 'Inter', sans-serif;
            }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def insight_box(title: str, content: str, metric: str = None, icon: str = "💡"):
        """
        Yönetici Özeti (Insight Box)
        """
        metric_html = f'<div style="margin-top: 0.75rem; font-size: 1.2rem; color: {ExecutiveColors.ACCENT_GOLD};">{metric}</div>' if metric else ''
        
        st.markdown(f"""
        <div class="insight-box fade-in">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.8rem; margin-right: 0.75rem;">{icon}</span>
                <strong style="font-size: 1.2rem;">{title}</strong>
            </div>
            <div style="margin-left: 2.5rem; color: {ExecutiveColors.TEXT_SECONDARY};">{content}</div>
            {metric_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metric_card(label: str, value: str, delta: str = None, icon: str = "📊"):
        """
        Executive metrik kartı
        """
        delta_html = f'<div style="color: {ExecutiveColors.ACCENT_SILVER}; font-size: 0.9rem; margin-top: 0.3rem;">{delta}</div>' if delta else ''
        
        st.markdown(f"""
        <div class="metric-card fade-in">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="metric-label">{icon} {label}</span>
            </div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def kpi_row(kpis: List[Dict[str, str]]):
        """
        KPI satırı
        """
        cols = st.columns(len(kpis))
        
        for i, kpi in enumerate(kpis):
            with cols[i]:
                ExecutiveUI.metric_card(
                    label=kpi.get('label', ''),
                    value=kpi.get('value', ''),
                    delta=kpi.get('delta', None),
                    icon=kpi.get('icon', '📊')
                )
    
    @staticmethod
    def investment_recommendation(growth_rate: float, market_share: float = None, risk_level: str = None):
        """
        Yatırım Tavsiyesi Kartı
        """
        if growth_rate > 20:
            rec_type = "AGRESİF BÜYÜME"
            color = ExecutiveColors.SUCCESS
            icon = "🚀"
            recommendation = "Kapasite artırımı ve yeni ürün geliştirme yatırımlarını hızlandırın."
        elif growth_rate > 10:
            rec_type = "SEÇİCİ BÜYÜME"
            color = ExecutiveColors.INFO
            icon = "📈"
            recommendation = "Karlılığı koruyarak kontrollü büyüme stratejisi uygulayın."
        elif growth_rate > 5:
            rec_type = "KORUMA"
            color = ExecutiveColors.ACCENT_GOLD
            icon = "🛡️"
            recommendation = "Mevcut pazar payını koruyun, maliyet optimizasyonuna odaklanın."
        elif growth_rate > 0:
            rec_type = "BEKLE-GÖR"
            color = ExecutiveColors.WARNING
            icon = "👁️"
            recommendation = "Pazar gelişmelerini izleyin, acil yatırım kararlarından kaçının."
        else:
            rec_type = "RİSK YÖNETİMİ"
            color = ExecutiveColors.DANGER
            icon = "⚠️"
            recommendation = "Portföy optimizasyonu yapın, zayıf ürünlerden çıkış stratejisi planlayın."
        
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, {color}20, transparent);
                    border: 2px solid {color};
                    border-radius: 15px;
                    padding: 1.5rem;
                    margin: 1rem 0;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="font-size: 2.5rem; margin-right: 1rem;">{icon}</span>
                <div>
                    <span style="color: {color}; font-size: 1.5rem; font-weight: 800;">{rec_type}</span>
                    <span style="color: white; font-size: 1.2rem; margin-left: 1rem;">(%{growth_rate:.1f} Büyüme)</span>
                </div>
            </div>
            <p style="color: {ExecutiveColors.TEXT_SECONDARY}; font-size: 1.1rem; margin-left: 3.5rem;">
                {recommendation}
            </p>
        </div>
        """, unsafe_allow_html=True)

# ================================================
# 18. ANA UYGULAMA - ENTERPRISE DASHBOARD
# ================================================

def main():
    """Ana uygulama fonksiyonu - 4500+ satır"""
    
    # Session state başlatma
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.df = None
        st.session_state.df_original = None
        st.session_state.data_loaded = False
        st.session_state.selected_molecule = 'Tüm Moleküller'
        st.session_state.prodpack_analyzer = None
        st.session_state.forecast_engine = None
        st.session_state.risk_detector = None
        st.session_state.segmentation_engine = None
        st.session_state.forecast_results = {}
        st.session_state.cannibalization_df = None
        st.session_state.view_limit = 5000
    
    # Executive tema uygula
    ExecutiveUI.apply_theme()
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; 
                    background: linear-gradient(145deg, {ExecutiveColors.PRIMARY}, {ExecutiveColors.SECONDARY});
                    border-radius: 15px;
                    border-bottom: 4px solid {ExecutiveColors.ACCENT_GOLD};
                    margin-bottom: 2rem;">
            <h1 style="color: {ExecutiveColors.ACCENT_GOLD}; font-size: 2.2rem; margin: 0;">💊 PharmaIntel</h1>
            <p style="color: {ExecutiveColors.ACCENT_SILVER}; margin: 0.5rem 0 0 0;">Enterprise v8.0</p>
            <p style="color: {ExecutiveColors.TEXT_SECONDARY}; font-size: 0.8rem; margin-top: 0.5rem;">ProdPack Derinlik Analizi</p>
        </div>
        """, unsafe_allow_html=True)
        
        # VERİ YÜKLEME
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <span style="color: {ExecutiveColors.ACCENT_GOLD}; font-size: 1.2rem; font-weight: 700;">📁 VERİ YÖNETİMİ</span>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Excel veya CSV yükleyin",
            type=['xlsx', 'csv', 'xls'],
            help="Satış, molekül, paket, şirket bilgilerini içeren dosya"
        )
        
        if uploaded_file:
            if st.button("🚀 VERİYİ İŞLE VE ANALİZ ET", use_container_width=True):
                with st.spinner("🔮 Veri işleniyor... (Regex yıl ayıklama, tip dönüşümü)"):
                    df = AdvancedDataEngine.load_and_clean_data(uploaded_file)
                    
                    if not df.empty:
                        st.session_state.df_original = df.copy()
                        st.session_state.df = df.copy()
                        st.session_state.data_loaded = True
                        
                        # Analiz motorlarını başlat
                        st.session_state.prodpack_analyzer = ProdPackDeepDive(df)
                        st.session_state.forecast_engine = StrategicForecastEngine(df)
                        st.session_state.risk_detector = RiskAnomalyDetector(df)
                        st.session_state.segmentation_engine = SegmentationEngine(df)
                        
                        # Ön analizler
                        with st.spinner("🧠 AI modülleri çalıştırılıyor..."):
                            # Tahmin
                            st.session_state.forecast_results['Holt-Winters'] = st.session_state.forecast_engine.forecast_holt_winters(8)
                            st.session_state.forecast_results['Ensemble'] = st.session_state.forecast_engine.forecast_ensemble(8)
                            
                            # Risk
                            st.session_state.risk_detector.ensemble_anomaly_detection()
                            st.session_state.risk_detector.calculate_financial_risk()
                            st.session_state.df = st.session_state.risk_detector.df
                            
                            # Segmentasyon
                            st.session_state.segmentation_engine.ensemble_segmentation()
                            st.session_state.df = st.session_state.segmentation_engine.df
                            
                            # ProdPack hiyerarşisi
                            st.session_state.prodpack_analyzer.df = st.session_state.df
                        
                        st.success(f"✅ Veri başarıyla işlendi! ({len(df):,} satır, {len(df.columns)} sütun)")
                        st.balloons()
        
        st.markdown("---")
        
        # PRODPACK KONTROLLERİ
        if st.session_state.data_loaded and st.session_state.df is not None:
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <span style="color: {ExecutiveColors.ACCENT_GOLD}; font-size: 1.2rem; font-weight: 700;">🔬 PRODPACK ANALİZİ</span>
            </div>
            """, unsafe_allow_html=True)
            
            molecules = ['Tüm Moleküller'] + sorted(st.session_state.df['Molekul'].unique().tolist())
            selected = st.selectbox(
                "Molekül Seçin",
                molecules,
                index=0,
                key='molecule_selector'
            )
            st.session_state.selected_molecule = selected
            
            # Kanibalizasyon analizi
            if selected != 'Tüm Moleküller' and st.session_state.prodpack_analyzer:
                with st.spinner("🔄 Kanibalizasyon analizi yapılıyor..."):
                    st.session_state.cannibalization_df = st.session_state.prodpack_analyzer.analyze_cannibalization_matrix(selected)
        
        st.markdown("---")
        
        # PERFORMANS BİLGİLERİ
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <span style="color: {ExecutiveColors.ACCENT_GOLD}; font-size: 1.2rem; font-weight: 700;">⚙️ PERFORMANS</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"📊 Gösterim limiti: {st.session_state.view_limit:,} satır")
        st.caption(f"💾 Cache: st.cache_data aktif (TTL: 3600s)")
        
        if st.session_state.df is not None:
            memory_usage = st.session_state.df.memory_usage(deep=True).sum() / 1024**2
            st.caption(f"🧠 Bellek: {memory_usage:.1f} MB")
            
        st.markdown("---")
        
        # VERSİYON
        st.markdown(f"""
        <div style="text-align: center; color: {ExecutiveColors.TEXT_MUTED}; font-size: 0.8rem; padding: 1rem;">
            © 2024 PharmaIntelligence<br>
            v8.0.0 | Enterprise
        </div>
        """, unsafe_allow_html=True)
    
    # ========== ANA İÇERİK ==========
    if not st.session_state.data_loaded or st.session_state.df is None:
        # HOŞGELDİN EKRANI
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 3rem; 
                        background: rgba(30, 58, 95, 0.5);
                        backdrop-filter: blur(10px);
                        border-radius: 30px;
                        border: 2px dashed {ExecutiveColors.ACCENT_GOLD};
                        margin-top: 3rem;">
                <span style="font-size: 6rem;">💊</span>
                <h1 style="color: {ExecutiveColors.ACCENT_GOLD}; font-size: 2.8rem; margin: 1rem 0;">
                    PharmaIntelligence Pro
                </h1>
                <h3 style="color: {ExecutiveColors.TEXT_SECONDARY}; margin-bottom: 2rem;">
                    ProdPack Derinlik Analizi · AI Öngörü · Executive Dashboard
                </h3>
                <p style="color: {ExecutiveColors.TEXT_PRIMARY}; font-size: 1.2rem; margin-bottom: 2rem;">
                    Molekül ➔ Şirket ➔ Marka ➔ Paket hiyerarşisi<br>
                    Kanibalizasyon analizi · Pazar tahmini · Risk tespiti · Segmentasyon
                </p>
                <div style="background: {ExecutiveColors.PRIMARY}; padding: 1.5rem; border-radius: 15px;">
                    <p style="color: {ExecutiveColors.ACCENT_SILVER};">
                        🚀 Başlamak için sol panelden veri dosyanızı yükleyin
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # ========== VERİ YÜKLENDİ - ANA DASHBOARD ==========
    df = st.session_state.df
    prodpack = st.session_state.prodpack_analyzer
    selected_mol = st.session_state.selected_molecule
    
    # Executive Header
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
        <div>
            <h1 class="executive-title">💊 PharmaIntel Pro</h1>
            <p style="color: {ExecutiveColors.TEXT_SECONDARY}; font-size: 1.2rem;">
                ProdPack Derinlik Analizi · {selected_mol}
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: {ExecutiveColors.ACCENT_GOLD}; color: {ExecutiveColors.PRIMARY}; 
                        padding: 0.5rem 1.5rem; border-radius: 30px; font-weight: 700;">
                {datetime.now().strftime('%d.%m.%Y')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI SATIRI
    sales_cols = prodpack.sales_cols
    if sales_cols:
        latest_sales = sales_cols[-1]
        total_market = df[latest_sales].sum()
        
        if selected_mol != 'Tüm Moleküller':
            mol_df = df[df['Molekul'] == selected_mol]
            mol_sales = mol_df[latest_sales].sum()
            mol_share = (mol_sales / total_market * 100) if total_market > 0 else 0
        else:
            mol_sales = total_market
            mol_share = 100
        
        # Büyüme oranı
        if prodpack.latest_growth_col:
            if selected_mol != 'Tüm Moleküller':
                growth_rate = mol_df[prodpack.latest_growth_col].mean()
            else:
                growth_rate = df[prodpack.latest_growth_col].mean()
        else:
            growth_rate = 0
        
        kpis = [
            {'label': 'PAZAR BÜYÜKLÜĞÜ', 'value': f'{total_market:,.0f}₺', 'icon': '💰'},
            {'label': 'SEÇİLİ MOLEKÜL', 'value': f'{mol_sales:,.0f}₺', 'icon': '💊'},
            {'label': 'PAZAR PAYI', 'value': f'%{mol_share:.1f}', 'icon': '📊'},
            {'label': 'BÜYÜME ORANI', 'value': f'%{growth_rate:.1f}', 
             'delta': '📈' if growth_rate > 0 else '📉', 'icon': '📈'}
        ]
        
        ExecutiveUI.kpi_row(kpis)
    
    # SEKMELER
    tabs = st.tabs([
        "🔬 PRODPACK DERİNLİK",
        "📈 TAHMİN & ÖNGÖRÜ",
        "⚠️ RİSK & ANOMALİ",
        "🎯 SEGMENTASYON",
        "📊 EXECUTIVE DASHBOARD"
    ])
    
    # ========== TAB 1: PRODPACK DERİNLİK ==========
    with tabs[0]:
        st.markdown(f"""
        <h2 class="section-title">🔬 ProdPack Derinlik Analizi</h2>
        """, unsafe_allow_html=True)
        
        # Hiyerarşi oluştur
        if selected_mol != 'Tüm Moleküller':
            root = prodpack.build_hierarchy(selected_mol)
        else:
            root = prodpack.build_hierarchy(None)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📊 Hiyerarşik Görselleştirme")
            
            viz_type = st.radio(
                "Görselleştirme Tipi",
                ["Sunburst", "Sankey", "Treemap"],
                horizontal=True,
                key='viz_type'
            )
            
            if viz_type == "Sunburst":
                fig = prodpack.create_sunburst_diagram(root)
            elif viz_type == "Sankey":
                fig = prodpack.create_sankey_diagram(root)
            else:
                fig = prodpack.create_treemap(root)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔄 Kanibalizasyon Matrisi")
            
            if selected_mol != 'Tüm Moleküller' and st.session_state.cannibalization_df is not None:
                cannibal_df = st.session_state.cannibalization_df
                
                if not cannibal_df.empty:
                    # Özet istatistikler
                    high_cannibal = len(cannibal_df[cannibal_df['Kanibalizasyon_Tipi'].str.contains('Yüksek')])
                    medium_cannibal = len(cannibal_df[cannibal_df['Kanibalizasyon_Tipi'].str.contains('Orta')])
                    
                    st.metric("🔴 Yüksek Kanibalizasyon", high_cannibal)
                    st.metric("🟡 Orta Kanibalizasyon", medium_cannibal)
                    
                    # Tablo
                    st.dataframe(
                        cannibal_df[['Paket', 'Şirket', 'Pazar_Payı', 'Büyüme_Oranı', 'Kanibalizasyon_Tipi']].head(10),
                        use_container_width=True,
                        height=300
                    )
                    
                    # Insight Box
                    if high_cannibal > 0:
                        top_cannibal = cannibal_df.iloc[0]
                        ExecutiveUI.insight_box(
                            "🔄 Kanibalizasyon Uyarısı",
                            f"**{selected_mol}** molekülünde **{top_cannibal['Paket']}** paketi yüksek kanibalizasyon riski taşıyor. "
                            f"Pazar payı %{top_cannibal['Pazar_Payı']:.1f}, büyüme oranı %{top_cannibal['Büyüme_Oranı']:.1f}.",
                            f"Risk Skoru: {top_cannibal['Kanibalizasyon_Skoru']:.0f}",
                            icon="⚠️"
                        )
                else:
                    st.info("Kanibalizasyon analizi için yeterli veri yok.")
            else:
                st.info("👈 Kanibalizasyon analizi için sol panelden bir molekül seçin.")
        
        # Paket Performans Tablosu
        st.markdown("---")
        st.subheader("📋 Paket Performans Detayı")
        
        pack_perf_df = prodpack.get_pack_performance_table(selected_mol, n_top=st.session_state.view_limit)
        
        if not pack_perf_df.empty:
            # Formatlama
            format_dict = {}
            if prodpack.latest_sales_col in pack_perf_df.columns:
                format_dict[prodpack.latest_sales_col] = '{:,.0f}'
            if prodpack.prev_sales_col in pack_perf_df.columns:
                format_dict[prodpack.prev_sales_col] = '{:,.0f}'
            if prodpack.latest_growth_col in pack_perf_df.columns:
                format_dict[prodpack.latest_growth_col] = '{:.1f}%'
            if 'Pazar_Payi_2024' in pack_perf_df.columns:
                format_dict['Pazar_Payi_2024'] = '{:.2f}%'
            if 'CAGR_3Y' in pack_perf_df.columns:
                format_dict['CAGR_3Y'] = '{:.1f}%'
            
            st.dataframe(
                pack_perf_df.style.format(format_dict),
                use_container_width=True,
                height=500
            )
    
    # ========== TAB 2: TAHMİN & ÖNGÖRÜ ==========
    with tabs[1]:
        st.markdown(f"""
        <h2 class="section-title">📈 Stratejik Tahmin & Yatırım Öngörüsü</h2>
        """, unsafe_allow_html=True)
        
        if st.session_state.forecast_results:
            forecast_ens = st.session_state.forecast_results.get('Ensemble')
            forecast_hw = st.session_state.forecast_results.get('Holt-Winters')
            
            col_f1, col_f2 = st.columns([2, 1])
            
            with col_f1:
                # Tahmin grafiği
                fig_forecast = go.Figure()
                
                # Tarihsel veri
                fig_forecast.add_trace(go.Scatter(
                    x=prodpack.years,
                    y=prodpack.sales_values,
                    mode='lines+markers',
                    name='Tarihsel Satış',
                    line=dict(color=ExecutiveColors.ACCENT_GOLD, width=4),
                    marker=dict(size=10)
                ))
                
                # Ensemble tahmin
                if forecast_ens:
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_ens.periods,
                        y=forecast_ens.predictions,
                        mode='lines+markers',
                        name='Ensemble Tahmin',
                        line=dict(color=ExecutiveColors.SUCCESS, width=3, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    # Güven aralığı
                    fig_forecast.add_trace(go.Scatter(
                        x=forecast_ens.periods + forecast_ens.periods[::-1],
                        y=forecast_ens.upper_bound_95 + forecast_ens.lower_bound_95[::-1],
                        fill='toself',
                        fillcolor='rgba(46, 204, 113, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='%95 Güven Aralığı'
                    ))
                
                fig_forecast.update_layout(
                    title='Pazar Tahmini 2025-2026',
                    xaxis_title='Yıl',
                    yaxis_title='Satış (₺)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=ExecutiveColors.TEXT_PRIMARY),
                    height=500,
                    template=ChartTemplates.executive_template()
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            with col_f2:
                if forecast_ens:
                    st.subheader("🎯 Tahmin Metrikleri")
                    
                    st.metric("2025 Tahmini", f"{forecast_ens.predictions[0]:,.0f}₺")
                    st.metric("2026 Tahmini", f"{forecast_ens.predictions[3]:,.0f}₺")
                    st.metric("Büyüme Oranı", f"%{forecast_ens.growth_rate:.1f}")
                    st.metric("CAGR (2025-26)", f"%{forecast_ens.cagr_forecast:.1f}")
                    st.metric("Model Doğruluğu (R²)", f"{forecast_ens.r2:.3f}")
                    st.metric("Tahmin Hatası (MAPE)", f"%{forecast_ens.mape:.1f}")
            
            # Yatırım Tavsiyesi
            if forecast_ens:
                st.markdown("---")
                st.subheader("💎 Yatırım Tavsiyesi")
                
                ExecutiveUI.investment_recommendation(
                    forecast_ens.growth_rate,
                    market_share=mol_share if selected_mol != 'Tüm Moleküller' else None
                )
                
                # Insight Box
                if forecast_ens.growth_rate > 15:
                    ExecutiveUI.insight_box(
                        "🚀 Yüksek Büyüme Fırsatı",
                        f"Pazarın önümüzdeki 2 yılda %{forecast_ens.growth_rate:.1f} büyümesi bekleniyor. "
                        f"Bu dönemde kapasite artırımı ve yeni ürün geliştirme yatırımlarına öncelik verin.",
                        f"Tahmini Pazar: {forecast_ens.predictions[-1]:,.0f}₺",
                        icon="💎"
                    )
                elif forecast_ens.growth_rate > 5:
                    ExecutiveUI.insight_box(
                        "📊 İstikrarlı Büyüme",
                        f"Pazar %{forecast_ens.growth_rate:.1f} büyüme trendinde. "
                        f"Mevcut pazar payını korumaya ve karlılığı optimize etmeye odaklanın.",
                        icon="📈"
                    )
                else:
                    ExecutiveUI.insight_box(
                        "⚠️ Durgun Pazar Uyarısı",
                        f"Pazar büyümesi %{forecast_ens.growth_rate:.1f} ile yavaşlıyor. "
                        f"Maliyet optimizasyonu ve portföy çeşitlendirmesi önerilir.",
                        icon="⚠️"
                    )
        else:
            st.warning("Tahmin analizi için en az 4 yıllık veri gereklidir.")
    
    # ========== TAB 3: RİSK & ANOMALİ ==========
    with tabs[2]:
        st.markdown(f"""
        <h2 class="section-title">⚠️ Risk ve Anomali İzleme</h2>
        """, unsafe_allow_html=True)
        
        if st.session_state.risk_detector:
            risk_summary = st.session_state.risk_detector.get_risk_summary()
            
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                critical = risk_summary.get('critical_count', 0)
                st.metric("🔴 Kritik Risk", critical, delta_color="inverse")
            
            with col_r2:
                high = risk_summary.get('high_count', 0)
                st.metric("🟠 Yüksek Risk", high, delta_color="inverse")
            
            with col_r3:
                medium = risk_summary.get('medium_count', 0)
                st.metric("🟡 Orta Risk", medium)
            
            with col_r4:
                low = risk_summary.get('low_count', 0)
                st.metric("🟢 Düşük Risk", low)
            
            # Risk Dashboard
            fig_risk = st.session_state.risk_detector.plot_risk_dashboard()
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Kritik riskli ürünler
            st.subheader("🚨 Kritik Riskli Paketler")
            
            if 'Risk_Seviyesi' in df.columns:
                critical_df = df[df['Risk_Seviyesi'] == '🔴 Kritik Risk']
                
                if not critical_df.empty:
                    display_cols = ['Paket', 'Sirket', 'Molekul', 'Anomali_Skoru_Ensemble']
                    available_cols = [col for col in display_cols if col in critical_df.columns]
                    
                    if prodpack.latest_sales_col:
                        available_cols.append(prodpack.latest_sales_col)
                    
                    st.dataframe(
                        critical_df[available_cols].head(20),
                        use_container_width=True
                    )
                    
                    ExecutiveUI.insight_box(
                        "🚨 Acil Müdahale Gereken Ürünler",
                        f"{len(critical_df)} paket kritik risk kategorisinde. "
                        f"Toplam satışların %{(critical_df[prodpack.latest_sales_col].sum() / df[prodpack.latest_sales_col].sum() * 100):.1f}''sini oluşturuyor.",
                        f"{len(critical_df)} Kritik Risk",
                        icon="🔥"
                    )
                else:
                    st.success("✅ Kritik risk seviyesinde paket bulunmuyor.")
    
    # ========== TAB 4: SEGMENTASYON ==========
    with tabs[3]:
        st.markdown(f"""
        <h2 class="section-title">🎯 Gelişmiş Segmentasyon Analizi</h2>
        """, unsafe_allow_html=True)
        
        if st.session_state.segmentation_engine:
            col_s1, col_s2 = st.columns([2, 1])
            
            with col_s1:
                # Segmentasyon dashboard
                fig_seg = st.session_state.segmentation_engine.plot_segmentation_dashboard()
                st.plotly_chart(fig_seg, use_container_width=True)
            
            with col_s2:
                st.subheader("📊 Segment Özeti")
                
                seg_summary = st.session_state.segmentation_engine.get_segmentation_summary()
                
                if not seg_summary.empty:
                    st.dataframe(
                        seg_summary[['Segment', 'Ürün Sayısı', 'Satış Oranı', 'Ort. Büyüme']].head(8),
                        use_container_width=True
                    )
                    
                    # En büyük segment
                    top_segment = seg_summary.iloc[0]
                    
                    ExecutiveUI.insight_box(
                        "🎯 Stratejik Segment Önerisi",
                        f"**{top_segment['Segment']}** segmenti pazarın %{top_segment['Satış Oranı']}''sini oluşturuyor. "
                        f"Bu segmentte {top_segment['Ürün Sayısı']} ürün bulunuyor.",
                        f"Ort. Büyüme: {top_segment['Ort. Büyüme']:.1f}%",
                        icon="💎"
                    )
            
            # Segment stratejileri
            st.markdown("---")
            st.subheader("🎯 Segment Bazlı Stratejiler")
            
            strategy_cols = st.columns(4)
            
            strategies = [
                {"title": "👑 Pazar Liderleri", "desc": "Yatırımı artır, yenilikçi ürünler geliştir", "color": ExecutiveColors.ACCENT_GOLD},
                {"title": "🚀 Yükselen Yıldızlar", "desc": "Büyümeyi destekle, pazarlamaya yatırım yap", "color": ExecutiveColors.SUCCESS},
                {"title": "🐄 Nakit İnekleri", "desc": "Karlılığı koru, nakit akışını optimize et", "color": ExecutiveColors.INFO},
                {"title": "⚠️ Gerileyen Ürünler", "desc": "Portföyden çıkar veya yeniden konumlandır", "color": ExecutiveColors.WARNING}
            ]
            
            for i, strategy in enumerate(strategies):
                with strategy_cols[i]:
                    st.markdown(f"""
                    <div style="background: {strategy['color']}20; 
                                border: 1px solid {strategy['color']};
                                border-radius: 10px;
                                padding: 1rem;
                                height: 150px;">
                        <h4 style="color: {strategy['color']};">{strategy['title']}</h4>
                        <p style="color: {ExecutiveColors.TEXT_SECONDARY};">{strategy['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ========== TAB 5: EXECUTIVE DASHBOARD ==========
    with tabs[4]:
        st.markdown(f"""
        <h2 class="section-title">📊 Executive Dashboard</h2>
        """, unsafe_allow_html=True)
        
        # Özet kartlar
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        
        with col_d1:
            st.markdown(f"""
            <div class="executive-card" style="text-align: center;">
                <span style="font-size: 2.5rem;">💊</span>
                <h3 style="color: {ExecutiveColors.ACCENT_GOLD};">{df['Molekul'].nunique()}</h3>
                <p style="color: {ExecutiveColors.TEXT_SECONDARY};">Aktif Molekül</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d2:
            st.markdown(f"""
            <div class="executive-card" style="text-align: center;">
                <span style="font-size: 2.5rem;">🏢</span>
                <h3 style="color: {ExecutiveColors.ACCENT_GOLD};">{df['Sirket'].nunique() if 'Sirket' in df.columns else 0}</h3>
                <p style="color: {ExecutiveColors.TEXT_SECONDARY};">Aktif Şirket</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d3:
            st.markdown(f"""
            <div class="executive-card" style="text-align: center;">
                <span style="font-size: 2.5rem;">📦</span>
                <h3 style="color: {ExecutiveColors.ACCENT_GOLD};">{df['Paket'].nunique()}</h3>
                <p style="color: {ExecutiveColors.TEXT_SECONDARY};">Toplam Paket</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d4:
            st.markdown(f"""
            <div class="executive-card" style="text-align: center;">
                <span style="font-size: 2.5rem;">📈</span>
                <h3 style="color: {ExecutiveColors.ACCENT_GOLD};">%{df['Buyume_Orani'].mean():.1f}</h3>
                <p style="color: {ExecutiveColors.TEXT_SECONDARY};">Ort. Büyüme</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Hiyerarşi özeti
        st.markdown("---")
        st.subheader("🏆 ProdPack Hiyerarşi Özeti")
        
        if selected_mol != 'Tüm Moleküller':
            root = prodpack.build_hierarchy(selected_mol)
            
            # Toplam şirket, marka, paket sayısı
            company_count = len([c for c in root.children])
            brand_count = sum([len(c.children) for c in root.children])
            pack_count = sum([sum([len(b.children) for b in c.children]) for c in root.children])
            
            col_h1, col_h2, col_h3 = st.columns(3)
            
            with col_h1:
                st.metric("🏢 Şirket Sayısı", company_count)
            with col_h2:
                st.metric("🏷️ Marka Sayısı", brand_count)
            with col_h3:
                st.metric("📦 Paket Sayısı", pack_count)
        
        # En iyi performans gösterenler
        st.markdown("---")
        col_top1, col_top2 = st.columns(2)
        
        with col_top1:
            st.subheader("🏆 En Yüksek Satış")
            if prodpack.latest_sales_col:
                top_sales = df.nlargest(10, prodpack.latest_sales_col)[
                    ['Paket', 'Sirket', prodpack.latest_sales_col]
                ]
                st.dataframe(top_sales, use_container_width=True)
        
        with col_top2:
            st.subheader("🚀 En Hızlı Büyüme")
            if prodpack.latest_growth_col:
                top_growth = df.nlargest(10, prodpack.latest_growth_col)[
                    ['Paket', 'Sirket', prodpack.latest_growth_col]
                ]
                st.dataframe(top_growth, use_container_width=True)
        
        # Executive Insight
        ExecutiveUI.insight_box(
            "📊 Executive Özet",
            f"**{selected_mol}** pazarında **{df['Paket'].nunique()}** farklı paket, "
            f"**{df['Sirket'].nunique()}** şirket tarafından rekabet ediyor. "
            f"Pazarın toplam büyüklüğü **{total_market:,.0f}₺**, yıllık büyüme **%{growth_rate:.1f}**. "
            f"Önümüzdeki 2 yılda **%{forecast_ens.growth_rate:.1f}** büyüme bekleniyor." if forecast_ens else "",
            icon="💎"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {ExecutiveColors.TEXT_MUTED}; padding: 2rem;">
        <span style="font-size: 1.2rem; color: {ExecutiveColors.ACCENT_GOLD};">PharmaIntelligence Pro v8.0</span>
        <br>
        <span style="font-size: 0.9rem;">Enterprise Karar Destek Platformu · ProdPack Derinlik Analizi · AI Öngörü</span>
        <br>
        <span style="font-size: 0.8rem;">© 2024 PharmaIntelligence Inc. Tüm hakları saklıdır.</span>
    </div>
    """, unsafe_allow_html=True)

# ================================================
# 19. UYGULAMA GİRİŞ NOKTASI
# ================================================

if __name__ == "__main__":
    try:
        # Bellek optimizasyonu
        gc.enable()
        
        # Uygulamayı başlat
        main()
        
    except Exception as e:
        st.error("""
        ## ⚠️ Kritik Uygulama Hatası
        
        PharmaIntelligence Pro v8.0'da beklenmeyen bir hata oluştu.
        Lütfen sayfayı yenileyin veya destek ekibiyle iletişime geçin.
        """)
        
        st.error(f"**Hata Detayı:** {str(e)}")
        
        with st.expander("🔍 Hata Ayıklama Detayları"):
            st.code(traceback.format_exc())
        
        # Kurtarma seçenekleri
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Uygulamayı Yeniden Başlat", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📋 Hata Raporu Oluştur", use_container_width=True):
                error_report = {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'session_state': {k: str(v)[:100] for k, v in st.session_state.items()}
                }
                
                st.download_button(
                    label="📥 Hata Raporunu İndir",
                    data=json.dumps(error_report, indent=2, default=str),
                    file_name=f"pharma_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

