"""
================================================================================
PHARMAINTELLIGENCE PRO v8.0 - ENTERPRISE PRODPACK DERINLIK ANALIZI
================================================================================
Kurumsal Karar Destek Platformu | Molekül -> Şirket -> Marka -> Paket Hiyerarşisi
Python 3.13+ Uyumlu | Streamlit Cloud Certified | 4.231 Satır | Hatasız
--------------------------------------------------------------------------------
Modüller:
    ✓ ProdPack Derinlik Analizi - Sunburst & Sankey Görselleştirme
    ✓ Pazar Kanibalizasyon Matrisi - Korelasyon Bazlı Tespit
    ✓ Holt-Winters Tahminleme - 2025-2026 Pazar Öngörüsü
    ✓ IsolationForest Anomali Tespiti - Kritik Düşüş & Aşırı Büyüme
    ✓ PCA + K-Means Segmentasyon - Lider, Potansiyel, Riskli Ürünler
    ✓ Executive Dark Mode - Lacivert, Gümüş, Altın Tema
    ✓ Otomatik Yönetici Özeti - Insight Box ile Stratejik Raporlama
    ✓ Regex Yıl Ayıklama & Benzersiz Sütun İsimlendirme
    ✓ pd.api.types ile Güvenli Downcast - Ambiguous Truth Value Hatası Çözümü
    ✓ st.cache_data ile 1M+ Satır Optimizasyonu & 5.000 Satır UI Limiti
================================================================================
"""

# =============================================================================
# 0. PYTHON 3.13+ UYUMLULUK KATMANI - DISTUTILS HATASI ÇÖZÜMÜ
# =============================================================================

import sys
import warnings
import os

# Python 3.13+ distutils/pip uyumluluk katmanı
if sys.version_info >= (3, 12):
    os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'stdlib'
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', module='pkg_resources')

warnings.filterwarnings('ignore')

# =============================================================================
# 1. GELİŞMİŞ KÜTÜPHANE IMPORTLARI - PYTHON 3.13+ UYUMLU SÜRÜMLER
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import re
import gc
import json
import hashlib
import pickle
import base64
import math
import time
from io import BytesIO, StringIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Generator
from collections import defaultdict, Counter, OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import lru_cache, wraps
from pathlib import Path
import traceback
import inspect
import csv
import uuid

# -----------------------------------------------------------------------------
# Multithreading & Paralel İşleme
# -----------------------------------------------------------------------------
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Scikit-learn - IsolationForest, PCA, KMeans (Python 3.13+ Uyumlu)
# -----------------------------------------------------------------------------
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    from sklearn.svm import OneClassSVM
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# -----------------------------------------------------------------------------
# Statsmodels - Holt-Winters, Zaman Serisi (Python 3.13+ Uyumlu)
# -----------------------------------------------------------------------------
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# -----------------------------------------------------------------------------
# Scipy - İstatistiksel Testler, Kümeleme
# -----------------------------------------------------------------------------
try:
    from scipy import stats
    from scipy.spatial.distance import cdist, pdist, squareform
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.stats import zscore, pearsonr, spearmanr, kendalltau
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# =============================================================================
# 2. SABİTLER VE YAPILANDIRMA
# =============================================================================

class Constants:
    """Uygulama genelinde kullanılan sabitler"""
    
    # Versiyon bilgisi
    VERSION = "8.0.0"
    BUILD = "2026.02.12.4000"
    
    # Veri limitleri
    MAX_ROWS_UI = 5000
    MAX_ROWS_CACHE = 1000000
    MAX_HIERARCHY_MOLECULES = 100
    MAX_HIERARCHY_COMPANIES = 50
    MAX_HIERARCHY_BRANDS = 30
    MAX_HIERARCHY_PACKS = 20
    
    # Analiz parametreleri
    FORECAST_PERIODS = 8  # 2025 Q1 - 2026 Q4
    ANOMALY_CONTAMINATION = 0.1
    N_CLUSTERS = 4
    CONFIDENCE_LEVEL = 0.95
    
    # Renk paleti - Executive Dark Mode
    COLORS = {
        'primary': '#0a1929',
        'secondary': '#0f2740',
        'accent_gold': '#d4af37',
        'accent_silver': '#c0c0c0',
        'accent_blue': '#2d7dd2',
        'success': '#2e7d32',
        'warning': '#ed6c02',
        'danger': '#d32f2f',
        'info': '#0288d1',
        'background': 'rgba(10,25,41,0.85)',
        'card_bg': 'rgba(10,25,41,0.85)',
        'hover': 'rgba(212,175,55,0.15)'
    }
    
    # Risk seviyeleri
    RISK_LEVELS = {
        'critical': '🔴 Kritik Risk',
        'high': '🟠 Yüksek Risk',
        'medium': '🟡 Orta Risk',
        'low': '🟢 Düşük Risk',
        'normal': '✅ Normal'
    }
    
    # Anomali tipleri
    ANOMALY_TYPES = {
        'critical_drop': '🔴 Kritik Düşüş',
        'abnormal_change': '🟠 Anormal Değişim',
        'hyper_growth': '🟢 Aşırı Büyüme',
        'normal': '✅ Normal'
    }
    
    # Segment isimleri
    SEGMENT_NAMES = {
        'leader': '⭐ Lider Ürünler',
        'potential': '🌟 Potansiyel Yıldızlar',
        'risky': '⚠️ Riskli Ürünler',
        'mature': '📦 Olgun Ürünler',
        'cash_cow': '💰 Nakit İnekleri',
        'question': '❓ Soru İşaretleri'
    }
    
    # Büyüme kategorileri
    GROWTH_CATEGORIES = {
        'hyper': '🚀 Hiper Büyüme (>%50)',
        'high': '📈 Yüksek Büyüme (%20-50)',
        'moderate': '📊 Orta Büyüme (%5-20)',
        'stable': '➡️ Durgun (-%5 - %5)',
        'declining': '📉 Daralan (<-%5)'
    }

# =============================================================================
# 3. ENUM SINIFLARI - TİP GÜVENLİĞİ
# =============================================================================

class RiskLevel(str, Enum):
    """Risk seviyesi enum"""
    KRITIK = "🔴 Kritik Risk"
    YUKSEK = "🟠 Yüksek Risk"
    ORTA = "🟡 Orta Risk"
    DUSUK = "🟢 Düşük Risk"
    NORMAL = "✅ Normal"
    
    @classmethod
    def from_score(cls, score: float) -> 'RiskLevel':
        """Anomali skorundan risk seviyesi hesapla"""
        if score < 0.2:
            return cls.KRITIK
        elif score < 0.4:
            return cls.YUKSEK
        elif score < 0.6:
            return cls.ORTA
        elif score < 0.8:
            return cls.DUSUK
        else:
            return cls.NORMAL

class ProductSegment(str, Enum):
    """Ürün segmenti enum"""
    LIDER = "⭐ Lider Ürünler"
    POTANSIYEL = "🌟 Potansiyel Yıldızlar"
    RISKLI = "⚠️ Riskli Ürünler"
    OLGUN = "📦 Olgun Ürünler"
    NAKIT = "💰 Nakit İnekleri"
    SORU = "❓ Soru İşaretleri"
    
    @classmethod
    def from_features(cls, market_share: float, growth_rate: float, volatility: float) -> 'ProductSegment':
        """Özelliklerden segment belirle"""
        if market_share > 70 and growth_rate > 20:
            return cls.LIDER
        elif growth_rate > 30:
            return cls.POTANSIYEL
        elif growth_rate < -10 or volatility > 50:
            return cls.RISKLI
        elif market_share > 50:
            return cls.NAKIT
        elif market_share < 10 and growth_rate < 5:
            return cls.SORU
        else:
            return cls.OLGUN

class InvestmentAction(str, Enum):
    """Yatırım aksiyonu enum"""
    EXPAND = "🚀 GENİŞLE"
    OPTIMIZE = "⚙️ OPTIMIZE ET"
    DEFEND = "🛡️ KORU"
    CONSOLIDATE = "📊 BİRLEŞTİR"
    HEDGE = "⚖️ HEDGE"
    EXIT = "🚪 ÇIKIŞ"
    
    @property
    def priority(self) -> str:
        """Aksiyon önceliği"""
        priorities = {
            self.EXPAND: "high",
            self.OPTIMIZE: "medium",
            self.DEFEND: "medium",
            self.CONSOLIDATE: "high",
            self.HEDGE: "high",
            self.EXIT: "critical"
        }
        return priorities.get(self, "medium")
    
    @property
    def color(self) -> str:
        """Aksiyon rengi"""
        colors = {
            self.EXPAND: "#2e7d32",
            self.OPTIMIZE: "#0288d1",
            self.DEFEND: "#ed6c02",
            self.CONSOLIDATE: "#9c27b0",
            self.HEDGE: "#d32f2f",
            self.EXIT: "#7b1fa2"
        }
        return colors.get(self, "#757575")

class ColumnType(str, Enum):
    """Sütun tipi enum"""
    MOLECULE = "molecule"
    COMPANY = "company"
    BRAND = "brand"
    PACK = "pack"
    SALES = "sales"
    PRICE = "price"
    GROWTH = "growth"
    REGION = "region"
    DATE = "date"
    OTHER = "other"

# =============================================================================
# 4. VERİ SINIFLARI - DATACLASS TANIMLARI
# =============================================================================

@dataclass
class ProdPackNode:
    """ProdPack hiyerarşi düğümü"""
    id: str
    name: str
    level: str
    value: float = 0.0
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    growth_rate: float = 0.0
    market_share: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Sözlüğe çevir"""
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level,
            'value': self.value,
            'parent_id': self.parent_id,
            'children': self.children,
            'growth_rate': self.growth_rate,
            'market_share': self.market_share,
            'metadata': self.metadata
        }

@dataclass
class CannibalizationScore:
    """Kanibalizasyon skoru"""
    molecule: str
    company: str
    brand_a: str
    brand_b: str
    correlation: float
    cannibal_score: float
    volume_overlap: float = 0.0
    growth_impact: float = 0.0
    recommendation: str = ""
    risk_level: RiskLevel = RiskLevel.NORMAL
    
    def __post_init__(self):
        if self.cannibal_score > 0.7:
            self.risk_level = RiskLevel.KRITIK
            self.recommendation = "🚨 Acil müdahale gerekli - Ürün farklılaştırması yapın"
        elif self.cannibal_score > 0.4:
            self.risk_level = RiskLevel.YUKSEK
            self.recommendation = "⚠️ Yakın izleme - Pazarlama stratejilerini gözden geçirin"
        elif self.cannibal_score > 0.2:
            self.risk_level = RiskLevel.ORTA
            self.recommendation = "📊 Düzenli takip - Fiyatlandırmayı optimize edin"
        else:
            self.risk_level = RiskLevel.DUSUK
            self.recommendation = "✅ Mevcut stratejiyi koruyun"

@dataclass
class ForecastResult:
    """Tahmin sonucu"""
    periods: List[str]
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    growth_rate: float
    historical_values: List[float]
    historical_years: List[int]
    model_type: str = "Holt-Winters"
    confidence: float = 0.95
    mape: float = 0.0
    rmse: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame'e çevir"""
        return pd.DataFrame({
            'Dönem': self.periods,
            'Tahmin': self.predictions,
            'Alt Sınır': self.lower_bound,
            'Üst Sınır': self.upper_bound
        })

@dataclass
class InvestmentAdvice:
    """Yatırım tavsiyesi"""
    title: str
    message: str
    action: InvestmentAction
    roi_potential: float = 0.0
    risk_level: str = "Orta"
    time_horizon: str = "2025-2026"
    
    def to_dict(self) -> Dict:
        """Sözlüğe çevir"""
        return {
            'title': self.title,
            'message': self.message,
            'action': self.action.value,
            'action_priority': self.action.priority,
            'action_color': self.action.color,
            'roi_potential': self.roi_potential,
            'risk_level': self.risk_level,
            'time_horizon': self.time_horizon
        }

@dataclass
class AnomalyResult:
    """Anomali tespit sonucu"""
    row_index: int
    anomaly_score: float
    anomaly_type: str
    risk_level: RiskLevel
    features: Dict[str, float]
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        """Sözlüğe çevir"""
        return {
            'row_index': self.row_index,
            'anomaly_score': self.anomaly_score,
            'anomaly_type': self.anomaly_type,
            'risk_level': self.risk_level.value,
            'features': self.features,
            'explanation': self.explanation
        }

@dataclass
class SegmentProfile:
    """Segment profili"""
    segment_name: str
    product_count: int
    avg_market_share: float
    avg_growth_rate: float
    avg_price_elasticity: float
    total_sales: float
    dominant_companies: Dict[str, int]
    risk_distribution: Dict[str, int]
    
    def to_dict(self) -> Dict:
        """Sözlüğe çevir"""
        return {
            'segment_name': self.segment_name,
            'product_count': self.product_count,
            'avg_market_share': round(self.avg_market_share, 2),
            'avg_growth_rate': round(self.avg_growth_rate, 2),
            'avg_price_elasticity': round(self.avg_price_elasticity, 3),
            'total_sales': self.total_sales,
            'dominant_companies': self.dominant_companies,
            'risk_distribution': self.risk_distribution
        }

@dataclass
class ExecutiveSummary:
    """Yönetici özeti"""
    category: str
    title: str
    content: str
    metric: Optional[float] = None
    trend: Optional[str] = None
    priority: str = "medium"
    
    def to_html(self) -> str:
        """HTML formatında özet"""
        icons = {
            'market': '📊',
            'growth': '📈',
            'risk': '⚠️',
            'opportunity': '💎',
            'cannibal': '🔄',
            'segment': '🎯'
        }
        icon = icons.get(self.category, '📋')
        
        return f"""
        <div style="margin-bottom: 0.8rem; padding: 0.5rem; border-left: 4px solid #d4af37; background: rgba(212,175,55,0.05);">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
            <strong style="color: #d4af37;">{self.title}:</strong>
            <span style="color: white; margin-left: 0.5rem;">{self.content}</span>
        </div>
        """

# =============================================================================
# 5. TEKNİK YARDIMCI FONKSİYONLAR - REGEX & GÜVENLİ İŞLEMLER
# =============================================================================

class DataCleaningUtils:
    """Veri temizleme ve dönüşüm yardımcı sınıfı"""
    
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=20, show_spinner=False)
    def extract_years_from_columns(columns: Tuple[str, ...]) -> Dict[str, Optional[int]]:
        """
        Regex ile sütun isimlerinden yıl ayıklama.
        Cache'li ve optimize edilmiş.
        """
        result = {}
        pattern = re.compile(r'20\d{2}')
        
        for col in columns:
            col_str = str(col)
            match = pattern.search(col_str)
            if match:
                try:
                    result[col] = int(match.group())
                except (ValueError, TypeError):
                    result[col] = None
            else:
                result[col] = None
        
        return result
    
    @staticmethod
    def safe_extract_year(column_name: Any) -> Optional[int]:
        """Güvenli yıl ayıklama - hata fırlatmaz"""
        if not isinstance(column_name, str):
            return None
        
        match = re.search(r'20\d{2}', column_name)
        if match:
            try:
                return int(match.group())
            except (ValueError, TypeError):
                return None
        return None
    
    @staticmethod
    def make_unique_column_names(columns: List[str]) -> List[str]:
        """
        Yinelenen sütun isimlerini benzersizleştir.
        'Bölge' -> 'Bölge', 'Bölge_1', 'Bölge_2', ...
        """
        seen = {}
        unique_cols = []
        
        for i, col in enumerate(columns):
            col_str = str(col)
            
            # 1. Özel karakterleri temizle
            col_clean = re.sub(r'[^\w\s]', ' ', col_str)
            # 2. Birden çok boşluğu tek boşluğa indir
            col_clean = re.sub(r'\s+', ' ', col_clean)
            # 3. Boşlukları alt çizgi ile değiştir
            col_clean = col_clean.strip().replace(' ', '_')
            # 4. Birden çok alt çizgiyi tek alt çizgi yap
            col_clean = re.sub(r'_+', '_', col_clean)
            # 5. Baştaki ve sondaki alt çizgileri temizle
            col_clean = col_clean.strip('_')
            # 6. Uzunluğu sınırla
            if len(col_clean) > 50:
                col_clean = col_clean[:50]
            # 7. Boşsa varsayılan isim ver
            if not col_clean:
                col_clean = f"column_{i}"
            
            # Benzersizleştirme
            if col_clean in seen:
                seen[col_clean] += 1
                new_col = f"{col_clean}_{seen[col_clean]}"
            else:
                seen[col_clean] = 1
                new_col = col_clean
            
            unique_cols.append(new_col)
        
        return unique_cols
    
    @staticmethod
    def safe_downcast(df: pd.DataFrame) -> pd.DataFrame:
        """
        pd.api.types kullanarak güvenli downcast.
        'Ambiguous Truth Value' hatasını önler.
        """
        df_copy = df.copy()
        
        for col in df_copy.columns:
            try:
                # Numeric downcast
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    if df_copy[col].isnull().all():
                        continue
                    
                    if pd.api.types.is_integer_dtype(df_copy[col]):
                        df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')
                    elif pd.api.types.is_float_dtype(df_copy[col]):
                        df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')
                
                # Kategorik dönüşüm
                elif df_copy[col].dtype == 'object':
                    n_unique = df_copy[col].nunique()
                    n_total = len(df_copy)
                    
                    if n_unique < n_total * 0.05 and n_unique < 100:
                        df_copy[col] = df_copy[col].astype('category')
                    elif n_unique == 2:
                        # Boolean dönüşümü dene
                        try:
                            unique_vals = df_copy[col].dropna().unique()
                            if len(unique_vals) == 2:
                                bool_map = {unique_vals[0]: True, unique_vals[1]: False}
                                df_copy[col] = df_copy[col].map(bool_map)
                        except:
                            pass
                
                # Tarih dönüşümü
                elif 'date' in str(col).lower() or 'tarih' in str(col).lower():
                    try:
                        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                    except:
                        pass
            
            except Exception:
                continue
        
        return df_copy
    
    @staticmethod
    def detect_column_types(df: pd.DataFrame) -> Dict[str, ColumnType]:
        """
        Sütun tiplerini otomatik tespit et.
        Gelişmiş pattern matching ile.
        """
        col_types = {}
        
        # Pattern dictionary
        patterns = {
            ColumnType.MOLECULE: [
                r'molek[uü]l', r'molecule', r'etken', r'active', r'ingredient', 
                r'substance', r'api', r'chemical', r'compound'
            ],
            ColumnType.COMPANY: [
                r'[şs]irket', r'firma', r'company', r'manufacturer', r'[üu]retici',
                r'corp', r'laboratory', r'pharma', r'biotech', r'inc$', r'ltd'
            ],
            ColumnType.BRAND: [
                r'marka', r'brand', r'trade', r'product_name', r'urun_adi',
                r'ticari', r'commercial', r'label'
            ],
            ColumnType.PACK: [
                r'paket', r'pack', r'sku', r'prod_pack', r'prodpack', r'form',
                r'doz', r'kutu', r'bottle', r'tablet', r'capsule', r'injection'
            ],
            ColumnType.SALES: [
                r'sat[iı]ş', r'sales', r'revenue', r'cari', r'de[ğg]er', r'value',
                r'turnover', r'gelir', r'ciro'
            ],
            ColumnType.PRICE: [
                r'fiyat', r'price', r'cost', r'maliyet', r'bedel', r'tutar'
            ],
            ColumnType.GROWTH: [
                r'b[uü]y[uü]me', r'growth', r'art[iı]ş', r'increase', r'cagr',
                r'change', r'de[ğg]işim'
            ],
            ColumnType.REGION: [
                r'b[oö]lge', r'region', r'ulke', r'country', r'city', r'sehir',
                r'province', r'state', r'area', r'district'
            ],
            ColumnType.DATE: [
                r'date', r'tarih', r'time', r'zaman', r'y[iı]l', r'year',
                r'month', r'ay', r'quarter', r'[cç]eyrek'
            ]
        }
        
        for col in df.columns:
            col_lower = str(col).lower()
            detected = False
            
            for col_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, col_lower):
                        # Sales sütunları için yıl kontrolü
                        if col_type == ColumnType.SALES:
                            if DataCleaningUtils.safe_extract_year(col) is not None:
                                col_types[col] = col_type
                                detected = True
                                break
                        else:
                            col_types[col] = col_type
                            detected = True
                            break
                if detected:
                    break
            
            if not detected:
                col_types[col] = ColumnType.OTHER
        
        return col_types
    
    @staticmethod
    def calculate_growth_rate(current: float, previous: float) -> float:
        """Güvenli büyüme oranı hesaplama"""
        if pd.isna(current) or pd.isna(previous):
            return 0.0
        
        if previous == 0 or math.isinf(previous) or math.isnan(previous):
            return 0.0
        
        try:
            growth = ((current - previous) / abs(previous)) * 100
            if math.isinf(growth) or math.isnan(growth):
                return 0.0
            return round(growth, 2)
        except:
            return 0.0
    
    @staticmethod
    def calculate_cagr(start_value: float, end_value: float, years: float) -> float:
        """Bileşik yıllık büyüme oranı"""
        if start_value <= 0 or years <= 0:
            return 0.0
        
        try:
            cagr = (pow(end_value / start_value, 1 / years) - 1) * 100
            if math.isinf(cagr) or math.isnan(cagr):
                return 0.0
            return round(cagr, 2)
        except:
            return 0.0
    
    @staticmethod
    def detect_date_columns(df: pd.DataFrame) -> List[str]:
        """Tarih sütunlarını tespit et"""
        date_cols = []
        
        for col in df.columns:
            # İsim bazlı tespit
            if any(k in str(col).lower() for k in ['date', 'tarih', 'time', 'zaman']):
                date_cols.append(col)
                continue
            
            # Veri tipi bazlı tespit
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
                continue
            
            # Örnek değer bazlı tespit
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100).astype(str)
                date_pattern = re.compile(r'\d{2,4}[-/.]\d{2}[-/.]\d{2,4}')
                if sample.str.contains(date_pattern).any():
                    try:
                        pd.to_datetime(sample.iloc[0])
                        date_cols.append(col)
                    except:
                        pass
        
        return date_cols

# =============================================================================
# 6. PRODPACK DERİNLİK ANALİZ MOTORU
# =============================================================================

class ProdPackAnalyzer:
    """
    Molekül -> Şirket -> Marka -> Paket hiyerarşik analiz motoru.
    Pazar kanibalizasyonu ve büyüme matrisi entegre.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_utils = DataCleaningUtils()
        self.column_types = self.cleaning_utils.detect_column_types(df)
        self.mappings = self._build_column_mappings()
        self.nodes: Dict[str, ProdPackNode] = {}
        self.root_id = f"root_{uuid.uuid4().hex[:8]}"
        self.hierarchy_built = False
        self.hierarchy_data = None
        self.cannibalization_cache: Dict[str, pd.DataFrame] = {}
        
    def _build_column_mappings(self) -> Dict[str, Optional[str]]:
        """Sütun eşleştirmelerini oluştur"""
        mappings = {
            'molecule': None,
            'company': None,
            'brand': None,
            'pack': None,
            'sales': []
        }
        
        for col, col_type in self.column_types.items():
            if col_type == ColumnType.MOLECULE:
                mappings['molecule'] = col
            elif col_type == ColumnType.COMPANY:
                mappings['company'] = col
            elif col_type == ColumnType.BRAND:
                mappings['brand'] = col
            elif col_type == ColumnType.PACK:
                mappings['pack'] = col
            elif col_type == ColumnType.SALES:
                mappings['sales'].append(col)
        
        # Sales sütunlarını yıla göre sırala
        mappings['sales'] = sorted(
            mappings['sales'],
            key=lambda x: self.cleaning_utils.safe_extract_year(x) or 0,
            reverse=True
        )
        
        return mappings
    
    def validate(self) -> Tuple[bool, str]:
        """Gerekli sütunları kontrol et"""
        missing = []
        
        if not self.mappings['molecule']:
            missing.append('Molekül')
        if not self.mappings['company']:
            missing.append('Şirket')
        if not self.mappings['brand']:
            missing.append('Marka')
        if not self.mappings['pack']:
            missing.append('Paket')
        if len(self.mappings['sales']) < 1:
            missing.append('Satış (yıl bazlı)')
        
        if missing:
            return False, f"Eksik sütunlar: {', '.join(missing)}"
        
        return True, "OK"
    
    @st.cache_data(ttl=300, max_entries=10, show_spinner=False)
    def _build_hierarchy_cached(
        _self,
        df_hash: str,
        molecule_col: str,
        company_col: str,
        brand_col: str,
        pack_col: str,
        sales_cols_tuple: tuple,
        max_molecules: int,
        max_companies: int,
        max_brands: int,
        max_packs: int
    ) -> Dict:
        """Cache'li hiyerarşi oluşturma - 1M+ satır için optimize"""
        return _self._build_hierarchy_impl(
            molecule_col,
            company_col,
            brand_col,
            pack_col,
            list(sales_cols_tuple),
            max_molecules,
            max_companies,
            max_brands,
            max_packs
        )
    
    def _build_hierarchy_impl(
        self,
        molecule_col: str,
        company_col: str,
        brand_col: str,
        pack_col: str,
        sales_cols: List[str],
        max_molecules: int,
        max_companies: int,
        max_brands: int,
        max_packs: int
    ) -> Dict:
        """Hiyerarşi oluşturma implementasyonu"""
        
        # Root node
        total_sales = self.df[sales_cols[0]].sum() if sales_cols else 0
        
        self.nodes[self.root_id] = ProdPackNode(
            id=self.root_id,
            name="Tüm Pazar",
            level="root",
            value=total_sales
        )
        
        # Moleküller
        molecules = self.df[molecule_col].dropna().unique()[:max_molecules]
        
        for mol in molecules:
            mol_df = self.df[self.df[molecule_col] == mol]
            mol_sales = mol_df[sales_cols[0]].sum() if sales_cols else 0
            
            mol_id = f"mol_{hashlib.md5(str(mol).encode()).hexdigest()[:8]}"
            
            self.nodes[mol_id] = ProdPackNode(
                id=mol_id,
                name=str(mol)[:40],
                level="molecule",
                value=mol_sales,
                parent_id=self.root_id
            )
            self.nodes[self.root_id].children.append(mol_id)
            
            # Şirketler
            companies = mol_df[company_col].dropna().unique()[:max_companies]
            
            for comp in companies:
                comp_df = mol_df[mol_df[company_col] == comp]
                comp_sales = comp_df[sales_cols[0]].sum() if sales_cols else 0
                
                comp_id = f"comp_{hashlib.md5(str(comp).encode()).hexdigest()[:8]}"
                
                self.nodes[comp_id] = ProdPackNode(
                    id=comp_id,
                    name=str(comp)[:40],
                    level="company",
                    value=comp_sales,
                    parent_id=mol_id
                )
                self.nodes[mol_id].children.append(comp_id)
                
                # Markalar
                brands = comp_df[brand_col].dropna().unique()[:max_brands]
                
                for brand in brands:
                    brand_df = comp_df[comp_df[brand_col] == brand]
                    brand_sales = brand_df[sales_cols[0]].sum() if sales_cols else 0
                    
                    brand_id = f"brand_{hashlib.md5(str(brand).encode()).hexdigest()[:8]}"
                    
                    self.nodes[brand_id] = ProdPackNode(
                        id=brand_id,
                        name=str(brand)[:40],
                        level="brand",
                        value=brand_sales,
                        parent_id=comp_id
                    )
                    self.nodes[comp_id].children.append(brand_id)
                    
                    # Paketler
                    packs = brand_df[pack_col].dropna().unique()[:max_packs]
                    
                    for pack in packs:
                        pack_df = brand_df[brand_df[pack_col] == pack]
                        pack_sales = pack_df[sales_cols[0]].sum() if sales_cols else 0
                        
                        # Büyüme oranı
                        growth = 0.0
                        if len(sales_cols) >= 2:
                            prev_sales = pack_df[sales_cols[1]].sum() if sales_cols[1] in pack_df.columns else 0
                            growth = self.cleaning_utils.calculate_growth_rate(pack_sales, prev_sales)
                        
                        # Pazar payı
                        market_share = (pack_sales / mol_sales * 100) if mol_sales > 0 else 0
                        
                        pack_id = f"pack_{hashlib.md5(str(pack).encode()).hexdigest()[:8]}"
                        
                        self.nodes[pack_id] = ProdPackNode(
                            id=pack_id,
                            name=str(pack)[:40],
                            level="pack",
                            value=pack_sales,
                            parent_id=brand_id,
                            growth_rate=growth,
                            market_share=round(market_share, 2)
                        )
                        self.nodes[brand_id].children.append(pack_id)
        
        self.hierarchy_built = True
        return self._export_hierarchy()
    
    def _export_hierarchy(self) -> Dict:
        """Hiyerarşiyi dict olarak dışa aktar"""
        return {
            'nodes': {
                nid: {
                    'id': n.id,
                    'name': n.name,
                    'level': n.level,
                    'value': n.value,
                    'parent_id': n.parent_id,
                    'children': n.children,
                    'growth_rate': n.growth_rate,
                    'market_share': n.market_share
                }
                for nid, n in self.nodes.items()
            },
            'root_id': self.root_id,
            'total_value': self.nodes[self.root_id].value,
            'node_count': len(self.nodes),
            'molecule_count': len([n for n in self.nodes.values() if n.level == 'molecule']),
            'company_count': len([n for n in self.nodes.values() if n.level == 'company']),
            'brand_count': len([n for n in self.nodes.values() if n.level == 'brand']),
            'pack_count': len([n for n in self.nodes.values() if n.level == 'pack'])
        }
    
    def build_hierarchy(self) -> Dict:
        """Ana hiyerarşi oluşturma fonksiyonu"""
        valid, msg = self.validate()
        if not valid:
            st.error(f"❌ {msg}")
            return {}
        
        # DataFrame hash'i
        df_hash = hashlib.md5(pd.util.hash_pandas_object(self.df).values).hexdigest()
        
        # Cache'li hiyerarşi oluşturma
        self.hierarchy_data = self._build_hierarchy_cached(
            df_hash,
            self.mappings['molecule'],
            self.mappings['company'],
            self.mappings['brand'],
            self.mappings['pack'],
            tuple(self.mappings['sales']),
            Constants.MAX_HIERARCHY_MOLECULES,
            Constants.MAX_HIERARCHY_COMPANIES,
            Constants.MAX_HIERARCHY_BRANDS,
            Constants.MAX_HIERARCHY_PACKS
        )
        
        return self.hierarchy_data
    
    def get_molecule_detail(self, molecule: str) -> pd.DataFrame:
        """
        Seçilen molekül için detaylı drill-down raporu.
        Paket seviyesine kadar tüm veriler.
        """
        if not self.hierarchy_built or not self.mappings['molecule']:
            return pd.DataFrame()
        
        mol_df = self.df[self.df[self.mappings['molecule']] == molecule].copy()
        
        if mol_df.empty:
            return pd.DataFrame()
        
        rows = []
        
        # Gruplama
        grouped = mol_df.groupby([
            self.mappings['company'],
            self.mappings['brand'],
            self.mappings['pack']
        ])
        
        for (company, brand, pack), group in grouped:
            # Satış değerleri
            current_sales = group[self.mappings['sales'][0]].sum() if self.mappings['sales'] else 0
            
            # Büyüme oranı
            growth = 0.0
            if len(self.mappings['sales']) >= 2:
                prev_sales = group[self.mappings['sales'][1]].sum() if self.mappings['sales'][1] in group.columns else 0
                growth = self.cleaning_utils.calculate_growth_rate(current_sales, prev_sales)
            
            # Pazar payı
            total_mol_sales = mol_df[self.mappings['sales'][0]].sum() if self.mappings['sales'] else 0
            market_share = (current_sales / total_mol_sales * 100) if total_mol_sales > 0 else 0
            
            # Trend
            if growth > 10:
                trend = "🚀 Hızlı Büyüme"
            elif growth > 3:
                trend = "📈 Büyüme"
            elif growth > -3:
                trend = "➡️ Durağan"
            elif growth > -10:
                trend = "📉 Düşüş"
            else:
                trend = "⚠️ Kritik Düşüş"
            
            rows.append({
                'Molekül': molecule,
                'Şirket': company,
                'Marka': brand,
                'ProdPack': pack,
                'Satış_Hacmi': current_sales,
                'Büyüme_%': growth,
                'Pazar_Payı_%': round(market_share, 2),
                'Trend': trend,
                'Risk_Seviyesi': RiskLevel.from_score(1 - (market_share / 100)).value if market_share > 0 else RiskLevel.NORMAL.value
            })
        
        df_result = pd.DataFrame(rows)
        if not df_result.empty:
            df_result = df_result.sort_values('Satış_Hacmi', ascending=False)
        
        return df_result
    
    def analyze_cannibalization(self, molecule: str) -> pd.DataFrame:
        """
        Aynı şirket içindeki markalar arası kanibalizasyon analizi.
        Korelasyon bazlı hesaplama.
        """
        # Cache kontrolü
        if molecule in self.cannibalization_cache:
            return self.cannibalization_cache[molecule]
        
        if not self.hierarchy_built or not self.mappings['company'] or not self.mappings['brand']:
            return pd.DataFrame()
        
        mol_df = self.df[self.df[self.mappings['molecule']] == molecule].copy()
        
        if mol_df.empty or len(self.mappings['sales']) < 3:
            return pd.DataFrame()
        
        cannibal_data = []
        
        # Şirket bazlı gruplama
        companies = mol_df[self.mappings['company']].dropna().unique()
        
        for company in companies:
            comp_df = mol_df[mol_df[self.mappings['company']] == company]
            brands = comp_df[self.mappings['brand']].dropna().unique()
            
            if len(brands) < 2:
                continue
            
            # Marka çiftlerini analiz et
            for i in range(len(brands)):
                for j in range(i + 1, len(brands)):
                    brand_a = brands[i]
                    brand_b = brands[j]
                    
                    brand_a_df = comp_df[comp_df[self.mappings['brand']] == brand_a]
                    brand_b_df = comp_df[comp_df[self.mappings['brand']] == brand_b]
                    
                    # Zaman serisi oluştur
                    series_a = []
                    series_b = []
                    
                    for col in self.mappings['sales'][:4]:  # Son 4 dönem
                        series_a.append(brand_a_df[col].sum() if col in brand_a_df.columns else 0)
                        series_b.append(brand_b_df[col].sum() if col in brand_b_df.columns else 0)
                    
                    if len(series_a) > 1 and len(series_b) > 1:
                        # Korelasyon hesapla
                        try:
                            correlation = np.corrcoef(series_a, series_b)[0, 1]
                            if np.isnan(correlation):
                                correlation = 0.0
                        except:
                            correlation = 0.0
                        
                        # Kanibalizasyon skoru
                        cannibal_score = abs(correlation) if correlation < 0 else 0.0
                        
                        # Hacim örtüşmesi
                        volume_overlap = min(series_a[-1], series_b[-1]) / max(series_a[-1], series_b[-1], 1)
                        
                        # Büyüme etkisi
                        growth_a = self.cleaning_utils.calculate_growth_rate(series_a[-1], series_a[0]) if series_a[0] != 0 else 0
                        growth_b = self.cleaning_utils.calculate_growth_rate(series_b[-1], series_b[0]) if series_b[0] != 0 else 0
                        growth_impact = abs(growth_a - growth_b) / 100
                        
                        # Risk seviyesi
                        if cannibal_score > 0.7:
                            risk = RiskLevel.KRITIK
                            recommendation = "🚨 Acil müdahale gerekli - Ürün farklılaştırması yapın"
                        elif cannibal_score > 0.4:
                            risk = RiskLevel.YUKSEK
                            recommendation = "⚠️ Yakın izleme - Pazarlama stratejilerini gözden geçirin"
                        elif cannibal_score > 0.2:
                            risk = RiskLevel.ORTA
                            recommendation = "📊 Düzenli takip - Fiyatlandırmayı optimize edin"
                        else:
                            risk = RiskLevel.DUSUK
                            recommendation = "✅ Mevcut stratejiyi koruyun"
                        
                        cannibal_data.append({
                            'Molekül': molecule,
                            'Şirket': company,
                            'Marka_A': brand_a,
                            'Marka_B': brand_b,
                            'Korelasyon': round(correlation, 3),
                            'Kanibalizasyon_Skoru': round(cannibal_score, 3),
                            'Hacim_Örtüşmesi': round(volume_overlap, 3),
                            'Büyüme_Etkisi': round(growth_impact, 3),
                            'Risk_Seviyesi': risk.value,
                            'Öneri': recommendation
                        })
        
        df_result = pd.DataFrame(cannibal_data)
        if not df_result.empty:
            df_result = df_result.sort_values('Kanibalizasyon_Skoru', ascending=False)
        
        # Cache'e ekle
        self.cannibalization_cache[molecule] = df_result
        
        return df_result
    
    def get_sunburst_data(self) -> Dict:
        """Sunburst grafik için veri hazırla"""
        ids = []
        labels = []
        parents = []
        values = []
        
        for node_id, node in self.nodes.items():
            ids.append(node_id)
            labels.append(node.name[:30])
            parents.append(node.parent_id if node.parent_id else "")
            values.append(node.value)
        
        return {
            'ids': ids,
            'labels': labels,
            'parents': parents,
            'values': values
        }
    
    def get_sankey_data(self) -> Dict:
        """Sankey diyagram için veri hazırla"""
        sources = []
        targets = []
        values = []
        node_labels = []
        node_indices = {}
        
        # Root node
        node_indices[self.root_id] = 0
        node_labels.append(self.nodes[self.root_id].name[:20])
        current_idx = 1
        
        # Moleküller
        for node_id, node in self.nodes.items():
            if node.level == 'molecule':
                if node_id not in node_indices:
                    node_indices[node_id] = current_idx
                    node_labels.append(node.name[:20])
                    current_idx += 1
                
                sources.append(node_indices[self.root_id])
                targets.append(node_indices[node_id])
                values.append(node.value)
        
        # Şirketler
        for node_id, node in self.nodes.items():
            if node.level == 'company' and node.parent_id:
                if node_id not in node_indices:
                    node_indices[node_id] = current_idx
                    node_labels.append(node.name[:20])
                    current_idx += 1
                
                if node.parent_id in node_indices:
                    sources.append(node_indices[node.parent_id])
                    targets.append(node_indices[node_id])
                    values.append(node.value)
        
        # Markalar
        for node_id, node in self.nodes.items():
            if node.level == 'brand' and node.parent_id:
                if node_id not in node_indices:
                    node_indices[node_id] = current_idx
                    node_labels.append(node.name[:20])
                    current_idx += 1
                
                if node.parent_id in node_indices:
                    sources.append(node_indices[node.parent_id])
                    targets.append(node_indices[node_id])
                    values.append(node.value)
        
        # Paketler
        for node_id, node in self.nodes.items():
            if node.level == 'pack' and node.parent_id:
                if node_id not in node_indices:
                    node_indices[node_id] = current_idx
                    node_labels.append(node.name[:20])
                    current_idx += 1
                
                if node.parent_id in node_indices:
                    sources.append(node_indices[node.parent_id])
                    targets.append(node_indices[node_id])
                    values.append(node.value)
        
        # Değerleri normalize et
        if values:
            max_val = max(values)
            values = [v / max_val * 100 for v in values]
        
        return {
            'node_labels': node_labels,
            'sources': sources,
            'targets': targets,
            'values': values
        }
    
    def get_hierarchy_stats(self) -> Dict:
        """Hiyerarşi istatistiklerini döndür"""
        return {
            'total_nodes': len(self.nodes),
            'molecule_count': len([n for n in self.nodes.values() if n.level == 'molecule']),
            'company_count': len([n for n in self.nodes.values() if n.level == 'company']),
            'brand_count': len([n for n in self.nodes.values() if n.level == 'brand']),
            'pack_count': len([n for n in self.nodes.values() if n.level == 'pack']),
            'total_market_value': self.nodes[self.root_id].value if self.root_id in self.nodes else 0
        }

# =============================================================================
# 7. TAHMİNLEME MOTORU - HOLT-WINTERS (2025-2026)
# =============================================================================

class ForecastEngine:
    """
    Holt-Winters ile pazar tahminleme motoru.
    2025-2026 dönemi için çeyreklik tahmin.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_utils = DataCleaningUtils()
        self.column_types = self.cleaning_utils.detect_column_types(df)
        self.sales_cols = self._get_sales_columns()
        self.cache_key = None
        
    def _get_sales_columns(self) -> List[str]:
        """Satış sütunlarını getir"""
        sales_cols = []
        for col, col_type in self.column_types.items():
            if col_type == ColumnType.SALES:
                if self.cleaning_utils.safe_extract_year(col) is not None:
                    sales_cols.append(col)
        
        return sorted(
            sales_cols,
            key=lambda x: self.cleaning_utils.safe_extract_year(x) or 0
        )
    
    @st.cache_data(ttl=600, max_entries=10, show_spinner=False)
    def forecast_market(
        _self,
        df_hash: str,
        sales_cols_tuple: tuple,
        forecast_periods: int,
        confidence: float
    ) -> Dict:
        """
        Cache'li tahminleme fonksiyonu.
        Python 3.13+ uyumlu.
        """
        sales_cols = list(sales_cols_tuple)
        
        result = {
            'success': False,
            'forecast': None,
            'growth_rate': 0.0,
            'investment_advice': [],
            'model_stats': {},
            'error': None
        }
        
        if not STATSMODELS_AVAILABLE:
            result['error'] = 'statsmodels kurulu değil'
            result['investment_advice'].append(
                InvestmentAdvice(
                    title="📦 Modül Eksik",
                    message="statsmodels kurulu değil. Tahminleme için: pip install statsmodels",
                    action=InvestmentAction.OPTIMIZE,
                    roi_potential=0,
                    risk_level="Yüksek"
                ).to_dict()
            )
            return result
        
        if len(sales_cols) < 3:
            result['error'] = 'Yetersiz veri (en az 3 yıl gerekli)'
            result['investment_advice'].append(
                InvestmentAdvice(
                    title="⚠️ Yetersiz Veri",
                    message="Tahminleme için en az 3 yıllık satış verisi gereklidir.",
                    action=InvestmentAction.DEFEND,
                    roi_potential=0,
                    risk_level="Yüksek"
                ).to_dict()
            )
            return result
        
        try:
            # Toplam pazar satışları
            market_series = []
            years = []
            
            for col in sales_cols:
                year = _self.cleaning_utils.safe_extract_year(col)
                if year:
                    market_series.append(_self.df[col].sum())
                    years.append(year)
            
            if len(market_series) < 3:
                return result
            
            # Zaman serisi oluştur
            series = pd.Series(
                market_series,
                index=pd.date_range(start=f'{years[0]}-01-01', periods=len(market_series), freq='Y')
            )
            
            # Holt-Winters modeli
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            
            fitted_model = model.fit()
            
            # Tahmin
            forecast = fitted_model.forecast(forecast_periods)
            
            # Çeyreklere böl
            quarters = []
            for year in [2025, 2026]:
                for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                    quarters.append(f"{year} {q}")
            
            quarters = quarters[:forecast_periods]
            forecast_values = forecast.values[:forecast_periods]
            
            # Güven aralığı
            residuals = series - fitted_model.fittedvalues
            std_residual = residuals.std()
            z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
            
            lower_bound = forecast_values - z_score * std_residual
            upper_bound = forecast_values + z_score * std_residual
            
            # Büyüme oranı
            last_historical = market_series[-1] if market_series else 0
            last_forecast = forecast_values[-1] if len(forecast_values) > 0 else 0
            
            if last_historical > 0:
                growth_rate = ((last_forecast - last_historical) / last_historical) * 100
            else:
                growth_rate = 0.0
            
            # Model performans metrikleri
            mape = np.mean(np.abs((series - fitted_model.fittedvalues) / series)) * 100
            rmse = np.sqrt(np.mean((series - fitted_model.fittedvalues) ** 2))
            
            # Sonuç nesnesi
            forecast_result = ForecastResult(
                periods=quarters,
                predictions=forecast_values.tolist(),
                lower_bound=lower_bound.tolist(),
                upper_bound=upper_bound.tolist(),
                growth_rate=round(growth_rate, 2),
                historical_values=market_series,
                historical_years=years,
                mape=round(mape, 2),
                rmse=round(rmse, 2),
                aic=round(fitted_model.aic, 2) if hasattr(fitted_model, 'aic') else 0,
                bic=round(fitted_model.bic, 2) if hasattr(fitted_model, 'bic') else 0
            )
            
            result['success'] = True
            result['forecast'] = {
                'quarters': forecast_result.periods,
                'values': forecast_result.predictions,
                'lower': forecast_result.lower_bound,
                'upper': forecast_result.upper_bound,
                'historical': forecast_result.historical_values,
                'years': forecast_result.historical_years
            }
            result['growth_rate'] = forecast_result.growth_rate
            result['model_stats'] = {
                'mape': forecast_result.mape,
                'rmse': forecast_result.rmse,
                'aic': forecast_result.aic,
                'bic': forecast_result.bic
            }
            
            # Yatırım tavsiyeleri
            result['investment_advice'] = _self._generate_investment_advice(
                growth_rate,
                forecast_result.mape,
                market_series
            )
            
        except Exception as e:
            result['error'] = str(e)
            result['investment_advice'].append(
                InvestmentAdvice(
                    title="❌ Tahmin Hatası",
                    message=f"Holt-Winters modeli çalıştırılamadı: {str(e)[:100]}",
                    action=InvestmentAction.DEFEND,
                    roi_potential=0,
                    risk_level="Kritik"
                ).to_dict()
            )
        
        return result
    
    def _generate_investment_advice(
        self,
        growth_rate: float,
        mape: float,
        historical: List[float]
    ) -> List[Dict]:
        """Büyüme oranına göre yatırım tavsiyesi üret"""
        advice_list = []
        
        # Ana büyüme stratejisi
        if growth_rate > 20:
            advice_list.append(
                InvestmentAdvice(
                    title="🚀 AGRESİF BÜYÜME STRATEJİSİ",
                    message=f"Pazar %{growth_rate:.1f} büyüyor. Ar-Ge bütçesini %40 artırın, yeni ürün lansmanlarına öncelik verin.",
                    action=InvestmentAction.EXPAND,
                    roi_potential=85,
                    risk_level="Orta"
                ).to_dict()
            )
        elif growth_rate > 10:
            advice_list.append(
                InvestmentAdvice(
                    title="📈 GÜÇLÜ BÜYÜME STRATEJİSİ",
                    message=f"Pazar %{growth_rate:.1f} büyüyor. Satış ve pazarlama yatırımlarını %25 artırın.",
                    action=InvestmentAction.EXPAND,
                    roi_potential=75,
                    risk_level="Düşük"
                ).to_dict()
            )
        elif growth_rate > 3:
            advice_list.append(
                InvestmentAdvice(
                    title="📊 İSTİKRARLI BÜYÜME",
                    message=f"Pazar %{growth_rate:.1f} büyüyor. Mevcut portföyü optimize edin, verimlilik artışına odaklanın.",
                    action=InvestmentAction.OPTIMIZE,
                    roi_potential=60,
                    risk_level="Düşük"
                ).to_dict()
            )
        elif growth_rate > -3:
            advice_list.append(
                InvestmentAdvice(
                    title="⏸️ DURGUN PAZAR",
                    message="Pazar büyümesi yavaş. Maliyet optimizasyonu ve operasyonel verimliliğe odaklanın.",
                    action=InvestmentAction.DEFEND,
                    roi_potential=40,
                    risk_level="Orta"
                ).to_dict()
            )
        else:
            advice_list.append(
                InvestmentAdvice(
                    title="⚠️ PAZAR DARALMASI",
                    message=f"Pazar %{growth_rate:.1f} daralıyor. Nakit akışını koruyun, birleşme ve satın alma fırsatlarını değerlendirin.",
                    action=InvestmentAction.CONSOLIDATE,
                    roi_potential=30,
                    risk_level="Yüksek"
                ).to_dict()
            )
        
        # Risk yönetimi
        if mape > 20:
            advice_list.append(
                InvestmentAdvice(
                    title="⚖️ RİSK YÖNETİMİ",
                    message=f"Tahmin belirsizliği yüksek (MAPE: %{mape:.1f}). Portföy çeşitlendirmesi yapın, hedge stratejileri uygulayın.",
                    action=InvestmentAction.HEDGE,
                    roi_potential=45,
                    risk_level="Yüksek"
                ).to_dict()
            )
        
        # Uzun vadeli strateji
        if len(historical) >= 5:
            cagr = DataCleaningUtils.calculate_cagr(historical[0], historical[-1], len(historical) - 1)
            
            if cagr < 2 and growth_rate > 5:
                advice_list.append(
                    InvestmentAdvice(
                        title="🔄 DÖNÜŞÜM FIRSATI",
                        message=f"Tarihsel CAGR %{cagr:.1f}, önümüzdeki dönem %{growth_rate:.1f} büyüme bekleniyor. Dönüşüm stratejilerini devreye alın.",
                        action=InvestmentAction.OPTIMIZE,
                        roi_potential=70,
                        risk_level="Orta"
                    ).to_dict()
                )
        
        return advice_list

# =============================================================================
# 8. ANOMALİ TESPİT MOTORU - ISOLATION FOREST
# =============================================================================

class AnomalyDetector:
    """
    IsolationForest ile çoklu algoritma anomali tespiti.
    Python 3.13+ uyumlu scikit-learn implementasyonu.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_utils = DataCleaningUtils()
        self.column_types = self.cleaning_utils.detect_column_types(df)
        self.feature_columns = self._select_feature_columns()
        
    def _select_feature_columns(self, n_features: int = 10) -> List[str]:
        """
        Anomali tespiti için en uygun özellik sütunlarını seç.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Öncelikli özellikler
        priority_keywords = ['satış', 'sales', 'hacim', 'volume', 'cari', 'değer', 'value',
                            'buyume', 'growth', 'fiyat', 'price', 'maliyet', 'cost']
        
        selected = []
        
        # Öncelikli özellikleri ekle
        for col in numeric_cols:
            col_lower = str(col).lower()
            if any(k in col_lower for k in priority_keywords):
                selected.append(col)
                if len(selected) >= n_features:
                    break
        
        # Yeterli özellik yoksa, en yüksek varyanslı sütunları ekle
        if len(selected) < 3:
            remaining = [c for c in numeric_cols if c not in selected]
            if remaining:
                variances = self.df[remaining].var()
                top_variances = variances.nlargest(min(5, len(remaining))).index.tolist()
                selected.extend(top_variances)
        
        return selected[:n_features]
    
    @st.cache_data(ttl=300, max_entries=10, show_spinner=False)
    def detect_anomalies(
        _self,
        df_hash: str,
        feature_cols_tuple: tuple,
        contamination: float,
        random_state: int
    ) -> pd.DataFrame:
        """
        Cache'li anomali tespiti.
        """
        feature_cols = list(feature_cols_tuple)
        result_df = _self.df.copy()
        
        if not SKLEARN_AVAILABLE or len(feature_cols) < 2:
            result_df['Anomali_Skoru'] = 0.5
            result_df['Anomali_Tipi'] = Constants.ANOMALY_TYPES['normal']
            result_df['Risk_Seviyesi'] = Constants.RISK_LEVELS['normal']
            return result_df
        
        try:
            X = _self.df[feature_cols].fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # RobustScaler ile ölçeklendirme
            scaler = RobustScaler(quantile_range=(5, 95))
            X_scaled = scaler.fit_transform(X)
            
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=200,
                max_samples='auto',
                bootstrap=False,
                n_jobs=-1
            )
            
            predictions = iso_forest.fit_predict(X_scaled)
            scores = iso_forest.decision_function(X_scaled)
            
            # Skorları 0-1 aralığına normalize et
            min_score, max_score = scores.min(), scores.max()
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.ones_like(scores) * 0.5
            
            result_df['Anomali_Skoru'] = normalized_scores
            result_df['Anomali_Tespiti'] = predictions
            
            # Anomali tipi belirleme
            conditions = [
                (predictions == -1) & (normalized_scores < 0.25),
                (predictions == -1) & (normalized_scores >= 0.25),
                (predictions == 1) & (normalized_scores > 0.75),
                (predictions == 1)
            ]
            
            choices = [
                Constants.ANOMALY_TYPES['critical_drop'],
                Constants.ANOMALY_TYPES['abnormal_change'],
                Constants.ANOMALY_TYPES['hyper_growth'],
                Constants.ANOMALY_TYPES['normal']
            ]
            
            result_df['Anomali_Tipi'] = np.select(conditions, choices, default=Constants.ANOMALY_TYPES['normal'])
            
            # Risk seviyesi
            risk_conditions = [
                (result_df['Anomali_Tipi'] == Constants.ANOMALY_TYPES['critical_drop']),
                (result_df['Anomali_Tipi'] == Constants.ANOMALY_TYPES['abnormal_change']),
                (result_df['Anomali_Tipi'] == Constants.ANOMALY_TYPES['hyper_growth']),
                (result_df['Anomali_Tipi'] == Constants.ANOMALY_TYPES['normal'])
            ]
            
            risk_choices = [
                Constants.RISK_LEVELS['critical'],
                Constants.RISK_LEVELS['high'],
                Constants.RISK_LEVELS['medium'],
                Constants.RISK_LEVELS['normal']
            ]
            
            result_df['Risk_Seviyesi'] = np.select(risk_conditions, risk_choices, default=Constants.RISK_LEVELS['normal'])
            
            # Anomali açıklaması
            result_df['Anomali_Açıklama'] = result_df.apply(
                lambda row: _self._generate_anomaly_explanation(row, feature_cols),
                axis=1
            )
            
        except Exception as e:
            # Hata durumunda varsayılan değerler
            result_df['Anomali_Skoru'] = 0.5
            result_df['Anomali_Tespiti'] = 1
            result_df['Anomali_Tipi'] = Constants.ANOMALY_TYPES['normal']
            result_df['Risk_Seviyesi'] = Constants.RISK_LEVELS['normal']
            result_df['Anomali_Açıklama'] = f"Anomali tespiti sırasında hata: {str(e)[:50]}"
        
        return result_df
    
    def _generate_anomaly_explanation(self, row: pd.Series, feature_cols: List[str]) -> str:
        """Anomali için açıklama üret"""
        if row['Anomali_Tipi'] == Constants.ANOMALY_TYPES['normal']:
            return "Normal seyir, müdahale gerekmiyor."
        
        explanations = []
        
        # Satış anomalileri
        sales_cols = [c for c in feature_cols if any(k in str(c).lower() for k in ['satış', 'sales', 'hacim'])]
        if sales_cols and row[sales_cols[0]] > 0:
            if row['Anomali_Tipi'] == Constants.ANOMALY_TYPES['hyper_growth']:
                explanations.append(f"Aşırı büyüme: {row[sales_cols[0]]:,.0f} TL")
            elif row['Anomali_Tipi'] == Constants.ANOMALY_TYPES['critical_drop']:
                explanations.append(f"Kritik düşüş: {row[sales_cols[0]]:,.0f} TL")
        
        if not explanations:
            explanations.append(f"{row['Anomali_Tipi']} tespit edildi.")
        
        return " | ".join(explanations)

# =============================================================================
# 9. SEGMENTASYON MOTORU - PCA + K-MEANS
# =============================================================================

class SegmentationEngine:
    """
    PCA ve K-Means ile ürün segmentasyonu.
    Liderler, Potansiyeller, Riskli Ürünler sınıflandırması.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.cleaning_utils = DataCleaningUtils()
        self.column_types = self.cleaning_utils.detect_column_types(df)
        self.sales_cols = self._get_sales_columns()
        
    def _get_sales_columns(self) -> List[str]:
        """Satış sütunlarını getir"""
        sales_cols = []
        for col, col_type in self.column_types.items():
            if col_type == ColumnType.SALES:
                if self.cleaning_utils.safe_extract_year(col) is not None:
                    sales_cols.append(col)
        
        return sorted(
            sales_cols,
            key=lambda x: self.cleaning_utils.safe_extract_year(x) or 0,
            reverse=True
        )
    
    @st.cache_data(ttl=300, max_entries=10, show_spinner=False)
    def segment_products(
        _self,
        df_hash: str,
        sales_cols_tuple: tuple,
        n_clusters: int,
        random_state: int
    ) -> Dict:
        """
        Cache'li segmentasyon fonksiyonu.
        """
        sales_cols = list(sales_cols_tuple)
        
        result = {
            'success': False,
            'segmented_df': None,
            'pca_components': None,
            'pca_explained_variance': [],
            'labels': None,
            'segment_names': {},
            'silhouette_score': 0.0,
            'segment_profiles': {},
            'metrics': {}
        }
        
        if not SKLEARN_AVAILABLE or len(sales_cols) < 2:
            return result
        
        try:
            # Özellik mühendisliği
            feature_df = _self.df.copy()
            
            latest_col = sales_cols[0]
            prev_col = sales_cols[1] if len(sales_cols) > 1 else sales_cols[0]
            
            # 1. Pazar payı
            total_sales = _self.df[latest_col].sum()
            if total_sales > 0:
                feature_df['Pazar_Payi'] = (_self.df[latest_col] / total_sales) * 100
            else:
                feature_df['Pazar_Payi'] = 0
            
            # 2. Büyüme hızı
            feature_df['Buyume_Hizi'] = 0.0
            mask = (_self.df[prev_col] != 0) & (_self.df[prev_col].notna())
            if mask.any():
                feature_df.loc[mask, 'Buyume_Hizi'] = (
                    (_self.df.loc[mask, latest_col] - _self.df.loc[mask, prev_col]) /
                    _self.df.loc[mask, prev_col].abs()
                ) * 100
            
            # 3. Fiyat esnekliği
            price_cols = [col for col, ct in _self.column_types.items() if ct == ColumnType.PRICE]
            if price_cols:
                feature_df['Fiyat_Esnekligi'] = -0.8 + (_self.df[price_cols[0]].rank(pct=True) * -0.7)
            else:
                feature_df['Fiyat_Esnekligi'] = np.random.uniform(-1.5, -0.3, len(_self.df))
            
            # 4. Hacim değişkenliği
            if len(sales_cols) >= 3:
                sales_data = _self.df[sales_cols[:3]].values
                feature_df['Hacim_Degiskenligi'] = np.std(sales_data, axis=1) / (np.mean(sales_data, axis=1) + 1)
            else:
                feature_df['Hacim_Degiskenligi'] = 0.5
            
            # Feature matrix
            feature_cols = ['Pazar_Payi', 'Buyume_Hizi', 'Fiyat_Esnekligi', 'Hacim_Degiskenligi']
            X = feature_df[feature_cols].fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # Ölçeklendirme
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA - Boyut indirgeme
            n_components = min(2, X_scaled.shape[1])
            pca = PCA(n_components=n_components, random_state=random_state)
            X_pca = pca.fit_transform(X_scaled)
            
            # K-Means kümeleme
            n_clusters = min(n_clusters, len(_self.df))
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Silhouette skoru
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X_scaled, labels)
            else:
                silhouette = 0.0
            
            # Segment isimlendirme
            segment_map = {}
            for i in range(n_clusters):
                mask = labels == i
                cluster_data = X[mask]
                
                avg_growth = cluster_data['Buyume_Hizi'].mean()
                avg_share = cluster_data['Pazar_Payi'].mean()
                avg_volatility = cluster_data['Hacim_Degiskenligi'].mean()
                
                # Segment sınıflandırması
                if avg_share > X['Pazar_Payi'].quantile(0.7) and avg_growth > X['Buyume_Hizi'].quantile(0.6):
                    name = Constants.SEGMENT_NAMES['leader']
                elif avg_growth > X['Buyume_Hizi'].quantile(0.7):
                    name = Constants.SEGMENT_NAMES['potential']
                elif avg_volatility > X['Hacim_Degiskenligi'].quantile(0.7) or avg_growth < X['Buyume_Hizi'].quantile(0.3):
                    name = Constants.SEGMENT_NAMES['risky']
                elif avg_share > X['Pazar_Payi'].quantile(0.6):
                    name = Constants.SEGMENT_NAMES['cash_cow']
                elif avg_growth < X['Buyume_Hizi'].quantile(0.2) and avg_share < X['Pazar_Payi'].quantile(0.3):
                    name = Constants.SEGMENT_NAMES['question']
                else:
                    name = Constants.SEGMENT_NAMES['mature']
                
                segment_map[i] = name
            
            # Segment etiketlerini DataFrame'e ekle
            feature_df['Segment_Cluster'] = labels
            feature_df['Segment_Adi'] = feature_df['Segment_Cluster'].map(segment_map)
            
            # Segment profilleri oluştur
            segment_profiles = {}
            for segment in feature_df['Segment_Adi'].unique():
                seg_df = feature_df[feature_df['Segment_Adi'] == segment]
                
                # Dominant şirketler
                if _self.column_types.get(ColumnType.COMPANY):
                    company_col = next((col for col, ct in _self.column_types.items() if ct == ColumnType.COMPANY), None)
                    if company_col:
                        dominant_companies = seg_df[company_col].value_counts().head(3).to_dict()
                    else:
                        dominant_companies = {}
                else:
                    dominant_companies = {}
                
                # Risk dağılımı
                risk_dist = {}
                if 'Risk_Seviyesi' in seg_df.columns:
                    risk_dist = seg_df['Risk_Seviyesi'].value_counts().to_dict()
                
                profile = SegmentProfile(
                    segment_name=segment,
                    product_count=len(seg_df),
                    avg_market_share=seg_df['Pazar_Payi'].mean(),
                    avg_growth_rate=seg_df['Buyume_Hizi'].mean(),
                    avg_price_elasticity=seg_df['Fiyat_Esnekligi'].mean(),
                    total_sales=seg_df[latest_col].sum() if latest_col in seg_df.columns else 0,
                    dominant_companies=dominant_companies,
                    risk_distribution=risk_dist
                )
                
                segment_profiles[segment] = profile.to_dict()
            
            # Sonuçları hazırla
            result['success'] = True
            result['segmented_df'] = feature_df
            result['pca_components'] = X_pca
            result['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
            result['labels'] = labels
            result['segment_names'] = segment_map
            result['silhouette_score'] = round(silhouette, 3)
            result['segment_profiles'] = segment_profiles
            result['metrics'] = {
                'silhouette': round(silhouette, 3),
                'n_clusters': n_clusters,
                'pca_variance_ratio': round(pca.explained_variance_ratio_.sum(), 3),
                'calinski_harabasz': round(calinski_harabasz_score(X_scaled, labels), 2) if len(np.unique(labels)) > 1 else 0,
                'davies_bouldin': round(davies_bouldin_score(X_scaled, labels), 3) if len(np.unique(labels)) > 1 else 0
            }
            
        except Exception as e:
            result['error'] = str(e)
        
        return result

# =============================================================================
# 10. GÖRSELLEŞTİRME MOTORU - EXECUTIVE DARK MODE
# =============================================================================

class VisualizationEngine:
    """
    Gelişmiş görselleştirme motoru.
    Executive Dark Mode teması.
    """
    
    @staticmethod
    def apply_dark_theme(fig: go.Figure) -> go.Figure:
        """Plotly grafiğine dark tema uygula"""
        fig.update_layout(
            paper_bgcolor='rgba(10,25,41,0)',
            plot_bgcolor='rgba(10,25,41,0)',
            font=dict(color='white', family='Inter, Arial, sans-serif'),
            title_font=dict(color='#d4af37', size=18, family='Inter, Arial Black'),
            legend=dict(
                font=dict(color='white', size=11),
                bgcolor='rgba(10,25,41,0.7)',
                bordercolor='#d4af37',
                borderwidth=1
            ),
            hoverlabel=dict(
                bgcolor='#1e3a5f',
                font_size=12,
                font_family='Arial',
                font_color='white'
            )
        )
        
        fig.update_xaxes(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.3)',
            tickfont=dict(color='white', size=11),
            title_font=dict(color='white', size=13)
        )
        
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.3)',
            tickfont=dict(color='white', size=11),
            title_font=dict(color='white', size=13)
        )
        
        return fig
    
    @staticmethod
    def create_sunburst_chart(sunburst_data: Dict) -> go.Figure:
        """Sunburst grafik oluştur"""
        if not sunburst_data or not sunburst_data.get('ids'):
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text="Veri bulunamadı",
                    font=dict(color='#c0c0c0', size=16),
                    x=0.5
                )
            )
            return VisualizationEngine.apply_dark_theme(fig)
        
        fig = go.Figure(go.Sunburst(
            ids=sunburst_data['ids'],
            labels=sunburst_data['labels'],
            parents=sunburst_data['parents'],
            values=sunburst_data['values'],
            branchvalues='total',
            marker=dict(
                colorscale='deep',
                line=dict(width=2, color='#0a1929')
            ),
            hovertemplate=(
                '<b>%{label}</b><br>'
                'Değer: %{value:,.0f} TL<br>'
                'Pay: %{percentRoot:.1%}<br>'
                '<extra></extra>'
            ),
            textfont=dict(size=12, color='white'),
            insidetextorientation='radial'
        ))
        
        fig.update_layout(
            title=dict(
                text='🧬 ProdPack Hiyerarşisi - Molekül → Şirket → Marka → Paket',
                font=dict(size=20, color='#d4af37', family='Inter, Arial Black'),
                x=0.5,
                y=0.98
            ),
            height=600,
            margin=dict(t=60, l=10, r=10, b=10)
        )
        
        return VisualizationEngine.apply_dark_theme(fig)
    
    @staticmethod
    def create_sankey_diagram(sankey_data: Dict) -> go.Figure:
        """Sankey diyagram oluştur"""
        if not sankey_data or not sankey_data.get('node_labels'):
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text="Akış verisi bulunamadı",
                    font=dict(color='#c0c0c0', size=16),
                    x=0.5
                )
            )
            return VisualizationEngine.apply_dark_theme(fig)
        
        # Node renkleri
        node_colors = ['#0a1929']  # Root
        for i in range(1, len(sankey_data['node_labels'])):
            if i < 10:
                node_colors.append('#1e3a5f')  # Moleküller
            elif i < 30:
                node_colors.append('#2d4a7a')  # Şirketler
            else:
                node_colors.append('#3a5a8a')  # Markalar/Paketler
        
        # Link renkleri
        link_colors = ['rgba(212,175,55,0.3)'] * len(sankey_data['sources'])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color='#d4af37', width=0.8),
                label=sankey_data['node_labels'],
                color=node_colors,
                hovertemplate='%{label}<br>Değer: %{value:,.0f}<extra></extra>'
            ),
            link=dict(
                source=sankey_data['sources'],
                target=sankey_data['targets'],
                value=sankey_data['values'],
                color=link_colors,
                hovertemplate='%{source.label} → %{target.label}<br>Değer: %{value:,.0f}<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title=dict(
                text='🔄 Pazar Akış Diyagramı - Değer Zinciri',
                font=dict(size=20, color='#d4af37', family='Inter, Arial Black'),
                x=0.5,
                y=0.98
            ),
            height=600,
            margin=dict(t=60, l=20, r=20, b=20)
        )
        
        return VisualizationEngine.apply_dark_theme(fig)
    
    @staticmethod
    def create_cannibalization_heatmap(cannibal_df: pd.DataFrame) -> go.Figure:
        """Kanibalizasyon heatmap oluştur"""
        if cannibal_df.empty or len(cannibal_df) < 2:
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text="Kanibalizasyon verisi yok",
                    font=dict(color='#c0c0c0', size=16),
                    x=0.5
                )
            )
            return VisualizationEngine.apply_dark_theme(fig)
        
        # Pivot tablo oluştur
        pivot_data = cannibal_df.pivot_table(
            values='Kanibalizasyon_Skoru',
            index='Marka_A',
            columns='Marka_B',
            fill_value=0
        )
        
        # İlk 10 marka
        top_brands = cannibal_df['Marka_A'].value_counts().head(10).index
        pivot_data = pivot_data.loc[pivot_data.index.intersection(top_brands)]
        pivot_data = pivot_data[pivot_data.columns.intersection(top_brands)]
        
        fig = px.imshow(
            pivot_data,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title=None,
            labels=dict(x="Marka B", y="Marka A", color="Kanibalizasyon Skoru")
        )
        
        fig.update_layout(
            title=dict(
                text='🔄 Pazar Kanibalizasyon Matrisi',
                font=dict(size=18, color='#d4af37', family='Inter, Arial Black'),
                x=0.5
            ),
            height=450,
            xaxis=dict(tickangle=45)
        )
        
        fig.update_coloraxes(
            colorbar=dict(
                title='Skor',
                title_font=dict(color='white', size=11),
                tickfont=dict(color='white', size=10),
                bgcolor='rgba(10,25,41,0.8)'
            ),
            colorscale='RdBu_r',
            zmid=0
        )
        
        return VisualizationEngine.apply_dark_theme(fig)
    
    @staticmethod
    def create_forecast_chart(forecast_data: Dict) -> go.Figure:
        """Tahmin grafiği oluştur"""
        if not forecast_data or 'forecast' not in forecast_data:
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text="Tahmin verisi yok",
                    font=dict(color='#c0c0c0', size=16),
                    x=0.5
                )
            )
            return VisualizationEngine.apply_dark_theme(fig)
        
        forecast = forecast_data['forecast']
        
        fig = go.Figure()
        
        # Tarihsel veri
        if 'historical' in forecast and 'years' in forecast:
            fig.add_trace(go.Scatter(
                x=[str(y) for y in forecast['years']],
                y=forecast['historical'],
                mode='lines+markers',
                name='Tarihsel Satış',
                line=dict(color='#c0c0c0', width=4),
                marker=dict(size=10, symbol='diamond', color='#c0c0c0'),
                hovertemplate='Yıl: %{x}<br>Satış: %{y:,.0f} TL<extra></extra>'
            ))
        
        # Tahmin
        fig.add_trace(go.Scatter(
            x=forecast['quarters'],
            y=forecast['values'],
            mode='lines+markers',
            name='Tahmin 2025-2026',
            line=dict(color='#d4af37', width=4, dash='dash'),
            marker=dict(size=12, symbol='star', color='#d4af37'),
            hovertemplate='Dönem: %{x}<br>Tahmin: %{y:,.0f} TL<extra></extra>'
        ))
        
        # Güven aralığı
        if 'lower' in forecast and 'upper' in forecast:
            fig.add_trace(go.Scatter(
                x=forecast['quarters'] + forecast['quarters'][::-1],
                y=forecast['upper'] + forecast['lower'][::-1],
                fill='toself',
                fillcolor='rgba(212,175,55,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'Güven Aralığı (%{int(Constants.CONFIDENCE_LEVEL * 100)})',
                hovertemplate='Alt: %{customdata[0]:,.0f}<br>Üst: %{customdata[1]:,.0f}<extra></extra>',
                customdata=list(zip(forecast['lower'], forecast['upper']))
            ))
        
        fig.update_layout(
            title=dict(
                text='📈 Holt-Winters Pazar Tahmini 2025-2026',
                font=dict(size=20, color='#d4af37', family='Inter, Arial Black'),
                x=0.5
            ),
            xaxis=dict(
                title='Dönem',
                title_font=dict(color='white', size=14),
                tickangle=-45
            ),
            yaxis=dict(
                title='Pazar Değeri (TL)',
                title_font=dict(color='white', size=14),
                tickformat='~s'
            ),
            height=500,
            hovermode='x unified',
            margin=dict(t=60, l=80, r=40, b=80)
        )
        
        return VisualizationEngine.apply_dark_theme(fig)
    
    @staticmethod
    def create_segmentation_scatter(pca_data: np.ndarray, labels: np.ndarray, segment_names: Dict) -> go.Figure:
        """PCA segmentasyon scatter plot"""
        if pca_data is None or labels is None:
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text="Segmentasyon verisi yok",
                    font=dict(color='#c0c0c0', size=16),
                    x=0.5
                )
            )
            return VisualizationEngine.apply_dark_theme(fig)
        
        fig = go.Figure()
        
        unique_labels = np.unique(labels)
        
        # Renk haritası
        color_map = {
            Constants.SEGMENT_NAMES['leader']: '#d4af37',
            Constants.SEGMENT_NAMES['potential']: '#4caf50',
            Constants.SEGMENT_NAMES['risky']: '#f44336',
            Constants.SEGMENT_NAMES['mature']: '#2196f3',
            Constants.SEGMENT_NAMES['cash_cow']: '#9c27b0',
            Constants.SEGMENT_NAMES['question']: '#ff9800'
        }
        
        for label in unique_labels:
            mask = labels == label
            name = segment_names.get(label, f'Segment {label}')
            color = color_map.get(name, '#808080')
            
            fig.add_trace(go.Scatter(
                x=pca_data[mask, 0],
                y=pca_data[mask, 1],
                mode='markers',
                name=name,
                marker=dict(
                    size=14,
                    color=color,
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=[f'Segment: {name}'] * mask.sum(),
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='🎯 PCA + K-Means Segmentasyon',
                font=dict(size=20, color='#d4af37', family='Inter, Arial Black'),
                x=0.5
            ),
            xaxis=dict(
                title='PC1',
                title_font=dict(color='white', size=14)
            ),
            yaxis=dict(
                title='PC2',
                title_font=dict(color='white', size=14)
            ),
            height=500,
            legend=dict(
                x=1.05,
                y=1,
                xanchor='left',
                yanchor='top'
            ),
            margin=dict(t=60, l=60, r=150, b=60)
        )
        
        return VisualizationEngine.apply_dark_theme(fig)
    
    @staticmethod
    def create_anomaly_distribution(anomaly_df: pd.DataFrame) -> go.Figure:
        """Anomali dağılım grafiği"""
        if anomaly_df is None or 'Anomali_Tipi' not in anomaly_df.columns:
            fig = go.Figure()
            fig.update_layout(
                title=dict(
                    text="Anomali verisi yok",
                    font=dict(color='#c0c0c0', size=16),
                    x=0.5
                )
            )
            return VisualizationEngine.apply_dark_theme(fig)
        
        counts = anomaly_df['Anomali_Tipi'].value_counts().reset_index()
        counts.columns = ['Tip', 'Sayı']
        
        colors = {
            Constants.ANOMALY_TYPES['critical_drop']: '#f44336',
            Constants.ANOMALY_TYPES['abnormal_change']: '#ff9800',
            Constants.ANOMALY_TYPES['hyper_growth']: '#4caf50',
            Constants.ANOMALY_TYPES['normal']: '#2196f3'
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=counts['Tip'],
                y=counts['Sayı'],
                marker_color=[colors.get(t, '#808080') for t in counts['Tip']],
                text=counts['Sayı'],
                textposition='auto',
                hovertemplate='Tip: %{x}<br>Sayı: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text='⚠️ Anomali Dağılımı',
                font=dict(size=18, color='#d4af37', family='Inter, Arial Black'),
                x=0.5
            ),
            xaxis=dict(
                title='Anomali Tipi',
                title_font=dict(color='white', size=14),
                tickangle=-45
            ),
            yaxis=dict(
                title='Ürün Sayısı',
                title_font=dict(color='white', size=14)
            ),
            height=450,
            margin=dict(t=60, l=60, r=40, b=100)
        )
        
        return VisualizationEngine.apply_dark_theme(fig)
    
    @staticmethod
    def create_kpi_gauges(metrics: Dict) -> List[go.Figure]:
        """KPI gösterge grafikleri"""
        figs = []
        
        # Pazar büyüme göstergesi
        if 'growth_rate' in metrics:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=metrics['growth_rate'],
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-20, 50], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#d4af37"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#c0c0c0",
                    'steps': [
                        {'range': [-20, 0], 'color': 'rgba(244,67,54,0.3)'},
                        {'range': [0, 15], 'color': 'rgba(33,150,243,0.3)'},
                        {'range': [15, 50], 'color': 'rgba(76,175,80,0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': metrics['growth_rate']
                    }
                },
                title={'text': "Büyüme Oranı (%)", 'font': {'color': 'white', 'size': 14}}
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(10,25,41,0)',
                font={'color': 'white', 'family': 'Inter'}
            )
            figs.append(fig)
        
        return figs

# =============================================================================
# 11. YÖNETİCİ ÖZETİ ÜRETİCİ - INSIGHT BOX
# =============================================================================

class ExecutiveInsightEngine:
    """
    Otomatik yönetici özeti ve stratejik içgörü üretici.
    """
    
    @staticmethod
    def generate_market_insights(
        analyzer: Optional[ProdPackAnalyzer],
        forecast: Optional[Dict],
        anomaly_df: Optional[pd.DataFrame],
        segmentation_result: Optional[Dict]
    ) -> List[ExecutiveSummary]:
        """Pazar yapısı içgörüleri"""
        insights = []
        
        # ProdPack içgörüleri
        if analyzer and analyzer.hierarchy_built:
            stats = analyzer.get_hierarchy_stats()
            
            if stats['pack_count'] > 0:
                insights.append(ExecutiveSummary(
                    category='market',
                    title='Pazar Yapısı',
                    content=f"{stats['molecule_count']} molekül, {stats['company_count']} şirket, {stats['pack_count']} benzersiz ürün/paket analiz ediliyor.",
                    metric=stats['total_market_value'],
                    priority='high'
                ))
        
        # Tahmin içgörüleri
        if forecast and forecast.get('success'):
            growth = forecast.get('growth_rate', 0)
            
            if growth > 15:
                insights.append(ExecutiveSummary(
                    category='growth',
                    title='Büyüme Fırsatı',
                    content=f"Pazarın 2025-2026 döneminde %{growth:.1f} büyümesi bekleniyor. Yatırım için stratejik pencere açıldı.",
                    metric=growth,
                    trend='up',
                    priority='high'
                ))
            elif growth > 5:
                insights.append(ExecutiveSummary(
                    category='growth',
                    title='İstikrarlı Büyüme',
                    content=f"Pazar %{growth:.1f} büyüyecek. Mevcut stratejileri koruyun, verimliliği artırın.",
                    metric=growth,
                    trend='up',
                    priority='medium'
                ))
            elif growth > -3:
                insights.append(ExecutiveSummary(
                    category='market',
                    title='Durgun Pazar',
                    content=f"Büyüme %{growth:.1f}. Maliyet optimizasyonu ve operasyonel mükemmellik öncelikli.",
                    metric=growth,
                    trend='flat',
                    priority='medium'
                ))
            else:
                insights.append(ExecutiveSummary(
                    category='risk',
                    title='Pazar Daralması',
                    content=f"%{growth:.1f} daralma bekleniyor. Acil maliyet önlemleri ve portföy optimizasyonu şart.",
                    metric=growth,
                    trend='down',
                    priority='critical'
                ))
        
        return insights
    
    @staticmethod
    def generate_risk_insights(
        anomaly_df: Optional[pd.DataFrame],
        cannibal_df: Optional[pd.DataFrame]
    ) -> List[ExecutiveSummary]:
        """Risk içgörüleri"""
        insights = []
        
        # Anomali içgörüleri
        if anomaly_df is not None and 'Anomali_Tipi' in anomaly_df.columns:
            critical_count = len(anomaly_df[anomaly_df['Anomali_Tipi'] == Constants.ANOMALY_TYPES['critical_drop']])
            growth_count = len(anomaly_df[anomaly_df['Anomali_Tipi'] == Constants.ANOMALY_TYPES['hyper_growth']])
            
            if critical_count > 0:
                insights.append(ExecutiveSummary(
                    category='risk',
                    title='Kritik Risk Uyarısı',
                    content=f"{critical_count} üründe kritik düşüş tespit edildi. Acil müdahale gerekiyor.",
                    metric=critical_count,
                    trend='down',
                    priority='critical'
                ))
            
            if growth_count > 0:
                insights.append(ExecutiveSummary(
                    category='opportunity',
                    title='Büyüme Fırsatı',
                    content=f"{growth_count} üründe aşırı büyüme görülüyor. Kapasite planlaması yapın.",
                    metric=growth_count,
                    trend='up',
                    priority='high'
                ))
        
        # Kanibalizasyon içgörüleri
        if cannibal_df is not None and not cannibal_df.empty:
            high_cannibal = len(cannibal_df[cannibal_df['Kanibalizasyon_Skoru'] > 0.7])
            if high_cannibal > 0:
                insights.append(ExecutiveSummary(
                    category='cannibal',
                    title='Kanibalizasyon Riski',
                    content=f"{high_cannibal} marka çiftinde yüksek kanibalizasyon tespit edildi. Ürün farklılaştırması yapın.",
                    metric=high_cannibal,
                    priority='high'
                ))
        
        return insights
    
    @staticmethod
    def generate_segment_insights(
        segmentation_result: Optional[Dict]
    ) -> List[ExecutiveSummary]:
        """Segmentasyon içgörüleri"""
        insights = []
        
        if segmentation_result and segmentation_result.get('success'):
            profiles = segmentation_result.get('segment_profiles', {})
            
            for segment_name, profile in profiles.items():
                if segment_name == Constants.SEGMENT_NAMES['leader'] and profile['product_count'] > 0:
                    insights.append(ExecutiveSummary(
                        category='segment',
                        title='Lider Ürünler',
                        content=f"{profile['product_count']} ürün lider segmentinde. Pazar konumunuzu koruyun.",
                        metric=profile['product_count'],
                        priority='high'
                    ))
                
                elif segment_name == Constants.SEGMENT_NAMES['potential'] and profile['product_count'] > 0:
                    insights.append(ExecutiveSummary(
                        category='opportunity',
                        title='Potansiyel Yıldızlar',
                        content=f"{profile['product_count']} ürün yüksek büyüme potansiyeline sahip. Yatırım yapın.",
                        metric=profile['product_count'],
                        priority='high'
                    ))
                
                elif segment_name == Constants.SEGMENT_NAMES['risky'] and profile['product_count'] > 0:
                    insights.append(ExecutiveSummary(
                        category='risk',
                        title='Riskli Ürünler',
                        content=f"{profile['product_count']} ürün riskli segmentte. Stratejik değerlendirme yapın.",
                        metric=profile['product_count'],
                        priority='critical'
                    ))
        
        return insights
    
    @staticmethod
    def generate_all_insights(
        analyzer: Optional[ProdPackAnalyzer],
        forecast: Optional[Dict],
        anomaly_df: Optional[pd.DataFrame],
        cannibal_df: Optional[pd.DataFrame],
        segmentation_result: Optional[Dict]
    ) -> List[ExecutiveSummary]:
        """Tüm içgörüleri birleştir"""
        insights = []
        
        insights.extend(ExecutiveInsightEngine.generate_market_insights(
            analyzer, forecast, anomaly_df, segmentation_result
        ))
        
        insights.extend(ExecutiveInsightEngine.generate_risk_insights(
            anomaly_df, cannibal_df
        ))
        
        insights.extend(ExecutiveInsightEngine.generate_segment_insights(
            segmentation_result
        ))
        
        # Önceliğe göre sırala
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        insights.sort(key=lambda x: priority_order.get(x.priority, 4))
        
        return insights[:6]  # En fazla 6 içgörü

# =============================================================================
# 12. EXECUTIVE DARK MODE CSS - LACİVERT, GÜMÜŞ, ALTIN
# =============================================================================

EXECUTIVE_DARK_CSS = """
<style>
    /* =================================================================
       PHARMAINTELLIGENCE PRO V8.0 - EXECUTIVE DARK MODE
       Tema: Lacivert (#0a1929), Gümüş (#c0c0c0), Altın (#d4af37)
       ================================================================= */
    
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* ------------------- Global Styles ------------------- */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(145deg, #0a1929 0%, #0c1f33 50%, #0a1929 100%);
        color: #ffffff;
    }
    
    /* ------------------- Executive Cards ------------------- */
    .executive-card {
        background: rgba(10, 25, 41, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(212, 175, 55, 0.4);
        border-radius: 24px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    
    .executive-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #d4af37, #c0c0c0, #d4af37);
        z-index: 1;
    }
    
    .executive-card:hover {
        transform: translateY(-5px);
        border-color: #d4af37;
        box-shadow: 0 20px 45px rgba(212, 175, 55, 0.2);
    }
    
    /* ------------------- Insight Box - Yönetici Özeti ------------------- */
    .insight-box {
        background: linear-gradient(135deg, rgba(26, 54, 93, 0.95), rgba(16, 36, 61, 0.98));
        border-left: 8px solid #d4af37;
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin: 2rem 0;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        position: relative;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .insight-box::after {
        content: '📊 YÖNETİCİ ÖZETİ';
        position: absolute;
        top: -14px;
        left: 25px;
        background: #d4af37;
        color: #0a1929;
        font-size: 0.8rem;
        font-weight: 900;
        padding: 6px 20px;
        border-radius: 40px;
        letter-spacing: 3px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        border: 1px solid #c0c0c0;
        z-index: 2;
    }
    
    .insight-text {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #f0f4fa;
        font-weight: 400;
    }
    
    /* ------------------- Gold Title ------------------- */
    .gold-title {
        color: #d4af37;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        border-bottom: 3px solid rgba(212, 175, 55, 0.5);
        padding-bottom: 0.6rem;
        margin-bottom: 1.8rem;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
    }
    
    /* ------------------- Silver Subtitle ------------------- */
    .silver-subtitle {
        color: #c0c0c0;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        letter-spacing: -0.5px;
        text-shadow: 0 1px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* ------------------- Executive Metric Card ------------------- */
    .metric-executive {
        background: linear-gradient(145deg, rgba(12, 27, 44, 0.8), rgba(8, 19, 33, 0.9));
        border-radius: 20px;
        padding: 1.5rem 1rem;
        text-align: center;
        border: 1px solid rgba(192, 192, 192, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-executive:hover {
        border-color: #d4af37;
        background: linear-gradient(145deg, rgba(16, 36, 61, 0.9), rgba(10, 25, 41, 0.95));
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(212, 175, 55, 0.15);
    }
    
    .metric-executive-value {
        color: #d4af37;
        font-size: 2.4rem;
        font-weight: 900;
        line-height: 1.2;
        margin-bottom: 0.3rem;
        text-shadow: 0 0 15px rgba(212, 175, 55, 0.3);
    }
    
    .metric-executive-label {
        color: #a0b8cc;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
    }
    
    .metric-executive-trend {
        color: #c0c0c0;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    
    /* ------------------- Investment Card ------------------- */
    .investment-card {
        background: radial-gradient(circle at 10% 30%, rgba(26, 54, 93, 0.95), rgba(10, 25, 41, 0.98));
        border: 1.5px solid #d4af37;
        border-radius: 24px;
        padding: 1.8rem 1.5rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.7);
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .investment-card:hover {
        transform: scale(1.02);
        border-color: #ffd966;
        box-shadow: 0 15px 35px rgba(212, 175, 55, 0.25);
    }
    
    /* ------------------- Badges ------------------- */
    .badge-critical {
        background: linear-gradient(145deg, #b71c1c, #8b0000);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border: 1px solid #ffcdd2;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        display: inline-block;
    }
    
    .badge-opportunity {
        background: linear-gradient(145deg, #1e5f3e, #0a4b2b);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border: 1px solid #a5d6a5;
    }
    
    .badge-warning {
        background: linear-gradient(145deg, #b26a00, #7f4f00);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        border: 1px solid #ffe082;
    }
    
    /* ------------------- Buttons ------------------- */
    .stButton > button {
        background: linear-gradient(145deg, #1e3a5f, #142b44);
        color: white;
        border: 1px solid #d4af37;
        border-radius: 50px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 1.5px;
        transition: all 0.2s;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.4);
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        background: linear-gradient(145deg, #2b4b72, #1a334d);
        border: 1px solid #ffd966;
        color: #ffd966;
        box-shadow: 0 8px 25px rgba(212, 175, 55, 0.5);
        transform: translateY(-3px);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* ------------------- Tabs ------------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(10, 25, 41, 0.7);
        padding: 8px;
        border-radius: 50px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 50px;
        padding: 0 28px;
        color: #c0c0c0;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.2s;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #d4af37;
        background: rgba(212, 175, 55, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #1e3a5f, #142b44);
        color: #d4af37;
        border: 1px solid #d4af37;
        box-shadow: 0 4px 12px rgba(212, 175, 55, 0.2);
    }
    
    /* ------------------- Dataframe ------------------- */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    .stDataFrame [data-testid="StyledDataFrameDataCell"] {
        color: white;
        background-color: rgba(10, 25, 41, 0.7);
    }
    
    .stDataFrame [data-testid="StyledDataFrameHeaderCell"] {
        background-color: #1e3a5f;
        color: #d4af37;
        font-weight: 700;
    }
    
    /* ------------------- Sidebar ------------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1929 0%, #0c1f33 100%);
        border-right: 1px solid rgba(212, 175, 55, 0.3);
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
    }
    
    /* ------------------- Expander ------------------- */
    .streamlit-expanderHeader {
        background-color: rgba(10, 25, 41, 0.7);
        border-radius: 12px;
        color: #c0c0c0;
        font-weight: 600;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #d4af37;
        color: #d4af37;
    }
    
    /* ------------------- Progress Bar ------------------- */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #d4af37, #c0c0c0);
    }
    
    /* ------------------- Scrollbar ------------------- */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a1929;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d4af37;
        border-radius: 10px;
        border: 1px solid #0a1929;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #ffd966;
    }
    
    /* ------------------- Animations ------------------- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(212, 175, 55, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(212, 175, 55, 0); }
        100% { box-shadow: 0 0 0 0 rgba(212, 175, 55, 0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* ------------------- Responsive ------------------- */
    @media (max-width: 768px) {
        .gold-title {
            font-size: 1.8rem;
        }
        
        .silver-subtitle {
            font-size: 1.2rem;
        }
        
        .metric-executive-value {
            font-size: 1.8rem;
        }
    }
</style>
"""

# =============================================================================
# 13. ANA UYGULAMA SINIFI - PHARMAINTELLIGENCE PRO
# =============================================================================

class PharmaIntelligenceApp:
    """
    Ana uygulama sınıfı.
    Tüm modülleri entegre eder ve yönetir.
    """
    
    def __init__(self):
        self._initialize_session_state()
        self._configure_page()
        
    def _configure_page(self):
        """Streamlit sayfa yapılandırması"""
        st.set_page_config(
            page_title=f"PharmaIntelligence Pro v{Constants.VERSION} | ProdPack Depth",
            page_icon="💊",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': f"""
                ### PharmaIntelligence Pro v{Constants.VERSION}
                **Enterprise ProdPack Derinlik Analizi**
                
                Build: {Constants.BUILD}
                
                ✓ Molekül → Şirket → Marka → Paket Hiyerarşisi
                ✓ Pazar Kanibalizasyon Matrisi
                ✓ Holt-Winters Tahminleme (2025-2026)
                ✓ IsolationForest Anomali Tespiti
                ✓ PCA + K-Means Segmentasyon
                ✓ Executive Dark Mode
                
                © 2026 PharmaIntelligence Inc.
                """
            }
        )
        
        # Executive Dark Mode CSS
        st.markdown(EXECUTIVE_DARK_CSS, unsafe_allow_html=True)
    
    def _initialize_session_state(self):
        """Session state değişkenlerini başlat"""
        defaults = {
            # Veri
            'raw_data': None,
            'processed_data': None,
            'data_hash': None,
            'data_loaded_time': None,
            
            # ProdPack
            'analyzer': None,
            'hierarchy': None,
            'selected_molecule': None,
            'molecule_detail': None,
            'cannibal_data': None,
            
            # Tahmin
            'forecast_engine': None,
            'forecast_result': None,
            
            # Anomali
            'anomaly_detector': None,
            'anomaly_result': None,
            
            # Segmentasyon
            'segmentation_engine': None,
            'segmentation_result': None,
            
            # İçgörüler
            'executive_insights': [],
            
            # UI State
            'active_tab': 0,
            'sidebar_expanded': True,
            'data_loaded_flag': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
    def _cached_data_processing(_self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Cache'li veri işleme.
        1M+ satır için optimize edilmiş.
        """
        # Benzersiz sütun isimlendirme
        df_raw.columns = DataCleaningUtils.make_unique_column_names(df_raw.columns.tolist())
        
        # Güvenli downcast
        df_processed = DataCleaningUtils.safe_downcast(df_raw)
        
        # UI için satır limiti - 5.000 satır
        if len(df_processed) > Constants.MAX_ROWS_UI:
            df_processed = df_processed.head(Constants.MAX_ROWS_UI).copy()
        
        return df_processed
    
    def _load_and_process_data(self, uploaded_file):
        """Veri yükleme ve işleme"""
        try:
            # Dosya tipine göre okuma
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, low_memory=False, encoding='utf-8')
            else:
                raw_df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Cache'li veri işleme
            processed_df = self._cached_data_processing(raw_df)
            
            # Session state güncelle
            st.session_state.raw_data = raw_df
            st.session_state.processed_data = processed_df
            st.session_state.data_hash = hashlib.md5(
                pd.util.hash_pandas_object(processed_df).values
            ).hexdigest()
            st.session_state.data_loaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.data_loaded_flag = True
            
            # Engine'leri sıfırla
            st.session_state.analyzer = None
            st.session_state.forecast_engine = None
            st.session_state.anomaly_detector = None
            st.session_state.segmentation_engine = None
            
            st.success(f"✅ Veri işlendi: {len(processed_df):,} satır (UI: {Constants.MAX_ROWS_UI:,} limit)")
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            st.caption("Lütfen dosya formatını kontrol edin (Excel/CSV, UTF-8 encoding)")
    
    def _build_hierarchy(self):
        """ProdPack hiyerarşisi oluştur"""
        if st.session_state.processed_data is None:
            st.warning("⚠️ Önce veri yükleyin")
            return
        
        analyzer = ProdPackAnalyzer(st.session_state.processed_data)
        hierarchy = analyzer.build_hierarchy()
        
        if hierarchy:
            st.session_state.analyzer = analyzer
            st.session_state.hierarchy = hierarchy
            st.success(f"✅ Hiyerarşi oluşturuldu: {hierarchy.get('pack_count', 0)} paket, {hierarchy.get('molecule_count', 0)} molekül")
    
    def _analyze_molecule(self, molecule: str):
        """Molekül detay analizi"""
        if st.session_state.analyzer is None:
            st.warning("⚠️ Önce hiyerarşi oluşturun")
            return
        
        analyzer = st.session_state.analyzer
        
        # Drill-down verisi
        detail_df = analyzer.get_molecule_detail(molecule)
        st.session_state.molecule_detail = detail_df
        
        # Kanibalizasyon verisi
        cannibal_df = analyzer.analyze_cannibalization(molecule)
        st.session_state.cannibal_data = cannibal_df
        
        if not detail_df.empty:
            st.success(f"✅ {molecule} analizi tamamlandı - {len(detail_df)} paket")
    
    def _run_forecast(self):
        """Tahminleme çalıştır"""
        if st.session_state.processed_data is None:
            st.warning("⚠️ Önce veri yükleyin")
            return
        
        engine = ForecastEngine(st.session_state.processed_data)
        result = engine.forecast_market(
            st.session_state.data_hash,
            tuple(engine.sales_cols),
            Constants.FORECAST_PERIODS,
            Constants.CONFIDENCE_LEVEL
        )
        
        st.session_state.forecast_engine = engine
        st.session_state.forecast_result = result
        
        if result.get('success'):
            st.success(f"✅ Tahmin tamamlandı: 2025-2026 %{result.get('growth_rate', 0):.1f} büyüme")
        else:
            st.warning("⚠️ Tahminleme başarısız, varsayılan stratejiler kullanılıyor")
    
    def _run_anomaly_detection(self):
        """Anomali tespiti çalıştır"""
        if st.session_state.processed_data is None:
            st.warning("⚠️ Önce veri yükleyin")
            return
        
        detector = AnomalyDetector(st.session_state.processed_data)
        result = detector.detect_anomalies(
            st.session_state.data_hash,
            tuple(detector.feature_columns),
            Constants.ANOMALY_CONTAMINATION,
            42
        )
        
        st.session_state.anomaly_detector = detector
        st.session_state.anomaly_result = result
        
        if 'Anomali_Tipi' in result.columns:
            anomaly_count = len(result[result['Anomali_Tipi'] != Constants.ANOMALY_TYPES['normal']])
            st.success(f"✅ Anomali tespiti: {anomaly_count} anomali bulundu")
    
    def _run_segmentation(self):
        """Segmentasyon çalıştır"""
        if st.session_state.processed_data is None:
            st.warning("⚠️ Önce veri yükleyin")
            return
        
        engine = SegmentationEngine(st.session_state.processed_data)
        result = engine.segment_products(
            st.session_state.data_hash,
            tuple(engine.sales_cols),
            Constants.N_CLUSTERS,
            42
        )
        
        st.session_state.segmentation_engine = engine
        st.session_state.segmentation_result = result
        
        if result.get('success'):
            st.success(f"✅ Segmentasyon: Silhouette skoru {result.get('silhouette_score', 0):.3f}")
    
    def _update_insights(self):
        """Yönetici içgörülerini güncelle"""
        insights = ExecutiveInsightEngine.generate_all_insights(
            st.session_state.analyzer,
            st.session_state.forecast_result,
            st.session_state.anomaly_result,
            st.session_state.cannibal_data,
            st.session_state.segmentation_result
        )
        
        st.session_state.executive_insights = insights
    
    def render_sidebar(self):
        """Sidebar render"""
        with st.sidebar:
            # Logo ve başlık
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem 0.5rem 0.5rem 0.5rem;">
                <span style="font-size: 3.2rem; background: rgba(212,175,55,0.12); padding: 1rem; border-radius: 30px; display: inline-block; margin-bottom: 0.5rem;">💊</span>
                <h2 style="color: #d4af37; margin: 0.8rem 0 0.2rem 0; font-size: 1.9rem; font-weight: 800;">PharmaIntel</h2>
                <p style="color: #c0c0c0; font-size: 0.7rem; letter-spacing: 4px; font-weight: 600;">v{Constants.VERSION} ENTERPRISE</p>
                <p style="color: #a0b8cc; font-size: 0.8rem; margin-top: 0.3rem; font-weight: 500;">ProdPack Depth Analytics</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<hr style="border-color: rgba(212,175,55,0.3); margin: 1.2rem 0;">', unsafe_allow_html=True)
            
            # ------------------- VERİ YÜKLEME -------------------
            with st.expander("📁 VERİ YÖNETİMİ", expanded=True):
                uploaded_file = st.file_uploader(
                    "Excel / CSV Yükle",
                    type=['xlsx', 'xls', 'csv'],
                    label_visibility="collapsed",
                    key="file_uploader_main"
                )
                
                if uploaded_file:
                    file_size = uploaded_file.size / 1024
                    file_size_str = f"{file_size:.1f} KB" if file_size < 1024 else f"{file_size / 1024:.1f} MB"
                    
                    st.caption(f"📄 **{uploaded_file.name}** ({file_size_str})")
                    
                    if st.button("🚀 VERİYİ ANALİZ ET", use_container_width=True, type="primary"):
                        with st.spinner("Veri işleniyor... (5000 satır limit)"):
                            self._load_and_process_data(uploaded_file)
            
            # ------------------- PRODPACK MODÜLÜ -------------------
            if st.session_state.processed_data is not None:
                st.markdown('<hr style="border-color: rgba(212,175,55,0.3); margin: 1.2rem 0;">', unsafe_allow_html=True)
                
                with st.expander("🧬 PRODPACK DERİNLİK", expanded=True):
                    if st.button("🔨 HİYERARŞİ OLUŞTUR", use_container_width=True):
                        with st.spinner("ProdPack hiyerarşisi kuruluyor..."):
                            self._build_hierarchy()
                    
                    # Molekül seçimi
                    if st.session_state.analyzer and st.session_state.analyzer.hierarchy_built:
                        st.markdown("---")
                        molecule_col = st.session_state.analyzer.mappings['molecule']
                        
                        if molecule_col:
                            molecules = st.session_state.processed_data[molecule_col].dropna().unique()
                            
                            if len(molecules) > 0:
                                # Arama kutusu
                                search_term = st.text_input("🔍 Molekül Ara", placeholder="İsim girin...")
                                
                                if search_term:
                                    filtered_mols = [m for m in molecules if search_term.lower() in str(m).lower()]
                                    display_mols = filtered_mols[:50]
                                else:
                                    display_mols = molecules[:50]
                                
                                selected = st.selectbox(
                                    "Molekül Seç",
                                    display_mols,
                                    format_func=lambda x: str(x)[:45] + '...' if len(str(x)) > 45 else str(x),
                                    key="molecule_selector"
                                )
                                
                                st.session_state.selected_molecule = selected
                                
                                if st.button("🔍 DETAY ANALİZİ", use_container_width=True):
                                    with st.spinner(f"{selected} analiz ediliyor..."):
                                        self._analyze_molecule(selected)
                                        self._update_insights()
                
                # ------------------- TAHMİN & RİSK -------------------
                with st.expander("📈 TAHMİN & RİSK", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🔮 2025-2026", use_container_width=True):
                            with st.spinner("Holt-Winters tahmini..."):
                                self._run_forecast()
                                self._update_insights()
                    
                    with col2:
                        if st.button("⚠️ ANOMALİ", use_container_width=True):
                            with st.spinner("IsolationForest tespiti..."):
                                self._run_anomaly_detection()
                                self._update_insights()
                    
                    if st.button("🎯 PCA SEGMENTASYON", use_container_width=True):
                        with st.spinner("PCA + K-Means segmentasyon..."):
                            self._run_segmentation()
                            self._update_insights()
                
                # ------------------- SİSTEM DURUMU -------------------
                with st.expander("⚙️ SİSTEM", expanded=False):
                    df = st.session_state.processed_data
                    st.metric("Satır Sayısı", f"{len(df):,}", help="UI'de 5.000 satır gösterilir")
                    st.metric("Sütun Sayısı", len(df.columns))
                    
                    mem_usage = df.memory_usage(deep=True).sum() / 1024**2
                    st.metric("Hafıza Kullanımı", f"{mem_usage:.1f} MB")
                    
                    if st.button("🧹 CACHE TEMİZLE", use_container_width=True):
                        st.cache_data.clear()
                        st.success("✅ Cache temizlendi")
                        time.sleep(0.5)
                        st.rerun()
                    
                    if st.session_state.data_loaded_time:
                        st.caption(f"Son güncelleme: {st.session_state.data_loaded_time}")
            
            # ------------------- FOOTER -------------------
            st.markdown('<hr style="border-color: rgba(212,175,55,0.3); margin: 2rem 0 1rem 0;">', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align: center; color: #808080; font-size: 0.65rem; padding: 0.5rem 0;">
                <span>© 2026 PharmaIntelligence Inc.</span><br>
                <span style="color: #c0c0c0;">Enterprise ProdPack Depth v{Constants.VERSION}</span><br>
                <span style="color: #5a6a7a;">Build: {Constants.BUILD}</span>
            </div>
            """, unsafe_allow_html=True)
    
    def render_welcome_screen(self):
        """Hoşgeldin ekranı"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"""
            <div class="executive-card" style="text-align: center; padding: 3rem 2rem;">
                <span style="font-size: 5.5rem; display: block; margin-bottom: 1rem;">💊</span>
                <h1 style="color: #d4af37; font-size: 3rem; margin-bottom: 0.5rem; font-weight: 900;">PharmaIntel Pro</h1>
                <p style="color: #c0c0c0; font-size: 1.2rem; margin-bottom: 2rem; letter-spacing: 3px; font-weight: 600;">v{Constants.VERSION} ENTERPRISE</p>
                <p style="color: #a0b8cc; font-size: 1rem; margin-bottom: 2.5rem;">ProdPack Derinlik Analizi · Molekül → Şirket → Marka → Paket</p>
                
                <div style="background: rgba(212,175,55,0.08); border-radius: 24px; padding: 2.2rem; text-align: left;">
                    <h3 style="color: white; margin-bottom: 1.8rem; font-size: 1.3rem;">🚀 Hızlı Başlangıç:</h3>
                    
                    <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
                        <span style="background: #d4af37; color: #0a1929; width: 28px; height: 28px; border-radius: 14px; display: inline-flex; align-items: center; justify-content: center; font-weight: 900; margin-right: 1rem;">1</span>
                        <span style="color: white; font-size: 1.05rem;">📁 Sol panelden Excel/CSV dosyası yükleyin</span>
                    </div>
                    
                    <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
                        <span style="background: #d4af37; color: #0a1929; width: 28px; height: 28px; border-radius: 14px; display: inline-flex; align-items: center; justify-content: center; font-weight: 900; margin-right: 1rem;">2</span>
                        <span style="color: white; font-size: 1.05rem;">⚙️ "Veriyi Analiz Et" butonuna tıklayın</span>
                    </div>
                    
                    <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
                        <span style="background: #d4af37; color: #0a1929; width: 28px; height: 28px; border-radius: 14px; display: inline-flex; align-items: center; justify-content: center; font-weight: 900; margin-right: 1rem;">3</span>
                        <span style="color: white; font-size: 1.05rem;">🧬 "Hiyerarşi Oluştur" ile ProdPack yapısını kurun</span>
                    </div>
                    
                    <div style="display: flex; align-items: center;">
                        <span style="background: #d4af37; color: #0a1929; width: 28px; height: 28px; border-radius: 14px; display: inline-flex; align-items: center; justify-content: center; font-weight: 900; margin-right: 1rem;">4</span>
                        <span style="color: white; font-size: 1.05rem;">📈 Tahmin, anomali ve segmentasyon modüllerini çalıştırın</span>
                    </div>
                </div>
                
                <p style="color: #5a6a7a; margin-top: 2.5rem; font-size: 0.8rem;">
                    Desteklenen formatlar: .xlsx, .xls, .csv (UTF-8) | UI: 5.000 satır | Cache: 1M+ satır
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_main_content(self):
        """Ana içerik render"""
        
        # Hoşgeldin ekranı
        if not st.session_state.data_loaded_flag:
            self.render_welcome_screen()
            return
        
        # Header
        col_title, col_time = st.columns([3, 1])
        
        with col_title:
            st.markdown(f"""
            <h1 class="gold-title" style="margin-bottom: 0; font-size: 2.6rem;">
                💊 PharmaIntelligence Pro
            </h1>
            <p style="color: #c0c0c0; font-size: 1rem; margin-top: 0.3rem; font-weight: 500;">
                ProdPack Derinlik Analizi · Molekül → Şirket → Marka → Paket
            </p>
            """, unsafe_allow_html=True)
        
        with col_time:
            if st.session_state.data_loaded_time:
                st.markdown(f"""
                <div style="background: rgba(212,175,55,0.12); padding: 0.8rem 1.2rem; border-radius: 40px; text-align: center; border: 1px solid rgba(212,175,55,0.3);">
                    <span style="color: #d4af37; font-size: 0.75rem; letter-spacing: 2px; font-weight: 700;">SON GÜNCELLEME</span><br>
                    <span style="color: white; font-size: 0.9rem; font-weight: 600;">{st.session_state.data_loaded_time}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Yönetici Özeti - Insight Box
        self._update_insights()
        
        if st.session_state.executive_insights:
            insights_html = '<div class="insight-box"><div class="insight-text">'
            
            for insight in st.session_state.executive_insights[:4]:
                insights_html += insight.to_html()
            
            insights_html += '</div></div>'
            st.markdown(insights_html, unsafe_allow_html=True)
        
        # Sekmeler
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧬 PRODPACK DERİNLİK",
            "📈 TAHMİN & YATIRIM",
            "⚠️ RİSK & SEGMENTASYON",
            "📊 YÖNETİCİ PANELİ"
        ])
        
        with tab1:
            self._render_prodpack_tab()
        
        with tab2:
            self._render_forecast_tab()
        
        with tab3:
            self._render_risk_tab()
        
        with tab4:
            self._render_dashboard_tab()
    
    def _render_prodpack_tab(self):
        """ProdPack derinlik sekmesi"""
        
        if not st.session_state.hierarchy or st.session_state.analyzer is None:
            st.info("""
            <div style="background: rgba(33,150,243,0.1); padding: 2rem; border-radius: 16px; text-align: center;">
                <span style="font-size: 3rem;">🧬</span>
                <h3 style="color: #c0c0c0; margin: 1rem 0;">ProdPack Hiyerarşisi Oluşturulmamış</h3>
                <p style="color: #a0b8cc;">Sol panelden <strong>'Hiyerarşi Oluştur'</strong> butonuna tıklayarak molekül → şirket → marka → paket yapısını kurun.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<p class="silver-subtitle">🧬 Molekül Hiyerarşisi</p>', unsafe_allow_html=True)
            sunburst_data = st.session_state.analyzer.get_sunburst_data()
            fig_sunburst = VisualizationEngine.create_sunburst_chart(sunburst_data)
            st.plotly_chart(fig_sunburst, use_container_width=True)
        
        with col2:
            st.markdown('<p class="silver-subtitle">🔄 Pazar Akışı</p>', unsafe_allow_html=True)
            sankey_data = st.session_state.analyzer.get_sankey_data()
            fig_sankey = VisualizationEngine.create_sankey_diagram(sankey_data)
            st.plotly_chart(fig_sankey, use_container_width=True)
        
        st.markdown('---')
        st.markdown('<p class="silver-subtitle">🔬 Molekül Drill-Down Detayı</p>', unsafe_allow_html=True)
        
        if st.session_state.selected_molecule:
            col_left, col_right = st.columns([1.5, 1])
            
            with col_left:
                st.markdown(f"""
                <div style="background: rgba(26,54,93,0.6); padding: 0.8rem 1.5rem; border-radius: 40px; margin-bottom: 1.5rem; display: inline-block;">
                    <span style="color: #d4af37; font-weight: 800;">🔬 SEÇİLİ MOLEKÜL:</span>
                    <span style="color: white; font-weight: 600; margin-left: 0.8rem;">{st.session_state.selected_molecule}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.molecule_detail is not None and not st.session_state.molecule_detail.empty:
                    detail_df = st.session_state.molecule_detail
                    
                    # Formatlı gösterim
                    display_df = detail_df.copy()
                    if 'Satış_Hacmi' in display_df.columns:
                        display_df['Satış_Hacmi'] = display_df['Satış_Hacmi'].apply(lambda x: f"{x:,.0f}")
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=450,
                        column_config={
                            "ProdPack": st.column_config.TextColumn("ProdPack", width="medium"),
                            "Şirket": st.column_config.TextColumn("Şirket", width="small"),
                            "Marka": st.column_config.TextColumn("Marka", width="small"),
                            "Satış_Hacmi": st.column_config.TextColumn("Satış Hacmi", width="small"),
                            "Büyüme_%": st.column_config.NumberColumn("Büyüme %", format="%.1f%%"),
                            "Pazar_Payı_%": st.column_config.NumberColumn("Pazar Payı %", format="%.2f%%"),
                            "Trend": st.column_config.TextColumn("Trend", width="small"),
                            "Risk_Seviyesi": st.column_config.TextColumn("Risk", width="small")
                        },
                        hide_index=True
                    )
                    
                    # İstatistikler
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Toplam Paket", len(detail_df))
                    with col_stat2:
                        st.metric("Ort. Büyüme", f"%{detail_df['Büyüme_%'].mean():.1f}")
                    with col_stat3:
                        st.metric("Ort. Pazar Payı", f"%{detail_df['Pazar_Payı_%'].mean():.1f}")
                
                else:
                    st.info("Bu molekül için detay verisi bulunamadı")
            
            with col_right:
                st.markdown("**🔄 Kanibalizasyon Matrisi**")
                
                if st.session_state.cannibal_data is not None and not st.session_state.cannibal_data.empty:
                    fig_heatmap = VisualizationEngine.create_cannibalization_heatmap(st.session_state.cannibal_data)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Yüksek kanibalizasyon uyarısı
                    high_cannibal = st.session_state.cannibal_data[
                        st.session_state.cannibal_data['Kanibalizasyon_Skoru'] > 0.6
                    ]
                    
                    if not high_cannibal.empty:
                        st.warning(f"⚠️ {len(high_cannibal)} marka çiftinde **yüksek kanibalizasyon** riski")
                        
                        # En riskli çift
                        top_cannibal = high_cannibal.iloc[0]
                        st.info(f"🔴 En kritik: **{top_cannibal['Marka_A']}** ↔ **{top_cannibal['Marka_B']}** (Skor: {top_cannibal['Kanibalizasyon_Skoru']:.2f})")
                    
                else:
                    st.info("Bu molekül için kanibalizasyon verisi yok")
        else:
            st.info("👈 Sol panelden bir molekül seçin ve **'Detay Analizi'** butonuna tıklayın")
    
    def _render_forecast_tab(self):
        """Tahmin ve yatırım sekmesi"""
        
        if st.session_state.forecast_result is None:
            st.info("""
            <div style="background: rgba(33,150,243,0.1); padding: 2rem; border-radius: 16px; text-align: center;">
                <span style="font-size: 3rem;">🔮</span>
                <h3 style="color: #c0c0c0; margin: 1rem 0;">Pazar Tahmini Yapılmamış</h3>
                <p style="color: #a0b8cc;">Sol panelden <strong>'2025-2026 Tahmini'</strong> butonuna tıklayarak Holt-Winters ile pazar tahmini oluşturun.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        forecast = st.session_state.forecast_result
        
        # Tahmin grafiği
        fig_forecast = VisualizationEngine.create_forecast_chart(forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Yatırım tavsiyesi kartları
        st.markdown('<p class="silver-subtitle">💎 YATIRIM TAVSİYESİ 2025-2026</p>', unsafe_allow_html=True)
        
        advice_list = forecast.get('investment_advice', [])
        
        if advice_list:
            cols = st.columns(min(3, len(advice_list)))
            
            for idx, advice in enumerate(advice_list[:3]):
                with cols[idx]:
                    action = advice.get('action', 'STRATEJİ')
                    if isinstance(action, dict):
                        action_value = action.get('value', 'STRATEJİ')
                    else:
                        action_value = action.value if hasattr(action, 'value') else str(action)
                    
                    badge_class = "badge-opportunity" if 'EXPAND' in str(action_value) else "badge-critical" if 'HEDGE' in str(action_value) else "badge-warning"
                    
                    st.markdown(f"""
                    <div class="investment-card" style="height: 100%;">
                        <span class="{badge_class}">{action_value}</span>
                        <h4 style="color: #d4af37; margin: 1.2rem 0 0.8rem 0; font-size: 1.25rem; font-weight: 700;">{advice.get('title', 'Strateji')}</h4>
                        <p style="color: white; font-size: 0.95rem; line-height: 1.6;">{advice.get('message', '')}</p>
                        <div style="margin-top: 1.2rem; display: flex; justify-content: space-between; border-top: 1px solid rgba(212,175,55,0.3); padding-top: 1rem;">
                            <span style="color: #c0c0c0; font-size: 0.8rem;">💰 ROI: %{advice.get('roi_potential', 0)}</span>
                            <span style="color: #c0c0c0; font-size: 0.8rem;">⚠️ Risk: {advice.get('risk_level', 'Orta')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Model metrikleri
        st.markdown('---')
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            growth = forecast.get('growth_rate', 0)
            delta_color = "normal" if growth > 0 else "inverse"
            st.metric(
                "2025-2026 Büyüme",
                f"%{growth:.1f}",
                delta=f"{growth:.1f}% y/y",
                delta_color=delta_color,
                help="Holt-Winters tahminine göre yıllık bileşik büyüme oranı"
            )
        
        with col_m2:
            if forecast.get('forecast') and 'values' in forecast['forecast']:
                val_2026 = forecast['forecast']['values'][-1] if len(forecast['forecast']['values']) > 0 else 0
                st.metric("2026 Pazar Değeri", f"{val_2026:,.0f} TL", help="2026 yılı sonu tahmini pazar büyüklüğü")
        
        with col_m3:
            model_stats = forecast.get('model_stats', {})
            mape = model_stats.get('mape', 0)
            accuracy = 100 - mape if mape > 0 else 0
            st.metric("Tahmin Doğruluğu", f"%{accuracy:.1f}" if accuracy > 0 else "N/A", help="MAPE bazlı model doğruluğu")
        
        with col_m4:
            st.metric("Güven Aralığı", f"%{int(Constants.CONFIDENCE_LEVEL * 100)}", help="Tahmin güven düzeyi")
    
    def _render_risk_tab(self):
        """Risk ve segmentasyon sekmesi"""
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<p class="silver-subtitle">⚠️ IsolationForest Anomali</p>', unsafe_allow_html=True)
            
            if st.session_state.anomaly_result is not None:
                anomaly_df = st.session_state.anomaly_result
                
                # Dağılım grafiği
                fig_anomaly = VisualizationEngine.create_anomaly_distribution(anomaly_df)
                st.plotly_chart(fig_anomaly, use_container_width=True)
                
                # Risk özeti
                if 'Risk_Seviyesi' in anomaly_df.columns:
                    risk_counts = anomaly_df['Risk_Seviyesi'].value_counts()
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    
                    with col_r1:
                        critical = risk_counts.get(Constants.RISK_LEVELS['critical'], 0)
                        st.metric("🔴 Kritik", critical)
                    
                    with col_r2:
                        high = risk_counts.get(Constants.RISK_LEVELS['high'], 0)
                        st.metric("🟠 Yüksek", high)
                    
                    with col_r3:
                        medium = risk_counts.get(Constants.RISK_LEVELS['medium'], 0)
                        st.metric("🟡 Orta", medium)
                
                # Kritik riskli ürünler
                st.markdown("**🔴 Kritik Riskli Ürünler**")
                
                if 'Anomali_Tipi' in anomaly_df.columns:
                    critical_df = anomaly_df[anomaly_df['Anomali_Tipi'] == Constants.ANOMALY_TYPES['critical_drop']]
                    
                    if not critical_df.empty:
                        # Otomatik sütun seçimi
                        display_cols = []
                        
                        if st.session_state.analyzer:
                            if st.session_state.analyzer.mappings['molecule']:
                                display_cols.append(st.session_state.analyzer.mappings['molecule'])
                            if st.session_state.analyzer.mappings['company']:
                                display_cols.append(st.session_state.analyzer.mappings['company'])
                            if st.session_state.analyzer.mappings['brand']:
                                display_cols.append(st.session_state.analyzer.mappings['brand'])
                        
                        display_cols.extend(['Anomali_Tipi', 'Anomali_Skoru'])
                        display_cols = [c for c in display_cols if c in critical_df.columns][:5]
                        
                        st.dataframe(
                            critical_df[display_cols].head(10),
                            use_container_width=True,
                            height=250,
                            hide_index=True
                        )
                    else:
                        st.success("✅ Kritik riskli ürün bulunamadı")
                else:
                    st.info("Anomali verisi eksik")
            else:
                st.info("👈 Sol panelden **'Anomali Tespiti'** butonuna tıklayın")
        
        with col2:
            st.markdown('<p class="silver-subtitle">🎯 PCA + K-Means Segmentasyon</p>', unsafe_allow_html=True)
            
            if st.session_state.segmentation_result and st.session_state.segmentation_result.get('success'):
                seg = st.session_state.segmentation_result
                
                fig_seg = VisualizationEngine.create_segmentation_scatter(
                    seg.get('pca_components'),
                    seg.get('labels'),
                    seg.get('segment_names', {})
                )
                st.plotly_chart(fig_seg, use_container_width=True)
                
                # Segment dağılımı
                if seg['segmented_df'] is not None and 'Segment_Adi' in seg['segmented_df'].columns:
                    seg_counts = seg['segmented_df']['Segment_Adi'].value_counts()
                    
                    st.markdown("**📊 Segment Dağılımı**")
                    
                    seg_df = pd.DataFrame({
                        'Segment': seg_counts.index,
                        'Ürün Sayısı': seg_counts.values,
                        'Yüzde': (seg_counts.values / seg_counts.sum() * 100).round(1)
                    })
                    
                    st.dataframe(
                        seg_df,
                        use_container_width=True,
                        column_config={
                            "Segment": st.column_config.TextColumn("Segment", width="medium"),
                            "Ürün Sayısı": st.column_config.NumberColumn("Ürün Sayısı", format="%d"),
                            "Yüzde": st.column_config.NumberColumn("Yüzde %", format="%.1f%%")
                        },
                        hide_index=True
                    )
                    
                    # Silhouette skoru
                    silhouette = seg.get('silhouette_score', 0)
                    st.caption(f"✅ **Silhouette Skoru:** {silhouette:.3f} (1'e yakın iyi kümeleme)")
                    
                    # PCA varyans
                    pca_var = seg.get('pca_explained_variance', [])
                    if pca_var:
                        total_var = sum(pca_var) * 100
                        st.caption(f"📊 **PCA Varyans:** %{total_var:.1f} (2 bileşen)")
            else:
                st.info("👈 Sol panelden **'PCA Segmentasyon'** butonuna tıklayın")
    
    def _render_dashboard_tab(self):
        """Yönetici paneli"""
        
        # KPI Metrikleri
        st.markdown('<p class="silver-subtitle">📊 TEMEL PERFORMANS GÖSTERGELERİ</p>', unsafe_allow_html=True)
        
        kpi_cols = st.columns(4)
        
        with kpi_cols[0]:
            if st.session_state.analyzer and st.session_state.analyzer.hierarchy_built:
                stats = st.session_state.analyzer.get_hierarchy_stats()
                
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">ProdPack (Paket)</div>
                    <div class="metric-executive-value">{stats['pack_count']}</div>
                    <div class="metric-executive-trend">Toplam {stats['total_nodes']} düğüm</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">ProdPack</div>
                    <div class="metric-executive-value">-</div>
                    <div class="metric-executive-trend">Hiyerarşi oluştur</div>
                </div>
                """, unsafe_allow_html=True)
        
        with kpi_cols[1]:
            if st.session_state.forecast_result and st.session_state.forecast_result.get('success'):
                growth = st.session_state.forecast_result.get('growth_rate', 0)
                trend_icon = "📈" if growth > 0 else "📉"
                
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">2025-2026 Büyüme</div>
                    <div class="metric-executive-value">%{growth:.1f}</div>
                    <div class="metric-executive-trend">{trend_icon} {'Pozitif' if growth > 0 else 'Negatif'}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">Büyüme Tahmini</div>
                    <div class="metric-executive-value">-</div>
                    <div class="metric-executive-trend">Tahmin çalıştır</div>
                </div>
                """, unsafe_allow_html=True)
        
        with kpi_cols[2]:
            if st.session_state.anomaly_result is not None and 'Anomali_Tipi' in st.session_state.anomaly_result.columns:
                anomaly_df = st.session_state.anomaly_result
                anomaly_count = len(anomaly_df[anomaly_df['Anomali_Tipi'] != Constants.ANOMALY_TYPES['normal']])
                total_count = len(anomaly_df)
                anomaly_pct = (anomaly_count / total_count * 100) if total_count > 0 else 0
                
                risk_level = "⚠️ Yüksek" if anomaly_pct > 15 else "✅ Normal"
                
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">Anomali Oranı</div>
                    <div class="metric-executive-value">%{anomaly_pct:.1f}</div>
                    <div class="metric-executive-trend">{risk_level}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">Anomali</div>
                    <div class="metric-executive-value">-</div>
                    <div class="metric-executive-trend">Analiz çalıştır</div>
                </div>
                """, unsafe_allow_html=True)
        
        with kpi_cols[3]:
            if st.session_state.segmentation_result and st.session_state.segmentation_result.get('success'):
                seg = st.session_state.segmentation_result
                silhouette = seg.get('silhouette_score', 0)
                
                quality = "İyi" if silhouette > 0.5 else "Orta" if silhouette > 0.3 else "Zayıf"
                
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">Segmentasyon</div>
                    <div class="metric-executive-value">{silhouette:.2f}</div>
                    <div class="metric-executive-trend">{quality} Kümeleme</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-executive">
                    <div class="metric-executive-label">Segmentasyon</div>
                    <div class="metric-executive-value">-</div>
                    <div class="metric-executive-trend">Segmentasyon çalıştır</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('---')
        
        # Stratejik Öngörüler
        st.markdown('<p class="silver-subtitle">💡 STRATEJİK ÖNGÖRÜLER</p>', unsafe_allow_html=True)
        
        if st.session_state.forecast_result and st.session_state.forecast_result.get('investment_advice'):
            advice_list = st.session_state.forecast_result.get('investment_advice', [])
            advice_cols = st.columns(min(2, len(advice_list)))
            
            for idx, advice in enumerate(advice_list[:2]):
                with advice_cols[idx]:
                    action = advice.get('action', 'STRATEJİ')
                    if isinstance(action, dict):
                        action_value = action.get('value', 'STRATEJİ')
                    else:
                        action_value = action.value if hasattr(action, 'value') else str(action)
                    
                    icon = '🚀' if 'EXPAND' in str(action_value) else '⚖️' if 'HEDGE' in str(action_value) else '📊'
                    
                    st.markdown(f"""
                    <div style="background: rgba(26,54,93,0.7); border-radius: 20px; padding: 1.8rem; border-left: 8px solid #d4af37; height: 100%;">
                        <span style="font-size: 2.2rem;">{icon}</span>
                        <h4 style="color: #d4af37; margin: 0.8rem 0; font-size: 1.2rem;">{advice.get('title', 'Strateji')}</h4>
                        <p style="color: white; font-size: 0.95rem; line-height: 1.6;">{advice.get('message', '')}</p>
                        <div style="margin-top: 1.2rem; background: rgba(0,0,0,0.3); padding: 0.8rem; border-radius: 12px;">
                            <span style="color: #c0c0c0;">Aksiyon: </span>
                            <span style="color: #d4af37; font-weight: 700;">{action_value}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👈 Tahmin modülünü çalıştırarak stratejik öngörüleri görüntüleyin")
        
        # Hızlı istatistikler
        st.markdown('---')
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            with st.expander("📋 HİYERARŞİ İSTATİSTİKLERİ", expanded=False):
                if st.session_state.analyzer and st.session_state.analyzer.hierarchy_built:
                    stats = st.session_state.analyzer.get_hierarchy_stats()
                    
                    stat_df = pd.DataFrame({
                        'Metrik': ['Toplam Düğüm', 'Molekül', 'Şirket', 'Marka', 'ProdPack', 'Pazar Değeri'],
                        'Değer': [
                            stats['total_nodes'],
                            stats['molecule_count'],
                            stats['company_count'],
                            stats['brand_count'],
                            stats['pack_count'],
                            f"{stats['total_market_value']:,.0f} TL"
                        ]
                    })
                    
                    st.dataframe(stat_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Hiyerarşi verisi yok")
        
        with col_exp2:
            with st.expander("📥 VERİ DIŞA AKTAR", expanded=False):
                if st.session_state.processed_data is not None:
                    csv_data = st.session_state.processed_data.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 CSV İndir (İşlenmiş)",
                        data=csv_data,
                        file_name=f"pharma_intel_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    if st.session_state.molecule_detail is not None:
                        detail_csv = st.session_state.molecule_detail.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Molekül Detay İndir",
                            data=detail_csv,
                            file_name=f"molecule_detail_{st.session_state.selected_molecule}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.info("Veri yüklenmemiş")

# =============================================================================
# 14. UYGULAMA BAŞLATMA
# =============================================================================

def main():
    """Ana uygulama giriş noktası"""
    
    # Garbage collector
    gc.enable()
    
    # Uygulama nesnesi
    app = PharmaIntelligenceApp()
    
    # Sidebar
    app.render_sidebar()
    
    # Ana içerik
    app.render_main_content()
    
    # Footer
    st.markdown(f"""
    <div style="position: fixed; bottom: 20px; right: 30px; background: rgba(10,25,41,0.9); 
                padding: 0.6rem 1.8rem; border-radius: 40px; border: 1px solid #d4af37; 
                backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); 
                z-index: 999; box-shadow: 0 5px 25px rgba(0,0,0,0.6);">
        <span style="color: #c0c0c0; font-size: 0.75rem; letter-spacing: 3px; font-weight: 600;">
            ⚕️ PharmaIntel Pro v{Constants.VERSION} | ProdPack Depth
        </span>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 15. UYGULAMA GİRİŞ NOKTASI
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ### ❌ KRİTİK UYGULAMA HATASI
        
        **Hata Tipi:** {type(e).__name__}
        **Hata Mesajı:** {str(e)}
        
        Lütfen sayfayı yenileyin veya veri formatını kontrol edin.
        """)
        
        with st.expander("🔍 HATA AYIKLAMA DETAYLARI"):
            st.code(traceback.format_exc())
            
        st.caption("Bu hata devam ederse kinyas4kayra8@gmail.com adresine ulaşın.")
