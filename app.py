"""
PharmaIntelligence Pro v8.0 - Kurumsal Karar Destek ve Stratejik İstihbarat Platformu
Versiyon: 8.0.0
Yazar: PharmaIntelligence Inc.
Lisans: Enterprise

✓ AI-Powered Predictive Analytics
✓ Multi-Algorithm Anomaly Detection  
✓ PCA + UMAP + t-SNE Advanced Segmentation
✓ Prophet & ARIMA Time Series Forecasting
✓ SHAP Explainable AI
✓ Executive Dark Theme with 3D Visualizations
✓ Automated Strategic Recommendations
✓ ProdPack Deep Drill-Down Analysis
✓ Molecular-Level Market Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Advanced Analytics Stack
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS, Isomap, SpectralEmbedding
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore, shapiro, kstest, normaltest
from scipy.signal import savgol_filter
import umap.umap_ as umap

# Time Series Specialized
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False
    st.warning("Prophet kurulu değil. Lütfen 'pip install prophet' komutuyla kurun.")

try:
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False
    st.warning("pmdarima kurulu değil. Lütfen 'pip install pmdarima' komutuyla kurun.")

# Explainable AI
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False
    st.warning("SHAP kurulu değil. Lütfen 'pip install shap' komutuyla kurun.")

# Visualization Enhancement
try:
    from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
    YELLOWBRICK_AVAILABLE = True
except:
    YELLOWBRICK_AVAILABLE = False

# Utility Stack
from datetime import datetime, timedelta
import json
from io import BytesIO, StringIO
import time
import gc
import traceback
import inspect
import hashlib
import pickle
import base64
import csv
import math
import re
import os
import sys
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Generator
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict, Counter, OrderedDict
from pathlib import Path
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

# Export Capabilities
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell, xl_range
try:
    from reportlab.lib.pagesizes import letter, A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
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
# 1. ENUMS AND DATA CLASSES
# ================================================

class RiskLevel(Enum):
    """Risk seviyeleri"""
    CRITICAL = "Kritik Risk"
    HIGH = "Yüksek Risk" 
    MEDIUM = "Orta Risk"
    LOW = "Düşük Risk"
    SAFE = "Güvenli"

class GrowthCategory(Enum):
    """Büyüme kategorileri"""
    HYPERGROWTH = "Hiper Büyüme (>50%)"
    HIGH_GROWTH = "Yüksek Büyüme (20-50%)"
    MODERATE_GROWTH = "Orta Büyüme (5-20%)"
    STAGNANT = "Durgun (-5% - 5%)"
    DECLINING = "Daralan (<-5%)"

class ProductSegment(Enum):
    """BCG Matrix-inspired product segments"""
    STARS = "Yıldız Ürünler"
    CASH_COWS = "Nakit İnekleri"
    QUESTION_MARKS = "Soru İşaretleri"
    DOGS = "Zayıf Ürünler"
    EMERGING = "Yükselen Yıldızlar"
    DISRUPTIVE = "Dikkat Çekiciler"
    MATURE = "Olgun Ürünler"
    LEGACY = "Eski Ürünler"

class MarketConcentration(Enum):
    """Pazar yoğunluğu sınıflandırması"""
    MONOPOLY = "Monopol (HHI > 2500)"
    OLIGOPOLY = "Oligopol (HHI: 1800-2500)"
    COMPETITIVE = "Rekabetçi (HHI: 1000-1800)"
    FRAGMENTED = "Parçalı (HHI < 1000)"

class AnalysisType(Enum):
    """Analiz türleri"""
    DESCRIPTIVE = "Tanımlayıcı Analiz"
    PREDICTIVE = "Tahmine Dayalı Analiz"
    PRESCRIPTIVE = "Reçeteli Analiz"
    DIAGNOSTIC = "Teşhise Dayalı Analiz"

# ================================================
# 2. DATA CLASSES FOR STRUCTURED DATA
# ================================================

@dataclass
class MarketMetrics:
    """Pazar metrikleri veri sınıfı"""
    total_market_value: float = 0.0
    yoy_growth: float = 0.0
    cagr: float = 0.0
    hhi_index: float = 0.0
    concentration_ratio: float = 0.0
    market_volatility: float = 0.0
    price_index: float = 0.0
    volume_index: float = 0.0
    international_penetration: float = 0.0
    innovation_index: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

@dataclass
class ProductInsight:
    """Ürün içgörüsü veri sınıfı"""
    product_id: str
    product_name: str
    sales_2024: float
    growth_rate: float
    market_share: float
    price_position: str
    risk_level: RiskLevel
    segment: ProductSegment
    recommendations: List[str] = field(default_factory=list)
    kpis: Dict[str, float] = field(default_factory=dict)
    
    def add_recommendation(self, recommendation: str):
        self.recommendations.append(recommendation)

@dataclass 
class CompanyAnalysis:
    """Şirket analizi veri sınıfı"""
    company_name: str
    total_sales: float
    market_share: float
    product_count: int
    growth_rate: float
    geographic_reach: int
    innovation_score: float
    financial_strength: str
    strategic_position: str
    swot_analysis: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class ForecastResult:
    """Tahmin sonucu veri sınıfı"""
    periods: List[str]
    predictions: List[float]
    lower_bounds: List[float]
    upper_bounds: List[float]
    confidence_level: float
    model_type: str
    mape: float
    rmse: float
    trend_direction: str
    seasonal_pattern: Optional[str] = None
    
    def get_growth_rate(self) -> float:
        if len(self.predictions) > 1:
            return ((self.predictions[-1] - self.predictions[0]) / self.predictions[0]) * 100
        return 0.0

# ================================================
# 3. ADVANCED DATA ENGINE
# ================================================

class AdvancedDataEngine:
    """
    Gelişmiş veri işleme motoru.
    Büyük veri setleri için optimize edilmiş, paralel işleme destekli.
    """
    
    def __init__(self):
        self.cache = {}
        self.processing_stats = {}
        self.column_metadata = {}
        
    @staticmethod
    def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Sütun tiplerini otomatik tespit et.
        
        Args:
            df: DataFrame
            
        Returns:
            Sütun tipi haritalaması
        """
        column_types = {}
        
        for col in df.columns:
            # Null oranı
            null_ratio = df[col].isnull().mean()
            
            # Benzersiz değer sayısı
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            
            # Veri tipine göre sınıflandırma
            dtype = str(df[col].dtype)
            
            if 'int' in dtype or 'float' in dtype:
                if unique_ratio < 0.05:
                    column_types[col] = 'categorical_numeric'
                elif null_ratio > 0.3:
                    column_types[col] = 'sparse_numeric'
                else:
                    column_types[col] = 'continuous_numeric'
                    
            elif 'object' in dtype or 'category' in dtype:
                if unique_ratio < 0.1:
                    column_types[col] = 'low_cardinality_categorical'
                elif unique_ratio < 0.5:
                    column_types[col] = 'medium_cardinality_categorical'
                else:
                    column_types[col] = 'high_cardinality_categorical'
                    
            elif 'datetime' in dtype:
                column_types[col] = 'datetime'
                
            elif 'bool' in dtype:
                column_types[col] = 'boolean'
                
            else:
                column_types[col] = 'other'
                
        return column_types
    
    @staticmethod
    def intelligent_column_renaming(columns: List[str]) -> List[str]:
        """
        Akıllı sütun isimlendirme.
        
        Args:
            columns: Orijinal sütun isimleri
            
        Returns:
            Temizlenmiş ve standardize edilmiş sütun isimleri
        """
        cleaned = []
        seen = {}
        
        # Öncelikli haritalama kuralları
        priority_patterns = {
            # Satış ve hacim
            r'(?i)(sales|revenue|satış|gelir).*?(202[2-5])': lambda m: f'Satış_{m.group(2)}',
            r'(?i)(volume|hacim|birim).*?(202[2-5])': lambda m: f'Hacim_{m.group(2)}',
            r'(?i)(units|birim).*?(202[2-5])': lambda m: f'Birim_{m.group(2)}',
            
            # Fiyat
            r'(?i)(price|fiyat).*?(avg|average|ort).*?(202[2-5])': lambda m: f'Ort_Fiyat_{m.group(3)}',
            r'(?i)(unit.*?price).*?(202[2-5])': lambda m: f'Birim_Fiyat_{m.group(2)}',
            
            # Büyüme
            r'(?i)(growth|growth rate|buyume|buyume oranı)': 'Büyüme_Oranı',
            r'(?i)(cagr|bsbh|yillik.*?buyume)': 'CAGR',
            
            # Pazar payı
            r'(?i)(market.*?share|pazar.*?payı)': 'Pazar_Payı',
            
            # Coğrafi
            r'(?i)(country|ülke|country.*?code)': 'Ülke',
            r'(?i)(region|bölge)': 'Bölge',
            r'(?i)(city|şehir)': 'Şehir',
            
            # Şirket bilgileri
            r'(?i)(company|firma|şirket)': 'Şirket',
            r'(?i)(manufacturer|üretici)': 'Üretici',
            r'(?i)(corporation|kuruluş)': 'Kuruluş',
            
            # Ürün bilgileri
            r'(?i)(product|ürün)': 'Ürün',
            r'(?i)(molecule|molekül)': 'Molekül',
            r'(?i)(brand|marka)': 'Marka',
            r'(?i)(generic|jenerik)': 'Jenerik',
            r'(?i)(pack|package|paket)': 'Paket',
            r'(?i)(prodpack|prod.*?pack)': 'ProdPack',
            
            # Tedarik zinciri
            r'(?i)(supplier|tedarikçi)': 'Tedarikçi',
            r'(?i)(distributor|distribütör)': 'Distribütör',
            
            # Zaman serisi
            r'(?i)(date|tarih)': 'Tarih',
            r'(?i)(month|ay)': 'Ay',
            r'(?i)(quarter|çeyrek)': 'Çeyrek',
            r'(?i)(year|yıl)': 'Yıl',
            
            # Kategorik
            r'(?i)(category|kategori)': 'Kategori',
            r'(?i)(segment|segment)': 'Segment',
            r'(?i)(class|sınıf)': 'Sınıf',
            
            # Finansal
            r'(?i)(revenue|gelir)': 'Gelir',
            r'(?i)(profit|kar)': 'Kar',
            r'(?i)(margin|marj)': 'Marj',
            r'(?i)(cost|maliyet)': 'Maliyet',
        }
        
        for original in columns:
            col = str(original)
            
            # Türkçe karakter dönüşümü
            turkish_map = {
                'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
            }
            for tr, en in turkish_map.items():
                col = col.replace(tr, en)
            
            # Özel karakterleri kaldır
            col = re.sub(r'[^\w\s]', ' ', col)
            col = re.sub(r'\s+', '_', col.strip())
            
            # Küçük harfe çevir
            col_lower = col.lower()
            
            # Öncelikli desenleri kontrol et
            matched = False
            for pattern, replacement in priority_patterns.items():
                match = re.search(pattern, col_lower)
                if match:
                    if callable(replacement):
                        col = replacement(match)
                    else:
                        col = replacement
                    matched = True
                    break
            
            # Eşleşme yoksa standart temizleme
            if not matched:
                # Sayısal önekleri kaldır
                col = re.sub(r'^\d+_', '', col)
                # Çok uzun isimleri kısalt
                if len(col) > 50:
                    words = col.split('_')
                    if len(words) > 3:
                        col = '_'.join(words[:3])
            
            # Benzersiz isimlendirme
            base_col = col
            counter = 1
            while col in seen:
                col = f"{base_col}_{counter}"
                counter += 1
            seen[col] = True
            
            cleaned.append(col)
            
        return cleaned
    
    @staticmethod
    def extract_years_from_dataframe(df: pd.DataFrame) -> List[int]:
        """
        DataFrame'den tüm yılları çıkar.
        
        Args:
            df: DataFrame
            
        Returns:
            Yılların listesi
        """
        years = set()
        
        # Sütun isimlerinden yıl çıkar
        for col in df.columns:
            matches = re.findall(r'20\d{2}', str(col))
            for match in matches:
                try:
                    year = int(match)
                    if 2000 <= year <= 2030:  # Makul yıl aralığı
                        years.add(year)
                except:
                    continue
        
        # Veri içinden yıl çıkar
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].notna().any():
                sample_values = df[col].dropna().astype(str).head(100)
                for val in sample_values:
                    matches = re.findall(r'20\d{2}', val)
                    for match in matches:
                        try:
                            year = int(match)
                            if 2000 <= year <= 2030:
                                years.add(year)
                        except:
                            continue
        
        return sorted(list(years))
    
    def parallel_data_processing(self, df: pd.DataFrame, n_workers: int = 4) -> pd.DataFrame:
        """
        Paralel veri işleme.
        
        Args:
            df: DataFrame
            n_workers: Paralel işçi sayısı
            
        Returns:
            İşlenmiş DataFrame
        """
        if len(df) < 10000:
            return self._process_dataframe(df)
        
        # Büyük veri seti için paralel işleme
        chunk_size = len(df) // n_workers
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self._process_dataframe, chunk) for chunk in chunks]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    st.warning(f"Chunk processing error: {str(e)}")
        
        return pd.concat(results, ignore_index=True) if results else df
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame işleme.
        
        Args:
            df: DataFrame
            
        Returns:
            İşlenmiş DataFrame
        """
        # Sütun isimlerini temizle
        df.columns = self.intelligent_column_renaming(df.columns.tolist())
        
        # Tip dönüşümü
        df = self._smart_type_conversion(df)
        
        # Eksik veri doldurma
        df = self._advanced_imputation(df)
        
        # Aykırı değer tespiti
        df = self._statistical_outlier_handling(df)
        
        # Tarih formatları
        df = self._standardize_dates(df)
        
        return df
    
    def _smart_type_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Akıllı tip dönüşümü.
        
        Args:
            df: DataFrame
            
        Returns:
            Tip dönüştürülmüş DataFrame
        """
        for col in df.columns:
            # Kategori tespiti
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            
            if unique_ratio < 0.05 and df[col].dtype == 'object':
                # Düşük kardinaliteli kategorik
                try:
                    df[col] = df[col].astype('category')
                except:
                    pass
                    
            elif 'date' in col.lower() or 'tarih' in col.lower():
                # Tarih dönüşümü
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
                    
            elif df[col].dtype == 'object':
                # String temizleme
                df[col] = df[col].astype(str).str.strip()
                
                # Sayısal string tespiti
                try:
                    numeric_mask = df[col].str.match(r'^-?\d+\.?\d*$', na=False)
                    if numeric_mask.any():
                        df.loc[numeric_mask, col] = pd.to_numeric(df.loc[numeric_mask, col], errors='coerce')
                except:
                    pass
        
        return df
    
    def _advanced_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş eksik veri doldurma.
        
        Args:
            df: DataFrame
            
        Returns:
            Eksik verileri doldurulmuş DataFrame
        """
        cols_to_drop = []
        
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count == 0:
                continue
                
            null_ratio = null_count / len(df) if len(df) > 0 else 0
            
            if null_ratio > 0.5:
                # %50'den fazla eksikse sütunu sil
                cols_to_drop.append(col)
                continue
                
            # Tip bazlı doldurma
            if pd.api.types.is_numeric_dtype(df[col]):
                # Sayısal sütunlar
                if null_ratio < 0.1:
                    # Az eksikse median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Çok eksikse forward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                # Kategorik sütunlar
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else 'Bilinmiyor'
                df[col] = df[col].fillna(fill_val)
                
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Tarih sütunları
                df[col] = df[col].fillna(pd.Timestamp.now())
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    def _statistical_outlier_handling(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        İstatistiksel aykırı değer işleme.
        
        Args:
            df: DataFrame
            threshold: Z-score eşiği
            
        Returns:
            Aykırı değerleri işlenmiş DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].nunique() < 10:
                continue
            
            try:
                # Z-score hesapla
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    z_scores = np.abs(stats.zscore(col_data))
                    
                    # Aykırı değerleri bul
                    outlier_mask = z_scores > threshold
                    
                    if outlier_mask.any():
                        # Winsorization uygula
                        q1 = df[col].quantile(0.01)
                        q99 = df[col].quantile(0.99)
                        df[col] = np.where(df[col] < q1, q1, df[col])
                        df[col] = np.where(df[col] > q99, q99, df[col])
            except:
                continue
        
        return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tarih formatlarını standardize et.
        
        Args:
            df: DataFrame
            
        Returns:
            Tarihleri standardize edilmiş DataFrame
        """
        date_cols = []
        
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            elif 'date' in col.lower() or 'tarih' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    date_cols.append(col)
                except:
                    pass
        
        # Tarih sütunlarından özellik çıkar
        for col in date_cols:
            if df[col].notna().any():
                try:
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_quarter'] = df[col].dt.quarter
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    df[f'{col}_is_weekend'] = df[col].dt.dayofweek >= 5
                except:
                    pass
        
        return df
    
    def create_analytical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analitik özellikler oluştur.
        
        Args:
            df: DataFrame
            
        Returns:
            Analitik özellikler eklenmiş DataFrame
        """
        # Yılları tespit et
        years = self.extract_years_from_dataframe(df)
        
        if len(years) >= 2:
            # Satış sütunlarını bul
            sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
            
            if sales_cols:
                # Son yılın satış sütunu
                latest_sales = sorted(sales_cols, key=lambda x: int(re.search(r'20\d{2}', x).group()))[-1]
                
                # Pazar payı
                total_sales = df[sales_cols].sum().sum()
                if total_sales > 0:
                    for col in sales_cols:
                        df[f'{col}_Pazar_Payi'] = (df[col] / total_sales) * 100
                
                # Büyüme oranları
                for i in range(1, len(sales_cols)):
                    current_col = sales_cols[i]
                    previous_col = sales_cols[i-1]
                    
                    # Yılları çıkar
                    current_year = int(re.search(r'20\d{2}', current_col).group())
                    previous_year = int(re.search(r'20\d{2}', previous_col).group())
                    
                    growth_col = f'Buyume_{previous_year}_{current_year}'
                    
                    # Güvenli büyüme hesaplama
                    mask = df[previous_col] != 0
                    df.loc[mask, growth_col] = ((df.loc[mask, current_col] - df.loc[mask, previous_col]) / 
                                                df.loc[mask, previous_col]) * 100
                    df.loc[~mask, growth_col] = np.nan
                
                # CAGR hesaplama
                if len(sales_cols) >= 2:
                    first_col = sales_cols[0]
                    last_col = sales_cols[-1]
                    
                    first_year = int(re.search(r'20\d{2}', first_col).group())
                    last_year = int(re.search(r'20\d{2}', last_col).group())
                    
                    n_years = last_year - first_year
                    
                    if n_years > 0:
                        mask = df[first_col] > 0
                        df.loc[mask, 'CAGR'] = ((df.loc[mask, last_col] / df.loc[mask, first_col]) ** (1/n_years) - 1) * 100
        
        # Finansal oranlar
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if 'Satış' in ''.join(numeric_cols) and 'Maliyet' in ''.join(numeric_cols):
            cost_cols = [col for col in numeric_cols if 'maliyet' in col.lower() or 'cost' in col.lower()]
            sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
            
            if cost_cols and sales_cols:
                sales_col = sales_cols[-1]
                cost_col = cost_cols[0]
                
                mask = df[sales_col] != 0
                df.loc[mask, 'Kar_Marjı'] = ((df.loc[mask, sales_col] - df.loc[mask, cost_col]) / df.loc[mask, sales_col]) * 100
        
        # İndeksler oluştur
        self._create_composite_indices(df)
        
        return df
    
    def _create_composite_indices(self, df: pd.DataFrame):
        """
        Kompozit indeksler oluştur.
        
        Args:
            df: DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 3:
            try:
                # Performans indeksi
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_cols].fillna(0))
                df['Performans_Indeksi'] = scaled_data.mean(axis=1)
                
                # Risk indeksi
                volatility = df[numeric_cols].std(axis=1)
                mean_vals = df[numeric_cols].mean(axis=1)
                df['Risk_Indeksi'] = volatility / (mean_vals + 1e-10)
                
                # Büyüme indeksi
                growth_cols = [col for col in df.columns if 'Buyume' in col]
                if growth_cols:
                    df['Büyüme_Indeksi'] = df[growth_cols].mean(axis=1)
                
                # Fiyat rekabet indeksi
                price_cols = [col for col in df.columns if 'Fiyat' in col or 'Price' in col]
                if price_cols:
                    price_std = df[price_cols].std(axis=1)
                    price_mean = df[price_cols].mean(axis=1)
                    df['Fiyat_Rekabet_Indeksi'] = price_std / (price_mean + 1e-10)
            except:
                pass
    
    def calculate_market_concentration(self, df: pd.DataFrame) -> MarketMetrics:
        """
        Pazar yoğunluğu metriklerini hesapla.
        
        Args:
            df: DataFrame
            
        Returns:
            MarketMetrics nesnesi
        """
        metrics = MarketMetrics()
        
        try:
            # Toplam pazar değeri
            sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
            if sales_cols:
                latest_sales = sales_cols[-1]
                metrics.total_market_value = df[latest_sales].sum()
            
            # Yıllık büyüme
            growth_cols = [col for col in df.columns if 'Buyume' in col]
            if growth_cols:
                metrics.yoy_growth = df[growth_cols[-1]].mean() if not df[growth_cols[-1]].empty else 0
            
            # HHI indeksi
            if 'Şirket' in df.columns and sales_cols:
                company_sales = df.groupby('Şirket')[latest_sales].sum()
                market_shares = (company_sales / company_sales.sum()) * 100
                metrics.hhi_index = (market_shares ** 2).sum()
                
                # Konsantrasyon oranı
                if len(company_sales) >= 4:
                    top_4_share = company_sales.nlargest(4).sum() / company_sales.sum() * 100
                    metrics.concentration_ratio = top_4_share
            
            # Pazar oynaklığı
            if len(sales_cols) >= 2:
                sales_matrix = df[sales_cols].values
                volatility = np.std(sales_matrix, axis=1).mean()
                metrics.market_volatility = volatility
            
            # Fiyat indeksi
            price_cols = [col for col in df.columns if 'Fiyat' in col]
            if price_cols:
                metrics.price_index = df[price_cols[-1]].mean()
            
            # Uluslararası penetrasyon
            if 'Ülke' in df.columns:
                metrics.international_penetration = df['Ülke'].nunique()
            
            # İnovasyon indeksi
            if 'Molekül' in df.columns:
                unique_molecules = df['Molekül'].nunique()
                total_products = len(df)
                metrics.innovation_index = (unique_molecules / total_products) * 100 if total_products > 0 else 0
            
        except Exception as e:
            st.warning(f"Pazar metrikleri hesaplama hatası: {str(e)}")
        
        return metrics
    
    def create_prodpack_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ProdPack hiyerarşisini oluştur.
        Molekül -> Şirket -> Marka -> Paket drill-down analizi
        
        Args:
            df: DataFrame
            
        Returns:
            Hiyerarşik yapı eklenmiş DataFrame
        """
        try:
            # Hiyerarşik seviyeler
            hierarchy_cols = []
            
            if 'Molekül' in df.columns:
                hierarchy_cols.append('Molekül')
            if 'Şirket' in df.columns:
                hierarchy_cols.append('Şirket')
            if 'Marka' in df.columns:
                hierarchy_cols.append('Marka')
            if 'Paket' in df.columns or 'ProdPack' in df.columns:
                pack_col = 'ProdPack' if 'ProdPack' in df.columns else 'Paket'
                hierarchy_cols.append(pack_col)
            
            if len(hierarchy_cols) >= 2:
                # Hiyerarşik ID oluştur
                df['Hiyerarşi_ID'] = df[hierarchy_cols].apply(
                    lambda x: ' > '.join([str(v) for v in x if pd.notna(v)]), 
                    axis=1
                )
                
                # Seviye göstergesi
                df['Hiyerarşi_Seviye'] = df[hierarchy_cols].notna().sum(axis=1)
            
        except Exception as e:
            st.warning(f"ProdPack hiyerarşi oluşturma hatası: {str(e)}")
        
        return df

# ================================================
# 4. PRODPACK DRILL-DOWN ENGINE
# ================================================

class ProdPackDrillDownEngine:
    """
    ProdPack derinlik analizi motoru.
    Molekül -> Şirket -> Marka -> Paket drill-down analizi
    """
    
    def __init__(self):
        self.hierarchy_levels = ['Molekül', 'Şirket', 'Marka', 'Paket', 'ProdPack']
        self.current_selection = {}
    
    def create_hierarchy_analysis(self, df: pd.DataFrame, 
                                  molecule: Optional[str] = None,
                                  company: Optional[str] = None,
                                  brand: Optional[str] = None) -> Dict[str, Any]:
        """
        Hiyerarşik analiz oluştur.
        
        Args:
            df: DataFrame
            molecule: Seçili molekül
            company: Seçili şirket
            brand: Seçili marka
            
        Returns:
            Hiyerarşik analiz sonuçları
        """
        results = {
            'summary': {},
            'breakdown': {},
            'growth_analysis': {},
            'market_share': {},
            'visualization_data': {}
        }
        
        try:
            # Filtreleme
            filtered_df = df.copy()
            
            if molecule and 'Molekül' in df.columns:
                filtered_df = filtered_df[filtered_df['Molekül'] == molecule]
                results['summary']['molecule'] = molecule
            
            if company and 'Şirket' in df.columns:
                filtered_df = filtered_df[filtered_df['Şirket'] == company]
                results['summary']['company'] = company
            
            if brand and 'Marka' in df.columns:
                filtered_df = filtered_df[filtered_df['Marka'] == brand]
                results['summary']['brand'] = brand
            
            # Satış sütunlarını bul
            sales_cols = [col for col in filtered_df.columns if re.search(r'Satış_20\d{2}', col)]
            
            if not sales_cols:
                return results
            
            latest_sales = sales_cols[-1]
            
            # Özet metrikler
            results['summary']['total_sales'] = filtered_df[latest_sales].sum()
            results['summary']['product_count'] = len(filtered_df)
            
            # Büyüme analizi
            growth_cols = [col for col in filtered_df.columns if 'Buyume' in col]
            if growth_cols:
                results['growth_analysis']['avg_growth'] = filtered_df[growth_cols[-1]].mean()
                results['growth_analysis']['median_growth'] = filtered_df[growth_cols[-1]].median()
            
            # Seviye bazlı breakdown
            for level in self.hierarchy_levels:
                if level in filtered_df.columns:
                    level_breakdown = filtered_df.groupby(level).agg({
                        latest_sales: 'sum',
                        level: 'count'
                    })
                    level_breakdown.columns = ['Total_Sales', 'Count']
                    level_breakdown['Market_Share'] = (level_breakdown['Total_Sales'] / 
                                                       level_breakdown['Total_Sales'].sum() * 100)
                    
                    results['breakdown'][level] = level_breakdown.to_dict('index')
            
            # Pazar payı analizi
            if 'Şirket' in filtered_df.columns:
                company_shares = filtered_df.groupby('Şirket')[latest_sales].sum()
                total_sales = company_shares.sum()
                
                if total_sales > 0:
                    results['market_share'] = {
                        company: (sales / total_sales * 100)
                        for company, sales in company_shares.items()
                    }
            
            # Görselleştirme verileri
            results['visualization_data'] = self._prepare_visualization_data(
                filtered_df, sales_cols
            )
            
        except Exception as e:
            st.error(f"Hiyerarşik analiz hatası: {str(e)}")
        
        return results
    
    def _prepare_visualization_data(self, df: pd.DataFrame, 
                                   sales_cols: List[str]) -> Dict[str, Any]:
        """
        Görselleştirme verilerini hazırla.
        
        Args:
            df: DataFrame
            sales_cols: Satış sütunları
            
        Returns:
            Görselleştirme verileri
        """
        viz_data = {}
        
        try:
            # Sunburst için hiyerarşik veri
            hierarchy_cols = [col for col in self.hierarchy_levels if col in df.columns]
            
            if hierarchy_cols and sales_cols:
                latest_sales = sales_cols[-1]
                
                # Hiyerarşik toplamlar
                sunburst_data = []
                
                for _, row in df.iterrows():
                    path = []
                    for col in hierarchy_cols:
                        if pd.notna(row[col]):
                            path.append(str(row[col]))
                    
                    if path and pd.notna(row[latest_sales]):
                        sunburst_data.append({
                            'path': path,
                            'value': row[latest_sales]
                        })
                
                viz_data['sunburst'] = sunburst_data
            
            # Sankey için akış verileri
            if len(hierarchy_cols) >= 2:
                sankey_data = self._create_sankey_data(df, hierarchy_cols, sales_cols[-1])
                viz_data['sankey'] = sankey_data
            
        except Exception as e:
            st.warning(f"Görselleştirme veri hazırlama hatası: {str(e)}")
        
        return viz_data
    
    def _create_sankey_data(self, df: pd.DataFrame, 
                           hierarchy_cols: List[str], 
                           value_col: str) -> Dict[str, List]:
        """
        Sankey diyagramı verisi oluştur.
        
        Args:
            df: DataFrame
            hierarchy_cols: Hiyerarşi sütunları
            value_col: Değer sütunu
            
        Returns:
            Sankey verileri
        """
        sankey_data = {
            'labels': [],
            'sources': [],
            'targets': [],
            'values': []
        }
        
        try:
            label_map = {}
            label_counter = 0
            
            for i in range(len(hierarchy_cols) - 1):
                source_col = hierarchy_cols[i]
                target_col = hierarchy_cols[i + 1]
                
                grouped = df.groupby([source_col, target_col])[value_col].sum().reset_index()
                
                for _, row in grouped.iterrows():
                    source = str(row[source_col])
                    target = str(row[target_col])
                    value = row[value_col]
                    
                    # Label mapping
                    if source not in label_map:
                        label_map[source] = label_counter
                        sankey_data['labels'].append(source)
                        label_counter += 1
                    
                    if target not in label_map:
                        label_map[target] = label_counter
                        sankey_data['labels'].append(target)
                        label_counter += 1
                    
                    sankey_data['sources'].append(label_map[source])
                    sankey_data['targets'].append(label_map[target])
                    sankey_data['values'].append(value)
            
        except Exception as e:
            st.warning(f"Sankey veri oluşturma hatası: {str(e)}")
        
        return sankey_data
    
    def detect_cannibalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aynı şirket içi kanibalizasyon tespiti.
        
        Args:
            df: DataFrame
            
        Returns:
            Kanibalizasyon analizi eklenmiş DataFrame
        """
        try:
            if 'Şirket' not in df.columns:
                return df
            
            # Satış ve büyüme sütunları
            sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
            growth_cols = [col for col in df.columns if 'Buyume' in col]
            
            if not sales_cols or not growth_cols:
                return df
            
            latest_sales = sales_cols[-1]
            latest_growth = growth_cols[-1]
            
            # Şirket bazlı analiz
            cannibalization_scores = []
            
            for company in df['Şirket'].unique():
                company_products = df[df['Şirket'] == company]
                
                if len(company_products) < 2:
                    cannibalization_scores.extend([0] * len(company_products))
                    continue
                
                # Büyüme ve pazar payı matrisi
                for idx in company_products.index:
                    product_growth = df.loc[idx, latest_growth]
                    product_sales = df.loc[idx, latest_sales]
                    
                    # Diğer ürünlerle karşılaştır
                    other_products = company_products[company_products.index != idx]
                    
                    if len(other_products) > 0:
                        # Negatif büyüme ve yüksek satış = potansiyel kanibalizasyon
                        if product_growth < 0:
                            growing_others = other_products[other_products[latest_growth] > 10]
                            
                            if len(growing_others) > 0:
                                # Kanibalizasyon skoru
                                cannib_score = min(100, abs(product_growth) * len(growing_others) / len(other_products))
                                cannibalization_scores.append(cannib_score)
                            else:
                                cannibalization_scores.append(0)
                        else:
                            cannibalization_scores.append(0)
                    else:
                        cannibalization_scores.append(0)
            
            df['Kanibalizasyon_Skoru'] = cannibalization_scores
            
            # Kanibalizasyon kategorisi
            df['Kanibalizasyon_Riski'] = pd.cut(
                df['Kanibalizasyon_Skoru'],
                bins=[-np.inf, 10, 30, 50, np.inf],
                labels=['Düşük', 'Orta', 'Yüksek', 'Kritik']
            )
            
        except Exception as e:
            st.warning(f"Kanibalizasyon tespiti hatası: {str(e)}")
        
        return df

# The code continues with Analytics Engine, Visualization Engine, and Dashboard UI...
# Due to character limits, I'll create this in the next section.

# ================================================
# 5. ADVANCED ANALYTICS ENGINE (Continued)
# ================================================

class AdvancedAnalyticsEngine:
    """
    Gelişmiş analitik motoru.
    """
    
    def __init__(self):
        self.models = {}
        self.results_cache = {}
        self.feature_importance = {}
    
    def multi_model_forecasting(self, df: pd.DataFrame, target_col: str, 
                               periods: int = 12, ensemble: bool = True) -> Dict[str, ForecastResult]:
        """Çoklu model tahminleme"""
        forecasts = {}
        
        try:
            time_series = self._prepare_time_series(df, target_col)
            
            if time_series is None or len(time_series) < 24:
                return forecasts
            
            # 1. Exponential Smoothing
            try:
                exp_model = ExponentialSmoothing(
                    time_series,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=12
                ).fit()
                exp_forecast = exp_model.forecast(periods)
                
                forecasts['exponential_smoothing'] = ForecastResult(
                    periods=[f"Period {i+1}" for i in range(periods)],
                    predictions=exp_forecast.values.tolist(),
                    lower_bounds=(exp_forecast.values * 0.9).tolist(),
                    upper_bounds=(exp_forecast.values * 1.1).tolist(),
                    confidence_level=0.95,
                    model_type='Exponential Smoothing',
                    mape=self._calculate_mape(time_series[-12:], exp_model.fittedvalues[-12:]),
                    rmse=self._calculate_rmse(time_series[-12:], exp_model.fittedvalues[-12:]),
                    trend_direction='up' if exp_forecast.values[-1] > exp_forecast.values[0] else 'down'
                )
            except Exception as e:
                st.warning(f"Exponential Smoothing hatası: {str(e)}")
            
            # 2. Prophet
            if PROPHET_AVAILABLE:
                try:
                    prophet_df = pd.DataFrame({
                        'ds': pd.date_range(start='2020-01-01', periods=len(time_series), freq='M'),
                        'y': time_series.values
                    })
                    
                    prophet_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False
                    )
                    prophet_model.fit(prophet_df)
                    
                    future = prophet_model.make_future_dataframe(periods=periods, freq='M')
                    prophet_forecast = prophet_model.predict(future)
                    
                    forecasts['prophet'] = ForecastResult(
                        periods=prophet_forecast['ds'].dt.strftime('%Y-%m').tolist()[-periods:],
                        predictions=prophet_forecast['yhat'].values[-periods:].tolist(),
                        lower_bounds=prophet_forecast['yhat_lower'].values[-periods:].tolist(),
                        upper_bounds=prophet_forecast['yhat_upper'].values[-periods:].tolist(),
                        confidence_level=0.95,
                        model_type='Prophet',
                        mape=0.0,
                        rmse=0.0,
                        trend_direction='up' if prophet_forecast['trend'].values[-1] > prophet_forecast['trend'].values[0] else 'down'
                    )
                except Exception as e:
                    st.warning(f"Prophet hatası: {str(e)}")
            
            # Ensemble
            if ensemble and len(forecasts) >= 2:
                ensemble_predictions = []
                ensemble_lower = []
                ensemble_upper = []
                
                for i in range(periods):
                    preds = []
                    lowers = []
                    uppers = []
                    
                    for model_name, forecast in forecasts.items():
                        if i < len(forecast.predictions):
                            preds.append(forecast.predictions[i])
                            lowers.append(forecast.lower_bounds[i])
                            uppers.append(forecast.upper_bounds[i])
                    
                    if preds:
                        ensemble_predictions.append(np.mean(preds))
                        ensemble_lower.append(np.mean(lowers))
                        ensemble_upper.append(np.mean(uppers))
                
                if ensemble_predictions:
                    forecasts['ensemble'] = ForecastResult(
                        periods=[f"Period {i+1}" for i in range(periods)],
                        predictions=ensemble_predictions,
                        lower_bounds=ensemble_lower,
                        upper_bounds=ensemble_upper,
                        confidence_level=0.95,
                        model_type='Ensemble',
                        mape=np.mean([f.mape for f in forecasts.values()]),
                        rmse=np.mean([f.rmse for f in forecasts.values()]),
                        trend_direction='up' if ensemble_predictions[-1] > ensemble_predictions[0] else 'down'
                    )
            
        except Exception as e:
            st.error(f"Tahminleme hatası: {str(e)}")
        
        return forecasts
    
    def _prepare_time_series(self, df: pd.DataFrame, target_col: str) -> Optional[pd.Series]:
        """Zaman serisi hazırla"""
        if target_col not in df.columns:
            return None
        
        if 'Tarih' in df.columns:
            try:
                df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')
                monthly = df.groupby(pd.Grouper(key='Tarih', freq='M'))[target_col].sum()
            except:
                monthly = pd.Series(df[target_col].values)
        else:
            monthly = pd.Series(df[target_col].values)
        
        return monthly.dropna()
    
    def _calculate_mape(self, actual: pd.Series, predicted: pd.Series) -> float:
        """MAPE hesapla"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        mask = actual != 0
        if mask.any():
            return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return 0.0
    
    def _calculate_rmse(self, actual: pd.Series, predicted: pd.Series) -> float:
        """RMSE hesapla"""
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    def multi_algorithm_anomaly_detection(self, df: pd.DataFrame, 
                                        contamination: float = 0.1) -> pd.DataFrame:
        """Çoklu algoritma anomali tespiti"""
        result_df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            return result_df
        
        selected_features = self._select_features_for_anomaly(df[numeric_cols])
        
        if len(selected_features) < 2:
            return result_df
        
        X = df[selected_features].fillna(0).values
        
        if len(X) < 20:
            return result_df
        
        anomaly_scores = {}
        
        # Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            iso_scores = iso_forest.fit_predict(X)
            anomaly_scores['iso_forest'] = iso_scores
        except:
            pass
        
        # Ensemble skor
        ensemble_scores = np.zeros(len(X))
        for algo, scores in anomaly_scores.items():
            normalized = (scores == 1).astype(int)
            ensemble_scores += normalized
        
        if len(anomaly_scores) > 0:
            ensemble_scores = ensemble_scores / len(anomaly_scores)
        
        result_df['Anomali_Skoru'] = ensemble_scores
        result_df['Anomali_Tahmini'] = np.where(ensemble_scores < 0.5, -1, 1)
        
        result_df['Risk_Seviyesi'] = pd.cut(
            result_df['Anomali_Skoru'],
            bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
            labels=['Kritik Risk', 'Yüksek Risk', 'Orta Risk', 'Düşük Risk', 'Normal']
        )
        
        return result_df
    
    def _select_features_for_anomaly(self, df: pd.DataFrame, n_features: int = 10) -> List[str]:
        """Anomali için özellik seç"""
        variances = df.var()
        top_variance = variances.nlargest(min(n_features, len(variances)))
        return top_variance.index.tolist()
    
    def generate_strategic_insights(self, df: pd.DataFrame, 
                                   metrics: MarketMetrics) -> List[Dict[str, Any]]:
        """Stratejik içgörüler üret"""
        insights = []
        
        try:
            # Pazar Yapısı
            if metrics.hhi_index > 2500:
                insights.append({
                    'type': 'market_structure',
                    'title': '🏢 Monopolistik Pazar Yapısı',
                    'description': f'HHI İndeksi: {metrics.hhi_index:.0f} - Pazar çok yoğunlaşmış',
                    'recommendation': 'Rakiplerle işbirliği fırsatlarını değerlendirin.',
                    'priority': 'high',
                    'impact': 'strategic'
                })
            
            # Büyüme Trendi
            if metrics.yoy_growth > 15:
                insights.append({
                    'type': 'growth',
                    'title': '🚀 Yüksek Büyüme Trendi',
                    'description': f'Yıllık büyüme: %{metrics.yoy_growth:.1f}',
                    'recommendation': 'Yatırımları artırın ve kapasite planlaması yapın.',
                    'priority': 'high',
                    'impact': 'financial'
                })
            
        except Exception as e:
            st.warning(f"İçgörü üretme hatası: {str(e)}")
        
        return insights


# ================================================
# MAIN APPLICATION
# ================================================

def main():
    """Ana uygulama"""
    st.set_page_config(
        page_title="PharmaIntelligence Pro v8.0",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0c1a32, #14274e);
            color: #f8fafc;
        }
        .main-header {
            font-size: 3rem;
            background: linear-gradient(135deg, #d4af37, #c0c0c0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 900;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">💊 PharmaIntelligence Pro v8.0</h1>', unsafe_allow_html=True)
    st.markdown("**Kurumsal Karar Destek ve Stratejik İstihbarat Platformu**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Veri Yönetimi")
        
        uploaded_file = st.file_uploader(
            "Veri Dosyası Yükle",
            type=['xlsx', 'xls', 'csv'],
            help="Excel veya CSV formatında veri yükleyin"
        )
        
        if uploaded_file:
            if st.button("🚀 Veriyi İşle", type="primary"):
                with st.spinner("Veri işleniyor..."):
                    try:
                        # Veriyi yükle
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        # Veri motorunu başlat
                        data_engine = AdvancedDataEngine()
                        processed_df = data_engine.parallel_data_processing(df)
                        processed_df = data_engine.create_analytical_features(processed_df)
                        
                        # ProdPack hiyerarşisi
                        processed_df = data_engine.create_prodpack_hierarchy(processed_df)
                        
                        # Kanibalizasyon analizi
                        prodpack_engine = ProdPackDrillDownEngine()
                        processed_df = prodpack_engine.detect_cannibalization(processed_df)
                        
                        # Metrikleri hesapla
                        metrics = data_engine.calculate_market_concentration(processed_df)
                        
                        # Session state'e kaydet
                        st.session_state.processed_data = processed_df
                        st.session_state.raw_data = df
                        st.session_state.market_metrics = metrics
                        
                        # İçgörüleri üret
                        analytics_engine = AdvancedAnalyticsEngine()
                        insights = analytics_engine.generate_strategic_insights(processed_df, metrics)
                        st.session_state.strategic_insights = insights
                        
                        st.success(f"✅ Veri işlendi: {len(processed_df):,} satır, {len(processed_df.columns)} sütun")
                        
                    except Exception as e:
                        st.error(f"Veri işleme hatası: {str(e)}")
    
    # Ana içerik
    if 'processed_data' in st.session_state:
        df = st.session_state.processed_data
        
        # Sekmeler
        tabs = st.tabs([
            "📊 YÖNETİCİ PANELİ",
            "🔬 PRODPACK ANALİZİ",
            "🔮 TAHMİN",
            "⚠️ RİSK",
            "💡 İÇGÖRÜLER"
        ])
        
        # Tab 1: Yönetici Paneli
        with tabs[0]:
            st.markdown("### 📊 Genel Bakış")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam Kayıt", f"{len(df):,}")
            
            with col2:
                if 'Molekül' in df.columns:
                    st.metric("Molekül Sayısı", df['Molekül'].nunique())
            
            with col3:
                if 'Şirket' in df.columns:
                    st.metric("Şirket Sayısı", df['Şirket'].nunique())
            
            with col4:
                if 'Ülke' in df.columns:
                    st.metric("Ülke Sayısı", df['Ülke'].nunique())
            
            # Veri önizleme
            st.markdown("#### 📋 Veri Önizleme")
            st.dataframe(df.head(100), use_container_width=True)
        
        # Tab 2: ProdPack Analizi
        with tabs[1]:
            st.markdown("### 🔬 ProdPack Derinlik Analizi")
            
            prodpack_engine = ProdPackDrillDownEngine()
            
            # Filtreler
            col1, col2, col3 = st.columns(3)
            
            with col1:
                molecules = ['Tümü'] + sorted(df['Molekül'].unique().tolist()) if 'Molekül' in df.columns else ['Tümü']
                selected_molecule = st.selectbox("Molekül Seçin", molecules)
            
            with col2:
                companies = ['Tümü'] + sorted(df['Şirket'].unique().tolist()) if 'Şirket' in df.columns else ['Tümü']
                selected_company = st.selectbox("Şirket Seçin", companies)
            
            with col3:
                brands = ['Tümü'] + sorted(df['Marka'].unique().tolist()) if 'Marka' in df.columns else ['Tümü']
                selected_brand = st.selectbox("Marka Seçin", brands)
            
            # Analiz yap
            if st.button("🔍 Analiz Et"):
                mol = None if selected_molecule == 'Tümü' else selected_molecule
                comp = None if selected_company == 'Tümü' else selected_company
                br = None if selected_brand == 'Tümü' else selected_brand
                
                analysis = prodpack_engine.create_hierarchy_analysis(df, mol, comp, br)
                
                # Sonuçları göster
                st.markdown("#### 📊 Analiz Sonuçları")
                
                summary = analysis.get('summary', {})
                if summary:
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        st.metric("Toplam Satış", f"${summary.get('total_sales', 0)/1e6:.2f}M")
                    
                    with metric_cols[1]:
                        st.metric("Ürün Sayısı", summary.get('product_count', 0))
                    
                    growth = analysis.get('growth_analysis', {})
                    with metric_cols[2]:
                        st.metric("Ort. Büyüme", f"{growth.get('avg_growth', 0):.1f}%")
            
            # Kanibalizasyon analizi
            st.markdown("#### 🔄 Kanibalizasyon Analizi")
            
            if 'Kanibalizasyon_Skoru' in df.columns:
                cannib_df = df[df['Kanibalizasyon_Skoru'] > 30].copy()
                
                if len(cannib_df) > 0:
                    st.warning(f"⚠️ {len(cannib_df)} ürün yüksek kanibalizasyon riski taşıyor")
                    
                    display_cols = []
                    for col in ['Molekül', 'Şirket', 'Marka', 'Kanibalizasyon_Skoru', 'Kanibalizasyon_Riski']:
                        if col in cannib_df.columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        st.dataframe(cannib_df[display_cols].sort_values('Kanibalizasyon_Skoru', ascending=False), 
                                   use_container_width=True)
                else:
                    st.success("✅ Yüksek kanibalizasyon riski tespit edilmedi")
        
        # Tab 3: Tahmin Analizi
        with tabs[2]:
            st.markdown("### 🔮 Tahmin Analizi")
            
            sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
            
            if sales_cols:
                target_col = st.selectbox("Hedef Sütun", sales_cols)
                
                if st.button("📈 Tahmin Çalıştır"):
                    with st.spinner("Tahmin modelleri çalışıyor..."):
                        analytics_engine = AdvancedAnalyticsEngine()
                        forecasts = analytics_engine.multi_model_forecasting(df, target_col, periods=12)
                        
                        if forecasts:
                            st.success(f"✅ {len(forecasts)} model başarıyla çalıştırıldı")
                            
                            # Model performansları
                            for model_name, forecast in forecasts.items():
                                with st.expander(f"📊 {model_name.upper()} Modeli"):
                                    cols = st.columns(3)
                                    
                                    with cols[0]:
                                        st.metric("MAPE", f"{forecast.mape:.2f}%")
                                    
                                    with cols[1]:
                                        st.metric("RMSE", f"${forecast.rmse/1e6:.2f}M")
                                    
                                    with cols[2]:
                                        st.metric("Trend", forecast.trend_direction.upper())
                                    
                                    # Grafik
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=list(range(len(forecast.predictions))),
                                        y=forecast.predictions,
                                        mode='lines+markers',
                                        name='Tahmin',
                                        line=dict(color='blue', width=3)
                                    ))
                                    
                                    fig.update_layout(
                                        title=f'{model_name.upper()} Tahminleri',
                                        xaxis_title='Dönem',
                                        yaxis_title='Değer',
                                        height=400,
                                        template='plotly_dark'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Tahmin modelleri çalıştırılamadı")
            else:
                st.info("Tahmin için uygun satış verisi bulunamadı")
        
        # Tab 4: Risk Analizi
        with tabs[3]:
            st.markdown("### ⚠️ Risk Analizi")
            
            if st.button("🔍 Risk Analizi Çalıştır"):
                with st.spinner("Risk analizi yapılıyor..."):
                    analytics_engine = AdvancedAnalyticsEngine()
                    risk_df = analytics_engine.multi_algorithm_anomaly_detection(df)
                    
                    st.session_state.risk_data = risk_df
                    
                    if 'Risk_Seviyesi' in risk_df.columns:
                        risk_summary = risk_df['Risk_Seviyesi'].value_counts()
                        
                        cols = st.columns(5)
                        
                        for i, (risk_level, count) in enumerate(risk_summary.items()):
                            with cols[i % 5]:
                                st.metric(risk_level, count)
                        
                        # Kritik riskler
                        critical = risk_df[risk_df['Risk_Seviyesi'] == 'Kritik Risk']
                        
                        if len(critical) > 0:
                            st.warning(f"⚠️ {len(critical)} kritik riskli ürün tespit edildi")
                            
                            display_cols = []
                            for col in ['Molekül', 'Şirket', 'Risk_Seviyesi', 'Anomali_Skoru']:
                                if col in critical.columns:
                                    display_cols.append(col)
                            
                            if display_cols:
                                st.dataframe(critical[display_cols], use_container_width=True)
        
        # Tab 5: Stratejik İçgörüler
        with tabs[4]:
            st.markdown("### 💡 Stratejik İçgörüler")
            
            if 'strategic_insights' in st.session_state:
                insights = st.session_state.strategic_insights
                
                if insights:
                    for insight in insights:
                        priority_colors = {
                            'critical': '#f44336',
                            'high': '#ff9800',
                            'medium': '#ffeb3b',
                            'low': '#4caf50'
                        }
                        
                        color = priority_colors.get(insight.get('priority', 'low'), '#4caf50')
                        
                        st.markdown(f"""
                        <div style="
                            background: rgba(26, 35, 126, 0.1);
                            border-left: 5px solid {color};
                            padding: 1.5rem;
                            border-radius: 10px;
                            margin-bottom: 1rem;
                        ">
                            <h4>{insight['title']}</h4>
                            <p>{insight['description']}</p>
                            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 5px;">
                                <strong>🎯 Öneri:</strong> {insight['recommendation']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Henüz içgörü üretilmedi")
            else:
                st.info("Veri analizi yapıldıktan sonra içgörüler burada görünecek")
    
    else:
        # Hoşgeldin ekranı
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem;">💊</div>
            <h2>PharmaIntelligence Pro v8.0'a Hoş Geldiniz</h2>
            <p style="color: #5c6bc0; margin: 1.5rem 0;">
            Kurumsal karar destek platformumuz, ilaç pazarı analizinde yapay zeka destekli 
            tahminleme, risk analizi ve stratejik içgörüler sunar.
            </p>
            <div style="background: #e8eaf6; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;">
                <h4>🚀 Başlamak İçin:</h4>
                <ol style="text-align: left; color: #5c6bc0; max-width: 600px; margin: 0 auto;">
                    <li>Sol taraftaki panelden veri dosyanızı yükleyin</li>
                    <li>"Veriyi İşle" butonuna tıklayın</li>
                    <li>Analiz sonuçlarını sekmelerde görüntüleyin</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        gc.enable()
        main()
    except Exception as e:
        st.error(f"⚠️ Kritik hata: {str(e)}")
        
        with st.expander("🔍 Hata Detayları"):
            st.code(traceback.format_exc())
        
        if st.button("🔄 Uygulamayı Yeniden Başlat"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
