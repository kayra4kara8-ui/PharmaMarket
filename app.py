

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
import aiohttp

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
    st.warning("PDF dışa aktarımı devre dışı: reportlab kurulu değil.")

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
            unique_ratio = df[col].nunique() / len(df)
            
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
                    years.add(int(match))
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
                            years.add(int(match))
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
                results.append(future.result())
        
        return pd.concat(results, ignore_index=True)
    
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
            unique_ratio = df[col].nunique() / len(df)
            
            if unique_ratio < 0.05 and df[col].dtype == 'object':
                # Düşük kardinaliteli kategorik
                df[col] = df[col].astype('category')
                
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
                numeric_mask = df[col].str.match(r'^-?\d+\.?\d*$', na=False)
                if numeric_mask.any():
                    try:
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
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count == 0:
                continue
                
            null_ratio = null_count / len(df)
            
            if null_ratio > 0.5:
                # %50'den fazla eksikse sütunu sil
                df = df.drop(columns=[col])
                continue
                
            # Tip bazlı doldurma
            if pd.api.types.is_numeric_dtype(df[col]):
                # Sayısal sütunlar
                if null_ratio < 0.1:
                    # Az eksikse ortalama
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Çok eksikse forward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                # Kategorik sütunlar
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Bilinmiyor')
                
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Tarih sütunları
                df[col] = df[col].fillna(pd.Timestamp.now())
        
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
                
            # Z-score hesapla
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            
            # Aykırı değerleri bul
            outlier_mask = z_scores > threshold
            
            if outlier_mask.any():
                # Winsorization uygula
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = np.where(df[col] < q1, q1, df[col])
                df[col] = np.where(df[col] > q99, q99, df[col])
        
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
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_quarter'] = df[col].dt.quarter
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_is_weekend'] = df[col].dt.dayofweek >= 5
        
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
            
            if cost_cols and sales_cols:
                sales_col = sales_cols[-1]
                cost_col = cost_cols[0]
                
                df['Kar_Marjı'] = ((df[sales_col] - df[cost_col]) / df[sales_col]) * 100
        
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
            # Performans indeksi
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[numeric_cols].fillna(0))
            df['Performans_Indeksi'] = scaled_data.mean(axis=1)
            
            # Risk indeksi
            volatility = df[numeric_cols].std(axis=1)
            df['Risk_Indeksi'] = volatility / (df[numeric_cols].mean(axis=1) + 1e-10)
            
            # Büyüme indeksi
            growth_cols = [col for col in df.columns if 'Buyume' in col]
            if growth_cols:
                df['Büyüme_Indeksi'] = df[growth_cols].mean(axis=1)
            
            # Fiyat rekabet indeksi
            price_cols = [col for col in df.columns if 'Fiyat' in col or 'Price' in col]
            if price_cols:
                df['Fiyat_Rekabet_Indeksi'] = df[price_cols].std(axis=1) / df[price_cols].mean(axis=1)
    
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
                top_4_share = company_sares.nlargest(4).sum() / company_sales.sum() * 100
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
                metrics.innovation_index = (unique_molecules / total_products) * 100
            
        except Exception as e:
            st.warning(f"Pazar metrikleri hesaplama hatası: {str(e)}")
        
        return metrics

# ================================================
# 4. ADVANCED ANALYTICS ENGINE
# ================================================

class AdvancedAnalyticsEngine:
    """
    Gelişmiş analitik motoru.
    Çoklu algoritma, ensemble yöntemleri ve derin öğrenme destekli.
    """
    
    def __init__(self):
        self.models = {}
        self.results_cache = {}
        self.feature_importance = {}
        
    def multi_model_forecasting(self, df: pd.DataFrame, target_col: str, 
                               periods: int = 12, ensemble: bool = True) -> Dict[str, ForecastResult]:
        """
        Çoklu model tahminleme.
        
        Args:
            df: DataFrame
            target_col: Hedef sütun
            periods: Tahmin periyodu
            ensemble: Ensemble model kullan
            
        Returns:
            Model tahminleri sözlüğü
        """
        forecasts = {}
        
        try:
            # Zaman serisi hazırlığı
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
                    periods=list(range(1, periods + 1)),
                    predictions=exp_forecast.values,
                    lower_bounds=exp_forecast.values * 0.9,
                    upper_bounds=exp_forecast.values * 1.1,
                    confidence_level=0.95,
                    model_type='Exponential Smoothing',
                    mape=self._calculate_mape(time_series[-12:], exp_model.fittedvalues[-12:]),
                    rmse=self._calculate_rmse(time_series[-12:], exp_model.fittedvalues[-12:]),
                    trend_direction='up' if exp_forecast.values[-1] > exp_forecast.values[0] else 'down'
                )
            except:
                pass
            
            # 2. Prophet (Facebook)
            if PROPHET_AVAILABLE:
                try:
                    prophet_df = pd.DataFrame({
                        'ds': pd.date_range(start='2020-01-01', periods=len(time_series), freq='M'),
                        'y': time_series.values
                    })
                    
                    prophet_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.05
                    )
                    prophet_model.fit(prophet_df)
                    
                    future = prophet_model.make_future_dataframe(periods=periods, freq='M')
                    prophet_forecast = prophet_model.predict(future)
                    
                    forecasts['prophet'] = ForecastResult(
                        periods=prophet_forecast['ds'].dt.strftime('%Y-%m').tolist()[-periods:],
                        predictions=prophet_forecast['yhat'].values[-periods:],
                        lower_bounds=prophet_forecast['yhat_lower'].values[-periods:],
                        upper_bounds=prophet_forecast['yhat_upper'].values[-periods:],
                        confidence_level=0.95,
                        model_type='Prophet',
                        mape=0.0,  # Prophet kendi hesaplar
                        rmse=0.0,
                        trend_direction='up' if prophet_forecast['trend'].values[-1] > prophet_forecast['trend'].values[0] else 'down',
                        seasonal_pattern='yearly'
                    )
                except:
                    pass
            
            # 3. ARIMA
            if ARIMA_AVAILABLE:
                try:
                    arima_model = auto_arima(
                        time_series,
                        seasonal=True,
                        m=12,
                        trace=False,
                        error_action='ignore',
                        suppress_warnings=True
                    )
                    
                    arima_forecast, conf_int = arima_model.predict(
                        n_periods=periods,
                        return_conf_int=True
                    )
                    
                    forecasts['arima'] = ForecastResult(
                        periods=list(range(1, periods + 1)),
                        predictions=arima_forecast.values,
                        lower_bounds=conf_int[:, 0],
                        upper_bounds=conf_int[:, 1],
                        confidence_level=0.95,
                        model_type='ARIMA',
                        mape=self._calculate_mape(time_series[-12:], arima_model.predict_in_sample()[-12:]),
                        rmse=self._calculate_rmse(time_series[-12:], arima_model.predict_in_sample()[-12:]),
                        trend_direction='up' if arima_forecast.values[-1] > arima_forecast.values[0] else 'down'
                    )
                except:
                    pass
            
            # Ensemble tahmin
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
                        periods=list(range(1, periods + 1)),
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
        """
        Zaman serisi hazırla.
        
        Args:
            df: DataFrame
            target_col: Hedef sütun
            
        Returns:
            Zaman serisi
        """
        if target_col not in df.columns:
            return None
        
        # Aylık toplamları hesapla
        if 'Tarih' in df.columns:
            df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')
            monthly = df.groupby(pd.Grouper(key='Tarih', freq='M'))[target_col].sum()
        else:
            # Tarih yoksa sıralı zaman serisi
            monthly = pd.Series(df[target_col].values)
        
        return monthly.dropna()
    
    def _calculate_mape(self, actual: pd.Series, predicted: pd.Series) -> float:
        """
        MAPE hesapla.
        
        Args:
            actual: Gerçek değerler
            predicted: Tahmin edilen değerler
            
        Returns:
            MAPE değeri
        """
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        mask = actual != 0
        if mask.any():
            return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return 0.0
    
    def _calculate_rmse(self, actual: pd.Series, predicted: pd.Series) -> float:
        """
        RMSE hesapla.
        
        Args:
            actual: Gerçek değerler
            predicted: Tahmin edilen değerler
            
        Returns:
            RMSE değeri
        """
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    def multi_algorithm_anomaly_detection(self, df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
        """
        Çoklu algoritma ile anomali tespiti.
        
        Args:
            df: DataFrame
            contamination: Kontaminasyon oranı
            
        Returns:
            Anomali skorları eklenmiş DataFrame
        """
        result_df = df.copy()
        
        # Sayısal özellikleri seç
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            return result_df
        
        # Özellik seçimi
        selected_features = self._select_features_for_anomaly(df[numeric_cols])
        
        if len(selected_features) < 2:
            return result_df
        
        X = df[selected_features].fillna(0).values
        
        if len(X) < 20:
            return result_df
        
        # Çoklu algoritma skorları
        anomaly_scores = {}
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples='auto',
            bootstrap=False
        )
        iso_scores = iso_forest.fit_predict(X)
        anomaly_scores['iso_forest'] = iso_scores
        
        # 2. Local Outlier Factor
        try:
            lof = LocalOutlierFactor(
                contamination=contamination,
                novelty=False,
                n_neighbors=20
            )
            lof_scores = lof.fit_predict(X)
            anomaly_scores['lof'] = lof_scores
        except:
            pass
        
        # 3. One-Class SVM
        try:
            oc_svm = OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='auto'
            )
            svm_scores = oc_svm.fit_predict(X)
            anomaly_scores['svm'] = svm_scores
        except:
            pass
        
        # 4. Elliptic Envelope
        try:
            elliptic = EllipticEnvelope(
                contamination=contamination,
                random_state=42
            )
            elliptic_scores = elliptic.fit_predict(X)
            anomaly_scores['elliptic'] = elliptic_scores
        except:
            pass
        
        # Ensemble skor
        ensemble_scores = np.zeros(len(X))
        for algo, scores in anomaly_scores.items():
            # -1: outlier, 1: inlier -> 1: normal, 0: outlier'a çevir
            normalized = (scores == 1).astype(int)
            ensemble_scores += normalized
        
        # Normalize ensemble skor
        ensemble_scores = ensemble_scores / len(anomaly_scores)
        
        # Sonuçları DataFrame'e ekle
        result_df['Anomali_Skoru'] = ensemble_scores
        result_df['Anomali_Tahmini'] = np.where(ensemble_scores < 0.5, -1, 1)
        
        # Risk kategorileri
        result_df['Risk_Seviyesi'] = pd.cut(
            result_df['Anomali_Skoru'],
            bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
            labels=['Kritik Risk', 'Yüksek Risk', 'Orta Risk', 'Düşük Risk', 'Normal'],
            include_lowest=True
        )
        
        # Anomali tipi belirleme
        result_df['Anomali_Tipi'] = result_df.apply(
            lambda row: self._determine_anomaly_type(row, selected_features),
            axis=1
        )
        
        # Şüphe indeksi
        result_df['Şüphe_İndeksi'] = 1 - result_df['Anomali_Skoru']
        
        return result_df
    
    def _select_features_for_anomaly(self, df: pd.DataFrame, n_features: int = 10) -> List[str]:
        """
        Anomali tespiti için özellik seç.
        
        Args:
            df: DataFrame
            n_features: Seçilecek özellik sayısı
            
        Returns:
            Seçilmiş özellikler
        """
        # Varyansa göre özellik seç
        variances = df.var()
        top_variance = variances.nlargest(min(n_features, len(variances)))
        
        # Korelasyon analizi
        corr_matrix = df.corr().abs()
        mean_correlation = corr_matrix.mean()
        
        # Yüksek varyanslı ve düşük korelasyonlu özellikler
        selected = []
        for feature in top_variance.index:
            if feature not in selected:
                selected.append(feature)
                if len(selected) >= n_features:
                    break
        
        return selected
    
    def _determine_anomaly_type(self, row: pd.Series, features: List[str]) -> str:
        """
        Anomali tipini belirle.
        
        Args:
            row: Satır verisi
            features: Özellikler
            
        Returns:
            Anomali tipi
        """
        if row['Risk_Seviyesi'] not in ['Kritik Risk', 'Yüksek Risk']:
            return 'Normal'
        
        anomalies = []
        
        # Satış anomalileri
        sales_features = [f for f in features if 'Satış' in f or 'sales' in f.lower()]
        if sales_features:
            sales_values = row[sales_features]
            if len(sales_values) > 0:
                max_sales = sales_values.max()
                if max_sales > sales_values.quantile(0.95) if len(sales_values) > 1 else 0:
                    anomalies.append('Aşırı Satış')
                elif max_sales < sales_values.quantile(0.05) if len(sales_values) > 1 else 0:
                    anomalies.append('Düşük Satış')
        
        # Büyüme anomalileri
        growth_features = [f for f in features if 'Buyume' in f or 'growth' in f.lower()]
        if growth_features:
            growth_values = row[growth_features]
            if len(growth_values) > 0:
                max_growth = growth_values.max()
                if max_growth > 50:
                    anomalies.append('Aşırı Büyüme')
                elif max_growth < -30:
                    anomalies.append('Aşırı Daralma')
        
        # Fiyat anomalileri
        price_features = [f for f in features if 'Fiyat' in f or 'price' in f.lower()]
        if price_features:
            price_values = row[price_features]
            if len(price_values) > 0:
                price_std = price_values.std()
                if price_std > price_values.mean() * 0.5:
                    anomalies.append('Fiyat Oynaklığı')
        
        if not anomalies:
            return 'Genel Anomali'
        
        return ', '.join(anomalies[:2])
    
    def advanced_segmentation_pipeline(self, df: pd.DataFrame, n_clusters: int = 6) -> Dict[str, Any]:
        """
        Gelişmiş segmentasyon pipeline'ı.
        
        Args:
            df: DataFrame
            n_clusters: Küme sayısı
            
        Returns:
            Segmentasyon sonuçları
        """
        results = {
            'segmented_df': None,
            'clustering_metrics': {},
            'segment_profiles': {},
            'visualization_data': {}
        }
        
        try:
            # Özellik seçimi
            features = self._select_segmentation_features(df)
            
            if len(features) < 3:
                return results
            
            X = df[features].fillna(0)
            
            if len(X) < n_clusters * 10:
                return results
            
            # Ölçeklendirme
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Boyut indirgeme pipeline'ı
            dim_reduction_results = self._dimensionality_reduction_pipeline(X_scaled)
            
            # Kümeleme pipeline'ı
            clustering_results = self._clustering_pipeline(
                dim_reduction_results['best_representation'],
                n_clusters=n_clusters
            )
            
            # Segment isimlendirme
            segment_names = self._name_segments(
                clustering_results['labels'],
                df[features],
                n_clusters
            )
            
            # Sonuç DataFrame'i oluştur
            segmented_df = df.copy()
            segmented_df['Segment_Kodu'] = clustering_results['labels']
            segmented_df['Segment_Adı'] = segmented_df['Segment_Kodu'].map(segment_names)
            
            # Segment profilleri oluştur
            segment_profiles = self._create_segment_profiles(segmented_df, features)
            
            # Metrikleri hesapla
            metrics = self._calculate_clustering_metrics(
                dim_reduction_results['best_representation'],
                clustering_results['labels']
            )
            
            # Görselleştirme verileri
            viz_data = {
                'pca_2d': dim_reduction_results.get('pca_2d', None),
                'tsne_2d': dim_reduction_results.get('tsne_2d', None),
                'umap_2d': dim_reduction_results.get('umap_2d', None),
                'labels': clustering_results['labels']
            }
            
            results.update({
                'segmented_df': segmented_df,
                'clustering_metrics': metrics,
                'segment_profiles': segment_profiles,
                'visualization_data': viz_data,
                'feature_importance': clustering_results.get('feature_importance', {})
            })
            
        except Exception as e:
            st.error(f"Segmentasyon hatası: {str(e)}")
        
        return results
    
    def _select_segmentation_features(self, df: pd.DataFrame) -> List[str]:
        """
        Segmentasyon için özellik seç.
        
        Args:
            df: DataFrame
            
        Returns:
            Seçilmiş özellikler
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Öncelikli özellik kategorileri
        priority_categories = [
            'Satış', 'Buyume', 'Fiyat', 'Pazar', 'Payı',
            'Hacim', 'Birim', 'Kar', 'Marj', 'Cost'
        ]
        
        selected = []
        
        # Öncelikli özellikleri ekle
        for category in priority_categories:
            category_features = [col for col in numeric_cols if category in col]
            selected.extend(category_features[:2])  # Her kategoriden en fazla 2
        
        # Benzersiz hale getir
        selected = list(set(selected))
        
        # Eğer yeterli özellik yoksa, en yüksek varyanslıları ekle
        if len(selected) < 5:
            remaining_cols = [col for col in numeric_cols if col not in selected]
            if remaining_cols:
                variances = df[remaining_cols].var()
                top_variance = variances.nlargest(5 - len(selected))
                selected.extend(top_variance.index.tolist())
        
        return selected[:10]  # En fazla 10 özellik
    
    def _dimensionality_reduction_pipeline(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Boyut indirgeme pipeline'ı.
        
        Args:
            X: Ölçeklendirilmiş veri
            
        Returns:
            Boyut indirgeme sonuçları
        """
        results = {}
        
        try:
            # 1. PCA
            pca = PCA(n_components=min(10, X.shape[1]))
            X_pca = pca.fit_transform(X)
            results['pca'] = X_pca
            results['pca_variance'] = pca.explained_variance_ratio_
            
            # 2D PCA
            pca_2d = PCA(n_components=2)
            results['pca_2d'] = pca_2d.fit_transform(X)
            
            # 2. t-SNE (eğer örnek sayısı makul ise)
            if X.shape[0] <= 10000:
                tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                results['tsne_2d'] = tsne.fit_transform(X)
            
            # 3. UMAP
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            results['umap_2d'] = umap_reducer.fit_transform(X)
            
            # En iyi temsili seç
            results['best_representation'] = X_pca[:, :5]  # İlk 5 PCA bileşeni
            
        except Exception as e:
            st.warning(f"Boyut indirgeme hatası: {str(e)}")
            results['best_representation'] = X[:, :min(10, X.shape[1])]
        
        return results
    
    def _clustering_pipeline(self, X: np.ndarray, n_clusters: int = 6) -> Dict[str, Any]:
        """
        Kümeleme pipeline'ı.
        
        Args:
            X: Özellik matrisi
            n_clusters: Küme sayısı
            
        Returns:
            Kümeleme sonuçları
        """
        results = {}
        
        try:
            # 1. K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            kmeans_labels = kmeans.fit_predict(X)
            kmeans_score = silhouette_score(X, kmeans_labels)
            
            # 2. Gaussian Mixture
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm_labels = gmm.fit_predict(X)
            gmm_score = silhouette_score(X, gmm_labels)
            
            # 3. Agglomerative Clustering
            agg = AgglomerativeClustering(n_clusters=n_clusters)
            agg_labels = agg.fit_predict(X)
            agg_score = silhouette_score(X, agg_labels)
            
            # En iyi modeli seç
            scores = {
                'kmeans': kmeans_score,
                'gmm': gmm_score,
                'agg': agg_score
            }
            
            best_algo = max(scores, key=scores.get)
            
            if best_algo == 'kmeans':
                best_labels = kmeans_labels
                results['model'] = kmeans
            elif best_algo == 'gmm':
                best_labels = gmm_labels
                results['model'] = gmm
            else:
                best_labels = agg_labels
                results['model'] = agg
            
            results['labels'] = best_labels
            results['best_algo'] = best_algo
            results['silhouette_score'] = scores[best_algo]
            
            # Özellik önemliliği
            if best_algo == 'kmeans':
                # K-Means için küme merkezlerine göre özellik önemliliği
                centers = kmeans.cluster_centers_
                feature_importance = np.std(centers, axis=0)
                results['feature_importance'] = dict(zip(
                    range(X.shape[1]),
                    feature_importance
                ))
            
        except Exception as e:
            st.warning(f"Kümeleme hatası: {str(e)}")
            # Basit K-Means fallback
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            results['labels'] = kmeans.fit_predict(X)
            results['model'] = kmeans
            results['best_algo'] = 'kmeans'
            results['silhouette_score'] = silhouette_score(X, results['labels'])
        
        return results
    
    def _name_segments(self, labels: np.ndarray, X: pd.DataFrame, n_clusters: int) -> Dict[int, str]:
        """
        Segmentlere isim ver.
        
        Args:
            labels: Küme etiketleri
            X: Özellik verisi
            n_clusters: Küme sayısı
            
        Returns:
            Segment isimleri haritası
        """
        segment_names = {}
        
        # Her küme için ortalama değerler
        for cluster in range(n_clusters):
            cluster_mask = labels == cluster
            if not cluster_mask.any():
                segment_names[cluster] = f'Segment_{cluster}'
                continue
            
            cluster_data = X[cluster_mask]
            
            # Ortalama satış
            sales_cols = [col for col in X.columns if 'Satış' in col]
            avg_sales = cluster_data[sales_cols].mean().mean() if sales_cols else 0
            
            # Ortalama büyüme
            growth_cols = [col for col in X.columns if 'Buyume' in col]
            avg_growth = cluster_data[growth_cols].mean().mean() if growth_cols else 0
            
            # BCG Matrix benzeri isimlendirme
            if avg_sales > X[sales_cols].mean().mean() * 1.5 if sales_cols else 0:
                if avg_growth > 20:
                    name = ProductSegment.STARS.value
                elif avg_growth > 5:
                    name = ProductSegment.CASH_COWS.value
                else:
                    name = ProductSegment.MATURE.value
            elif avg_growth > 30:
                name = ProductSegment.EMERGING.value
            elif avg_growth < -10:
                name = ProductSegment.DECLINING.value
            elif avg_sales < X[sales_cols].mean().mean() * 0.5 if sales_cols else 0:
                name = ProductSegment.QUESTION_MARKS.value
            else:
                name = f'Segment_{cluster}'
            
            segment_names[cluster] = name
        
        return segment_names
    
    def _create_segment_profiles(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Segment profilleri oluştur.
        
        Args:
            df: Segmentlenmiş DataFrame
            features: Özellikler
            
        Returns:
            Segment profilleri
        """
        profiles = {}
        
        for segment in df['Segment_Adı'].unique():
            segment_df = df[df['Segment_Adı'] == segment]
            
            profile = {
                'urun_sayisi': len(segment_df),
                'ortalama_satis': segment_df[[f for f in features if 'Satış' in f]].mean().mean() if any('Satış' in f for f in features) else 0,
                'ortalama_buyume': segment_df[[f for f in features if 'Buyume' in f]].mean().mean() if any('Buyume' in f for f in features) else 0,
                'ortalama_pazar_payi': segment_df[[f for f in features if 'Pazar' in f]].mean().mean() if any('Pazar' in f for f in features) else 0,
                'dominant_ulkeler': segment_df['Ülke'].value_counts().head(3).to_dict() if 'Ülke' in df.columns else {},
                'dominant_sirketler': segment_df['Şirket'].value_counts().head(3).to_dict() if 'Şirket' in df.columns else {},
                'urun_cesitliligi': segment_df['Molekül'].nunique() if 'Molekül' in df.columns else 0,
                'risk_profili': segment_df['Risk_Seviyesi'].value_counts().to_dict() if 'Risk_Seviyesi' in df.columns else {}
            }
            
            profiles[segment] = profile
        
        return profiles
    
    def _calculate_clustering_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Kümeleme metriklerini hesapla.
        
        Args:
            X: Özellik matrisi
            labels: Küme etiketleri
            
        Returns:
            Kümeleme metrikleri
        """
        metrics = {}
        
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            
            # Küme içi varyans
            unique_labels = np.unique(labels)
            within_cluster_variance = 0
            for label in unique_labels:
                cluster_points = X[labels == label]
                if len(cluster_points) > 0:
                    centroid = cluster_points.mean(axis=0)
                    within_cluster_variance += np.sum((cluster_points - centroid) ** 2)
            
            metrics['within_cluster_variance'] = within_cluster_variance
            metrics['cluster_separation'] = 1 / (metrics['davies_bouldin_score'] + 1e-10)
            
        except Exception as e:
            st.warning(f"Kümeleme metrikleri hatası: {str(e)}")
        
        return metrics
    
    def explainable_ai_analysis(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Açıklanabilir AI analizi.
        
        Args:
            df: DataFrame
            target_col: Hedef sütun
            
        Returns:
            SHAP analiz sonuçları
        """
        results = {
            'shap_values': None,
            'feature_importance': {},
            'summary_plot': None,
            'dependence_plots': {},
            'model_performance': {}
        }
        
        if not SHAP_AVAILABLE:
            return results
        
        try:
            # Sayısal özellikleri seç
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if target_col not in numeric_cols:
                return results
            
            features = [col for col in numeric_cols if col != target_col]
            
            if len(features) < 2:
                return results
            
            X = df[features].fillna(0)
            y = df[target_col].fillna(0)
            
            # Random Forest modeli eğit
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            model.fit(X, y)
            
            # SHAP değerlerini hesapla
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Özellik önemliliği
            feature_importance = np.abs(shap_values).mean(axis=0)
            results['feature_importance'] = dict(zip(features, feature_importance))
            
            # Model performansı
            predictions = model.predict(X)
            results['model_performance'] = {
                'r2_score': model.score(X, y),
                'mse': np.mean((y - predictions) ** 2),
                'mae': np.mean(np.abs(y - predictions))
            }
            
            # SHAP değerlerini kaydet
            results['shap_values'] = shap_values
            
            # Bağımlılık plot'ları için önemli özellikleri seç
            top_features = sorted(results['feature_importance'].items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
            
            for feature, importance in top_features:
                results['dependence_plots'][feature] = {
                    'importance': importance,
                    'shap_dependence': shap_values[:, features.index(feature)]
                }
            
        except Exception as e:
            st.warning(f"SHAP analizi hatası: {str(e)}")
        
        return results
    
    def generate_strategic_insights(self, df: pd.DataFrame, metrics: MarketMetrics) -> List[Dict[str, Any]]:
        """
        Stratejik içgörüler üret.
        
        Args:
            df: DataFrame
            metrics: Pazar metrikleri
            
        Returns:
            Stratejik içgörüler listesi
        """
        insights = []
        
        try:
            # 1. Pazar Yapısı İçgörüsü
            if metrics.hhi_index > 2500:
                insights.append({
                    'type': 'market_structure',
                    'title': '🏢 Monopolistik Pazar Yapısı',
                    'description': f'HHI İndeksi: {metrics.hhi_index:.0f} - Pazar çok yoğunlaşmış',
                    'recommendation': 'Rakiplerle işbirliği fırsatlarını değerlendirin. Regülasyon riskine karşı hazırlıklı olun.',
                    'priority': 'high',
                    'impact': 'strategic'
                })
            elif metrics.hhi_index > 1800:
                insights.append({
                    'type': 'market_structure',
                    'title': '🏢 Oligopolistik Pazar',
                    'description': f'HHI İndeksi: {metrics.hhi_index:.0f} - Birkaç büyük oyuncu hakim',
                    'recommendation': 'Pazar payınızı korumak için fiyat stratejilerinizi gözden geçirin.',
                    'priority': 'medium',
                    'impact': 'operational'
                })
            
            # 2. Büyüme Trendi İçgörüsü
            if metrics.yoy_growth > 15:
                insights.append({
                    'type': 'growth',
                    'title': '🚀 Yüksek Büyüme Trendi',
                    'description': f'Yıllık büyüme: %{metrics.yoy_growth:.1f} - Pazar genişliyor',
                    'recommendation': 'Yatırımları artırın ve kapasite planlaması yapın.',
                    'priority': 'high',
                    'impact': 'financial'
                })
            elif metrics.yoy_growth < 0:
                insights.append({
                    'type': 'growth',
                    'title': '⚠️ Negatif Büyüme Uyarısı',
                    'description': f'Yıllık büyüme: %{metrics.yoy_growth:.1f} - Pazar daralıyor',
                    'recommendation': 'Maliyet optimizasyonu ve ürün çeşitlendirmesine odaklanın.',
                    'priority': 'critical',
                    'impact': 'survival'
                })
            
            # 3. Fiyat Rekabeti İçgörüsü
            if hasattr(metrics, 'price_index'):
                price_cols = [col for col in df.columns if 'Fiyat' in col]
                if price_cols:
                    price_variance = df[price_cols].std().mean()
                    if price_variance > df[price_cols].mean().mean() * 0.3:
                        insights.append({
                            'type': 'pricing',
                            'title': '💰 Yüksek Fiyat Oynaklığı',
                            'description': f'Fiyat varyasyonu: %{(price_variance/df[price_cols].mean().mean()*100):.1f}',
                            'recommendation': 'Fiyatlandırma stratejinizi stabilize edin ve dalgalanma riskini yönetin.',
                            'priority': 'medium',
                            'impact': 'revenue'
                        })
            
            # 4. Ürün Portföyü İçgörüsü
            if 'Segment_Adı' in df.columns:
                segment_distribution = df['Segment_Adı'].value_counts(normalize=True) * 100
                
                stars_ratio = segment_distribution.get('Yıldız Ürünler', 0)
                cash_cows_ratio = segment_distribution.get('Nakit İnekleri', 0)
                
                if stars_ratio < 10:
                    insights.append({
                        'type': 'portfolio',
                        'title': '⭐ Yıldız Ürün Eksikliği',
                        'description': f'Portföyde sadece %{stars_ratio:.1f} yıldız ürün var',
                        'recommendation': 'Ar-Ge yatırımlarını artırarak geleceğin yıldız ürünlerini geliştirin.',
                        'priority': 'high',
                        'impact': 'strategic'
                    })
                
                if cash_cows_ratio > 50:
                    insights.append({
                        'type': 'portfolio',
                        'title': '🐄 Aşırı Nakit İneği Bağımlılığı',
                        'description': f'Portföyün %{cash_cows_ratio:.1f}\'si nakit ineklerinden oluşuyor',
                        'recommendation': 'Nakit akışını yeni büyüme alanlarına yönlendirin.',
                        'priority': 'medium',
                        'impact': 'financial'
                    })
            
            # 5. Risk Konsantrasyonu İçgörüsü
            if 'Risk_Seviyesi' in df.columns:
                risk_distribution = df['Risk_Seviyesi'].value_counts(normalize=True) * 100
                high_risk_ratio = risk_distribution.get('Yüksek Risk', 0) + risk_distribution.get('Kritik Risk', 0)
                
                if high_risk_ratio > 20:
                    insights.append({
                        'type': 'risk',
                        'title': '🚨 Yüksek Risk Konsantrasyonu',
                        'description': f'Portföyün %{high_risk_ratio:.1f}\'si yüksek riskli',
                        'recommendation': 'Risk dağılımını iyileştirmek için ürün çeşitlendirmesi yapın.',
                        'priority': 'critical',
                        'impact': 'risk'
                    })
            
            # 6. Coğrafi Dağılım İçgörüsü
            if 'Ülke' in df.columns:
                country_distribution = df['Ülke'].value_counts(normalize=True) * 100
                top_country_ratio = country_distribution.iloc[0] if len(country_distribution) > 0 else 0
                
                if top_country_ratio > 50:
                    insights.append({
                        'type': 'geographic',
                        'title': '🌍 Coğrafi Konsantrasyon Riski',
                        'description': f'Pazarın %{top_country_ratio:.1f}\'si tek ülkede',
                        'recommendation': 'Coğrafi çeşitlendirme için yeni pazarlara açılın.',
                        'priority': 'medium',
                        'impact': 'strategic'
                    })
            
            # 7. İnovasyon İçgörüsü
            if metrics.innovation_index < 30:
                insights.append({
                    'type': 'innovation',
                    'title': '🔬 Düşük İnovasyon Yoğunluğu',
                    'description': f'İnovasyon indeksi: %{metrics.innovation_index:.1f}',
                    'recommendation': 'Ar-Ge yatırımlarını ve patent başvurularını artırın.',
                    'priority': 'high',
                    'impact': 'strategic'
                })
            
        except Exception as e:
            st.warning(f"İçgörü üretme hatası: {str(e)}")
        
        return insights

# ================================================
# 5. ADVANCED VISUALIZATION ENGINE
# ================================================

class AdvancedVizEngine:
    """
    Gelişmiş görselleştirme motoru.
    3D, interaktif ve animasyonlu grafikler.
    """
    
    def __init__(self):
        self.color_palettes = {
            'executive': ['#1a237e', '#283593', '#303f9f', '#3949ab', '#3f51b5',
                         '#5c6bc0', '#7986cb', '#9fa8da', '#c5cae9'],
            'pharma': ['#00695c', '#00796b', '#00897b', '#009688', '#26a69a',
                      '#4db6ac', '#80cbc4', '#b2dfdb', '#e0f2f1'],
            'risk': ['#b71c1c', '#d32f2f', '#f44336', '#ef5350', '#e57373',
                    '#ef9a9a', '#ffcdd2', '#ffebee'],
            'growth': ['#1b5e20', '#2e7d32', '#388e3c', '#43a047', '#4caf50',
                      '#66bb6a', '#81c784', '#a5d6a7', '#c8e6c9']
        }
        
        self.chart_templates = {
            'dark': 'plotly_dark',
            'white': 'plotly_white',
            'presentation': 'presentation',
            'seaborn': 'seaborn'
        }
    
    def create_executive_dashboard(self, df: pd.DataFrame, metrics: MarketMetrics, 
                                  insights: List[Dict]) -> Dict[str, go.Figure]:
        """
        Yönetici gösterge paneli oluştur.
        
        Args:
            df: DataFrame
            metrics: Pazar metrikleri
            insights: Stratejik içgörüler
            
        Returns:
            Grafik sözlüğü
        """
        dashboard = {}
        
        try:
            # 1. Ana Metrik Gösterge Tablosu
            dashboard['metrics_board'] = self._create_metrics_board(metrics)
            
            # 2. Pazar Yoğunluğu Grafiği
            dashboard['market_concentration'] = self._create_market_concentration_chart(df)
            
            # 3. Segmentasyon Haritası
            dashboard['segmentation_map'] = self._create_segmentation_map(df)
            
            # 4. Zaman Serisi Tahmini
            dashboard['time_series_forecast'] = self._create_time_series_forecast(df)
            
            # 5. Risk Heatmap
            dashboard['risk_heatmap'] = self._create_risk_heatmap(df)
            
            # 6. Coğrafi Dağılım
            dashboard['geographic_distribution'] = self._create_geographic_distribution(df)
            
            # 7. Ürün Portföy Matrisi
            dashboard['portfolio_matrix'] = self._create_portfolio_matrix(df)
            
            # 8. Şirket Performans Radar
            dashboard['company_radar'] = self._create_company_radar_chart(df)
            
        except Exception as e:
            st.error(f"Dashboard oluşturma hatası: {str(e)}")
        
        return dashboard
    
    def _create_metrics_board(self, metrics: MarketMetrics) -> go.Figure:
        """
        Metrik gösterge tablosu oluştur.
        
        Args:
            metrics: Pazar metrikleri
            
        Returns:
            Metrik board figürü
        """
        fig = make_subplots(
            rows=2, cols=4,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, 
                   {'type': 'indicator'}, {'type': 'indicator'}],
                  [{'type': 'indicator'}, {'type': 'indicator'}, 
                   {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=('Toplam Pazar', 'Büyüme Oranı', 'HHI İndeksi', 'Pazar Oynaklığı',
                          'Fiyat İndeksi', 'Uluslararasılık', 'İnovasyon', 'Konsantrasyon')
        )
        
        # Metrik 1: Toplam Pazar
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics.total_market_value / 1e6,
                number={'prefix': "$", 'suffix': "M", 'font': {'size': 30}},
                delta={'position': "bottom", 'reference': metrics.total_market_value / 1e6 * 0.9},
                title={'text': "Pazar Değeri"},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )
        
        # Metrik 2: Büyüme Oranı
        growth_color = 'green' if metrics.yoy_growth > 0 else 'red'
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics.yoy_growth,
                number={'suffix': "%", 'font': {'size': 30, 'color': growth_color}},
                delta={'position': "bottom", 'reference': 0},
                title={'text': "Yıllık Büyüme"},
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )
        
        # Metrik 3: HHI İndeksi
        hhi_status = 'Kritik' if metrics.hhi_index > 2500 else 'Yüksek' if metrics.hhi_index > 1800 else 'Orta'
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=metrics.hhi_index,
                number={'font': {'size': 30}},
                gauge={
                    'axis': {'range': [0, 5000]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1000], 'color': "lightgray"},
                        {'range': [1000, 1800], 'color': "gray"},
                        {'range': [1800, 2500], 'color': "darkgray"},
                        {'range': [2500, 5000], 'color': "black"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 2500
                    }
                },
                title={'text': f"HHI - {hhi_status}"},
                domain={'row': 0, 'column': 2}
            ),
            row=1, col=3
        )
        
        # Diğer metrikleri ekle
        metrics_data = [
            (metrics.market_volatility, "%", "Oynaklık", 1, 4),
            (metrics.price_index, "$", "Fiyat", 2, 1),
            (metrics.international_penetration, " ülke", "Kapsam", 2, 2),
            (metrics.innovation_index, "%", "İnovasyon", 2, 3),
            (metrics.concentration_ratio, "%", "Konsantrasyon", 2, 4)
        ]
        
        for value, suffix, title, row, col in metrics_data:
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=value,
                    number={'suffix': suffix, 'font': {'size': 24}},
                    title={'text': title},
                    domain={'row': row-1, 'column': col-1}
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=500,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    def _create_market_concentration_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Pazar yoğunluğu grafiği oluştur.
        
        Args:
            df: DataFrame
            
        Returns:
            Pazar yoğunluğu figürü
        """
        if 'Şirket' not in df.columns:
            return self._create_empty_chart("Şirket verisi yok")
        
        # Şirket bazlı satışlar
        sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
        if not sales_cols:
            return self._create_empty_chart("Satış verisi yok")
        
        latest_sales = sales_cols[-1]
        company_sales = df.groupby('Şirket')[latest_sales].sum().sort_values(ascending=False)
        
        # Lorenz eğrisi için kümülatif paylar
        total_sales = company_sales.sum()
        cumulative_percent = company_sales.cumsum() / total_sales * 100
        perfect_equality = np.linspace(0, 100, len(company_sales))
        
        # Gini katsayısı
        gini_coefficient = 1 - 2 * (cumulative_percent / 100).sum() / len(company_sales)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Pazar Payı Dağılımı', 'Lorenz Eğrisi & Gini Katsayısı'),
            specs=[[{'type': 'pie'}, {'type': 'xy'}]]
        )
        
        # Pasta grafik
        top_companies = company_sales.head(10)
        others_sales = company_sales[10:].sum() if len(company_sales) > 10 else 0
        
        if others_sales > 0:
            top_companies['Diğerleri'] = others_sales
        
        fig.add_trace(
            go.Pie(
                labels=top_companies.index,
                values=top_companies.values,
                hole=0.4,
                marker_colors=self.color_palettes['executive'],
                textinfo='percent+label',
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Lorenz eğrisi
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(cumulative_percent) + 1),
                y=cumulative_percent.values,
                mode='lines',
                name='Lorenz Eğrisi',
                line=dict(color='blue', width=3)
            ),
            row=1, col=2
        )
        
        # Mükemmel eşitlik çizgisi
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(perfect_equality) + 1),
                y=perfect_equality,
                mode='lines',
                name='Mükemmel Eşitlik',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            template='plotly_dark',
            title_text=f"Pazar Yoğunluğu Analizi | Gini Katsayısı: {gini_coefficient:.3f}",
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(title_text="Firma Sıralaması", row=1, col=2)
        fig.update_yaxes(title_text="Kümülatif Pazar Payı (%)", row=1, col=2)
        
        return fig
    
    def _create_segmentation_map(self, df: pd.DataFrame) -> go.Figure:
        """
        Segmentasyon haritası oluştur.
        
        Args:
            df: DataFrame
            
        Returns:
            Segmentasyon figürü
        """
        if 'Segment_Adı' not in df.columns:
            return self._create_empty_chart("Segmentasyon verisi yok")
        
        # PCA veya t-SNE koordinatları
        if 'PCA_1' in df.columns and 'PCA_2' in df.columns:
            x_col, y_col = 'PCA_1', 'PCA_2'
            x_title, y_title = 'PCA Bileşen 1', 'PCA Bileşen 2'
        elif 'TSNE_1' in df.columns and 'TSNE_2' in df.columns:
            x_col, y_col = 'TSNE_1', 'TSNE_2'
            x_title, y_title = 't-SNE 1', 't-SNE 2'
        elif 'UMAP_1' in df.columns and 'UMAP_2' in df.columns:
            x_col, y_col = 'UMAP_1', 'UMAP_2'
            x_title, y_title = 'UMAP 1', 'UMAP 2'
        else:
            return self._create_empty_chart("Boyut indirgeme verisi yok")
        
        # Segment renk haritası
        segments = df['Segment_Adı'].unique()
        colors = self.color_palettes['pharma'][:len(segments)]
        color_map = {seg: color for seg, color in zip(segments, colors)}
        
        fig = go.Figure()
        
        for segment in segments:
            segment_df = df[df['Segment_Adı'] == segment]
            
            # Satış büyüklüğü
            sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
            size_col = sales_cols[-1] if sales_cols else None
            
            if size_col:
                sizes = np.log10(segment_df[size_col] + 1) * 10
            else:
                sizes = 10
            
            fig.add_trace(
                go.Scatter(
                    x=segment_df[x_col],
                    y=segment_df[y_col],
                    mode='markers',
                    name=segment,
                    marker=dict(
                        color=color_map[segment],
                        size=sizes,
                        line=dict(width=1, color='white'),
                        opacity=0.7
                    ),
                    text=segment_df.get('Molekul', segment_df.index).astype(str),
                    hoverinfo='text+name',
                    hovertemplate='<b>%{text}</b><br>Segment: %{name}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                )
            )
        
        # Segment merkezleri
        for segment in segments:
            segment_df = df[df['Segment_Adı'] == segment]
            if len(segment_df) > 0:
                center_x = segment_df[x_col].median()
                center_y = segment_df[y_col].median()
                
                fig.add_trace(
                    go.Scatter(
                        x=[center_x],
                        y=[center_y],
                        mode='markers',
                        name=f'{segment} Merkezi',
                        marker=dict(
                            color=color_map[segment],
                            size=20,
                            symbol='star',
                            line=dict(width=2, color='white')
                        ),
                        showlegend=False,
                        hoverinfo='none'
                    )
                )
        
        fig.update_layout(
            title='Ürün Segmentasyon Haritası',
            xaxis_title=x_title,
            yaxis_title=y_title,
            height=600,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def _create_time_series_forecast(self, df: pd.DataFrame) -> go.Figure:
        """
        Zaman serisi tahmin grafiği oluştur.
        
        Args:
            df: DataFrame
            
        Returns:
            Zaman serisi figürü
        """
        # Satış sütunlarını bul
        sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
        
        if len(sales_cols) < 3:
            return self._create_empty_chart("Yeterli zaman serisi verisi yok")
        
        # Yıllık satışları topla
        yearly_sales = {}
        for col in sorted(sales_cols):
            year_match = re.search(r'20\d{2}', col)
            if year_match:
                year = int(year_match.group())
                yearly_sales[year] = df[col].sum()
        
        if len(yearly_sales) < 3:
            return self._create_empty_chart("Yeterli yıllık veri yok")
        
        # Zaman serisi oluştur
        years = list(yearly_sales.keys())
        sales = list(yearly_sales.values())
        
        fig = go.Figure()
        
        # Tarihsel veri
        fig.add_trace(
            go.Scatter(
                x=years,
                y=sales,
                mode='lines+markers',
                name='Tarihsel Satış',
                line=dict(color='blue', width=3),
                marker=dict(size=10, color='blue')
            )
        )
        
        # Trend çizgisi
        z = np.polyfit(years, sales, 1)
        p = np.poly1d(z)
        trend_line = p(years)
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=trend_line,
                mode='lines',
                name='Trend Çizgisi',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        # Büyüme oranları
        growth_rates = []
        for i in range(1, len(sales)):
            growth = ((sales[i] - sales[i-1]) / sales[i-1]) * 100
            growth_rates.append(growth)
        
        # İkinci eksen için büyüme grafiği
        fig.add_trace(
            go.Bar(
                x=years[1:],
                y=growth_rates,
                name='Yıllık Büyüme (%)',
                yaxis='y2',
                marker_color='green',
                opacity=0.5
            )
        )
        
        fig.update_layout(
            title='Satış Trendi ve Büyüme Oranları',
            xaxis_title='Yıl',
            yaxis_title='Satış (USD)',
            yaxis2=dict(
                title='Büyüme (%)',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            height=500,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
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
    
    def _create_risk_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """
        Risk heatmap oluştur.
        
        Args:
            df: DataFrame
            
        Returns:
            Risk heatmap figürü
        """
        if 'Risk_Seviyesi' not in df.columns:
            return self._create_empty_chart("Risk verisi yok")
        
        # Risk seviyelerine göre grupla
        risk_levels = ['Kritik Risk', 'Yüksek Risk', 'Orta Risk', 'Düşük Risk', 'Normal']
        
        # Şirket ve risk seviyesi çapraz tablo
        if 'Şirket' in df.columns:
            risk_by_company = pd.crosstab(df['Şirket'], df['Risk_Seviyesi'])
            risk_by_company = risk_by_company.reindex(columns=risk_levels, fill_value=0)
            
            # En riskli 15 şirket
            top_companies = risk_by_company.sum(axis=1).nlargest(15).index
            risk_by_company = risk_by_company.loc[top_companies]
            
            fig = px.imshow(
                risk_by_company.T,
                text_auto=True,
                color_continuous_scale='Reds',
                title='Şirket Bazında Risk Dağılımı',
                labels=dict(x="Şirket", y="Risk Seviyesi", color="Ürün Sayısı")
            )
        
        else:
            # Segment bazında risk
            if 'Segment_Adı' in df.columns:
                risk_by_segment = pd.crosstab(df['Segment_Adı'], df['Risk_Seviyesi'])
                risk_by_segment = risk_by_segment.reindex(columns=risk_levels, fill_value=0)
                
                fig = px.imshow(
                    risk_by_segment.T,
                    text_auto=True,
                    color_continuous_scale='Reds',
                    title='Segment Bazında Risk Dağılımı',
                    labels=dict(x="Segment", y="Risk Seviyesi", color="Ürün Sayısı")
                )
            else:
                return self._create_empty_chart("Risk analizi için yeterli veri yok")
        
        fig.update_layout(
            height=500,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        
        return fig
    
    def _create_geographic_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Coğrafi dağılım haritası oluştur.
        
        Args:
            df: DataFrame
            
        Returns:
            Coğrafi dağılım figürü
        """
        if 'Ülke' not in df.columns:
            return self._create_empty_chart("Coğrafi veri yok")
        
        # Ülke bazında satışlar
        sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
        if not sales_cols:
            return self._create_empty_chart("Satış verisi yok")
        
        latest_sales = sales_cols[-1]
        country_sales = df.groupby('Ülke')[latest_sales].sum().reset_index()
        country_sales.columns = ['Ülke', 'Satış']
        
        # Ülke isimlerini standardize et
        country_mapping = {
            'USA': 'United States',
            'US': 'United States',
            'U.S.A': 'United States',
            'UK': 'United Kingdom',
            'U.K': 'United Kingdom',
            'UAE': 'United Arab Emirates',
            'S. Korea': 'South Korea',
            'South Korea': 'Korea, Republic of',
            'Russia': 'Russian Federation',
            'Iran': 'Iran, Islamic Republic of',
            'Turkey': 'Türkiye',
            'Turkiye': 'Türkiye'
        }
        
        country_sales['Ülke'] = country_sales['Ülke'].replace(country_mapping)
        
        fig = px.choropleth(
            country_sales,
            locations='Ülke',
            locationmode='country names',
            color='Satış',
            hover_name='Ülke',
            hover_data={'Satış': ':.2f'},
            color_continuous_scale='Viridis',
            title='Global Pazar Dağılımı',
            projection='natural earth'
        )
        
        fig.update_layout(
            height=600,
            template='plotly_dark',
            geo=dict(
                bgcolor='rgba(0,0,0,0)',
                lakecolor='#1e3a5f',
                landcolor='#2d4a7a',
                subunitcolor='#64748b'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        
        return fig
    
    def _create_portfolio_matrix(self, df: pd.DataFrame) -> go.Figure:
        """
        Portföy matrisi (BCG Matrix) oluştur.
        
        Args:
            df: DataFrame
            
        Returns:
            Portföy matrisi figürü
        """
        # Pazar payı ve büyüme oranı sütunlarını bul
        market_share_col = None
        growth_col = None
        
        for col in df.columns:
            if 'Pazar_Payi' in col:
                market_share_col = col
            elif 'Buyume' in col:
                growth_col = col
        
        if not market_share_col or not growth_col:
            return self._create_empty_chart("Portföy analizi için gerekli veri yok")
        
        fig = go.Figure()
        
        # Segment renkleri
        if 'Segment_Adı' in df.columns:
            segments = df['Segment_Adı'].unique()
            colors = self.color_palettes['pharma'][:len(segments)]
            color_map = {seg: color for seg, color in zip(segments, colors)}
        else:
            color_map = {'All': '#1a237e'}
            df['Segment_Adı'] = 'All'
        
        for segment in df['Segment_Adı'].unique():
            segment_df = df[df['Segment_Adı'] == segment]
            
            # Boyut için satış değeri
            sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
            size_col = sales_cols[-1] if sales_cols else None
            
            if size_col:
                sizes = np.log10(segment_df[size_col] + 1) * 10
            else:
                sizes = 10
            
            fig.add_trace(
                go.Scatter(
                    x=segment_df[market_share_col],
                    y=segment_df[growth_col],
                    mode='markers',
                    name=segment,
                    marker=dict(
                        color=color_map[segment],
                        size=sizes,
                        line=dict(width=1, color='white'),
                        opacity=0.7
                    ),
                    text=segment_df.get('Molekul', segment_df.index).astype(str),
                    hoverinfo='text+name+x+y',
                    hovertemplate='<b>%{text}</b><br>Pazar Payı: %{x:.1f}%<br>Büyüme: %{y:.1f}%<extra></extra>'
                )
            )
        
        # BCG Matrix bölgeleri
        fig.add_shape(
            type="rect",
            x0=df[market_share_col].median(),
            y0=df[growth_col].median(),
            x1=df[market_share_col].max(),
            y1=df[growth_col].max(),
            line=dict(color="Gold", width=2),
            fillcolor="rgba(255, 215, 0, 0.1)",
            name="Yıldızlar"
        )
        
        fig.add_shape(
            type="rect",
            x0=df[market_share_col].median(),
            y0=df[growth_col].min(),
            x1=df[market_share_col].max(),
            y1=df[growth_col].median(),
            line=dict(color="Green", width=2),
            fillcolor="rgba(0, 128, 0, 0.1)",
            name="Nakit İnekleri"
        )
        
        fig.add_shape(
            type="rect",
            x0=df[market_share_col].min(),
            y0=df[growth_col].median(),
            x1=df[market_share_col].median(),
            y1=df[growth_col].max(),
            line=dict(color="Blue", width=2),
            fillcolor="rgba(0, 0, 255, 0.1)",
            name="Soru İşaretleri"
        )
        
        fig.add_shape(
            type="rect",
            x0=df[market_share_col].min(),
            y0=df[growth_col].min(),
            x1=df[market_share_col].median(),
            y1=df[growth_col].median(),
            line=dict(color="Red", width=2),
            fillcolor="rgba(255, 0, 0, 0.1)",
            name="Zayıf Ürünler"
        )
        
        # Bölge etiketleri
        fig.add_annotation(
            x=df[market_share_col].max() * 0.75,
            y=df[growth_col].max() * 0.75,
            text="⭐ Yıldızlar",
            showarrow=False,
            font=dict(size=14, color="Gold")
        )
        
        fig.add_annotation(
            x=df[market_share_col].max() * 0.75,
            y=df[growth_col].min() * 1.3,
            text="🐄 Nakit İnekleri",
            showarrow=False,
            font=dict(size=14, color="Green")
        )
        
        fig.add_annotation(
            x=df[market_share_col].min() * 1.3,
            y=df[growth_col].max() * 0.75,
            text="❓ Soru İşaretleri",
            showarrow=False,
            font=dict(size=14, color="Blue")
        )
        
        fig.add_annotation(
            x=df[market_share_col].min() * 1.3,
            y=df[growth_col].min() * 1.3,
            text="⚠️ Zayıf Ürünler",
            showarrow=False,
            font=dict(size=14, color="Red")
        )
        
        fig.update_layout(
            title='BCG Portföy Matrisi',
            xaxis_title='Pazar Payı (%)',
            yaxis_title='Büyüme Oranı (%)',
            height=600,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def _create_company_radar_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Şirket radar grafiği oluştur.
        
        Args:
            df: DataFrame
            
        Returns:
            Radar grafiği figürü
        """
        if 'Şirket' not in df.columns:
            return self._create_empty_chart("Şirket verisi yok")
        
        # En büyük 5 şirketi seç
        sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
        if not sales_cols:
            return self._create_empty_chart("Satış verisi yok")
        
        latest_sales = sales_cols[-1]
        top_companies = df.groupby('Şirket')[latest_sales].sum().nlargest(5).index.tolist()
        
        if len(top_companies) < 2:
            return self._create_empty_chart("Yeterli şirket verisi yok")
        
        # Metrikleri seç
        metrics = []
        
        # 1. Pazar Payı
        total_sales = df[latest_sales].sum()
        metrics.append(('Pazar Payı', latest_sales))
        
        # 2. Büyüme Oranı
        growth_cols = [col for col in df.columns if 'Buyume' in col]
        if growth_cols:
            metrics.append(('Büyüme', growth_cols[-1]))
        
        # 3. Risk Profili
        if 'Risk_Seviyesi' in df.columns:
            # Risk skorunu hesapla
            risk_scores = {'Kritik Risk': 0, 'Yüksek Risk': 25, 'Orta Risk': 50, 
                          'Düşük Risk': 75, 'Normal': 100}
            df['Risk_Skoru'] = df['Risk_Seviyesi'].map(risk_scores)
            metrics.append(('Risk', 'Risk_Skoru'))
        
        # 4. Ürün Çeşitliliği
        if 'Molekül' in df.columns:
            metrics.append(('Çeşitlilik', 'Molekül'))
        
        # 5. Fiyat Rekabeti
        price_cols = [col for col in df.columns if 'Fiyat' in col]
        if price_cols:
            metrics.append(('Fiyat', price_cols[-1]))
        
        if len(metrics) < 3:
            return self._create_empty_chart("Yeterli metrik yok")
        
        fig = go.Figure()
        
        for company in top_companies:
            company_df = df[df['Şirket'] == company]
            
            values = []
            theta = []
            
            for metric_name, metric_col in metrics:
                if metric_col == 'Molekül':
                    # Ürün çeşitliliği
                    value = company_df[metric_col].nunique()
                    max_value = df[metric_col].nunique()
                    normalized = (value / max_value * 100) if max_value > 0 else 0
                elif metric_col == 'Risk_Skoru':
                    # Risk skoru ortalaması
                    value = company_df[metric_col].mean()
                    normalized = value  # Zaten 0-100 arası
                else:
                    # Diğer metrikler
                    value = company_df[metric_col].mean()
                    max_value = df[metric_col].max()
                    normalized = (value / max_value * 100) if max_value > 0 else 0
                
                values.append(normalized)
                theta.append(metric_name)
            
            # Radar grafiği için ilk değeri tekrar ekle
            values.append(values[0])
            theta.append(theta[0])
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=theta,
                    fill='toself',
                    name=company,
                    opacity=0.7
                )
            )
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            title='Şirket Performans Karşılaştırması',
            height=600,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """
        Boş grafik oluştur.
        
        Args:
            message: Gösterilecek mesaj
            
        Returns:
            Boş grafik figürü
        """
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig

# ================================================
# 6. EXECUTIVE DASHBOARD UI
# ================================================

class ExecutiveDashboard:
    """
    Yönetici gösterge paneli UI.
    """
    
    def __init__(self):
        self.data_engine = AdvancedDataEngine()
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.viz_engine = AdvancedVizEngine()
        
        # Session state initialization
        self._init_session_state()
    
    def _init_session_state(self):
        """Session state'i başlat."""
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'market_metrics' not in st.session_state:
            st.session_state.market_metrics = None
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = None
        if 'anomaly_results' not in st.session_state:
            st.session_state.anomaly_results = None
        if 'segmentation_results' not in st.session_state:
            st.session_state.segmentation_results = None
        if 'strategic_insights' not in st.session_state:
            st.session_state.strategic_insights = []
        if 'dashboard_figures' not in st.session_state:
            st.session_state.dashboard_figures = {}
    
    def render_sidebar(self):
        """Yan çubuğu render et."""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(26, 35, 126, 0.8); 
                     border-radius: 10px; margin-bottom: 2rem;">
                <h2 style="color: white; margin: 0;">💊 PharmaIntel Pro</h2>
                <p style="color: #bbdefb; margin: 0.5rem 0 0 0;">v8.0 Enterprise</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Veri Yükleme
            with st.expander("📁 VERİ YÖNETİMİ", expanded=True):
                uploaded_file = st.file_uploader(
                    "Veri Dosyası Yükle",
                    type=['xlsx', 'xls', 'csv'],
                    help="Excel veya CSV formatında veri yükleyin"
                )
                
                if uploaded_file:
                    st.info(f"**Dosya:** {uploaded_file.name}")
                    
                    if st.button("🚀 Veriyi İşle ve Analiz Et", type="primary", use_container_width=True):
                        with st.spinner("Veri işleniyor..."):
                            self._load_and_process_data(uploaded_file)
            
            # Analiz Kontrolleri
            if st.session_state.processed_data is not None:
                with st.expander("🔧 ANALİZ KONTROLLERİ", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🔮 Tahmin Analizi", use_container_width=True):
                            with st.spinner("Tahmin analizi yapılıyor..."):
                                self._run_forecast_analysis()
                    
                    with col2:
                        if st.button("⚠️ Risk Analizi", use_container_width=True):
                            with st.spinner("Risk analizi yapılıyor..."):
                                self._run_anomaly_detection()
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        if st.button("🎯 Segmentasyon", use_container_width=True):
                            with st.spinner("Segmentasyon analizi yapılıyor..."):
                                self._run_segmentation_analysis()
                    
                    with col4:
                        if st.button("📊 Dashboard", use_container_width=True):
                            with st.spinner("Dashboard oluşturuluyor..."):
                                self._create_dashboard()
                    
                    # İleri Analiz Seçenekleri
                    st.markdown("---")
                    st.markdown("**🎯 İleri Analiz**")
                    
                    analysis_type = st.selectbox(
                        "Analiz Türü",
                        ["BCG Portföy Analizi", "Rekabet Analizi", "Fiyat Optimizasyonu", 
                         "Ürün Yaşam Döngüsü", "Regresyon Analizi"]
                    )
                    
                    if st.button("▶️ Analizi Çalıştır", use_container_width=True):
                        st.info(f"{analysis_type} çalıştırılıyor...")
            
            # Raporlama
            with st.expander("📑 RAPORLAMA", expanded=False):
                report_type = st.selectbox(
                    "Rapor Türü",
                    ["Yönetici Özeti", "Detaylı Analiz", "Stratejik Plan", "Risk Raporu"]
                )
                
                if st.button("📥 Rapor Oluştur", use_container_width=True):
                    st.info(f"{report_type} raporu oluşturuluyor...")
            
            # Sistem Durumu
            with st.expander("⚙️ SİSTEM DURUMU", expanded=False):
                if st.session_state.processed_data is not None:
                    data_info = st.session_state.processed_data
                    st.metric("Satır Sayısı", f"{len(data_info):,}")
                    st.metric("Sütun Sayısı", len(data_info.columns))
                    
                    memory_usage = data_info.memory_usage(deep=True).sum() / 1024**2
                    st.metric("Hafıza Kullanımı", f"{memory_usage:.1f} MB")
                else:
                    st.info("Veri yüklenmemiş")
    
    def _load_and_process_data(self, uploaded_file):
        """Veriyi yükle ve işle."""
        try:
            # Veriyi yükle
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, low_memory=False)
            else:
                raw_df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Veriyi işle
            processed_df = self.data_engine.parallel_data_processing(raw_df)
            processed_df = self.data_engine.create_analytical_features(processed_df)
            
            # Metrikleri hesapla
            market_metrics = self.data_engine.calculate_market_concentration(processed_df)
            
            # Session state'i güncelle
            st.session_state.raw_data = raw_df
            st.session_state.processed_data = processed_df
            st.session_state.market_metrics = market_metrics
            
            # İçgörüleri üret
            insights = self.analytics_engine.generate_strategic_insights(
                processed_df, market_metrics
            )
            st.session_state.strategic_insights = insights
            
            st.success(f"✅ Veri işlendi: {len(processed_df):,} satır, {len(processed_df.columns)} sütun")
            
        except Exception as e:
            st.error(f"Veri işleme hatası: {str(e)}")
    
    def _run_forecast_analysis(self):
        """Tahmin analizi çalıştır."""
        if st.session_state.processed_data is None:
            st.warning("Lütfen önce veri yükleyin")
            return
        
        df = st.session_state.processed_data
        
        # Satış sütununu bul
        sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
        if not sales_cols:
            st.warning("Satış verisi bulunamadı")
            return
        
        target_col = sales_cols[-1]
        
        # Tahmin analizi çalıştır
        forecasts = self.analytics_engine.multi_model_forecasting(
            df, target_col, periods=12, ensemble=True
        )
        
        st.session_state.forecast_results = forecasts
        
        if forecasts:
            st.success(f"✅ {len(forecasts)} tahmin modeli oluşturuldu")
        else:
            st.warning("Tahmin analizi başarısız")
    
    def _run_anomaly_detection(self):
        """Anomali tespiti çalıştır."""
        if st.session_state.processed_data is None:
            st.warning("Lütfen önce veri yükleyin")
            return
        
        df = st.session_state.processed_data
        
        # Anomali tespiti çalıştır
        anomaly_df = self.analytics_engine.multi_algorithm_anomaly_detection(
            df, contamination=0.1
        )
        
        st.session_state.anomaly_results = anomaly_df
        
        if anomaly_df is not None:
            critical_risk = len(anomaly_df[anomaly_df['Risk_Seviyesi'] == 'Kritik Risk'])
            high_risk = len(anomaly_df[anomaly_df['Risk_Seviyesi'] == 'Yüksek Risk'])
            
            st.success(f"✅ Risk analizi tamamlandı: {critical_risk} kritik, {high_risk} yüksek riskli ürün")
        else:
            st.warning("Risk analizi başarısız")
    
    def _run_segmentation_analysis(self):
        """Segmentasyon analizi çalıştır."""
        if st.session_state.processed_data is None:
            st.warning("Lütfen önce veri yükleyin")
            return
        
        df = st.session_state.processed_data
        
        # Segmentasyon analizi çalıştır
        segmentation_results = self.analytics_engine.advanced_segmentation_pipeline(
            df, n_clusters=6
        )
        
        st.session_state.segmentation_results = segmentation_results
        
        if segmentation_results['segmented_df'] is not None:
            segments = segmentation_results['segmented_df']['Segment_Adı'].unique()
            silhouette = segmentation_results['clustering_metrics'].get('silhouette_score', 0)
            
            st.success(f"✅ Segmentasyon tamamlandı: {len(segments)} segment, Silhouette: {silhouette:.3f}")
        else:
            st.warning("Segmentasyon analizi başarısız")
    
    def _create_dashboard(self):
        """Dashboard oluştur."""
        if st.session_state.processed_data is None:
            st.warning("Lütfen önce veri yükleyin")
            return
        
        df = st.session_state.processed_data
        metrics = st.session_state.market_metrics
        insights = st.session_state.strategic_insights
        
        # Dashboard oluştur
        dashboard_figures = self.viz_engine.create_executive_dashboard(
            df, metrics, insights
        )
        
        st.session_state.dashboard_figures = dashboard_figures
        
        st.success("✅ Dashboard oluşturuldu")
    
    def render_main_content(self):
        """Ana içeriği render et."""
        # Header
        st.markdown("""
        <div style="padding: 2rem; background: linear-gradient(135deg, #1a237e, #283593); 
                 border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">PharmaIntelligence Pro v8.0</h1>
            <p style="color: #bbdefb; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Kurumsal Karar Destek ve Stratejik İstihbarat Platformu
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Eğer veri yoksa hoşgeldin ekranı
        if st.session_state.processed_data is None:
            self._render_welcome_screen()
            return
        
        # Sekmeler
        tabs = st.tabs([
            "📊 YÖNETİCİ PANELİ",
            "🔮 TAHMİN ANALİZİ",
            "⚠️ RİK ANALİZİ",
            "🎯 SEGMENTASYON",
            "📈 GÖRSELLEŞTİRME",
            "💡 STRATEJİK İÇGÖRÜLER",
            "📑 RAPORLAMA"
        ])
        
        with tabs[0]:
            self._render_executive_dashboard()
        
        with tabs[1]:
            self._render_forecast_analysis()
        
        with tabs[2]:
            self._render_risk_analysis()
        
        with tabs[3]:
            self._render_segmentation_analysis()
        
        with tabs[4]:
            self._render_visualization_gallery()
        
        with tabs[5]:
            self._render_strategic_insights()
        
        with tabs[6]:
            self._render_reporting()
    
    def _render_welcome_screen(self):
        """Hoşgeldin ekranını render et."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: rgba(26, 35, 126, 0.1); 
                     border-radius: 20px; border: 2px dashed #3949ab;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">💊</div>
                <h2 style="color: #1a237e;">PharmaIntelligence Pro v8.0'a Hoş Geldiniz</h2>
                <p style="color: #5c6bc0; margin: 1.5rem 0;">
                Kurumsal karar destek platformumuz, ilaç pazarı analizinde yapay zeka destekli 
                tahminleme, risk analizi ve stratejik içgörüler sunar.
                </p>
                <div style="background: #e8eaf6; padding: 1.5rem; border-radius: 10px; margin-top: 2rem;">
                    <h4 style="color: #1a237e; margin-top: 0;">🚀 Başlamak İçin:</h4>
                    <ol style="text-align: left; color: #5c6bc0;">
                        <li>Sol taraftaki panelden veri dosyanızı yükleyin</li>
                        <li>"Veriyi İşle ve Analiz Et" butonuna tıklayın</li>
                        <li>Analiz sonuçlarını sekme sekmelerinde görüntüleyin</li>
                    </ol>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_executive_dashboard(self):
        """Yönetici panelini render et."""
        st.markdown("### 📊 Yönetici Gösterge Paneli")
        
        if not st.session_state.dashboard_figures:
            st.info("Dashboard oluşturmak için sol panelden 'Dashboard' butonuna tıklayın")
            return
        
        figures = st.session_state.dashboard_figures
        
        # Row 1: Metrikler
        if 'metrics_board' in figures:
            st.plotly_chart(figures['metrics_board'], use_container_width=True)
        
        # Row 2: Pazar Yoğunluğu ve Segmentasyon
        col1, col2 = st.columns(2)
        
        with col1:
            if 'market_concentration' in figures:
                st.plotly_chart(figures['market_concentration'], use_container_width=True)
        
        with col2:
            if 'segmentation_map' in figures:
                st.plotly_chart(figures['segmentation_map'], use_container_width=True)
        
        # Row 3: Zaman Serisi ve Risk
        col3, col4 = st.columns(2)
        
        with col3:
            if 'time_series_forecast' in figures:
                st.plotly_chart(figures['time_series_forecast'], use_container_width=True)
        
        with col4:
            if 'risk_heatmap' in figures:
                st.plotly_chart(figures['risk_heatmap'], use_container_width=True)
        
        # Row 4: Coğrafi Dağılım ve Portföy
        col5, col6 = st.columns(2)
        
        with col5:
            if 'geographic_distribution' in figures:
                st.plotly_chart(figures['geographic_distribution'], use_container_width=True)
        
        with col6:
            if 'portfolio_matrix' in figures:
                st.plotly_chart(figures['portfolio_matrix'], use_container_width=True)
        
        # Row 5: Radar Chart
        if 'company_radar' in figures:
            st.plotly_chart(figures['company_radar'], use_container_width=True)
    
    def _render_forecast_analysis(self):
        """Tahmin analizini render et."""
        st.markdown("### 🔮 Tahmin Analizi")
        
        if st.session_state.forecast_results is None:
            st.info("Tahmin analizi çalıştırmak için sol panelden 'Tahmin Analizi' butonuna tıklayın")
            return
        
        forecasts = st.session_state.forecast_results
        
        # Model performans karşılaştırması
        st.markdown("#### 📈 Model Performans Karşılaştırması")
        
        performance_data = []
        for model_name, forecast in forecasts.items():
            performance_data.append({
                'Model': model_name.upper(),
                'MAPE': f"{forecast.mape:.2f}%" if forecast.mape > 0 else "N/A",
                'RMSE': f"${forecast.rmse/1e6:.2f}M" if forecast.rmse > 0 else "N/A",
                'Trend': forecast.trend_direction,
                'Güven Seviyesi': f"%{forecast.confidence_level * 100:.0f}"
            })
        
        st.table(pd.DataFrame(performance_data))
        
        # Tahmin grafikleri
        st.markdown("#### 📊 Tahmin Grafikleri")
        
        for model_name, forecast in forecasts.items():
            with st.expander(f"{model_name.upper()} Modeli Tahminleri"):
                # Tahmin tablosu
                forecast_df = pd.DataFrame({
                    'Dönem': forecast.periods,
                    'Tahmin': forecast.predictions,
                    'Alt Sınır': forecast.lower_bounds,
                    'Üst Sınır': forecast.upper_bounds
                })
                
                st.dataframe(forecast_df.style.format({
                    'Tahmin': '${:,.0f}',
                    'Alt Sınır': '${:,.0f}',
                    'Üst Sınır': '${:,.0f}'
                }))
                
                # Tahmin grafiği
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=forecast.periods,
                    y=forecast.predictions,
                    mode='lines+markers',
                    name='Tahmin',
                    line=dict(color='blue', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast.periods + forecast.periods[::-1],
                    y=forecast.upper_bounds + forecast.lower_bounds[::-1],
                    fill='toself',
                    fillcolor='rgba(0, 100, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Güven Aralığı'
                ))
                
                fig.update_layout(
                    title=f'{model_name.upper()} Tahminleri',
                    xaxis_title='Dönem',
                    yaxis_title='Değer (USD)',
                    height=400,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Stratejik öneriler
        st.markdown("#### 💡 Stratejik Öneriler")
        
        ensemble_forecast = forecasts.get('ensemble')
        if ensemble_forecast:
            growth_rate = ensemble_forecast.get_growth_rate()
            
            if growth_rate > 15:
                st.success(f"**📈 Olumlu Tahmin:** Pazarın {growth_rate:.1f}% büyümesi bekleniyor. Yatırımları artırmak için uygun zaman.")
            elif growth_rate > 5:
                st.info(f"**📊 Orta Seviye Büyüme:** {growth_rate:.1f}% büyüme bekleniyor. Kontrollü genişleme stratejisi önerilir.")
            else:
                st.warning(f"**⚠️ Düşük Büyüme:** Sadece {growth_rate:.1f}% büyüme bekleniyor. Maliyet optimizasyonuna odaklanın.")
    
    def _render_risk_analysis(self):
        """Risk analizini render et."""
        st.markdown("### ⚠️ Risk Analizi")
        
        if st.session_state.anomaly_results is None:
            st.info("Risk analizi çalıştırmak için sol panelden 'Risk Analizi' butonuna tıklayın")
            return
        
        anomaly_df = st.session_state.anomaly_results
        
        # Risk özeti
        st.markdown("#### 📊 Risk Özeti")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            critical = len(anomaly_df[anomaly_df['Risk_Seviyesi'] == 'Kritik Risk'])
            st.metric("Kritik Risk", critical, delta_color="inverse")
        
        with col2:
            high = len(anomaly_df[anomaly_df['Risk_Seviyesi'] == 'Yüksek Risk'])
            st.metric("Yüksek Risk", high, delta_color="inverse")
        
        with col3:
            medium = len(anomaly_df[anomaly_df['Risk_Seviyesi'] == 'Orta Risk'])
            st.metric("Orta Risk", medium)
        
        with col4:
            low = len(anomaly_df[anomaly_df['Risk_Seviyesi'] == 'Düşük Risk'])
            st.metric("Düşük Risk", low)
        
        with col5:
            safe = len(anomaly_df[anomaly_df['Risk_Seviyesi'] == 'Normal'])
            st.metric("Güvenli", safe)
        
        # Kritik riskli ürünler
        st.markdown("#### 🚨 Kritik Riskli Ürünler")
        
        critical_products = anomaly_df[anomaly_df['Risk_Seviyesi'] == 'Kritik Risk']
        
        if len(critical_products) > 0:
            # Görüntülenecek sütunlar
            display_cols = []
            
            if 'Molekul' in critical_products.columns:
                display_cols.append('Molekul')
            if 'Şirket' in critical_products.columns:
                display_cols.append('Şirket')
            if 'Ülke' in critical_products.columns:
                display_cols.append('Ülke')
            
            display_cols.extend(['Anomali_Tipi', 'Anomali_Skoru', 'Şüphe_İndeksi'])
            
            # Satış sütununu ekle
            sales_cols = [col for col in critical_products.columns if re.search(r'Satış_20\d{2}', col)]
            if sales_cols:
                display_cols.append(sales_cols[-1])
            
            st.dataframe(
                critical_products[display_cols].sort_values('Anomali_Skoru').head(20),
                use_container_width=True
            )
            
            # Risk haritası
            st.markdown("#### 🗺️ Risk Dağılım Haritası")
            
            fig = px.scatter(
                critical_products,
                x=sales_cols[-1] if sales_cols else 'Anomali_Skoru',
                y='Şüphe_İndeksi',
                color='Anomali_Tipi',
                size='Anomali_Skoru',
                hover_name='Molekul' if 'Molekul' in critical_products.columns else None,
                title='Kritik Riskli Ürünler Haritası'
            )
            
            fig.update_layout(height=500, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ Kritik riskli ürün tespit edilmedi")
        
        # Risk azaltma önerileri
        st.markdown("#### 🛡️ Risk Azaltma Önerileri")
        
        if len(critical_products) > 0:
            st.warning("""
            **Acil Eylem Gerektiren Durumlar:**
            
            1. **Fiyat İstikrarsızlığı:** Aşırı fiyat oynaklığı gösteren ürünler için fiyatlandırma politikalarını gözden geçirin.
            2. **Anormal Büyüme:** Beklenmedik büyüme gösteren ürünler için pazar araştırması yapın.
            3. **Satış Dalgalanmaları:** Düzensiz satış paternleri için stok yönetimini optimize edin.
            """)
        else:
            st.info("""
            **Risk Yönetimi İyi Uygulamaları:**
            
            1. Düzenli risk izleme ve raporlama
            2. Erken uyarı sistemleri kurma
            3. Risk dağılımını çeşitlendirme
            4. Senaryo analizleri yapma
            """)
    
    def _render_segmentation_analysis(self):
        """Segmentasyon analizini render et."""
        st.markdown("### 🎯 Segmentasyon Analizi")
        
        if st.session_state.segmentation_results is None:
            st.info("Segmentasyon analizi çalıştırmak için sol panelden 'Segmentasyon' butonuna tıklayın")
            return
        
        results = st.session_state.segmentation_results
        
        if results['segmented_df'] is None:
            st.warning("Segmentasyon analizi başarısız")
            return
        
        segmented_df = results['segmented_df']
        metrics = results['clustering_metrics']
        profiles = results['segment_profiles']
        
        # Segment dağılımı
        st.markdown("#### 📊 Segment Dağılımı")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            silhouette = metrics.get('silhouette_score', 0)
            st.metric("Silhouette Skoru", f"{silhouette:.3f}")
        
        with col2:
            segments = segmented_df['Segment_Adı'].unique()
            st.metric("Segment Sayısı", len(segments))
        
        with col3:
            separation = metrics.get('cluster_separation', 0)
            st.metric("Küme Ayrımı", f"{separation:.3f}")
        
        # Segment profilleri
        st.markdown("#### 👥 Segment Profilleri")
        
        for segment_name, profile in profiles.items():
            with st.expander(f"{segment_name} - {profile['urun_sayisi']} ürün"):
                cols = st.columns(4)
                
                with cols[0]:
                    st.metric("Ürün Sayısı", profile['urun_sayisi'])
                
                with cols[1]:
                    st.metric("Ort. Satış", f"${profile['ortalama_satis']/1e6:.2f}M")
                
                with cols[2]:
                    st.metric("Ort. Büyüme", f"%{profile['ortalama_buyume']:.1f}")
                
                with cols[3]:
                    st.metric("Ürün Çeşitliliği", profile['urun_cesitliligi'])
                
                # Dominant şirketler ve ülkeler
                if profile['dominant_sirketler']:
                    st.write("**🏢 Dominant Şirketler:**")
                    for company, count in list(profile['dominant_sirketler'].items())[:3]:
                        st.write(f"- {company}: {count} ürün")
                
                if profile['dominant_ulkeler']:
                    st.write("**🌍 Dominant Ülkeler:**")
                    for country, count in list(profile['dominant_ulkeler'].items())[:3]:
                        st.write(f"- {country}: {count} ürün")
        
        # Segmentasyon haritası
        st.markdown("#### 🗺️ Segmentasyon Haritası")
        
        if 'visualization_data' in results and results['visualization_data'].get('pca_2d') is not None:
            viz_data = results['visualization_data']
            
            # PCA haritası
            fig = go.Figure()
            
            for segment in segmented_df['Segment_Adı'].unique():
                segment_mask = segmented_df['Segment_Adı'] == segment
                
                fig.add_trace(go.Scatter(
                    x=viz_data['pca_2d'][segment_mask, 0],
                    y=viz_data['pca_2d'][segment_mask, 1],
                    mode='markers',
                    name=segment,
                    marker=dict(size=10, opacity=0.7)
                ))
            
            fig.update_layout(
                title='PCA Segmentasyon Haritası',
                xaxis_title='PCA Bileşen 1',
                yaxis_title='PCA Bileşen 2',
                height=500,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment stratejileri
        st.markdown("#### 🎯 Segment Stratejileri")
        
        strategy_recommendations = {
            'Yıldız Ürünler': 'Yatırımı artırın ve pazar liderliğini pekiştirin',
            'Nakit İnekleri': 'Kar marjını koruyun ve nakit akışını yönetin',
            'Soru İşaretleri': 'Pazar potansiyelini araştırın, başarılı olanlara yatırım yapın',
            'Zayıf Ürünler': 'Değerlendirme yapın, gerekirse portföyden çıkarın',
            'Yükselen Yıldızlar': 'Büyümeyi destekleyin, kaynak ayırın',
            'Olgun Ürünler': 'Verimliliği artırın, maliyetleri optimize edin'
        }
        
        for segment in segments:
            if segment in strategy_recommendations:
                with st.expander(f"📋 {segment} Stratejisi"):
                    st.info(strategy_recommendations[segment])
    
    def _render_visualization_gallery(self):
        """Görselleştirme galerisini render et."""
        st.markdown("### 📈 Görselleştirme Galerisi")
        
        if st.session_state.processed_data is None:
            st.info("Görselleştirmeler için önce veri yükleyin")
            return
        
        df = st.session_state.processed_data
        
        # Görselleştirme seçenekleri
        viz_options = st.multiselect(
            "Görselleştirmeleri Seçin",
            [
                "Korelasyon Matrisi",
                "Dağılım Grafikleri", 
                "Kutu Grafikleri",
                "Yoğunluk Grafikleri",
                "Zaman Serisi Analizi",
                "Coğrafi Analiz",
                "Karşılaştırmalı Analiz"
            ],
            default=["Korelasyon Matrisi", "Dağılım Grafikleri"]
        )
        
        # Korelasyon Matrisi
        if "Korelasyon Matrisi" in viz_options:
            st.markdown("#### 🔗 Korelasyon Matrisi")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # En önemli 10 sütun
                important_cols = []
                
                # Satış sütunları
                sales_cols = [col for col in numeric_cols if re.search(r'Satış_20\d{2}', col)]
                if sales_cols:
                    important_cols.append(sales_cols[-1])
                
                # Büyüme sütunları
                growth_cols = [col for col in numeric_cols if 'Buyume' in col]
                if growth_cols:
                    important_cols.append(growth_cols[-1])
                
                # Diğer önemli sütunlar
                other_important = ['Pazar_Payi', 'CAGR', 'Risk_Indeksi', 'Performans_Indeksi']
                for col in other_important:
                    if col in numeric_cols and col not in important_cols:
                        important_cols.append(col)
                
                # Varyansı yüksek sütunlar ekle
                if len(important_cols) < 10:
                    remaining_cols = [col for col in numeric_cols if col not in important_cols]
                    if remaining_cols:
                        variances = df[remaining_cols].var()
                        top_variance = variances.nlargest(10 - len(important_cols))
                        important_cols.extend(top_variance.index.tolist())
                
                if len(important_cols) >= 2:
                    corr_matrix = df[important_cols].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu',
                        title='Korelasyon Matrisi',
                        width=800,
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Dağılım Grafikleri
        if "Dağılım Grafikleri" in viz_options:
            st.markdown("#### 📊 Dağılım Grafikleri")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox(
                    "X Ekseni",
                    df.select_dtypes(include=[np.number]).columns.tolist(),
                    key='dist_x'
                )
            
            with col2:
                y_axis = st.selectbox(
                    "Y Ekseni", 
                    df.select_dtypes(include=[np.number]).columns.tolist(),
                    key='dist_y'
                )
            
            if x_axis and y_axis:
                color_by = st.selectbox(
                    "Renk Sınıflandırması",
                    ['None'] + df.select_dtypes(include=['object', 'category']).columns.tolist(),
                    key='dist_color'
                )
                
                if color_by == 'None':
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=f'{x_axis} vs {y_axis}')
                else:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, 
                                   title=f'{x_axis} vs {y_axis} - {color_by} Bazında')
                
                fig.update_layout(height=500, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_strategic_insights(self):
        """Stratejik içgörüleri render et."""
        st.markdown("### 💡 Stratejik İçgörüler")
        
        if not st.session_state.strategic_insights:
            st.info("İçgörüleri görüntülemek için önce veri analizini çalıştırın")
            return
        
        insights = st.session_state.strategic_insights
        
        # İçgörü istatistikleri
        st.markdown("#### 📊 İçgörü Özeti")
        
        insight_types = {}
        for insight in insights:
            insight_type = insight.get('type', 'other')
            insight_types[insight_type] = insight_types.get(insight_type, 0) + 1
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Toplam İçgörü", len(insights))
        
        with col2:
            st.metric("Kritik Öncelikli", len([i for i in insights if i.get('priority') == 'critical']))
        
        with col3:
            st.metric("Yüksek Öncelikli", len([i for i in insights if i.get('priority') == 'high']))
        
        with col4:
            st.metric("Stratejik Etki", len([i for i in insights if i.get('impact') == 'strategic']))
        
        # İçgörü kartları
        st.markdown("#### 🎯 Detaylı İçgörüler")
        
        # Öncelik sırasına göre sırala
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_insights = sorted(insights, key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
        
        for insight in sorted_insights:
            # İkon belirle
            icons = {
                'market_structure': '🏢',
                'growth': '📈',
                'pricing': '💰',
                'portfolio': '📊',
                'risk': '⚠️',
                'geographic': '🌍',
                'innovation': '🔬'
            }
            
            icon = icons.get(insight['type'], '💡')
            
            # Renk belirle
            colors = {
                'critical': '#f44336',
                'high': '#ff9800',
                'medium': '#ffeb3b',
                'low': '#4caf50'
            }
            
            color = colors.get(insight.get('priority', 'low'), '#4caf50')
            
            # İçgörü kartı
            st.markdown(f"""
            <div style="
                background: rgba(26, 35, 126, 0.1);
                border-left: 5px solid {color};
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                    <h4 style="margin: 0; color: #1a237e;">{insight['title']}</h4>
                    <span style="margin-left: auto; background: {color}; color: white; 
                           padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem;">
                        {insight.get('priority', 'medium').upper()}
                    </span>
                </div>
                <p style="color: #5c6bc0; margin: 0.5rem 0 1rem 0;">{insight['description']}</p>
                <div style="background: rgba(255, 255, 255, 0.5); padding: 1rem; border-radius: 5px;">
                    <strong style="color: #1a237e;">🎯 Öneri:</strong> {insight['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Eylem planı özeti
        st.markdown("#### 📋 Özet Eylem Planı")
        
        critical_actions = [i for i in insights if i.get('priority') in ['critical', 'high']]
        
        if critical_actions:
            st.warning("""
            **🔄 Acil Eylem Gerektiren Alanlar:**
            
            1. **Kritik Risk Yönetimi:** Riskli ürünler için acil müdahale planları geliştirin
            2. **Pazar Yapısı Analizi:** Rekabet stratejilerini gözden geçirin
            3. **Büyüme Odaklı Yatırım:** Yüksek büyüme potansiyelli alanlara kaynak ayırın
            4. **Fiyatlandırma Stratejisi:** Fiyat oynaklığını azaltacak politikalar uygulayın
            """)
        
        # İndirme seçenekleri
        st.markdown("#### 📥 İçgörü Raporu")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Excel Raporu Oluştur", use_container_width=True):
                self._generate_insights_excel(insights)
        
        with col2:
            if st.button("📄 PDF Özet Oluştur", use_container_width=True):
                self._generate_insights_pdf(insights)
    
    def _generate_insights_excel(self, insights):
        """İçgörü Excel raporu oluştur."""
        try:
            # Excel dosyası oluştur
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # İçgörü sayfası
                insights_df = pd.DataFrame(insights)
                insights_df.to_excel(writer, sheet_name='Stratejik İçgörüler', index=False)
                
                # Özet sayfası
                summary_data = {
                    'Metrik': ['Toplam İçgörü', 'Kritik Öncelikli', 'Yüksek Öncelikli', 
                              'Orta Öncelikli', 'Stratejik Etkili', 'Operasyonel Etkili'],
                    'Değer': [
                        len(insights),
                        len([i for i in insights if i.get('priority') == 'critical']),
                        len([i for i in insights if i.get('priority') == 'high']),
                        len([i for i in insights if i.get('priority') == 'medium']),
                        len([i for i in insights if i.get('impact') == 'strategic']),
                        len([i for i in insights if i.get('impact') == 'operational'])
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Özet', index=False)
                
                # Eylem planı sayfası
                action_plan = []
                for i, insight in enumerate(insights[:10], 1):
                    action_plan.append({
                        'Sıra': i,
                        'İçgörü': insight['title'],
                        'Öncelik': insight.get('priority', 'medium'),
                        'Eylem': insight['recommendation'],
                        'Sorumlu': 'İlgili Departman',
                        'Tamamlanma': '-'
                    })
                
                action_df = pd.DataFrame(action_plan)
                action_df.to_excel(writer, sheet_name='Eylem Planı', index=False)
            
            output.seek(0)
            
            # İndirme butonu
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.download_button(
                label="📥 İçgörü Raporunu İndir",
                data=output,
                file_name=f"pharma_insights_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            st.success("✅ Excel raporu oluşturuldu")
            
        except Exception as e:
            st.error(f"Excel raporu oluşturma hatası: {str(e)}")
    
    def _generate_insights_pdf(self, insights):
        """İçgörü PDF raporu oluştur."""
        if not REPORTLAB_AVAILABLE:
            st.warning("PDF oluşturmak için reportlab kurulu değil")
            return
        
        try:
            # PDF oluştur
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            
            story = []
            styles = getSampleStyleSheet()
            
            # Başlık
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1a237e'),
                spaceAfter=30
            )
            
            story.append(Paragraph("PharmaIntelligence Pro - Stratejik İçgörü Raporu", title_style))
            story.append(Paragraph(f"Oluşturulma Tarihi: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Özet
            story.append(Paragraph("Özet", styles['Heading2']))
            
            summary_data = [
                ['Metrik', 'Değer'],
                ['Toplam İçgörü', str(len(insights))],
                ['Kritik Öncelikli', str(len([i for i in insights if i.get('priority') == 'critical']))],
                ['Yüksek Öncelikli', str(len([i for i in insights if i.get('priority') == 'high']))],
                ['Stratejik Etkili', str(len([i for i in insights if i.get('impact') == 'strategic']))]
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e8eaf6')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 30))
            
            # İçgörüler
            story.append(Paragraph("Detaylı İçgörüler", styles['Heading2']))
            
            for i, insight in enumerate(insights[:10], 1):
                # İçgörü başlığı
                insight_title = f"{i}. {insight['title']}"
                story.append(Paragraph(insight_title, styles['Heading3']))
                
                # Açıklama
                story.append(Paragraph(insight['description'], styles['Normal']))
                
                # Öneri
                story.append(Paragraph(f"<b>Öneri:</b> {insight['recommendation']}", styles['Normal']))
                
                # Öncelik
                priority_color = {
                    'critical': '#f44336',
                    'high': '#ff9800',
                    'medium': '#ffeb3b',
                    'low': '#4caf50'
                }.get(insight.get('priority', 'medium'), '#4caf50')
                
                story.append(Paragraph(
                    f"<b>Öncelik:</b> <font color='{priority_color}'>{insight.get('priority', 'medium').upper()}</font>",
                    styles['Normal']
                ))
                
                story.append(Spacer(1, 15))
            
            # PDF'yi oluştur
            doc.build(story)
            buffer.seek(0)
            
            # İndirme butonu
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.download_button(
                label="📥 PDF Raporunu İndir",
                data=buffer,
                file_name=f"pharma_insights_{timestamp}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            st.success("✅ PDF raporu oluşturuldu")
            
        except Exception as e:
            st.error(f"PDF raporu oluşturma hatası: {str(e)}")
    
    def _render_reporting(self):
        """Raporlama bölümünü render et."""
        st.markdown("### 📑 Raporlama ve Dışa Aktarım")
        
        if st.session_state.processed_data is None:
            st.info("Rapor oluşturmak için önce veri yükleyin")
            return
        
        df = st.session_state.processed_data
        
        # Rapor türleri
        st.markdown("#### 📊 Rapor Türleri")
        
        report_type = st.selectbox(
            "Rapor Türü Seçin",
            [
                "Yönetici Özet Raporu",
                "Detaylı Analiz Raporu",
                "Risk ve Uyum Raporu",
                "Stratejik Planlama Raporu",
                "Portföy Analiz Raporu",
                "Rekabet Analizi Raporu"
            ]
        )
        
        # Rapor seçenekleri
        st.markdown("#### ⚙️ Rapor Ayarları")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_charts = st.checkbox("📈 Grafikleri Ekle", value=True)
            include_tables = st.checkbox("📋 Tabloları Ekle", value=True)
        
        with col2:
            include_insights = st.checkbox("💡 İçgörüleri Ekle", value=True)
            include_recommendations = st.checkbox("🎯 Önerileri Ekle", value=True)
        
        # Format seçimi
        export_format = st.radio(
            "Dışa Aktarma Formatı",
            ["Excel (.xlsx)", "CSV (.csv)", "PDF (.pdf)", "HTML (.html)"],
            horizontal=True
        )
        
        # Rapor oluşturma
        if st.button("🚀 Rapor Oluştur", type="primary", use_container_width=True):
            with st.spinner(f"{report_type} oluşturuluyor..."):
                self._generate_report(
                    df, report_type, export_format,
                    include_charts, include_tables,
                    include_insights, include_recommendations
                )
        
        # Hızlı istatistikler
        st.markdown("#### 📈 Hızlı İstatistikler")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Toplam Satır", f"{len(df):,}")
        
        with stats_col2:
            st.metric("Sütun Sayısı", len(df.columns))
        
        with stats_col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Sayısal Sütunlar", numeric_cols)
        
        with stats_col4:
            memory = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Hafıza Kullanımı", f"{memory:.1f} MB")
        
        # Veri önizleme
        st.markdown("#### 👁️ Veri Önizleme")
        
        preview_rows = st.slider("Gösterilecek Satır Sayısı", 10, 5000, 100, 10)
        
        # Önemli sütunları otomatik seç
        important_cols = []
        
        for col in ['Molekul', 'Şirket', 'Ülke', 'Segment_Adı', 'Risk_Seviyesi']:
            if col in df.columns:
                important_cols.append(col)
        
        # Satış sütunları
        sales_cols = [col for col in df.columns if re.search(r'Satış_20\d{2}', col)]
        if sales_cols:
            important_cols.append(sales_cols[-1])
        
        # Büyüme sütunları
        growth_cols = [col for col in df.columns if 'Buyume' in col]
        if growth_cols:
            important_cols.append(growth_cols[-1])
        
        # Sütun seçimi
        selected_cols = st.multiselect(
            "Gösterilecek Sütunlar",
            df.columns.tolist(),
            default=important_cols[:min(8, len(important_cols))]
        )
        
        if selected_cols:
            display_df = df[selected_cols].head(preview_rows)
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.dataframe(df.head(preview_rows), use_container_width=True, height=400)
    
    def _generate_report(self, df, report_type, export_format, include_charts, 
                        include_tables, include_insights, include_recommendations):
        """Rapor oluştur."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if 'Excel' in export_format:
                # Excel raporu
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Ana veri sayfası
                    df.to_excel(writer, sheet_name='Ham Veri', index=False)
                    
                    # Özet sayfası
                    summary_data = self._create_summary_data(df)
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Özet', index=False)
                    
                    # İçgörü sayfası
                    if include_insights and st.session_state.strategic_insights:
                        insights_df = pd.DataFrame(st.session_state.strategic_insights)
                        insights_df.to_excel(writer, sheet_name='Stratejik İçgörüler', index=False)
                
                output.seek(0)
                
                st.download_button(
                    label="📥 Excel Raporunu İndir",
                    data=output,
                    file_name=f"pharma_report_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
            elif 'CSV' in export_format:
                # CSV raporu
                csv_data = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 CSV Verisini İndir",
                    data=csv_data,
                    file_name=f"pharma_data_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.success(f"✅ {report_type} oluşturuldu")
            
        except Exception as e:
            st.error(f"Rapor oluşturma hatası: {str(e)}")
    
    def _create_summary_data(self, df):
        """Özet verisi oluştur."""
        summary = []
        
        # Temel istatistikler
        summary.append(['Toplam Satır', len(df)])
        summary.append(['Toplam Sütun', len(df.columns)])
        
        # Sayısal sütunlar
        numeric_cols = df.select_dtypes(include=[np.number])
        summary.append(['Sayısal Sütunlar', len(numeric_cols.columns)])
        
        # Kategorik sütunlar
        categorical_cols = df.select_dtypes(include=['object', 'category'])
        summary.append(['Kategorik Sütunlar', len(categorical_cols.columns)])
        
        # Eksik veri yüzdesi
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        summary.append(['Eksik Veri Yüzdesi', f'{missing_percentage:.2f}%'])
        
        # Hafıza kullanımı
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        summary.append(['Hafıza Kullanımı', f'{memory_usage:.2f} MB'])
        
        # Benzersiz değerler
        if 'Molekül' in df.columns:
            summary.append(['Benzersiz Moleküller', df['Molekül'].nunique()])
        
        if 'Şirket' in df.columns:
            summary.append(['Benzersiz Şirketler', df['Şirket'].nunique()])
        
        if 'Ülke' in df.columns:
            summary.append(['Benzersiz Ülkeler', df['Ülke'].nunique()])
        
        return summary

# ================================================
# 7. MAIN APPLICATION
# ================================================

def main():
    """Ana uygulama fonksiyonu."""
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="PharmaIntelligence Pro v8.0 | Enterprise Decision Support",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://pharmaintelligence.com/enterprise-support',
            'Report a bug': 'https://pharmaintelligence.com/enterprise-bug-report',
            'About': '''
            PharmaIntelligence Pro v8.0 - Enterprise Decision Support Platform
            © 2024 PharmaIntelligence Inc. All Rights Reserved.
            
            Advanced Features:
            • AI-Powered Predictive Analytics
            • Multi-Algorithm Risk Detection
            • PCA + UMAP + t-SNE Segmentation
            • Prophet & ARIMA Time Series
            • SHAP Explainable AI
            • Executive Dashboard with 3D Visualizations
            '''
        }
    )
    
    # Özel CSS
    EXECUTIVE_CSS = """
    <style>
        /* Ana tema değişkenleri */
        :root {
            --primary-dark: #0c1a32;
            --secondary-dark: #14274e;
            --accent-gold: #d4af37;
            --accent-silver: #c0c0c0;
            --accent-blue: #2d7dd2;
            --success: #2dd2a3;
            --warning: #f2c94c;
            --danger: #eb5757;
            --info: #2d7dd2;
        }
        
        /* Ana arkaplan */
        .stApp {
            background: linear-gradient(135deg, var(--primary-dark), var(--secondary-dark));
            background-attachment: fixed;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #f8fafc;
        }
        
        /* Cam efekti kartları */
        .glass-card {
            background: rgba(30, 58, 95, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(212, 175, 55, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            padding: 2rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent-gold), var(--accent-silver));
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6);
            border-color: var(--accent-gold);
        }
        
        /* Başlıklar */
        .executive-title {
            font-size: 3.5rem;
            background: linear-gradient(135deg, var(--accent-gold), var(--accent-silver), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 900;
            letter-spacing: -1px;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .executive-subtitle {
            font-size: 1.3rem;
            color: #bbdefb;
            font-weight: 300;
            margin-bottom: 2rem;
            max-width: 800px;
            line-height: 1.6;
        }
        
        /* Bölüm başlıkları */
        .section-title {
            font-size: 2rem;
            color: var(--accent-gold);
            font-weight: 700;
            margin: 2.5rem 0 1.5rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid rgba(212, 175, 55, 0.3);
        }
        
        /* Metrik kartları */
        .metric-card {
            background: rgba(41, 53, 146, 0.3);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .metric-card:hover {
            background: rgba(41, 53, 146, 0.5);
            transform: translateY(-3px);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: white;
            margin: 0.5rem 0;
            line-height: 1;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #bbdefb;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }
        
        .metric-trend {
            font-size: 0.8rem;
            color: #90a4ae;
            margin-top: 0.5rem;
        }
        
        /* İçgörü kartları */
        .insight-card {
            background: rgba(30, 58, 95, 0.6);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 5px solid;
            transition: all 0.2s ease;
        }
        
        .insight-card:hover {
            background: rgba(30, 58, 95, 0.8);
            transform: translateX(5px);
        }
        
        .insight-card.critical {
            border-left-color: var(--danger);
        }
        
        .insight-card.high {
            border-left-color: var(--warning);
        }
        
        .insight-card.medium {
            border-left-color: var(--info);
        }
        
        .insight-card.low {
            border-left-color: var(--success);
        }
        
        /* Butonlar */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-blue), #4a9fe3);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #4a9fe3, var(--accent-blue));
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(45, 125, 210, 0.4);
        }
        
        /* Sekmeler */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: rgba(30, 58, 95, 0.5);
            padding: 5px;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 8px;
            color: #bbdefb;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(212, 175, 55, 0.2);
            color: var(--accent-gold);
        }
        
        /* Progress bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--accent-gold), var(--accent-silver));
        }
        
        /* Tooltip */
        .stTooltip {
            background: rgba(30, 58, 95, 0.9);
            border: 1px solid var(--accent-gold);
            border-radius: 8px;
        }
        
        /* Dataframe stilleri */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--primary-dark), var(--secondary-dark));
        }
        
        /* Animasyonlar */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
            animation: fadeIn 0.6s ease-out;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(212, 175, 55, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(212, 175, 55, 0); }
            100% { box-shadow: 0 0 0 0 rgba(212, 175, 55, 0); }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
    </style>
    """
    
    st.markdown(EXECUTIVE_CSS, unsafe_allow_html=True)
    
    # Dashboard'u başlat
    dashboard = ExecutiveDashboard()
    
    # UI render
    dashboard.render_sidebar()
    dashboard.render_main_content()

# ================================================
# 8. APPLICATION ENTRY POINT
# ================================================

if __name__ == "__main__":
    try:
        # Garbage collection'ı etkinleştir
        gc.enable()
        
        # Uygulamayı başlat
        main()
        
    except Exception as e:
        # Hata yönetimi
        st.error("""
        ## ⚠️ Kritik Uygulama Hatası
        
        PharmaIntelligence Pro v8.0'da beklenmeyen bir hata oluştu.
        """)
        
        st.error(f"**Hata Detayı:** {str(e)}")
        
        # Hata ayıklama bilgisi
        with st.expander("🔍 Hata Ayıklama Detayları"):
            st.code(traceback.format_exc())
        
        # Kurtarma seçenekleri
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Uygulamayı Yeniden Başlat", use_container_width=True):
                # Session state'i temizle
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📋 Hata Raporu Oluştur", use_container_width=True):
                error_report = {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'session_keys': list(st.session_state.keys())
                }
                
                st.download_button(
                    label="📥 Hata Raporunu İndir",
                    data=json.dumps(error_report, indent=2),
                    file_name=f"pharma_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )]



