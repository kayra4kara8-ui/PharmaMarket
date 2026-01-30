"""
PharmaIntelligence Enterprise v4.0
Profesyonel İlaç Pazarı Analytics Platformu
"""

# ================================================
# IMPORTS AND CONFIGURATION
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import base64
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Advanced Analytics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats

# ================================================
# ENUMS AND DATA CLASSES
# ================================================

from enum import Enum
from dataclasses import dataclass, field

class ProductType(Enum):
    INTERNATIONAL = "international"
    LOCAL = "local"
    REGIONAL = "regional"
    GLOBAL = "global"

class MarketSegment(Enum):
    PREMIUM = "premium"
    STANDARD = "standard"
    ECONOMY = "economy"
    NICHE = "niche"

class GrowthCategory(Enum):
    HIGH_GROWTH = "high_growth"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"

@dataclass
class ProductMetrics:
    """Ürün metrikleri için data class"""
    sales_2024: float = 0.0
    sales_2023: float = 0.0
    growth_rate: float = 0.0
    market_share: float = 0.0
    price: float = 0.0
    units: int = 0
    performance_score: float = 0.0
    risk_score: float = 0.0
    
@dataclass
class MarketInsight:
    """Pazar içgörüleri"""
    title: str
    description: str
    insight_type: str
    severity: str
    confidence: float
    recommendations: List[str]
    affected_products: List[str] = field(default_factory=list)

# ================================================
# APPLICATION CONFIGURATION
# ================================================

class AppConfig:
    """Uygulama konfigürasyonu"""
    
    # Page configuration
    PAGE_CONFIG = {
        'page_title': 'PharmaIntelligence Enterprise',
        'page_icon': '💊',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # Color schemes
    COLORS = {
        'primary': '#2563eb',
        'primary_dark': '#1d4ed8',
        'primary_light': '#3b82f6',
        'secondary': '#0ea5e9',
        'accent': '#06b6d4',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#8b5cf6',
        'dark': '#1e293b',
        'darker': '#0f172a',
        'light': '#64748b',
        'lighter': '#94a3b8'
    }
    
    # Chart colors
    CHART_COLORS = [
        '#2563eb', '#0ea5e9', '#06b6d4', '#10b981',
        '#84cc16', '#f59e0b', '#f97316', '#ef4444',
        '#8b5cf6', '#d946ef', '#ec4899', '#6366f1'
    ]

# ================================================
# PROFESSIONAL CSS STYLING
# ================================================

class ProfessionalStyler:
    """Profesyonel CSS stilleri"""
    
    @staticmethod
    def get_css():
        """CSS stillerini döndür"""
        return f"""
        <style>
        /* Reset and Base */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        /* Application Styling */
        .stApp {{
            background: linear-gradient(135deg, {AppConfig.COLORS['darker']}, {AppConfig.COLORS['dark']});
            color: #f8fafc;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            min-height: 100vh;
        }}
        
        /* Main Header */
        .main-header {{
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(14, 165, 233, 0.05));
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
        }}
        
        .main-title {{
            font-size: 2.8rem;
            font-weight: 900;
            background: linear-gradient(90deg, {AppConfig.COLORS['primary']}, {AppConfig.COLORS['secondary']}, {AppConfig.COLORS['accent']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }}
        
        .main-subtitle {{
            font-size: 1.1rem;
            color: {AppConfig.COLORS['lighter']};
            font-weight: 400;
            max-width: 800px;
            line-height: 1.6;
        }}
        
        /* Section Headers */
        .section-header {{
            font-size: 1.8rem;
            font-weight: 800;
            color: #f1f5f9;
            margin: 2.5rem 0 1.5rem 0;
            padding-left: 1rem;
            border-left: 5px solid {AppConfig.COLORS['primary']};
            background: linear-gradient(90deg, rgba(37, 99, 235, 0.1), transparent);
            padding: 1rem;
            border-radius: 8px;
        }}
        
        .subsection-header {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #e2e8f0;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(148, 163, 184, 0.2);
        }}
        
        /* Metric Cards */
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        @media (max-width: 1200px) {{
            .metric-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
        
        @media (max-width: 768px) {{
            .metric-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .metric-card {{
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, {AppConfig.COLORS['primary']}, {AppConfig.COLORS['accent']});
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            border-color: {AppConfig.COLORS['primary']};
            box-shadow: 0 8px 30px rgba(37, 99, 235, 0.2);
        }}
        
        .metric-card.primary {{
            background: linear-gradient(135deg, {AppConfig.COLORS['primary']}, {AppConfig.COLORS['primary_dark']});
        }}
        
        .metric-card.success {{
            background: linear-gradient(135deg, {AppConfig.COLORS['success']}, #059669);
        }}
        
        .metric-card.warning {{
            background: linear-gradient(135deg, {AppConfig.COLORS['warning']}, #d97706);
        }}
        
        .metric-card.danger {{
            background: linear-gradient(135deg, {AppConfig.COLORS['danger']}, #dc2626);
        }}
        
        .metric-card.info {{
            background: linear-gradient(135deg, {AppConfig.COLORS['info']}, #7c3aed);
        }}
        
        .metric-value {{
            font-size: 2.2rem;
            font-weight: 900;
            color: #f8fafc;
            margin: 0.5rem 0;
            line-height: 1;
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: {AppConfig.COLORS['lighter']};
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        .metric-trend {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.5rem;
            font-size: 0.8rem;
        }}
        
        .trend-up {{ color: {AppConfig.COLORS['success']}; }}
        .trend-down {{ color: {AppConfig.COLORS['danger']}; }}
        .trend-neutral {{ color: {AppConfig.COLORS['light']}; }}
        
        /* Insight Cards */
        .insight-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        
        .insight-card {{
            background: rgba(30, 41, 59, 0.8);
            border-radius: 10px;
            padding: 1.5rem;
            border-left: 5px solid;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }}
        
        .insight-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        }}
        
        .insight-card.info {{ border-left-color: {AppConfig.COLORS['primary']}; }}
        .insight-card.success {{ border-left-color: {AppConfig.COLORS['success']}; }}
        .insight-card.warning {{ border-left-color: {AppConfig.COLORS['warning']}; }}
        .insight-card.danger {{ border-left-color: {AppConfig.COLORS['danger']}; }}
        
        .insight-icon {{
            font-size: 2rem;
            margin-bottom: 1rem;
        }}
        
        .insight-title {{
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }}
        
        .insight-content {{
            color: {AppConfig.COLORS['lighter']};
            line-height: 1.6;
            font-size: 0.95rem;
        }}
        
        /* Sidebar Styling */
        .sidebar-section {{
            background: rgba(15, 23, 42, 0.9);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }}
        
        .sidebar-title {{
            font-size: 1.2rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {AppConfig.COLORS['primary']};
        }}
        
        /* Welcome Screen */
        .welcome-container {{
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
            border-radius: 20px;
            padding: 4rem;
            text-align: center;
            margin: 2rem auto;
            max-width: 1000px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}
        
        .welcome-icon {{
            font-size: 5rem;
            background: linear-gradient(135deg, {AppConfig.COLORS['primary']}, {AppConfig.COLORS['accent']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }}
        
        .feature-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin: 3rem 0;
        }}
        
        .feature-card {{
            background: linear-gradient(145deg, rgba(37, 99, 235, 0.1), rgba(6, 182, 212, 0.05));
            padding: 2rem;
            border-radius: 12px;
            border-left: 4px solid;
            transition: all 0.3s;
        }}
        
        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.2);
        }}
        
        .feature-card.blue {{ border-left-color: {AppConfig.COLORS['primary']}; }}
        .feature-card.teal {{ border-left-color: {AppConfig.COLORS['accent']}; }}
        .feature-card.green {{ border-left-color: {AppConfig.COLORS['success']}; }}
        .feature-card.purple {{ border-left-color: {AppConfig.COLORS['info']}; }}
        
        .feature-icon {{
            font-size: 2.5rem;
            margin-bottom: 1rem;
            opacity: 0.9;
        }}
        
        .feature-title {{
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 0.5rem;
            font-size: 1.2rem;
        }}
        
        .feature-description {{
            color: {AppConfig.COLORS['lighter']};
            font-size: 0.95rem;
            line-height: 1.6;
        }}
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(30, 41, 59, 0.5);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {AppConfig.COLORS['primary']};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {AppConfig.COLORS['primary_dark']};
        }}
        </style>
        """
    
    @staticmethod
    def apply_style():
        """CSS stillerini uygula"""
        st.markdown(ProfessionalStyler.get_css(), unsafe_allow_html=True)

# ================================================
# DATA PROCESSING ENGINE
# ================================================

class DataProcessingEngine:
    """Gelişmiş veri işleme motoru"""
    
    def __init__(self):
        self.df = None
        self.original_df = None
        
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
    def load_dataset(file):
        """Veri setini yükle"""
        try:
            start_time = time.time()
            
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file, engine='openpyxl')
            elif file.name.endswith('.xls'):
                df = pd.read_excel(file, engine='xlrd')
            else:
                st.error(f"Desteklenmeyen dosya formatı: {file.name}")
                return None
            
            load_time = time.time() - start_time
            st.success(f"✅ Veri yüklendi: {len(df):,} satır, {len(df.columns)} sütun ({load_time:.2f}s)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            return None
    
    def clean_and_prepare(self, df):
        """Veriyi temizle ve hazırla"""
        try:
            # Sütun isimlerini temizle
            df.columns = self._clean_column_names(df.columns)
            
            # Veri tiplerini optimize et
            df = self._optimize_dtypes(df)
            
            # Eksik değerleri işle
            df = self._handle_missing_values(df)
            
            # Analiz sütunlarını oluştur
            df = self._create_analysis_columns(df)
            
            # International product flag ekle
            df = self._identify_international_products(df)
            
            return df
            
        except Exception as e:
            st.warning(f"Veri hazırlama hatası: {str(e)}")
            return df
    
    def _clean_column_names(self, columns):
        """Sütun isimlerini temizle"""
        cleaned = []
        for col in columns:
            col_str = str(col)
            
            # Türkçe karakter düzeltme
            turkish_chars = {'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                           'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                           'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'}
            
            for tr, en in turkish_chars.items():
                col_str = col_str.replace(tr, en)
            
            # Özel formatları düzelt
            if 'USD' in col_str and 'MNF' in col_str:
                if '2024' in col_str:
                    if 'Units' in col_str:
                        col_str = 'Units_2024'
                    elif 'Avg Price' in col_str:
                        col_str = 'Avg_Price_2024'
                    else:
                        col_str = 'Sales_2024'
                elif '2023' in col_str:
                    if 'Units' in col_str:
                        col_str = 'Units_2023'
                    elif 'Avg Price' in col_str:
                        col_str = 'Avg_Price_2023'
                    else:
                        col_str = 'Sales_2023'
            
            # Temizleme
            col_str = col_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            col_str = ' '.join(col_str.split())
            col_str = col_str.strip()
            
            cleaned.append(col_str)
        
        return cleaned
    
    def _optimize_dtypes(self, df):
        """Veri tiplerini optimize et"""
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:
                    df[col] = df[col].astype('category')
        
        return df
    
    def _handle_missing_values(self, df):
        """Eksik değerleri işle"""
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            
            if missing_pct > 0:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
                elif pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _create_analysis_columns(self, df):
        """Analiz sütunlarını oluştur"""
        try:
            # Büyüme oranları
            if 'Sales_2023' in df.columns and 'Sales_2024' in df.columns:
                df['Growth_2023_2024'] = ((df['Sales_2024'] - df['Sales_2023']) / 
                                         df['Sales_2023'].replace(0, np.nan)) * 100
            
            # Pazar payı
            if 'Sales_2024' in df.columns:
                total_sales = df['Sales_2024'].sum()
                if total_sales > 0:
                    df['Market_Share_2024'] = (df['Sales_2024'] / total_sales) * 100
            
            # Ortalama fiyat
            if 'Sales_2024' in df.columns and 'Units_2024' in df.columns:
                if 'Avg_Price_2024' not in df.columns:
                    df['Avg_Price_2024'] = df['Sales_2024'] / df['Units_2024'].replace(0, np.nan)
            
            return df
            
        except Exception as e:
            st.warning(f"Analiz sütunları oluşturma hatası: {str(e)}")
            return df
    
    def _identify_international_products(self, df):
        """International product'ları tanımla"""
        try:
            if 'Molecule' in df.columns:
                # International product kontrolü
                intl_flags = []
                
                for idx, row in df.iterrows():
                    molecule = row['Molecule']
                    
                    # Aynı molekül için diğer kayıtları bul
                    same_molecule = df[df['Molecule'] == molecule]
                    
                    corp_count = same_molecule['Corporation'].nunique() if 'Corporation' in df.columns else 1
                    country_count = same_molecule['Country'].nunique() if 'Country' in df.columns else 1
                    
                    # International kriteri
                    is_international = (corp_count > 1 or country_count > 1)
                    intl_flags.append(is_international)
                
                df['Is_International'] = intl_flags
            
            return df
            
        except Exception as e:
            st.warning(f"International product tanımlama hatası: {str(e)}")
            return df

# ================================================
# ADVANCED ANALYTICS ENGINE
# ================================================

class AdvancedAnalyticsEngine:
    """Gelişmiş analitik motoru"""
    
    @staticmethod
    def calculate_comprehensive_metrics(df):
        """Kapsamlı pazar metrikleri"""
        metrics = {}
        
        try:
            # Temel metrikler
            metrics['total_products'] = len(df)
            metrics['total_columns'] = len(df.columns)
            
            # Satış metrikleri
            if 'Sales_2024' in df.columns:
                sales_2024 = df['Sales_2024']
                metrics['total_sales_2024'] = sales_2024.sum()
                metrics['avg_sales_2024'] = sales_2024.mean()
                metrics['median_sales_2024'] = sales_2024.median()
                metrics['sales_std_2024'] = sales_2024.std()
            
            # Büyüme metrikleri
            if 'Growth_2023_2024' in df.columns:
                growth = df['Growth_2023_2024']
                metrics['avg_growth'] = growth.mean()
                metrics['growth_std'] = growth.std()
                metrics['positive_growth_count'] = (growth > 0).sum()
                metrics['high_growth_count'] = (growth > 20).sum()
            
            # Pazar konsantrasyonu
            if 'Corporation' in df.columns and 'Sales_2024' in df.columns:
                corp_sales = df.groupby('Corporation')['Sales_2024'].sum().sort_values(ascending=False)
                total_sales = corp_sales.sum()
                
                if total_sales > 0:
                    # HHI indeksi
                    market_shares = (corp_sales / total_sales * 100)
                    metrics['hhi_index'] = (market_shares ** 2).sum() / 10000
                    
                    # Konsantrasyon oranları
                    metrics['top_3_share'] = corp_sales.nlargest(3).sum() / total_sales * 100
                    metrics['top_5_share'] = corp_sales.nlargest(5).sum() / total_sales * 100
            
            # International product metrikleri
            if 'Is_International' in df.columns:
                intl_metrics = AdvancedAnalyticsEngine._calculate_international_metrics(df)
                metrics.update(intl_metrics)
            
            # Fiyat metrikleri
            if 'Avg_Price_2024' in df.columns:
                prices = df['Avg_Price_2024']
                metrics['avg_price'] = prices.mean()
                metrics['price_std'] = prices.std()
                metrics['price_cv'] = (prices.std() / prices.mean() * 100) if prices.mean() > 0 else 0
            
            # Coğrafi metrikler
            if 'Country' in df.columns:
                metrics['country_count'] = df['Country'].nunique()
                if 'Sales_2024' in df.columns:
                    country_sales = df.groupby('Country')['Sales_2024'].sum()
                    metrics['top_country_share'] = (country_sales.max() / country_sales.sum() * 100) if country_sales.sum() > 0 else 0
            
            # Molekül çeşitliliği
            if 'Molecule' in df.columns:
                metrics['unique_molecules'] = df['Molecule'].nunique()
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return {}
    
    @staticmethod
    def _calculate_international_metrics(df):
        """International product metrikleri"""
        metrics = {}
        
        try:
            if 'Is_International' in df.columns:
                intl_df = df[df['Is_International']]
                local_df = df[~df['Is_International']]
                
                metrics['intl_product_count'] = len(intl_df)
                metrics['local_product_count'] = len(local_df)
                metrics['intl_percentage'] = (len(intl_df) / len(df) * 100) if len(df) > 0 else 0
                
                if 'Sales_2024' in df.columns:
                    intl_sales = intl_df['Sales_2024'].sum()
                    total_sales = df['Sales_2024'].sum()
                    
                    if total_sales > 0:
                        metrics['intl_sales_share'] = (intl_sales / total_sales) * 100
                        metrics['intl_sales_avg'] = intl_df['Sales_2024'].mean() if len(intl_df) > 0 else 0
                
                if 'Growth_2023_2024' in df.columns:
                    metrics['intl_avg_growth'] = intl_df['Growth_2023_2024'].mean() if len(intl_df) > 0 else 0
                    metrics['local_avg_growth'] = local_df['Growth_2023_2024'].mean() if len(local_df) > 0 else 0
            
            return metrics
            
        except Exception as e:
            return {}
    
    @staticmethod
    def detect_strategic_insights(df):
        """Stratejik içgörüleri tespit et"""
        insights = []
        
        try:
            # En çok satan ürünler
            if 'Sales_2024' in df.columns:
                top_products = df.nlargest(5, 'Sales_2024')
                top_sales = top_products['Sales_2024'].sum()
                total_sales = df['Sales_2024'].sum()
                
                if total_sales > 0:
                    insights.append(MarketInsight(
                        title="🏆 Top Performanslı Ürünler",
                        description=f"Top 5 ürün toplam pazarın %{(top_sales/total_sales*100):.1f}'ini oluşturuyor.",
                        insight_type="success",
                        severity="low",
                        confidence=0.9,
                        recommendations=[
                            "Bu ürünlerin büyüme potansiyelini değerlendirin",
                            "Rekabet analizi yapın",
                            "Fiyat optimizasyonu uygulayın"
                        ],
                        affected_products=top_products['Molecule'].tolist() if 'Molecule' in df.columns else []
                    ))
            
            # En hızlı büyüyen ürünler
            if 'Growth_2023_2024' in df.columns:
                top_growth = df.nlargest(5, 'Growth_2023_2024')
                avg_growth = top_growth['Growth_2023_2024'].mean()
                
                insights.append(MarketInsight(
                    title="🚀 Yüksek Büyüme Potansiyeli",
                    description=f"En hızlı büyüyen 5 ürün ortalama %{avg_growth:.1f} büyüme gösteriyor.",
                    insight_type="info",
                    severity="medium",
                    confidence=0.8,
                    recommendations=[
                        "Büyüme trendlerini izleyin",
                        "Pazar genişletme fırsatlarını değerlendirin",
                        "Rekabetçi pozisyonu güçlendirin"
                    ],
                    affected_products=top_growth['Molecule'].tolist() if 'Molecule' in df.columns else []
                ))
            
            # International product fırsatları
            if 'Is_International' in df.columns and 'Growth_2023_2024' in df.columns:
                intl_growth = df[df['Is_International']]['Growth_2023_2024'].mean()
                local_growth = df[~df['Is_International']]['Growth_2023_2024'].mean()
                
                if intl_growth > local_growth:
                    insights.append(MarketInsight(
                        title="🌍 International Product Avantajı",
                        description=f"International product'lar yerel ürünlerden %{intl_growth-local_growth:.1f} daha hızlı büyüyor.",
                        insight_type="success",
                        severity="medium",
                        confidence=0.7,
                        recommendations=[
                            "International product portföyünü genişletin",
                            "Yeni coğrafyalara açılın",
                            "Global pazarlama stratejileri geliştirin"
                        ]
                    ))
            
            # Pazar konsantrasyonu uyarıları
            metrics = AdvancedAnalyticsEngine.calculate_comprehensive_metrics(df)
            hhi_index = metrics.get('hhi_index', 0)
            
            if hhi_index > 2500:
                insights.append(MarketInsight(
                    title="⚠️ Yüksek Pazar Konsantrasyonu",
                    description=f"HHI indeksi {hhi_index:.0f} - Pazar yüksek derecede konsantre.",
                    insight_type="warning",
                    severity="high",
                    confidence=0.9,
                    recommendations=[
                        "Rekabet stratejilerini gözden geçirin",
                        "Yeni pazar fırsatları araştırın",
                        "Ürün farklılaştırmaya odaklanın"
                    ]
                ))
            
            return insights
            
        except Exception as e:
            st.warning(f"İçgörü tespit hatası: {str(e)}")
            return []
    
    @staticmethod
    def perform_market_segmentation(df, n_clusters=4, method='kmeans'):
        """Pazar segmentasyonu analizi"""
        try:
            # Özellik seçimi
            features = []
            
            if 'Sales_2024' in df.columns:
                features.append('Sales_2024')
            
            if 'Growth_2023_2024' in df.columns:
                features.append('Growth_2023_2024')
            
            if 'Avg_Price_2024' in df.columns:
                features.append('Avg_Price_2024')
            
            if len(features) < 2:
                st.warning("Segmentasyon için yeterli özellik bulunamadı")
                return None
            
            # Veri hazırlama
            seg_data = df[features].fillna(0)
            
            if len(seg_data) < n_clusters * 10:
                st.warning("Segmentasyon için yeterli veri noktası yok")
                return None
            
            # Normalizasyon
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(seg_data)
            
            # Segmentasyon
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif method == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                model = KMeans(n_clusters=n_clusters, random_state=42)
            
            clusters = model.fit_predict(X_scaled)
            
            # Segment isimlendirme
            segment_names = {
                0: 'Star Ürünler',
                1: 'Nakit İnekleri',
                2: 'Soru İşaretleri',
                3: 'Köpek Ürünler',
                4: 'Yükselen Yıldızlar',
                5: 'Olgun Ürünler',
                6: 'Yenilikçi Ürünler'
            }
            
            df_segmented = df.copy()
            df_segmented['Segment'] = clusters
            df_segmented['Segment_Name'] = df_segmented['Segment'].apply(
                lambda x: segment_names.get(x, f'Segment {x}')
            )
            
            # Segment metrikleri
            segment_stats = []
            for seg_num in range(n_clusters):
                seg_df = df_segmented[df_segmented['Segment'] == seg_num]
                
                stats = {
                    'Segment': seg_num,
                    'Segment_Name': segment_names.get(seg_num, f'Segment {seg_num}'),
                    'Product_Count': len(seg_df),
                    'Avg_Sales': seg_df['Sales_2024'].mean() if 'Sales_2024' in seg_df.columns else 0,
                    'Avg_Growth': seg_df['Growth_2023_2024'].mean() if 'Growth_2023_2024' in seg_df.columns else 0,
                    'Avg_Price': seg_df['Avg_Price_2024'].mean() if 'Avg_Price_2024' in seg_df.columns else 0
                }
                segment_stats.append(stats)
            
            return {
                'segmented_data': df_segmented,
                'segment_stats': pd.DataFrame(segment_stats),
                'n_clusters': len(np.unique(clusters))
            }
            
        except Exception as e:
            st.warning(f"Segmentasyon hatası: {str(e)}")
            return None

# ================================================
# VISUALIZATION ENGINE
# ================================================

class VisualizationEngine:
    """Gelişmiş görselleştirme motoru"""
    
    @staticmethod
    def create_dashboard_metrics(metrics):
        """Dashboard metrik kartlarını oluştur"""
        html_output = """
        <div class="metric-grid">
        """
        
        # Toplam Pazar Değeri
        total_sales = metrics.get('total_sales_2024', 0)
        html_output += f"""
            <div class="metric-card primary">
                <div class="metric-label">TOPLAM PAZAR DEĞERİ</div>
                <div class="metric-value">${total_sales/1e9:.2f}B</div>
                <div class="metric-trend">
                    <span>Global Pazar 2024</span>
                </div>
            </div>
        """
        
        # Ortalama Büyüme
        avg_growth = metrics.get('avg_growth', 0)
        growth_class = "success" if avg_growth > 0 else "danger"
        html_output += f"""
            <div class="metric-card {growth_class}">
                <div class="metric-label">ORTALAMA BÜYÜME</div>
                <div class="metric-value">{avg_growth:.1f}%</div>
                <div class="metric-trend">
                    <span>Yıllık Büyüme</span>
                </div>
            </div>
        """
        
        # International Product Payı
        intl_share = metrics.get('intl_sales_share', 0)
        intl_color = "success" if intl_share > 20 else "warning" if intl_share > 10 else "info"
        html_output += f"""
            <div class="metric-card {intl_color}">
                <div class="metric-label">INTERNATIONAL PAYI</div>
                <div class="metric-value">{intl_share:.1f}%</div>
                <div class="metric-trend">
                    <span>Global Yayılım</span>
                </div>
            </div>
        """
        
        # HHI İndeksi
        hhi = metrics.get('hhi_index', 0)
        hhi_status = "danger" if hhi > 2500 else "warning" if hhi > 1500 else "success"
        html_output += f"""
            <div class="metric-card {hhi_status}">
                <div class="metric-label">REKABET YOĞUNLUĞU</div>
                <div class="metric-value">{hhi:.0f}</div>
                <div class="metric-trend">
                    <span>{'Monopol' if hhi > 2500 else 'Oligopol' if hhi > 1500 else 'Rekabetçi'}</span>
                </div>
            </div>
        """
        
        # Molekül Çeşitliliği
        unique_molecules = metrics.get('unique_molecules', 0)
        html_output += f"""
            <div class="metric-card">
                <div class="metric-label">MOLEKÜL ÇEŞİTLİLİĞİ</div>
                <div class="metric-value">{unique_molecules:,}</div>
                <div class="metric-trend">
                    <span>Farklı Molekül</span>
                </div>
            </div>
        """
        
        # Ortalama Fiyat
        avg_price = metrics.get('avg_price', 0)
        html_output += f"""
            <div class="metric-card">
                <div class="metric-label">ORTALAMA FİYAT</div>
                <div class="metric-value">${avg_price:.2f}</div>
                <div class="metric-trend">
                    <span>Birim Başına</span>
                </div>
            </div>
        """
        
        # Yüksek Büyüyen Ürünler
        high_growth = metrics.get('high_growth_count', 0)
        total_products = metrics.get('total_products', 0)
        high_growth_pct = (high_growth / total_products * 100) if total_products > 0 else 0
        html_output += f"""
            <div class="metric-card success">
                <div class="metric-label">YÜKSEK BÜYÜME</div>
                <div class="metric-value">{high_growth_pct:.1f}%</div>
                <div class="metric-trend">
                    <span>{high_growth} ürün > %20</span>
                </div>
            </div>
        """
        
        # Coğrafi Yayılım
        country_count = metrics.get('country_count', 0)
        html_output += f"""
            <div class="metric-card">
                <div class="metric-label">COĞRAFİ YAYILIM</div>
                <div class="metric-value">{country_count}</div>
                <div class="metric-trend">
                    <span>Ülke Sayısı</span>
                </div>
            </div>
        """
        
        html_output += "</div>"
        st.markdown(html_output, unsafe_allow_html=True)
    
    @staticmethod
    def create_sales_trend_chart(df):
        """Satış trend grafiği"""
        try:
            yearly_data = []
            
            # Tüm satış sütunlarını bul
            for col in df.columns:
                if 'Sales_' in col:
                    year = col.split('_')[-1]
                    if year.isdigit():
                        yearly_data.append({
                            'Year': year,
                            'Total_Sales': df[col].sum()
                        })
            
            if len(yearly_data) < 2:
                return None
            
            yearly_df = pd.DataFrame(yearly_data)
            yearly_df = yearly_df.sort_values('Year')
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=yearly_df['Year'],
                y=yearly_df['Total_Sales'],
                name='Toplam Satış',
                marker_color=AppConfig.COLORS['primary']
            ))
            
            fig.update_layout(
                title='Yıllık Satış Trendleri',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                xaxis_title='Yıl',
                yaxis_title='Satış ($)'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Trend grafiği oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_market_share_chart(df):
        """Pazar payı analiz grafiği"""
        try:
            if 'Corporation' not in df.columns or 'Sales_2024' not in df.columns:
                return None
            
            corp_sales = df.groupby('Corporation')['Sales_2024'].sum().sort_values(ascending=False)
            top_corps = corp_sales.nlargest(10)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=top_corps.values,
                y=top_corps.index,
                orientation='h',
                marker_color=AppConfig.CHART_COLORS,
                text=[f'${x/1e6:.1f}M' for x in top_corps.values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Top 10 Şirket - Pazar Payı',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                xaxis_title='Satış ($)',
                yaxis_title='Şirket'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Pazar payı grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_price_volume_analysis(df):
        """Fiyat-hacim analiz grafiği"""
        try:
            if 'Avg_Price_2024' not in df.columns or 'Units_2024' not in df.columns:
                return None
            
            # Veriyi hazırla
            analysis_df = df[['Avg_Price_2024', 'Units_2024']].dropna()
            analysis_df = analysis_df[(analysis_df['Avg_Price_2024'] > 0) & (analysis_df['Units_2024'] > 0)]
            
            if len(analysis_df) < 10:
                return None
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=analysis_df['Avg_Price_2024'],
                y=analysis_df['Units_2024'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=np.log1p(analysis_df['Units_2024']),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Log(Hacim)")
                ),
                name='Ürünler'
            ))
            
            fig.update_layout(
                title='Fiyat-Hacim İlişkisi',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                xaxis_title='Fiyat ($)',
                yaxis_title='Hacim'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Fiyat-hacim grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_price_elasticity_chart(df):
        """Fiyat esnekliği analiz grafiği"""
        try:
            if 'Avg_Price_2024' not in df.columns or 'Units_2024' not in df.columns:
                return None, 0
            
            # Veriyi hazırla
            analysis_df = df[['Avg_Price_2024', 'Units_2024']].dropna()
            analysis_df = analysis_df[(analysis_df['Avg_Price_2024'] > 0) & (analysis_df['Units_2024'] > 0)]
            
            if len(analysis_df) < 10:
                return None, 0
            
            # Korelasyon hesapla
            correlation = analysis_df['Avg_Price_2024'].corr(analysis_df['Units_2024'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=analysis_df['Avg_Price_2024'],
                y=analysis_df['Units_2024'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=analysis_df['Units_2024'],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Hacim")
                ),
                name='Ürünler'
            ))
            
            fig.update_layout(
                title=f'Fiyat Esnekliği Analizi (r={correlation:.3f})',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                xaxis_title='Fiyat ($)',
                yaxis_title='Hacim'
            )
            
            return fig, correlation
            
        except Exception as e:
            st.warning(f"Fiyat esnekliği grafiği hatası: {str(e)}")
            return None, 0
    
    @staticmethod
    def create_international_product_analysis(df):
        """International product analiz grafiği"""
        try:
            if 'Is_International' not in df.columns:
                return None
            
            # International product analizi
            intl_df = df[df['Is_International']]
            local_df = df[~df['Is_International']]
            
            if len(intl_df) == 0:
                return None
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('International vs Local', 'Satış Karşılaştırması',
                               'Ülke Dağılımı', 'Büyüme Karşılaştırması')
            )
            
            # International vs Local
            fig.add_trace(
                go.Pie(
                    labels=['International', 'Local'],
                    values=[len(intl_df), len(local_df)],
                    hole=0.3,
                    marker_colors=[AppConfig.COLORS['primary'], AppConfig.COLORS['light']]
                ),
                row=1, col=1
            )
            
            # Satış karşılaştırması
            if 'Sales_2024' in df.columns:
                intl_sales = intl_df['Sales_2024'].sum()
                local_sales = local_df['Sales_2024'].sum()
                
                fig.add_trace(
                    go.Bar(
                        x=['International', 'Local'],
                        y=[intl_sales, local_sales],
                        marker_color=[AppConfig.COLORS['primary'], AppConfig.COLORS['light']]
                    ),
                    row=1, col=2
                )
            
            # Coğrafi yayılım
            if 'Country' in df.columns:
                country_counts = intl_df['Country'].value_counts().head(10)
                
                fig.add_trace(
                    go.Bar(
                        x=country_counts.values,
                        y=country_counts.index,
                        orientation='h',
                        marker_color=AppConfig.COLORS['success']
                    ),
                    row=2, col=1
                )
            
            # Büyüme karşılaştırması
            if 'Growth_2023_2024' in df.columns:
                intl_growth = intl_df['Growth_2023_2024'].mean()
                local_growth = local_df['Growth_2023_2024'].mean()
                
                if not pd.isna(intl_growth) and not pd.isna(local_growth):
                    fig.add_trace(
                        go.Bar(
                            x=['International', 'Local'],
                            y=[intl_growth, local_growth],
                            marker_color=[AppConfig.COLORS['primary'], AppConfig.COLORS['light']]
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"International product grafiği hatası: {str(e)}")
            return None

# ================================================
# FILTER SYSTEM
# ================================================

class AdvancedFilterSystem:
    """Gelişmiş filtreleme sistemi"""
    
    def __init__(self, df):
        self.df = df
        self.filters = {}
    
    def render_filter_sidebar(self):
        """Filtre sidebar'ını oluştur"""
        with st.sidebar:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">🔍 Arama ve Filtreleme</div>', unsafe_allow_html=True)
            
            # Global arama
            search_term = st.text_input(
                "Global Arama",
                placeholder="Molekül, Şirket, Ülke...",
                key="global_search"
            )
            
            self.filters['search'] = search_term
            
            # Kategori filtreleri
            if 'Country' in self.df.columns:
                countries = sorted(self.df['Country'].dropna().unique())
                selected_countries = st.multiselect(
                    "🌍 Ülkeler",
                    options=countries
                )
                if selected_countries:
                    self.filters['countries'] = selected_countries
            
            if 'Corporation' in self.df.columns:
                corporations = sorted(self.df['Corporation'].dropna().unique())
                selected_corps = st.multiselect(
                    "🏢 Şirketler",
                    options=corporations
                )
                if selected_corps:
                    self.filters['corporations'] = selected_corps
            
            # Numerik filtreler
            st.markdown("---")
            st.markdown("📊 **Numerik Filtreler**")
            
            if 'Sales_2024' in self.df.columns:
                min_sales = float(self.df['Sales_2024'].min())
                max_sales = float(self.df['Sales_2024'].max())
                
                sales_range = st.slider(
                    "Satış Aralığı ($)",
                    min_value=min_sales,
                    max_value=max_sales,
                    value=(min_sales, max_sales)
                )
                self.filters['sales_range'] = sales_range
            
            if 'Growth_2023_2024' in self.df.columns:
                min_growth = float(self.df['Growth_2023_2024'].min())
                max_growth = float(self.df['Growth_2023_2024'].max())
                
                growth_range = st.slider(
                    "Büyüme Oranı (%)",
                    min_value=min_growth,
                    max_value=max_growth,
                    value=(min(min_growth, -50.0), max(max_growth, 150.0))
                )
                self.filters['growth_range'] = growth_range
            
            # Filtre butonları
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                apply_clicked = st.button("✅ Uygula", use_container_width=True)
            with col2:
                clear_clicked = st.button("🗑️ Temizle", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            return apply_clicked, clear_clicked
    
    def apply_filters(self):
        """Filtreleri uygula"""
        filtered_df = self.df.copy()
        
        # Global arama
        if self.filters.get('search'):
            search_mask = pd.Series(False, index=filtered_df.index)
            for col in filtered_df.columns:
                try:
                    search_mask = search_mask | filtered_df[col].astype(str).str.contains(
                        self.filters['search'], case=False, na=False
                    )
                except:
                    continue
            filtered_df = filtered_df[search_mask]
        
        # Kategori filtreleri
        if 'countries' in self.filters and self.filters['countries']:
            filtered_df = filtered_df[filtered_df['Country'].isin(self.filters['countries'])]
        
        if 'corporations' in self.filters and self.filters['corporations']:
            filtered_df = filtered_df[filtered_df['Corporation'].isin(self.filters['corporations'])]
        
        # Numerik filtreler
        if 'sales_range' in self.filters and 'Sales_2024' in filtered_df.columns:
            min_val, max_val = self.filters['sales_range']
            filtered_df = filtered_df[
                (filtered_df['Sales_2024'] >= min_val) & 
                (filtered_df['Sales_2024'] <= max_val)
            ]
        
        if 'growth_range' in self.filters and 'Growth_2023_2024' in filtered_df.columns:
            min_val, max_val = self.filters['growth_range']
            filtered_df = filtered_df[
                (filtered_df['Growth_2023_2024'] >= min_val) & 
                (filtered_df['Growth_2023_2024'] <= max_val)
            ]
        
        return filtered_df

# ================================================
# REPORTING SYSTEM
# ================================================

class ReportingSystem:
    """Profesyonel raporlama sistemi"""
    
    @staticmethod
    def generate_excel_report(df, metrics, insights, segmentation_results=None):
        """Excel raporu oluştur"""
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Ham veri
                df.to_excel(writer, sheet_name='HAM_VERI', index=False)
                
                # Metrikler
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['METRİK', 'DEĞER'])
                metrics_df.to_excel(writer, sheet_name='OZET_METRIKLER', index=False)
                
                # Pazar payı analizi
                if 'Corporation' in df.columns and 'Sales_2024' in df.columns:
                    market_share = df.groupby('Corporation')['Sales_2024'].sum().sort_values(ascending=False)
                    market_share_df = market_share.reset_index()
                    market_share_df.columns = ['ŞİRKET', 'SATIŞ']
                    market_share_df['PAY (%)'] = (market_share_df['SATIŞ'] / market_share_df['SATIŞ'].sum()) * 100
                    market_share_df.to_excel(writer, sheet_name='PAZAR_PAYI', index=False)
                
                # International product analizi
                if 'Is_International' in df.columns:
                    intl_df = df[df['Is_International']]
                    if len(intl_df) > 0:
                        intl_summary = intl_df.groupby('Molecule').agg({
                            'Sales_2024': 'sum',
                            'Corporation': 'nunique',
                            'Country': 'nunique'
                        }).round(2)
                        intl_summary.to_excel(writer, sheet_name='INTERNATIONAL_ANALIZ')
            
            output.seek(0)
            return output
            
        except Exception as e:
            st.error(f"Excel rapor oluşturma hatası: {str(e)}")
            return None

# ================================================
# MAIN APPLICATION
# ================================================

class PharmaIntelligenceApp:
    """Ana uygulama sınıfı"""
    
    def __init__(self):
        self.data_processor = DataProcessingEngine()
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.visualization_engine = VisualizationEngine()
        self.reporting_system = ReportingSystem()
        
        self.initialize_session_state()
        self.setup_page()
    
    def initialize_session_state(self):
        """Session state'leri başlat"""
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'filtered_df' not in st.session_state:
            st.session_state.filtered_df = None
        if 'metrics' not in st.session_state:
            st.session_state.metrics = None
        if 'insights' not in st.session_state:
            st.session_state.insights = []
        if 'segmentation_results' not in st.session_state:
            st.session_state.segmentation_results = None
    
    def setup_page(self):
        """Sayfa ayarlarını yap"""
        st.set_page_config(**AppConfig.PAGE_CONFIG)
        ProfessionalStyler.apply_style()
        
        # Ana başlık
        st.markdown("""
        <div class="main-header">
            <h1 class="main-title">💊 PHARMAINTELLIGENCE ENTERPRISE</h1>
            <p class="main-subtitle">
            Advanced Pharmaceutical Market Analytics Platform | International Product Analysis | 
            Predictive Insights | Strategic Recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Sidebar'ı oluştur"""
        with st.sidebar:
            # Logo ve başlık
            st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #f1f5f9; font-size: 1.5rem; margin-bottom: 0.5rem;">🎛️ KONTROL PANELİ</h2>
                <p style="color: #94a3b8; font-size: 0.9rem;">PharmaIntelligence v4.0</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Veri yükleme
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">📁 VERİ YÜKLEME</div>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Excel/CSV Dosyası Yükleyin",
                type=['xlsx', 'xls', 'csv']
            )
            
            if uploaded_file:
                if st.button("🚀 Veriyi Yükle & Analiz Et", type="primary", use_container_width=True):
                    with st.spinner("Veri işleniyor..."):
                        # Veriyi yükle
                        df = DataProcessingEngine.load_dataset(uploaded_file)
                        
                        if df is not None and len(df) > 0:
                            # Veriyi hazırla
                            df = self.data_processor.clean_and_prepare(df)
                            
                            # Session state'e kaydet
                            st.session_state.df = df
                            st.session_state.filtered_df = df.copy()
                            
                            # Analizleri yap
                            st.session_state.metrics = self.analytics_engine.calculate_comprehensive_metrics(df)
                            st.session_state.insights = self.analytics_engine.detect_strategic_insights(df)
                            
                            st.success(f"✅ {len(df):,} satır başarıyla yüklendi ve analiz edildi!")
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Filtreleme (eğer veri yüklüyse)
            if st.session_state.df is not None:
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                
                filter_system = AdvancedFilterSystem(st.session_state.df)
                apply_clicked, clear_clicked = filter_system.render_filter_sidebar()
                
                if apply_clicked:
                    with st.spinner("Filtreler uygulanıyor..."):
                        filtered_df = filter_system.apply_filters()
                        st.session_state.filtered_df = filtered_df
                        st.session_state.metrics = self.analytics_engine.calculate_comprehensive_metrics(filtered_df)
                        st.session_state.insights = self.analytics_engine.detect_strategic_insights(filtered_df)
                        st.success(f"✅ Filtreler uygulandı: {len(filtered_df):,} satır")
                        st.rerun()
                
                if clear_clicked:
                    st.session_state.filtered_df = st.session_state.df.copy()
                    st.session_state.metrics = self.analytics_engine.calculate_comprehensive_metrics(st.session_state.df)
                    st.session_state.insights = self.analytics_engine.detect_strategic_insights(st.session_state.df)
                    st.success("✅ Filtreler temizlendi")
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Footer
            st.markdown("""
            <div style="text-align: center; margin-top: 2rem; color: #64748b; font-size: 0.8rem;">
                <hr style="border-color: rgba(148, 163, 184, 0.2); margin: 1rem 0;">
                <p><strong>PharmaIntelligence Enterprise</strong><br>
                v4.0 | © 2024 Tüm hakları saklıdır.</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_welcome_screen(self):
        """Hoşgeldiniz ekranını göster"""
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">💊</div>
            <h2 style="color: #f1f5f9; margin-bottom: 1rem;">PharmaIntelligence Enterprise'a Hoşgeldiniz</h2>
            <p style="color: #cbd5e1; margin-bottom: 2rem; line-height: 1.6;">
            İleri düzey ilaç pazarı analitik platformu. Verilerinizi yükleyin, International Product'ları analiz edin, 
            pazar trendlerini keşfedin ve stratejik kararlar alın.
            </p>
            
            <div class="feature-grid">
                <div class="feature-card blue">
                    <div class="feature-icon">🌍</div>
                    <div class="feature-title">International Product Analytics</div>
                    <div class="feature-description">Çoklu pazar ürün analizi ve global strateji geliştirme</div>
                </div>
                <div class="feature-card teal">
                    <div class="feature-icon">📈</div>
                    <div class="feature-title">Pazar Trend Analizi</div>
                    <div class="feature-description">Derin pazar içgörüleri ve yapay zeka destekli tahminler</div>
                </div>
                <div class="feature-card green">
                    <div class="feature-icon">💰</div>
                    <div class="feature-title">Fiyat İstihbaratı</div>
                    <div class="feature-description">Rekabetçi fiyatlandırma, esneklik analizi ve optimizasyon</div>
                </div>
                <div class="feature-card purple">
                    <div class="feature-icon">🏆</div>
                    <div class="feature-title">Rekabet Analizi</div>
                    <div class="feature-description">HHI indeksi, pazar konsantrasyonu ve rakip takibi</div>
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(6, 182, 212, 0.1));
                        padding: 1.5rem; border-radius: 12px; margin-top: 2rem; border: 1px solid rgba(37, 99, 235, 0.3);">
                <div style="font-weight: 600; color: #3b82f6; margin-bottom: 0.8rem; font-size: 1.1rem;">
                    🎯 Başlamak İçin
                </div>
                <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                1. Sol taraftaki panelden Excel/CSV dosyanızı yükleyin<br>
                2. "Veriyi Yükle & Analiz Et" butonuna tıklayın<br>
                3. Analiz sonuçlarını görmek için tabları kullanın
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_overview_tab(self):
        """Genel bakış tab'ı"""
        st.markdown('<div class="section-header">📊 Genel Bakış ve Performans Göstergeleri</div>', unsafe_allow_html=True)
        
        # Metrik kartları
        if st.session_state.metrics:
            self.visualization_engine.create_dashboard_metrics(st.session_state.metrics)
        
        # Stratejik içgörüler
        st.markdown('<div class="subsection-header">🔍 Stratejik İçgörüler</div>', unsafe_allow_html=True)
        
        if st.session_state.insights:
            insight_cols = st.columns(2)
            
            for idx, insight in enumerate(st.session_state.insights[:6]):
                with insight_cols[idx % 2]:
                    icon = "💡"
                    if insight.insight_type == 'warning':
                        icon = "⚠️"
                    elif insight.insight_type == 'success':
                        icon = "✅"
                    elif insight.insight_type == 'info':
                        icon = "ℹ️"
                    
                    st.markdown(f"""
                    <div class="insight-card {insight.insight_type}">
                        <div class="insight-icon">{icon}</div>
                        <div class="insight-title">{insight.title}</div>
                        <div class="insight-content">{insight.description}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Verileriniz analiz ediliyor... Stratejik içgörüler burada görünecek.")
        
        # Veri önizleme
        st.markdown('<div class="subsection-header">📋 Veri Önizleme</div>', unsafe_allow_html=True)
        
        if st.session_state.filtered_df is not None:
            df = st.session_state.filtered_df
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                rows_to_show = st.slider("Gösterilecek Satır Sayısı", 10, 1000, 100)
                
                available_columns = df.columns.tolist()
                default_columns = []
                
                priority_columns = ['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_2023_2024']
                for col in priority_columns:
                    if col in available_columns:
                        default_columns.append(col)
                
                if not default_columns:
                    default_columns = available_columns[:5]
                
                selected_columns = st.multiselect(
                    "Sütunlar",
                    options=available_columns,
                    default=default_columns,
                    key="overview_columns"
                )
            
            with col2:
                if selected_columns:
                    st.dataframe(df[selected_columns].head(rows_to_show), use_container_width=True, height=400)
                else:
                    st.dataframe(df.head(rows_to_show), use_container_width=True, height=400)
    
    def render_market_analysis_tab(self):
        """Pazar analizi tab'ı"""
        st.markdown('<div class="section-header">📈 Pazar Analizi ve Trendler</div>', unsafe_allow_html=True)
        
        if st.session_state.filtered_df is not None:
            df = st.session_state.filtered_df
            
            # Satış trendleri
            st.markdown('<div class="subsection-header">📈 Satış Trendleri</div>', unsafe_allow_html=True)
            trend_fig = self.visualization_engine.create_sales_trend_chart(df)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.info("Satış trend analizi için yeterli yıllık veri bulunamadı.")
            
            # Molekül bazlı analiz
            if 'Molecule' in df.columns:
                st.markdown('<div class="subsection-header">🧪 Molekül Bazlı Analiz</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Sales_2024' in df.columns:
                        top_molecules = df.groupby('Molecule')['Sales_2024'].sum().nlargest(15)
                        fig = px.bar(
                            top_molecules,
                            orientation='h',
                            title='Top 15 Molekül - Satış Performansı',
                            color=top_molecules.values,
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#f1f5f9'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Growth_2023_2024' in df.columns:
                        molecule_growth = df.groupby('Molecule')['Growth_2023_2024'].mean().nlargest(15)
                        fig = px.bar(
                            molecule_growth,
                            orientation='h',
                            title='Top 15 Molekül - Büyüme Oranları',
                            color=molecule_growth.values,
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#f1f5f9'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def render_price_analysis_tab(self):
        """Fiyat analizi tab'ı"""
        st.markdown('<div class="section-header">💰 Fiyat Analizi ve Optimizasyon</div>', unsafe_allow_html=True)
        
        if st.session_state.filtered_df is not None:
            df = st.session_state.filtered_df
            
            # Fiyat-hacim analizi
            st.markdown('<div class="subsection-header">💰 Fiyat-Hacim Analizi</div>', unsafe_allow_html=True)
            price_fig = self.visualization_engine.create_price_volume_analysis(df)
            if price_fig:
                st.plotly_chart(price_fig, use_container_width=True)
            else:
                st.info("Fiyat-hacim analizi için yeterli veri bulunamadı.")
            
            # Fiyat esnekliği analizi
            st.markdown('<div class="subsection-header">📉 Fiyat Esnekliği Analizi</div>', unsafe_allow_html=True)
            elasticity_fig, correlation = self.visualization_engine.create_price_elasticity_chart(df)
            
            if elasticity_fig:
                st.plotly_chart(elasticity_fig, use_container_width=True)
                
                # Esneklik metrikleri
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Korelasyon Katsayısı", f"{correlation:.3f}")
                
                with col2:
                    if correlation < -0.3:
                        elasticity_status = "Yüksek Esneklik"
                    elif correlation > 0.3:
                        elasticity_status = "Düşük Esneklik"
                    else:
                        elasticity_status = "Nötr"
                    st.metric("Esneklik Durumu", elasticity_status)
                
                with col3:
                    if correlation < -0.3:
                        recommendation = "Fiyat Artışı Riskli"
                    elif correlation > 0.3:
                        recommendation = "Fiyat Artışı Mümkün"
                    else:
                        recommendation = "Kısıtlı Artış"
                    st.metric("Öneri", recommendation)
            else:
                st.info("Fiyat esnekliği analizi için yeterli veri bulunamadı.")
    
    def render_competition_analysis_tab(self):
        """Rekabet analizi tab'ı"""
        st.markdown('<div class="section-header">🏆 Rekabet Analizi ve Pazar Yapısı</div>', unsafe_allow_html=True)
        
        if st.session_state.filtered_df is not None:
            df = st.session_state.filtered_df
            
            # Pazar payı analizi
            st.markdown('<div class="subsection-header">🏆 Pazar Payı Analizi</div>', unsafe_allow_html=True)
            share_fig = self.visualization_engine.create_market_share_chart(df)
            if share_fig:
                st.plotly_chart(share_fig, use_container_width=True)
            else:
                st.info("Pazar payı analizi için gerekli veri bulunamadı.")
            
            # Rekabet metrikleri
            st.markdown('<div class="subsection-header">📊 Rekabet Yoğunluğu Metrikleri</div>', unsafe_allow_html=True)
            
            if st.session_state.metrics:
                cols = st.columns(4)
                
                with cols[0]:
                    hhi = st.session_state.metrics.get('hhi_index', 0)
                    if hhi > 2500:
                        hhi_status = "Monopol"
                    elif hhi > 1800:
                        hhi_status = "Oligopol"
                    else:
                        hhi_status = "Rekabetçi"
                    st.metric("HHI İndeksi", f"{hhi:.0f}", hhi_status)
                
                with cols[1]:
                    top3 = st.session_state.metrics.get('top_3_share', 0)
                    if top3 > 50:
                        concentration = "Yüksek"
                    elif top3 > 30:
                        concentration = "Orta"
                    else:
                        concentration = "Düşük"
                    st.metric("Top 3 Payı", f"{top3:.1f}%", concentration)
                
                with cols[2]:
                    top5 = st.session_state.metrics.get('top_5_share', 0)
                    st.metric("Top 5 Payı", f"{top5:.1f}%")
                
                with cols[3]:
                    price_cv = st.session_state.metrics.get('price_cv', 0)
                    st.metric("Fiyat Varyasyonu", f"{price_cv:.1f}%")
            
            # Şirket performans analizi
            if 'Corporation' in df.columns:
                st.markdown('<div class="subsection-header">📈 Şirket Performans Analizi</div>', unsafe_allow_html=True)
                
                if 'Sales_2024' in df.columns:
                    company_stats = df.groupby('Corporation').agg({
                        'Sales_2024': ['sum', 'mean', 'count'],
                        'Growth_2023_2024': 'mean'
                    }).round(2)
                    
                    company_stats.columns = ['_'.join(col).strip() for col in company_stats.columns.values]
                    company_stats = company_stats.sort_values('Sales_2024_sum', ascending=False)
                    
                    with st.expander("📋 Detaylı Şirket Performans Tablosu"):
                        st.dataframe(company_stats.head(20), use_container_width=True, height=400)
    
    def render_international_product_tab(self):
        """International product tab'ı"""
        st.markdown('<div class="section-header">🌍 International Product Analizi</div>', unsafe_allow_html=True)
        
        if st.session_state.filtered_df is not None:
            df = st.session_state.filtered_df
            
            if 'Is_International' not in df.columns:
                st.warning("International product analizi için gerekli sütun bulunamadı.")
                return
            
            # International product metrikleri
            st.markdown('<div class="subsection-header">📊 International Product Metrikleri</div>', unsafe_allow_html=True)
            
            if st.session_state.metrics:
                cols = st.columns(4)
                
                with cols[0]:
                    intl_count = st.session_state.metrics.get('intl_product_count', 0)
                    total_products = st.session_state.metrics.get('total_products', 0)
                    intl_pct = (intl_count / total_products * 100) if total_products > 0 else 0
                    st.metric("International Product", f"{intl_count}", f"%{intl_pct:.1f}")
                
                with cols[1]:
                    intl_share = st.session_state.metrics.get('intl_sales_share', 0)
                    st.metric("Pazar Payı", f"{intl_share:.1f}%")
                
                with cols[2]:
                    intl_growth = st.session_state.metrics.get('intl_avg_growth', 0)
                    st.metric("Ort. Büyüme", f"{intl_growth:.1f}%")
                
                with cols[3]:
                    local_growth = st.session_state.metrics.get('local_avg_growth', 0)
                    st.metric("Local Büyüme", f"{local_growth:.1f}%")
            
            # International product analiz grafikleri
            st.markdown('<div class="subsection-header">📈 International Product Analiz Grafikleri</div>', unsafe_allow_html=True)
            intl_fig = self.visualization_engine.create_international_product_analysis(df)
            if intl_fig:
                st.plotly_chart(intl_fig, use_container_width=True)
            
            # International product detaylı listesi
            st.markdown('<div class="subsection-header">📋 International Product Detaylı Listesi</div>', unsafe_allow_html=True)
            
            intl_df = df[df['Is_International']]
            
            if len(intl_df) > 0:
                display_columns = []
                for col in ['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_2023_2024']:
                    if col in intl_df.columns:
                        display_columns.append(col)
                
                if display_columns:
                    display_df = intl_df[display_columns].copy()
                    
                    # Formatlama
                    if 'Sales_2024' in display_df.columns:
                        display_df['Sales_2024'] = display_df['Sales_2024'].apply(
                            lambda x: f"${x/1e6:.2f}M" if pd.notnull(x) else "N/A"
                        )
                    
                    st.dataframe(display_df, use_container_width=True, height=400)
            else:
                st.info("International product bulunamadı.")
    
    def render_segmentation_tab(self):
        """Segmentasyon tab'ı"""
        st.markdown('<div class="section-header">🎯 Pazar Segmentasyonu</div>', unsafe_allow_html=True)
        
        if st.session_state.filtered_df is not None:
            df = st.session_state.filtered_df
            
            # Segmentasyon parametreleri
            col1, col2 = st.columns(2)
            
            with col1:
                n_clusters = st.slider("Segment Sayısı", 2, 6, 4)
            
            with col2:
                method = st.selectbox(
                    "Segmentasyon Metodu",
                    ['kmeans', 'agglomerative'],
                    index=0
                )
            
            if st.button("🔍 Segmentasyon Analizi Yap", type="primary"):
                with st.spinner("Pazar segmentasyonu analiz ediliyor..."):
                    segmentation_results = self.analytics_engine.perform_market_segmentation(
                        df, n_clusters, method
                    )
                    
                    if segmentation_results:
                        st.session_state.segmentation_results = segmentation_results
                        st.success(f"{segmentation_results['n_clusters']} segment tespit edildi!")
                        st.rerun()
            
            # Segmentasyon sonuçları
            if st.session_state.segmentation_results:
                results = st.session_state.segmentation_results
                
                # Segment dağılımı
                st.markdown('<div class="subsection-header">📊 Segment Dağılımı</div>', unsafe_allow_html=True)
                
                segment_counts = results['segmented_data']['Segment_Name'].value_counts()
                
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title='Pazar Segmentleri Dağılımı',
                    color_discrete_sequence=AppConfig.CHART_COLORS
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Segment istatistikleri
                st.markdown('<div class="subsection-header">📈 Segment İstatistikleri</div>', unsafe_allow_html=True)
                
                st.dataframe(
                    results['segment_stats'],
                    use_container_width=True,
                    height=300
                )
                
                # Segment detayları
                with st.expander("📋 Segment Detayları"):
                    segment_data = results['segmented_data']
                    
                    selected_segment = st.selectbox(
                        "Segment Seçin",
                        options=sorted(segment_data['Segment'].unique())
                    )
                    
                    seg_df = segment_data[segment_data['Segment'] == selected_segment]
                    
                    display_cols = []
                    for col in ['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_2023_2024']:
                        if col in seg_df.columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        st.dataframe(
                            seg_df[display_cols].head(20),
                            use_container_width=True,
                            height=400
                        )
    
    def render_reporting_tab(self):
        """Raporlama tab'ı"""
        st.markdown('<div class="section-header">📑 Raporlama ve İndirme</div>', unsafe_allow_html=True)
        
        if st.session_state.filtered_df is not None:
            df = st.session_state.filtered_df
            
            # Rapor türleri
            st.markdown('<div class="subsection-header">📊 Rapor Türleri</div>', unsafe_allow_html=True)
            
            report_type = st.radio(
                "Rapor türü seçin:",
                ["Excel Detaylı Rapor", "CSV Ham Veri"],
                horizontal=True
            )
            
            # Rapor oluşturma
            st.markdown('<div class="subsection-header">🛠️ Rapor Oluşturma</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📈 Excel Raporu Oluştur", use_container_width=True):
                    if report_type == "Excel Detaylı Rapor":
                        with st.spinner("Excel raporu oluşturuluyor..."):
                            excel_report = self.reporting_system.generate_excel_report(
                                df,
                                st.session_state.metrics,
                                st.session_state.insights,
                                st.session_state.segmentation_results
                            )
                            
                            if excel_report:
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                st.download_button(
                                    label="⬇️ Excel İndir",
                                    data=excel_report,
                                    file_name=f"pharma_intelligence_raporu_{timestamp}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
            
            with col2:
                if st.button("🔄 Analizi Sıfırla", use_container_width=True):
                    for key in ['df', 'filtered_df', 'metrics', 'insights', 'segmentation_results']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            
            with col3:
                if st.button("💾 CSV İndir", use_container_width=True):
                    csv = df.to_csv(index=False)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="⬇️ CSV İndir",
                        data=csv,
                        file_name=f"pharma_data_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Veri istatistikleri
            st.markdown('<div class="subsection-header">📈 Veri İstatistikleri</div>', unsafe_allow_html=True)
            
            cols = st.columns(4)
            
            with cols[0]:
                st.metric("Toplam Satır", f"{len(df):,}")
            
            with cols[1]:
                st.metric("Toplam Sütun", len(df.columns))
            
            with cols[2]:
                null_count = df.isnull().sum().sum()
                null_pct = (null_count / (len(df) * len(df.columns))) * 100
                st.metric("Eksik Veri", f"{null_pct:.1f}%")
            
            with cols[3]:
                intl_count = df['Is_International'].sum() if 'Is_International' in df.columns else 0
                st.metric("International Product", intl_count)
    
    def run(self):
        """Uygulamayı çalıştır"""
        # Sidebar'ı render et
        self.render_sidebar()
        
        # Veri yüklü değilse hoşgeldiniz ekranını göster
        if st.session_state.df is None:
            self.render_welcome_screen()
            return
        
        # Tab'ları oluştur
        tabs = st.tabs([
            "📊 Genel Bakış",
            "📈 Pazar Analizi",
            "💰 Fiyat Analizi",
            "🏆 Rekabet Analizi",
            "🌍 International Product",
            "🎯 Segmentasyon",
            "📑 Raporlama"
        ])
        
        # Her tab'ı render et
        with tabs[0]:
            self.render_overview_tab()
        
        with tabs[1]:
            self.render_market_analysis_tab()
        
        with tabs[2]:
            self.render_price_analysis_tab()
        
        with tabs[3]:
            self.render_competition_analysis_tab()
        
        with tabs[4]:
            self.render_international_product_tab()
        
        with tabs[5]:
            self.render_segmentation_tab()
        
        with tabs[6]:
            self.render_reporting_tab()

# ================================================
# APPLICATION ENTRY POINT
# ================================================

if __name__ == "__main__":
    try:
        # Garbage collection'ı etkinleştir
        gc.enable()
        
        # Uygulamayı başlat
        app = PharmaIntelligenceApp()
        app.run()
        
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        
        if st.button("🔄 Sayfayı Yenile", use_container_width=True):
            st.rerun()
