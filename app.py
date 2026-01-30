# app.py - Profesyonel İlaç Pazarı Dashboard (TÜM HATALAR DÜZELTİLMİŞ)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Advanced analytics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import statsmodels.api as sm
from scipy import stats

# Utilities
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple
import math

# ================================================
# 1. PROFESYONEL KONFİGÜRASYON VE STİL AYARLARI
# ================================================
st.set_page_config(
    page_title="PharmaAnalytics Pro | İlaç Pazarı Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://pharmaanalytics.com/support',
        'Report a bug': "https://pharmaanalytics.com/bug",
        'About': "### PharmaAnalytics Pro v4.0\nInternational Product Analytics Dahil"
    }
)

# PROFESYONEL Mavi Tema CSS Stilleri
PROFESSIONAL_CSS = """
<style>
    /* === ROOT VARIABLES === */
    :root {
        --primary-dark: #0a192f;
        --secondary-dark: #112240;
        --accent-blue: #2563eb;
        --accent-blue-light: #3b82f6;
        --accent-blue-dark: #1d4ed8;
        --accent-cyan: #06b6d4;
        --accent-teal: #14b8a6;
        --accent-green: #10b981;
        --accent-yellow: #f59e0b;
        --accent-red: #ef4444;
        
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        
        --bg-primary: #0a192f;
        --bg-secondary: #112240;
        --bg-card: #1e293b;
        --bg-hover: #334155;
        --bg-surface: #1e293b;
        
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
        
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.4);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.5);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.6);
        --shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.7);
        
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
        
        --transition-fast: 200ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-normal: 300ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 500ms cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* === GLOBAL STYLES === */
    .stApp {
        background: linear-gradient(135deg, var(--primary-dark), var(--secondary-dark));
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text-primary);
        min-height: 100vh;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* === TYPOGRAPHY === */
    .pharma-title {
        font-size: 2.8rem;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .pharma-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 800px;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.8rem;
        color: var(--text-primary);
        font-weight: 800;
        margin: 2.5rem 0 1.5rem 0;
        padding-left: 1rem;
        border-left: 5px solid var(--accent-blue);
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.1), transparent);
        padding: 1rem;
        border-radius: var(--radius-sm);
    }
    
    .subsection-title {
        font-size: 1.4rem;
        color: var(--text-primary);
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--bg-hover);
    }
    
    /* === CUSTOM METRIC CARDS === */
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
    }
    
    .custom-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    }
    
    .custom-metric-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-xl);
        border-color: var(--accent-blue);
    }
    
    .custom-metric-card.premium {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    }
    
    .custom-metric-card.warning {
        background: linear-gradient(135deg, var(--accent-yellow), #f97316);
    }
    
    .custom-metric-card.danger {
        background: linear-gradient(135deg, var(--accent-red), #dc2626);
    }
    
    .custom-metric-card.success {
        background: linear-gradient(135deg, var(--accent-green), #059669);
    }
    
    .custom-metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        margin: 0.5rem 0;
        color: var(--text-primary);
        line-height: 1;
    }
    
    .custom-metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .custom-metric-trend {
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
        margin-top: 0.5rem;
    }
    
    .trend-up { color: var(--accent-green); }
    .trend-down { color: var(--accent-red); }
    .trend-neutral { color: var(--text-muted); }
    
    /* === INSIGHT CARDS === */
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
    }
    
    .insight-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
    }
    
    .insight-card.info { border-left-color: var(--accent-blue); }
    .insight-card.success { border-left-color: var(--accent-green); }
    .insight-card.warning { border-left-color: var(--accent-yellow); }
    .insight-card.danger { border-left-color: var(--accent-red); }
    
    .insight-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .insight-title {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .insight-content {
        color: var(--text-secondary);
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* === FILTER SECTION === */
    .filter-section {
        background: var(--bg-card);
        padding: 1.2rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        margin-bottom: 1rem;
        border: 1px solid var(--bg-hover);
    }
    
    .filter-title {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* === FILTER STATUS === */
    .filter-status {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.2), rgba(6, 182, 212, 0.2));
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--accent-blue);
        box-shadow: var(--shadow-md);
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    /* === SIDEBAR === */
    .sidebar-title {
        font-size: 1.4rem;
        color: var(--text-primary);
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--accent-blue);
    }
    
    /* === WELCOME CONTAINER === */
    .welcome-container {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-secondary));
        padding: 3rem;
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow-xl);
        text-align: center;
        margin: 2rem auto;
        max-width: 900px;
        border: 1px solid var(--bg-hover);
    }
    
    .welcome-icon {
        font-size: 5rem;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* === FEATURE CARDS === */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-hover));
        padding: 1.5rem;
        border-radius: var(--radius-md);
        border-left: 4px solid;
        transition: all var(--transition-normal);
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg);
    }
    
    .feature-card-blue { border-left-color: var(--accent-blue); }
    .feature-card-cyan { border-left-color: var(--accent-cyan); }
    .feature-card-green { border-left-color: var(--accent-green); }
    .feature-card-yellow { border-left-color: var(--accent-yellow); }
    
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
        opacity: 0.9;
    }
    
    .feature-title {
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .feature-description {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* === BADGES === */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: var(--accent-green);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: var(--accent-yellow);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: var(--accent-red);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .badge-info {
        background: rgba(37, 99, 235, 0.2);
        color: var(--accent-blue);
        border: 1px solid rgba(37, 99, 235, 0.3);
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ================================================
# 2. OPTİMİZE VERİ İŞLEME SİSTEMİ
# ================================================

class OptimizedDataProcessor:
    """Optimize edilmiş veri işleme sınıfı"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
    def load_large_dataset(file, sample_size=None):
        """Büyük veri setlerini optimize şekilde yükle"""
        try:
            start_time = time.time()
            
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, engine='openpyxl')
            
            df = OptimizedDataProcessor.optimize_dataframe(df)
            
            load_time = time.time() - start_time
            st.success(f"✅ Veri yükleme tamamlandı: {len(df):,} satır, {len(df.columns)} sütun ({load_time:.2f}s)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            return None
    
    @staticmethod
    def optimize_dataframe(df):
        """DataFrame'i optimize et"""
        try:
            # Sütun isimlerini temizle
            df.columns = OptimizedDataProcessor.clean_column_names(df.columns)
            
            # Tarih sütunlarını işle
            date_patterns = ['date', 'time', 'year', 'month', 'day']
            for col in df.columns:
                col_lower = str(col).lower()
                if any(pattern in col_lower for pattern in date_patterns):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
            
            return df
            
        except Exception as e:
            st.warning(f"Optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def clean_column_names(columns):
        """Sütun isimlerini temizle"""
        cleaned = []
        for col in columns:
            if isinstance(col, str):
                # Türkçe karakterleri düzelt
                replacements = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in replacements.items():
                    col = col.replace(tr, en)
                
                # Özel formatları düzelt
                if 'USD' in col and 'MNF' in col and 'MAT' in col:
                    if '2022' in col:
                        if 'Units' in col:
                            col = 'Units_2022'
                        elif 'Avg Price' in col:
                            col = 'Avg_Price_2022'
                        else:
                            col = 'Sales_2022'
                    elif '2023' in col:
                        if 'Units' in col:
                            col = 'Units_2023'
                        elif 'Avg Price' in col:
                            col = 'Avg_Price_2023'
                        else:
                            col = 'Sales_2023'
                    elif '2024' in col:
                        if 'Units' in col:
                            col = 'Units_2024'
                        elif 'Avg Price' in col:
                            col = 'Avg_Price_2024'
                        else:
                            col = 'Sales_2024'
                
                col = col.strip()
            
            cleaned.append(str(col).strip())
        
        return cleaned
    
    @staticmethod
    def prepare_analytics_data(df):
        """Analiz için veriyi hazırla"""
        try:
            # Büyüme hesaplamaları
            for year in [2022, 2023, 2024]:
                sales_col = f'Sales_{year}'
                units_col = f'Units_{year}'
                price_col = f'Avg_Price_{year}'
                
                if sales_col in df.columns and units_col in df.columns:
                    # Average price hesapla (eğer yoksa)
                    if price_col not in df.columns:
                        df[price_col] = df[sales_col] / df[units_col].replace(0, np.nan)
            
            # Yıllık büyüme hesapla
            if 'Sales_2023' in df.columns and 'Sales_2024' in df.columns:
                df['Growth_2023_2024'] = ((df['Sales_2024'] - df['Sales_2023']) / 
                                         df['Sales_2023'].replace(0, np.nan)) * 100
            
            if 'Sales_2022' in df.columns and 'Sales_2023' in df.columns:
                df['Growth_2022_2023'] = ((df['Sales_2023'] - df['Sales_2022']) / 
                                         df['Sales_2022'].replace(0, np.nan)) * 100
            
            # Market share hesapla
            if 'Sales_2024' in df.columns:
                total_sales = df['Sales_2024'].sum()
                if total_sales > 0:
                    df['Market_Share_2024'] = (df['Sales_2024'] / total_sales) * 100
            
            # Performans skoru
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                try:
                    scaler = StandardScaler()
                    numeric_data = df[numeric_cols].fillna(0)
                    scaled_data = scaler.fit_transform(numeric_data)
                    df['Performance_Score'] = scaled_data.mean(axis=1)
                except:
                    pass
            
            return df
            
        except Exception as e:
            st.warning(f"Analiz verisi hazırlama hatası: {str(e)}")
            return df

# ================================================
# 3. GELİŞMİŞ FİLTRELEME SİSTEMİ
# ================================================

class AdvancedFilterSystem:
    """Gelişmiş filtreleme sistemi"""
    
    @staticmethod
    def create_filter_sidebar(df):
        """Filtreleme sidebar'ını oluştur"""
        with st.sidebar.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
            st.markdown('<div class="filter-title">🔍 Arama ve Filtreleme</div>', unsafe_allow_html=True)
            
            search_term = st.text_input(
                "Global Arama",
                placeholder="Molekül, Şirket, Ülke...",
                help="Tüm sütunlarda arama yapın",
                key="global_search"
            )
            
            filter_config = {}
            
            # Ülke filtresi
            if 'Country' in df.columns:
                countries = sorted(df['Country'].dropna().unique())
                selected_countries = st.multiselect(
                    "Ülkeler",
                    options=countries,
                    default=countries[:min(5, len(countries))],
                    help="Filtrelenecek ülkeleri seçin"
                )
                if selected_countries:
                    filter_config['Country'] = selected_countries
            
            # Şirket filtresi
            if 'Corporation' in df.columns:
                companies = sorted(df['Corporation'].dropna().unique())
                selected_companies = st.multiselect(
                    "Şirketler",
                    options=companies,
                    default=companies[:min(5, len(companies))],
                    help="Filtrelenecek şirketleri seçin"
                )
                if selected_companies:
                    filter_config['Corporation'] = selected_companies
            
            # Molekül filtresi
            if 'Molecule' in df.columns:
                molecules = sorted(df['Molecule'].dropna().unique())
                selected_molecules = st.multiselect(
                    "Moleküller",
                    options=molecules,
                    default=molecules[:min(5, len(molecules))],
                    help="Filtrelenecek molekülleri seçin"
                )
                if selected_molecules:
                    filter_config['Molecule'] = selected_molecules
            
            st.markdown("---")
            st.markdown('<div class="filter-title">📊 Numerik Filtreler</div>', unsafe_allow_html=True)
            
            # Satış filtresi
            if 'Sales_2024' in df.columns:
                min_sales = float(df['Sales_2024'].min())
                max_sales = float(df['Sales_2024'].max())
                
                sales_range = st.slider(
                    "Satış Aralığı ($)",
                    min_value=min_sales,
                    max_value=max_sales,
                    value=(min_sales, max_sales),
                    step=(max_sales - min_sales) / 100,
                    help="Satış aralığını seçin"
                )
                filter_config['sales_range'] = sales_range
            
            # Büyüme filtresi
            if 'Growth_2023_2024' in df.columns:
                min_growth = float(df['Growth_2023_2024'].min())
                max_growth = float(df['Growth_2023_2024'].max())
                
                growth_range = st.slider(
                    "Büyüme Oranı (%)",
                    min_value=min_growth,
                    max_value=max_growth,
                    value=(min(min_growth, -50.0), max(max_growth, 150.0)),
                    step=5.0,
                    help="Büyüme oranı aralığını seçin"
                )
                filter_config['growth_range'] = growth_range
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                apply_filter = st.button("✅ Filtre Uygula", width='stretch', key="apply_filter")
            with col2:
                clear_filter = st.button("🗑️ Temizle", width='stretch', key="clear_filter")
            
            return search_term, filter_config, apply_filter, clear_filter
    
    @staticmethod
    def apply_filters(df, search_term, filter_config):
        """Filtreleri uygula"""
        filtered_df = df.copy()
        
        # Global arama
        if search_term:
            search_mask = pd.Series(False, index=filtered_df.index)
            for col in filtered_df.columns:
                try:
                    search_mask = search_mask | filtered_df[col].astype(str).str.contains(
                        search_term, case=False, na=False
                    )
                except:
                    continue
            filtered_df = filtered_df[search_mask]
        
        # Kategori filtreleri
        for column, values in filter_config.items():
            if column in ['Country', 'Corporation', 'Molecule'] and values:
                filtered_df = filtered_df[filtered_df[column].isin(values)]
        
        # Satış aralığı filtresi
        if 'sales_range' in filter_config and 'Sales_2024' in filtered_df.columns:
            min_val, max_val = filter_config['sales_range']
            filtered_df = filtered_df[
                (filtered_df['Sales_2024'] >= min_val) & 
                (filtered_df['Sales_2024'] <= max_val)
            ]
        
        # Büyüme aralığı filtresi
        if 'growth_range' in filter_config and 'Growth_2023_2024' in filtered_df.columns:
            min_val, max_val = filter_config['growth_range']
            filtered_df = filtered_df[
                (filtered_df['Growth_2023_2024'] >= min_val) & 
                (filtered_df['Growth_2023_2024'] <= max_val)
            ]
        
        return filtered_df

# ================================================
# 4. GELİŞMİŞ ANALİTİK MOTORU
# ================================================

class AdvancedPharmaAnalytics:
    """Gelişmiş farma analitik motoru"""
    
    @staticmethod
    def calculate_comprehensive_metrics(df):
        """Kapsamlı pazar metrikleri"""
        metrics = {}
        
        try:
            metrics['Toplam_Satır'] = len(df)
            metrics['Toplam_Sütun'] = len(df.columns)
            
            # Satış metrikleri
            if 'Sales_2024' in df.columns:
                metrics['Toplam_Pazar_Değeri'] = df['Sales_2024'].sum()
                metrics['Ortalama_Satış'] = df['Sales_2024'].mean()
                metrics['Medyan_Satış'] = df['Sales_2024'].median()
                metrics['Satış_Std_Sapma'] = df['Sales_2024'].std()
            
            # Büyüme metrikleri
            if 'Growth_2023_2024' in df.columns:
                metrics['Ortalama_Büyüme'] = df['Growth_2023_2024'].mean()
                metrics['Pozitif_Büyüyen_Ürünler'] = (df['Growth_2023_2024'] > 0).sum()
                metrics['Negatif_Büyüyen_Ürünler'] = (df['Growth_2023_2024'] < 0).sum()
                metrics['Yüksek_Büyüyen_Ürünler'] = (df['Growth_2023_2024'] > 20).sum()
            
            # Pazar konsantrasyonu
            if 'Corporation' in df.columns and 'Sales_2024' in df.columns:
                corp_sales = df.groupby('Corporation')['Sales_2024'].sum().sort_values(ascending=False)
                total_sales = corp_sales.sum()
                
                if total_sales > 0:
                    market_shares = (corp_sales / total_sales * 100)
                    metrics['HHI_İndeksi'] = (market_shares ** 2).sum() / 10000
                    metrics['Top_3_Payı'] = corp_sales.nlargest(3).sum() / total_sales * 100
                    metrics['Top_5_Payı'] = corp_sales.nlargest(5).sum() / total_sales * 100
            
            # Molekül çeşitliliği
            if 'Molecule' in df.columns:
                metrics['Benzersiz_Molekül'] = df['Molecule'].nunique()
            
            # Ülke kapsamı
            if 'Country' in df.columns:
                metrics['Ülke_Sayısı'] = df['Country'].nunique()
            
            # Fiyat metrikleri
            if 'Avg_Price_2024' in df.columns:
                metrics['Ortalama_Fiyat'] = df['Avg_Price_2024'].mean()
                metrics['Fiyat_Varyansı'] = df['Avg_Price_2024'].var()
            
            # International Product metrikleri
            if 'Molecule' in df.columns:
                intl_metrics = AdvancedPharmaAnalytics.calculate_international_metrics(df)
                metrics.update(intl_metrics)
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_international_metrics(df):
        """International Product metrikleri"""
        metrics = {}
        
        try:
            international_products = {}
            
            # International Product'ları tespit et
            if 'Molecule' in df.columns:
                for molecule in df['Molecule'].unique():
                    molecule_df = df[df['Molecule'] == molecule]
                    
                    unique_corporations = molecule_df['Corporation'].nunique() if 'Corporation' in df.columns else 0
                    unique_countries = molecule_df['Country'].nunique() if 'Country' in df.columns else 0
                    
                    if unique_corporations > 1 or unique_countries > 1:
                        international_products[molecule] = {
                            'corporation_count': unique_corporations,
                            'country_count': unique_countries,
                            'total_sales': molecule_df['Sales_2024'].sum() if 'Sales_2024' in df.columns else 0
                        }
            
            metrics['International_Product_Sayısı'] = len(international_products)
            
            if international_products and 'Sales_2024' in df.columns:
                intl_sales = sum(data['total_sales'] for data in international_products.values())
                total_sales = df['Sales_2024'].sum()
                
                if total_sales > 0:
                    metrics['International_Product_Payı'] = (intl_sales / total_sales) * 100
                
                # Ortalama ülke ve şirket sayısı
                if international_products:
                    metrics['Ort_Şirket_Sayısı'] = np.mean([data['corporation_count'] for data in international_products.values()])
                    metrics['Ort_Ülke_Sayısı'] = np.mean([data['country_count'] for data in international_products.values()])
            
            return metrics
            
        except Exception as e:
            return {}
    
    @staticmethod
    def analyze_international_products(df):
        """International Product analizi"""
        try:
            if 'Molecule' not in df.columns:
                return None
            
            analysis_results = []
            
            for molecule in df['Molecule'].unique():
                molecule_df = df[df['Molecule'] == molecule]
                
                unique_corporations = molecule_df['Corporation'].nunique() if 'Corporation' in df.columns else 0
                unique_countries = molecule_df['Country'].nunique() if 'Country' in df.columns else 0
                
                is_international = (unique_corporations > 1 or unique_countries > 1)
                
                analysis_results.append({
                    'Molekül': molecule,
                    'International_Ürün': is_international,
                    'Şirket_Sayısı': unique_corporations,
                    'Ülke_Sayısı': unique_countries,
                    'Toplam_Satış': molecule_df['Sales_2024'].sum() if 'Sales_2024' in df.columns else 0,
                    'Ortalama_Fiyat': molecule_df['Avg_Price_2024'].mean() if 'Avg_Price_2024' in df.columns else None,
                    'Ortalama_Büyüme': molecule_df['Growth_2023_2024'].mean() if 'Growth_2023_2024' in df.columns else None
                })
            
            return pd.DataFrame(analysis_results)
            
        except Exception as e:
            st.warning(f"International Product analiz hatası: {str(e)}")
            return None
    
    @staticmethod
    def detect_strategic_insights(df):
        """Stratejik içgörüleri tespit et"""
        insights = []
        
        try:
            # En çok satan ürünler
            if 'Sales_2024' in df.columns and 'Molecule' in df.columns:
                top_products = df.nlargest(5, 'Sales_2024')
                top_sales = top_products['Sales_2024'].sum()
                total_sales = df['Sales_2024'].sum()
                
                if total_sales > 0:
                    insights.append({
                        'type': 'success',
                        'title': '🏆 En Çok Satan Ürünler',
                        'description': f"Top 5 ürün toplam pazarın %{(top_sales / total_sales * 100):.1f}'ini oluşturuyor.",
                        'data': top_products[['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_2023_2024']]
                    })
            
            # En hızlı büyüyen ürünler
            if 'Growth_2023_2024' in df.columns:
                top_growth = df.nlargest(5, 'Growth_2023_2024')
                insights.append({
                    'type': 'info',
                    'title': '🚀 En Hızlı Büyüyen Ürünler',
                    'description': f"Ortalama %{top_growth['Growth_2023_2024'].mean():.1f} büyüme ile.",
                    'data': top_growth[['Molecule', 'Corporation', 'Country', 'Growth_2023_2024', 'Sales_2024']]
                })
            
            # Pazar lideri
            if 'Corporation' in df.columns and 'Sales_2024' in df.columns:
                top_company = df.groupby('Corporation')['Sales_2024'].sum().idxmax()
                company_sales = df[df['Corporation'] == top_company]['Sales_2024'].sum()
                company_share = (company_sales / df['Sales_2024'].sum()) * 100 if df['Sales_2024'].sum() > 0 else 0
                
                insights.append({
                    'type': 'warning',
                    'title': '🏢 Pazar Lideri',
                    'description': f"{top_company} %{company_share:.1f} pazar payı ile lider konumda.",
                    'data': None
                })
            
            return insights
            
        except Exception as e:
            st.warning(f"İçgörü tespiti hatası: {str(e)}")
            return []

# ================================================
# 5. GÖRSELLEŞTİRME MOTORU
# ================================================

class ProfessionalVisualization:
    """Profesyonel görselleştirme motoru"""
    
    @staticmethod
    def create_dashboard_metrics(df, metrics):
        """Dashboard metrik kartlarını oluştur"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sales = metrics.get('Toplam_Pazar_Değeri', 0)
                st.markdown(f"""
                <div class="custom-metric-card premium">
                    <div class="custom-metric-label">TOPLAM PAZAR DEĞERİ</div>
                    <div class="custom-metric-value">${total_sales/1e9:.2f}B</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">2024</span>
                        <span>Global Pazar</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_growth = metrics.get('Ortalama_Büyüme', 0)
                growth_class = "success" if avg_growth > 0 else "danger"
                st.markdown(f"""
                <div class="custom-metric-card {growth_class}">
                    <div class="custom-metric-label">ORTALAMA BÜYÜME</div>
                    <div class="custom-metric-value">{avg_growth:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">YoY</span>
                        <span>Yıllık Büyüme</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrics.get('HHI_İndeksi', 0)
                hhi_status = "danger" if hhi > 2500 else "warning" if hhi > 1500 else "success"
                st.markdown(f"""
                <div class="custom-metric-card {hhi_status}">
                    <div class="custom-metric-label">REKABET YOĞUNLUĞU</div>
                    <div class="custom-metric-value">{hhi:.0f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-warning">HHI Index</span>
                        <span>{'Monopol' if hhi > 2500 else 'Oligopol' if hhi > 1500 else 'Rekabetçi'}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                intl_share = metrics.get('International_Product_Payı', 0)
                intl_color = "success" if intl_share > 20 else "warning" if intl_share > 10 else "info"
                st.markdown(f"""
                <div class="custom-metric-card {intl_color}">
                    <div class="custom-metric-label">INTERNATIONAL PRODUCT PAYI</div>
                    <div class="custom-metric-value">{intl_share:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Global Yayılım</span>
                        <span>Multi-Market</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                unique_molecules = metrics.get('Benzersiz_Molekül', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">MOLEKÜL ÇEŞİTLİLİĞİ</div>
                    <div class="custom-metric-value">{unique_molecules:,}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">Benzersiz</span>
                        <span>Farklı Molekül</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                avg_price = metrics.get('Ortalama_Fiyat', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">ORTALAMA FİYAT</div>
                    <div class="custom-metric-value">${avg_price:.2f}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Birim Başına</span>
                        <span>Ortalama</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                high_growth = metrics.get('Yüksek_Büyüyen_Ürünler', 0)
                total_products = metrics.get('Toplam_Satır', 0)
                high_growth_pct = (high_growth / total_products * 100) if total_products > 0 else 0
                st.markdown(f"""
                <div class="custom-metric-card success">
                    <div class="custom-metric-label">YÜKSEK BÜYÜME</div>
                    <div class="custom-metric-value">{high_growth_pct:.1f}%</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-success">{high_growth} ürün</span>
                        <span>> %20 büyüme</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                country_coverage = metrics.get('Ülke_Sayısı', 0)
                st.markdown(f"""
                <div class="custom-metric-card">
                    <div class="custom-metric-label">COĞRAFİ YAYILIM</div>
                    <div class="custom-metric-value">{country_coverage}</div>
                    <div class="custom-metric-trend">
                        <span class="badge badge-info">Ülke</span>
                        <span>Global Kapsam</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Metrik kartları oluşturma hatası: {str(e)}")
    
    @staticmethod
    def create_international_product_analysis(analysis_df):
        """International Product analiz grafikleri"""
        try:
            if analysis_df is None or len(analysis_df) == 0:
                return None
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('International vs Local Dağılımı', 'International Product Satış Dağılımı',
                               'Coğrafi Yayılım', 'Büyüme Karşılaştırması'),
                specs=[[{'type': 'pie'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            # International vs Local dağılımı
            intl_counts = analysis_df['International_Ürün'].value_counts()
            labels = ['International', 'Local']
            values = [intl_counts.get(True, 0), intl_counts.get(False, 0)]
            
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker_colors=['#3b82f6', '#64748b']
                ),
                row=1, col=1
            )
            
            # Satış dağılımı
            intl_sales = analysis_df[analysis_df['International_Ürün']]['Toplam_Satış'].sum()
            local_sales = analysis_df[~analysis_df['International_Ürün']]['Toplam_Satış'].sum()
            
            fig.add_trace(
                go.Bar(
                    x=['International', 'Local'],
                    y=[intl_sales, local_sales],
                    marker_color=['#3b82f6', '#64748b']
                ),
                row=1, col=2
            )
            
            # Coğrafi yayılım
            if 'Ülke_Sayısı' in analysis_df.columns:
                intl_countries = analysis_df[analysis_df['International_Ürün']]['Ülke_Sayısı']
                if len(intl_countries) > 0:
                    country_dist = intl_countries.value_counts().sort_index()
                    fig.add_trace(
                        go.Bar(
                            x=country_dist.index.astype(str),
                            y=country_dist.values,
                            marker_color='#10b981'
                        ),
                        row=2, col=1
                    )
            
            # Büyüme karşılaştırması
            if 'Ortalama_Büyüme' in analysis_df.columns:
                intl_growth = analysis_df[analysis_df['International_Ürün']]['Ortalama_Büyüme'].mean()
                local_growth = analysis_df[~analysis_df['International_Ürün']]['Ortalama_Büyüme'].mean()
                
                if not pd.isna(intl_growth) and not pd.isna(local_growth):
                    fig.add_trace(
                        go.Bar(
                            x=['International', 'Local'],
                            y=[intl_growth, local_growth],
                            marker_color=['#3b82f6', '#64748b']
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                showlegend=True,
                title_text="International Product Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"International Product grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_sales_trend_chart(df):
        """Satış trend grafikleri"""
        try:
            yearly_data = []
            
            for year in [2022, 2023, 2024]:
                sales_col = f'Sales_{year}'
                if sales_col in df.columns:
                    yearly_data.append({
                        'Yıl': str(year),
                        'Toplam_Satış': df[sales_col].sum(),
                        'Ortalama_Satış': df[sales_col].mean(),
                        'Ürün_Sayısı': (df[sales_col] > 0).sum()
                    })
            
            if len(yearly_data) < 2:
                return None
            
            yearly_df = pd.DataFrame(yearly_data)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Yıllık Toplam Satış', 'Ortalama Satış Trendi', 
                               'Ürün Sayısı Trendi', 'Büyüme Oranları'),
                specs=[[{'type': 'bar'}, {'type': 'scatter'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            fig.add_trace(
                go.Bar(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Toplam_Satış'],
                    name='Toplam Satış',
                    marker_color='#3b82f6'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Ortalama_Satış'],
                    mode='lines+markers',
                    name='Ortalama Satış',
                    line=dict(color='#8b5cf6', width=3),
                    marker=dict(size=10)
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Ürün_Sayısı'],
                    name='Ürün Sayısı',
                    marker_color='#10b981'
                ),
                row=2, col=1
            )
            
            # Büyüme oranları
            if len(yearly_df) > 1:
                growth_rates = []
                for i in range(1, len(yearly_df)):
                    growth = ((yearly_df['Toplam_Satış'].iloc[i] - yearly_df['Toplam_Satış'].iloc[i-1]) / 
                              yearly_df['Toplam_Satış'].iloc[i-1] * 100) if yearly_df['Toplam_Satış'].iloc[i-1] > 0 else 0
                    growth_rates.append(growth)
                
                fig.add_trace(
                    go.Bar(
                        x=yearly_df['Yıl'].iloc[1:],
                        y=growth_rates,
                        name='Büyüme (%)',
                        marker_color=['#ef4444' if g < 0 else '#10b981' for g in growth_rates]
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                showlegend=False,
                title_text="Satış Trendleri Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Trend grafiği oluşturma hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_market_share_analysis(df):
        """Pazar payı analiz grafikleri"""
        try:
            if 'Corporation' not in df.columns or 'Sales_2024' not in df.columns:
                return None
            
            company_sales = df.groupby('Corporation')['Sales_2024'].sum().sort_values(ascending=False)
            top_companies = company_sales.nlargest(15)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Top 15 Şirket Pazar Payı', 'Top 10 Şirket Satışları'),
                specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                column_widths=[0.4, 0.6]
            )
            
            fig.add_trace(
                go.Pie(
                    labels=top_companies.index,
                    values=top_companies.values,
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Bold
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=top_companies.values[:10],
                    y=top_companies.index[:10],
                    orientation='h',
                    marker_color='#3b82f6'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                showlegend=False,
                title_text="Pazar Konsantrasyonu Analizi",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Pazar payı grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_price_volume_analysis(df):
        """Fiyat-hacim analiz grafikleri"""
        try:
            if 'Avg_Price_2024' not in df.columns or 'Units_2024' not in df.columns:
                return None
            
            sample_df = df[
                (df['Avg_Price_2024'] > 0) & 
                (df['Units_2024'] > 0)
            ].copy()
            
            if len(sample_df) > 10000:
                sample_df = sample_df.sample(10000, random_state=42)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Fiyat-Hacim İlişkisi', 'Fiyat Dağılımı',
                               'Hacim Dağılımı', 'Fiyat-Hacim Kategorileri'),
                specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
                       [{'type': 'histogram'}, {'type': 'box'}]]
            )
            
            # Fiyat-Hacim scatter plot
            fig.add_trace(
                go.Scatter(
                    x=sample_df['Avg_Price_2024'],
                    y=sample_df['Units_2024'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=sample_df['Units_2024'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Hacim")
                    )
                ),
                row=1, col=1
            )
            
            # Fiyat dağılımı
            fig.add_trace(
                go.Histogram(
                    x=df['Avg_Price_2024'],
                    nbinsx=50,
                    marker_color='#3b82f6',
                    name='Fiyat Dağılımı'
                ),
                row=1, col=2
            )
            
            # Hacim dağılımı
            fig.add_trace(
                go.Histogram(
                    x=df['Units_2024'],
                    nbinsx=50,
                    marker_color='#10b981',
                    name='Hacim Dağılımı'
                ),
                row=2, col=1
            )
            
            # Fiyat-hacim kategorileri
            if 'Corporation' in df.columns:
                top_companies = df['Corporation'].value_counts().nlargest(5).index
                company_data = df[df['Corporation'].isin(top_companies)]
                
                fig.add_trace(
                    go.Box(
                        x=company_data['Corporation'],
                        y=company_data['Avg_Price_2024'],
                        marker_color='#8b5cf6',
                        name='Şirket Bazlı Fiyat'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Fiyat-hacim grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_price_elasticity_analysis(df):
        """Fiyat esnekliği analizi"""
        try:
            if 'Avg_Price_2024' not in df.columns or 'Units_2024' not in df.columns:
                return None
            
            # Korelasyon hesapla
            correlation = df['Avg_Price_2024'].corr(df['Units_2024'])
            
            # Elasticity segments
            df_clean = df.dropna(subset=['Avg_Price_2024', 'Units_2024'])
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'Fiyat-Hacim Korelasyonu: {correlation:.3f}', 'Fiyat Esnekliği Segmentleri'),
                specs=[[{'type': 'scatter'}, {'type': 'pie'}]]
            )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df_clean['Avg_Price_2024'],
                    y=df_clean['Units_2024'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=np.log1p(df_clean['Units_2024']),
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=df_clean['Molecule'] if 'Molecule' in df_clean.columns else None
                ),
                row=1, col=1
            )
            
            # Elasticity classification
            df_clean['price_elasticity'] = 'Nötr'
            if correlation < -0.3:
                df_clean.loc[df_clean['Avg_Price_2024'] > df_clean['Avg_Price_2024'].median(), 'price_elasticity'] = 'Elastik'
            elif correlation > 0.3:
                df_clean.loc[df_clean['Avg_Price_2024'] > df_clean['Avg_Price_2024'].median(), 'price_elasticity'] = 'Elastik Olmayan'
            
            elasticity_counts = df_clean['price_elasticity'].value_counts()
            
            fig.add_trace(
                go.Pie(
                    labels=elasticity_counts.index,
                    values=elasticity_counts.values,
                    hole=0.3,
                    marker_colors=['#3b82f6', '#10b981', '#64748b']
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                showlegend=True
            )
            
            return fig, correlation
            
        except Exception as e:
            st.warning(f"Fiyat esnekliği analiz hatası: {str(e)}")
            return None, 0

# ================================================
# 6. RAPORLAMA SİSTEMİ
# ================================================

class ProfessionalReporting:
    """Profesyonel raporlama sistemi"""
    
    @staticmethod
    def generate_excel_report(df, metrics, insights, analysis_df=None):
        """Excel raporu oluştur"""
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Ham_Veri', index=False)
                
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metrik', 'Değer'])
                metrics_df.to_excel(writer, sheet_name='Özet_Metrikler', index=False)
                
                if 'Corporation' in df.columns and 'Sales_2024' in df.columns:
                    market_share = df.groupby('Corporation')['Sales_2024'].sum().sort_values(ascending=False)
                    market_share_df = market_share.reset_index()
                    market_share_df.columns = ['Şirket', 'Satış']
                    market_share_df['Pazar_Payı'] = (market_share_df['Satış'] / market_share_df['Satış'].sum()) * 100
                    market_share_df.to_excel(writer, sheet_name='Pazar_Payı', index=False)
                
                if analysis_df is not None:
                    analysis_df.to_excel(writer, sheet_name='International_Analiz', index=False)
                
                writer.save()
            
            output.seek(0)
            return output
            
        except Exception as e:
            st.error(f"Excel rapor oluşturma hatası: {str(e)}")
            return None

# ================================================
# 7. ANA UYGULAMA
# ================================================

def main():
    """Ana uygulama fonksiyonu"""
    
    st.markdown("""
    <div class="animate-fade-in">
        <h1 class="pharma-title">💊 PHARMAANALYTICS PRO</h1>
        <p class="pharma-subtitle">
        İlaç pazarı analitik platformu - International Product analizi, gelişmiş filtreleme ve stratejik içgörüler
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Session state'leri başlat
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'international_analysis' not in st.session_state:
        st.session_state.international_analysis = None
    
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">🎛️ KONTROL PANELİ</h2>', unsafe_allow_html=True)
        
        with st.expander("📁 VERİ YÜKLEME", expanded=True):
            uploaded_file = st.file_uploader(
                "Excel/CSV Dosyası Yükleyin",
                type=['xlsx', 'xls', 'csv'],
                help="İlaç pazarı verilerinizi yükleyin"
            )
            
            if uploaded_file:
                if st.button("🚀 Veriyi Yükle & Analiz Et", type="primary", width='stretch'):
                    with st.spinner("Veri işleniyor..."):
                        processor = OptimizedDataProcessor()
                        df = processor.load_large_dataset(uploaded_file)
                        
                        if df is not None and len(df) > 0:
                            df = processor.prepare_analytics_data(df)
                            
                            st.session_state.df = df
                            st.session_state.filtered_df = df.copy()
                            
                            analytics = AdvancedPharmaAnalytics()
                            st.session_state.metrics = analytics.calculate_comprehensive_metrics(df)
                            st.session_state.insights = analytics.detect_strategic_insights(df)
                            st.session_state.international_analysis = analytics.analyze_international_products(df)
                            
                            st.success(f"✅ {len(df):,} satır başarıyla yüklendi!")
                            st.rerun()
        
        if st.session_state.df is not None:
            st.markdown("---")
            df = st.session_state.df
            
            filter_system = AdvancedFilterSystem()
            search_term, filter_config, apply_filter, clear_filter = filter_system.create_filter_sidebar(df)
            
            if apply_filter:
                with st.spinner("Filtreler uygulanıyor..."):
                    filtered_df = filter_system.apply_filters(df, search_term, filter_config)
                    st.session_state.filtered_df = filtered_df
                    
                    analytics = AdvancedPharmaAnalytics()
                    st.session_state.metrics = analytics.calculate_comprehensive_metrics(filtered_df)
                    st.session_state.insights = analytics.detect_strategic_insights(filtered_df)
                    st.session_state.international_analysis = analytics.analyze_international_products(filtered_df)
                    
                    st.success(f"✅ Filtreler uygulandı: {len(filtered_df):,} satır")
                    st.rerun()
            
            if clear_filter:
                st.session_state.filtered_df = st.session_state.df.copy()
                st.session_state.metrics = AdvancedPharmaAnalytics().calculate_comprehensive_metrics(st.session_state.df)
                st.session_state.insights = AdvancedPharmaAnalytics().detect_strategic_insights(st.session_state.df)
                st.session_state.international_analysis = AdvancedPharmaAnalytics().analyze_international_products(st.session_state.df)
                st.success("✅ Filtreler temizlendi")
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #64748b;">
        <strong>PharmaAnalytics Pro</strong><br>
        v4.0 | International Product Analytics<br>
        © 2024 Tüm hakları saklıdır.
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.df is None:
        show_welcome_screen()
        return
    
    df = st.session_state.filtered_df
    metrics = st.session_state.metrics
    insights = st.session_state.insights
    intl_analysis = st.session_state.international_analysis
    
    # YENİ TAB EKLENDİ: INTERNATIONAL PRODUCT ANALİZİ
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 GENEL BAKIŞ",
        "📈 PAZAR ANALİZİ",
        "💰 FİYAT ANALİZİ",
        "🏆 REKABET ANALİZİ",
        "🌍 INTERNATIONAL PRODUCT",
        "📑 RAPORLAMA"
    ])
    
    with tab1:
        show_overview_tab(df, metrics, insights)
    
    with tab2:
        show_market_analysis_tab(df)
    
    with tab3:
        show_price_analysis_tab(df)
    
    with tab4:
        show_competition_analysis_tab(df, metrics)
    
    with tab5:
        show_international_product_tab(df, intl_analysis, metrics)
    
    with tab6:
        show_reporting_tab(df, metrics, insights, intl_analysis)

def show_welcome_screen():
    """Hoşgeldiniz ekranını göster"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">💊</div>
            <h2 style="color: #f1f5f9; margin-bottom: 1rem;">PharmaAnalytics Pro'ya Hoşgeldiniz</h2>
            <p style="color: #cbd5e1; margin-bottom: 2rem; line-height: 1.6;">
            İlaç pazarı verilerinizi yükleyin ve güçlü analitik özelliklerin kilidini açın.<br>
            International Product analizi ile çoklu pazar stratejilerinizi optimize edin.
            </p>
            
            <div class="feature-grid">
                <div class="feature-card feature-card-blue">
                    <div class="feature-icon">🌍</div>
                    <div class="feature-title">International Product Analizi</div>
                    <div class="feature-description">Çoklu pazar ürün analizi ve strateji geliştirme</div>
                </div>
                <div class="feature-card feature-card-cyan">
                    <div class="feature-icon">📈</div>
                    <div class="feature-title">Pazar Trend Analizi</div>
                    <div class="feature-description">Derin pazar içgörüleri ve trend analizi</div>
                </div>
                <div class="feature-card feature-card-green">
                    <div class="feature-icon">💰</div>
                    <div class="feature-title">Fiyat Analizi</div>
                    <div class="feature-description">Rekabetçi fiyatlandırma ve optimizasyon analizi</div>
                </div>
                <div class="feature-card feature-card-yellow">
                    <div class="feature-icon">🏆</div>
                    <div class="feature-title">Rekabet Analizi</div>
                    <div class="feature-description">Rakiplerinizi analiz edin ve fırsatları belirleyin</div>
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(6, 182, 212, 0.1));
                        padding: 1.5rem; border-radius: var(--radius-lg); margin-top: 2rem;
                        border: 1px solid rgba(37, 99, 235, 0.3);">
                <div style="font-weight: 600; color: var(--accent-blue); margin-bottom: 0.8rem; font-size: 1.1rem;">
                    🎯 Başlamak İçin
                </div>
                <div style="color: var(--text-secondary); font-size: 0.95rem; line-height: 1.6;">
                1. Sol taraftaki panelden veri dosyanızı yükleyin<br>
                2. "Veriyi Yükle & Analiz Et" butonuna tıklayın<br>
                3. Analiz sonuçlarını görmek için tabları kullanın
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_overview_tab(df, metrics, insights):
    """Genel Bakış tab'ını göster"""
    st.markdown('<h2 class="section-title">Genel Bakış ve Performans Göstergeleri</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    viz.create_dashboard_metrics(df, metrics)
    
    st.markdown('<h3 class="subsection-title">🔍 Stratejik İçgörüler</h3>', unsafe_allow_html=True)
    
    if insights:
        insight_cols = st.columns(2)
        
        for idx, insight in enumerate(insights[:6]):
            with insight_cols[idx % 2]:
                icon = "💡"
                if insight['type'] == 'warning':
                    icon = "⚠️"
                elif insight['type'] == 'success':
                    icon = "✅"
                elif insight['type'] == 'info':
                    icon = "ℹ️"
                
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <div class="insight-icon">{icon}</div>
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-content">{insight['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if insight.get('data') is not None and not insight['data'].empty:
                    with st.expander("📋 Detaylı Liste"):
                        display_columns = []
                        for col in ['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_2023_2024']:
                            if col in insight['data'].columns:
                                display_columns.append(col)
                        
                        if display_columns:
                            st.dataframe(
                                insight['data'][display_columns].head(10),
                                use_container_width=True
                            )
    else:
        st.info("Verileriniz analiz ediliyor... Stratejik içgörüler burada görünecek.")
    
    st.markdown('<h3 class="subsection-title">📋 Veri Önizleme</h3>', unsafe_allow_html=True)
    
    preview_col1, preview_col2 = st.columns([1, 3])
    
    with preview_col1:
        rows_to_show = st.slider("Gösterilecek Satır Sayısı", 10, 1000, 100, 10, key="rows_preview")
        
        available_columns = df.columns.tolist()
        default_columns = []
        
        priority_columns = ['Molecule', 'Corporation', 'Country', 'Sales_2024', 'Growth_2023_2024']
        for col in priority_columns:
            if col in available_columns:
                default_columns.append(col)
            if len(default_columns) >= 5:
                break
        
        if len(default_columns) < 5:
            default_columns.extend([col for col in available_columns[:5] if col not in default_columns])
        
        show_columns = st.multiselect(
            "Gösterilecek Sütunlar",
            options=available_columns,
            default=default_columns[:min(5, len(default_columns))],
            key="columns_preview"
        )
    
    with preview_col2:
        if show_columns:
            st.dataframe(
                df[show_columns].head(rows_to_show),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(
                df.head(rows_to_show),
                use_container_width=True,
                height=400
            )

def show_market_analysis_tab(df):
    """Pazar Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Pazar Analizi ve Trendler</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    
    st.markdown('<h3 class="subsection-title">📈 Satış Trendleri</h3>', unsafe_allow_html=True)
    trend_fig = viz.create_sales_trend_chart(df)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("Satış trend analizi için yeterli yıllık veri bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">🧪 Molekül Bazlı Analiz</h3>', unsafe_allow_html=True)
    
    if 'Molecule' in df.columns and 'Sales_2024' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            top_molecules = df.groupby('Molecule')['Sales_2024'].sum().nlargest(15)
            fig = px.bar(
                top_molecules,
                orientation='h',
                title='Top 15 Molekül - Satış Performansı',
                color=top_molecules.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f1f5f9',
                xaxis_title='Satış (USD)',
                yaxis_title='Molekül'
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
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9',
                    xaxis_title='Büyüme Oranı (%)',
                    yaxis_title='Molekül'
                )
                st.plotly_chart(fig, use_container_width=True)

def show_price_analysis_tab(df):
    """Fiyat Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Fiyat Analizi ve Optimizasyon</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    
    st.markdown('<h3 class="subsection-title">💰 Fiyat-Hacim İlişkisi</h3>', unsafe_allow_html=True)
    price_fig = viz.create_price_volume_analysis(df)
    if price_fig:
        st.plotly_chart(price_fig, use_container_width=True)
    else:
        st.info("Fiyat-hacim analizi için yeterli veri bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">📉 Fiyat Esnekliği Analizi</h3>', unsafe_allow_html=True)
    
    elasticity_fig, correlation = viz.create_price_elasticity_analysis(df)
    if elasticity_fig:
        st.plotly_chart(elasticity_fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fiyat-Hacim Korelasyonu", f"{correlation:.3f}")
        
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
                recommendation = "Limitli Fiyat Artışı"
            st.metric("Öneri", recommendation)
    
    st.markdown('<h3 class="subsection-title">🎯 Fiyat Segmentasyonu</h3>', unsafe_allow_html=True)
    
    if 'Avg_Price_2024' in df.columns:
        price_data = df['Avg_Price_2024'].dropna()
        if len(price_data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                price_segments = pd.cut(
                    price_data,
                    bins=[0, 10, 50, 100, 500, float('inf')],
                    labels=['Ekonomik (<$10)', 'Standart ($10-$50)', 'Premium ($50-$100)', 
                           'Süper Premium ($100-$500)', 'Lüks (>$500)']
                )
                
                segment_counts = price_segments.value_counts()
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title='Fiyat Segmentleri Dağılımı',
                    color_discrete_sequence=px.colors.qualitative.Set3
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
                    df_temp = df.copy()
                    df_temp['Price_Segment'] = pd.cut(
                        df_temp['Avg_Price_2024'],
                        bins=[0, 10, 50, 100, 500, float('inf')],
                        labels=['Ekonomik', 'Standart', 'Premium', 'Süper Premium', 'Lüks']
                    )
                    
                    segment_growth = df_temp.groupby('Price_Segment')['Growth_2023_2024'].mean().dropna()
                    
                    if len(segment_growth) > 0:
                        fig = px.bar(
                            segment_growth,
                            orientation='v',
                            title='Fiyat Segmenti Bazlı Büyüme',
                            color=segment_growth.values,
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#f1f5f9',
                            xaxis_title='Fiyat Segmenti',
                            yaxis_title='Ortalama Büyüme (%)',
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig, use_container_width=True)

def show_competition_analysis_tab(df, metrics):
    """Rekabet Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">Rekabet Analizi ve Pazar Yapısı</h2>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualization()
    
    st.markdown('<h3 class="subsection-title">🏆 Pazar Payı Analizi</h3>', unsafe_allow_html=True)
    share_fig = viz.create_market_share_analysis(df)
    if share_fig:
        st.plotly_chart(share_fig, use_container_width=True)
    else:
        st.info("Pazar payı analizi için gerekli veri bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">📊 Rekabet Yoğunluğu Metrikleri</h3>', unsafe_allow_html=True)
    
    comp_cols = st.columns(4)
    
    with comp_cols[0]:
        hhi = metrics.get('HHI_İndeksi', 0)
        if hhi > 2500:
            hhi_status = "Monopolistik"
        elif hhi > 1800:
            hhi_status = "Oligopol"
        else:
            hhi_status = "Rekabetçi"
        st.metric("HHI İndeksi", f"{hhi:.0f}", hhi_status)
    
    with comp_cols[1]:
        top3_share = metrics.get('Top_3_Payı', 0)
        if top3_share > 50:
            concentration = "Yüksek"
        elif top3_share > 30:
            concentration = "Orta"
        else:
            concentration = "Düşük"
        st.metric("Top 3 Payı", f"{top3_share:.1f}%", concentration)
    
    with comp_cols[2]:
        top5_share = metrics.get('Top_5_Payı', 0)
        st.metric("Top 5 Payı", f"{top5_share:.1f}%")
    
    with comp_cols[3]:
        unique_molecules = metrics.get('Benzersiz_Molekül', 0)
        st.metric("Benzersiz Molekül", f"{unique_molecules:,}")
    
    st.markdown('<h3 class="subsection-title">📈 Şirket Performans Analizi</h3>', unsafe_allow_html=True)
    
    if 'Corporation' in df.columns and 'Sales_2024' in df.columns:
        company_metrics = df.groupby('Corporation').agg({
            'Sales_2024': ['sum', 'mean', 'count']
        }).round(2)
        
        company_metrics.columns = ['_'.join(col).strip() for col in company_metrics.columns.values]
        company_metrics = company_metrics.sort_values('Sales_2024_sum', ascending=False)
        
        with st.expander("📋 Detaylı Şirket Performans Tablosu"):
            st.dataframe(
                company_metrics.head(50),
                use_container_width=True,
                height=400
            )

def show_international_product_tab(df, analysis_df, metrics):
    """International Product Analizi tab'ını göster"""
    st.markdown('<h2 class="section-title">🌍 International Product Analizi</h2>', unsafe_allow_html=True)
    
    if analysis_df is None or len(analysis_df) == 0:
        st.warning("International Product analizi için gerekli veri bulunamadı.")
        return
    
    viz = ProfessionalVisualization()
    
    # Genel bakış metrikleri
    st.markdown('<h3 class="subsection-title">📊 International Product Genel Bakış</h3>', unsafe_allow_html=True)
    
    intl_cols = st.columns(4)
    
    with intl_cols[0]:
        intl_count = metrics.get('International_Product_Sayısı', 0)
        total_molecules = metrics.get('Benzersiz_Molekül', 0)
        intl_percentage = (intl_count / total_molecules * 100) if total_molecules > 0 else 0
        st.metric("International Product Sayısı", f"{intl_count}", f"%{intl_percentage:.1f}")
    
    with intl_cols[1]:
        intl_share = metrics.get('International_Product_Payı', 0)
        st.metric("Pazar Payı", f"%{intl_share:.1f}")
    
    with intl_cols[2]:
        avg_countries = metrics.get('Ort_Ülke_Sayısı', 0)
        st.metric("Ortalama Ülke Sayısı", f"{avg_countries:.1f}")
    
    with intl_cols[3]:
        avg_companies = metrics.get('Ort_Şirket_Sayısı', 0)
        st.metric("Ortalama Şirket Sayısı", f"{avg_companies:.1f}")
    
    # Grafik analizi
    st.markdown('<h3 class="subsection-title">📈 International Product Analiz Grafikleri</h3>', unsafe_allow_html=True)
    
    intl_fig = viz.create_international_product_analysis(analysis_df)
    if intl_fig:
        st.plotly_chart(intl_fig, use_container_width=True)
    
    # Detaylı tablo
    st.markdown('<h3 class="subsection-title">📋 International Product Detaylı Listesi</h3>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Tüm International Product'lar", "Top Performanslılar"])
    
    with tab1:
        display_columns = [
            'Molekül', 'International_Ürün', 'Toplam_Satış', 'Şirket_Sayısı',
            'Ülke_Sayısı', 'Ortalama_Fiyat', 'Ortalama_Büyüme'
        ]
        
        display_columns = [col for col in display_columns if col in analysis_df.columns]
        
        intl_df_display = analysis_df[display_columns].copy()
        
        # Formatlama
        if 'Toplam_Satış' in intl_df_display.columns:
            intl_df_display['Toplam_Satış'] = intl_df_display['Toplam_Satış'].apply(
                lambda x: f"${x/1e6:.2f}M" if not pd.isna(x) and x is not None else "N/A"
            )
        
        if 'Ortalama_Büyüme' in intl_df_display.columns:
            intl_df_display['Ortalama_Büyüme'] = intl_df_display['Ortalama_Büyüme'].apply(
                lambda x: f"{x:.1f}%" if not pd.isna(x) and x is not None else "N/A"
            )
        
        if 'Ortalama_Fiyat' in intl_df_display.columns:
            intl_df_display['Ortalama_Fiyat'] = intl_df_display['Ortalama_Fiyat'].apply(
                lambda x: f"${x:,.2f}" if not pd.isna(x) and x is not None else "N/A"
            )
        
        st.dataframe(
            intl_df_display,
            use_container_width=True,
            height=400
        )
    
    with tab2:
        top_intl = analysis_df[analysis_df['International_Ürün']].nlargest(20, 'Toplam_Satış')
        
        if len(top_intl) > 0:
            top_display_columns = [
                'Molekül', 'Toplam_Satış', 'Şirket_Sayısı', 'Ülke_Sayısı',
                'Ortalama_Büyüme'
            ]
            
            top_display_columns = [col for col in top_display_columns if col in top_intl.columns]
            
            top_intl_display = top_intl[top_display_columns].copy()
            
            # Formatlama
            if 'Toplam_Satış' in top_intl_display.columns:
                top_intl_display['Toplam_Satış'] = top_intl_display['Toplam_Satış'].apply(
                    lambda x: f"${x/1e6:.2f}M" if not pd.isna(x) and x is not None else "N/A"
                )
            
            if 'Ortalama_Büyüme' in top_intl_display.columns:
                top_intl_display['Ortalama_Büyüme'] = top_intl_display['Ortalama_Büyüme'].apply(
                    lambda x: f"{x:.1f}%" if not pd.isna(x) and x is not None else "N/A"
                )
            
            st.dataframe(
                top_intl_display,
                use_container_width=True,
                height=400
            )
    
    # Strateji önerileri
    st.markdown('<h3 class="subsection-title">🎯 Strateji Önerileri</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-card info">
            <div class="insight-title">🚀 International Product Büyüme Stratejisi</div>
            <div class="insight-content">
            1. Yüksek büyüme gösteren International Product'ları belirleyin<br>
            2. Bu ürünlerin diğer ülkelere yayılma potansiyelini değerlendirin<br>
            3. Yerel pazarlarda lider olan ürünleri International Product'a dönüştürün
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-card success">
            <div class="insight-title">💰 International Product Fiyatlandırma</div>
            <div class="insight-content">
            1. Ülke bazında fiyatlandırma stratejileri geliştirin<br>
            2. Premium segmentteki International Product'ların fiyatını optimize edin<br>
            3. Fiyat esnekliği düşük ürünlere odaklanın
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_reporting_tab(df, metrics, insights, analysis_df):
    """Raporlama tab'ını göster"""
    st.markdown('<h2 class="section-title">Raporlama ve İndirme</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-title">📊 Rapor Türleri</h3>', unsafe_allow_html=True)
    
    report_type = st.radio(
        "Rapor Türü Seçin",
        ['Excel Detaylı Rapor', 'CSV Ham Veri'],
        horizontal=True,
        key="report_type"
    )
    
    st.markdown('<h3 class="subsection-title">🛠️ Rapor Oluşturma</h3>', unsafe_allow_html=True)
    
    report_cols = st.columns(3)
    
    with report_cols[0]:
        if st.button("📈 Excel Raporu Oluştur", width='stretch', key="excel_report"):
            with st.spinner("Excel raporu oluşturuluyor..."):
                reporting = ProfessionalReporting()
                excel_report = reporting.generate_excel_report(df, metrics, insights, analysis_df)
                
                if excel_report:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="⬇️ Excel İndir",
                        data=excel_report,
                        file_name=f"pharma_raporu_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch',
                        key="download_excel"
                    )
                else:
                    st.error("Excel raporu oluşturulamadı.")
    
    with report_cols[1]:
        if st.button("🔄 Analizi Sıfırla", width='stretch', key="reset_analysis"):
            for key in ['df', 'filtered_df', 'metrics', 'insights', 'international_analysis']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with report_cols[2]:
        if st.button("💾 International Product CSV", width='stretch', key="intl_csv"):
            if analysis_df is not None:
                csv = analysis_df.to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="⬇️ CSV İndir",
                    data=csv,
                    file_name=f"international_products_{timestamp}.csv",
                    mime="text/csv",
                    width='stretch',
                    key="download_intl_csv"
                )
            else:
                st.warning("International Product analizi bulunamadı.")
    
    st.markdown('<h3 class="subsection-title">📈 Hızlı İstatistikler</h3>', unsafe_allow_html=True)
    
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.metric("Toplam Satır", f"{len(df):,}")
    
    with stat_cols[1]:
        st.metric("Toplam Sütun", len(df.columns))
    
    with stat_cols[2]:
        mem_usage = df.memory_usage(deep=True).sum()/1024**2
        st.metric("Bellek Kullanımı", f"{mem_usage:.1f} MB")
    
    with stat_cols[3]:
        intl_count = metrics.get('International_Product_Sayısı', 0)
        st.metric("International Product", intl_count)

# ================================================
# 8. UYGULAMA BAŞLATMA
# ================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        if st.button("🔄 Sayfayı Yenile", width='stretch'):
            st.rerun()
