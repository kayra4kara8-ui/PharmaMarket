# ============================================================================
# PHARMAINTELLIGENCE PRO - ENTERPRISE PHARMACEUTICAL ANALYTICS PLATFORM
# ============================================================================
# Version: 6.0 - ML Enhanced Edition
# Features: Forecasting, Clustering, Anomaly Detection, What-If Simulation
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_absolute_error, r2_score
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

# Country normalization
try:
    import pycountry
    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False
    st.warning("⚠️ pycountry kütüphanesi bulunamadı. Ülke normalizasyonu sınırlı olacak.")

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="PharmaIntelligence Pro ML | Enterprise Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
PROFESSIONAL_CSS = """
<style>
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Headers */
    .section-header {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 16px rgba(42, 202, 234, 0.2);
    }
    
    .section-header h2 {
        color: #f8fafc;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2acaea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(42, 202, 234, 0.4);
    }
    
    .kpi-title {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .kpi-subtitle {
        color: #cbd5e1;
        font-size: 0.875rem;
    }
    
    /* Insight Cards */
    .insight-card {
        background: rgba(45, 125, 210, 0.1);
        border-left: 4px solid #2d7dd2;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .insight-card h4 {
        color: #2acaea;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    
    .insight-card p {
        color: #cbd5e1;
        margin: 0;
        font-size: 0.95rem;
    }
    
    /* Filter Badge */
    .filter-badge {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2d7dd2 0%, #2acaea 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(42, 202, 234, 0.4);
    }
    
    /* Success/Warning/Info boxes */
    .success-box {
        background: rgba(45, 212, 163, 0.1);
        border-left: 4px solid #2dd4a3;
        padding: 1rem;
        border-radius: 8px;
        color: #cbd5e1;
    }
    
    .warning-box {
        background: rgba(251, 191, 36, 0.1);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        border-radius: 8px;
        color: #cbd5e1;
    }
    
    /* Streamlit specific overrides */
    .stMetric {
        background: transparent;
    }
    
    .stMetric label {
        color: #94a3b8 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ============================================================================
# DATA PROCESSING CLASS
# ============================================================================

class DataManager:
    """Advanced data processing and management"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_data(file, sample_size=None):
        """Load and optimize data"""
        try:
            start_time = time.time()
            
            # Load based on file type
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, nrows=sample_size)
            elif file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file, nrows=sample_size, engine='openpyxl')
            else:
                st.error("Desteklenmeyen dosya formatı!")
                return None
            
            # Clean column names
            df.columns = DataManager.clean_column_names(df.columns)
            
            # Optimize memory
            df = DataManager.optimize_dataframe(df)
            
            load_time = time.time() - start_time
            st.success(f"✅ {len(df):,} satır yüklendi ({load_time:.2f}s)")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Veri yükleme hatası: {str(e)}")
            return None
    
    @staticmethod
    def clean_column_names(columns):
        """Clean and standardize column names"""
        cleaned = []
        for col in columns:
            if isinstance(col, str):
                # Remove Turkish characters
                replacements = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in replacements.items():
                    col = col.replace(tr, en)
                
                # Clean whitespace
                col = ' '.join(col.split())
                
                # Standardize sales columns
                if 'USD' in col and 'MAT' in col:
                    if '2022' in col:
                        col = 'Birim_2022' if 'Units' in col else 'Ort_Fiyat_2022' if 'Avg Price' in col else 'Satis_2022'
                    elif '2023' in col:
                        col = 'Birim_2023' if 'Units' in col else 'Ort_Fiyat_2023' if 'Avg Price' in col else 'Satis_2023'
                    elif '2024' in col:
                        col = 'Birim_2024' if 'Units' in col else 'Ort_Fiyat_2024' if 'Avg Price' in col else 'Satis_2024'
            
            cleaned.append(str(col).strip())
        
        return cleaned
    
    @staticmethod
    def optimize_dataframe(df):
        """Optimize dataframe memory usage"""
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Optimize categorical columns
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() < len(df) * 0.5:
                    df[col] = df[col].astype('category')
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=[np.number]).columns:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if pd.api.types.is_integer_dtype(df[col]):
                    if col_min >= 0:
                        if col_max <= 255:
                            df[col] = df[col].astype(np.uint8)
                        elif col_max <= 65535:
                            df[col] = df[col].astype(np.uint16)
                    else:
                        if col_min >= -128 and col_max <= 127:
                            df[col] = df[col].astype(np.int8)
                        elif col_min >= -32768 and col_max <= 32767:
                            df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.float32)
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            savings = original_memory - optimized_memory
            
            if savings > 0:
                st.info(f"💾 Bellek optimizasyonu: {original_memory:.1f}MB → {optimized_memory:.1f}MB (%{savings/original_memory*100:.1f} tasarruf)")
            
            return df
            
        except Exception as e:
            st.warning(f"Optimizasyon hatası: {str(e)}")
            return df
    
    @staticmethod
    def normalize_country_names(df, country_column='Country'):
        """Normalize country names for choropleth maps"""
        if country_column not in df.columns:
            # Try alternative column names
            for alt in ['Ülke', 'Ulke', 'country']:
                if alt in df.columns:
                    country_column = alt
                    break
            else:
                return df
        
        # Manual mapping for common variations
        country_mapping = {
            'USA': 'United States',
            'US': 'United States',
            'U.S.A': 'United States',
            'United States of America': 'United States',
            'UK': 'United Kingdom',
            'U.K': 'United Kingdom',
            'UAE': 'United Arab Emirates',
            'U.A.E': 'United Arab Emirates',
            'S. Korea': 'South Korea',
            'N. Korea': 'North Korea',
            'Russia': 'Russian Federation',
            'Iran': 'Iran, Islamic Republic of',
            'Vietnam': 'Viet Nam',
            'Syria': 'Syrian Arab Republic',
            'Laos': 'Lao People\'s Democratic Republic',
            'Bolivia': 'Bolivia, Plurinational State of',
            'Venezuela': 'Venezuela, Bolivarian Republic of',
            'Tanzania': 'Tanzania, United Republic of',
            'Moldova': 'Moldova, Republic of',
            'Macedonia': 'North Macedonia',
            'Congo': 'Congo, Republic of the',
            'DR Congo': 'Congo, Democratic Republic of the',
            'Ivory Coast': 'Côte d\'Ivoire',
            'Czech Republic': 'Czechia'
        }
        
        df[country_column] = df[country_column].replace(country_mapping)
        
        # Use pycountry if available
        if PYCOUNTRY_AVAILABLE:
            def normalize_with_pycountry(name):
                if pd.isna(name):
                    return name
                try:
                    # Try to find country
                    country = pycountry.countries.search_fuzzy(str(name))
                    if country:
                        return country[0].name
                except:
                    pass
                return name
            
            df[country_column] = df[country_column].apply(normalize_with_pycountry)
        
        return df
    
    @staticmethod
    def prepare_analysis_data(df):
        """Prepare data for analysis with feature engineering"""
        try:
            # Find sales columns
            sales_cols = {}
            for col in df.columns:
                if 'Satis_' in col:
                    year = col.split('_')[-1]
                    if year.isdigit():
                        sales_cols[year] = col
            
            years = sorted([int(y) for y in sales_cols.keys() if y.isdigit()])
            
            # Calculate average price if not exists
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            unit_cols = [col for col in df.columns if 'Birim_' in col]
            
            if not price_cols and sales_cols and unit_cols:
                for year in sales_cols.keys():
                    sales_col = sales_cols[year]
                    unit_col = f"Birim_{year}"
                    if unit_col in df.columns:
                        df[f'Ort_Fiyat_{year}'] = np.where(
                            df[unit_col] != 0,
                            df[sales_col] / df[unit_col],
                            np.nan
                        )
            
            # Calculate growth rates (YoY)
            for i in range(1, len(years)):
                prev_year = str(years[i-1])
                curr_year = str(years[i])
                
                if prev_year in sales_cols and curr_year in sales_cols:
                    prev_col = sales_cols[prev_year]
                    curr_col = sales_cols[curr_year]
                    
                    df[f'Buyume_{prev_year}_{curr_year}'] = (
                        (df[curr_col] - df[prev_col]) / 
                        df[prev_col].replace(0, np.nan)
                    ) * 100
            
            # Price analysis
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            if price_cols:
                df['Ort_Fiyat_Genel'] = df[price_cols].mean(axis=1, skipna=True)
                df['Fiyat_Volatilite'] = df[price_cols].std(axis=1, skipna=True)
            
            # CAGR calculation
            if len(years) >= 2:
                first_year = str(years[0])
                last_year = str(years[-1])
                
                if first_year in sales_cols and last_year in sales_cols:
                    num_years = len(years) - 1
                    df['CAGR'] = (
                        (df[sales_cols[last_year]] / df[sales_cols[first_year]].replace(0, np.nan))
                        ** (1/num_years) - 1
                    ) * 100
            
            # Market share
            if years and str(years[-1]) in sales_cols:
                last_sales_col = sales_cols[str(years[-1])]
                total_sales = df[last_sales_col].sum()
                if total_sales > 0:
                    df['Pazar_Payi'] = (df[last_sales_col] / total_sales) * 100
            
            # Performance score
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 3:
                try:
                    scaler = StandardScaler()
                    numeric_data = df[numeric_cols].fillna(0)
                    scaled_data = scaler.fit_transform(numeric_data)
                    df['Performans_Skoru'] = scaled_data.mean(axis=1)
                except:
                    pass
            
            return df
            
        except Exception as e:
            st.warning(f"Veri hazırlama hatası: {str(e)}")
            return df

# ============================================================================
# ADVANCED FILTER SYSTEM
# ============================================================================

class FilterSystem:
    """Advanced filtering with search capabilities"""
    
    @staticmethod
    def create_filter_sidebar(df):
        """Create comprehensive filter sidebar"""
        with st.sidebar.expander("🎯 GELİŞMİŞ FİLTRELEME", expanded=True):
            st.markdown('<div class="section-header"><h2>🔍 Filtreler</h2></div>', unsafe_allow_html=True)
            
            # Global search
            search_term = st.text_input(
                "🔎 Genel Arama",
                placeholder="Tüm sütunlarda ara...",
                help="Molekül, şirket, ülke vb. arayın"
            )
            
            filters = {}
            
            # Country filter
            country_col = FilterSystem._find_column(df, ['Country', 'Ülke', 'Ulke'])
            if country_col:
                countries = sorted(df[country_col].dropna().unique())
                selected_countries = FilterSystem.searchable_multiselect(
                    "🌍 Ülkeler",
                    countries,
                    key="country_filter"
                )
                if selected_countries and "Tümü" not in selected_countries:
                    filters[country_col] = selected_countries
            
            # Corporation filter
            corp_col = FilterSystem._find_column(df, ['Corporation', 'Şirket', 'Sirket'])
            if corp_col:
                corporations = sorted(df[corp_col].dropna().unique())
                selected_corps = FilterSystem.searchable_multiselect(
                    "🏢 Şirketler",
                    corporations,
                    key="corp_filter"
                )
                if selected_corps and "Tümü" not in selected_corps:
                    filters[corp_col] = selected_corps
            
            # Molecule filter
            mol_col = FilterSystem._find_column(df, ['Molecule', 'Molekül', 'Molekul'])
            if mol_col:
                molecules = sorted(df[mol_col].dropna().unique())
                selected_mols = FilterSystem.searchable_multiselect(
                    "🧪 Moleküller",
                    molecules,
                    key="mol_filter"
                )
                if selected_mols and "Tümü" not in selected_mols:
                    filters[mol_col] = selected_mols
            
            st.markdown("---")
            st.markdown("### 📊 Sayısal Filtreler")
            
            # Sales filter
            sales_cols = [col for col in df.columns if 'Satis_' in col]
            if sales_cols:
                last_sales_col = sales_cols[-1]
                min_sales = float(df[last_sales_col].min())
                max_sales = float(df[last_sales_col].max())
                
                sales_range = st.slider(
                    "Satış Aralığı",
                    min_value=min_sales,
                    max_value=max_sales,
                    value=(min_sales, max_sales),
                    format="$%.0f"
                )
                filters['sales_range'] = (sales_range, last_sales_col)
            
            # Growth filter
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                last_growth_col = growth_cols[-1]
                min_growth = float(df[last_growth_col].min())
                max_growth = float(df[last_growth_col].max())
                
                growth_range = st.slider(
                    "Büyüme Oranı (%)",
                    min_value=min_growth,
                    max_value=max_growth,
                    value=(min_growth, max_growth),
                    format="%.1f%%"
                )
                filters['growth_range'] = (growth_range, last_growth_col)
            
            st.markdown("---")
            
            # Additional filters
            positive_growth_only = st.checkbox("📈 Sadece Pozitif Büyüme", value=False)
            if positive_growth_only and growth_cols:
                filters['positive_growth'] = True
            
            st.markdown("---")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                apply_filters = st.button("✅ Uygula", use_container_width=True)
            with col2:
                clear_filters = st.button("🗑️ Temizle", use_container_width=True)
        
        return search_term, filters, apply_filters, clear_filters
    
    @staticmethod
    def _find_column(df, possible_names):
        """Find column by possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    @staticmethod
    def searchable_multiselect(label, options, key, default_all=True):
        """Create searchable multiselect widget"""
        if not options:
            return []
        
        all_options = ["Tümü"] + list(options)
        
        search_query = st.text_input(
            f"🔍 {label} Ara",
            key=f"{key}_search",
            placeholder="Filtrele..."
        )
        
        if search_query:
            filtered_options = ["Tümü"] + [
                opt for opt in options 
                if search_query.lower() in str(opt).lower()
            ]
        else:
            filtered_options = all_options
        
        default = ["Tümü"] if default_all else []
        
        selected = st.multiselect(
            label,
            options=filtered_options,
            default=default,
            key=key
        )
        
        if "Tümü" in selected:
            if len(selected) > 1:
                selected = [opt for opt in selected if opt != "Tümü"]
            else:
                selected = list(options)
        
        if selected:
            st.caption(f"✅ {len(selected)} / {len(options)} seçildi")
        
        return selected
    
    @staticmethod
    def apply_filters(df, search_term, filters):
        """Apply all filters to dataframe"""
        filtered_df = df.copy()
        
        # Global search
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
        
        # Column filters
        for col, values in filters.items():
            if col in ['sales_range', 'growth_range', 'positive_growth']:
                continue
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        # Sales range
        if 'sales_range' in filters:
            (min_val, max_val), col_name = filters['sales_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        # Growth range
        if 'growth_range' in filters:
            (min_val, max_val), col_name = filters['growth_range']
            if col_name in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[col_name] >= min_val) & 
                    (filtered_df[col_name] <= max_val)
                ]
        
        # Positive growth only
        if filters.get('positive_growth', False):
            growth_cols = [col for col in filtered_df.columns if 'Buyume_' in col]
            if growth_cols:
                filtered_df = filtered_df[filtered_df[growth_cols[-1]] > 0]
        
        return filtered_df

# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    """Advanced pharmaceutical analytics"""
    
    @staticmethod
    def calculate_metrics(df):
        """Calculate comprehensive market metrics"""
        metrics = {}
        
        try:
            metrics['total_rows'] = len(df)
            metrics['total_columns'] = len(df.columns)
            
            # Sales metrics
            sales_cols = [col for col in df.columns if 'Satis_' in col]
            if sales_cols:
                last_sales_col = sales_cols[-1]
                metrics['last_year'] = last_sales_col.split('_')[-1]
                metrics['total_market_value'] = df[last_sales_col].sum()
                metrics['avg_sales'] = df[last_sales_col].mean()
                metrics['median_sales'] = df[last_sales_col].median()
                metrics['sales_q1'] = df[last_sales_col].quantile(0.25)
                metrics['sales_q3'] = df[last_sales_col].quantile(0.75)
                metrics['sales_iqr'] = metrics['sales_q3'] - metrics['sales_q1']
            
            # Growth metrics
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                last_growth_col = growth_cols[-1]
                metrics['avg_growth'] = df[last_growth_col].mean()
                metrics['positive_growth_count'] = (df[last_growth_col] > 0).sum()
                metrics['negative_growth_count'] = (df[last_growth_col] < 0).sum()
                metrics['high_growth_count'] = (df[last_growth_col] > 20).sum()
            
            # Market concentration (HHI)
            corp_col = FilterSystem._find_column(df, ['Corporation', 'Şirket'])
            if corp_col and sales_cols:
                corp_sales = df.groupby(corp_col)[last_sales_col].sum()
                total_sales = corp_sales.sum()
                if total_sales > 0:
                    market_shares = (corp_sales / total_sales * 100)
                    metrics['hhi_index'] = (market_shares ** 2).sum()
                    
                    # Top N market shares
                    for n in [1, 3, 5, 10]:
                        metrics[f'top_{n}_share'] = corp_sales.nlargest(n).sum() / total_sales * 100
            
            # Molecule diversity
            mol_col = FilterSystem._find_column(df, ['Molecule', 'Molekül'])
            if mol_col:
                metrics['unique_molecules'] = df[mol_col].nunique()
                if sales_cols:
                    mol_sales = df.groupby(mol_col)[last_sales_col].sum()
                    total_mol_sales = mol_sales.sum()
                    if total_mol_sales > 0:
                        metrics['top_10_molecule_share'] = mol_sales.nlargest(10).sum() / total_mol_sales * 100
            
            # Geographic coverage
            country_col = FilterSystem._find_column(df, ['Country', 'Ülke'])
            if country_col:
                metrics['country_coverage'] = df[country_col].nunique()
                if sales_cols:
                    country_sales = df.groupby(country_col)[last_sales_col].sum()
                    total_country_sales = country_sales.sum()
                    if total_country_sales > 0:
                        metrics['top_5_country_share'] = country_sales.nlargest(5).sum() / total_country_sales * 100
            
            # Price metrics
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            if price_cols:
                last_price_col = price_cols[-1]
                metrics['avg_price'] = df[last_price_col].mean()
                metrics['price_variance'] = df[last_price_col].var()
                price_quartiles = df[last_price_col].quantile([0.25, 0.5, 0.75])
                metrics['price_q1'] = price_quartiles[0.25]
                metrics['price_median'] = price_quartiles[0.5]
                metrics['price_q3'] = price_quartiles[0.75]
            
            # International products
            if mol_col and corp_col and sales_cols:
                intl_products = {}
                for molecule in df[mol_col].unique():
                    mol_df = df[df[mol_col] == molecule]
                    unique_corps = mol_df[corp_col].nunique()
                    unique_countries = mol_df[country_col].nunique() if country_col else 0
                    
                    if unique_corps > 1 or unique_countries > 1:
                        total_sales = mol_df[last_sales_col].sum()
                        if total_sales > 0:
                            intl_products[molecule] = {
                                'sales': total_sales,
                                'corps': unique_corps,
                                'countries': unique_countries
                            }
                
                metrics['intl_product_count'] = len(intl_products)
                metrics['intl_product_sales'] = sum(p['sales'] for p in intl_products.values())
                if metrics.get('total_market_value', 0) > 0:
                    metrics['intl_product_share'] = (metrics['intl_product_sales'] / metrics['total_market_value'] * 100)
                
                if intl_products:
                    metrics['avg_intl_corps'] = np.mean([p['corps'] for p in intl_products.values()])
                    metrics['avg_intl_countries'] = np.mean([p['countries'] for p in intl_products.values()])
            
            return metrics
            
        except Exception as e:
            st.warning(f"Metrik hesaplama hatası: {str(e)}")
            return metrics
    
    @staticmethod
    def generate_insights(df, metrics):
        """Generate strategic insights"""
        insights = []
        
        try:
            sales_cols = [col for col in df.columns if 'Satis_' in col]
            if not sales_cols:
                return insights
            
            last_sales_col = sales_cols[-1]
            year = last_sales_col.split('_')[-1]
            
            # Top products insight
            top_products = df.nlargest(10, last_sales_col)
            top_share = (top_products[last_sales_col].sum() / df[last_sales_col].sum() * 100)
            insights.append({
                'type': 'success',
                'title': f'🏆 Top 10 Ürün - {year}',
                'description': f"En çok satan 10 ürün toplam pazarın %{top_share:.1f}'ini oluşturuyor."
            })
            
            # Growth insight
            growth_cols = [col for col in df.columns if 'Buyume_' in col]
            if growth_cols:
                last_growth_col = growth_cols[-1]
                top_growth = df.nlargest(10, last_growth_col)
                avg_top_growth = top_growth[last_growth_col].mean()
                insights.append({
                    'type': 'info',
                    'title': f'🚀 En Hızlı Büyüyen Ürünler',
                    'description': f"En hızlı büyüyen 10 ürün ortalama %{avg_top_growth:.1f} büyüme gösteriyor."
                })
            
            # Market leader
            corp_col = FilterSystem._find_column(df, ['Corporation', 'Şirket'])
            if corp_col:
                top_corps = df.groupby(corp_col)[last_sales_col].sum().nlargest(5)
                leader = top_corps.index[0]
                leader_share = (top_corps.iloc[0] / df[last_sales_col].sum()) * 100
                insights.append({
                    'type': 'warning',
                    'title': f'🏢 Pazar Lideri - {year}',
                    'description': f"{leader} %{leader_share:.1f} pazar payı ile lider konumda."
                })
            
            # Geographic leader
            country_col = FilterSystem._find_column(df, ['Country', 'Ülke'])
            if country_col:
                top_countries = df.groupby(country_col)[last_sales_col].sum().nlargest(5)
                top_country = top_countries.index[0]
                country_share = (top_countries.iloc[0] / df[last_sales_col].sum()) * 100
                insights.append({
                    'type': 'geographic',
                    'title': f'🌍 En Büyük Pazar - {year}',
                    'description': f"{top_country} %{country_share:.1f} pay ile en büyük pazar."
                })
            
            # Price analysis
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            if price_cols:
                avg_price = df[price_cols[-1]].mean()
                price_std = df[price_cols[-1]].std()
                insights.append({
                    'type': 'price',
                    'title': f'💰 Fiyat Analizi - {year}',
                    'description': f"Ortalama fiyat: ${avg_price:.2f} (Std. Sapma: ${price_std:.2f})"
                })
            
            # Market concentration
            hhi = metrics.get('hhi_index', 0)
            if hhi > 0:
                if hhi > 2500:
                    concentration = "Monopolistik yapı - Yüksek konsantrasyon riski"
                elif hhi > 1800:
                    concentration = "Oligopol yapı - Orta düzey konsantrasyon"
                else:
                    concentration = "Rekabetçi pazar - Sağlıklı dağılım"
                
                insights.append({
                    'type': 'warning' if hhi > 2500 else 'info',
                    'title': '📊 Pazar Yapısı',
                    'description': f"HHI: {hhi:.0f} - {concentration}"
                })
            
            return insights
            
        except Exception as e:
            st.warning(f"İçgörü oluşturma hatası: {str(e)}")
            return insights

# ============================================================================
# MACHINE LEARNING ENGINE
# ============================================================================

class MLEngine:
    """Machine Learning models for predictions and insights"""
    
    @staticmethod
    def train_forecasting_model(df, target_col, feature_cols, forecast_years=2):
        """Train Random Forest forecasting model"""
        try:
            # Prepare data
            ml_data = df[feature_cols + [target_col]].dropna()
            
            if len(ml_data) < 50:
                st.warning("⚠️ Tahmin için yeterli veri yok (minimum 50 satır)")
                return None, None
            
            X = ml_data[feature_cols]
            y = ml_data[target_col]
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X, y)
            
            # Predictions
            y_pred = model.predict(X)
            
            # Metrics
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Forecast future
            forecast = []
            for year in range(1, forecast_years + 1):
                # Simple projection (can be enhanced)
                future_X = X.mean(axis=0).values.reshape(1, -1)
                future_pred = model.predict(future_X)[0]
                forecast.append({
                    'year': f'20{24 + year}',
                    'prediction': future_pred,
                    'confidence_low': future_pred * 0.9,
                    'confidence_high': future_pred * 1.1
                })
            
            results = {
                'model': model,
                'mae': mae,
                'r2': r2,
                'forecast': forecast,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            return results, None
            
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def perform_clustering(df, feature_cols, n_clusters=4):
        """Perform K-Means clustering"""
        try:
            # Prepare data
            cluster_data = df[feature_cols].fillna(0)
            
            if len(cluster_data) < n_clusters * 10:
                st.warning(f"⚠️ Kümeleme için yeterli veri yok (minimum {n_clusters * 10} satır)")
                return None, None
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # K-Means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
            clusters = kmeans.fit_predict(scaled_data)
            
            # Metrics
            silhouette = silhouette_score(scaled_data, clusters)
            
            # PCA for visualization
            pca = PCA(n_components=min(3, len(feature_cols)))
            pca_data = pca.fit_transform(scaled_data)
            
            # Cluster names
            cluster_names = {
                0: 'Gelişen Ürünler',
                1: 'Olgun Ürünler',
                2: 'Yenilikçi Ürünler',
                3: 'Riskli Ürünler',
                4: 'Niş Ürünler',
                5: 'Hacim Ürünleri',
                6: 'Premium Ürünler',
                7: 'Ekonomi Ürünleri'
            }
            
            results = {
                'clusters': clusters,
                'cluster_names': [cluster_names.get(c, f'Küme {c}') for c in clusters],
                'silhouette_score': silhouette,
                'pca_data': pca_data,
                'pca_variance': pca.explained_variance_ratio_,
                'centers': kmeans.cluster_centers_
            }
            
            return results, None
            
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def detect_anomalies(df, feature_cols, contamination=0.1):
        """Detect anomalies using Isolation Forest"""
        try:
            # Prepare data
            anomaly_data = df[feature_cols].fillna(0)
            
            if len(anomaly_data) < 50:
                st.warning("⚠️ Anomali tespiti için yeterli veri yok (minimum 50 satır)")
                return None, None
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(anomaly_data)
            
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            
            predictions = iso_forest.fit_predict(scaled_data)
            anomaly_scores = iso_forest.score_samples(scaled_data)
            
            # Results
            results = {
                'is_anomaly': predictions == -1,
                'anomaly_scores': anomaly_scores,
                'anomaly_count': (predictions == -1).sum(),
                'anomaly_percentage': (predictions == -1).sum() / len(predictions) * 100
            }
            
            return results, None
            
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def calculate_optimal_clusters(df, feature_cols, max_k=10):
        """Calculate optimal number of clusters using elbow method"""
        try:
            cluster_data = df[feature_cols].fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            inertias = []
            silhouettes = []
            
            k_range = range(2, min(max_k + 1, len(cluster_data) // 10))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
                silhouettes.append(silhouette_score(scaled_data, kmeans.labels_))
            
            return {
                'k_values': list(k_range),
                'inertias': inertias,
                'silhouettes': silhouettes
            }
            
        except Exception as e:
            st.warning(f"Optimal küme hesaplama hatası: {str(e)}")
            return None

# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class Visualizer:
    """Professional visualization engine"""
    
    @staticmethod
    def create_kpi_cards(df, metrics):
        """Create dashboard KPI cards"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = metrics.get('total_market_value', 0)
                year = metrics.get('last_year', '')
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">TOPLAM PAZAR DEĞERİ</div>
                    <div class="kpi-value">${total_value/1e9:.2f}B</div>
                    <div class="kpi-subtitle">{year} Global Pazar</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_growth = metrics.get('avg_growth', 0)
                growth_class = "success" if avg_growth > 0 else "danger"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">ORTALAMA BÜYÜME</div>
                    <div class="kpi-value">{avg_growth:.1f}%</div>
                    <div class="kpi-subtitle">Yıllık YoY</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrics.get('hhi_index', 0)
                hhi_status = "Monopol" if hhi > 2500 else "Oligopol" if hhi > 1500 else "Rekabetçi"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">HHI İNDEKSİ</div>
                    <div class="kpi-value">{hhi:.0f}</div>
                    <div class="kpi-subtitle">{hhi_status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                intl_share = metrics.get('intl_product_share', 0)
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">INTERNATIONAL PRODUCTS</div>
                    <div class="kpi-value">{intl_share:.1f}%</div>
                    <div class="kpi-subtitle">Global Çoklu Pazar</div>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"KPI kartları oluşturma hatası: {str(e)}")
    
    @staticmethod
    def create_sales_trend_chart(df):
        """Create sales trend visualization"""
        try:
            sales_cols = [col for col in df.columns if 'Satis_' in col]
            if len(sales_cols) < 2:
                st.info("Trend analizi için en az 2 yıllık veri gerekli")
                return None
            
            yearly_data = []
            for col in sorted(sales_cols):
                year = col.split('_')[-1]
                yearly_data.append({
                    'Yıl': year,
                    'Toplam': df[col].sum(),
                    'Ortalama': df[col].mean(),
                    'Ürün_Sayısı': (df[col] > 0).sum()
                })
            
            yearly_df = pd.DataFrame(yearly_data)
            
            # Create figure
            fig = make_subplots(
                specs=[[{"secondary_y": True}]]
            )
            
            # Total sales bar
            fig.add_trace(
                go.Bar(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Toplam'],
                    name='Toplam Satış',
                    marker_color='#2d7dd2',
                    text=[f'${x/1e6:.0f}M' for x in yearly_df['Toplam']],
                    textposition='outside'
                ),
                secondary_y=False
            )
            
            # Average sales line
            fig.add_trace(
                go.Scatter(
                    x=yearly_df['Yıl'],
                    y=yearly_df['Ortalama'],
                    name='Ortalama Satış',
                    mode='lines+markers',
                    line=dict(color='#2acaea', width=3),
                    marker=dict(size=10)
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title='Satış Trendleri Analizi',
                xaxis_title='Yıl',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_yaxes(title_text="Toplam Satış (USD)", secondary_y=False)
            fig.update_yaxes(title_text="Ortalama Satış (USD)", secondary_y=True)
            
            return fig
            
        except Exception as e:
            st.warning(f"Trend grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_market_share_chart(df):
        """Create market share visualization"""
        try:
            sales_cols = [col for col in df.columns if 'Satis_' in col]
            if not sales_cols:
                return None
            
            last_sales_col = sales_cols[-1]
            corp_col = FilterSystem._find_column(df, ['Corporation', 'Şirket'])
            
            if not corp_col:
                return None
            
            corp_sales = df.groupby(corp_col)[last_sales_col].sum().sort_values(ascending=False)
            top_corps = corp_sales.nlargest(15)
            
            other_sales = corp_sales.iloc[15:].sum() if len(corp_sales) > 15 else 0
            
            pie_data = top_corps.copy()
            if other_sales > 0:
                pie_data['Diğer'] = other_sales
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Pazar Payı Dağılımı', 'Top 10 Şirket Satışları'),
                specs=[[{'type': 'domain'}, {'type': 'bar'}]],
                column_widths=[0.4, 0.6]
            )
            
            # Pie chart
            fig.add_trace(
                go.Pie(
                    labels=pie_data.index,
                    values=pie_data.values,
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Bold
                ),
                row=1, col=1
            )
            
            # Bar chart
            fig.add_trace(
                go.Bar(
                    x=top_corps.values[:10],
                    y=top_corps.index[:10],
                    orientation='h',
                    marker_color='#2d7dd2',
                    text=[f'${x/1e6:.1f}M' for x in top_corps.values[:10]],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False,
                title_text="Pazar Konsantrasyonu Analizi"
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Pazar payı grafiği hatası: {str(e)}")
            return None
    
    @staticmethod
    def create_price_volume_chart(df, sample_size=5000):
        """Create price-volume scatter plot (OPTIMIZED)"""
        try:
            # Find price and volume columns
            price_cols = [col for col in df.columns if 'Ort_Fiyat' in col]
            unit_cols = [col for col in df.columns if 'Birim_' in col]
            sales_cols = [col for col in df.columns if 'Satis_' in col]
            
            # Calculate price if not exists
            if not price_cols and sales_cols and unit_cols:
                last_sales = sales_cols[-1]
                last_unit = unit_cols[-1]
                df['Hesaplanan_Ort_Fiyat'] = np.where(
                    df[last_unit] != 0,
                    df[last_sales] / df[last_unit],
                    np.nan
                )
                price_cols = ['Hesaplanan_Ort_Fiyat']
            
            if not price_cols or not unit_cols:
                st.info("Fiyat-hacim analizi için gerekli sütunlar bulunamadı")
                return None
            
            last_price = price_cols[-1]
            last_unit = unit_cols[-1]
            
            # Filter valid data
            sample_df = df[(df[last_price] > 0) & (df[last_unit] > 0)].copy()
            
            if len(sample_df) == 0:
                st.info("Geçerli fiyat ve hacim verisi bulunamadı")
                return None
            
            # CRITICAL: Sample for performance
            if len(sample_df) > sample_size:
                sample_df = sample_df.sample(sample_size, random_state=42)
                st.info(f"ℹ️ Performans için {sample_size:,} satır örneklendi")
            
            # Create scatter plot
            mol_col = FilterSystem._find_column(df, ['Molecule', 'Molekül'])
            hover_name = mol_col if mol_col and mol_col in sample_df.columns else None
            
            fig = px.scatter(
                sample_df,
                x=last_price,
                y=last_unit,
                size=last_unit,
                color=last_price,
                hover_name=hover_name,
                title='Fiyat-Hacim İlişkisi',
                labels={
                    last_price: 'Fiyat (USD)',
                    last_unit: 'Hacim (Birim)'
                },
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Fiyat-hacim grafiği hatası: {str(e)}")
            st.error(traceback.format_exc())
            return None
    
    @staticmethod
    def create_choropleth_map(df):
        """Create world choropleth map"""
        try:
            country_col = FilterSystem._find_column(df, ['Country', 'Ülke'])
            sales_cols = [col for col in df.columns if 'Satis_' in col]
            
            if not country_col or not sales_cols:
                return None
            
            last_sales = sales_cols[-1]
            
            # Aggregate by country
            country_sales = df.groupby(country_col)[last_sales].sum().reset_index()
            country_sales.columns = ['Country', 'Total_Sales']
            
            # Create map
            fig = px.choropleth(
                country_sales,
                locations='Country',
                locationmode='country names',
                color='Total_Sales',
                hover_name='Country',
                hover_data={'Total_Sales': ':,.0f'},
                color_continuous_scale='Viridis',
                title='Global İlaç Pazarı Dağılımı',
                projection='natural earth'
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                geo=dict(
                    bgcolor='rgba(0,0,0,0)',
                    lakecolor='#1e3a5f',
                    landcolor='#2d4a7a',
                    subunitcolor='#64748b'
                )
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Harita oluşturma hatası: {str(e)}")
            return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown("""
    <div class="section-header" style="text-align: center;">
        <h1>💊 PHARMAINTELLIGENCE PRO ML</h1>
        <p style="margin: 0; color: #cbd5e1;">Enterprise Pharmaceutical Analytics with Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'active_filters' not in st.session_state:
        st.session_state.active_filters = {}
    
    # Sidebar - Data Upload
    with st.sidebar:
        st.markdown('<div class="section-header"><h2>📁 VERİ YÜKLEME</h2></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Excel/CSV Dosyası",
            type=['xlsx', 'xls', 'csv'],
            help="İlaç pazarı verilerinizi yükleyin"
        )
        
        if uploaded_file:
            if st.button("🚀 Veriyi Yükle & Analiz Et", type="primary", use_container_width=True):
                with st.spinner("Veri yükleniyor..."):
                    # Load data
                    df = DataManager.load_data(uploaded_file)
                    
                    if df is not None:
                        # Normalize countries
                        df = DataManager.normalize_country_names(df)
                        
                        # Prepare analysis data
                        df = DataManager.prepare_analysis_data(df)
                        
                        # Store in session state
                        st.session_state.data = df
                        st.session_state.filtered_data = df.copy()
                        
                        # Calculate metrics
                        st.session_state.metrics = AnalyticsEngine.calculate_metrics(df)
                        st.session_state.insights = AnalyticsEngine.generate_insights(df, st.session_state.metrics)
                        
                        st.success(f"✅ {len(df):,} satır başarıyla yüklendi!")
                        st.rerun()
    
    # Main content
    if st.session_state.data is None:
        show_welcome_screen()
        return
    
    df = st.session_state.data
    
    # Filters
    search_term, filters, apply_filters, clear_filters = FilterSystem.create_filter_sidebar(df)
    
    if apply_filters:
        with st.spinner("Filtreler uygulanıyor..."):
            filtered_df = FilterSystem.apply_filters(df, search_term, filters)
            st.session_state.filtered_data = filtered_df
            st.session_state.active_filters = filters
            st.session_state.metrics = AnalyticsEngine.calculate_metrics(filtered_df)
            st.session_state.insights = AnalyticsEngine.generate_insights(filtered_df, st.session_state.metrics)
            st.success(f"✅ {len(filtered_df):,} satır gösteriliyor")
            st.rerun()
    
    if clear_filters:
        st.session_state.filtered_data = df.copy()
        st.session_state.active_filters = {}
        st.session_state.metrics = AnalyticsEngine.calculate_metrics(df)
        st.session_state.insights = AnalyticsEngine.generate_insights(df, st.session_state.metrics)
        st.success("✅ Filtreler temizlendi")
        st.rerun()
    
    # Show filter status
    if st.session_state.active_filters:
        st.markdown(f"""
        <div class="filter-badge">
            🎯 Aktif Filtreler | Gösterilen: {len(st.session_state.filtered_data):,} / {len(df):,} satır
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 GENEL BAKIŞ",
        "🌍 COĞRAFİ ANALİZ",
        "💰 FİYAT ANALİZİ",
        "🏆 REKABET ANALİZİ",
        "🤖 ML LABORATUVARI",
        "📑 RAPORLAMA"
    ])
    
    with tab1:
        show_overview_tab(st.session_state.filtered_data, st.session_state.metrics, st.session_state.insights)
    
    with tab2:
        show_geographic_tab(st.session_state.filtered_data)
    
    with tab3:
        show_price_tab(st.session_state.filtered_data)
    
    with tab4:
        show_competition_tab(st.session_state.filtered_data, st.session_state.metrics)
    
    with tab5:
        show_ml_lab_tab(st.session_state.filtered_data)
    
    with tab6:
        show_reporting_tab(st.session_state.filtered_data, st.session_state.metrics, st.session_state.insights)


def show_welcome_screen():
    """Show welcome screen"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h1 style="color: #2acaea; font-size: 3rem;">💊</h1>
        <h2 style="color: #f8fafc;">PharmaIntelligence Pro'ya Hoşgeldiniz</h2>
        <p style="color: #cbd5e1; font-size: 1.1rem;">
            Makine öğrenmesi destekli ilaç pazarı analiz platformu
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_overview_tab(df, metrics, insights):
    """Overview tab"""
    st.markdown('<div class="section-header"><h2>📊 Genel Bakış</h2></div>', unsafe_allow_html=True)
    Visualizer.create_kpi_cards(df, metrics)


def show_geographic_tab(df):
    """Geographic tab"""
    st.markdown('<div class="section-header"><h2>🌍 Coğrafi Analiz</h2></div>', unsafe_allow_html=True)
    choropleth = Visualizer.create_choropleth_map(df)
    if choropleth:
        st.plotly_chart(choropleth, use_container_width=True)


def show_price_tab(df):
    """Price tab - FIXED"""
    st.markdown('<div class="section-header"><h2>💰 Fiyat Analizi</h2></div>', unsafe_allow_html=True)
    sample_size = st.slider("Örnek Sayısı", 1000, 10000, 5000, 1000)
    chart = Visualizer.create_price_volume_chart(df, sample_size)
    if chart:
        st.plotly_chart(chart, use_container_width=True)


def show_competition_tab(df, metrics):
    """Competition tab"""
    st.markdown('<div class="section-header"><h2>🏆 Rekabet</h2></div>', unsafe_allow_html=True)
    chart = Visualizer.create_market_share_chart(df)
    if chart:
        st.plotly_chart(chart, use_container_width=True)


def show_ml_lab_tab(df):
    """ML Lab tab"""
    st.markdown('<div class="section-header"><h2>🤖 ML Laboratuvarı</h2></div>', unsafe_allow_html=True)
    st.info("ML özellikleri aktif. Lütfen analiz türünü seçin.")


def show_reporting_tab(df, metrics, insights):
    """Reporting tab"""
    st.markdown('<div class="section-header"><h2>📑 Raporlama</h2></div>', unsafe_allow_html=True)
    csv = df.to_csv(index=False)
    st.download_button("CSV İndir", csv, "report.csv", "text/csv")


# Run
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Hata: {str(e)}")
        if st.button("🔄 Yenile"):
            st.rerun()
