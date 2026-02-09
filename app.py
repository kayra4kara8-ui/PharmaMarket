# app.py - PharmaIntelligence Enterprise Dashboard v6.0
# Global Pharmaceutical Consulting Standards (McKinsey/BCG Level)
# © 2024 - All Rights Reserved

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Advanced Analytics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from scipy import stats, integrate

# Utilities
from datetime import datetime, timedelta
import json
from io import BytesIO
import time
import gc
import traceback
from typing import Dict, List, Optional, Tuple
import math
import re

# ================================================
# 1. PROFESSIONAL CONFIGURATION & STYLING
# ================================================

st.set_page_config(
    page_title="PharmaIntelligence Enterprise | Strategic Market Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PROFESSIONAL CSS - GLOBAL CONSULTING STANDARD
PROFESSIONAL_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 100%);
        color: #f8fafc;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%);
        border-left: 4px solid #2acaea;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #2acaea;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .metric-delta {
        font-size: 13px;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #2d7dd2 0%, #2acaea 100%);
        padding: 15px 25px;
        border-radius: 8px;
        margin: 20px 0;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(42, 202, 234, 0.2);
    }
    
    /* Insight Cards */
    .insight-card {
        background: rgba(45, 125, 210, 0.1);
        border-left: 3px solid #2d7dd2;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Status Badges */
    .badge-success {
        background: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .badge-warning {
        background: #f59e0b;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .badge-danger {
        background: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Streamlit Override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 58, 95, 0.5);
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #94a3b8;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2d7dd2 0%, #2acaea 100%);
        color: white;
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# ================================================
# 2. OPTIMIZED DATA PROCESSING SYSTEM (ERROR-PROOF)
# ================================================

class OptimizedDataProcessor:
    """Enterprise-grade data processing with error handling"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
    def load_large_dataset(file, sample_size=None):
        """Load large datasets with optimization"""
        try:
            start_time = time.time()
            
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, low_memory=False, nrows=sample_size)
            elif file.name.endswith(('.xlsx', '.xls')):
                if sample_size:
                    chunks = []
                    chunk_size = 50000
                    with st.spinner(f"📥 Loading dataset..."):
                        for i in range(0, sample_size, chunk_size):
                            chunk = pd.read_excel(file, skiprows=i, nrows=min(chunk_size, sample_size-i), engine='openpyxl')
                            if chunk.empty:
                                break
                            chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_excel(file, engine='openpyxl')
            
            # Optimize dataframe
            df = OptimizedDataProcessor.optimize_dataframe(df)
            
            load_time = time.time() - start_time
            st.success(f"✅ Dataset loaded: {len(df):,} rows, {len(df.columns)} columns ({load_time:.2f}s)")
            
            return df
            
        except Exception as e:
            st.error(f"BCG Matrix chart error: {str(e)}")
            return None
    
    # ==========================================
    # PRICE-VOLUME ANALYSIS (FIXED)
    # ==========================================
    
    def price_volume_analysis(self, df: pd.DataFrame) -> Optional[go.Figure]:
        """
        CRITICAL FIX: Price-Volume Analysis with guaranteed unique column names
        """
        try:
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            units_cols = [col for col in df.columns if 'Units_' in col]
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            
            # Calculate price if not present
            if not price_cols and sales_cols and units_cols:
                last_sales = sales_cols[-1]
                last_units = units_cols[-1]
                
                temp_df = df.copy()
                temp_df['Calculated_Price'] = temp_df.apply(
                    lambda row: self.utils.safe_divide(row[last_sales], row[last_units]),
                    axis=1
                )
                price_cols = ['Calculated_Price']
                df = temp_df
            
            if not price_cols or not units_cols:
                st.info("Price-volume analysis requires price and units data")
                return None
            
            last_price = price_cols[-1]
            last_units = units_cols[-1]
            
            # CRITICAL FIX: Create unique temporary columns using hash
            unique_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            temp_price = f'TempPrice_{unique_id}'
            temp_units = f'TempUnits_{unique_id}'
            
            # Create plot dataframe
            plot_df = df[[last_price, last_units]].copy()
            plot_df = plot_df.rename(columns={
                last_price: temp_price,
                last_units: temp_units
            })
            
            # Filter valid data
            plot_df = plot_df[
                (plot_df[temp_price] > 0) &
                (plot_df[temp_units] > 0)
            ].copy()
            
            if len(plot_df) == 0:
                st.info("No valid price-volume data found")
                return None
            
            # Sample if too large
            if len(plot_df) > 10000:
                plot_df = plot_df.sample(10000, random_state=42)
            
            # Add hover information
            if 'Molecule' in df.columns:
                plot_df['Product'] = df.loc[plot_df.index, 'Molecule']
                hover_name = 'Product'
            else:
                hover_name = None
            
            # Create scatter plot
            fig = px.scatter(
                plot_df,
                x=temp_price,
                y=temp_units,
                size=temp_units,
                color=temp_price,
                hover_name=hover_name,
                title='Price-Volume Relationship Analysis',
                labels={
                    temp_price: 'Average Price (USD)',
                    temp_units: 'Volume (Units)'
                },
                color_continuous_scale='Viridis',
                size_max=40
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Price-volume analysis error: {str(e)}")
            return None
    
    # ==========================================
    # FORECAST VISUALIZATION
    # ==========================================
    
    def forecast_chart(self, forecast_data: Dict) -> Optional[go.Figure]:
        """
        Visualize sales forecasting with confidence intervals
        """
        try:
            if not forecast_data:
                return None
            
            fig = go.Figure()
            
            # Historical data
            hist_df = forecast_data['historical']
            fig.add_trace(go.Scatter(
                x=hist_df['Year'],
                y=hist_df['Sales'],
                mode='lines+markers',
                name='Historical',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=10)
            ))
            
            # Linear regression forecast
            lr_data = forecast_data['linear_regression']
            fig.add_trace(go.Scatter(
                x=lr_data['years'],
                y=lr_data['forecast'],
                mode='lines+markers',
                name='Forecast (Linear)',
                line=dict(color=self.colors['success'], width=3, dash='dash'),
                marker=dict(size=8)
            ))
            
            # Confidence interval
            if 'confidence_interval' in forecast_data:
                ci = forecast_data['confidence_interval']
                
                fig.add_trace(go.Scatter(
                    x=lr_data['years'],
                    y=ci['upper'],
                    mode='lines',
                    name='Upper CI (95%)',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=lr_data['years'],
                    y=ci['lower'],
                    mode='lines',
                    name='Lower CI (95%)',
                    line=dict(width=0),
                    fillcolor='rgba(16, 185, 129, 0.2)',
                    fill='tonexty',
                    showlegend=True
                ))
            
            fig.update_layout(
                title='Sales Forecast with Confidence Intervals',
                xaxis_title='Year',
                yaxis_title='Sales (USD)',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Forecast chart error: {str(e)}")
            return None

# ================================================
# METRICS CALCULATION ENGINE
# ================================================

def calculate_comprehensive_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive metrics for dashboard
    Returns dictionary of all KPIs
    """
    metrics = {}
    utils = UtilityFunctions()
    
    try:
        sales_cols = [col for col in df.columns if 'Sales_' in col and col.split('_')[-1].isdigit()]
        growth_cols = [col for col in df.columns if 'Growth_' in col]
        
        # Basic counts
        metrics['Total_Rows'] = len(df)
        metrics['Total_Columns'] = len(df.columns)
        
        # Sales metrics
        if sales_cols:
            years = sorted([int(col.split('_')[-1]) for col in sales_cols])
            latest_year = years[-1]
            latest_sales_col = f"Sales_{latest_year}"
            
            metrics['Latest_Year'] = latest_year
            metrics['Total_Market_Value'] = df[latest_sales_col].sum()
            metrics['Avg_Product_Sales'] = df[latest_sales_col].mean()
            metrics['Median_Product_Sales'] = df[latest_sales_col].median()
            
            # Previous year for delta
            if len(years) >= 2:
                prev_year = years[-2]
                prev_sales_col = f"Sales_{prev_year}"
                metrics['Previous_Year_Value'] = df[prev_sales_col].sum()
        
        # Growth metrics
        if growth_cols:
            last_growth = growth_cols[-1]
            metrics['Avg_Growth_Rate'] = df[last_growth].mean()
            metrics['Median_Growth_Rate'] = df[last_growth].median()
            metrics['High_Growth_Percentage'] = (df[last_growth] > 20).sum() / len(df) * 100
            metrics['Declining_Percentage'] = (df[last_growth] < 0).sum() / len(df) * 100
        
        # Market concentration (HHI Index)
        if 'Company' in df.columns and sales_cols:
            company_sales = df.groupby('Company')[sales_cols[-1]].sum()
            total_sales = company_sales.sum()
            
            if total_sales > 0:
                market_shares = (company_sales / total_sales * 100)
                hhi = (market_shares ** 2).sum()
                metrics['HHI_Index'] = hhi
                
                # Top N concentration
                top3_sales = company_sales.nlargest(3).sum()
                top5_sales = company_sales.nlargest(5).sum()
                top10_sales = company_sales.nlargest(10).sum()
                
                metrics['Top_3_Share'] = (top3_sales / total_sales * 100) if total_sales > 0 else 0
                metrics['Top_5_Share'] = (top5_sales / total_sales * 100) if total_sales > 0 else 0
                metrics['Top_10_Share'] = (top10_sales / total_sales * 100) if total_sales > 0 else 0
        
        # International Product metrics
        if 'International_Product' in df.columns:
            intl_count = (df['International_Product'] == 1).sum()
            total_count = len(df)
            
            metrics['International_Count'] = intl_count
            metrics['International_Percentage'] = (intl_count / total_count * 100) if total_count > 0 else 0
            
            if sales_cols:
                intl_sales = df[df['International_Product'] == 1][sales_cols[-1]].sum()
                total_sales = df[sales_cols[-1]].sum()
                metrics['International_Share'] = (intl_sales / total_sales * 100) if total_sales > 0 else 0
        
        # Diversity metrics
        if 'Molecule' in df.columns:
            metrics['Unique_Molecules'] = df['Molecule'].nunique()
        
        if 'Company' in df.columns:
            metrics['Unique_Companies'] = df['Company'].nunique()
        
        if 'Country' in df.columns:
            metrics['Country_Coverage'] = df['Country'].nunique()
        
        if 'Therapeutic_Area' in df.columns:
            metrics['Therapeutic_Areas'] = df['Therapeutic_Area'].nunique()
        
        # Price metrics
        price_cols = [col for col in df.columns if 'Avg_Price' in col]
        if price_cols:
            last_price = price_cols[-1]
            metrics['Avg_Price'] = df[last_price].mean()
            metrics['Median_Price'] = df[last_price].median()
            metrics['Price_Std'] = df[last_price].std()
        
        # Performance metrics
        if 'Performance_Score' in df.columns:
            metrics['Avg_Performance_Score'] = df['Performance_Score'].mean()
        
    except Exception as e:
        st.warning(f"Metrics calculation warning: {str(e)}")
    
    return metrics

# ================================================
# MAIN APPLICATION FUNCTION
# ================================================

def main():
    """
    Main application entry point
    Orchestrates the entire dashboard
    """
    
    # Header
    st.markdown(f"""
    <div style='text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #1e3a5f 0%, #2d4a7a 100%); border-radius: 15px; margin-bottom: 30px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);'>
        <h1 style='color: #2acaea; font-size: 48px; font-weight: 800; margin: 0; letter-spacing: -1px;'>
            💊 {APP_CONFIG['APP_NAME']}
        </h1>
        <p style='color: #94a3b8; font-size: 18px; margin: 10px 0 5px 0;'>
            Ultimate Strategic Pharmaceutical Market Analytics Platform
        </p>
        <p style='color: #64748b; font-size: 14px; margin: 0;'>
            Version {APP_CONFIG['VERSION']} | BCG Matrix • Pareto • Forecasting • Anomaly Detection • ML Segmentation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in ss:
        ss.data = None
    if 'filtered_data' not in ss:
        ss.filtered_data = None
    if 'metrics' not in ss:
        ss.metrics = None
    if 'bcg_analysis' not in ss:
        ss.bcg_analysis = None
    if 'pareto_analysis' not in ss:
        ss.pareto_analysis = None
    if 'segmentation' not in ss:
        ss.segmentation = None
    if 'forecast' not in ss:
        ss.forecast = None
    
    # ==========================================
    # SIDEBAR - CONTROL PANEL
    # ==========================================
    
    with st.sidebar:
        st.markdown('<div class="section-header">🎛️ CONTROL PANEL</div>', unsafe_allow_html=True)
        
        # Data Upload
        with st.expander("📁 DATA UPLOAD", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload Dataset",
                type=['xlsx', 'xls', 'csv'],
                help=f"Maximum file size: {APP_CONFIG['MAX_UPLOAD_SIZE']}MB"
            )
            
            if uploaded_file:
                file_details = f"""
                **File**: {uploaded_file.name}  
                **Size**: {uploaded_file.size / 1024**2:.2f} MB  
                **Type**: {uploaded_file.type}
                """
                st.info(file_details)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🚀 Load Full Dataset", type="primary", use_container_width=True):
                        with st.spinner("Processing dataset..."):
                            processor = AdvancedDataProcessor()
                            data = processor.load_dataset(uploaded_file, sample_size=None)
                            
                            if data is not None and len(data) > 0:
                                data = processor.prepare_analysis_data(data)
                                ss.data = data
                                ss.filtered_data = data.copy()
                                
                                # Calculate metrics
                                metrics = calculate_comprehensive_metrics(data)
                                ss.metrics = metrics
                                
                                # Initialize analyses
                                analyzer = StrategicAnalysisEngine()
                                ss.bcg_analysis = analyzer.bcg_matrix_analysis(data)
                                ss.pareto_analysis = analyzer.pareto_analysis(data)
                                ss.segmentation = analyzer.product_segmentation(data)
                                ss.forecast = analyzer.forecast_sales(data)
                                
                                st.success("✅ Dataset loaded and analyzed")
                                st.rerun()
                
                with col2:
                    sample_size = st.number_input(
                        "Sample Size",
                        min_value=1000,
                        max_value=1000000,
                        value=100000,
                        step=10000,
                        help="Load a sample for faster analysis"
                    )
                    
                    if st.button("⚡ Load Sample", use_container_width=True):
                        with st.spinner(f"Loading {sample_size:,} rows..."):
                            processor = AdvancedDataProcessor()
                            data = processor.load_dataset(uploaded_file, sample_size=sample_size)
                            
                            if data is not None and len(data) > 0:
                                data = processor.prepare_analysis_data(data)
                                ss.data = data
                                ss.filtered_data = data.copy()
                                
                                metrics = calculate_comprehensive_metrics(data)
                                ss.metrics = metrics
                                
                                st.success(f"✅ Sample loaded: {len(data):,} rows")
                                st.rerun()
        
        # Filters
        if ss.data is not None:
            with st.expander("🔍 FILTERS", expanded=True):
                data = ss.data
                
                # Global search
                search_term = st.text_input(
                    "🔎 Global Search",
                    placeholder="Search across all columns...",
                    key="global_search"
                )
                
                # Country filter
                if 'Country' in data.columns:
                    countries = sorted(data['Country'].dropna().unique())
                    selected_countries = st.multiselect(
                        "🌍 Countries",
                        options=countries,
                        default=countries[:min(5, len(countries))],
                        key="country_filter"
                    )
                
                # Company filter
                if 'Company' in data.columns:
                    companies = sorted(data['Company'].dropna().unique())
                    selected_companies = st.multiselect(
                        "🏢 Companies",
                        options=companies,
                        default=companies[:min(5, len(companies))],
                        key="company_filter"
                    )
                
                # Molecule filter
                if 'Molecule' in data.columns:
                    molecules = sorted(data['Molecule'].dropna().unique())
                    
                    # Search within molecules
                    molecule_search = st.text_input(
                        "Search Molecules",
                        placeholder="Filter molecule list...",
                        key="molecule_search"
                    )
                    
                    if molecule_search:
                        filtered_molecules = [m for m in molecules if molecule_search.lower() in str(m).lower()]
                    else:
                        filtered_molecules = molecules
                    
                    selected_molecules = st.multiselect(
                        "🧪 Molecules",
                        options=filtered_molecules,
                        default=filtered_molecules[:min(3, len(filtered_molecules))],
                        key="molecule_filter"
                    )
                
                # Growth filter
                growth_cols = [col for col in data.columns if 'Growth_' in col]
                if growth_cols:
                    last_growth = growth_cols[-1]
                    
                    growth_range = st.slider(
                        f"📈 Growth Rate ({last_growth})",
                        min_value=float(data[last_growth].min()),
                        max_value=float(data[last_growth].max()),
                        value=(
                            float(data[last_growth].quantile(0.1)),
                            float(data[last_growth].quantile(0.9))
                        ),
                        key="growth_filter"
                    )
                
                # International Product filter
                if 'International_Product' in data.columns:
                    intl_filter = st.selectbox(
                        "🌍 International Product",
                        options=['All', 'International Only', 'Local Only'],
                        key="intl_filter"
                    )
                
                # Apply filters button
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Apply Filters", use_container_width=True, key="apply_filters"):
                        filtered = data.copy()
                        
                        # Apply search
                        if search_term:
                            mask = pd.Series(False, index=filtered.index)
                            for col in filtered.columns:
                                try:
                                    mask = mask | filtered[col].astype(str).str.contains(
                                        search_term, case=False, na=False
                                    )
                                except:
                                    continue
                            filtered = filtered[mask]
                        
                        # Apply country filter
                        if 'Country' in data.columns and selected_countries:
                            filtered = filtered[filtered['Country'].isin(selected_countries)]
                        
                        # Apply company filter
                        if 'Company' in data.columns and selected_companies:
                            filtered = filtered[filtered['Company'].isin(selected_companies)]
                        
                        # Apply molecule filter
                        if 'Molecule' in data.columns and selected_molecules:
                            filtered = filtered[filtered['Molecule'].isin(selected_molecules)]
                        
                        # Apply growth filter
                        if growth_cols:
                            filtered = filtered[
                                (filtered[last_growth] >= growth_range[0]) &
                                (filtered[last_growth] <= growth_range[1])
                            ]
                        
                        # Apply international filter
                        if 'International_Product' in data.columns:
                            if intl_filter == 'International Only':
                                filtered = filtered[filtered['International_Product'] == 1]
                            elif intl_filter == 'Local Only':
                                filtered = filtered[filtered['International_Product'] == 0]
                        
                        ss.filtered_data = filtered
                        st.success(f"✅ Filters applied: {len(filtered):,} rows")
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Reset Filters", use_container_width=True, key="reset_filters"):
                        ss.filtered_data = ss.data.copy()
                        st.success("✅ Filters reset")
                        st.rerun()
        
        # Settings
        with st.expander("⚙️ SETTINGS"):
            st.markdown("**Display Options**")
            
            max_display_rows = st.slider(
                "Max display rows",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key="max_display_rows"
            )
            
            chart_theme = st.selectbox(
                "Chart theme",
                options=['plotly_dark', 'plotly', 'seaborn', 'simple_white'],
                key="chart_theme"
            )
            
            st.markdown("**Analysis Options**")
            
            n_clusters = st.slider(
                "K-Means clusters",
                min_value=2,
                max_value=10,
                value=4,
                key="n_clusters"
            )
            
            forecast_periods = st.slider(
                "Forecast periods",
                min_value=1,
                max_value=24,
                value=12,
                key="forecast_periods"
            )
    
    # ==========================================
    # MAIN CONTENT AREA
    # ==========================================
    
    if ss.data is None:
        # Welcome screen
        show_welcome_screen()
        return
    
    data = ss.filtered_data
    metrics = ss.metrics
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 EXECUTIVE OVERVIEW",
        "🎯 STRATEGIC ANALYSIS",
        "🏆 COMPETITIVE INTELLIGENCE",
        "🌍 INTERNATIONAL MARKETS",
        "🔮 PREDICTIVE ANALYTICS",
        "📑 REPORTS & EXPORT"
    ])
    
    with tab1:
        show_executive_overview(data, metrics)
    
    with tab2:
        show_strategic_analysis(data, metrics)
    
    with tab3:
        show_competitive_intelligence(data, metrics)
    
    with tab4:
        show_international_markets(data, metrics)
    
    with tab5:
        show_predictive_analytics(data, metrics)
    
    with tab6:
        show_reports_export(data, metrics)

# ================================================
# TAB CONTENT FUNCTIONS
# ================================================

def show_welcome_screen():
    """Display welcome screen when no data is loaded"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 60px 40px; background: linear-gradient(135deg, rgba(30, 58, 95, 0.6) 0%, rgba(45, 74, 122, 0.4) 100%); border-radius: 20px; backdrop-filter: blur(10px);'>
            <h2 style='color: #2acaea; font-size: 36px; margin-bottom: 20px;'>
                Welcome to PharmaIntelligence Enterprise Ultimate
            </h2>
            <p style='color: #94a3b8; font-size: 18px; margin-bottom: 40px;'>
                Upload your pharmaceutical market data to unlock powerful analytics and strategic insights
            </p>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 40px;'>
                <div style='background: rgba(45, 125, 210, 0.15); padding: 25px; border-radius: 12px; border-left: 4px solid #2d7dd2;'>
                    <h3 style='color: #2d7dd2; font-size: 20px; margin-bottom: 10px;'>🎯 BCG Matrix</h3>
                    <p style='color: #94a3b8; font-size: 14px;'>Strategic portfolio classification into Stars, Cash Cows, Question Marks, and Dogs</p>
                </div>
                
                <div style='background: rgba(16, 185, 129, 0.15); padding: 25px; border-radius: 12px; border-left: 4px solid #10b981;'>
                    <h3 style='color: #10b981; font-size: 20px; margin-bottom: 10px;'>📊 Pareto Analysis</h3>
                    <p style='color: #94a3b8; font-size: 14px;'>Identify the critical 20% of products driving 80% of revenue</p>
                </div>
                
                <div style='background: rgba(245, 158, 11, 0.15); padding: 25px; border-radius: 12px; border-left: 4px solid #f59e0b;'>
                    <h3 style='color: #f59e0b; font-size: 20px; margin-bottom: 10px;'>🔮 Forecasting</h3>
                    <p style='color: #94a3b8; font-size: 14px;'>Advanced time-series forecasting with confidence intervals</p>
                </div>
                
                <div style='background: rgba(59, 130, 246, 0.15); padding: 25px; border-radius: 12px; border-left: 4px solid #3b82f6;'>
                    <h3 style='color: #3b82f6; font-size: 20px; margin-bottom: 10px;'>🎬 Action Plans</h3>
                    <p style='color: #94a3b8; font-size: 14px;'>Automated, rule-based strategic recommendations</p>
                </div>
                
                <div style='background: rgba(239, 68, 68, 0.15); padding: 25px; border-radius: 12px; border-left: 4px solid #ef4444;'>
                    <h3 style='color: #ef4444; font-size: 20px; margin-bottom: 10px;'>🚨 Anomaly Detection</h3>
                    <p style='color: #94a3b8; font-size: 14px;'>ML-powered anomaly detection to identify unusual patterns</p>
                </div>
                
                <div style='background: rgba(168, 85, 247, 0.15); padding: 25px; border-radius: 12px; border-left: 4px solid #a855f7;'>
                    <h3 style='color: #a855f7; font-size: 20px; margin-bottom: 10px;'>🧩 Segmentation</h3>
                    <p style='color: #94a3b8; font-size: 14px;'>K-Means clustering for automatic product segmentation</p>
                </div>
            </div>
            
            <div style='margin-top: 50px; padding: 30px; background: rgba(42, 202, 234, 0.1); border-radius: 12px;'>
                <h4 style='color: #2acaea; font-size: 18px; margin-bottom: 15px;'>🚀 Getting Started</h4>
                <ol style='text-align: left; color: #94a3b8; font-size: 14px; line-height: 1.8;'>
                    <li>Click on <strong>"📁 DATA UPLOAD"</strong> in the sidebar</li>
                    <li>Select your Excel (.xlsx) or CSV file</li>
                    <li>Click <strong>"🚀 Load Full Dataset"</strong> or <strong>"⚡ Load Sample"</strong></li>
                    <li>Explore the analytics tabs and generate insights</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_executive_overview(df: pd.DataFrame, metrics: Dict):
    """Executive Overview Tab"""
    st.markdown('<div class="section-header">Executive Dashboard</div>', unsafe_allow_html=True)
    
    viz = EnterpriseVisualizationEngine()
    viz.create_executive_dashboard(df, metrics)
    
    # Market trends
    st.markdown('<div class="section-header">📈 Market Performance Trends</div>', unsafe_allow_html=True)
    
    sales_cols = [col for col in df.columns if 'Sales_' in col and col.split('_')[-1].isdigit()]
    
    if len(sales_cols) >= 2:
        years = sorted([int(col.split('_')[-1]) for col in sales_cols])
        
        yearly_data = []
        for year in years:
            col = f"Sales_{year}"
            if col in df.columns:
                yearly_data.append({
                    'Year': year,
                    'Total_Sales': df[col].sum(),
                    'Avg_Sales': df[col].mean(),
                    'Products': (df[col] > 0).sum()
                })
        
        yearly_df = pd.DataFrame(yearly_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total Market Sales Evolution', 'Active Products Over Time'),
            specs=[[{'secondary_y': False}, {'type': 'bar'}]]
        )
        
        # Sales trend
        fig.add_trace(
            go.Scatter(
                x=yearly_df['Year'],
                y=yearly_df['Total_Sales'],
                mode='lines+markers',
                name='Total Sales',
                line=dict(color='#2acaea', width=3),
                marker=dict(size=10),
                text=[f'${x/1e6:.0f}M' for x in yearly_df['Total_Sales']],
                textposition='top center'
            ),
            row=1, col=1
        )
        
        # Product count
        fig.add_trace(
            go.Bar(
                x=yearly_df['Year'],
                y=yearly_df['Products'],
                name='Active Products',
                marker_color='#10b981',
                text=yearly_df['Products'],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.markdown('<div class="section-header">🔍 Dataset Preview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        display_columns = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=[col for col in ['Molecule', 'Company', 'Country', 'Sales_2024', 'CAGR', 'Market_Share'] if col in df.columns][:6],
            key="preview_columns"
        )
        
        sort_by = st.selectbox(
            "Sort by",
            options=display_columns if display_columns else df.columns.tolist(),
            key="sort_column"
        )
        
        sort_order = st.radio(
            "Sort order",
            options=['Descending', 'Ascending'],
            key="sort_order"
        )
    
    with col2:
        if display_columns:
            sorted_df = df[display_columns].sort_values(
                by=sort_by,
                ascending=(sort_order == 'Ascending')
            )
            
            st.dataframe(
                sorted_df.head(st.session_state.get('max_display_rows', 100)),
                use_container_width=True,
                height=400
            )
        else:
            st.info("Please select at least one column to display")

def show_strategic_analysis(df: pd.DataFrame, metrics: Dict):
    """Strategic Analysis Tab"""
    st.markdown('<div class="section-header">Strategic Portfolio Analysis</div>', unsafe_allow_html=True)
    
    analyzer = StrategicAnalysisEngine()
    viz = EnterpriseVisualizationEngine()
    
    # BCG Matrix
    st.markdown('<div class="section-header">📊 BCG Matrix (Growth-Share Matrix)</div>', unsafe_allow_html=True)
    
    bcg_df = ss.bcg_analysis
    
    if bcg_df is not None and len(bcg_df) > 0:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            bcg_chart = viz.bcg_matrix_chart(bcg_df)
            if bcg_chart:
                st.plotly_chart(bcg_chart, use_container_width=True, config={'displayModeBar': True})
        
        with col2:
            st.markdown("**📈 Category Distribution**")
            
            category_counts = bcg_df['BCG_Category'].value_counts()
            total_products = len(bcg_df)
            
            for category in ['Star', 'Cash Cow', 'Question Mark', 'Dog']:
                if category in category_counts.index:
                    count = category_counts[category]
                    pct = (count / total_products * 100)
                    
                    color_map = {
                        'Star': '#10b981',
                        'Cash Cow': '#3b82f6',
                        'Question Mark': '#f59e0b',
                        'Dog': '#ef4444'
                    }
                    
                    icon_map = {
                        'Star': '⭐',
                        'Cash Cow': '💰',
                        'Question Mark': '❓',
                        'Dog': '🐕'
                    }
                    
                    st.markdown(f"""
                    <div style='background: rgba(30, 58, 95, 0.4); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid {color_map[category]};'>
                        <h4 style='color: {color_map[category]}; margin: 0; font-size: 16px;'>{icon_map[category]} {category}s</h4>
                        <p style='color: #f8fafc; margin: 5px 0; font-size: 24px; font-weight: 700;'>{count}</p>
                        <p style='color: #94a3b8; margin: 0; font-size: 12px;'>{pct:.1f}% of portfolio</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show product lists
            st.markdown("**📋 Product Lists**")
            
            for category in ['Star', 'Cash Cow', 'Question Mark', 'Dog']:
                products = bcg_df[bcg_df['BCG_Category'] == category]
                if len(products) > 0 and 'Product_Name' in products.columns:
                    with st.expander(f"{category}s ({len(products)} products)"):
                        product_list = products['Product_Name'].tolist()[:10]
                        for p in product_list:
                            st.write(f"• {p}")
                        if len(products) > 10:
                            st.caption(f"...and {len(products) - 10} more")
    else:
        st.info("BCG Matrix analysis requires sales and growth data. Please ensure your dataset contains the necessary columns.")
    
    # Pareto Analysis
    st.markdown('<div class="section-header">📊 Pareto Analysis (80/20 Rule)</div>', unsafe_allow_html=True)
    
    pareto_df = ss.pareto_analysis
    
    if pareto_df is not None and len(pareto_df) > 0:
        st.success(f"🎯 **{len(pareto_df)} products** generate **80%** of total revenue ({len(pareto_df)/len(df)*100:.1f}% of products)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Pareto chart
            sales_cols = [col for col in pareto_df.columns if 'Sales_' in col]
            if sales_cols:
                fig = go.Figure()
                
                # Cumulative line
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(pareto_df) + 1)),
                    y=pareto_df['Cumulative_Pct'],
                    mode='lines',
                    name='Cumulative %',
                    line=dict(color='#2acaea', width=3),
                    yaxis='y2'
                ))
                
                # Individual bars
                fig.add_trace(go.Bar(
                    x=list(range(1, len(pareto_df) + 1)),
                    y=pareto_df[sales_cols[0]],
                    name='Sales',
                    marker_color='#2d7dd2'
                ))
                
                # 80% line
                fig.add_hline(
                    y=80,
                    line_dash="dash",
                    line_color="#ef4444",
                    opacity=0.7,
                    annotation_text="80%",
                    annotation_position="right",
                    yref='y2'
                )
                
                fig.update_layout(
                    title='Pareto Chart: Revenue Concentration',
                    xaxis_title='Product Rank',
                    yaxis_title='Sales (USD)',
                    yaxis2=dict(
                        title='Cumulative %',
                        overlaying='y',
                        side='right',
                        range=[0, 100]
                    ),
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#f8fafc'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top Revenue Drivers**")
            
            if 'Product_Name' in pareto_df.columns and sales_cols:
                top_10 = pareto_df[['Product_Name', sales_cols[0], 'Individual_Pct']].head(10)
                top_10.columns = ['Product', 'Sales', 'Share %']
                
                # Format sales
                top_10['Sales'] = top_10['Sales'].apply(lambda x: f"${x/1e6:.1f}M")
                top_10['Share %'] = top_10['Share %'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(top_10, use_container_width=True, hide_index=True)
    
    # Action Plan
    st.markdown('<div class="section-header">🎬 Automated Strategic Action Plan</div>', unsafe_allow_html=True)
    
    recommendations = analyzer.generate_action_plan(df, bcg_df)
    
    if recommendations:
        # Filter by priority
        priority_filter = st.multiselect(
            "Filter by priority",
            options=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
            default=['CRITICAL', 'HIGH'],
            key="action_plan_filter"
        )
        
        filtered_recs = [r for r in recommendations if r['priority'] in priority_filter]
        
        for rec in filtered_recs[:15]:
            priority_colors = {
                'CRITICAL': '#ef4444',
                'HIGH': '#f59e0b',
                'MEDIUM': '#3b82f6',
                'LOW': '#64748b'
            }
            
            priority_icons = {
                'CRITICAL': '🚨',
                'HIGH': '⚠️',
                'MEDIUM': 'ℹ️',
                'LOW': '📌'
            }
            
            color = priority_colors.get(rec['priority'], '#64748b')
            icon = priority_icons.get(rec['priority'], '•')
            
            st.markdown(f"""
            <div class="insight-card">
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                    <span class='badge' style='background: {color};'>{icon} {rec['priority']}</span>
                    <div style='display: flex; gap: 10px;'>
                        <span style='color: #94a3b8; font-size: 12px; background: rgba(45, 125, 210, 0.2); padding: 4px 10px; border-radius: 8px;'>
                            {rec['category']}
                        </span>
                        <span style='color: #64748b; font-size: 12px;'>
                            {rec.get('timeframe', 'TBD')}
                        </span>
                    </div>
                </div>
                <p style='margin: 10px 0 0 0; color: #f8fafc; line-height: 1.6;'>{rec['action']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recommendations generated. Ensure your dataset contains sufficient data for analysis.")

def show_competitive_intelligence(df: pd.DataFrame, metrics: Dict):
    """Competitive Intelligence Tab"""
    st.markdown('<div class="section-header">Competitive Market Intelligence</div>', unsafe_allow_html=True)
    
    viz = EnterpriseVisualizationEngine()
    
    # Molecule Comparison
    if 'Molecule' in df.columns:
        st.markdown('<div class="section-header">🧪 Molecule Head-to-Head Comparison</div>', unsafe_allow_html=True)
        
        molecules = sorted(df['Molecule'].dropna().unique())
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_molecules = st.multiselect(
                "Select 2 or more molecules to compare",
                options=molecules,
                default=molecules[:min(3, len(molecules))],
                key="molecule_comparison_select"
            )
        
        with col2:
            comparison_type = st.radio(
                "Comparison type",
                options=['Side-by-Side', 'Radar Chart'],
                key="comparison_type"
            )
        
        if len(selected_molecules) >= 2:
            if comparison_type == 'Side-by-Side':
                comparison_chart = viz.molecule_comparison_chart(df, selected_molecules)
                if comparison_chart:
                    st.plotly_chart(comparison_chart, use_container_width=True)
            else:
                # TODO: Add radar chart implementation
                st.info("Radar chart visualization coming soon")
    
    # Company Battle
    if 'Company' in df.columns:
        st.markdown('<div class="section-header">⚔️ Company Market Share Battle</div>', unsafe_allow_html=True)
        
        companies = sorted(df['Company'].dropna().unique())
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            company1 = st.selectbox(
                "Company 1",
                options=companies,
                index=0,
                key="battle_company1"
            )
        
        with col2:
            company2 = st.selectbox(
                "Company 2",
                options=companies,
                index=min(1, len(companies) - 1),
                key="battle_company2"
            )
        
        with col3:
            if st.button("⚔️ Compare", use_container_width=True, type="primary"):
                if company1 != company2:
                    battle_chart = viz.company_battle_chart(df, company1, company2)
                    if battle_chart:
                        st.plotly_chart(battle_chart, use_container_width=True)
                else:
                    st.warning("Please select two different companies")
    
    # Price-Volume Dynamics
    st.markdown('<div class="section-header">💰 Price-Volume Dynamics</div>', unsafe_allow_html=True)
    
    pv_chart = viz.price_volume_analysis(df)
    if pv_chart:
        st.plotly_chart(pv_chart, use_container_width=True)
    else:
        st.info("Price-volume analysis requires price and units data")

def show_international_markets(df: pd.DataFrame, metrics: Dict):
    """International Markets Tab"""
    st.markdown('<div class="section-header">🌍 International Market Analysis</div>', unsafe_allow_html=True)
    
    if 'International_Product' not in df.columns:
        st.warning("⚠️ International Product column not found. Please check your dataset or use the column mapping feature in the sidebar.")
        return
    
    intl_df = df[df['International_Product'] == 1].copy()
    local_df = df[df['International_Product'] == 0].copy()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        intl_count = len(intl_df)
        total_count = len(df)
        intl_pct = (intl_count / total_count * 100) if total_count > 0 else 0
        
        st.metric(
            "International Products",
            f"{intl_count:,}",
            f"{intl_pct:.1f}%"
        )
    
    with col2:
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        if sales_cols:
            last_sales = sales_cols[-1]
            intl_sales = intl_df[last_sales].sum()
            total_sales = df[last_sales].sum()
            intl_share = (intl_sales / total_sales * 100) if total_sales > 0 else 0
            
            st.metric(
                "Revenue Share",
                f"{intl_share:.1f}%",
                f"${intl_sales/1e6:.1f}M"
            )
    
    with col3:
        if 'Country' in intl_df.columns:
            intl_countries = intl_df['Country'].nunique()
            total_countries = df['Country'].nunique()
            
            st.metric(
                "Countries (Intl)",
                f"{intl_countries}",
                f"of {total_countries}"
            )
    
    with col4:
        if 'Molecule' in intl_df.columns:
            intl_molecules = intl_df['Molecule'].nunique()
            total_molecules = df['Molecule'].nunique()
            
            st.metric(
                "Molecules (Intl)",
                f"{intl_molecules}",
                f"of {total_molecules}"
            )
    
    # Comparison charts
    st.markdown('<div class="section-header">📊 International vs. Local Comparison</div>', unsafe_allow_html=True)
    
    if sales_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales distribution
            fig = go.Figure()
            
            intl_total = intl_df[last_sales].sum()
            local_total = local_df[last_sales].sum()
            
            fig.add_trace(go.Bar(
                x=['International', 'Local'],
                y=[intl_total, local_total],
                marker=dict(
                    color=['#2d7dd2', '#64748b'],
                    line=dict(color='#f8fafc', width=2)
                ),
                text=[f'${intl_total/1e6:.1f}M', f'${local_total/1e6:.1f}M'],
                textposition='auto',
                textfont=dict(size=14, color='#f8fafc', weight='bold')
            ))
            
            fig.update_layout(
                title='Total Sales Distribution',
                yaxis_title='Sales (USD)',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Product count pie
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=['International', 'Local'],
                values=[len(intl_df), len(local_df)],
                marker=dict(colors=['#2d7dd2', '#64748b']),
                hole=0.5,
                textinfo='percent+label',
                textfont=dict(size=14, color='#f8fafc')
            ))
            
            fig.update_layout(
                title='Product Distribution',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # International products table
    if len(intl_df) > 0:
        st.markdown('<div class="section-header">📋 International Product Catalog</div>', unsafe_allow_html=True)
        
        display_cols = []
        for col in ['Molecule', 'Company', 'Country', 'Sales_2024', 'CAGR', 'Market_Share', 'Avg_Price_2024']:
            if col in intl_df.columns:
                display_cols.append(col)
        
        if display_cols:
            st.dataframe(
                intl_df[display_cols].head(st.session_state.get('max_display_rows', 100)),
                use_container_width=True,
                height=400
            )
    else:
        st.info("No international products found in the filtered dataset")

def show_predictive_analytics(df: pd.DataFrame, metrics: Dict):
    """Predictive Analytics Tab"""
    st.markdown('<div class="section-header">🔮 Predictive Analytics & Forecasting</div>', unsafe_allow_html=True)
    
    analyzer = StrategicAnalysisEngine()
    viz = EnterpriseVisualizationEngine()
    
    # Sales Forecasting
    st.markdown('<div class="section-header">📈 Sales Forecasting</div>', unsafe_allow_html=True)
    
    forecast_data = ss.forecast
    
    if forecast_data:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            forecast_chart = viz.forecast_chart(forecast_data)
            if forecast_chart:
                st.plotly_chart(forecast_chart, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Forecast Summary**")
            
            # Extract forecast values
            lr_forecast = forecast_data['linear_regression']['forecast']
            years = forecast_data['linear_regression']['years']
            
            # Display forecast values
            for i, (year, value) in enumerate(zip(years, lr_forecast)):
                if 'confidence_interval' in forecast_data:
                    ci_lower = forecast_data['confidence_interval']['lower'][i]
                    ci_upper = forecast_data['confidence_interval']['upper'][i]
                    
                    st.markdown(f"""
                    <div style='background: rgba(30, 58, 95, 0.4); padding: 12px; border-radius: 8px; margin: 8px 0;'>
                        <p style='color: #2acaea; margin: 0; font-size: 14px; font-weight: 600;'>{int(year)}</p>
                        <p style='color: #f8fafc; margin: 5px 0; font-size: 20px; font-weight: 700;'>${value/1e6:.1f}M</p>
                        <p style='color: #94a3b8; margin: 0; font-size: 11px;'>CI: ${ci_lower/1e6:.1f}M - ${ci_upper/1e6:.1f}M</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Forecasting requires at least 3 years of historical sales data")
    
    # Product Segmentation
    st.markdown('<div class="section-header">🧩 Product Segmentation (K-Means Clustering)</div>', unsafe_allow_html=True)
    
    segmentation = ss.segmentation
    
    if segmentation is not None and len(segmentation) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Segment distribution
            segment_counts = segmentation['Segment_Label'].value_counts()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=segment_counts.index,
                y=segment_counts.values,
                marker=dict(
                    color=['#2d7dd2', '#10b981', '#f59e0b', '#ef4444'][:len(segment_counts)],
                    line=dict(color='#f8fafc', width=2)
                ),
                text=segment_counts.values,
                textposition='auto',
                textfont=dict(size=16, color='#f8fafc', weight='bold')
            ))
            
            fig.update_layout(
                title='Product Segment Distribution',
                xaxis_title='Segment',
                yaxis_title='Number of Products',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**🏷️ Segment Descriptions**")
            
            for segment in segment_counts.index:
                count = segment_counts[segment]
                pct = (count / len(segmentation) * 100)
                
                st.markdown(f"""
                <div style='background: rgba(45, 125, 210, 0.15); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2d7dd2;'>
                    <h4 style='color: #2acaea; margin: 0; font-size: 14px;'>{segment}</h4>
                    <p style='color: #f8fafc; margin: 5px 0; font-size: 18px; font-weight: 600;'>{count} products</p>
                    <p style='color: #94a3b8; margin: 0; font-size: 11px;'>{pct:.1f}% of portfolio</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Product segmentation requires multiple numeric features (sales, growth, price, etc.)")
    
    # Anomaly Detection
    st.markdown('<div class="section-header">🚨 Anomaly Detection</div>', unsafe_allow_html=True)
    
    with st.expander("⚙️ Configure Anomaly Detection"):
        selected_features = st.multiselect(
            "Select features for anomaly detection",
            options=df.select_dtypes(include=[np.number]).columns.tolist(),
            default=[col for col in ['Sales_2024', 'CAGR', 'Market_Share'] if col in df.columns][:3],
            key="anomaly_features"
        )
        
        contamination = st.slider(
            "Contamination factor (expected % of anomalies)",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            key="contamination"
        )
        
        if st.button("🔍 Detect Anomalies", type="primary"):
            if selected_features:
                anomaly_df = analyzer.anomaly_detection(df, features=selected_features)
                
                if anomaly_df is not None:
                    anomaly_count = (anomaly_df['Is_Anomaly'] == 1).sum()
                    
                    st.success(f"✅ Detected {anomaly_count} anomalies ({anomaly_count/len(anomaly_df)*100:.2f}%)")
                    
                    # Show anomalies
                    if anomaly_count > 0:
                        anomalies = anomaly_df[anomaly_df['Is_Anomaly'] == 1]
                        
                        if 'Product_Name' in anomalies.columns:
                            st.markdown("**🔴 Detected Anomalies:**")
                            
                            display_cols = ['Product_Name'] + selected_features + ['Anomaly_Score']
                            st.dataframe(
                                anomalies[display_cols].sort_values('Anomaly_Score').head(20),
                                use_container_width=True
                            )
            else:
                st.warning("Please select at least 2 features for anomaly detection")

def show_reports_export(df: pd.DataFrame, metrics: Dict):
    """Reports & Export Tab"""
    st.markdown('<div class="section-header">📑 Reports & Data Export</div>', unsafe_allow_html=True)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Full Dataset Export**")
        
        if st.button("📥 Download CSV", use_container_width=True):
            csv = df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.download_button(
                label="⬇️ Download CSV File",
                data=csv,
                file_name=f"pharma_dataset_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if st.button("📥 Download Excel", use_container_width=True):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.download_button(
                label="⬇️ Download Excel File",
                data=output.getvalue(),
                file_name=f"pharma_dataset_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    with col2:
        st.markdown("**🌍 International Products**")
        
        if 'International_Product' in df.columns:
            intl_df = df[df['International_Product'] == 1]
            
            if len(intl_df) > 0:
                if st.button("📥 Export International", use_container_width=True):
                    csv = intl_df.to_csv(index=False)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name=f"international_products_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("No international products found")
        else:
            st.info("International Product column not available")
    
    with col3:
        st.markdown("**🎯 Strategic Analysis**")
        
        if ss.bcg_analysis is not None:
            if st.button("📥 Export BCG Analysis", use_container_width=True):
                csv = ss.bcg_analysis.to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"bcg_analysis_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("BCG analysis not available")
    
    # Dataset statistics
    st.markdown('<div class="section-header">📈 Dataset Statistics</div>', unsafe_allow_html=True)
    
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    
    with stat_col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with stat_col2:
        st.metric("Total Columns", len(df.columns))
    
    with stat_col3:
        memory = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory:.1f} MB")
    
    with stat_col4:
        intl_count = (df['International_Product'] == 1).sum() if 'International_Product' in df.columns else 0
        st.metric("International", intl_count)
    
    with stat_col5:
        if st.button("🔄 Reset Application", use_container_width=True):
            # Clear all session state
            for key in list(ss.keys()):
                del ss[key]
            st.rerun()
    
    # Data quality report
    st.markdown('<div class="section-header">🔍 Data Quality Report</div>', unsafe_allow_html=True)
    
    quality_data = []
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df) * 100)
        unique_count = df[col].nunique()
        dtype = str(df[col].dtype)
        
        quality_data.append({
            'Column': col,
            'Type': dtype,
            'Null Count': null_count,
            'Null %': f"{null_pct:.2f}%",
            'Unique Values': unique_count,
            'Completeness': f"{100-null_pct:.2f}%"
        })
    
    quality_df = pd.DataFrame(quality_data)
    
    st.dataframe(quality_df, use_container_width=True, height=400)

# ================================================
# APPLICATION ENTRY POINT
# ================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        
        with st.expander("🔍 Error Details", expanded=True):
            st.code(traceback.format_exc())
        
        if st.button("🔄 Restart Application", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()"❌ Data loading error: {str(e)}")
            st.error(f"Detail: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def clean_column_names(columns):
        """
        CRITICAL FIX: Clean column names and ensure uniqueness
        This fixes the "Expected unique column names, got: 'Bölge' 2 times" error
        """
        cleaned = []
        seen = {}
        
        for col in columns:
            if isinstance(col, str):
                # Turkish character mapping
                tr_to_en = {
                    'İ': 'I', 'ı': 'i', 'Ş': 'S', 'ş': 's',
                    'Ğ': 'G', 'ğ': 'g', 'Ü': 'U', 'ü': 'u',
                    'Ö': 'O', 'ö': 'o', 'Ç': 'C', 'ç': 'c'
                }
                for tr, en in tr_to_en.items():
                    col = col.replace(tr, en)
                
                # Clean whitespace
                col = col.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                col = ' '.join(col.split()).strip()
                
                # COLUMN MAPPING - Standard naming
                mapping = {
                    "MAT Q3 2022 USD MNF": "Sales_2022",
                    "MAT Q3 2023 USD MNF": "Sales_2023",
                    "MAT Q3 2024 USD MNF": "Sales_2024",
                    "MAT Q3 2022 Units": "Units_2022",
                    "MAT Q3 2023 Units": "Units_2023",
                    "MAT Q3 2024 Units": "Units_2024",
                    "MAT Q3 2022 Unit Avg Price USD MNF": "Avg_Price_2022",
                    "MAT Q3 2023 Unit Avg Price USD MNF": "Avg_Price_2023",
                    "MAT Q3 2024 Unit Avg Price USD MNF": "Avg_Price_2024",
                    "Source.Name": "Source",
                    "Corporation": "Company",
                    "Manufacturer": "Manufacturer",
                    "Molecule": "Molecule",
                    "Molecule List": "Molecule_List",
                    "Chemical Salt": "Chemical_Salt",
                    "International Product": "International_Product",
                    "International Pack": "International_Pack",
                    "Specialty Product": "Specialty_Product",
                    "Region": "Region",
                    "Sub-Region": "Sub_Region",
                    "Country": "Country",
                    "Sector": "Sector",
                    "Panel": "Panel"
                }
                
                # Apply exact match first
                if col in mapping:
                    col = mapping[col]
                # Then partial match for variants
                else:
                    for key, value in mapping.items():
                        if key in col:
                            col = value
                            break
            
            # CRITICAL: Ensure uniqueness
            base_col = str(col).strip()
            if base_col in seen:
                seen[base_col] += 1
                col = f"{base_col}_{seen[base_col]}"
            else:
                seen[base_col] = 0
                col = base_col
            
            cleaned.append(col)
        
        return cleaned
    
    @staticmethod
    def optimize_dataframe(df):
        """Optimize DataFrame memory usage"""
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            
            # Clean column names - CRITICAL FIX
            df.columns = OptimizedDataProcessor.clean_column_names(df.columns)
            
            with st.spinner("Optimizing dataset..."):
                # Categorical columns
                for col in df.select_dtypes(include=['object']).columns:
                    unique_count = df[col].nunique()
                    total_count = len(df)
                    if unique_count < total_count * 0.5:
                        df[col] = df[col].astype('category')
                
                # Numeric columns
                for col in df.select_dtypes(include=[np.number]).columns:
                    try:
                        col_min = df[col].min()
                        col_max = df[col].max()
                        
                        if pd.api.types.is_integer_dtype(df[col]):
                            if col_min >= 0:
                                if col_max <= 255:
                                    df[col] = df[col].astype(np.uint8)
                                elif col_max <= 65535:
                                    df[col] = df[col].astype(np.uint16)
                                elif col_max <= 4294967295:
                                    df[col] = df[col].astype(np.uint32)
                            else:
                                if col_min >= -128 and col_max <= 127:
                                    df[col] = df[col].astype(np.int8)
                                elif col_min >= -32768 and col_max <= 32767:
                                    df[col] = df[col].astype(np.int16)
                        else:
                            df[col] = df[col].astype(np.float32)
                    except:
                        continue
                
                # String cleaning
                for col in df.select_dtypes(include=['object', 'category']).columns:
                    try:
                        df[col] = df[col].astype(str).str.strip()
                    except:
                        pass
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
            savings = original_memory - optimized_memory
            
            if savings > 0:
                st.success(f"💾 Memory optimized: {original_memory:.1f}MB → {optimized_memory:.1f}MB ({savings/original_memory*100:.1f}% saved)")
            
            return df
            
        except Exception as e:
            st.warning(f"Optimization error: {str(e)}")
            return df
    
    @staticmethod
    def prepare_analysis_data(df):
        """
        Prepare data for analysis with calculated metrics
        CRITICAL: Handles International Product detection
        """
        try:
            # Find sales columns
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            
            if not sales_cols:
                st.warning("⚠️ No sales columns found. Please check your data structure.")
                return df
            
            # Extract years
            years = []
            for col in sales_cols:
                try:
                    year = col.split('_')[-1]
                    if year.isdigit():
                        years.append(int(year))
                except:
                    continue
            
            years = sorted(set(years))
            
            # Calculate growth rates
            for i in range(1, len(years)):
                prev_year = str(years[i-1])
                curr_year = str(years[i])
                prev_col = f"Sales_{prev_year}"
                curr_col = f"Sales_{curr_year}"
                
                if prev_col in df.columns and curr_col in df.columns:
                    df[f'Growth_{prev_year}_{curr_year}'] = np.where(
                        df[prev_col] != 0,
                        ((df[curr_col] - df[prev_col]) / df[prev_col]) * 100,
                        np.nan
                    )
            
            # Calculate CAGR
            if len(years) >= 2:
                first_year = str(years[0])
                last_year = str(years[-1])
                first_col = f"Sales_{first_year}"
                last_col = f"Sales_{last_year}"
                
                if first_col in df.columns and last_col in df.columns:
                    n_years = len(years) - 1
                    df['CAGR'] = np.where(
                        df[first_col] > 0,
                        ((df[last_col] / df[first_col]) ** (1/n_years) - 1) * 100,
                        np.nan
                    )
            
            # Market share
            if years:
                last_year = str(years[-1])
                last_sales_col = f"Sales_{last_year}"
                if last_sales_col in df.columns:
                    total_sales = df[last_sales_col].sum()
                    if total_sales > 0:
                        df['Market_Share'] = (df[last_sales_col] / total_sales) * 100
            
            # CRITICAL: International Product Detection
            df = OptimizedDataProcessor.detect_international_product(df)
            
            # Calculate average prices if not present
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if not price_cols:
                for year in years:
                    sales_col = f"Sales_{year}"
                    units_col = f"Units_{year}"
                    if sales_col in df.columns and units_col in df.columns:
                        df[f'Avg_Price_{year}'] = np.where(
                            df[units_col] > 0,
                            df[sales_col] / df[units_col],
                            np.nan
                        )
            
            # Performance score (for segmentation)
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
            st.warning(f"Analysis data preparation error: {str(e)}")
            return df
    
    @staticmethod
    def detect_international_product(df):
        """
        CRITICAL FIX: Intelligent International Product detection
        Handles missing columns with user mapping interface
        """
        # Check for existing International Product column
        intl_cols = [col for col in df.columns if 'International' in col and 'Product' in col]
        
        if intl_cols:
            # Use the first matching column
            main_col = intl_cols[0]
            if main_col != 'International_Product':
                df['International_Product'] = df[main_col]
            
            # Convert to binary
            if df['International_Product'].dtype == 'object':
                df['International_Product'] = df['International_Product'].apply(
                    lambda x: 1 if str(x).lower() in ['yes', 'true', '1', 'y'] else 0
                )
            else:
                df['International_Product'] = df['International_Product'].fillna(0).astype(int)
            
            st.info(f"✅ International Product column detected: `{main_col}`")
        else:
            # Column not found - create mapping interface
            st.warning("⚠️ International Product column not found in dataset.")
            
            with st.expander("🔧 Map International Product Column", expanded=True):
                st.markdown("""
                **International Product column not automatically detected.**  
                Please select the column from your dataset that indicates international products:
                """)
                
                available_cols = [col for col in df.columns if df[col].dtype in ['object', 'category', 'int', 'float']]
                
                selected_col = st.selectbox(
                    "Select International Product Indicator Column",
                    options=['[None - Skip]'] + available_cols,
                    help="Select the column that indicates whether a product is international"
                )
                
                if selected_col and selected_col != '[None - Skip]':
                    # Create binary column
                    if df[selected_col].dtype == 'object' or df[selected_col].dtype == 'category':
                        unique_vals = df[selected_col].unique()[:10]
                        st.write(f"Sample values in `{selected_col}`: {unique_vals}")
                        
                        intl_value = st.text_input(
                            "Value indicating International Product",
                            help="Enter the exact value that indicates an international product (e.g., 'Yes', '1', 'International')"
                        )
                        
                        if intl_value:
                            df['International_Product'] = df[selected_col].apply(
                                lambda x: 1 if str(x).strip().lower() == intl_value.strip().lower() else 0
                            )
                            st.success(f"✅ International Product column created from `{selected_col}` where value = '{intl_value}'")
                    else:
                        df['International_Product'] = df[selected_col].fillna(0).astype(int)
                        st.success(f"✅ International Product column created from `{selected_col}`")
                else:
                    # Create dummy column
                    df['International_Product'] = 0
                    st.info("ℹ️ International Product analysis will be skipped (all products marked as local)")
        
        return df

# ================================================
# 3. STRATEGIC ANALYSIS ENGINE
# ================================================

class StrategicAnalyzer:
    """Strategic analysis engine for pharmaceutical market intelligence"""
    
    @staticmethod
    def bcg_matrix_analysis(df):
        """
        BCG Matrix Analysis - Classify products into Stars, Cash Cows, Question Marks, Dogs
        Returns: DataFrame with BCG classification
        """
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if len(sales_cols) < 2:
                return None
            
            last_sales_col = sales_cols[-1]
            
            # Get CAGR or growth rate
            if 'CAGR' in df.columns:
                growth_col = 'CAGR'
            else:
                growth_cols = [col for col in df.columns if 'Growth_' in col]
                if not growth_cols:
                    return None
                growth_col = growth_cols[-1]
            
            # Prepare data
            bcg_df = df[[last_sales_col, growth_col, 'Market_Share']].copy()
            bcg_df = bcg_df.dropna()
            
            if len(bcg_df) == 0:
                return None
            
            # Calculate median thresholds
            median_growth = bcg_df[growth_col].median()
            median_share = bcg_df['Market_Share'].median()
            
            # Classify products
            def classify_bcg(row):
                growth = row[growth_col]
                share = row['Market_Share']
                
                if growth >= median_growth and share >= median_share:
                    return 'Star'
                elif growth < median_growth and share >= median_share:
                    return 'Cash Cow'
                elif growth >= median_growth and share < median_share:
                    return 'Question Mark'
                else:
                    return 'Dog'
            
            bcg_df['BCG_Category'] = bcg_df.apply(classify_bcg, axis=1)
            
            # Add product names if available
            if 'Molecule' in df.columns:
                bcg_df['Product_Name'] = df['Molecule']
            elif 'Product' in df.columns:
                bcg_df['Product_Name'] = df['Product']
            else:
                bcg_df['Product_Name'] = df.index
            
            return bcg_df
            
        except Exception as e:
            st.warning(f"BCG Matrix analysis error: {str(e)}")
            return None
    
    @staticmethod
    def pareto_analysis(df, top_pct=80):
        """
        Pareto Analysis (80/20 Rule)
        Returns: DataFrame of products contributing to top_pct% of revenue
        """
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if not sales_cols:
                return None
            
            last_sales_col = sales_cols[-1]
            
            # Sort by sales
            pareto_df = df.sort_values(last_sales_col, ascending=False).copy()
            
            # Calculate cumulative percentage
            total_sales = pareto_df[last_sales_col].sum()
            pareto_df['Cumulative_Sales'] = pareto_df[last_sales_col].cumsum()
            pareto_df['Cumulative_Pct'] = (pareto_df['Cumulative_Sales'] / total_sales) * 100
            
            # Get products contributing to top_pct%
            pareto_products = pareto_df[pareto_df['Cumulative_Pct'] <= top_pct].copy()
            
            # Add product names
            if 'Molecule' in pareto_products.columns:
                pareto_products['Product_Name'] = pareto_products['Molecule']
            elif 'Product' in pareto_products.columns:
                pareto_products['Product_Name'] = pareto_products['Product']
            
            return pareto_products
            
        except Exception as e:
            st.warning(f"Pareto analysis error: {str(e)}")
            return None
    
    @staticmethod
    def product_segmentation(df, n_clusters=4):
        """
        Automatic Product Segmentation using KMeans clustering
        Returns: DataFrame with segment assignments
        """
        try:
            # Select features for clustering
            feature_cols = []
            
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if sales_cols:
                feature_cols.append(sales_cols[-1])
            
            if 'CAGR' in df.columns:
                feature_cols.append('CAGR')
            elif 'Growth_' in df.columns:
                growth_cols = [col for col in df.columns if 'Growth_' in col]
                if growth_cols:
                    feature_cols.append(growth_cols[-1])
            
            if 'Market_Share' in df.columns:
                feature_cols.append('Market_Share')
            
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            if price_cols:
                feature_cols.append(price_cols[-1])
            
            if len(feature_cols) < 2:
                return None
            
            # Prepare data
            segment_df = df[feature_cols].copy()
            segment_df = segment_df.dropna()
            
            if len(segment_df) < n_clusters * 3:
                return None
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(segment_df)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            segment_df['Segment'] = kmeans.fit_predict(scaled_features)
            
            # Label segments based on characteristics
            segment_labels = {
                0: 'High Value',
                1: 'Growing',
                2: 'Mature',
                3: 'Emerging'
            }
            
            segment_df['Segment_Label'] = segment_df['Segment'].map(segment_labels)
            
            # Add product names
            if 'Molecule' in df.columns:
                segment_df['Product_Name'] = df['Molecule']
            
            return segment_df
            
        except Exception as e:
            st.warning(f"Product segmentation error: {str(e)}")
            return None
    
    @staticmethod
    def action_plan_generator(df, bcg_df=None):
        """
        Rule-based Action Plan Generator
        Generates specific, actionable recommendations
        """
        recommendations = []
        
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            growth_cols = [col for col in df.columns if 'Growth_' in col]
            
            if not sales_cols:
                return recommendations
            
            last_sales_col = sales_cols[-1]
            
            # Rule 1: Declining high-value products
            if growth_cols and 'Molecule' in df.columns:
                last_growth_col = growth_cols[-1]
                declining_high_value = df[
                    (df[last_growth_col] < -10) & 
                    (df[last_sales_col] > df[last_sales_col].quantile(0.75))
                ]
                
                for _, row in declining_high_value.head(3).iterrows():
                    molecule = row['Molecule']
                    growth = row[last_growth_col]
                    sales = row[last_sales_col]
                    
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'Revenue Protection',
                        'molecule': molecule,
                        'action': f"**{molecule}**: Experiencing {abs(growth):.1f}% market decline despite ${sales/1e6:.1f}M revenue. **Action**: Conduct pricing elasticity study and consider 5-10% price optimization to protect market share."
                    })
            
            # Rule 2: High-growth opportunities
            if growth_cols and 'Molecule' in df.columns:
                high_growth = df[df[last_growth_col] > 30]
                
                for _, row in high_growth.head(3).iterrows():
                    molecule = row['Molecule']
                    growth = row[last_growth_col]
                    
                    recommendations.append({
                        'priority': 'HIGH',
                        'category': 'Growth Investment',
                        'molecule': molecule,
                        'action': f"**{molecule}**: Strong {growth:.1f}% growth momentum. **Action**: Increase marketing spend by 15-20% and expand distribution channels to capture market opportunity."
                    })
            
            # Rule 3: International expansion opportunities
            if 'International_Product' in df.columns and 'Molecule' in df.columns:
                local_high_performers = df[
                    (df['International_Product'] == 0) & 
                    (df[last_sales_col] > df[last_sales_col].quantile(0.80))
                ]
                
                for _, row in local_high_performers.head(2).iterrows():
                    molecule = row['Molecule']
                    sales = row[last_sales_col]
                    
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': 'International Expansion',
                        'molecule': molecule,
                        'action': f"**{molecule}**: Successful local product (${sales/1e6:.1f}M). **Action**: Evaluate international registration opportunities in similar therapeutic markets."
                    })
            
            # Rule 4: Price optimization opportunities
            if 'Market_Share' in df.columns and 'Avg_Price_2024' in df.columns:
                low_share_premium = df[
                    (df['Market_Share'] < 5) & 
                    (df['Avg_Price_2024'] > df['Avg_Price_2024'].quantile(0.75))
                ]
                
                for _, row in low_share_premium.head(2).iterrows():
                    if 'Molecule' in row:
                        molecule = row['Molecule']
                        price = row['Avg_Price_2024']
                        share = row['Market_Share']
                        
                        recommendations.append({
                            'priority': 'MEDIUM',
                            'category': 'Pricing Strategy',
                            'molecule': molecule,
                            'action': f"**{molecule}**: Premium pricing (${price:.2f}) with low market share ({share:.1f}%). **Action**: Consider 10-15% price reduction to improve market penetration."
                        })
            
            # Rule 5: BCG-based recommendations
            if bcg_df is not None and len(bcg_df) > 0:
                # Question Marks needing investment
                question_marks = bcg_df[bcg_df['BCG_Category'] == 'Question Mark']
                if len(question_marks) > 0:
                    for _, row in question_marks.head(2).iterrows():
                        if 'Product_Name' in row:
                            recommendations.append({
                                'priority': 'MEDIUM',
                                'category': 'Strategic Investment',
                                'molecule': row['Product_Name'],
                                'action': f"**{row['Product_Name']}**: Question Mark product with growth potential. **Action**: Invest in market development or divest if ROI target cannot be met within 12 months."
                            })
                
                # Dogs to consider divesting
                dogs = bcg_df[bcg_df['BCG_Category'] == 'Dog']
                if len(dogs) > 0:
                    for _, row in dogs.head(2).iterrows():
                        if 'Product_Name' in row:
                            recommendations.append({
                                'priority': 'LOW',
                                'category': 'Portfolio Optimization',
                                'molecule': row['Product_Name'],
                                'action': f"**{row['Product_Name']}**: Low growth, low share product. **Action**: Evaluate divestment or harvest strategy to reallocate resources."
                            })
            
            return recommendations
            
        except Exception as e:
            st.warning(f"Action plan generation error: {str(e)}")
            return recommendations

# ================================================
# 4. PROFESSIONAL VISUALIZATION ENGINE
# ================================================

class ProfessionalVisualizer:
    """Professional visualization engine"""
    
    @staticmethod
    def create_dashboard_metrics(df, metrics):
        """Create executive dashboard metric cards"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_market = metrics.get('Total_Market_Value', 0)
                year = metrics.get('Latest_Year', '')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">TOTAL MARKET VALUE</div>
                    <div class="metric-value">${total_market/1e6:.1f}M</div>
                    <div class="metric-delta">{year} Total Market</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_growth = metrics.get('Avg_Growth_Rate', 0)
                growth_class = "success" if avg_growth > 0 else "danger"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">AVERAGE GROWTH</div>
                    <div class="metric-value">{avg_growth:.1f}%</div>
                    <div class="metric-delta">YoY Annual Growth</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hhi = metrics.get('HHI_Index', 0)
                hhi_status = "Monopolistic" if hhi > 2500 else "Oligopoly" if hhi > 1500 else "Competitive"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MARKET CONCENTRATION</div>
                    <div class="metric-value">{hhi:.0f}</div>
                    <div class="metric-delta">HHI Index - {hhi_status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                intl_share = metrics.get('International_Share', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">INTERNATIONAL PRODUCTS</div>
                    <div class="metric-value">{intl_share:.1f}%</div>
                    <div class="metric-delta">Multi-Market Presence</div>
                </div>
                """, unsafe_allow_html=True)
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                unique_molecules = metrics.get('Unique_Molecules', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MOLECULE DIVERSITY</div>
                    <div class="metric-value">{unique_molecules:,}</div>
                    <div class="metric-delta">Unique Active Ingredients</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                avg_price = metrics.get('Avg_Price', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">AVERAGE PRICE</div>
                    <div class="metric-value">${avg_price:.2f}</div>
                    <div class="metric-delta">Per Unit Average</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col7:
                high_growth_pct = metrics.get('High_Growth_Percentage', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">HIGH GROWTH</div>
                    <div class="metric-value">{high_growth_pct:.1f}%</div>
                    <div class="metric-delta">Products >20% Growth</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col8:
                country_coverage = metrics.get('Country_Coverage', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">GEOGRAPHIC REACH</div>
                    <div class="metric-value">{country_coverage}</div>
                    <div class="metric-delta">Countries Covered</div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Metric card creation error: {str(e)}")
    
    @staticmethod
    def molecule_comparison_chart(df, selected_molecules):
        """
        Molecular Comparison Module
        Side-by-side comparison of 2+ molecules
        """
        try:
            if len(selected_molecules) < 2:
                st.info("Please select at least 2 molecules to compare")
                return None
            
            # Filter data
            comparison_df = df[df['Molecule'].isin(selected_molecules)].copy()
            
            if len(comparison_df) == 0:
                return None
            
            # Aggregate by molecule
            agg_dict = {}
            
            sales_cols = [col for col in comparison_df.columns if 'Sales_' in col]
            if sales_cols:
                agg_dict[sales_cols[-1]] = 'sum'
            
            if 'CAGR' in comparison_df.columns:
                agg_dict['CAGR'] = 'mean'
            
            if 'Market_Share' in comparison_df.columns:
                agg_dict['Market_Share'] = 'sum'
            
            price_cols = [col for col in comparison_df.columns if 'Avg_Price' in col]
            if price_cols:
                agg_dict[price_cols[-1]] = 'mean'
            
            if not agg_dict:
                return None
            
            mol_comparison = comparison_df.groupby('Molecule').agg(agg_dict).reset_index()
            
            # Create grouped bar chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Sales Comparison', 'CAGR Comparison', 
                              'Market Share', 'Average Price'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            molecules = mol_comparison['Molecule'].tolist()
            
            # Sales
            if sales_cols:
                fig.add_trace(
                    go.Bar(
                        x=molecules,
                        y=mol_comparison[sales_cols[-1]],
                        name='Sales',
                        marker_color='#2d7dd2',
                        text=[f'${x/1e6:.1f}M' for x in mol_comparison[sales_cols[-1]]],
                        textposition='auto'
                    ),
                    row=1, col=1
                )
            
            # CAGR
            if 'CAGR' in mol_comparison.columns:
                fig.add_trace(
                    go.Bar(
                        x=molecules,
                        y=mol_comparison['CAGR'],
                        name='CAGR',
                        marker_color='#2acaea',
                        text=[f'{x:.1f}%' for x in mol_comparison['CAGR']],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
            
            # Market Share
            if 'Market_Share' in mol_comparison.columns:
                fig.add_trace(
                    go.Bar(
                        x=molecules,
                        y=mol_comparison['Market_Share'],
                        name='Market Share',
                        marker_color='#2dd2a3',
                        text=[f'{x:.1f}%' for x in mol_comparison['Market_Share']],
                        textposition='auto'
                    ),
                    row=2, col=1
                )
            
            # Average Price
            if price_cols:
                fig.add_trace(
                    go.Bar(
                        x=molecules,
                        y=mol_comparison[price_cols[-1]],
                        name='Avg Price',
                        marker_color='#f59e0b',
                        text=[f'${x:.2f}' for x in mol_comparison[price_cols[-1]]],
                        textposition='auto'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                title_text="Molecule Head-to-Head Comparison",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Molecule comparison error: {str(e)}")
            return None
    
    @staticmethod
    def company_battle_chart(df, company1, company2):
        """
        Head-to-Head Company Market Share Battle
        Time series showing market share evolution
        """
        try:
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            if len(sales_cols) < 2:
                return None
            
            # Filter companies
            battle_df = df[df['Company'].isin([company1, company2])].copy()
            
            if len(battle_df) == 0:
                return None
            
            # Calculate market share over time
            years = sorted([col.split('_')[-1] for col in sales_cols])
            
            company_data = {company1: [], company2: []}
            
            for year in years:
                sales_col = f"Sales_{year}"
                if sales_col in battle_df.columns:
                    year_total = battle_df[sales_col].sum()
                    
                    for company in [company1, company2]:
                        company_sales = battle_df[battle_df['Company'] == company][sales_col].sum()
                        market_share = (company_sales / year_total * 100) if year_total > 0 else 0
                        company_data[company].append(market_share)
            
            # Create line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=years,
                y=company_data[company1],
                mode='lines+markers',
                name=company1,
                line=dict(color='#2d7dd2', width=3),
                marker=dict(size=10)
            ))
            
            fig.add_trace(go.Scatter(
                x=years,
                y=company_data[company2],
                mode='lines+markers',
                name=company2,
                line=dict(color='#ef4444', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title=f'Market Share Battle: {company1} vs {company2}',
                xaxis_title='Year',
                yaxis_title='Market Share (%)',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Company battle chart error: {str(e)}")
            return None
    
    @staticmethod
    def bcg_matrix_chart(bcg_df):
        """BCG Matrix visualization"""
        try:
            if bcg_df is None or len(bcg_df) == 0:
                return None
            
            # Get column names
            sales_cols = [col for col in bcg_df.columns if 'Sales_' in col]
            if not sales_cols:
                return None
            
            last_sales_col = sales_cols[-1]
            
            if 'CAGR' in bcg_df.columns:
                growth_col = 'CAGR'
            else:
                growth_cols = [col for col in bcg_df.columns if 'Growth_' in col]
                if not growth_cols:
                    return None
                growth_col = growth_cols[-1]
            
            # Create scatter plot
            fig = px.scatter(
                bcg_df,
                x='Market_Share',
                y=growth_col,
                size=last_sales_col,
                color='BCG_Category',
                hover_name='Product_Name' if 'Product_Name' in bcg_df.columns else None,
                title='BCG Growth-Share Matrix',
                labels={
                    'Market_Share': 'Relative Market Share (%)',
                    growth_col: 'Market Growth Rate (%)',
                    last_sales_col: 'Market Size (USD)'
                },
                color_discrete_map={
                    'Star': '#10b981',
                    'Cash Cow': '#3b82f6',
                    'Question Mark': '#f59e0b',
                    'Dog': '#ef4444'
                }
            )
            
            # Add quadrant lines
            median_growth = bcg_df[growth_col].median()
            median_share = bcg_df['Market_Share'].median()
            
            fig.add_hline(y=median_growth, line_dash="dash", line_color="#64748b", opacity=0.5)
            fig.add_vline(x=median_share, line_dash="dash", line_color="#64748b", opacity=0.5)
            
            # Add quadrant labels
            fig.add_annotation(
                x=bcg_df['Market_Share'].max() * 0.8,
                y=bcg_df[growth_col].max() * 0.9,
                text="<b>STARS</b>",
                showarrow=False,
                font=dict(size=14, color='#10b981')
            )
            
            fig.add_annotation(
                x=bcg_df['Market_Share'].max() * 0.8,
                y=bcg_df[growth_col].min() * 0.9,
                text="<b>CASH COWS</b>",
                showarrow=False,
                font=dict(size=14, color='#3b82f6')
            )
            
            fig.add_annotation(
                x=bcg_df['Market_Share'].min() * 1.2,
                y=bcg_df[growth_col].max() * 0.9,
                text="<b>QUESTION MARKS</b>",
                showarrow=False,
                font=dict(size=14, color='#f59e0b')
            )
            
            fig.add_annotation(
                x=bcg_df['Market_Share'].min() * 1.2,
                y=bcg_df[growth_col].min() * 0.9,
                text="<b>DOGS</b>",
                showarrow=False,
                font=dict(size=14, color='#ef4444')
            )
            
            fig.update_layout(
                height=700,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"BCG matrix chart error: {str(e)}")
            return None
    
    @staticmethod
    def price_volume_analysis(df):
        """
        FIXED: Price-Volume Analysis with unique column names
        """
        try:
            price_cols = [col for col in df.columns if 'Avg_Price' in col]
            units_cols = [col for col in df.columns if 'Units_' in col]
            sales_cols = [col for col in df.columns if 'Sales_' in col]
            
            # If no price columns, calculate them
            if not price_cols and sales_cols and units_cols:
                last_sales_col = sales_cols[-1]
                last_units_col = units_cols[-1]
                
                if last_sales_col in df.columns and last_units_col in df.columns:
                    temp_df = df.copy()
                    temp_df['Calculated_Avg_Price'] = np.where(
                        temp_df[last_units_col] != 0,
                        temp_df[last_sales_col] / temp_df[last_units_col],
                        np.nan
                    )
                    price_cols = ['Calculated_Avg_Price']
                    df = temp_df
            
            if not price_cols or not units_cols:
                st.info("Price-volume analysis requires price and units data.")
                return None
            
            last_price_col = price_cols[-1]
            last_units_col = units_cols[-1]
            
            # CRITICAL FIX: Create unique temporary column names using timestamp
            import hashlib
            unique_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            
            temp_price_col = f'TempPrice_{unique_id}'
            temp_units_col = f'TempUnits_{unique_id}'
            
            # Create temporary dataframe
            plot_df = df[[last_price_col, last_units_col]].copy()
            plot_df = plot_df.rename(columns={
                last_price_col: temp_price_col,
                last_units_col: temp_units_col
            })
            
            # Remove invalid values
            plot_df = plot_df[
                (plot_df[temp_price_col] > 0) & 
                (plot_df[temp_units_col] > 0)
            ].copy()
            
            if len(plot_df) == 0:
                st.info("No valid price-volume data found.")
                return None
            
            # Sample if too large
            if len(plot_df) > 10000:
                plot_df = plot_df.sample(10000, random_state=42)
            
            # Add hover name if available
            hover_col = None
            if 'Molecule' in df.columns:
                plot_df['Hover_Name'] = df.loc[plot_df.index, 'Molecule']
                hover_col = 'Hover_Name'
            
            # Create scatter plot
            fig = px.scatter(
                plot_df,
                x=temp_price_col,
                y=temp_units_col,
                size=temp_units_col,
                color=temp_price_col,
                hover_name=hover_col,
                title='Price-Volume Relationship Analysis',
                labels={
                    temp_price_col: 'Average Price (USD)',
                    temp_units_col: 'Volume (Units)'
                },
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"Price-volume analysis error: {str(e)}")
            st.error(traceback.format_exc())
            return None

# ================================================
# 5. MAIN APPLICATION
# ================================================

def calculate_metrics(df):
    """Calculate key metrics for dashboard"""
    metrics = {}
    
    try:
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        growth_cols = [col for col in df.columns if 'Growth_' in col]
        
        if sales_cols:
            last_sales_col = sales_cols[-1]
            metrics['Total_Market_Value'] = df[last_sales_col].sum()
            metrics['Latest_Year'] = last_sales_col.split('_')[-1]
        
        if growth_cols:
            last_growth_col = growth_cols[-1]
            metrics['Avg_Growth_Rate'] = df[last_growth_col].mean()
            metrics['High_Growth_Percentage'] = (df[last_growth_col] > 20).sum() / len(df) * 100
        
        # HHI Index
        if 'Company' in df.columns and sales_cols:
            company_shares = df.groupby('Company')[sales_cols[-1]].sum()
            total_market = company_shares.sum()
            if total_market > 0:
                market_shares = (company_shares / total_market * 100) ** 2
                metrics['HHI_Index'] = market_shares.sum()
        
        # International Product metrics
        if 'International_Product' in df.columns:
            intl_count = (df['International_Product'] == 1).sum()
            metrics['International_Count'] = intl_count
            
            if sales_cols:
                intl_sales = df[df['International_Product'] == 1][sales_cols[-1]].sum()
                total_sales = df[sales_cols[-1]].sum()
                metrics['International_Share'] = (intl_sales / total_sales * 100) if total_sales > 0 else 0
        
        # Other metrics
        if 'Molecule' in df.columns:
            metrics['Unique_Molecules'] = df['Molecule'].nunique()
        
        if 'Country' in df.columns:
            metrics['Country_Coverage'] = df['Country'].nunique()
        
        price_cols = [col for col in df.columns if 'Avg_Price' in col]
        if price_cols:
            metrics['Avg_Price'] = df[price_cols[-1]].mean()
        
    except Exception as e:
        st.warning(f"Metrics calculation error: {str(e)}")
    
    return metrics

def main():
    """Main application function"""
    
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #2acaea; font-size: 48px; font-weight: 700;'>💊 PHARMAINTELLIGENCE ENTERPRISE</h1>
        <p style='color: #94a3b8; font-size: 18px;'>Strategic Pharmaceutical Market Analytics Platform</p>
        <p style='color: #64748b; font-size: 14px;'>BCG Matrix | Pareto Analysis | Predictive Insights | Action Planning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="section-header">🎛️ CONTROL PANEL</div>', unsafe_allow_html=True)
        
        with st.expander("📁 DATA UPLOAD", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload Excel/CSV File",
                type=['xlsx', 'xls', 'csv'],
                help="Supports datasets with 1M+ rows"
            )
            
            if uploaded_file:
                st.info(f"File: {uploaded_file.name}")
                
                if st.button("🚀 Load & Analyze Dataset", type="primary", use_container_width=True):
                    with st.spinner("Processing dataset..."):
                        processor = OptimizedDataProcessor()
                        data = processor.load_large_dataset(uploaded_file, sample_size=None)
                        
                        if data is not None and len(data) > 0:
                            data = processor.prepare_analysis_data(data)
                            st.session_state.data = data
                            st.session_state.filtered_data = data.copy()
                            
                            # Calculate metrics
                            metrics = calculate_metrics(data)
                            st.session_state.metrics = metrics
                            
                            st.success(f"✅ Dataset loaded: {len(data):,} rows")
                            st.rerun()
        
        # Filtering
        if st.session_state.data is not None:
            with st.expander("🔍 FILTERS", expanded=True):
                data = st.session_state.data
                
                # Search
                search_term = st.text_input("🔎 Global Search", placeholder="Search...")
                
                # Country filter
                if 'Country' in data.columns:
                    countries = sorted(data['Country'].dropna().unique())
                    selected_countries = st.multiselect(
                        "Countries",
                        options=countries,
                        default=countries[:min(5, len(countries))]
                    )
                
                # Company filter
                if 'Company' in data.columns:
                    companies = sorted(data['Company'].dropna().unique())
                    selected_companies = st.multiselect(
                        "Companies",
                        options=companies,
                        default=companies[:min(5, len(companies))]
                    )
                
                # Apply filters
                if st.button("✅ Apply Filters", use_container_width=True):
                    filtered = data.copy()
                    
                    if search_term:
                        mask = pd.Series(False, index=filtered.index)
                        for col in filtered.columns:
                            try:
                                mask = mask | filtered[col].astype(str).str.contains(search_term, case=False, na=False)
                            except:
                                continue
                        filtered = filtered[mask]
                    
                    if 'Country' in data.columns and selected_countries:
                        filtered = filtered[filtered['Country'].isin(selected_countries)]
                    
                    if 'Company' in data.columns and selected_companies:
                        filtered = filtered[filtered['Company'].isin(selected_companies)]
                    
                    st.session_state.filtered_data = filtered
                    st.success(f"✅ {len(filtered):,} rows after filtering")
                    st.rerun()
    
    # Main content
    if st.session_state.data is None:
        show_welcome_screen()
        return
    
    data = st.session_state.filtered_data
    metrics = st.session_state.metrics
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 OVERVIEW",
        "🎯 STRATEGIC ANALYSIS",
        "🏆 COMPETITION",
        "🌍 INTERNATIONAL",
        "📑 REPORTS"
    ])
    
    with tab1:
        show_overview_tab(data, metrics)
    
    with tab2:
        show_strategic_tab(data)
    
    with tab3:
        show_competition_tab(data)
    
    with tab4:
        show_international_tab(data, metrics)
    
    with tab5:
        show_reports_tab(data, metrics)

# ================================================
# TAB FUNCTIONS
# ================================================

def show_welcome_screen():
    """Welcome screen"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 40px;'>
            <h2 style='color: #2acaea;'>Welcome to PharmaIntelligence Enterprise</h2>
            <p style='color: #94a3b8; font-size: 16px;'>
                Upload your pharmaceutical market data to unlock powerful analytics
            </p>
            <br>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 30px;'>
                <div style='background: rgba(45, 125, 210, 0.1); padding: 20px; border-radius: 8px;'>
                    <h3 style='color: #2d7dd2;'>🎯 BCG Matrix</h3>
                    <p style='color: #94a3b8; font-size: 14px;'>Strategic portfolio analysis</p>
                </div>
                <div style='background: rgba(45, 125, 210, 0.1); padding: 20px; border-radius: 8px;'>
                    <h3 style='color: #2d7dd2;'>📊 Pareto Analysis</h3>
                    <p style='color: #94a3b8; font-size: 14px;'>80/20 revenue drivers</p>
                </div>
                <div style='background: rgba(45, 125, 210, 0.1); padding: 20px; border-radius: 8px;'>
                    <h3 style='color: #2d7dd2;'>🌍 International</h3>
                    <p style='color: #94a3b8; font-size: 14px;'>Multi-market insights</p>
                </div>
                <div style='background: rgba(45, 125, 210, 0.1); padding: 20px; border-radius: 8px;'>
                    <h3 style='color: #2d7dd2;'>🎬 Action Plans</h3>
                    <p style='color: #94a3b8; font-size: 14px;'>Automated recommendations</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_overview_tab(df, metrics):
    """Overview tab"""
    st.markdown('<div class="section-header">Executive Overview</div>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualizer()
    viz.create_dashboard_metrics(df, metrics)
    
    st.markdown('<div class="section-header">📈 Market Trends</div>', unsafe_allow_html=True)
    
    # Sales trends
    sales_cols = [col for col in df.columns if 'Sales_' in col]
    if len(sales_cols) >= 2:
        years = sorted([col.split('_')[-1] for col in sales_cols])
        yearly_data = []
        
        for year in years:
            col = f"Sales_{year}"
            if col in df.columns:
                yearly_data.append({
                    'Year': year,
                    'Total_Sales': df[col].sum(),
                    'Avg_Sales': df[col].mean()
                })
        
        yearly_df = pd.DataFrame(yearly_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly_df['Year'],
            y=yearly_df['Total_Sales'],
            name='Total Sales',
            marker_color='#2d7dd2',
            text=[f'${x/1e6:.0f}M' for x in yearly_df['Total_Sales']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Sales Evolution',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.markdown('<div class="section-header">🔍 Data Preview</div>', unsafe_allow_html=True)
    
    display_cols = []
    for col in ['Molecule', 'Company', 'Country', 'Sales_2024', 'CAGR', 'Market_Share']:
        if col in df.columns:
            display_cols.append(col)
    
    if display_cols:
        st.dataframe(df[display_cols].head(100), use_container_width=True, height=400)

def show_strategic_tab(df):
    """Strategic Analysis tab"""
    st.markdown('<div class="section-header">Strategic Analysis & Planning</div>', unsafe_allow_html=True)
    
    analyzer = StrategicAnalyzer()
    viz = ProfessionalVisualizer()
    
    # BCG Matrix
    st.markdown('<div class="section-header">📊 BCG Matrix Analysis</div>', unsafe_allow_html=True)
    
    bcg_df = analyzer.bcg_matrix_analysis(df)
    
    if bcg_df is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            bcg_chart = viz.bcg_matrix_chart(bcg_df)
            if bcg_chart:
                st.plotly_chart(bcg_chart, use_container_width=True)
        
        with col2:
            st.markdown("**BCG Category Distribution**")
            
            category_counts = bcg_df['BCG_Category'].value_counts()
            
            for category in ['Star', 'Cash Cow', 'Question Mark', 'Dog']:
                if category in category_counts:
                    count = category_counts[category]
                    pct = count / len(bcg_df) * 100
                    
                    color_map = {
                        'Star': '#10b981',
                        'Cash Cow': '#3b82f6',
                        'Question Mark': '#f59e0b',
                        'Dog': '#ef4444'
                    }
                    
                    st.markdown(f"""
                    <div style='background: rgba(30, 58, 95, 0.3); padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {color_map[category]};'>
                        <h4 style='color: {color_map[category]}; margin: 0;'>{category}s</h4>
                        <p style='color: #94a3b8; margin: 5px 0;'>{count} products ({pct:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show product names by category
            st.markdown("**Product Lists**")
            
            for category in ['Star', 'Cash Cow', 'Question Mark', 'Dog']:
                products = bcg_df[bcg_df['BCG_Category'] == category]
                if len(products) > 0 and 'Product_Name' in products.columns:
                    with st.expander(f"{category}s ({len(products)})"):
                        product_list = products['Product_Name'].tolist()[:10]
                        for p in product_list:
                            st.write(f"• {p}")
    
    # Pareto Analysis
    st.markdown('<div class="section-header">📊 Pareto Analysis (80/20 Rule)</div>', unsafe_allow_html=True)
    
    pareto_df = analyzer.pareto_analysis(df, top_pct=80)
    
    if pareto_df is not None:
        st.success(f"🎯 **{len(pareto_df)} products** generate **80%** of total revenue")
        
        if 'Product_Name' in pareto_df.columns:
            with st.expander(f"View Top {len(pareto_df)} Revenue-Driving Products"):
                sales_cols = [col for col in pareto_df.columns if 'Sales_' in col]
                if sales_cols:
                    display_df = pareto_df[['Product_Name', sales_cols[-1], 'Cumulative_Pct']].head(20)
                    display_df.columns = ['Product', 'Sales', 'Cumulative %']
                    st.dataframe(display_df, use_container_width=True)
    
    # Action Plan Generator
    st.markdown('<div class="section-header">🎬 Automated Action Plan</div>', unsafe_allow_html=True)
    
    recommendations = analyzer.action_plan_generator(df, bcg_df)
    
    if recommendations:
        for rec in recommendations[:10]:
            priority_colors = {
                'HIGH': '#ef4444',
                'MEDIUM': '#f59e0b',
                'LOW': '#3b82f6'
            }
            
            color = priority_colors.get(rec['priority'], '#64748b')
            
            st.markdown(f"""
            <div class="insight-card">
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span class='badge-{rec['priority'].lower()}' style='background: {color};'>{rec['priority']}</span>
                    <span style='color: #94a3b8; font-size: 12px;'>{rec['category']}</span>
                </div>
                <p style='margin-top: 10px; color: #f8fafc;'>{rec['action']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No specific recommendations generated at this time.")

def show_competition_tab(df):
    """Competition Analysis tab"""
    st.markdown('<div class="section-header">Competition & Market Dynamics</div>', unsafe_allow_html=True)
    
    viz = ProfessionalVisualizer()
    
    # Molecule Comparison
    if 'Molecule' in df.columns:
        st.markdown('<div class="section-header">🧪 Molecule Head-to-Head Comparison</div>', unsafe_allow_html=True)
        
        molecules = sorted(df['Molecule'].dropna().unique())
        selected_molecules = st.multiselect(
            "Select 2+ molecules to compare",
            options=molecules,
            default=molecules[:min(3, len(molecules))]
        )
        
        if len(selected_molecules) >= 2:
            comparison_chart = viz.molecule_comparison_chart(df, selected_molecules)
            if comparison_chart:
                st.plotly_chart(comparison_chart, use_container_width=True)
    
    # Company Battle
    if 'Company' in df.columns:
        st.markdown('<div class="section-header">⚔️ Company Market Share Battle</div>', unsafe_allow_html=True)
        
        companies = sorted(df['Company'].dropna().unique())
        
        col1, col2 = st.columns(2)
        with col1:
            company1 = st.selectbox("Company 1", options=companies, index=0)
        with col2:
            company2 = st.selectbox("Company 2", options=companies, index=min(1, len(companies)-1))
        
        if company1 != company2:
            battle_chart = viz.company_battle_chart(df, company1, company2)
            if battle_chart:
                st.plotly_chart(battle_chart, use_container_width=True)
    
    # Price-Volume Analysis
    st.markdown('<div class="section-header">💰 Price-Volume Dynamics</div>', unsafe_allow_html=True)
    
    pv_chart = viz.price_volume_analysis(df)
    if pv_chart:
        st.plotly_chart(pv_chart, use_container_width=True)

def show_international_tab(df, metrics):
    """International Product Analysis tab"""
    st.markdown('<div class="section-header">🌍 International Product Analysis</div>', unsafe_allow_html=True)
    
    if 'International_Product' not in df.columns:
        st.warning("⚠️ International Product column not found in dataset")
        return
    
    intl_df = df[df['International_Product'] == 1].copy()
    local_df = df[df['International_Product'] == 0].copy()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        intl_count = len(intl_df)
        total_count = len(df)
        intl_pct = (intl_count / total_count * 100) if total_count > 0 else 0
        st.metric("International Products", f"{intl_count}", f"{intl_pct:.1f}%")
    
    with col2:
        sales_cols = [col for col in df.columns if 'Sales_' in col]
        if sales_cols:
            last_sales_col = sales_cols[-1]
            intl_sales = intl_df[last_sales_col].sum()
            total_sales = df[last_sales_col].sum()
            intl_share = (intl_sales / total_sales * 100) if total_sales > 0 else 0
            st.metric("Revenue Share", f"{intl_share:.1f}%", f"${intl_sales/1e6:.1f}M")
    
    with col3:
        if 'Country' in intl_df.columns:
            intl_countries = intl_df['Country'].nunique()
            st.metric("Countries", intl_countries)
    
    with col4:
        if 'Molecule' in intl_df.columns:
            intl_molecules = intl_df['Molecule'].nunique()
            st.metric("Molecules", intl_molecules)
    
    # Comparison charts
    if sales_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales distribution
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['International', 'Local'],
                y=[intl_df[last_sales_col].sum(), local_df[last_sales_col].sum()],
                marker_color=['#2d7dd2', '#64748b'],
                text=[f'${intl_df[last_sales_col].sum()/1e6:.1f}M', f'${local_df[last_sales_col].sum()/1e6:.1f}M'],
                textposition='auto'
            ))
            fig.update_layout(
                title='Sales Distribution',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Product count
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=['International', 'Local'],
                values=[len(intl_df), len(local_df)],
                marker_colors=['#2d7dd2', '#64748b'],
                hole=0.4
            ))
            fig.update_layout(
                title='Product Distribution',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # International products table
    if len(intl_df) > 0:
        st.markdown('<div class="section-header">📋 International Product Details</div>', unsafe_allow_html=True)
        
        display_cols = []
        for col in ['Molecule', 'Company', 'Country', 'Sales_2024', 'CAGR', 'Market_Share']:
            if col in intl_df.columns:
                display_cols.append(col)
        
        if display_cols:
            st.dataframe(intl_df[display_cols].head(50), use_container_width=True, height=400)

def show_reports_tab(df, metrics):
    """Reports & Export tab"""
    st.markdown('<div class="section-header">Reports & Data Export</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Full Dataset (CSV)", use_container_width=True):
            csv = df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"pharma_dataset_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if 'International_Product' in df.columns:
            if st.button("🌍 Download International Products", use_container_width=True):
                intl_df = df[df['International_Product'] == 1]
                csv = intl_df.to_csv(index=False)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"international_products_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    with col3:
        if st.button("🔄 Reset Analysis", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Quick stats
    st.markdown('<div class="section-header">📈 Dataset Statistics</div>', unsafe_allow_html=True)
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with stat_col2:
        st.metric("Total Columns", len(df.columns))
    
    with stat_col3:
        memory = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory:.1f} MB")
    
    with stat_col4:
        intl_count = (df['International_Product'] == 1).sum() if 'International_Product' in df.columns else 0
        st.metric("International Products", intl_count)

# ================================================
# APPLICATION ENTRY POINT
# ================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Detailed error:")
        st.code(traceback.format_exc())
        
        if st.button("🔄 Restart Application", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
