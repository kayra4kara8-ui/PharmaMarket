"""
================================================================================
PHARMACEUTICAL MARKET INTELLIGENCE PLATFORM
MAT Data Analytics Dashboard
Version: 4.0.0 | Production-Ready | Python 3.11+
================================================================================
Author: Analytics Team
Last Updated: 2024-01-27
Status: Production
================================================================================
"""

# ==============================================================================
# 1. STREAMLIT PAGE CONFIG - MUST BE FIRST!
# ==============================================================================
import streamlit as st

st.set_page_config(
    page_title='Pharma MAT Intelligence Platform',
    page_icon='💊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ==============================================================================
# 2. IMPORT LIBRARIES
# ==============================================================================
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import io
import re
import json
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ==============================================================================
# 3. CONFIGURATION & CONSTANTS
# ==============================================================================
@dataclass
class AppConfig:
    """Application configuration constants"""
    APP_VERSION = "4.0.0"
    APP_NAME = "Pharmaceutical Market Intelligence Platform"
    MAX_FILE_SIZE_MB = 200
    DEFAULT_TOP_N = 15
    CHART_HEIGHT = 500
    TABLE_HEIGHT = 400
    
    # Column name mappings
    METRIC_COLUMNS = {
        'SALES_USD': ['USD MNF', 'SALES', 'REVENUE'],
        'STANDARD_UNITS': ['STANDARD UNITS', 'SU'],
        'UNITS': ['UNITS'],
        'SU_AVG_PRICE': ['SU AVG PRICE', 'AVG PRICE'],
        'UNIT_AVG_PRICE': ['UNIT AVG PRICE']
    }
    
    # MAT period patterns
    MAT_PATTERNS = [
        r'MAT Q\d \d{4}',
        r'MAT_Q\d_\d{4}',
        r'MAT\d{1}Q\d{4}',
        r'Moving Annual Total Q\d \d{4}'
    ]

# ==============================================================================
# 4. CUSTOM CSS STYLING
# ==============================================================================
def apply_custom_styling():
    """Apply professional CSS styling"""
    st.markdown("""
    <style>
    /* ========== IMPORTS ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* ========== GLOBAL ========== */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        padding: 2rem;
        background: #f8f9fa;
    }
    
    /* ========== HEADER ========== */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
        padding: 1rem;
    }
    
    .main-subheader {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* ========== METRIC CARDS ========== */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 140px;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        color: #1e293b;
        line-height: 1.2;
        margin-top: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .metric-change {
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .metric-change.positive {
        color: #10b981;
    }
    
    .metric-change.negative {
        color: #ef4444;
    }
    
    /* ========== SUB HEADERS ========== */
    .sub-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        padding: 4px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        background-color: transparent;
        color: #64748b;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e8f0;
        color: #1e293b;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: white !important;
        margin-bottom: 1.5rem;
        padding: 1.25rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(102, 126, 234, 0.4);
    }
    
    /* ========== ALERTS ========== */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
    }
    
    /* ========== FILE UPLOADER ========== */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        border: 2px dashed #cbd5e1;
        transition: all 0.3s;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #f8f9fa;
    }
    
    /* ========== DATA TABLES ========== */
    .dataframe {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
    }
    
    /* ========== FOOTER ========== */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 4rem;
        color: #64748b;
        font-size: 0.875rem;
        border-top: 2px solid #e2e8f0;
        background: white;
        border-radius: 12px;
    }
    
    /* ========== INFO BOX ========== */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* ========== LOADING ========== */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    
    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. UTILITY FUNCTIONS
# ==============================================================================
class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def is_valid_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate if dataframe is usable"""
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        if len(df.columns) == 0:
            return False, "DataFrame has no columns"
        
        return True, "Valid"
    
    @staticmethod
    def check_required_columns(df: pd.DataFrame, required: List[str]) -> Tuple[bool, List[str]]:
        """Check if required columns exist"""
        missing = [col for col in required if col not in df.columns]
        return len(missing) == 0, missing
    
    @staticmethod
    def safe_numeric_conversion(value: Any) -> float:
        """Safely convert any value to numeric"""
        if pd.isna(value):
            return 0.0
        
        try:
            str_val = str(value).strip()
            # Handle European format (comma as decimal)
            str_val = str_val.replace(',', '.')
            # Remove non-numeric characters except dot and minus
            str_val = re.sub(r'[^\d\.\-]', '', str_val)
            
            if str_val == '' or str_val == '-':
                return 0.0
            
            return float(str_val)
        except Exception:
            return 0.0

class ErrorHandler:
    """Centralized error handling"""
    
    @staticmethod
    def handle_error(error: Exception, context: str = "Operation", show_traceback: bool = False):
        """Display user-friendly error messages"""
        st.error(f"❌ **Error in {context}**")
        st.error(f"**Message:** {str(error)}")
        
        if show_traceback:
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())
    
    @staticmethod
    def safe_execute(func, default_return=None, error_context="Operation"):
        """Execute function with error handling"""
        try:
            return func()
        except Exception as e:
            ErrorHandler.handle_error(e, error_context)
            return default_return

# ==============================================================================
# 6. DATA PROCESSING ENGINE
# ==============================================================================
class PharmaDataProcessor:
    """Advanced pharmaceutical data processing engine"""
    
    def __init__(self):
        self.config = AppConfig()
        self.validator = DataValidator()
    
    def load_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file with comprehensive error handling"""
        try:
            file_name = uploaded_file.name.lower()
            
            with st.spinner(f"Loading {uploaded_file.name}..."):
                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, low_memory=False)
                elif file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                else:
                    st.error("❌ Unsupported file format. Please upload CSV or Excel file.")
                    return None
                
                # Clean column names
                df.columns = [str(col).strip() for col in df.columns]
                
                # Validate
                is_valid, message = self.validator.is_valid_dataframe(df)
                if not is_valid:
                    st.error(f"❌ Invalid data: {message}")
                    return None
                
                return df
                
        except Exception as e:
            ErrorHandler.handle_error(e, "Data Loading", show_traceback=True)
            return None
    
    def detect_mat_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect MAT period columns with improved pattern matching"""
        if df is None or df.empty:
            return []
        
        mat_columns = []
        for col in df.columns:
            col_str = str(col)
            for pattern in self.config.MAT_PATTERNS:
                if re.search(pattern, col_str, re.IGNORECASE):
                    mat_columns.append(col)
                    break
        
        return mat_columns
    
    def parse_mat_period(self, column_name: str) -> Optional[str]:
        """Extract MAT period from column name"""
        match = re.search(r'MAT Q(\d) (\d{4})', str(column_name), re.IGNORECASE)
        if match:
            quarter = match.group(1)
            year = match.group(2)
            return f"MAT Q{quarter} {year}"
        return None
    
    def parse_mat_metric(self, column_name: str) -> Optional[str]:
        """Identify metric type from column name"""
        col_upper = str(column_name).upper()
        
        # Check each metric type
        for metric_key, patterns in self.config.METRIC_COLUMNS.items():
            for pattern in patterns:
                if pattern in col_upper:
                    # Additional checks for price metrics
                    if metric_key == 'SALES_USD' and 'AVG PRICE' in col_upper:
                        continue
                    if metric_key == 'STANDARD_UNITS' and 'PRICE' in col_upper:
                        continue
                    return metric_key
        
        return None
    
    def process_mat_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Transform MAT format data into analytical long format"""
        try:
            # Validate input
            is_valid, message = self.validator.is_valid_dataframe(df)
            if not is_valid:
                st.warning(f"⚠️ Cannot process data: {message}")
                return df
            
            # Detect MAT columns
            mat_columns = self.detect_mat_columns(df)
            
            if not mat_columns:
                st.warning("⚠️ No MAT period columns detected. Showing raw data.")
                return df
            
            st.info(f"✓ Detected {len(mat_columns)} MAT columns")
            
            # Identify non-MAT columns
            non_mat_cols = [col for col in df.columns if col not in mat_columns]
            
            # Build period-metric mapping
            periods = {}
            for col in mat_columns:
                period = self.parse_mat_period(col)
                metric = self.parse_mat_metric(col)
                
                if period and metric:
                    if period not in periods:
                        periods[period] = {}
                    periods[period][metric] = col
            
            if not periods:
                st.warning("⚠️ Could not parse MAT periods. Showing raw data.")
                return df
            
            # Transform to long format
            long_data = []
            
            progress_bar = st.progress(0)
            total_periods = len(periods)
            
            for idx, (period, metrics) in enumerate(periods.items()):
                # Update progress
                progress_bar.progress((idx + 1) / total_periods)
                
                # Create period dataframe
                period_df = df[non_mat_cols].copy()
                period_df['PERIOD'] = period
                
                # Add metrics with safe conversion
                for metric, col_name in metrics.items():
                    if col_name in df.columns:
                        period_df[metric] = df[col_name].apply(
                            self.validator.safe_numeric_conversion
                        )
                
                long_data.append(period_df)
            
            progress_bar.empty()
            
            if not long_data:
                st.warning("⚠️ No data could be processed.")
                return df
            
            # Combine all periods
            result_df = pd.concat(long_data, ignore_index=True)
            
            # Extract temporal components
            result_df['YEAR'] = result_df['PERIOD'].str.extract(r'(\d{4})')
            result_df['QUARTER'] = result_df['PERIOD'].str.extract(r'Q(\d)')
            result_df['PERIOD_SORT'] = result_df['YEAR'] + result_df['QUARTER'].fillna('0')
            
            # Sort by period
            result_df = result_df.sort_values('PERIOD_SORT')
            
            # Standardize column names
            result_df.columns = [col.upper() for col in result_df.columns]
            
            st.success(f"✅ Processed {len(result_df):,} records across {len(periods)} periods")
            
            return result_df
            
        except Exception as e:
            ErrorHandler.handle_error(e, "MAT Data Processing", show_traceback=True)
            return df
    
    def calculate_market_metrics(
        self, 
        df: pd.DataFrame, 
        group_by: str, 
        period: str, 
        metric: str
    ) -> pd.DataFrame:
        """Calculate comprehensive market metrics with growth analysis"""
        try:
            # Validate inputs
            if df is None or df.empty:
                return pd.DataFrame()
            
            if group_by not in df.columns:
                st.warning(f"⚠️ Column '{group_by}' not found in data")
                return pd.DataFrame()
            
            if metric not in df.columns:
                st.warning(f"⚠️ Metric '{metric}' not found in data")
                return pd.DataFrame()
            
            # Filter for current period
            period_data = df[df['PERIOD'] == period].copy()
            
            if period_data.empty:
                st.warning(f"⚠️ No data found for period '{period}'")
                return pd.DataFrame()
            
            # Calculate current period metrics
            current_metrics = period_data.groupby(group_by).agg({
                metric: ['sum', 'count', 'mean']
            }).reset_index()
            
            current_metrics.columns = [group_by, 'CURRENT_VALUE', 'PRODUCT_COUNT', 'AVG_VALUE']
            
            # Calculate market share
            total_value = current_metrics['CURRENT_VALUE'].sum()
            if total_value > 0:
                current_metrics['MARKET_SHARE_PCT'] = (
                    current_metrics['CURRENT_VALUE'] / total_value * 100
                ).round(2)
            else:
                current_metrics['MARKET_SHARE_PCT'] = 0
            
            # Get previous period for growth calculation
            periods = sorted(df['PERIOD'].unique())
            
            if period in periods:
                period_idx = periods.index(period)
                
                if period_idx > 0:
                    prev_period = periods[period_idx - 1]
                    prev_data = df[df['PERIOD'] == prev_period].copy()
                    
                    if not prev_data.empty:
                        # Calculate previous period metrics
                        prev_metrics = prev_data.groupby(group_by)[metric].sum().reset_index()
                        prev_metrics.columns = [group_by, 'PREVIOUS_VALUE']
                        
                        # Merge with current
                        current_metrics = pd.merge(
                            current_metrics, 
                            prev_metrics, 
                            on=group_by, 
                            how='left'
                        )
                        
                        current_metrics['PREVIOUS_VALUE'] = current_metrics['PREVIOUS_VALUE'].fillna(0)
                        
                        # Calculate growth metrics
                        current_metrics['ABS_GROWTH'] = (
                            current_metrics['CURRENT_VALUE'] - current_metrics['PREVIOUS_VALUE']
                        ).round(2)
                        
                        current_metrics['GROWTH_PCT'] = current_metrics.apply(
                            lambda x: (
                                (x['CURRENT_VALUE'] - x['PREVIOUS_VALUE']) / x['PREVIOUS_VALUE'] * 100
                            ).round(2) if x['PREVIOUS_VALUE'] > 0 else 0,
                            axis=1
                        )
            
            # Add rankings
            current_metrics = current_metrics.sort_values('CURRENT_VALUE', ascending=False)
            current_metrics['RANK'] = range(1, len(current_metrics) + 1)
            
            # Remove any NaN values
            current_metrics = current_metrics.fillna(0)
            
            return current_metrics
            
        except Exception as e:
            ErrorHandler.handle_error(e, f"Metric Calculation for {group_by}")
            return pd.DataFrame()

# ==============================================================================
# 7. VISUALIZATION ENGINE
# ==============================================================================
class PharmaVisualizer:
    """Professional visualization engine with consistent styling"""
    
    def __init__(self):
        self.config = AppConfig()
        self.color_scheme = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'info': '#3b82f6'
        }
    
    def create_dashboard_metrics(self, df: pd.DataFrame, period: str):
        """Create enhanced metric cards with change indicators"""
        try:
            period_data = df[df['PERIOD'] == period]
            
            if period_data.empty:
                st.warning(f"⚠️ No data available for period {period}")
                return
            
            # Get previous period for comparison
            periods = sorted(df['PERIOD'].unique())
            prev_period = None
            if period in periods:
                idx = periods.index(period)
                if idx > 0:
                    prev_period = periods[idx - 1]
            
            metrics = []
            
            # Sales metric
            if 'SALES_USD' in period_data.columns:
                current_sales = period_data['SALES_USD'].sum()
                prev_sales = 0
                if prev_period:
                    prev_data = df[df['PERIOD'] == prev_period]
                    if 'SALES_USD' in prev_data.columns:
                        prev_sales = prev_data['SALES_USD'].sum()
                
                change_pct = ((current_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
                
                metrics.append({
                    'label': '💰 Total Sales',
                    'value': f"${current_sales:,.0f}",
                    'subtext': period,
                    'change': f"{change_pct:+.1f}%" if prev_period else ""
                })
            
            # Volume metric
            if 'STANDARD_UNITS' in period_data.columns:
                volume = period_data['STANDARD_UNITS'].sum()
                metrics.append({
                    'label': '📦 Total Volume',
                    'value': f"{volume:,.0f}",
                    'subtext': 'Standard Units',
                    'change': ''
                })
            
            # Manufacturers
            if 'MANUFACTURER' in period_data.columns:
                mfg_count = period_data['MANUFACTURER'].nunique()
                metrics.append({
                    'label': '🏭 Manufacturers',
                    'value': f"{mfg_count}",
                    'subtext': 'Active Companies',
                    'change': ''
                })
            
            # Molecules
            if 'MOLECULE' in period_data.columns:
                mol_count = period_data['MOLECULE'].nunique()
                metrics.append({
                    'label': '🧪 Molecules',
                    'value': f"{mol_count}",
                    'subtext': 'Unique Products',
                    'change': ''
                })
            
            # Average Price
            if 'SU_AVG_PRICE' in period_data.columns:
                avg_price = period_data[period_data['SU_AVG_PRICE'] > 0]['SU_AVG_PRICE'].mean()
                if pd.notna(avg_price):
                    metrics.append({
                        'label': '💵 Avg Price',
                        'value': f"${avg_price:.2f}",
                        'subtext': 'Per Unit',
                        'change': ''
                    })
            
            # Display metrics
            if metrics:
                cols = st.columns(len(metrics))
                for idx, metric in enumerate(metrics):
                    with cols[idx]:
                        change_class = ''
                        if metric['change']:
                            change_class = 'positive' if '+' in metric['change'] else 'negative'
                        
                        change_html = f'<div class="metric-change {change_class}">{metric["change"]}</div>' if metric['change'] else ''
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{metric['label']}</div>
                            <div class="metric-value">{metric['value']}</div>
                            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">
                                {metric['subtext']}
                            </div>
                            {change_html}
                        </div>
                        """, unsafe_allow_html=True)
        
        except Exception as e:
            ErrorHandler.handle_error(e, "Dashboard Metrics")
    
    def create_market_share_chart(
        self, 
        data: pd.DataFrame, 
        title: str, 
        group_by: str = 'MANUFACTURER'
    ) -> Optional[go.Figure]:
        """Create professional market share visualization"""
        try:
            if data.empty or group_by not in data.columns:
                return None
            
            # Take top N
            plot_data = data.head(self.config.DEFAULT_TOP_N)
            
            fig = px.bar(
                plot_data,
                x='MARKET_SHARE_PCT',
                y=group_by,
                orientation='h',
                title=title,
                text='MARKET_SHARE_PCT',
                color='MARKET_SHARE_PCT',
                color_continuous_scale='Viridis',
                labels={'MARKET_SHARE_PCT': 'Market Share (%)'}
            )
            
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                textfont_size=11
            )
            
            fig.update_layout(
                height=self.config.CHART_HEIGHT,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title="Market Share (%)",
                yaxis_title="",
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                margin=dict(t=60, l=20, r=40, b=50),
                font=dict(family="Inter, sans-serif")
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
            
            return fig
            
        except Exception as e:
            ErrorHandler.handle_error(e, "Market Share Chart")
            return None
    
    def create_trend_chart(
        self, 
        df: pd.DataFrame, 
        metric: str, 
        group_by: Optional[str] = None
    ) -> Optional[go.Figure]:
        """Create trend analysis visualization"""
        try:
            if df.empty or metric not in df.columns:
                return None
            
            if group_by and group_by in df.columns:
                # Multi-line trend
                trend_data = df.groupby(['PERIOD', group_by])[metric].sum().reset_index()
                top_groups = trend_data.groupby(group_by)[metric].sum().nlargest(5).index
                filtered_data = trend_data[trend_data[group_by].isin(top_groups)]
                
                fig = px.line(
                    filtered_data,
                    x='PERIOD',
                    y=metric,
                    color=group_by,
                    markers=True,
                    title=f"{metric.replace('_', ' ').title()} Trend (Top 5 {group_by.title()})",
                    labels={metric: metric.replace('_', ' ').title()}
                )
            else:
                # Single line trend
                trend_data = df.groupby('PERIOD')[metric].sum().reset_index()
                
                fig = px.line(
                    trend_data,
                    x='PERIOD',
                    y=metric,
                    markers=True,
                    title=f"{metric.replace('_', ' ').title()} Trend",
                    labels={metric: metric.replace('_', ' ').title()}
                )
                
                fig.update_traces(line_color=self.color_scheme['primary'], line_width=3)
            
            fig.update_layout(
                height=450,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title="Period",
                yaxis_title=metric.replace('_', ' ').title(),
                hovermode='x unified',
                font=dict(family="Inter, sans-serif")
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
            
            return fig
            
        except Exception as e:
            ErrorHandler.handle_error(e, "Trend Chart")
            return None
    
    def create_growth_chart(
        self, 
        data: pd.DataFrame, 
        title: str, 
        group_by: str = 'MANUFACTURER'
    ) -> Optional[go.Figure]:
        """Create growth comparison chart"""
        try:
            if data.empty or 'GROWTH_PCT' not in data.columns:
                return None
            
            # Filter valid growth data and take top/bottom
            growth_data = data[data['GROWTH_PCT'] != 0].copy()
            growth_data = growth_data.sort_values('GROWTH_PCT', ascending=False).head(10)
            
            if growth_data.empty:
                return None
            
            fig = px.bar(
                growth_data,
                x=group_by,
                y='GROWTH_PCT',
                title=title,
                text='GROWTH_PCT',
                color='GROWTH_PCT',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                labels={'GROWTH_PCT': 'Growth (%)'}
            )
            
            fig.update_traces(
                texttemplate='%{text:+.1f}%',
                textposition='outside'
            )
            
            fig.update_layout(
                height=450,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title="",
                yaxis_title="Growth (%)",
                xaxis_tickangle=-45,
                showlegend=False,
                font=dict(family="Inter, sans-serif")
            )
            
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
            
            return fig
            
        except Exception as e:
            ErrorHandler.handle_error(e, "Growth Chart")
            return None

# ==============================================================================
# 8. DASHBOARD COMPONENTS
# ==============================================================================
class DashboardComponents:
    """Reusable UI components"""
    
    def __init__(self):
        self.config = AppConfig()
    
    def create_sidebar(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create enhanced sidebar with professional styling"""
        with st.sidebar:
            st.markdown(
                '<div class="sidebar-header">🔧 Control Center</div>', 
                unsafe_allow_html=True
            )
            
            filters = {}
            
            # Period Selection
            if 'PERIOD' in df.columns:
                periods = sorted(df['PERIOD'].unique(), reverse=True)
                if periods:
                    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                    filters['period'] = st.selectbox(
                        "📅 Analysis Period",
                        periods,
                        index=0,
                        help="Select the MAT period to analyze"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    filters['period'] = None
            else:
                filters['period'] = None
                st.warning("⚠️ No period data available")
            
            # Metric Selection
            available_metrics = []
            metric_display = {}
            
            if 'SALES_USD' in df.columns:
                available_metrics.append('SALES_USD')
                metric_display['SALES_USD'] = '💰 Sales (USD)'
            
            if 'STANDARD_UNITS' in df.columns:
                available_metrics.append('STANDARD_UNITS')
                metric_display['STANDARD_UNITS'] = '📦 Volume (Standard Units)'
            
            if 'SU_AVG_PRICE' in df.columns:
                available_metrics.append('SU_AVG_PRICE')
                metric_display['SU_AVG_PRICE'] = '💵 Average Price'
            
            if available_metrics:
                st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
                filters['metric'] = st.selectbox(
                    "📊 Primary Metric",
                    available_metrics,
                    format_func=lambda x: metric_display.get(x, x),
                    index=0,
                    help="Select the main metric for analysis"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                filters['metric'] = None
                st.error("❌ No metrics available in data")
            
            st.markdown("---")
            
            # Advanced Filters
            st.markdown("### 🎯 Advanced Filters")
            
            # Manufacturer Filter
            if 'MANUFACTURER' in df.columns:
                manufacturers = sorted(df['MANUFACTURER'].dropna().unique())
                if manufacturers:
                    filters['manufacturers'] = st.multiselect(
                        "🏭 Manufacturers",
                        manufacturers,
                        default=None,
                        help="Filter by specific manufacturers"
                    )
                else:
                    filters['manufacturers'] = []
            else:
                filters['manufacturers'] = []
            
            # Molecule Filter
            if 'MOLECULE' in df.columns:
                molecules = sorted(df['MOLECULE'].dropna().unique())
                if molecules:
                    filters['molecules'] = st.multiselect(
                        "🧪 Molecules",
                        molecules,
                        default=None,
                        help="Filter by specific molecules"
                    )
                else:
                    filters['molecules'] = []
            else:
                filters['molecules'] = []
            
            # Country Filter
            if 'COUNTRY' in df.columns:
                countries = sorted(df['COUNTRY'].dropna().unique())
                if countries:
                    filters['countries'] = st.multiselect(
                        "🌍 Countries",
                        countries,
                        default=None,
                        help="Filter by specific countries"
                    )
                else:
                    filters['countries'] = []
            else:
                filters['countries'] = []
            
            st.markdown("---")
            
            # Dataset Information
            self._show_dataset_info(df)
            
            st.markdown("---")
            
            # Action Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Export", use_container_width=True):
                    st.session_state['show_export'] = True
            
            with col2:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            
            return filters
    
    def _show_dataset_info(self, df: pd.DataFrame):
        """Display dataset information"""
        st.markdown("### 📊 Dataset Info")
        
        info_items = []
        info_items.append(f"**Records:** {len(df):,}")
        
        if 'PERIOD' in df.columns:
            periods = df['PERIOD'].nunique()
            latest = df['PERIOD'].max()
            info_items.append(f"**Periods:** {periods}")
            info_items.append(f"**Latest:** {latest}")
        
        if 'MANUFACTURER' in df.columns:
            info_items.append(f"**Manufacturers:** {df['MANUFACTURER'].nunique()}")
        
        if 'MOLECULE' in df.columns:
            info_items.append(f"**Molecules:** {df['MOLECULE'].nunique()}")
        
        if 'COUNTRY' in df.columns:
            info_items.append(f"**Countries:** {df['COUNTRY'].nunique()}")
        
        info_html = '<div class="sidebar-section">'
        for item in info_items:
            info_html += f'<p style="margin: 0.4rem 0; color: #f1f5f9 !important;">{item}</p>'
        info_html += '</div>'
        
        st.markdown(info_html, unsafe_allow_html=True)
    
    def show_export_dialog(self, df: pd.DataFrame):
        """Show export options dialog"""
        if st.session_state.get('show_export', False):
            st.markdown("### 📥 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"pharma_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel Export
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                
                st.download_button(
                    label="📊 Download Excel",
                    data=output.getvalue(),
                    file_name=f"pharma_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            if st.button("✖️ Close", use_container_width=True):
                st.session_state['show_export'] = False
                st.rerun()

# ==============================================================================
# 9. MAIN APPLICATION
# ==============================================================================
class PharmaAnalyticsApp:
    """Main application orchestrator"""
    
    def __init__(self):
        self.config = AppConfig()
        self.processor = PharmaDataProcessor()
        self.visualizer = PharmaVisualizer()
        self.components = DashboardComponents()
        apply_custom_styling()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'processed' not in st.session_state:
            st.session_state.processed = False
        if 'show_export' not in st.session_state:
            st.session_state.show_export = False
    
    def run(self):
        """Main application entry point"""
        self._show_header()
        
        # File Upload
        uploaded_file = st.file_uploader(
            "📁 Upload Pharmaceutical Data File",
            type=['csv', 'xlsx', 'xls'],
            help=f"Supported formats: CSV, Excel | Max size: {self.config.MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file:
            if not st.session_state.processed:
                self._process_file(uploaded_file)
            
            if st.session_state.data is not None:
                df = st.session_state.data
                
                # Show export dialog if requested
                self.components.show_export_dialog(df)
                
                # Create sidebar and get filters
                filters = self.components.create_sidebar(df)
                
                # Apply filters
                filtered_df = self._apply_filters(df, filters)
                
                if not filtered_df.empty:
                    # Render dashboard tabs
                    self._render_dashboard_tabs(filtered_df, filters)
                else:
                    st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")
        else:
            # Reset processed state when file is removed
            if st.session_state.processed:
                st.session_state.processed = False
                st.session_state.data = None
            
            self._show_welcome_screen()
        
        self._show_footer()
    
    def _show_header(self):
        """Display application header"""
        st.markdown(f"""
        <div class="main-header">
            {self.config.APP_NAME}
        </div>
        <div class="main-subheader">
            Advanced MAT Data Analytics | Version {self.config.APP_VERSION} | Production-Ready
        </div>
        """, unsafe_allow_html=True)
    
    def _show_welcome_screen(self):
        """Display welcome screen with instructions"""
        st.info("👆 **Get Started:** Upload your pharmaceutical MAT data file to begin analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("📋 **Data Format Guide**", expanded=True):
                st.markdown("""
                ### Expected Data Structure
                
                **Core Columns:**
                - `Manufacturer` - Company name
                - `Molecule` - Active pharmaceutical ingredient
                - `Country` - Market geography
                - `Sector` - Business sector
                
                **MAT Period Columns Format:**
                ```
                MAT Q3 2022 USD MNF
                MAT Q3 2022 Standard Units
                MAT Q3 2022 SU Avg Price USD MNF
                MAT Q3 2023 USD MNF
                MAT Q3 2023 Standard Units
                ...
                ```
                
                **Supported Metrics:**
                - Sales (USD MNF)
                - Volume (Standard Units, Units)
                - Pricing (SU Avg Price, Unit Avg Price)
                
                **File Requirements:**
                - Format: CSV or Excel (.xlsx, .xls)
                - Size: Up to 200 MB
                - Encoding: UTF-8 recommended
                """)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h3>✨ Key Features</h3>
            <ul>
                <li>📊 Executive Dashboard</li>
                <li>🏭 Manufacturer Analysis</li>
                <li>🧪 Molecule Tracking</li>
                <li>📈 Trend Analysis</li>
                <li>🌍 Geographic Insights</li>
                <li>💰 Price Analytics</li>
                <li>📥 Data Export</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def _process_file(self, uploaded_file):
        """Process uploaded data file"""
        with st.spinner("🔄 Processing data..."):
            # Load raw data
            raw_df = self.processor.load_data(uploaded_file)
            
            if raw_df is not None:
                # Process MAT format
                processed_df = self.processor.process_mat_data(raw_df)
                
                if processed_df is not None and not processed_df.empty:
                    st.session_state.data = processed_df
                    st.session_state.processed = True
                    
                    # Show summary
                    mat_cols = self.processor.detect_mat_columns(raw_df)
                    periods = processed_df['PERIOD'].nunique() if 'PERIOD' in processed_df.columns else 0
                    
                    st.success(f"""
                    ✅ **Data Processing Complete!**
                    
                    - MAT columns detected: **{len(mat_cols)}**
                    - Periods identified: **{periods}**
                    - Total records: **{len(processed_df):,}**
                    - Columns: **{len(processed_df.columns)}**
                    """)
                else:
                    st.error("❌ Failed to process data. Please check the file format.")
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply selected filters to dataframe"""
        filtered_df = df.copy()
        
        try:
            # Period filter
            if filters.get('period'):
                filtered_df = filtered_df[filtered_df['PERIOD'] == filters['period']]
            
            # Manufacturer filter
            if filters.get('manufacturers') and len(filters['manufacturers']) > 0:
                filtered_df = filtered_df[filtered_df['MANUFACTURER'].isin(filters['manufacturers'])]
            
            # Molecule filter
            if filters.get('molecules') and len(filters['molecules']) > 0:
                filtered_df = filtered_df[filtered_df['MOLECULE'].isin(filters['molecules'])]
            
            # Country filter
            if filters.get('countries') and len(filters['countries']) > 0:
                filtered_df = filtered_df[filtered_df['COUNTRY'].isin(filters['countries'])]
            
            return filtered_df
            
        except Exception as e:
            ErrorHandler.handle_error(e, "Filter Application")
            return df
    
    def _render_dashboard_tabs(self, df: pd.DataFrame, filters: Dict[str, Any]):
        """Render all dashboard tabs"""
        tabs = st.tabs([
            "📊 Dashboard",
            "🏭 Manufacturers",
            "🧪 Molecules",
            "📈 Trends",
            "🌍 Geography",
            "💰 Pricing",
            "📋 Data Explorer"
        ])
        
        with tabs[0]:
            self._render_dashboard_tab(df, filters)
        
        with tabs[1]:
            self._render_manufacturers_tab(df, filters)
        
        with tabs[2]:
            self._render_molecules_tab(df, filters)
        
        with tabs[3]:
            self._render_trends_tab(df, filters)
        
        with tabs[4]:
            self._render_geography_tab(df, filters)
        
        with tabs[5]:
            self._render_pricing_tab(df, filters)
        
        with tabs[6]:
            self._render_data_explorer_tab(df)
    
    def _render_dashboard_tab(self, df: pd.DataFrame, filters: Dict[str, Any]):
        """Render executive dashboard"""
        st.markdown('<div class="sub-header">📊 Executive Dashboard</div>', unsafe_allow_html=True)
        
        if not filters.get('period'):
            st.warning("⚠️ Please select a period from the sidebar")
            return
        
        # Key metrics
        self.visualizer.create_dashboard_metrics(df, filters['period'])
        
        st.markdown("---")
        
        # Charts
        if filters.get('metric'):
            col1, col2 = st.columns(2)
            
            with col1:
                # Market share
                market_data = self.processor.calculate_market_metrics(
                    df, 'MANUFACTURER', filters['period'], filters['metric']
                )
                
                if not market_data.empty:
                    fig = self.visualizer.create_market_share_chart(
                        market_data,
                        f"Top Manufacturers - {filters['period']}"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Growth
                if not market_data.empty and 'GROWTH_PCT' in market_data.columns:
                    fig = self.visualizer.create_growth_chart(
                        market_data,
                        f"Growth Leaders - {filters['period']}"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_manufacturers_tab(self, df: pd.DataFrame, filters: Dict[str, Any]):
        """Render manufacturer analysis"""
        st.markdown('<div class="sub-header">🏭 Manufacturer Analysis</div>', unsafe_allow_html=True)
        
        if not filters.get('period') or not filters.get('metric'):
            st.warning("⚠️ Please select both period and metric")
            return
        
        manufacturer_data = self.processor.calculate_market_metrics(
            df, 'MANUFACTURER', filters['period'], filters['metric']
        )
        
        if not manufacturer_data.empty:
            st.markdown(f"#### Top 20 Manufacturers - {filters['period']}")
            
            # Prepare display dataframe
            display_cols = ['RANK', 'MANUFACTURER', 'CURRENT_VALUE', 'MARKET_SHARE_PCT']
            if 'GROWTH_PCT' in manufacturer_data.columns:
                display_cols.append('GROWTH_PCT')
            
            display_df = manufacturer_data[display_cols].head(20).copy()
            display_df.columns = ['Rank', 'Manufacturer', 'Value', 'Market Share %'] + (['Growth %'] if 'GROWTH_PCT' in manufacturer_data.columns else [])
            
            # Format values
            display_df['Value'] = display_df['Value'].apply(lambda x: f"{x:,.0f}")
            display_df['Market Share %'] = display_df['Market Share %'].apply(lambda x: f"{x:.2f}%")
            if 'Growth %' in display_df.columns:
                display_df['Growth %'] = display_df['Growth %'].apply(lambda x: f"{x:+.1f}%")
            
            st.dataframe(display_df, use_container_width=True, height=self.config.TABLE_HEIGHT)
        else:
            st.info("ℹ️ No manufacturer data available for the selected filters")
    
    def _render_molecules_tab(self, df: pd.DataFrame, filters: Dict[str, Any]):
        """Render molecule analysis"""
        st.markdown('<div class="sub-header">🧪 Molecule Analysis</div>', unsafe_allow_html=True)
        
        if not filters.get('period') or not filters.get('metric'):
            st.warning("⚠️ Please select both period and metric")
            return
        
        molecule_data = self.processor.calculate_market_metrics(
            df, 'MOLECULE', filters['period'], filters['metric']
        )
        
        if not molecule_data.empty:
            fig = px.bar(
                molecule_data.head(15),
                x='MOLECULE',
                y='CURRENT_VALUE',
                title=f"Top Molecules - {filters['period']}",
                color='MARKET_SHARE_PCT',
                color_continuous_scale='Plasma',
                labels={'CURRENT_VALUE': 'Value', 'MARKET_SHARE_PCT': 'Market Share (%)'}
            )
            
            fig.update_layout(
                height=self.config.CHART_HEIGHT,
                xaxis_tickangle=-45,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No molecule data available")
    
    def _render_trends_tab(self, df: pd.DataFrame, filters: Dict[str, Any]):
        """Render trend analysis"""
        st.markdown('<div class="sub-header">📈 Trend Analysis</div>', unsafe_allow_html=True)
        
        if not filters.get('metric'):
            st.warning("⚠️ Please select a metric")
            return
        
        # Overall trend
        fig = self.visualizer.create_trend_chart(df, filters['metric'])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Grouped trends
        col1, col2 = st.columns(2)
        
        with col1:
            if 'MANUFACTURER' in df.columns:
                fig = self.visualizer.create_trend_chart(df, filters['metric'], 'MANUFACTURER')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'MOLECULE' in df.columns:
                fig = self.visualizer.create_trend_chart(df, filters['metric'], 'MOLECULE')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_geography_tab(self, df: pd.DataFrame, filters: Dict[str, Any]):
        """Render geographic analysis"""
        st.markdown('<div class="sub-header">🌍 Geographic Analysis</div>', unsafe_allow_html=True)
        
        if 'COUNTRY' not in df.columns:
            st.info("ℹ️ Geographic data not available in this dataset")
            return
        
        if not filters.get('period') or not filters.get('metric'):
            st.warning("⚠️ Please select both period and metric")
            return
        
        country_data = self.processor.calculate_market_metrics(
            df, 'COUNTRY', filters['period'], filters['metric']
        )
        
        if not country_data.empty:
            fig = px.bar(
                country_data.head(15),
                x='COUNTRY',
                y='CURRENT_VALUE',
                title=f"Top Countries - {filters['period']}",
                color='MARKET_SHARE_PCT',
                color_continuous_scale='Viridis',
                labels={'CURRENT_VALUE': 'Value', 'MARKET_SHARE_PCT': 'Market Share (%)'}
            )
            
            fig.update_layout(
                height=self.config.CHART_HEIGHT,
                xaxis_tickangle=-45,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_pricing_tab(self, df: pd.DataFrame, filters: Dict[str, Any]):
        """Render pricing analysis"""
        st.markdown('<div class="sub-header">💰 Price Analysis</div>', unsafe_allow_html=True)
        
        if 'SU_AVG_PRICE' not in df.columns:
            st.info("ℹ️ Pricing data not available in this dataset")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            price_data = df[df['SU_AVG_PRICE'] > 0]
            
            if len(price_data) > 0:
                fig = px.histogram(
                    price_data,
                    x='SU_AVG_PRICE',
                    nbins=50,
                    title="Price Distribution",
                    labels={'SU_AVG_PRICE': 'Average Price (USD)'}
                )
                
                fig.update_layout(
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Inter, sans-serif")
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price by manufacturer
            if filters.get('period') and 'MANUFACTURER' in df.columns:
                period_data = df[df['PERIOD'] == filters['period']]
                
                if not period_data.empty:
                    price_by_mfg = period_data.groupby('MANUFACTURER')['SU_AVG_PRICE'].mean().nlargest(10).reset_index()
                    
                    fig = px.bar(
                        price_by_mfg,
                        x='MANUFACTURER',
                        y='SU_AVG_PRICE',
                        title=f"Average Price by Manufacturer - {filters['period']}",
                        color='SU_AVG_PRICE',
                        color_continuous_scale='Viridis',
                        labels={'SU_AVG_PRICE': 'Avg Price (USD)'}
                    )
                    
                    fig.update_layout(
                        height=400,
                        xaxis_tickangle=-45,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_data_explorer_tab(self, df: pd.DataFrame):
        """Render data explorer"""
        st.markdown('<div class="sub-header">📋 Data Explorer</div>', unsafe_allow_html=True)
        
        # Data preview
        st.dataframe(df, use_container_width=True, height=500)
        
        # Data statistics
        st.markdown("### 📊 Data Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Basic Info**")
            st.write(f"Rows: {len(df):,}")
            st.write(f"Columns: {len(df.columns)}")
            st.write(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with col2:
            st.markdown("**Data Types**")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"{dtype}: {count}")
        
        with col3:
            st.markdown("**Missing Values**")
            missing = df.isnull().sum()
            total_missing = missing.sum()
            st.write(f"Total: {total_missing:,}")
            if total_missing > 0:
                st.write(f"Percentage: {total_missing/len(df)*100:.2f}%")
    
    def _show_footer(self):
        """Display application footer"""
        st.markdown(f"""
        <div class="footer">
            <p><strong>{self.config.APP_NAME}</strong> | Version {self.config.APP_VERSION}</p>
            <p style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.5rem;">
                Production-Ready Analytics Platform | Confidential - For Internal Use Only
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 10. APPLICATION ENTRY POINT
# ==============================================================================
def main():
    """Application entry point with error handling"""
    try:
        app = PharmaAnalyticsApp()
        app.run()
    except Exception as e:
        st.error("🚨 **Critical Application Error**")
        st.error(f"**Error:** {str(e)}")
        
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
        
        st.info("💡 **Troubleshooting:** Try refreshing the page or uploading a different file.")

if __name__ == "__main__":
    main()
