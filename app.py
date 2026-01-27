"""
================================================================================
PHARMACEUTICAL MARKET INTELLIGENCE PLATFORM
MAT Data Analytics Dashboard
Version: 3.1.0 | Python 3.11+ Compatible
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
from datetime import datetime
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# ==============================================================================
# 3. CUSTOM CSS STYLING
# ==============================================================================
def apply_custom_styling():
    """Apply custom CSS styling for professional appearance"""
    st.markdown("""
    <style>
    /* ========== MAIN STYLES ========== */
    .main {
        padding: 2rem;
    }
    
    /* ========== HEADER ========== */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #667eea 0%, #764ba2 100%) 1;
    }
    
    /* ========== METRIC CARDS ========== */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        transition: transform 0.3s ease;
        height: 140px;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #2d3748;
        line-height: 1.2;
        margin-top: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* ========== SUB HEADERS ========== */
    .sub-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        border: 1px solid #e2e8f0;
        border-bottom: none;
        background-color: #f8fafc;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #edf2f7;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #667eea;
        border-color: #667eea;
        border-bottom-color: white;
    }
    
    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* ========== DATA TABLES ========== */
    .dataframe {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* ========== FOOTER ========== */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 4rem;
        color: #718096;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
        background: #f8fafc;
    }
    
    /* ========== LOADING SPINNER ========== */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. DATA PROCESSING ENGINE
# ==============================================================================
class PharmaDataProcessor:
    """Advanced data processing engine for pharmaceutical MAT data"""
    
    @staticmethod
    def load_data(uploaded_file):
        """Load data from uploaded file"""
        try:
            file_name = uploaded_file.name.lower()
            
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, low_memory=False)
            elif file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error("Unsupported file format. Please upload CSV or Excel file.")
                return None
            
            # Clean column names
            df.columns = [str(col).strip() for col in df.columns]
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    @staticmethod
    def detect_mat_columns(df):
        """Detect MAT period columns in the dataframe"""
        mat_patterns = [
            r'MAT Q\d \d{4}',
            r'MAT_Q\d_\d{4}',
            r'MAT\d{1}Q\d{4}',
            r'Moving Annual Total Q\d \d{4}'
        ]
        
        mat_columns = []
        for col in df.columns:
            col_str = str(col)
            for pattern in mat_patterns:
                if re.search(pattern, col_str):
                    mat_columns.append(col)
                    break
        
        return mat_columns
    
    @staticmethod
    def parse_mat_period(column_name):
        """Parse MAT period from column name"""
        match = re.search(r'MAT Q(\d) (\d{4})', str(column_name))
        if match:
            quarter = match.group(1)
            year = match.group(2)
            return f"MAT Q{quarter} {year}"
        return None
    
    @staticmethod
    def parse_mat_metric(column_name):
        """Parse metric type from column name"""
        col_str = str(column_name).upper()
        
        if 'USD MNF' in col_str and 'AVG PRICE' not in col_str:
            return 'SALES_USD'
        elif 'STANDARD UNITS' in col_str:
            return 'STANDARD_UNITS'
        elif 'UNITS' in col_str and 'STANDARD' not in col_str:
            return 'UNITS'
        elif 'SU AVG PRICE USD MNF' in col_str:
            return 'SU_AVG_PRICE'
        elif 'UNIT AVG PRICE USD MNF' in col_str:
            return 'UNIT_AVG_PRICE'
        else:
            return None
    
    @staticmethod
    def convert_to_numeric(value):
        """Safely convert value to numeric, handling European decimals"""
        if pd.isna(value):
            return 0.0
        
        try:
            # Convert to string first
            str_val = str(value).strip()
            
            # Handle European decimal format (comma as decimal separator)
            str_val = str_val.replace(',', '.')
            
            # Remove any non-numeric characters except dot and minus
            str_val = re.sub(r'[^\d\.\-]', '', str_val)
            
            if str_val == '' or str_val == '-':
                return 0.0
            
            return float(str_val)
        except:
            return 0.0
    
    @staticmethod
    def process_mat_data(df):
        """Process MAT format data into long format"""
        try:
            # Detect MAT columns
            mat_columns = PharmaDataProcessor.detect_mat_columns(df)
            
            if not mat_columns:
                st.warning("No MAT period columns found in the data.")
                return df
            
            # Identify non-MAT columns
            non_mat_cols = [col for col in df.columns if col not in mat_columns]
            
            # Extract unique periods
            periods = {}
            for col in mat_columns:
                period = PharmaDataProcessor.parse_mat_period(col)
                metric = PharmaDataProcessor.parse_mat_metric(col)
                
                if period and metric:
                    if period not in periods:
                        periods[period] = {}
                    periods[period][metric] = col
            
            # Transform to long format
            long_data = []
            
            for period, metrics in periods.items():
                # Create base dataframe for this period
                period_df = df[non_mat_cols].copy()
                period_df['PERIOD'] = period
                
                # Add metrics
                for metric, col_name in metrics.items():
                    if col_name in df.columns:
                        # Convert to numeric safely
                        period_df[metric] = df[col_name].apply(PharmaDataProcessor.convert_to_numeric)
                
                long_data.append(period_df)
            
            # Combine all periods
            if long_data:
                result_df = pd.concat(long_data, ignore_index=True)
                
                # Extract year and quarter
                result_df['YEAR'] = result_df['PERIOD'].str.extract(r'(\d{4})')
                result_df['QUARTER'] = result_df['PERIOD'].str.extract(r'Q(\d)')
                result_df['PERIOD_SORT'] = result_df['YEAR'] + result_df['QUARTER']
                result_df = result_df.sort_values('PERIOD_SORT')
                
                # Clean up column names
                result_df.columns = [col.upper() for col in result_df.columns]
                
                return result_df
            else:
                return df
                
        except Exception as e:
            st.error(f"Error processing MAT data: {str(e)}")
            return df
    
    @staticmethod
    def calculate_market_metrics(df, group_by, period, metric='SALES_USD'):
        """Calculate market share and growth metrics"""
        try:
            # Filter for specific period
            period_data = df[df['PERIOD'] == period].copy()
            
            if len(period_data) == 0:
                return pd.DataFrame()
            
            # Get previous period
            periods = sorted(df['PERIOD'].unique())
            period_idx = periods.index(period)
            
            if period_idx > 0:
                prev_period = periods[period_idx - 1]
                prev_data = df[df['PERIOD'] == prev_period].copy()
            else:
                prev_data = pd.DataFrame()
            
            # Calculate current period metrics
            current_metrics = period_data.groupby(group_by)[metric].agg(['sum', 'count']).reset_index()
            current_metrics.columns = [group_by, 'CURRENT_VALUE', 'PRODUCT_COUNT']
            
            # Calculate market share
            total_value = current_metrics['CURRENT_VALUE'].sum()
            if total_value > 0:
                current_metrics['MARKET_SHARE_PCT'] = (current_metrics['CURRENT_VALUE'] / total_value * 100).round(2)
            else:
                current_metrics['MARKET_SHARE_PCT'] = 0
            
            # Calculate growth if previous period exists
            if not prev_data.empty:
                prev_metrics = prev_data.groupby(group_by)[metric].sum().reset_index()
                prev_metrics.columns = [group_by, 'PREVIOUS_VALUE']
                
                merged = pd.merge(current_metrics, prev_metrics, on=group_by, how='left')
                merged['PREVIOUS_VALUE'] = merged['PREVIOUS_VALUE'].fillna(0)
                
                # Calculate growth
                merged['ABS_GROWTH'] = (merged['CURRENT_VALUE'] - merged['PREVIOUS_VALUE']).round(2)
                merged['GROWTH_PCT'] = merged.apply(
                    lambda x: ((x['CURRENT_VALUE'] - x['PREVIOUS_VALUE']) / x['PREVIOUS_VALUE'] * 100).round(2) 
                    if x['PREVIOUS_VALUE'] > 0 else 0, 
                    axis=1
                )
                
                current_metrics = merged
            
            # Add rank
            current_metrics = current_metrics.sort_values('CURRENT_VALUE', ascending=False)
            current_metrics['RANK'] = range(1, len(current_metrics) + 1)
            
            return current_metrics
            
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return pd.DataFrame()

# ==============================================================================
# 5. VISUALIZATION ENGINE
# ==============================================================================
class PharmaVisualizer:
    """Professional visualization engine for pharmaceutical data"""
    
    @staticmethod
    def create_dashboard_metrics(df, period):
        """Create key metric cards"""
        period_data = df[df['PERIOD'] == period]
        
        if len(period_data) == 0:
            return
        
        metrics = []
        
        # Sales metric
        if 'SALES_USD' in period_data.columns:
            sales = period_data['SALES_USD'].sum()
            metrics.append({
                'label': '💰 Total Sales',
                'value': f"${sales:,.0f}",
                'subtext': period
            })
        
        # Volume metric
        if 'STANDARD_UNITS' in period_data.columns:
            volume = period_data['STANDARD_UNITS'].sum()
            metrics.append({
                'label': '📦 Total Volume',
                'value': f"{volume:,.0f}",
                'subtext': 'Standard Units'
            })
        
        # Manufacturers metric
        if 'MANUFACTURER' in period_data.columns:
            mfg_count = period_data['MANUFACTURER'].nunique()
            metrics.append({
                'label': '🏭 Manufacturers',
                'value': f"{mfg_count}",
                'subtext': 'Active'
            })
        
        # Molecules metric
        if 'MOLECULE' in period_data.columns:
            mol_count = period_data['MOLECULE'].nunique()
            metrics.append({
                'label': '🧪 Molecules',
                'value': f"{mol_count}",
                'subtext': 'Unique'
            })
        
        # Average Price metric
        if 'SU_AVG_PRICE' in period_data.columns:
            avg_price = period_data['SU_AVG_PRICE'].mean()
            metrics.append({
                'label': '💵 Avg Price',
                'value': f"${avg_price:.2f}",
                'subtext': 'Per Unit'
            })
        
        # Display metrics
        cols = st.columns(len(metrics))
        for idx, metric in enumerate(metrics):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric['label']}</div>
                    <div class="metric-value">{metric['value']}</div>
                    <div style="font-size: 0.8rem; color: #718096; margin-top: 0.5rem;">
                        {metric['subtext']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def create_market_share_chart(data, title, group_by='MANUFACTURER'):
        """Create horizontal bar chart for market share"""
        if data.empty:
            return None
        
        fig = px.bar(
            data.head(15),
            x='MARKET_SHARE_PCT',
            y=group_by,
            orientation='h',
            title=title,
            text='MARKET_SHARE_PCT',
            color='MARKET_SHARE_PCT',
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Market Share (%)",
            yaxis_title="",
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            margin=dict(t=50, l=0, r=0, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_trend_chart(df, metric='SALES_USD', group_by=None):
        """Create trend chart over periods"""
        if metric not in df.columns:
            return None
        
        if group_by and group_by in df.columns:
            # Grouped trend
            trend_data = df.groupby(['PERIOD', group_by])[metric].sum().reset_index()
            top_groups = trend_data.groupby(group_by)[metric].sum().nlargest(5).index
            
            filtered_data = trend_data[trend_data[group_by].isin(top_groups)]
            
            fig = px.line(
                filtered_data,
                x='PERIOD',
                y=metric,
                color=group_by,
                markers=True,
                title=f"{metric.replace('_', ' ')} Trend by {group_by}",
                labels={metric: metric.replace('_', ' ')}
            )
        else:
            # Overall trend
            trend_data = df.groupby('PERIOD')[metric].sum().reset_index()
            
            fig = px.line(
                trend_data,
                x='PERIOD',
                y=metric,
                markers=True,
                title=f"{metric.replace('_', ' ')} Trend",
                labels={metric: metric.replace('_', ' ')}
            )
        
        fig.update_layout(
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="MAT Period",
            yaxis_title=metric.replace('_', ' '),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_growth_chart(data, title, group_by='MANUFACTURER'):
        """Create growth comparison chart"""
        if data.empty or 'GROWTH_PCT' not in data.columns:
            return None
        
        growth_data = data.sort_values('GROWTH_PCT', ascending=False).head(10)
        
        fig = px.bar(
            growth_data,
            x=group_by,
            y='GROWTH_PCT',
            title=title,
            text='GROWTH_PCT',
            color='GROWTH_PCT',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig.update_layout(
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="",
            yaxis_title="Growth (%)",
            xaxis_tickangle=45,
            showlegend=False
        )
        
        return fig

# ==============================================================================
# 6. DASHBOARD COMPONENTS
# ==============================================================================
class DashboardComponents:
    """Reusable dashboard components"""
    
    @staticmethod
    def create_sidebar(df):
        """Create sidebar with filters and info"""
        with st.sidebar:
            # Sidebar header
            st.markdown('<div class="sidebar-header">🔧 Control Panel</div>', unsafe_allow_html=True)
            
            # Period selector
            if 'PERIOD' in df.columns:
                periods = sorted(df['PERIOD'].unique())
                selected_period = st.selectbox(
                    "📅 Select MAT Period",
                    periods,
                    index=len(periods)-1 if periods else 0
                )
            else:
                selected_period = None
            
            # Metric selector
            metric_options = []
            if 'SALES_USD' in df.columns:
                metric_options.append(('💰 Sales (USD)', 'SALES_USD'))
            if 'STANDARD_UNITS' in df.columns:
                metric_options.append(('📦 Volume (Units)', 'STANDARD_UNITS'))
            if 'SU_AVG_PRICE' in df.columns:
                metric_options.append(('💵 Avg Price', 'SU_AVG_PRICE'))
            
            if metric_options:
                selected_metric = st.selectbox(
                    "📊 Select Metric",
                    [opt[1] for opt in metric_options],
                    format_func=lambda x: dict(metric_options)[x],
                    index=0
                )
            else:
                selected_metric = None
            
            st.markdown("---")
            
            # Manufacturer filter
            if 'MANUFACTURER' in df.columns:
                manufacturers = sorted(df['MANUFACTURER'].dropna().unique())
                if len(manufacturers) > 0:
                    selected_mfgs = st.multiselect(
                        "🏭 Filter Manufacturers",
                        manufacturers,
                        default=None
                    )
                else:
                    selected_mfgs = None
            else:
                selected_mfgs = None
            
            # Molecule filter
            if 'MOLECULE' in df.columns:
                molecules = sorted(df['MOLECULE'].dropna().unique())
                if len(molecules) > 0:
                    selected_mols = st.multiselect(
                        "🧪 Filter Molecules",
                        molecules,
                        default=None
                    )
                else:
                    selected_mols = None
            else:
                selected_mols = None
            
            st.markdown("---")
            
            # Dataset info
            DashboardComponents._show_dataset_info(df)
            
            st.markdown("---")
            
            # Export button
            if st.button("📥 Export Data", use_container_width=True):
                DashboardComponents._export_data(df)
            
            # Refresh button
            if st.button("🔄 Refresh Analysis", use_container_width=True):
                st.rerun()
        
        return {
            'period': selected_period,
            'metric': selected_metric,
            'manufacturers': selected_mfgs,
            'molecules': selected_mols
        }
    
    @staticmethod
    def _show_dataset_info(df):
        """Show dataset information"""
        st.markdown("#### 📊 Dataset Info")
        
        info_html = """
        <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 8px; color: white;">
        """
        
        info_html += f'<p style="margin: 0.3rem 0;"><strong>Records:</strong> {len(df):,}</p>'
        
        if 'PERIOD' in df.columns:
            periods = df['PERIOD'].nunique()
            latest_period = df['PERIOD'].max()
            info_html += f'<p style="margin: 0.3rem 0;"><strong>MAT Periods:</strong> {periods}</p>'
            info_html += f'<p style="margin: 0.3rem 0;"><strong>Latest:</strong> {latest_period}</p>'
        
        if 'MANUFACTURER' in df.columns:
            mfg_count = df['MANUFACTURER'].nunique()
            info_html += f'<p style="margin: 0.3rem 0;"><strong>Manufacturers:</strong> {mfg_count}</p>'
        
        if 'MOLECULE' in df.columns:
            mol_count = df['MOLECULE'].nunique()
            info_html += f'<p style="margin: 0.3rem 0;"><strong>Molecules:</strong> {mol_count}</p>'
        
        if 'COUNTRY' in df.columns:
            country_count = df['COUNTRY'].nunique()
            info_html += f'<p style="margin: 0.3rem 0;"><strong>Countries:</strong> {country_count}</p>'
        
        info_html += "</div>"
        
        st.markdown(info_html, unsafe_allow_html=True)
    
    @staticmethod
    def _export_data(df):
        """Export data functionality"""
        st.info("Click the download button below to export your data")
        
        # CSV Export
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"pharma_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Excel Export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Analysis')
        
        st.download_button(
            label="📥 Download Excel",
            data=output.getvalue(),
            file_name=f"pharma_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    @staticmethod
    def create_tabs():
        """Create dashboard tabs"""
        tabs = st.tabs([
            "📊 Dashboard",
            "🏭 Manufacturers",
            "🧪 Molecules",
            "📈 Trends",
            "🌍 Geography",
            "💰 Pricing",
            "📋 Data"
        ])
        return tabs

# ==============================================================================
# 7. MAIN APPLICATION
# ==============================================================================
class PharmaAnalyticsApp:
    """Main application class"""
    
    def __init__(self):
        self.processor = PharmaDataProcessor()
        self.visualizer = PharmaVisualizer()
        self.components = DashboardComponents()
        apply_custom_styling()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'processed' not in st.session_state:
            st.session_state.processed = False
    
    def run(self):
        """Run the main application"""
        self._show_header()
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload Pharmaceutical Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload MAT format pharmaceutical data"
        )
        
        if uploaded_file:
            self._process_file(uploaded_file)
            
            if st.session_state.data is not None:
                df = st.session_state.data
                filters = self.components.create_sidebar(df)
                
                # Apply filters
                filtered_df = self._apply_filters(df, filters)
                
                if not filtered_df.empty:
                    tabs = self.components.create_tabs()
                    self._render_tabs(tabs, filtered_df, filters)
                else:
                    st.warning("No data matches the selected filters.")
        else:
            self._show_welcome()
    
    def _show_header(self):
        """Show application header"""
        st.markdown("""
        <div class="main-header">
            Pharmaceutical Market Intelligence Platform
        </div>
        <div style="text-align: center; color: #718096; margin-bottom: 2rem;">
            Advanced Analytics for MAT Data | Version 3.1.0
        </div>
        """, unsafe_allow_html=True)
    
    def _show_welcome(self):
        """Show welcome screen"""
        st.info("👆 Upload your pharmaceutical data file to begin analysis")
        
        with st.expander("📋 Expected Data Format", expanded=True):
            st.markdown("""
            ### MAT Format Pharmaceutical Data
            
            **Required Columns:**
            - `Manufacturer` - Company name
            - `Molecule` - Active ingredient
            - `Country` - Market country
            
            **MAT Period Columns (Example):**
            ```
            MAT Q3 2022 USD MNF
            MAT Q3 2022 Standard Units
            MAT Q3 2022 SU Avg Price USD MNF
            MAT Q3 2023 USD MNF
            MAT Q3 2023 Standard Units
            ```
            
            **Supported File Formats:**
            - CSV (.csv)
            - Excel (.xlsx, .xls)
            """)
    
    def _process_file(self, uploaded_file):
        """Process uploaded file"""
        if not st.session_state.processed:
            with st.spinner("Processing data..."):
                # Load data
                raw_df = self.processor.load_data(uploaded_file)
                
                if raw_df is not None:
                    # Process MAT data
                    processed_df = self.processor.process_mat_data(raw_df)
                    
                    if processed_df is not None:
                        st.session_state.data = processed_df
                        st.session_state.processed = True
                        
                        # Show success message
                        mat_cols = self.processor.detect_mat_columns(raw_df)
                        periods = processed_df['PERIOD'].nunique() if 'PERIOD' in processed_df.columns else 0
                        
                        st.success(f"""
                        ✅ **Data loaded successfully!**
                        
                        - **MAT columns detected:** {len(mat_cols)}
                        - **Periods processed:** {periods}
                        - **Total records:** {len(processed_df):,}
                        """)
    
    def _apply_filters(self, df, filters):
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Apply period filter
        if filters['period']:
            filtered_df = filtered_df[filtered_df['PERIOD'] == filters['period']]
        
        # Apply manufacturer filter
        if filters['manufacturers'] and len(filters['manufacturers']) > 0:
            filtered_df = filtered_df[filtered_df['MANUFACTURER'].isin(filters['manufacturers'])]
        
        # Apply molecule filter
        if filters['molecules'] and len(filters['molecules']) > 0:
            filtered_df = filtered_df[filtered_df['MOLECULE'].isin(filters['molecules'])]
        
        return filtered_df
    
    def _render_tabs(self, tabs, df, filters):
        """Render all dashboard tabs"""
        
        # Tab 1: Dashboard
        with tabs[0]:
            self._render_dashboard(df, filters)
        
        # Tab 2: Manufacturers
        with tabs[1]:
            self._render_manufacturers(df, filters)
        
        # Tab 3: Molecules
        with tabs[2]:
            self._render_molecules(df, filters)
        
        # Tab 4: Trends
        with tabs[3]:
            self._render_trends(df, filters)
        
        # Tab 5: Geography
        with tabs[4]:
            self._render_geography(df, filters)
        
        # Tab 6: Pricing
        with tabs[5]:
            self._render_pricing(df, filters)
        
        # Tab 7: Data
        with tabs[6]:
            self._render_data_explorer(df)
    
    def _render_dashboard(self, df, filters):
        """Render dashboard tab"""
        st.markdown('<div class="sub-header">📊 Executive Dashboard</div>', unsafe_allow_html=True)
        
        if filters['period']:
            # Key metrics
            self.visualizer.create_dashboard_metrics(df, filters['period'])
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Market share chart
                if filters['period'] and filters['metric']:
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
                # Growth chart
                if filters['period'] and filters['metric']:
                    market_data = self.processor.calculate_market_metrics(
                        df, 'MANUFACTURER', filters['period'], filters['metric']
                    )
                    
                    if not market_data.empty and 'GROWTH_PCT' in market_data.columns:
                        fig = self.visualizer.create_growth_chart(
                            market_data,
                            f"Top Growth - {filters['period']}"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
    
    def _render_manufacturers(self, df, filters):
        """Render manufacturers tab"""
        st.markdown('<div class="sub-header">🏭 Manufacturer Analysis</div>', unsafe_allow_html=True)
        
        if filters['period'] and filters['metric']:
            # Calculate metrics
            manufacturer_data = self.processor.calculate_market_metrics(
                df, 'MANUFACTURER', filters['period'], filters['metric']
            )
            
            if not manufacturer_data.empty:
                # Show top manufacturers
                st.markdown(f"#### Top 20 Manufacturers - {filters['period']}")
                
                display_cols = ['RANK', 'MANUFACTURER', 'CURRENT_VALUE', 'MARKET_SHARE_PCT']
                if 'GROWTH_PCT' in manufacturer_data.columns:
                    display_cols.append('GROWTH_PCT')
                
                display_df = manufacturer_data[display_cols].head(20).copy()
                display_df.columns = ['Rank', 'Manufacturer', 'Value', 'Market Share %']
                if 'GROWTH_PCT' in manufacturer_data.columns:
                    display_df['Growth %'] = manufacturer_data.head(20)['GROWTH_PCT']
                
                # Format values
                display_df['Value'] = display_df['Value'].apply(lambda x: f"{x:,.0f}")
                display_df['Market Share %'] = display_df['Market Share %'].apply(lambda x: f"{x:.1f}%")
                if 'Growth %' in display_df.columns:
                    display_df['Growth %'] = display_df['Growth %'].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
    
    def _render_molecules(self, df, filters):
        """Render molecules tab"""
        st.markdown('<div class="sub-header">🧪 Molecule Analysis</div>', unsafe_allow_html=True)
        
        if filters['period'] and filters['metric']:
            # Calculate metrics
            molecule_data = self.processor.calculate_market_metrics(
                df, 'MOLECULE', filters['period'], filters['metric']
            )
            
            if not molecule_data.empty:
                # Chart
                fig = px.bar(
                    molecule_data.head(15),
                    x='MOLECULE',
                    y='CURRENT_VALUE',
                    title=f"Top Molecules - {filters['period']}",
                    color='MARKET_SHARE_PCT',
                    color_continuous_scale='Plasma'
                )
                
                fig.update_layout(
                    height=500,
                    xaxis_tickangle=45,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_trends(self, df, filters):
        """Render trends tab"""
        st.markdown('<div class="sub-header">📈 Trend Analysis</div>', unsafe_allow_html=True)
        
        if filters['metric']:
            # Overall trend
            fig = self.visualizer.create_trend_chart(df, filters['metric'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Grouped trends
            col1, col2 = st.columns(2)
            
            with col1:
                if 'MANUFACTURER' in df.columns:
                    fig = self.visualizer.create_trend_chart(
                        df, filters['metric'], 'MANUFACTURER'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'MOLECULE' in df.columns:
                    fig = self.visualizer.create_trend_chart(
                        df, filters['metric'], 'MOLECULE'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_geography(self, df, filters):
        """Render geography tab"""
        st.markdown('<div class="sub-header">🌍 Geographic Analysis</div>', unsafe_allow_html=True)
        
        if 'COUNTRY' in df.columns and filters['period'] and filters['metric']:
            # Calculate country metrics
            country_data = self.processor.calculate_market_metrics(
                df, 'COUNTRY', filters['period'], filters['metric']
            )
            
            if not country_data.empty:
                # Top countries chart
                fig = px.bar(
                    country_data.head(15),
                    x='COUNTRY',
                    y='CURRENT_VALUE',
                    title=f"Top Countries - {filters['period']}",
                    color='MARKET_SHARE_PCT',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis_tickangle=45
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_pricing(self, df, filters):
        """Render pricing tab"""
        st.markdown('<div class="sub-header">💰 Price Analysis</div>', unsafe_allow_html=True)
        
        if 'SU_AVG_PRICE' in df.columns:
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
                        labels={'SU_AVG_PRICE': 'Average Price'}
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price by manufacturer
                if filters['period'] and 'MANUFACTURER' in df.columns:
                    period_data = df[df['PERIOD'] == filters['period']]
                    
                    if not period_data.empty:
                        price_by_mfg = period_data.groupby('MANUFACTURER')['SU_AVG_PRICE'].mean().nlargest(10).reset_index()
                        
                        fig = px.bar(
                            price_by_mfg,
                            x='MANUFACTURER',
                            y='SU_AVG_PRICE',
                            title=f"Average Price by Manufacturer - {filters['period']}",
                            color='SU_AVG_PRICE',
                            color_continuous_scale='Viridis'
                        )
                        
                        fig.update_layout(height=400, xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_data_explorer(self, df):
        """Render data explorer tab"""
        st.markdown('<div class="sub-header">📋 Data Explorer</div>', unsafe_allow_html=True)
        
        # Data preview
        st.dataframe(
            df,
            use_container_width=True,
            height=500
        )
        
        # Data statistics
        st.markdown("### 📊 Data Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Types**")
            st.write(df.dtypes)
        
        with col2:
            st.write("**Missing Values**")
            st.write(df.isnull().sum())

# ==============================================================================
# 8. APPLICATION ENTRY POINT
# ==============================================================================
def main():
    """Main application entry point"""
    try:
        # Initialize and run app
        app = PharmaAnalyticsApp()
        app.run()
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>© 2024 Pharma Intelligence Platform | Version 3.1.0</p>
            <p style="font-size: 0.8rem; color: #a0aec0;">
                Confidential - For Internal Use Only
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        st.info("Please refresh the page or contact support.")

# ==============================================================================
# 9. RUN APPLICATION
# ==============================================================================
if __name__ == "__main__":
    main()
