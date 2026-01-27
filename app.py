"""
================================================================================
PHARMACEUTICAL MARKET INTELLIGENCE PLATFORM
MAT Data Analytics Dashboard
Version: 3.0.0 | Python 3.11+ Compatible
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
from plotly.subplots import make_subplots
import warnings
import io
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# ==============================================================================
# 3. CUSTOM CSS STYLING
# ==============================================================================
def apply_custom_styling():
    """Apply custom CSS styling for professional appearance"""
    st.markdown("""
    <style>
    /* ========== GLOBAL STYLES ========== */
    .main {
        padding: 0rem 1rem;
    }
    
    /* ========== HEADER STYLES ========== */
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1.5rem;
        position: relative;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 25%;
        width: 50%;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #4a5568;
        margin: 1.5rem 0 1rem 0;
        padding-left: 0.5rem;
        border-left: 4px solid #3498db;
    }
    
    /* ========== METRIC CARDS ========== */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(180deg, #3498db 0%, #2ecc71 100%);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2d3748;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-change {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .metric-change.positive {
        background-color: rgba(72, 187, 120, 0.1);
        color: #38a169;
    }
    
    .metric-change.negative {
        background-color: rgba(245, 101, 101, 0.1);
        color: #e53e3e;
    }
    
    /* ========== TABS STYLING ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0.5rem 1.8rem;
        font-weight: 600;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #e2e8f0;
        color: #4a5568;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #edf2f7;
        border-color: #cbd5e0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
        border-color: #3498db;
        box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2);
    }
    
    /* ========== BUTTONS ========== */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.8rem;
        transition: all 0.3s ease;
        border: none;
        background: linear-gradient(135deg, #3498db 0%, #2ecc71 100%);
        color: white;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(52, 152, 219, 0.3);
    }
    
    /* ========== SIDEBAR ========== */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
        color: white;
    }
    
    /* ========== FOOTER ========== */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem;
        color: #718096;
        font-size: 0.9rem;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 4. DASHBOARD CONFIGURATION
# ==============================================================================
class DashboardConfig:
    """Dashboard configuration and constants"""
    APP_NAME = "Pharma MAT Intelligence Platform"
    APP_VERSION = "3.0.0"
    APP_DESCRIPTION = "MAT Data Analytics Dashboard"
    MAX_FILE_SIZE_MB = 200

# ==============================================================================
# 5. MAT DATA PROCESSING ENGINE
# ==============================================================================
class MATDataProcessor:
    """MAT format data processing engine"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=True)
    def load_and_process_data(uploaded_file) -> pd.DataFrame:
        """
        Load and process MAT format pharmaceutical data
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error(f"❌ Unsupported file format: {file_extension}")
                return pd.DataFrame()
            
            # Clean column names
            df.columns = MATDataProcessor._clean_column_names(df.columns)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _clean_column_names(columns: pd.Index) -> List[str]:
        """Clean column names"""
        cleaned = []
        for col in columns:
            col_str = str(col).strip()
            col_str = ' '.join(col_str.split())  # Remove multiple spaces
            cleaned.append(col_str)
        return cleaned
    
    @staticmethod
    def extract_mat_periods(df: pd.DataFrame) -> List[str]:
        """Extract MAT periods from column names"""
        mat_pattern = r'MAT Q\d \d{4}'
        mat_columns = []
        
        for col in df.columns:
            if re.search(mat_pattern, str(col)):
                mat_columns.append(col)
        
        return mat_columns
    
    @staticmethod
    def transform_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform wide MAT format to long format for analysis
        """
        # Identify MAT columns
        mat_pattern = r'MAT Q\d \d{4}'
        mat_columns = []
        
        for col in df.columns:
            if re.search(mat_pattern, str(col)):
                mat_columns.append(col)
        
        if not mat_columns:
            return df
        
        # Identify ID columns (non-MAT columns)
        id_columns = [col for col in df.columns if col not in mat_columns]
        
        # Extract unique periods
        periods = []
        for col in mat_columns:
            # Extract period from column name
            match = re.search(r'(MAT Q\d \d{4})', col)
            if match:
                period = match.group(1)
                if period not in periods:
                    periods.append(period)
        
        # Group MAT columns by period and metric type
        period_data = []
        
        for period in periods:
            # Find all columns for this period
            period_cols = [col for col in mat_columns if period in col]
            
            # Create mapping of metric types
            metrics = {}
            for col in period_cols:
                if 'USD MNF' in col:
                    metrics['Sales_USD'] = col
                elif 'Standard Units' in col:
                    metrics['Standard_Units'] = col
                elif 'Units' in col:
                    metrics['Units'] = col
                elif 'SU Avg Price USD MNF' in col:
                    metrics['SU_Avg_Price'] = col
                elif 'Unit Avg Price USD MNF' in col:
                    metrics['Unit_Avg_Price'] = col
            
            # Create period DataFrame
            period_df = df[id_columns].copy()
            period_df['Period'] = period
            
            # Add metrics
            for metric_name, col_name in metrics.items():
                # Handle comma decimal separator
                values = df[col_name].astype(str).str.replace(',', '.').astype(float, errors='coerce')
                period_df[metric_name] = values
            
            period_data.append(period_df)
        
        # Combine all periods
        if period_data:
            long_df = pd.concat(period_data, ignore_index=True)
            
            # Extract year and quarter
            long_df['Year'] = long_df['Period'].str.extract(r'(\d{4})')
            long_df['Quarter'] = long_df['Period'].str.extract(r'Q(\d)')
            long_df['Period_Index'] = long_df['Year'] + 'Q' + long_df['Quarter']
            
            # Sort by period
            long_df = long_df.sort_values('Period_Index')
            
            return long_df
        else:
            return df
    
    @staticmethod
    def calculate_mat_growth(df: pd.DataFrame, metric: str = 'Sales_USD') -> pd.DataFrame:
        """
        Calculate MAT period-over-period growth
        """
        if 'Period_Index' not in df.columns or metric not in df.columns:
            return pd.DataFrame()
        
        # Calculate growth by manufacturer and molecule
        growth_data = []
        
        for group_col in ['Manufacturer', 'Molecule', 'Country']:
            if group_col in df.columns:
                pivot = df.pivot_table(
                    index=group_col,
                    columns='Period',
                    values=metric,
                    aggfunc='sum'
                ).fillna(0)
                
                # Calculate growth between periods
                if len(pivot.columns) >= 2:
                    periods = sorted(pivot.columns)
                    for i in range(1, len(periods)):
                        current = periods[i]
                        previous = periods[i-1]
                        
                        growth = pivot[current] - pivot[previous]
                        growth_pct = (growth / pivot[previous].replace(0, np.nan)) * 100
                        
                        temp_df = pd.DataFrame({
                            group_col: pivot.index,
                            'Period': current,
                            f'{metric}_Current': pivot[current],
                            f'{metric}_Previous': pivot[previous],
                            f'{metric}_Growth': growth,
                            f'{metric}_Growth_Pct': growth_pct
                        })
                        
                        growth_data.append(temp_df)
        
        if growth_data:
            return pd.concat(growth_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def calculate_market_share(df: pd.DataFrame, 
                              group_column: str, 
                              period: str,
                              metric: str = 'Sales_USD') -> pd.DataFrame:
        """
        Calculate market share for specific period
        """
        period_data = df[df['Period'] == period]
        
        if len(period_data) == 0:
            return pd.DataFrame()
        
        # Group and calculate
        grouped = period_data.groupby(group_column)[metric].sum().reset_index()
        total = grouped[metric].sum()
        
        grouped['Market_Share_%'] = (grouped[metric] / total * 100).round(2)
        grouped['Rank'] = grouped[metric].rank(ascending=False, method='dense').astype(int)
        grouped = grouped.sort_values('Rank')
        
        return grouped

# ==============================================================================
# 6. VISUALIZATION ENGINE FOR MAT DATA
# ==============================================================================
class MATVisualizationEngine:
    """Visualization engine for MAT data"""
    
    @staticmethod
    def create_mat_metrics(df: pd.DataFrame, latest_period: str) -> None:
        """
        Create metric cards for MAT dashboard
        """
        latest_data = df[df['Period'] == latest_period]
        
        if len(latest_data) == 0:
            return
        
        metrics = {}
        
        if 'Sales_USD' in latest_data.columns:
            total_sales = latest_data['Sales_USD'].sum()
            metrics['💰 MAT Sales'] = {
                'value': f"${total_sales/1e6:.1f}M",
                'period': latest_period
            }
        
        if 'Standard_Units' in latest_data.columns:
            total_units = latest_data['Standard_Units'].sum()
            metrics['📦 MAT Volume'] = {
                'value': f"{total_units/1e6:.1f}M",
                'period': latest_period
            }
        
        if 'Manufacturer' in latest_data.columns:
            manufacturers = latest_data['Manufacturer'].nunique()
            metrics['🏭 Manufacturers'] = {
                'value': f"{manufacturers:,}",
                'period': latest_period
            }
        
        if 'Molecule' in latest_data.columns:
            molecules = latest_data['Molecule'].nunique()
            metrics['🧪 Molecules'] = {
                'value': f"{molecules:,}",
                'period': latest_period
            }
        
        # Display metrics in columns
        cols = st.columns(len(metrics))
        for idx, (title, data) in enumerate(metrics.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{title}</div>
                    <div class="metric-value">{data['value']}</div>
                    <div style="font-size: 0.8rem; color: #718096; margin-top: 0.5rem;">
                        {data['period']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def create_mat_trend_chart(df: pd.DataFrame, 
                              metric: str = 'Sales_USD',
                              group_by: str = None) -> go.Figure:
        """
        Create MAT trend chart over periods
        """
        if group_by and group_by in df.columns:
            # Grouped trend
            trend_data = df.groupby(['Period', group_by])[metric].sum().reset_index()
            
            fig = px.line(
                trend_data,
                x='Period',
                y=metric,
                color=group_by,
                markers=True,
                title=f"MAT {metric.replace('_', ' ')} Trend by {group_by}",
                labels={metric: metric.replace('_', ' ')}
            )
        else:
            # Overall trend
            trend_data = df.groupby('Period')[metric].sum().reset_index()
            
            fig = px.line(
                trend_data,
                x='Period',
                y=metric,
                markers=True,
                title=f"MAT {metric.replace('_', ' ')} Trend",
                labels={metric: metric.replace('_', ' ')}
            )
        
        fig.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="MAT Period",
            yaxis_title=metric.replace('_', ' '),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_mat_comparison_chart(df: pd.DataFrame,
                                   metric: str = 'Sales_USD',
                                   top_n: int = 10) -> go.Figure:
        """
        Create comparison chart between latest two MAT periods
        """
        periods = sorted(df['Period'].unique())
        
        if len(periods) < 2:
            st.warning("Need at least 2 MAT periods for comparison")
            return None
        
        latest_period = periods[-1]
        previous_period = periods[-2]
        
        # Get top performers in latest period
        latest_data = df[df['Period'] == latest_period]
        top_items = latest_data.groupby('Manufacturer')[metric].sum().nlargest(top_n).index
        
        # Get data for comparison
        comparison_data = df[df['Manufacturer'].isin(top_items)]
        comparison_data = comparison_data[comparison_data['Period'].isin([latest_period, previous_period])]
        
        # Pivot for comparison
        pivot_data = comparison_data.pivot_table(
            index='Manufacturer',
            columns='Period',
            values=metric,
            aggfunc='sum'
        ).fillna(0)
        
        # Calculate growth
        pivot_data['Growth'] = pivot_data[latest_period] - pivot_data[previous_period]
        pivot_data['Growth_%'] = (pivot_data['Growth'] / pivot_data[previous_period].replace(0, np.nan)) * 100
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=pivot_data.index,
            y=pivot_data[previous_period],
            name=previous_period,
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=pivot_data.index,
            y=pivot_data[latest_period],
            name=latest_period,
            marker_color='royalblue'
        ))
        
        fig.update_layout(
            title=f"MAT {metric.replace('_', ' ')} Comparison: {previous_period} vs {latest_period}",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            barmode='group',
            xaxis_title="Manufacturer",
            yaxis_title=metric.replace('_', ' ')
        )
        
        return fig
    
    @staticmethod
    def create_market_share_pie(df: pd.DataFrame,
                               period: str,
                               metric: str = 'Sales_USD',
                               top_n: int = 10) -> go.Figure:
        """
        Create market share pie chart for specific period
        """
        period_data = df[df['Period'] == period]
        
        if len(period_data) == 0:
            return None
        
        # Get top manufacturers
        top_manufacturers = period_data.groupby('Manufacturer')[metric].sum().nlargest(top_n).reset_index()
        
        # Calculate "Others"
        total = period_data[metric].sum()
        top_total = top_manufacturers[metric].sum()
        others_value = total - top_total
        
        if others_value > 0:
            others_df = pd.DataFrame({
                'Manufacturer': ['Others'],
                metric: [others_value]
            })
            top_manufacturers = pd.concat([top_manufacturers, others_df], ignore_index=True)
        
        # Create pie chart
        fig = px.pie(
            top_manufacturers,
            values=metric,
            names='Manufacturer',
            title=f"Market Share Distribution - {period}",
            hole=0.4
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            pull=[0.1] + [0] * (len(top_manufacturers) - 1)
        )
        
        fig.update_layout(height=500)
        
        return fig

# ==============================================================================
# 7. DASHBOARD COMPONENTS
# ==============================================================================
class MATDashboardComponents:
    """Components for MAT dashboard"""
    
    @staticmethod
    def create_sidebar_filters(df: pd.DataFrame) -> Dict:
        """
        Create sidebar filters for MAT data
        """
        filters = {}
        
        with st.sidebar:
            st.markdown("""
            <div style="padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     border-radius: 10px; margin-bottom: 1.5rem;">
                <h3 style="color: white; margin: 0;">🎛️ MAT Control Panel</h3>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Configure MAT Analysis
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Period selection
            if 'Period' in df.columns:
                periods = sorted(df['Period'].unique())
                default_periods = periods[-2:] if len(periods) >= 2 else periods
                
                st.markdown("#### 📅 MAT Periods")
                filters['periods'] = st.multiselect(
                    "Select MAT periods",
                    options=periods,
                    default=default_periods,
                    label_visibility="collapsed"
                )
            
            # Metric selection
            metric_options = []
            if 'Sales_USD' in df.columns:
                metric_options.append(('💰 Sales (USD MNF)', 'Sales_USD'))
            if 'Standard_Units' in df.columns:
                metric_options.append(('📦 Standard Units', 'Standard_Units'))
            if 'Units' in df.columns:
                metric_options.append(('📊 Units', 'Units'))
            
            if metric_options:
                st.markdown("#### 📊 Primary Metric")
                selected_metric = st.radio(
                    "",
                    options=[opt[1] for opt in metric_options],
                    format_func=lambda x: dict(metric_options)[x],
                    label_visibility="collapsed"
                )
                filters['metric'] = selected_metric
            
            st.markdown("---")
            
            # Manufacturer filter
            if 'Manufacturer' in df.columns:
                manufacturers = sorted(df['Manufacturer'].dropna().unique().tolist())
                if manufacturers:
                    st.markdown("#### 🏭 Manufacturers")
                    filters['manufacturers'] = st.multiselect(
                        "",
                        options=manufacturers,
                        default=manufacturers[:10] if len(manufacturers) > 10 else manufacturers,
                        label_visibility="collapsed"
                    )
            
            # Molecule filter
            if 'Molecule' in df.columns:
                molecules = sorted(df['Molecule'].dropna().unique().tolist())
                if molecules:
                    st.markdown("#### 🧪 Molecules")
                    filters['molecules'] = st.multiselect(
                        "",
                        options=molecules,
                        default=molecules[:10] if len(molecules) > 10 else molecules,
                        label_visibility="collapsed"
                    )
            
            # Country filter
            if 'Country' in df.columns:
                countries = sorted(df['Country'].dropna().unique().tolist())
                if countries:
                    st.markdown("#### 🌍 Countries")
                    filters['countries'] = st.multiselect(
                        "",
                        options=countries,
                        default=countries[:5] if len(countries) > 5 else countries,
                        label_visibility="collapsed"
                    )
            
            st.markdown("---")
            
            # Dataset info
            MATDashboardComponents._display_dataset_info(df)
            
            st.markdown("---")
            
            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("📥 Export Data", use_container_width=True):
                    MATDashboardComponents._export_data(df)
        
        return filters
    
    @staticmethod
    def _display_dataset_info(df: pd.DataFrame):
        """Display dataset information"""
        st.markdown("#### 📊 Dataset Info")
        
        info_html = f"""
        <div style="background: rgba(52, 152, 219, 0.1); padding: 1rem; border-radius: 8px;">
            <p style="margin: 0.2rem 0;"><strong>Total Records:</strong> {len(df):,}</p>
            <p style="margin: 0.2rem 0;"><strong>Columns:</strong> {len(df.columns)}</p>
        """
        
        if 'Period' in df.columns:
            periods = df['Period'].nunique()
            info_html += f'<p style="margin: 0.2rem 0;"><strong>MAT Periods:</strong> {periods}</p>'
            period_list = ', '.join(sorted(df['Period'].unique())[-5:])
            info_html += f'<p style="margin: 0.2rem 0; font-size: 0.9em;"><strong>Latest Periods:</strong> {period_list}</p>'
        
        if 'Manufacturer' in df.columns:
            info_html += f'<p style="margin: 0.2rem 0;"><strong>Manufacturers:</strong> {df["Manufacturer"].nunique():,}</p>'
        
        if 'Molecule' in df.columns:
            info_html += f'<p style="margin: 0.2rem 0;"><strong>Molecules:</strong> {df["Molecule"].nunique():,}</p>'
        
        if 'Country' in df.columns:
            info_html += f'<p style="margin: 0.2rem 0;"><strong>Countries:</strong> {df["Country"].nunique():,}</p>'
        
        info_html += "</div>"
        st.markdown(info_html, unsafe_allow_html=True)
    
    @staticmethod
    def _export_data(df: pd.DataFrame):
        """Export data functionality"""
        export_format = st.selectbox(
            "Select export format",
            ["CSV", "Excel", "JSON"]
        )
        
        if export_format == "CSV":
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="mat_pharma_analysis.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='MAT Analysis')
            st.download_button(
                label="📥 Download Excel",
                data=output.getvalue(),
                file_name="mat_pharma_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    @staticmethod
    def create_mat_analysis_tabs() -> List:
        """
        Create MAT analysis tabs
        """
        tab_titles = [
            "📊 MAT Executive Summary",
            "📈 MAT Trend Analysis",
            "🏭 Manufacturer Performance",
            "🧪 Molecule Analysis",
            "🌍 Geographic View",
            "💰 Price Analytics",
            "📋 Data Explorer"
        ]
        
        return st.tabs(tab_titles)

# ==============================================================================
# 8. MAIN DASHBOARD APPLICATION
# ==============================================================================
class MATPharmaAnalyticsDashboard:
    """Main MAT dashboard application"""
    
    def __init__(self):
        self.config = DashboardConfig()
        self.processor = MATDataProcessor()
        self.visualizer = MATVisualizationEngine()
        self.components = MATDashboardComponents()
        self._init_session_state()
        apply_custom_styling()
    
    def _init_session_state(self):
        """Initialize session state"""
        if 'mat_data_loaded' not in st.session_state:
            st.session_state.mat_data_loaded = False
        if 'processed_mat_data' not in st.session_state:
            st.session_state.processed_mat_data = None
    
    def run(self):
        """Run the dashboard"""
        self._display_header()
        uploaded_file = self._display_file_uploader()
        
        if uploaded_file is not None:
            self._process_uploaded_file(uploaded_file)
            
            if st.session_state.mat_data_loaded and st.session_state.processed_mat_data is not None:
                df = st.session_state.processed_mat_data
                filters = self.components.create_sidebar_filters(df)
                filtered_df = self._apply_filters(df, filters)
                
                if len(filtered_df) > 0:
                    tabs = self.components.create_mat_analysis_tabs()
                    self._render_tabs(tabs, filtered_df, filters)
                else:
                    st.warning("⚠️ No data matches the selected filters.")
        else:
            self._display_welcome_screen()
    
    def _display_header(self):
        """Display header"""
        st.markdown(f"""
        <div class="main-header">
            {self.config.APP_NAME}
        </div>
        <div style="text-align: center; color: #718096; margin-bottom: 2rem; font-size: 1.1rem;">
            {self.config.APP_DESCRIPTION} • Version {self.config.APP_VERSION}
        </div>
        """, unsafe_allow_html=True)
    
    def _display_file_uploader(self):
        """Display file uploader"""
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                uploaded_file = st.file_uploader(
                    "📁 Upload MAT Pharmaceutical Data",
                    type=['xlsx', 'xls', 'csv'],
                    help="Upload Excel or CSV files with MAT format data",
                    key="mat_file_uploader"
                )
                
                if uploaded_file:
                    file_size = uploaded_file.size / (1024 * 1024)
                    if file_size > self.config.MAX_FILE_SIZE_MB:
                        st.error(f"❌ File size exceeds {self.config.MAX_FILE_SIZE_MB}MB limit")
                        return None
                
                return uploaded_file
    
    def _display_welcome_screen(self):
        """Display welcome screen"""
        st.info("👆 Please upload a MAT format pharmaceutical data file to begin analysis")
        
        st.markdown("---")
        
        # MAT Data Format Explanation
        with st.expander("📋 MAT Data Format Guide", expanded=True):
            st.markdown("""
            ### MAT (Moving Annual Total) Data Format
            
            **Expected Column Structure:**
            ```
            Source.Name, Country, Sector, Panel, Region, Sub-Region,
            Corporation, Manufacturer, Molecule List, Molecule, Chemical Salt,
            International Product, Specialty Product, NFC123, International Pack,
            International Strength, International Size, International Volume,
            International Prescription,
            
            MAT Period Columns (example):
            "MAT Q3 2022 USD MNF"
            "MAT Q3 2022 Standard Units"
            "MAT Q3 2022 Units"
            "MAT Q3 2022 SU Avg Price USD MNF"
            "MAT Q3 2022 Unit Avg Price USD MNF"
            
            "MAT Q3 2023 USD MNF"
            "MAT Q3 2023 Standard Units"
            ... etc
            ```
            
            **Required MAT Period Metrics:**
            1. USD MNF (Sales value)
            2. Standard Units (Volume)
            3. Units
            4. SU Avg Price USD MNF
            5. Unit Avg Price USD MNF
            
            **Example Data:**
            ```
            Manufacturer: ABBOTT
            Molecule: PENICILLIN G
            Country: ALGERIA
            MAT Q3 2022 USD MNF: 2265
            MAT Q3 2022 Standard Units: 7065
            MAT Q3 2023 USD MNF: 2270
            MAT Q3 2023 Standard Units: 6881
            ```
            """)
        
        st.markdown("---")
        
        # Features showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e2e8f0; height: 100%;">
                <h3 style="color: #2d3748; margin-top: 0;">📊 MAT Trend Analysis</h3>
                <ul style="color: #4a5568;">
                    <li>Quarterly MAT period comparisons</li>
                    <li>Growth rate calculations</li>
                    <li>Trend visualization</li>
                    <li>Period-over-period analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e2e8f0; height: 100%;">
                <h3 style="color: #2d3748; margin-top: 0;">🏭 Manufacturer Intelligence</h3>
                <ul style="color: #4a5568;">
                    <li>MAT market share analysis</li>
                    <li>Growth ranking</li>
                    <li>Competitive benchmarking</li>
                    <li>Performance tracking</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e2e8f0; height: 100%;">
                <h3 style="color: #2d3748; margin-top: 0;">💰 Price Analytics</h3>
                <ul style="color: #4a5568;">
                    <li>Average price trends</li>
                    <li>Price vs volume analysis</li>
                    <li>Market basket analysis</li>
                    <li>Pricing strategy insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def _process_uploaded_file(self, uploaded_file):
        """Process uploaded MAT file"""
        if not st.session_state.mat_data_loaded:
            with st.spinner("🔄 Processing MAT data..."):
                # Load raw data
                raw_data = self.processor.load_and_process_data(uploaded_file)
                
                if not raw_data.empty:
                    # Extract MAT periods info
                    mat_columns = self.processor.extract_mat_periods(raw_data)
                    
                    if mat_columns:
                        st.info(f"✅ Found {len(mat_columns)} MAT period columns")
                        
                        # Transform to long format
                        long_data = self.processor.transform_to_long_format(raw_data)
                        
                        if not long_data.empty:
                            st.session_state.processed_mat_data = long_data
                            st.session_state.mat_data_loaded = True
                            
                            # Display processing summary
                            periods = long_data['Period'].unique()
                            
                            st.success(f"""
                            ✅ **MAT Data Successfully Processed!**
                            
                            - **Total Records:** {len(long_data):,}
                            - **MAT Periods:** {len(periods)}
                            - **Latest Period:** {max(periods) if len(periods) > 0 else 'N/A'}
                            - **Manufacturers:** {long_data['Manufacturer'].nunique() if 'Manufacturer' in long_data.columns else 'N/A'}
                            - **Molecules:** {long_data['Molecule'].nunique() if 'Molecule' in long_data.columns else 'N/A'}
                            """)
                        else:
                            st.error("❌ Failed to transform MAT data to long format")
                    else:
                        st.error("❌ No MAT period columns found in the data")
                        st.info("Please ensure your data contains columns like 'MAT Q3 2022 USD MNF'")
                else:
                    st.error("❌ Failed to load data")
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to MAT data"""
        filtered_df = df.copy()
        
        # Apply period filter
        if 'periods' in filters and filters['periods']:
            filtered_df = filtered_df[filtered_df['Period'].isin(filters['periods'])]
        
        # Apply manufacturer filter
        if 'manufacturers' in filters and filters['manufacturers']:
            filtered_df = filtered_df[filtered_df['Manufacturer'].isin(filters['manufacturers'])]
        
        # Apply molecule filter
        if 'molecules' in filters and filters['molecules']:
            filtered_df = filtered_df[filtered_df['Molecule'].isin(filters['molecules'])]
        
        # Apply country filter
        if 'countries' in filters and filters['countries']:
            filtered_df = filtered_df[filtered_df['Country'].isin(filters['countries'])]
        
        return filtered_df
    
    def _render_tabs(self, tabs, df: pd.DataFrame, filters: Dict):
        """Render MAT analysis tabs"""
        
        # Tab 1: MAT Executive Summary
        with tabs[0]:
            self._render_mat_executive_summary(df, filters)
        
        # Tab 2: MAT Trend Analysis
        with tabs[1]:
            self._render_mat_trend_analysis(df, filters)
        
        # Tab 3: Manufacturer Performance
        with tabs[2]:
            self._render_manufacturer_performance(df, filters)
        
        # Tab 4: Molecule Analysis
        with tabs[3]:
            self._render_molecule_analysis(df, filters)
        
        # Tab 5: Geographic View
        with tabs[4]:
            self._render_geographic_view(df, filters)
        
        # Tab 6: Price Analytics
        with tabs[5]:
            self._render_price_analytics(df, filters)
        
        # Tab 7: Data Explorer
        with tabs[6]:
            self._render_data_explorer(df)
    
    def _render_mat_executive_summary(self, df: pd.DataFrame, filters: Dict):
        """Render MAT executive summary"""
        st.markdown('<div class="sub-header">📊 MAT Executive Dashboard</div>', 
                   unsafe_allow_html=True)
        
        # Get latest period
        periods = sorted(df['Period'].unique())
        latest_period = periods[-1] if len(periods) > 0 else None
        
        if latest_period:
            # Display MAT metrics
            self.visualizer.create_mat_metrics(df, latest_period)
            
            st.markdown("---")
            
            # Top charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Market share pie chart
                if filters.get('metric'):
                    fig = self.visualizer.create_market_share_pie(
                        df,
                        latest_period,
                        filters['metric']
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # MAT comparison chart
                if len(periods) >= 2 and filters.get('metric'):
                    fig = self.visualizer.create_mat_comparison_chart(
                        df,
                        filters['metric']
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_mat_trend_analysis(self, df: pd.DataFrame, filters: Dict):
        """Render MAT trend analysis"""
        st.markdown('<div class="sub-header">📈 MAT Trend Analysis</div>', 
                   unsafe_allow_html=True)
        
        if filters.get('metric'):
            # Overall trend
            fig = self.visualizer.create_mat_trend_chart(
                df,
                filters['metric']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Grouped trends
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Manufacturer' in df.columns:
                    top_manufacturers = df.groupby('Manufacturer')[filters['metric']].sum().nlargest(5).index.tolist()
                    filtered_df = df[df['Manufacturer'].isin(top_manufacturers)]
                    
                    fig = self.visualizer.create_mat_trend_chart(
                        filtered_df,
                        filters['metric'],
                        'Manufacturer'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Molecule' in df.columns:
                    top_molecules = df.groupby('Molecule')[filters['metric']].sum().nlargest(5).index.tolist()
                    filtered_df = df[df['Molecule'].isin(top_molecules)]
                    
                    fig = self.visualizer.create_mat_trend_chart(
                        filtered_df,
                        filters['metric'],
                        'Molecule'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_manufacturer_performance(self, df: pd.DataFrame, filters: Dict):
        """Render manufacturer performance analysis"""
        st.markdown('<div class="sub-header">🏭 Manufacturer Performance Analysis</div>', 
                   unsafe_allow_html=True)
        
        if 'Manufacturer' in df.columns and filters.get('metric'):
            # Calculate market share for latest period
            periods = sorted(df['Period'].unique())
            latest_period = periods[-1] if len(periods) > 0 else None
            
            if latest_period:
                market_share = self.processor.calculate_market_share(
                    df, 'Manufacturer', latest_period, filters['metric']
                )
                
                if not market_share.empty:
                    # Display top manufacturers
                    st.markdown(f"#### 🏆 Top 20 Manufacturers - {latest_period}")
                    
                    display_df = market_share.head(20)[['Rank', 'Manufacturer', 'Market_Share_%', filters['metric']]]
                    display_df.columns = ['Rank', 'Manufacturer', 'Market Share %', 'Value']
                    
                    # Format values
                    display_df['Market Share %'] = display_df['Market Share %'].apply(lambda x: f'{x:.1f}%')
                    display_df['Value'] = display_df['Value'].apply(lambda x: f'{x:,.0f}')
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Growth analysis
                    st.markdown("#### 📈 Growth Analysis")
                    
                    if len(periods) >= 2:
                        growth_data = self.processor.calculate_mat_growth(df, filters['metric'])
                        
                        if not growth_data.empty:
                            manufacturer_growth = growth_data[growth_data['Manufacturer'].notna()]
                            
                            if not manufacturer_growth.empty:
                                latest_growth = manufacturer_growth.sort_values(f'{filters["metric"]}_Growth', ascending=False).head(10)
                                
                                fig = px.bar(
                                    latest_growth,
                                    x='Manufacturer',
                                    y=f'{filters["metric"]}_Growth_Pct',
                                    title=f"Top Growth Manufacturers (Latest Period)",
                                    labels={f'{filters["metric"]}_Growth_Pct': 'Growth %'}
                                )
                                
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
    
    def _render_molecule_analysis(self, df: pd.DataFrame, filters: Dict):
        """Render molecule analysis"""
        st.markdown('<div class="sub-header">🧪 Molecule Performance Analysis</div>', 
                   unsafe_allow_html=True)
        
        if 'Molecule' in df.columns and filters.get('metric'):
            # Calculate market share for latest period
            periods = sorted(df['Period'].unique())
            latest_period = periods[-1] if len(periods) > 0 else None
            
            if latest_period:
                molecule_share = self.processor.calculate_market_share(
                    df, 'Molecule', latest_period, filters['metric']
                )
                
                if not molecule_share.empty:
                    # Top molecules chart
                    top_molecules = molecule_share.head(15)
                    
                    fig = px.bar(
                        top_molecules,
                        x='Molecule',
                        y=filters['metric'],
                        title=f"Top Molecules - {latest_period}",
                        color='Market_Share_%',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(height=500, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_geographic_view(self, df: pd.DataFrame, filters: Dict):
        """Render geographic analysis"""
        st.markdown('<div class="sub-header">🌍 Geographic Market Analysis</div>', 
                   unsafe_allow_html=True)
        
        if 'Country' in df.columns and filters.get('metric'):
            # Country performance
            periods = sorted(df['Period'].unique())
            latest_period = periods[-1] if len(periods) > 0 else None
            
            if latest_period:
                country_data = df[df['Period'] == latest_period]
                
                if len(country_data) > 0:
                    # Top countries
                    top_countries = country_data.groupby('Country')[filters['metric']].sum().nlargest(15).reset_index()
                    
                    fig = px.bar(
                        top_countries,
                        x='Country',
                        y=filters['metric'],
                        title=f"Top Countries - {latest_period}",
                        color=filters['metric'],
                        color_continuous_scale='Plasma'
                    )
                    
                    fig.update_layout(height=500, xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Country growth
                    if len(periods) >= 2:
                        growth_data = self.processor.calculate_mat_growth(df, filters['metric'])
                        country_growth = growth_data[growth_data['Country'].notna()]
                        
                        if not country_growth.empty:
                            latest_country_growth = country_growth.sort_values(f'{filters["metric"]}_Growth', ascending=False).head(10)
                            
                            fig = px.bar(
                                latest_country_growth,
                                x='Country',
                                y=f'{filters["metric"]}_Growth_Pct',
                                title="Top Growing Countries",
                                labels={f'{filters["metric"]}_Growth_Pct': 'Growth %'}
                            )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
    
    def _render_price_analytics(self, df: pd.DataFrame, filters: Dict):
        """Render price analytics"""
        st.markdown('<div class="sub-header">💰 Price & Volume Analytics</div>', 
                   unsafe_allow_html=True)
        
        if 'SU_Avg_Price' in df.columns:
            # Price distribution
            price_data = df[df['SU_Avg_Price'] > 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    price_data,
                    x='SU_Avg_Price',
                    nbins=50,
                    title="Price Distribution",
                    labels={'SU_Avg_Price': 'Average Price per Unit'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price trend over periods
                if 'Period' in df.columns:
                    price_trend = df.groupby('Period')['SU_Avg_Price'].mean().reset_index()
                    
                    fig = px.line(
                        price_trend,
                        x='Period',
                        y='SU_Avg_Price',
                        title="Average Price Trend",
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_data_explorer(self, df: pd.DataFrame):
        """Render data explorer"""
        st.markdown('<div class="sub-header">📋 MAT Data Explorer</div>', 
                   unsafe_allow_html=True)
        
        # Data preview
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
        
        # Export options
        st.markdown("---")
        st.markdown("#### 💾 Export Options")
        self.components._export_data(df)

# ==============================================================================
# 9. APPLICATION ENTRY POINT
# ==============================================================================
def main():
    """Main application entry point"""
    try:
        # Create and run dashboard
        dashboard = MATPharmaAnalyticsDashboard()
        dashboard.run()
        
        # Add footer
        st.markdown("""
        <div class="footer">
            <p>MAT Pharma Intelligence Platform v3.0.0 • © 2024 Pharmaceutical Analytics</p>
            <p style="font-size: 0.8rem; color: #a0aec0;">
                For internal use only • Data confidentiality required
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")

# ==============================================================================
# 10. RUN APPLICATION
# ==============================================================================
if __name__ == "__main__":
    main()
