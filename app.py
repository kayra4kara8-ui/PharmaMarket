"""
================================================================================
PHARMACEUTICAL MARKET INTELLIGENCE PLATFORM
Enterprise-Grade Analytics Dashboard
Version: 3.0.0 | Python 3.11+ Compatible
================================================================================
"""

# ==============================================================================
# IMPORT LIBRARIES
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import math
import sys
import json

warnings.filterwarnings('ignore')

# ==============================================================================
# DASHBOARD CONFIGURATION
# ==============================================================================
class DashboardConfig:
    """Dashboard configuration and constants"""
    
    # Application metadata
    APP_NAME = "Pharma Intelligence Platform"
    APP_VERSION = "3.0.0"
    APP_DESCRIPTION = "Enterprise Pharmaceutical Market Analytics"
    
    # Page configuration
    PAGE_CONFIG = {
        'page_title': 'Pharma Intelligence Platform',
        'page_icon': '💊',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # Color schemes
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ffbb78',
        'info': '#98df8a',
        'light': '#f7f7f7',
        'dark': '#262730'
    }
    
    # Chart color palettes
    COLOR_PALETTES = {
        'sequential': ['Blues', 'Viridis', 'Plasma', 'Inferno', 'Magma'],
        'diverging': ['RdYlGn', 'RdBu', 'PiYG', 'PRGn', 'BrBG'],
        'qualitative': ['Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2']
    }
    
    # Data processing
    CHUNK_SIZE = 50000
    CACHE_DURATION = 3600  # seconds
    
    # Performance settings
    ENABLE_CACHING = True
    MAX_FILE_SIZE_MB = 200

# ==============================================================================
# CUSTOM STYLING
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
    
    .sidebar-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* ========== DATA TABLES ========== */
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .dataframe thead th {
        background-color: #f7fafc;
        font-weight: 700;
        color: #2d3748;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* ========== PROGRESS BARS ========== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%);
    }
    
    /* ========== TOOLTIPS ========== */
    [data-testid="stTooltip"] {
        background-color: #2d3748 !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.8rem !important;
        font-size: 0.9rem !important;
    }
    
    /* ========== BADGES ========== */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 10rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-primary { background-color: #3498db; color: white; }
    .badge-success { background-color: #38a169; color: white; }
    .badge-warning { background-color: #d69e2e; color: white; }
    .badge-danger { background-color: #e53e3e; color: white; }
    .badge-info { background-color: #4299e1; color: white; }
    
    /* ========== ALERTS ========== */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* ========== CHARTS CONTAINER ========== */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
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
# DATA PROCESSING ENGINE
# ==============================================================================
class DataProcessor:
    """Advanced data processing and transformation engine"""
    
    @staticmethod
    @st.cache_data(ttl=DashboardConfig.CACHE_DURATION, show_spinner=True)
    def load_data(uploaded_file, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load pharmaceutical sales data with intelligent format detection
        
        Args:
            uploaded_file: Uploaded file object
            sample_size: Number of rows to sample (for large files)
            
        Returns:
            Processed DataFrame
        """
        try:
            # Detect file type and load
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(
                            uploaded_file, 
                            encoding=encoding,
                            low_memory=False,
                            on_bad_lines='skip'
                        )
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("❌ Unable to read CSV file with standard encodings")
                    return pd.DataFrame()
                    
            elif file_extension in ['xlsx', 'xls']:
                # Read Excel with optimized settings
                df = pd.read_excel(
                    uploaded_file,
                    engine='openpyxl',
                    dtype=str  # Read all as string initially
                )
            else:
                st.error(f"❌ Unsupported file format: {file_extension}")
                return pd.DataFrame()
            
            # Apply sampling if specified
            if sample_size and len(df) > sample_size:
                df = df.sample(min(sample_size, len(df)), random_state=42)
            
            # Clean column names
            df.columns = DataProcessor._clean_column_names(df.columns)
            
            # Auto-detect MAT period columns
            df = DataProcessor._detect_mat_periods(df)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def _clean_column_names(columns: pd.Index) -> List[str]:
        """Clean and standardize column names"""
        cleaned = []
        for col in columns:
            if isinstance(col, str):
                # Remove extra whitespace and special characters
                col = col.strip()
                col = col.replace('\n', ' ').replace('\r', ' ')
                col = ' '.join(col.split())  # Remove multiple spaces
                cleaned.append(col)
            else:
                cleaned.append(str(col))
        return cleaned
    
    @staticmethod
    def _detect_mat_periods(df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect MAT period columns in the dataset"""
        
        # Common MAT period patterns
        mat_patterns = [
            r'MAT Q\d \d{4}',
            r'MAT\d{1}Q\d{4}',
            r'MAT \d{1}Q\d{4}',
            r'MAT_Q\d_\d{4}',
            r'Moving Annual Total Q\d \d{4}'
        ]
        
        # Check for MAT columns
        mat_columns = []
        for col in df.columns:
            if any(pd.Series(col).str.contains(pattern, na=False, regex=True).any() 
                  for pattern in mat_patterns):
                mat_columns.append(col)
        
        if mat_columns:
            # Add metadata about MAT columns
            df.attrs['mat_columns'] = mat_columns
            df.attrs['has_mat_data'] = True
            
            # Extract periods from column names
            periods = []
            for col in mat_columns:
                # Try to extract period information
                match = None
                for pattern in mat_patterns:
                    match = pd.Series(col).str.extract(pattern)
                    if not match.isna().all().all():
                        break
                
                if match is not None and not match.isna().all().all():
                    periods.append(col)
            
            df.attrs['periods'] = sorted(set(periods))
        
        return df
    
    @staticmethod
    def transform_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform wide format data to long format for analysis
        
        Args:
            df: Wide format DataFrame
            
        Returns:
            Long format DataFrame
        """
        try:
            if 'mat_columns' not in df.attrs:
                st.warning("⚠️ No MAT period columns detected. Using original format.")
                return df
            
            mat_columns = df.attrs.get('mat_columns', [])
            
            # Define ID columns (non-MAT columns)
            id_columns = [col for col in df.columns if col not in mat_columns]
            
            # Extract unique periods
            periods = []
            for col in mat_columns:
                # Simple extraction logic
                parts = col.split()
                if len(parts) >= 3:
                    period = f"{parts[0]} {parts[1]} {parts[2]}"
                    if period not in periods:
                        periods.append(period)
            
            if not periods:
                st.warning("⚠️ Could not extract periods from column names")
                return df
            
            # Melt data for each period
            melted_data = []
            
            for period in periods:
                period_df = df[id_columns].copy()
                period_df['Period'] = period
                
                # Add metric columns for this period
                metric_patterns = [
                    f'{period} USD MNF',
                    f'{period} Standard Units',
                    f'{period} Units',
                    f'{period} SU Avg Price USD MNF',
                    f'{period} Unit Avg Price USD MNF'
                ]
                
                for pattern in metric_patterns:
                    # Find matching columns
                    matching_cols = [col for col in mat_columns if pattern in col]
                    if matching_cols:
                        col_name = matching_cols[0]
                        metric_name = pattern.split()[-1] if 'USD' in pattern else 'Units'
                        period_df[f'{metric_name}_{period}'] = pd.to_numeric(
                            df[col_name], errors='coerce'
                        )
                
                melted_data.append(period_df)
            
            # Combine all periods
            result = pd.concat(melted_data, ignore_index=True)
            
            # Clean and validate
            result = DataProcessor._clean_transformed_data(result)
            
            st.success(f"✅ Transformed to long format: {len(result):,} records")
            return result
            
        except Exception as e:
            st.error(f"❌ Error transforming data: {str(e)}")
            return df
    
    @staticmethod
    def _clean_transformed_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate transformed data"""
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fill NaN values
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
        
        # Remove rows where all metrics are zero
        if numeric_cols:
            df = df[df[numeric_cols].sum(axis=1) != 0]
        
        # Add calculated columns
        if 'Period' in df.columns:
            # Extract Year and Quarter
            df['Year'] = df['Period'].str.extract(r'(\d{4})')
            df['Quarter'] = df['Period'].str.extract(r'Q(\d)')
            
            # Create period index for sorting
            df['Period_Index'] = df['Year'] + df['Quarter'].str.zfill(2)
        
        return df
    
    @staticmethod
    def calculate_market_share(df: pd.DataFrame, 
                              group_column: str, 
                              value_column: str) -> pd.DataFrame:
        """
        Calculate market share and growth metrics
        
        Args:
            df: Input DataFrame
            group_column: Column to group by (e.g., 'Manufacturer', 'Molecule')
            value_column: Value column for calculations
            
        Returns:
            DataFrame with market share metrics
        """
        try:
            # Get latest periods
            periods = sorted(df['Period'].unique())
            if len(periods) < 2:
                return pd.DataFrame()
            
            current_period = periods[-1]
            previous_period = periods[-2]
            
            # Calculate current period metrics
            current = (
                df[df['Period'] == current_period]
                .groupby(group_column)[value_column]
                .sum()
                .reset_index()
                .rename(columns={value_column: 'Current_Value'})
            )
            
            # Calculate previous period metrics
            previous = (
                df[df['Period'] == previous_period]
                .groupby(group_column)[value_column]
                .sum()
                .reset_index()
                .rename(columns={value_column: 'Previous_Value'})
            )
            
            # Merge and calculate
            merged = pd.merge(current, previous, on=group_column, how='left').fillna(0)
            
            # Calculate market share
            total_current = merged['Current_Value'].sum()
            merged['Market_Share_%'] = (
                merged['Current_Value'] / total_current * 100
            ).round(2)
            
            # Calculate growth
            merged['Growth_%'] = (
                (merged['Current_Value'] - merged['Previous_Value']) / 
                merged['Previous_Value'].replace(0, np.nan) * 100
            ).round(2).fillna(0)
            
            # Calculate absolute growth
            merged['Abs_Growth'] = (
                merged['Current_Value'] - merged['Previous_Value']
            ).round(2)
            
            # Rank by current value
            merged['Rank'] = merged['Current_Value'].rank(
                ascending=False, method='dense'
            ).astype(int)
            
            # Sort by rank
            merged = merged.sort_values('Rank')
            
            return merged
            
        except Exception as e:
            st.error(f"Error calculating market share: {str(e)}")
            return pd.DataFrame()

# ==============================================================================
# VISUALIZATION ENGINE
# ==============================================================================
class VisualizationEngine:
    """Professional visualization engine with enterprise-grade charts"""
    
    @staticmethod
    def create_dashboard_metrics(df: pd.DataFrame, current_period: str) -> None:
        """
        Create professional metric cards for dashboard
        
        Args:
            df: Processed DataFrame
            current_period: Current period for calculations
        """
        current_data = df[df['Period'] == current_period]
        
        # Calculate metrics
        metrics = {
            '💰 Total Sales': {
                'value': f"${current_data['Sales_USD'].sum()/1e6:.1f}M",
                'change': None,
                'icon': '💰'
            },
            '📦 Total Units': {
                'value': f"{current_data['Standard_Units'].sum()/1e6:.1f}M",
                'change': None,
                'icon': '📦'
            },
            '💵 Avg Price': {
                'value': f"${current_data['SU_Avg_Price'].mean():.2f}",
                'change': None,
                'icon': '💵'
            },
            '🏭 Manufacturers': {
                'value': f"{current_data['Manufacturer'].nunique():,}",
                'change': None,
                'icon': '🏭'
            },
            '🧪 Molecules': {
                'value': f"{current_data['Molecule'].nunique():,}",
                'change': None,
                'icon': '🧪'
            }
        }
        
        # Display metrics in columns
        cols = st.columns(5)
        for idx, (title, data) in enumerate(metrics.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{data['icon']} {title}</div>
                    <div class="metric-value">{data['value']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def create_market_share_chart(data: pd.DataFrame, 
                                 title: str,
                                 category_col: str = 'Manufacturer',
                                 top_n: int = 15) -> go.Figure:
        """
        Create horizontal bar chart for market share analysis
        
        Args:
            data: Market share DataFrame
            title: Chart title
            category_col: Category column name
            top_n: Number of top items to display
            
        Returns:
            Plotly Figure object
        """
        # Get top N items
        top_data = data.head(top_n).copy()
        
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            y=top_data[category_col],
            x=top_data['Market_Share_%'],
            orientation='h',
            marker=dict(
                color=top_data['Market_Share_%'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Market Share %")
            ),
            text=top_data['Market_Share_%'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hoverinfo='text+name',
            hovertext=[
                f"<b>{row[category_col]}</b><br>"
                f"Market Share: {row['Market_Share_%']:.1f}%<br>"
                f"Growth: {row['Growth_%']:+.1f}%<br>"
                f"Rank: #{row['Rank']}"
                for _, row in top_data.iterrows()
            ]
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color='#2d3748'),
                x=0.5,
                xanchor='center'
            ),
            height=max(400, top_n * 25),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=80, l=0, r=0, b=50),
            xaxis=dict(
                title='Market Share (%)',
                gridcolor='#e2e8f0',
                showgrid=True,
                zeroline=False,
                tickformat='.1f'
            ),
            yaxis=dict(
                title='',
                autorange='reversed',
                gridcolor='#e2e8f0',
                showgrid=False
            ),
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        return fig
    
    @staticmethod
    def create_growth_heatmap(growth_matrix: np.ndarray,
                             x_categories: List[str],
                             y_categories: List[str],
                             title: str = "Growth Heatmap") -> go.Figure:
        """
        Create growth heatmap visualization
        
        Args:
            growth_matrix: 2D array of growth percentages
            x_categories: X-axis categories
            y_categories: Y-axis categories
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=growth_matrix,
            x=x_categories,
            y=y_categories,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(growth_matrix, 1),
            texttemplate='%{text}%',
            textfont=dict(size=10),
            colorbar=dict(
                title="Growth %",
                titleside="right",
                tickformat='.1f'
            ),
            hovertemplate=(
                "<b>%{y} × %{x}</b><br>" +
                "Growth: %{z:.1f}%<br>" +
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color='#2d3748'),
                x=0.5,
                xanchor='center'
            ),
            height=600,
            xaxis=dict(
                title="Molecules",
                tickangle=45,
                gridcolor='#e2e8f0'
            ),
            yaxis=dict(
                title="Manufacturers",
                gridcolor='#e2e8f0'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=80, l=100, r=50, b=100)
        )
        
        return fig
    
    @staticmethod
    def create_geographic_map(df: pd.DataFrame, 
                             metric: str = 'Sales_USD',
                             title: str = "Global Market Distribution") -> go.Figure:
        """
        Create geographic choropleth map
        
        Args:
            df: Processed DataFrame
            metric: Metric to visualize
            title: Chart title
            
        Returns:
            Plotly Figure object
        """
        # Country to ISO code mapping
        country_to_iso = {
            'USA': 'USA', 'UNITED STATES': 'USA',
            'UK': 'GBR', 'UNITED KINGDOM': 'GBR',
            'GERMANY': 'DEU', 'FRANCE': 'FRA', 'ITALY': 'ITA',
            'SPAIN': 'ESP', 'NETHERLANDS': 'NLD',
            'BELGIUM': 'BEL', 'SWITZERLAND': 'CHE',
            'AUSTRIA': 'AUT', 'PORTUGAL': 'PRT',
            'GREECE': 'GRC', 'SWEDEN': 'SWE',
            'NORWAY': 'NOR', 'DENMARK': 'DNK',
            'FINLAND': 'FIN', 'IRELAND': 'IRL',
            # Add more mappings as needed
        }
        
        # Aggregate data by country
        country_data = df.groupby('Country').agg({
            metric: 'sum',
            'Manufacturer': 'nunique',
            'Molecule': 'nunique'
        }).reset_index()
        
        # Map country names to ISO codes
        country_data['ISO'] = country_data['Country'].map(country_to_iso)
        country_data = country_data.dropna(subset=['ISO'])
        
        # Create map
        fig = go.Figure(data=go.Choropleth(
            locations=country_data['ISO'],
            z=country_data[metric],
            text=country_data['Country'],
            colorscale='Blues',
            autocolorscale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar=dict(
                title=f"{metric.replace('_', ' ')}",
                titleside='right'
            ),
            hovertemplate=(
                "<b>%{text}</b><br>" +
                f"{metric.replace('_', ' ')}: %{{z:,.0f}}<br>" +
                "Manufacturers: %{customdata[0]:,}<br>" +
                "Molecules: %{customdata[1]:,}<br>" +
                "<extra></extra>"
            ),
            customdata=country_data[['Manufacturer', 'Molecule']].values
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, color='#2d3748'),
                x=0.5,
                xanchor='center'
            ),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)',
                lakecolor='rgb(255, 255, 255)',
                showocean=True,
                oceancolor='rgb(230, 242, 255)'
            ),
            height=700,
            margin=dict(t=80, l=0, r=0, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_trend_chart(df: pd.DataFrame, 
                          metric: str = 'Sales_USD',
                          group_column: str = None) -> go.Figure:
        """
        Create time series trend chart
        
        Args:
            df: Processed DataFrame
            metric: Metric to visualize
            group_column: Column to group by (optional)
            
        Returns:
            Plotly Figure object
        """
        if group_column:
            # Grouped trend
            trend_data = df.groupby(['Period', group_column])[metric].sum().reset_index()
            
            fig = px.line(
                trend_data,
                x='Period',
                y=metric,
                color=group_column,
                markers=True,
                title=f"{metric.replace('_', ' ')} Trend by {group_column}",
                labels={metric: metric.replace('_', ' '), 'Period': 'Period'}
            )
        else:
            # Overall trend
            trend_data = df.groupby('Period')[metric].sum().reset_index()
            
            fig = px.line(
                trend_data,
                x='Period',
                y=metric,
                markers=True,
                title=f"{metric.replace('_', ' ')} Trend Over Time",
                labels={metric: metric.replace('_', ' '), 'Period': 'Period'}
            )
        
        # Update layout
        fig.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            hovermode='x unified',
            xaxis=dict(
                gridcolor='#e2e8f0',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='#e2e8f0',
                showgrid=True
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#e2e8f0',
                borderwidth=1
            )
        )
        
        return fig

# ==============================================================================
# DASHBOARD COMPONENTS
# ==============================================================================
class DashboardComponents:
    """Reusable dashboard components and widgets"""
    
    @staticmethod
    def create_sidebar_filters(df: pd.DataFrame) -> Dict:
        """
        Create interactive sidebar filters
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary of filter values
        """
        filters = {}
        
        with st.sidebar:
            # Sidebar header
            st.markdown("""
            <div style="padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     border-radius: 10px; margin-bottom: 1.5rem;">
                <h3 style="color: white; margin: 0;">🎛️ Control Panel</h3>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                    Configure your analysis
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metric selection
            st.markdown("#### 📊 Primary Metric")
            filters['metric'] = st.radio(
                "",
                options=['Sales_USD', 'Standard_Units'],
                format_func=lambda x: '💰 Sales (USD)' if x == 'Sales_USD' else '📦 Volume (Units)',
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Period selection
            if 'Period' in df.columns:
                periods = sorted(df['Period'].unique())
                default_periods = periods[-4:] if len(periods) >= 4 else periods
                
                st.markdown("#### 📅 Time Period")
                filters['periods'] = st.multiselect(
                    "Select periods to analyze",
                    options=periods,
                    default=default_periods,
                    label_visibility="collapsed"
                )
            
            # Region filter
            if 'Region' in df.columns:
                regions = ['All'] + sorted(df['Region'].dropna().unique().tolist())
                st.markdown("#### 🌍 Region")
                filters['region'] = st.selectbox(
                    "",
                    options=regions,
                    label_visibility="collapsed"
                )
            
            # Country filter
            if 'Country' in df.columns:
                if filters.get('region') != 'All':
                    countries = ['All'] + sorted(
                        df[df['Region'] == filters['region']]['Country']
                        .dropna().unique().tolist()
                    )
                else:
                    countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
                
                st.markdown("#### 🏴 Country")
                filters['country'] = st.selectbox(
                    "",
                    options=countries,
                    label_visibility="collapsed"
                )
            
            # Manufacturer filter
            if 'Manufacturer' in df.columns:
                manufacturers = sorted(df['Manufacturer'].dropna().unique().tolist())
                st.markdown("#### 🏭 Manufacturers")
                filters['manufacturers'] = st.multiselect(
                    "Select manufacturers (empty for all)",
                    options=manufacturers,
                    default=[],
                    label_visibility="collapsed"
                )
            
            # Molecule filter
            if 'Molecule' in df.columns:
                molecules = sorted(df['Molecule'].dropna().unique().tolist())
                st.markdown("#### 🧪 Molecules")
                filters['molecules'] = st.multiselect(
                    "Select molecules (empty for all)",
                    options=molecules,
                    default=[],
                    label_visibility="collapsed"
                )
            
            st.markdown("---")
            
            # Dataset information
            DashboardComponents._display_dataset_info(df)
            
            # Analysis settings
            with st.expander("⚙️ Advanced Settings", expanded=False):
                st.checkbox("Enable advanced analytics", value=True)
                st.slider("Confidence level", 80, 99, 95)
                st.selectbox("Chart theme", ["Light", "Dark", "Corporate"])
            
            # Quick actions
            st.markdown("---")
            st.markdown("#### ⚡ Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("📥 Export", use_container_width=True):
                    DashboardComponents._export_data(df)
        
        return filters
    
    @staticmethod
    def _display_dataset_info(df: pd.DataFrame):
        """Display dataset information in sidebar"""
        st.markdown("#### 📊 Dataset Info")
        
        info_metrics = [
            ("Total Records", f"{len(df):,}"),
            ("Columns", len(df.columns)),
            ("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        ]
        
        if 'Period' in df.columns:
            info_metrics.append(("Time Periods", df['Period'].nunique()))
        
        if 'Country' in df.columns:
            info_metrics.append(("Countries", df['Country'].nunique()))
        
        if 'Manufacturer' in df.columns:
            info_metrics.append(("Manufacturers", df['Manufacturer'].nunique()))
        
        if 'Molecule' in df.columns:
            info_metrics.append(("Molecules", df['Molecule'].nunique()))
        
        # Create metric cards
        for label, value in info_metrics:
            st.markdown(f"""
            <div style="background: rgba(52, 152, 219, 0.1); padding: 0.8rem; 
                     border-radius: 8px; margin: 0.3rem 0; border-left: 3px solid #3498db;">
                <div style="font-size: 0.8rem; color: #4a5568; font-weight: 600;">{label}</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #2d3748;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
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
                file_name="pharma_analysis.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Analysis')
            st.download_button(
                label="📥 Download Excel",
                data=output.getvalue(),
                file_name="pharma_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif export_format == "JSON":
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="pharma_analysis.json",
                mime="application/json"
            )
    
    @staticmethod
    def create_analysis_tabs() -> List:
        """
        Create dashboard tabs
        
        Returns:
            List of tab objects
        """
        tab_titles = [
            "📊 Executive Summary",
            "🌍 Geographic Insights",
            "🏭 Manufacturer Analysis",
            "🧪 Product Portfolio",
            "📈 Trend Analytics",
            "💰 Price Intelligence",
            "🔬 Advanced Metrics",
            "📋 Data Explorer"
        ]
        
        return st.tabs(tab_titles)

# ==============================================================================
# MAIN DASHBOARD APPLICATION
# ==============================================================================
class PharmaAnalyticsDashboard:
    """Main dashboard application class"""
    
    def __init__(self):
        self.config = DashboardConfig()
        self.processor = DataProcessor()
        self.visualizer = VisualizationEngine()
        self.components = DashboardComponents()
        
        # Initialize session state
        self._init_session_state()
        
        # Apply styling
        apply_custom_styling()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
    
    def run(self):
        """Run the dashboard application"""
        # Apply page configuration
        st.set_page_config(**self.config.PAGE_CONFIG)
        
        # Display application header
        self._display_header()
        
        # File upload section
        uploaded_file = self._display_file_uploader()
        
        if uploaded_file is not None:
            # Process uploaded file
            self._process_uploaded_file(uploaded_file)
            
            if st.session_state.data_loaded and st.session_state.processed_data is not None:
                # Get processed data
                df = st.session_state.processed_data
                
                # Create sidebar filters
                filters = self.components.create_sidebar_filters(df)
                
                # Apply filters to data
                filtered_df = self._apply_filters(df, filters)
                
                if len(filtered_df) > 0:
                    # Create analysis tabs
                    tabs = self.components.create_analysis_tabs()
                    
                    # Render each tab
                    self._render_tabs(tabs, filtered_df, filters)
                else:
                    st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")
        else:
            # Display welcome screen
            self._display_welcome_screen()
    
    def _display_header(self):
        """Display the main dashboard header"""
        st.markdown(f"""
        <div class="main-header">
            {self.config.APP_NAME}
        </div>
        <div style="text-align: center; color: #718096; margin-bottom: 2rem; font-size: 1.1rem;">
            {self.config.APP_DESCRIPTION} • Version {self.config.APP_VERSION}
        </div>
        """, unsafe_allow_html=True)
    
    def _display_file_uploader(self):
        """Display file uploader component"""
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                uploaded_file = st.file_uploader(
                    "📁 Upload Pharmaceutical Data",
                    type=['xlsx', 'xls', 'csv'],
                    help="Upload Excel or CSV files with sales data",
                    key="file_uploader"
                )
                
                if uploaded_file:
                    file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
                    if file_size > self.config.MAX_FILE_SIZE_MB:
                        st.error(f"❌ File size exceeds {self.config.MAX_FILE_SIZE_MB}MB limit")
                        return None
                
                return uploaded_file
    
    def _display_welcome_screen(self):
        """Display welcome screen when no data is uploaded"""
        st.info("👆 Please upload a pharmaceutical sales data file to begin analysis")
        
        st.markdown("---")
        
        # Features showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e2e8f0; height: 100%;">
                <h3 style="color: #2d3748; margin-top: 0;">🌍 Geographic Intelligence</h3>
                <ul style="color: #4a5568;">
                    <li>Interactive world map visualization</li>
                    <li>Regional market share analysis</li>
                    <li>Country-level performance metrics</li>
                    <li>Market penetration insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e2e8f0; height: 100%;">
                <h3 style="color: #2d3748; margin-top: 0;">📈 Market Analytics</h3>
                <ul style="color: #4a5568;">
                    <li>Manufacturer ranking & benchmarking</li>
                    <li>Product portfolio analysis</li>
                    <li>Growth trend identification</li>
                    <li>Competitive landscape mapping</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e2e8f0; height: 100%;">
                <h3 style="color: #2d3748; margin-top: 0;">💰 Price Intelligence</h3>
                <ul style="color: #4a5568;">
                    <li>Price positioning analysis</li>
                    <li>Volume vs price correlation</li>
                    <li>Market basket analysis</li>
                    <li>Price trend forecasting</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data format guidance
        with st.expander("📋 Data Format Requirements", expanded=True):
            st.markdown("""
            ### Expected Data Structure
            
            **Required Columns:**
            - `Manufacturer`: Company name
            - `Molecule`: Active ingredient
            - `Country`: Market country
            - `Region`: Geographic region
            
            **MAT Period Columns (example formats):**
            ```
            MAT Q4 2023 USD MNF
            MAT Q4 2023 Standard Units
            MAT Q3 2023 USD MNF
            MAT Q3 2023 Standard Units
            ```
            
            **Optional Columns:**
            - `Corporation`: Parent company
            - `International Product`: Product name
            - `Chemical Salt`: Chemical formulation
            - `Sector`: Market sector
            - `Panel`: Market panel
            """)
    
    def _process_uploaded_file(self, uploaded_file):
        """Process uploaded file and load data"""
        if not st.session_state.data_loaded:
            with st.spinner("🔄 Processing data... This may take a moment for large files."):
                # Load raw data
                raw_data = self.processor.load_data(uploaded_file)
                
                if not raw_data.empty:
                    # Transform to long format
                    processed_data = self.processor.transform_to_long_format(raw_data)
                    
                    if not processed_data.empty:
                        st.session_state.processed_data = processed_data
                        st.session_state.data_loaded = True
                        
                        # Display success message
                        st.success(f"""
                        ✅ **Data Successfully Loaded!**
                        
                        - **Records:** {len(processed_data):,}
                        - **Columns:** {len(processed_data.columns)}
                        - **Periods:** {processed_data['Period'].nunique() if 'Period' in processed_data.columns else 'N/A'}
                        """)
                    else:
                        st.error("❌ Failed to process data")
                else:
                    st.error("❌ Failed to load data")
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to the dataframe"""
        filtered_df = df.copy()
        
        # Apply period filter
        if 'periods' in filters and filters['periods']:
            filtered_df = filtered_df[filtered_df['Period'].isin(filters['periods'])]
        
        # Apply region filter
        if 'region' in filters and filters['region'] != 'All':
            filtered_df = filtered_df[filtered_df['Region'] == filters['region']]
        
        # Apply country filter
        if 'country' in filters and filters['country'] != 'All':
            filtered_df = filtered_df[filtered_df['Country'] == filters['country']]
        
        # Apply manufacturer filter
        if 'manufacturers' in filters and filters['manufacturers']:
            filtered_df = filtered_df[filtered_df['Manufacturer'].isin(filters['manufacturers'])]
        
        # Apply molecule filter
        if 'molecules' in filters and filters['molecules']:
            filtered_df = filtered_df[filtered_df['Molecule'].isin(filters['molecules'])]
        
        return filtered_df
    
    def _render_tabs(self, tabs, df: pd.DataFrame, filters: Dict):
        """Render content for each tab"""
        
        # Tab 1: Executive Summary
        with tabs[0]:
            self._render_executive_summary(df, filters)
        
        # Tab 2: Geographic Insights
        with tabs[1]:
            self._render_geographic_insights(df, filters)
        
        # Tab 3: Manufacturer Analysis
        with tabs[2]:
            self._render_manufacturer_analysis(df, filters)
        
        # Tab 4: Product Portfolio
        with tabs[3]:
            self._render_product_portfolio(df, filters)
        
        # Tab 5: Trend Analytics
        with tabs[4]:
            self._render_trend_analytics(df, filters)
        
        # Tab 6: Price Intelligence
        with tabs[5]:
            self._render_price_intelligence(df, filters)
        
        # Tab 7: Advanced Metrics
        with tabs[6]:
            self._render_advanced_metrics(df, filters)
        
        # Tab 8: Data Explorer
        with tabs[7]:
            self._render_data_explorer(df)
    
    def _render_executive_summary(self, df: pd.DataFrame, filters: Dict):
        """Render executive summary tab"""
        st.markdown('<div class="sub-header">📊 Executive Dashboard</div>', 
                   unsafe_allow_html=True)
        
        # Get current period
        current_period = sorted(df['Period'].unique())[-1] if 'Period' in df.columns else 'N/A'
        
        # Display metrics
        self.visualizer.create_dashboard_metrics(df, current_period)
        
        st.markdown("---")
        
        # Top charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Top manufacturers
            if 'Manufacturer' in df.columns and filters.get('metric'):
                manufacturer_metrics = self.processor.calculate_market_share(
                    df, 'Manufacturer', filters['metric']
                )
                
                if not manufacturer_metrics.empty:
                    fig = self.visualizer.create_market_share_chart(
                        manufacturer_metrics,
                        "🏆 Top Manufacturers by Market Share"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Market distribution
            if 'Country' in df.columns:
                country_dist = df.groupby('Country')[filters.get('metric', 'Sales_USD')].sum().reset_index()
                country_dist = country_dist.sort_values(filters.get('metric', 'Sales_USD'), ascending=False).head(10)
                
                fig = px.pie(
                    country_dist,
                    values=filters.get('metric', 'Sales_USD'),
                    names='Country',
                    hole=0.4,
                    title="🌍 Top 10 Countries Distribution"
                )
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    pull=[0.1] + [0] * 9
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Growth heatmap
        st.markdown('<div class="sub-header">🔥 Market Growth Matrix</div>', 
                   unsafe_allow_html=True)
        
        # This is a simplified version - implement full heatmap logic as needed
        st.info("Market growth matrix visualization requires specific data structure.")
    
    def _render_geographic_insights(self, df: pd.DataFrame, filters: Dict):
        """Render geographic insights tab"""
        st.markdown('<div class="sub-header">🌍 Geographic Market Intelligence</div>', 
                   unsafe_allow_html=True)
        
        # World map
        if 'Country' in df.columns:
            fig = self.visualizer.create_geographic_map(
                df, 
                filters.get('metric', 'Sales_USD')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Region' in df.columns:
                region_data = df.groupby('Region')[filters.get('metric', 'Sales_USD')].sum().reset_index()
                region_data = region_data.sort_values(filters.get('metric', 'Sales_USD'), ascending=False)
                
                fig = px.bar(
                    region_data,
                    x='Region',
                    y=filters.get('metric', 'Sales_USD'),
                    color=filters.get('metric', 'Sales_USD'),
                    color_continuous_scale='Viridis',
                    title="📊 Regional Performance"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Country' in df.columns:
                country_data = df.groupby('Country')[filters.get('metric', 'Sales_USD')].sum().reset_index()
                top_countries = country_data.nlargest(15, filters.get('metric', 'Sales_USD'))
                
                fig = px.bar(
                    top_countries,
                    x=filters.get('metric', 'Sales_USD'),
                    y='Country',
                    orientation='h',
                    color=filters.get('metric', 'Sales_USD'),
                    color_continuous_scale='Teal',
                    title="🏆 Top 15 Countries"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_manufacturer_analysis(self, df: pd.DataFrame, filters: Dict):
        """Render manufacturer analysis tab"""
        st.markdown('<div class="sub-header">🏭 Manufacturer Intelligence</div>', 
                   unsafe_allow_html=True)
        
        if 'Manufacturer' in df.columns and filters.get('metric'):
            # Calculate manufacturer metrics
            manufacturer_metrics = self.processor.calculate_market_share(
                df, 'Manufacturer', filters['metric']
            )
            
            if not manufacturer_metrics.empty:
                # Market leaders visualization
                fig = px.scatter(
                    manufacturer_metrics.head(20),
                    x='Market_Share_%',
                    y='Growth_%',
                    size='Current_Value',
                    color='Growth_%',
                    hover_name='Manufacturer',
                    color_continuous_scale='RdYlGn',
                    size_max=60,
                    title="🎯 Market Leaders Analysis"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Rankings table
                st.markdown('<div class="sub-header">📋 Manufacturer Rankings</div>', 
                           unsafe_allow_html=True)
                
                display_df = manufacturer_metrics[['Rank', 'Manufacturer', 'Market_Share_%', 'Growth_%', 'Current_Value']].head(20)
                display_df.columns = ['Rank', 'Manufacturer', 'Market Share %', 'Growth %', 'Current Value']
                
                # Format the DataFrame
                formatted_df = display_df.copy()
                formatted_df['Market Share %'] = formatted_df['Market Share %'].apply(lambda x: f'{x:.1f}%')
                formatted_df['Growth %'] = formatted_df['Growth %'].apply(lambda x: f'{x:+.1f}%')
                formatted_df['Current Value'] = formatted_df['Current Value'].apply(lambda x: f'{x:,.0f}')
                
                st.dataframe(
                    formatted_df,
                    use_container_width=True,
                    height=600
                )
    
    def _render_product_portfolio(self, df: pd.DataFrame, filters: Dict):
        """Render product portfolio tab"""
        st.markdown('<div class="sub-header">🧪 Product Portfolio Analysis</div>', 
                   unsafe_allow_html=True)
        
        if 'Molecule' in df.columns and filters.get('metric'):
            # Calculate molecule metrics
            molecule_metrics = self.processor.calculate_market_share(
                df, 'Molecule', filters['metric']
            )
            
            if not molecule_metrics.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top molecules treemap
                    fig = px.treemap(
                        molecule_metrics.head(15),
                        path=['Molecule'],
                        values='Current_Value',
                        color='Growth_%',
                        color_continuous_scale='RdYlGn',
                        title="🔬 Top Molecules Market View"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Growth vs Market Share
                    fig = px.scatter(
                        molecule_metrics.head(20),
                        x='Market_Share_%',
                        y='Growth_%',
                        size='Current_Value',
                        color='Growth_%',
                        hover_name='Molecule',
                        color_continuous_scale='RdYlGn',
                        title="📈 Growth vs Market Share"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_trend_analytics(self, df: pd.DataFrame, filters: Dict):
        """Render trend analytics tab"""
        st.markdown('<div class="sub-header">📈 Market Trend Analytics</div>', 
                   unsafe_allow_html=True)
        
        if 'Period' in df.columns and filters.get('metric'):
            # Overall trend
            fig = self.visualizer.create_trend_chart(
                df, 
                filters['metric']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Grouped trends
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Manufacturer' in df.columns:
                    fig = self.visualizer.create_trend_chart(
                        df, 
                        filters['metric'],
                        'Manufacturer'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Molecule' in df.columns:
                    fig = self.visualizer.create_trend_chart(
                        df, 
                        filters['metric'],
                        'Molecule'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_price_intelligence(self, df: pd.DataFrame, filters: Dict):
        """Render price intelligence tab"""
        st.markdown('<div class="sub-header">💰 Price & Volume Analytics</div>', 
                   unsafe_allow_html=True)
        
        if 'SU_Avg_Price' in df.columns and 'Standard_Units' in df.columns:
            # Price distribution
            price_data = df[df['SU_Avg_Price'] > 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    price_data,
                    x='SU_Avg_Price',
                    nbins=50,
                    title="📊 Price Distribution",
                    labels={'SU_Avg_Price': 'Price per Unit (USD)'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price statistics
                st.markdown("#### 💵 Price Statistics")
                
                stats = price_data['SU_Avg_Price'].describe()
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25%', '75%'],
                    'Value': [
                        f"${stats['mean']:.2f}",
                        f"${stats['50%']:.2f}",
                        f"${stats['std']:.2f}",
                        f"${stats['min']:.2f}",
                        f"${stats['max']:.2f}",
                        f"${stats['25%']:.2f}",
                        f"${stats['75%']:.2f}"
                    ]
                })
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    def _render_advanced_metrics(self, df: pd.DataFrame, filters: Dict):
        """Render advanced metrics tab"""
        st.markdown('<div class="sub-header">🔬 Advanced Market Metrics</div>', 
                   unsafe_allow_html=True)
        
        if 'Manufacturer' in df.columns and filters.get('metric'):
            # Calculate concentration metrics
            manufacturer_metrics = self.processor.calculate_market_share(
                df, 'Manufacturer', filters['metric']
            )
            
            if not manufacturer_metrics.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # HHI Index
                    market_shares = manufacturer_metrics['Market_Share_%'] / 100
                    hhi = (market_shares ** 2).sum() * 10000
                    
                    st.metric(
                        "Herfindahl-Hirschman Index",
                        f"{hhi:.0f}",
                        help="HHI < 1500: Competitive\n1500-2500: Moderate\n>2500: Concentrated"
                    )
                
                with col2:
                    # CR4
                    cr4 = manufacturer_metrics.head(4)['Market_Share_%'].sum()
                    st.metric(
                        "CR4 (Top 4 Concentration)",
                        f"{cr4:.1f}%"
                    )
                
                with col3:
                    # Effective competitors
                    n_effective = 1 / (market_shares ** 2).sum() if (market_shares ** 2).sum() > 0 else 0
                    st.metric(
                        "Effective Competitors",
                        f"{n_effective:.1f}"
                    )
    
    def _render_data_explorer(self, df: pd.DataFrame):
        """Render data explorer tab"""
        st.markdown('<div class="sub-header">📋 Data Explorer</div>', 
                   unsafe_allow_html=True)
        
        # Interactive data table
        st.dataframe(
            df,
            use_container_width=True,
            height=600
        )
        
        # Data export
        st.markdown("---")
        st.markdown("#### 💾 Export Options")
        
        self.components._export_data(df)

# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================
def main():
    """Main application entry point"""
    try:
        # ✅ SET PAGE CONFIG HERE - AT THE VERY BEGINNING
        st.set_page_config(
            page_title='Pharma Intelligence Platform',
            page_icon='💊',
            layout='wide',
            initial_sidebar_state='expanded'
        )
        
        # Apply custom styling (page config'tan sonra)
        apply_custom_styling()
        
        # Create and run dashboard
        dashboard = PharmaAnalyticsDashboard()
        dashboard.run()
        
        # Add footer
        st.markdown("""
        <div class="footer">
            <p>Pharma Intelligence Platform v3.0.0 • © 2024 Pharmaceutical Analytics</p>
            <p style="font-size: 0.8rem; color: #a0aec0;">
                For internal use only • Data confidentiality required
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")

# ==============================================================================
# RUN APPLICATION
# ==============================================================================
if __name__ == "__main__":
    main()


