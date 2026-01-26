"""
Pharmaceutical Sales Analytics Dashboard
Version: 2.1.0
Author: Data Science Team
Description: Enterprise-grade analytics platform for pharmaceutical market intelligence
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import io
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class DashboardConfig:
    """Dashboard configuration and constants"""
    
    # Page config
    PAGE_TITLE = "💊 Pharma Intelligence Platform"
    PAGE_ICON = "💊"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # Colors and themes
    PRIMARY_COLOR = "#1f77b4"
    SECONDARY_COLOR = "#ff7f0e"
    SUCCESS_COLOR = "#2ca02c"
    WARNING_COLOR = "#d62728"
    NEUTRAL_COLOR = "#7f7f7f"
    
    COLOR_SCALES = {
        'sequential': ['Blues', 'Viridis', 'Plasma', 'Inferno', 'Magma'],
        'diverging': ['RdYlGn', 'RdBu', 'PiYG', 'PRGn', 'BrBG'],
        'qualitative': ['Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2']
    }
    
    # Data processing
    CHUNK_SIZE = 100000
    CACHE_TTL = 3600  # seconds

# ============================================================================
# CUSTOM CSS
# ============================================================================
def load_custom_css():
    """Load custom CSS for enhanced styling"""
    st.markdown("""
    <style>
        /* Main container */
        .main {
            padding: 0rem 1rem;
        }
        
        /* Header styling */
        .dashboard-header {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem;
            text-align: center;
            padding: 1rem;
        }
        
        /* Subheader */
        .section-header {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #3498db;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(145deg, #ffffff, #f0f2f6);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 5px 5px 15px #d1d9e6, -5px -5px 15px #ffffff;
            border-left: 5px solid #3498db;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #2c3e50;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            border-radius: 10px 10px 0 0;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3498db;
            color: white;
            border-color: #3498db;
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.5rem 2rem;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted black;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Custom badges */
        .badge {
            display: inline-block;
            padding: 0.25em 0.6em;
            font-size: 75%;
            font-weight: 700;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 10px;
        }
        
        .badge-success {
            color: #fff;
            background-color: #28a745;
        }
        
        .badge-warning {
            color: #212529;
            background-color: #ffc107;
        }
        
        .badge-danger {
            color: #fff;
            background-color: #dc3545;
        }
        
        /* Progress bars */
        .progress {
            height: 10px;
            border-radius: 5px;
            background-color: #e9ecef;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# DATA PROCESSING ENGINE
# ============================================================================
class DataProcessor:
    """Advanced data processing engine for pharmaceutical data"""
    
    @staticmethod
    @st.cache_data(ttl=DashboardConfig.CACHE_TTL, show_spinner=False)
    def load_data(uploaded_file) -> pd.DataFrame:
        """
        Load and validate pharmaceutical sales data
        
        Args:
            uploaded_file: Uploaded file object
            
        Returns:
            Processed DataFrame
        """
        with st.spinner("🔄 Loading and validating data..."):
            try:
                # Detect file type and load
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(
                        uploaded_file, 
                        low_memory=False,
                        encoding='utf-8',
                        on_bad_lines='skip'
                    )
                else:
                    df = pd.read_excel(
                        uploaded_file, 
                        engine='openpyxl',
                        dtype=str
                    )
                
                # Validate required columns
                required_columns = ['Manufacturer', 'Molecule', 'Country', 'Region']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.warning(f"⚠️ Missing columns: {missing_columns}. Some features may be limited.")
                
                return df
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return pd.DataFrame()
    
    @staticmethod
    def transform_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transform wide format to long format
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (transformed DataFrame, list of periods)
        """
        with st.spinner("🔄 Transforming data structure..."):
            # Identify MAT period columns
            mat_pattern = r'MAT Q\d \d{4}'
            mat_columns = [col for col in df.columns if pd.Series(col).str.contains(mat_pattern, na=False).any()]
            
            if not mat_columns:
                st.error("❌ No MAT period columns found. Please check data format.")
                return pd.DataFrame(), []
            
            # Extract unique periods
            periods = []
            for col in mat_columns:
                parts = col.split()
                if len(parts) >= 3:
                    period = f"{parts[0]} {parts[1]} {parts[2]}"
                    if period not in periods:
                        periods.append(period)
            
            periods = sorted(set(periods))
            
            # Define ID columns
            id_columns = [
                'Source.Name', 'Country', 'Sector', 'Panel', 'Region', 
                'Sub-Region', 'Corporation', 'Manufacturer', 'Molecule List',
                'Molecule', 'Chemical Salt', 'International Product', 
                'Specialty Product', 'NFC123', 'International Pack',
                'International Strength', 'International Size', 
                'International Volume', 'International Prescription'
            ]
            
            # Keep only existing columns
            id_columns = [col for col in id_columns if col in df.columns]
            
            # Melt data for each period
            melted_data = []
            
            for period in periods:
                period_data = df[id_columns].copy()
                period_data['Period'] = period
                
                # Add metric columns
                metric_mapping = {
                    f'{period} USD MNF': 'Sales_USD',
                    f'{period} Standard Units': 'Standard_Units',
                    f'{period} Units': 'Units',
                    f'{period} SU Avg Price USD MNF': 'SU_Avg_Price',
                    f'{period} Unit Avg Price USD MNF': 'Unit_Avg_Price'
                }
                
                for source_col, target_col in metric_mapping.items():
                    if source_col in df.columns:
                        period_data[target_col] = pd.to_numeric(
                            df[source_col], 
                            errors='coerce'
                        )
                    else:
                        period_data[target_col] = np.nan
                
                melted_data.append(period_data)
            
            # Combine all periods
            result = pd.concat(melted_data, ignore_index=True)
            
            # Clean and enrich data
            result = DataProcessor._clean_data(result)
            result = DataProcessor._enrich_data(result)
            
            return result, periods
    
    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        # Fill missing values
        numeric_cols = ['Sales_USD', 'Standard_Units', 'Units', 'SU_Avg_Price', 'Unit_Avg_Price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Remove rows with all metrics zero
        metric_cols = [col for col in numeric_cols if col in df.columns]
        if metric_cols:
            df = df[df[metric_cols].sum(axis=1) > 0]
        
        return df
    
    @staticmethod
    def _enrich_data(df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated columns"""
        # Extract year and quarter
        df['Year'] = df['Period'].str.extract(r'(\d{4})')[0]
        df['Quarter'] = df['Period'].str.extract(r'Q(\d)')[0]
        
        # Add calculated metrics
        if 'Standard_Units' in df.columns and 'Sales_USD' in df.columns:
            mask = df['Standard_Units'] > 0
            df.loc[mask, 'Calculated_Price'] = df.loc[mask, 'Sales_USD'] / df.loc[mask, 'Standard_Units']
        
        # Add period index for sorting
        df['Period_Index'] = df['Year'].astype(str) + df['Quarter'].str.zfill(2)
        
        return df
    
    @staticmethod
    def calculate_market_metrics(
        df: pd.DataFrame, 
        groupby_col: str, 
        metric: str = 'Sales_USD'
    ) -> pd.DataFrame:
        """
        Calculate comprehensive market metrics
        
        Args:
            df: Input DataFrame
            groupby_col: Column to group by
            metric: Metric to analyze
            
        Returns:
            DataFrame with market metrics
        """
        try:
            periods = sorted(df['Period'].unique())
            if len(periods) < 2:
                return pd.DataFrame()
            
            latest_period = periods[-1]
            previous_period = periods[-2]
            
            # Calculate metrics for each period
            latest_data = (
                df[df['Period'] == latest_period]
                .groupby(groupby_col)[metric]
                .sum()
                .reset_index()
                .rename(columns={metric: 'Current_Period'})
            )
            
            previous_data = (
                df[df['Period'] == previous_period]
                .groupby(groupby_col)[metric]
                .sum()
                .reset_index()
                .rename(columns={metric: 'Previous_Period'})
            )
            
            # Merge and calculate
            merged = pd.merge(
                latest_data, 
                previous_data, 
                on=groupby_col, 
                how='left'
            ).fillna(0)
            
            total_current = merged['Current_Period'].sum()
            
            merged['Market_Share_Pct'] = (
                merged['Current_Period'] / total_current * 100
            ).round(2)
            
            merged['Growth_Pct'] = (
                (merged['Current_Period'] - merged['Previous_Period']) / 
                merged['Previous_Period'].replace(0, np.nan) * 100
            ).round(2).fillna(0)
            
            merged['Abs_Growth'] = (
                merged['Current_Period'] - merged['Previous_Period']
            ).round(2)
            
            # Calculate rankings
            merged['Rank'] = merged['Current_Period'].rank(
                ascending=False, 
                method='dense'
            ).astype(int)
            
            # Sort by rank
            merged = merged.sort_values('Rank')
            
            # Add contribution to growth
            total_growth = merged['Abs_Growth'].sum()
            if total_growth != 0:
                merged['Contribution_to_Growth_Pct'] = (
                    merged['Abs_Growth'] / total_growth * 100
                ).round(2)
            else:
                merged['Contribution_to_Growth_Pct'] = 0
            
            return merged
            
        except Exception as e:
            st.error(f"Error calculating market metrics: {str(e)}")
            return pd.DataFrame()


# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================
class VisualizationEngine:
    """Advanced visualization engine for creating interactive charts"""
    
    @staticmethod
    def create_metric_cards(df: pd.DataFrame, latest_period: str) -> None:
        """Create dashboard metric cards"""
        latest_data = df[df['Period'] == latest_period]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = [
            ('💰 Total Sales', f"${latest_data['Sales_USD'].sum()/1e6:.1f}M", 
             'Total sales in millions USD'),
            ('📦 Total Units', f"{latest_data['Standard_Units'].sum()/1e6:.1f}M", 
             'Total standard units in millions'),
            ('💵 Avg Price', f"${latest_data['SU_Avg_Price'].mean():.2f}", 
             'Average price per standard unit'),
            ('🏷️ Products', f"{latest_data['International Product'].nunique():,}", 
             'Number of unique products'),
            ('🏭 Manufacturers', f"{latest_data['Manufacturer'].nunique():,}", 
             'Number of manufacturers')
        ]
        
        cols = [col1, col2, col3, col4, col5]
        for col, (title, value, help_text) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{title}</div>
                    <div class="metric-value">{value}</div>
                    <div style="font-size: 0.8rem; color: #95a5a6; margin-top: 0.5rem;">
                        {help_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def create_market_share_chart(
        data: pd.DataFrame, 
        title: str,
        x_col: str = 'Market_Share_Pct',
        y_col: str = 'Manufacturer',
        color_col: str = 'Growth_Pct'
    ) -> go.Figure:
        """Create horizontal bar chart for market share"""
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            orientation='h',
            text=x_col,
            color=color_col,
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            labels={
                x_col: 'Market Share (%)',
                color_col: 'Growth (%)',
                y_col: ''
            },
            title=title
        )
        
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            marker_line_width=0
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_growth_heatmap(
        data: pd.DataFrame,
        x_categories: List[str],
        y_categories: List[str],
        metric: str = 'Sales_USD'
    ) -> go.Figure:
        """Create growth heatmap"""
        # Prepare heatmap data
        heatmap_data = []
        annotations = []
        
        for i, y_cat in enumerate(y_categories):
            row = []
            for j, x_cat in enumerate(x_categories):
                subset = data[
                    (data['Manufacturer'] == y_cat) & 
                    (data['Molecule'] == x_cat)
                ]
                
                if len(subset) >= 2:
                    periods = sorted(subset['Period'].unique())
                    latest = subset[subset['Period'] == periods[-1]][metric].sum()
                    previous = subset[subset['Period'] == periods[-2]][metric].sum()
                    growth = ((latest - previous) / previous * 100) if previous > 0 else 0
                else:
                    growth = 0
                
                row.append(growth)
                annotations.append(
                    dict(
                        x=x_cat,
                        y=y_cat,
                        text=f"{growth:.1f}%",
                        showarrow=False,
                        font=dict(color='white' if abs(growth) > 20 else 'black')
                    )
                )
            
            heatmap_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=x_categories,
            y=y_categories,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(
                title="Growth %",
                titleside="right",
                titlefont=dict(size=14)
            ),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Growth Heatmap: Manufacturer × Molecule",
            xaxis_title="Molecule",
            yaxis_title="Manufacturer",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        return fig
    
    @staticmethod
    def create_geographic_map(
        data: pd.DataFrame,
        metric: str = 'Sales_USD'
    ) -> go.Figure:
        """Create geographic choropleth map"""
        # Country to ISO mapping
        country_mapping = {
            'ALGERIA': 'DZA', 'ARGENTINA': 'ARG', 'AUSTRALIA': 'AUS', 'AUSTRIA': 'AUT',
            'BELGIUM': 'BEL', 'BRAZIL': 'BRA', 'CANADA': 'CAN', 'CHILE': 'CHL',
            'CHINA': 'CHN', 'COLOMBIA': 'COL', 'CZECH REPUBLIC': 'CZE', 'DENMARK': 'DNK',
            'EGYPT': 'EGY', 'FINLAND': 'FIN', 'FRANCE': 'FRA', 'GERMANY': 'DEU',
            'GREECE': 'GRC', 'HONG KONG': 'HKG', 'HUNGARY': 'HUN', 'INDIA': 'IND',
            'INDONESIA': 'IDN', 'IRELAND': 'IRL', 'ISRAEL': 'ISR', 'ITALY': 'ITA',
            'JAPAN': 'JPN', 'MALAYSIA': 'MYS', 'MEXICO': 'MEX', 'NETHERLANDS': 'NLD',
            'NEW ZEALAND': 'NZL', 'NORWAY': 'NOR', 'PAKISTAN': 'PAK', 'PERU': 'PER',
            'PHILIPPINES': 'PHL', 'POLAND': 'POL', 'PORTUGAL': 'PRT', 'ROMANIA': 'ROU',
            'RUSSIA': 'RUS', 'SAUDI ARABIA': 'SAU', 'SINGAPORE': 'SGP', 'SOUTH AFRICA': 'ZAF',
            'SOUTH KOREA': 'KOR', 'SPAIN': 'ESP', 'SWEDEN': 'SWE', 'SWITZERLAND': 'CHE',
            'TAIWAN': 'TWN', 'THAILAND': 'THA', 'TURKEY': 'TUR', 'UAE': 'ARE',
            'UK': 'GBR', 'UKRAINE': 'UKR', 'USA': 'USA', 'VENEZUELA': 'VEN',
            'VIETNAM': 'VNM'
        }
        
        # Aggregate data
        agg_data = data.groupby('Country').agg({
            metric: 'sum',
            'Manufacturer': 'nunique',
            'Molecule': 'nunique'
        }).reset_index()
        
        agg_data['ISO'] = agg_data['Country'].str.upper().map(country_mapping)
        agg_data = agg_data.dropna(subset=['ISO'])
        
        # Create map
        fig = px.choropleth(
            agg_data,
            locations='ISO',
            color=metric,
            hover_name='Country',
            hover_data={
                'ISO': False,
                metric: ':,.0f',
                'Manufacturer': True,
                'Molecule': True
            },
            color_continuous_scale='Blues',
            title='Global Market Distribution',
            labels={metric: 'Sales (USD)' if metric == 'Sales_USD' else 'Units'}
        )
        
        fig.update_layout(
            height=600,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                landcolor='rgb(243, 243, 243)',
                countrycolor='rgb(204, 204, 204)'
            ),
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_portfolio_scatter(
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        size_col: str,
        color_col: str,
        title: str
    ) -> go.Figure:
        """Create portfolio analysis scatter plot"""
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            size=size_col,
            color=color_col,
            hover_name='Manufacturer',
            text='Manufacturer',
            color_continuous_scale='Plasma',
            size_max=50,
            title=title,
            labels={
                x_col: 'Number of Molecules',
                y_col: 'Number of Products',
                size_col: 'Total Sales',
                color_col: 'Total Sales'
            }
        )
        
        fig.update_traces(
            textposition='top center',
            textfont=dict(size=9),
            marker=dict(line=dict(width=0.5, color='DarkSlateGrey'))
        )
        
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        )
        
        return fig


# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================
class DashboardLayout:
    """Dashboard layout manager"""
    
    @staticmethod
    def create_sidebar_filters(df: pd.DataFrame) -> Dict:
        """Create sidebar filters"""
        st.sidebar.header("🎛️ Control Panel")
        
        filters = {}
        
        # Metric selection
        filters['metric'] = st.sidebar.radio(
            "📊 Select Primary Metric",
            options=['Sales_USD', 'Standard_Units'],
            format_func=lambda x: '💰 Sales (USD)' if x == 'Sales_USD' else '📦 Volume (Units)',
            help="Choose the primary metric for analysis"
        )
        
        st.sidebar.markdown("---")
        
        # Period filter
        available_periods = sorted(df['Period'].unique())
        filters['periods'] = st.sidebar.multiselect(
            "📅 Select Periods",
            options=available_periods,
            default=available_periods[-4:] if len(available_periods) >= 4 else available_periods,
            help="Select periods to analyze"
        )
        
        # Region filter
        regions = ['All'] + sorted(df['Region'].dropna().unique().tolist())
        filters['region'] = st.sidebar.selectbox(
            "🌍 Region", 
            regions,
            help="Filter by geographic region"
        )
        
        # Country filter
        if filters['region'] != 'All':
            countries = ['All'] + sorted(
                df[df['Region'] == filters['region']]['Country']
                .dropna().unique().tolist()
            )
        else:
            countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
        
        filters['country'] = st.sidebar.selectbox(
            "🏴 Country", 
            countries,
            help="Filter by country"
        )
        
        # Manufacturer filter
        manufacturers = sorted(df['Manufacturer'].dropna().unique().tolist())
        filters['manufacturers'] = st.sidebar.multiselect(
            "🏭 Manufacturers",
            options=manufacturers,
            default=[],
            help="Select specific manufacturers (empty for all)"
        )
        
        # Molecule filter
        molecules = sorted(df['Molecule'].dropna().unique().tolist())
        filters['molecules'] = st.sidebar.multiselect(
            "🧪 Molecules",
            options=molecules,
            default=[],
            help="Select specific molecules (empty for all)"
        )
        
        st.sidebar.markdown("---")
        
        # Dataset info
        st.sidebar.subheader("📊 Dataset Info")
        DashboardLayout._display_dataset_info(df)
        
        return filters
    
    @staticmethod
    def _display_dataset_info(df: pd.DataFrame):
        """Display dataset information in sidebar"""
        info_metrics = [
            ("Total Records", f"{len(df):,}"),
            ("Date Range", f"{df['Period'].nunique()} periods"),
            ("Countries", df['Country'].nunique()),
            ("Manufacturers", df['Manufacturer'].nunique()),
            ("Molecules", df['Molecule'].nunique()),
            ("Products", df['International Product'].nunique() if 'International Product' in df.columns else "N/A")
        ]
        
        for label, value in info_metrics:
            st.sidebar.metric(label, value)
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        if filters['periods']:
            filtered_df = filtered_df[filtered_df['Period'].isin(filters['periods'])]
        
        if filters['region'] != 'All':
            filtered_df = filtered_df[filtered_df['Region'] == filters['region']]
        
        if filters['country'] != 'All':
            filtered_df = filtered_df[filtered_df['Country'] == filters['country']]
        
        if filters['manufacturers']:
            filtered_df = filtered_df[filtered_df['Manufacturer'].isin(filters['manufacturers'])]
        
        if filters['molecules']:
            filtered_df = filtered_df[filtered_df['Molecule'].isin(filters['molecules'])]
        
        return filtered_df


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class PharmaAnalyticsDashboard:
    """Main dashboard application"""
    
    def __init__(self):
        self.config = DashboardConfig()
        self.data_processor = DataProcessor()
        self.visualizer = VisualizationEngine()
        self.layout = DashboardLayout()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'periods' not in st.session_state:
            st.session_state.periods = []
    
    def run(self):
        """Run the dashboard application"""
        # Page configuration
        st.set_page_config(
            page_title=self.config.PAGE_TITLE,
            page_icon=self.config.PAGE_ICON,
            layout=self.config.LAYOUT,
            initial_sidebar_state=self.config.INITIAL_SIDEBAR_STATE
        )
        
        # Load custom CSS
        load_custom_css()
        
        # Display header
        self._display_header()
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📁 Upload Pharmaceutical Sales Data",
            type=['xlsx', 'xls', 'csv'],
            help="Upload Excel (.xlsx, .xls) or CSV files containing sales data"
        )
        
        if uploaded_file is None:
            self._display_welcome_screen()
            return
        
        # Process data
        if not st.session_state.data_loaded or st.session_state.processed_data is None:
            with st.spinner("🔄 Processing data... This may take a moment."):
                raw_data = self.data_processor.load_data(uploaded_file)
                if not raw_data.empty:
                    processed_data, periods = self.data_processor.transform_data(raw_data)
                    if not processed_data.empty:
                        st.session_state.processed_data = processed_data
                        st.session_state.periods = periods
                        st.session_state.data_loaded = True
                        st.success(f"✅ Successfully processed {len(processed_data):,} records")
                    else:
                        st.error("❌ Failed to process data")
                        return
                else:
                    return
        
        # Get processed data
        df = st.session_state.processed_data
        
        # Create sidebar filters
        filters = self.layout.create_sidebar_filters(df)
        
        # Apply filters
        filtered_df = self.layout.apply_filters(df, filters)
        
        if len(filtered_df) == 0:
            st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")
            return
        
        # Create tabs
        tab_names = [
            "📊 Executive Summary",
            "🌍 Geographic Analysis", 
            "🏭 Manufacturer Intel",
            "🧪 Molecule Deep Dive",
            "💰 Price Analytics",
            "📈 Trend Analysis",
            "🔬 Advanced Insights"
        ]
        
        tabs = st.tabs(tab_names)
        
        # Tab 1: Executive Summary
        with tabs[0]:
            self._render_executive_summary(filtered_df, filters['metric'])
        
        # Tab 2: Geographic Analysis
        with tabs[1]:
            self._render_geographic_analysis(filtered_df, filters['metric'])
        
        # Tab 3: Manufacturer Intelligence
        with tabs[2]:
            self._render_manufacturer_intel(filtered_df, filters['metric'])
        
        # Tab 4: Molecule Analysis
        with tabs[3]:
            self._render_molecule_analysis(filtered_df, filters['metric'])
        
        # Tab 5: Price Analytics
        with tabs[4]:
            self._render_price_analytics(filtered_df)
        
        # Tab 6: Trend Analysis
        with tabs[5]:
            self._render_trend_analysis(filtered_df, filters['metric'])
        
        # Tab 7: Advanced Insights
        with tabs[6]:
            self._render_advanced_insights(filtered_df, filters['metric'])
    
    def _display_header(self):
        """Display dashboard header"""
        st.markdown(f"""
        <div class="dashboard-header">
            {self.config.PAGE_TITLE}
        </div>
        <div style="text-align: center; color: #7f8c8d; margin-bottom: 2rem;">
            Enterprise-grade analytics platform for pharmaceutical market intelligence • v2.1.0
        </div>
        """, unsafe_allow_html=True)
    
    def _display_welcome_screen(self):
        """Display welcome screen when no data is uploaded"""
        st.info("👆 Please upload a pharmaceutical sales data file to begin analysis")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🌍 **Geographic Analysis**
            - Interactive world map visualization
            - Regional performance benchmarking
            - Country-level market insights
            - Sub-region drill-down capabilities
            """)
        
        with col2:
            st.markdown("""
            ### 📈 **Market Intelligence**
            - Manufacturer ranking & benchmarking
            - Molecule performance analysis
            - Price positioning strategy
            - Competitive landscape mapping
            """)
        
        with col3:
            st.markdown("""
            ### ⚡ **Advanced Features**
            - Growth heatmaps & trend analysis
            - Portfolio optimization insights
            - Market concentration metrics
            - Export-ready reports
            """)
        
        st.markdown("---")
        
        # Sample data format
        with st.expander("📋 Expected Data Format", expanded=False):
            st.markdown("""
            The dashboard expects pharmaceutical sales data in a specific format:
            
            **Required Columns:**
            - `Manufacturer`: Company name
            - `Molecule`: Active ingredient
            - `Country`: Market country
            - `Region`: Geographic region
            
            **MAT Period Columns (example formats):**
            - `MAT Q4 2023 USD MNF`
            - `MAT Q4 2023 Standard Units`
            - `MAT Q3 2023 USD MNF`
            - `MAT Q3 2023 Standard Units`
            
            **Optional Columns:**
            - `Corporation`: Parent company
            - `International Product`: Product name
            - `Chemical Salt`: Chemical formulation
            - `Sector`: Market sector
            """)
    
    def _render_executive_summary(self, df: pd.DataFrame, metric: str):
        """Render executive summary tab"""
        st.markdown('<div class="section-header">📊 Executive Summary</div>', 
                   unsafe_allow_html=True)
        
        latest_period = sorted(df['Period'].unique())[-1]
        
        # Display metric cards
        self.visualizer.create_metric_cards(df, latest_period)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top manufacturers by market share
            manufacturer_metrics = self.data_processor.calculate_market_metrics(
                df, 'Manufacturer', metric
            )
            if not manufacturer_metrics.empty:
                fig = self.visualizer.create_market_share_chart(
                    manufacturer_metrics.head(10),
                    "🏆 Top 10 Manufacturers by Market Share"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top molecules
            molecule_data = df[df['Period'] == latest_period]
            top_molecules = (
                molecule_data.groupby('Molecule')[metric]
                .sum()
                .reset_index()
                .sort_values(metric, ascending=False)
                .head(10)
            )
            
            fig = px.pie(
                top_molecules,
                values=metric,
                names='Molecule',
                hole=0.4,
                title="🧪 Top 10 Molecules Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                pull=[0.1] + [0] * 9
            )
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Growth heatmap
        st.markdown('<div class="section-header">🔥 Growth Heatmap Analysis</div>', 
                   unsafe_allow_html=True)
        
        top_manufacturers = manufacturer_metrics.head(10)['Manufacturer'].tolist()
        molecule_metrics = self.data_processor.calculate_market_metrics(df, 'Molecule', metric)
        top_molecules = molecule_metrics.head(10)['Molecule'].tolist()
        
        if top_manufacturers and top_molecules:
            fig = self.visualizer.create_growth_heatmap(
                df, top_molecules, top_manufacturers, metric
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_geographic_analysis(self, df: pd.DataFrame, metric: str):
        """Render geographic analysis tab"""
        st.markdown('<div class="section-header">🌍 Geographic Analysis</div>', 
                   unsafe_allow_html=True)
        
        # World map
        latest_period = sorted(df['Period'].unique())[-1]
        latest_data = df[df['Period'] == latest_period]
        
        fig = self.visualizer.create_geographic_map(latest_data, metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional performance
            region_data = latest_data.groupby('Region')[metric].sum().reset_index()
            region_data = region_data.sort_values(metric, ascending=False)
            
            fig = px.bar(
                region_data,
                x='Region',
                y=metric,
                color=metric,
                color_continuous_scale='Viridis',
                title="📊 Regional Performance",
                labels={metric: 'Sales' if metric == 'Sales_USD' else 'Units'}
            )
            fig.update_traces(
                texttemplate='%{y:.2s}',
                textposition='outside',
                marker_line_width=0
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top countries
            country_data = latest_data.groupby('Country')[metric].sum().reset_index()
            top_countries = country_data.nlargest(15, metric)
            
            fig = px.bar(
                top_countries,
                x=metric,
                y='Country',
                orientation='h',
                color=metric,
                color_continuous_scale='Teal',
                title="🏆 Top 15 Countries",
                labels={metric: 'Sales' if metric == 'Sales_USD' else 'Units'}
            )
            fig.update_traces(
                texttemplate='%{x:.2s}',
                textposition='outside',
                marker_line_width=0
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_manufacturer_intel(self, df: pd.DataFrame, metric: str):
        """Render manufacturer intelligence tab"""
        st.markdown('<div class="section-header">🏭 Manufacturer Intelligence</div>', 
                   unsafe_allow_html=True)
        
        manufacturer_metrics = self.data_processor.calculate_market_metrics(
            df, 'Manufacturer', metric
        )
        
        if manufacturer_metrics.empty:
            st.warning("Insufficient data for manufacturer analysis")
            return
        
        # Market leaders bubble chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.scatter(
                manufacturer_metrics.head(20),
                x='Market_Share_Pct',
                y='Growth_Pct',
                size='Current_Period',
                color='Growth_Pct',
                hover_name='Manufacturer',
                text='Manufacturer',
                color_continuous_scale='RdYlGn',
                size_max=60,
                title="🎯 Market Leaders Analysis",
                labels={
                    'Market_Share_Pct': 'Market Share (%)',
                    'Growth_Pct': 'Growth (%)',
                    'Current_Period': 'Current Period Value'
                }
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(
                x=manufacturer_metrics['Market_Share_Pct'].median(),
                line_dash="dash", 
                line_color="gray"
            )
            fig.update_traces(
                textposition='top center',
                textfont=dict(size=9)
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rankings table
            st.subheader("📋 Manufacturer Rankings")
            display_df = manufacturer_metrics[['Rank', 'Manufacturer', 'Market_Share_Pct', 'Growth_Pct']].head(15)
            
            # Apply styling
            styled_df = display_df.style.format({
                'Market_Share_Pct': '{:.1f}%',
                'Growth_Pct': '{:+.1f}%'
            }).background_gradient(
                subset=['Market_Share_Pct'], 
                cmap='Blues'
            ).background_gradient(
                subset=['Growth_Pct'], 
                cmap='RdYlGn',
                vmin=-50, 
                vmax=50
            )
            
            st.dataframe(
                styled_df,
                height=500,
                use_container_width=True,
                hide_index=True
            )
        
        # Portfolio analysis
        st.markdown('<div class="section-header">📦 Portfolio Diversity Analysis</div>', 
                   unsafe_allow_html=True)
        
        latest_period = sorted(df['Period'].unique())[-1]
        latest_data = df[df['Period'] == latest_period]
        
        portfolio_data = latest_data.groupby('Manufacturer').agg({
            'Molecule': 'nunique',
            'International Product': 'nunique',
            metric: 'sum'
        }).reset_index()
        
        portfolio_data.columns = [
            'Manufacturer', 
            'Unique_Molecules', 
            'Unique_Products', 
            'Total_Value'
        ]
        
        portfolio_data = portfolio_data.sort_values('Total_Value', ascending=False).head(20)
        
        fig = self.visualizer.create_portfolio_scatter(
            portfolio_data,
            'Unique_Molecules',
            'Unique_Products',
            'Total_Value',
            'Total_Value',
            '📊 Portfolio Efficiency: Diversity vs Concentration'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_molecule_analysis(self, df: pd.DataFrame, metric: str):
        """Render molecule analysis tab"""
        st.markdown('<div class="section-header">🧪 Molecule Deep Dive</div>', 
                   unsafe_allow_html=True)
        
        molecule_metrics = self.data_processor.calculate_market_metrics(
            df, 'Molecule', metric
        )
        
        if molecule_metrics.empty:
            st.warning("Insufficient data for molecule analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top molecules treemap
            top_molecules = molecule_metrics.head(20)
            
            fig = px.treemap(
                top_molecules,
                path=['Molecule'],
                values='Current_Period',
                color='Growth_Pct',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                hover_data=['Market_Share_Pct', 'Growth_Pct'],
                title="🔬 Top 20 Molecules Market View"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Growth vs Market Share
            fig = px.scatter(
                molecule_metrics.head(30),
                x='Market_Share_Pct',
                y='Growth_Pct',
                size='Current_Period',
                color='Growth_Pct',
                hover_name='Molecule',
                color_continuous_scale='RdYlGn',
                size_max=40,
                title="📈 Growth vs Market Share Analysis",
                labels={
                    'Market_Share_Pct': 'Market Share (%)',
                    'Growth_Pct': 'Growth (%)'
                }
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(
                x=molecule_metrics['Market_Share_Pct'].median(),
                line_dash="dash", 
                line_color="gray"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Manufacturer analysis for selected molecule
        st.markdown('<div class="section-header">🏭 Manufacturer Share by Molecule</div>', 
                   unsafe_allow_html=True)
        
        # Select molecule
        available_molecules = molecule_metrics.head(20)['Molecule'].tolist()
        selected_molecule = st.selectbox(
            "Select a molecule for detailed analysis",
            options=available_molecules,
            index=0,
            help="Choose a molecule to analyze manufacturer distribution"
        )
        
        latest_period = sorted(df['Period'].unique())[-1]
        molecule_data = df[
            (df['Period'] == latest_period) & 
            (df['Molecule'] == selected_molecule)
        ]
        
        if not molecule_data.empty:
            manufacturer_share = (
                molecule_data.groupby('Manufacturer')[metric]
                .sum()
                .reset_index()
                .sort_values(metric, ascending=False)
                .head(10)
            )
            
            fig = px.bar(
                manufacturer_share,
                x='Manufacturer',
                y=metric,
                color=metric,
                color_continuous_scale='Viridis',
                title=f"🏭 Manufacturer Distribution for {selected_molecule}",
                labels={metric: 'Value'}
            )
            fig.update_traces(
                texttemplate='%{y:.2s}',
                textposition='outside'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_price_analytics(self, df: pd.DataFrame):
        """Render price analytics tab"""
        st.markdown('<div class="section-header">💰 Price Analytics</div>', 
                   unsafe_allow_html=True)
        
        latest_period = sorted(df['Period'].unique())[-1]
        latest_data = df[df['Period'] == latest_period]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            price_data = latest_data[latest_data['SU_Avg_Price'] > 0]['SU_Avg_Price']
            
            if len(price_data) > 0:
                fig = px.histogram(
                    price_data,
                    nbins=50,
                    title="📊 Price Distribution",
                    labels={'value': 'Price per Unit (USD)', 'count': 'Frequency'},
                    color_discrete_sequence=['#636EFA'],
                    opacity=0.8
                )
                
                # Add mean line
                mean_price = price_data.mean()
                fig.add_vline(
                    x=mean_price, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Mean: ${mean_price:.2f}",
                    annotation_position="top right"
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price statistics
            st.subheader("💵 Price Statistics")
            
            if len(price_data) > 0:
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.metric("Mean Price", f"${price_data.mean():.2f}")
                    st.metric("Median Price", f"${price_data.median():.2f}")
                    st.metric("Std Deviation", f"${price_data.std():.2f}")
                
                with stats_col2:
                    st.metric("Min Price", f"${price_data.min():.2f}")
                    st.metric("Max Price", f"${price_data.max():.2f}")
                    st.metric("IQR", f"${price_data.quantile(0.75) - price_data.quantile(0.25):.2f}")
            else:
                st.warning("No price data available")
        
        # Price vs Volume analysis
        st.markdown('<div class="section-header">🎯 Price Positioning Analysis</div>', 
                   unsafe_allow_html=True)
        
        scatter_data = latest_data[latest_data['SU_Avg_Price'] > 0].groupby('Manufacturer').agg({
            'SU_Avg_Price': 'mean',
            'Standard_Units': 'sum',
            'Sales_USD': 'sum'
        }).reset_index()
        
        scatter_data = scatter_data.sort_values('Sales_USD', ascending=False).head(30)
        
        if not scatter_data.empty:
            fig = px.scatter(
                scatter_data,
                x='SU_Avg_Price',
                y='Standard_Units',
                size='Sales_USD',
                color='SU_Avg_Price',
                hover_name='Manufacturer',
                text='Manufacturer',
                color_continuous_scale='Turbo',
                size_max=50,
                title="Price vs Volume: Manufacturer Positioning",
                labels={
                    'SU_Avg_Price': 'Average Price (USD)',
                    'Standard_Units': 'Total Volume'
                }
            )
            fig.update_traces(
                textposition='top center',
                textfont=dict(size=8)
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Price trends
        st.markdown('<div class="section-header">📈 Price Trends Over Time</div>', 
                   unsafe_allow_html=True)
        
        # Get top molecules
        top_molecules = (
            latest_data.groupby('Molecule')['Sales_USD']
            .sum()
            .nlargest(5)
            .index.tolist()
        )
        
        if top_molecules:
            price_trend_data = df[
                (df['Molecule'].isin(top_molecules)) & 
                (df['SU_Avg_Price'] > 0)
            ]
            
            price_trend_agg = price_trend_data.groupby(['Period', 'Molecule'])[
                'SU_Avg_Price'
            ].mean().reset_index()
            
            fig = px.line(
                price_trend_agg,
                x='Period',
                y='SU_Avg_Price',
                color='Molecule',
                markers=True,
                title="Price Evolution of Top 5 Molecules",
                labels={'SU_Avg_Price': 'Average Price (USD)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_trend_analysis(self, df: pd.DataFrame, metric: str):
        """Render trend analysis tab"""
        st.markdown('<div class="section-header">📈 Trend Analysis</div>', 
                   unsafe_allow_html=True)
        
        # Time series analysis
        time_data = df.groupby('Period').agg({
            'Sales_USD': 'sum',
            'Standard_Units': 'sum',
            'Manufacturer': 'nunique',
            'Molecule': 'nunique'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=('Sales USD Trend', 'Standard Units Trend',
                          'Manufacturer Count', 'Molecule Count'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Sales trend
        fig.add_trace(
            go.Scatter(
                x=time_data['Period'], 
                y=time_data['Sales_USD'],
                mode='lines+markers', 
                name='Sales USD',
                line=dict(color='#1f77b4', width=3)
            ),
            row=1, col=1
        )
        
        # Units trend
        fig.add_trace(
            go.Scatter(
                x=time_data['Period'], 
                y=time_data['Standard_Units'],
                mode='lines+markers', 
                name='Standard Units',
                line=dict(color='#ff7f0e', width=3)
            ),
            row=1, col=2
        )
        
        # Manufacturer count
        fig.add_trace(
            go.Bar(
                x=time_data['Period'], 
                y=time_data['Manufacturer'],
                name='Manufacturers',
                marker_color='#2ca02c'
            ),
            row=2, col=1
        )
        
        # Molecule count
        fig.add_trace(
            go.Bar(
                x=time_data['Period'], 
                y=time_data['Molecule'],
                name='Molecules',
                marker_color='#d62728'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600, 
            showlegend=False,
            title_text="Market Evolution Trends"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth analysis
        st.markdown('<div class="section-header">📊 Growth Performance Analysis</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top growers
            manufacturer_metrics = self.data_processor.calculate_market_metrics(
                df, 'Manufacturer', metric
            )
            
            if not manufacturer_metrics.empty:
                top_growers = manufacturer_metrics.nlargest(10, 'Growth_Pct')[
                    ['Manufacturer', 'Growth_Pct', 'Market_Share_Pct']
                ]
                
                fig = px.bar(
                    top_growers,
                    x='Growth_Pct',
                    y='Manufacturer',
                    orientation='h',
                    color='Growth_Pct',
                    color_continuous_scale='Greens',
                    title="🚀 Top 10 Growers",
                    labels={'Growth_Pct': 'Growth (%)'}
                )
                fig.update_traces(
                    texttemplate='%{x:.1f}%',
                    textposition='outside'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Decliners
            if not manufacturer_metrics.empty:
                decliners = manufacturer_metrics.nsmallest(10, 'Growth_Pct')[
                    ['Manufacturer', 'Growth_Pct', 'Market_Share_Pct']
                ]
                
                fig = px.bar(
                    decliners,
                    x='Growth_Pct',
                    y='Manufacturer',
                    orientation='h',
                    color='Growth_Pct',
                    color_continuous_scale='Reds',
                    title="📉 Top 10 Decliners",
                    labels={'Growth_Pct': 'Growth (%)'}
                )
                fig.update_traces(
                    texttemplate='%{x:.1f}%',
                    textposition='outside'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Quarterly analysis
        st.markdown('<div class="section-header">📅 Quarterly Performance Analysis</div>', 
                   unsafe_allow_html=True)
        
        quarterly_data = df.groupby(['Year', 'Quarter']).agg({
            metric: 'sum'
        }).reset_index()
        
        quarterly_data['Period_Label'] = (
            'Q' + quarterly_data['Quarter'] + ' ' + quarterly_data['Year']
        )
        
        fig = px.bar(
            quarterly_data,
            x='Period_Label',
            y=metric,
            color=metric,
            color_continuous_scale='Blues',
            title="Quarterly Performance",
            labels={
                'Period_Label': 'Quarter',
                metric: 'Value'
            }
        )
        
        fig.update_traces(
            texttemplate='%{y:.2s}',
            textposition='outside'
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_advanced_insights(self, df: pd.DataFrame, metric: str):
        """Render advanced insights tab"""
        st.markdown('<div class="section-header">🔬 Advanced Market Insights</div>', 
                   unsafe_allow_html=True)
        
        # Market concentration metrics
        st.subheader("📊 Market Concentration Analysis")
        
        manufacturer_metrics = self.data_processor.calculate_market_metrics(
            df, 'Manufacturer', metric
        )
        
        if not manufacturer_metrics.empty:
            col1, col2, col3 = st.columns(3)
            
            # HHI calculation
            market_shares = manufacturer_metrics['Market_Share_Pct'] / 100
            hhi = (market_shares ** 2).sum() * 10000
            
            with col1:
                st.metric(
                    "Herfindahl-Hirschman Index (HHI)",
                    f"{hhi:.0f}",
                    help="HHI < 1500: Competitive market\n"
                         "1500-2500: Moderately concentrated\n"
                         ">2500: Highly concentrated"
                )
            
            # Concentration ratios
            cr4 = manufacturer_metrics.head(4)['Market_Share_Pct'].sum()
            cr8 = manufacturer_metrics.head(8)['Market_Share_Pct'].sum()
            
            with col2:
                st.metric(
                    "CR4 (Top 4 Concentration)",
                    f"{cr4:.1f}%",
                    delta=f"CR8: {cr8:.1f}%",
                    delta_color="off",
                    help="Market share of top 4 players"
                )
            
            # Number of effective competitors
            n_effective = 1 / (market_shares ** 2).sum() if (market_shares ** 2).sum() > 0 else 0
            
            with col3:
                st.metric(
                    "Effective Number of Competitors",
                    f"{n_effective:.1f}",
                    help="Number of equally-sized competitors needed to match concentration"
                )
        
        # Pareto analysis
        st.markdown('<div class="section-header">📈 Pareto Analysis (80/20 Rule)</div>', 
                   unsafe_allow_html=True)
        
        if not manufacturer_metrics.empty:
            pareto_data = manufacturer_metrics.copy()
            pareto_data['Cumulative_Share_Pct'] = pareto_data['Market_Share_Pct'].cumsum()
            pareto_data['Manufacturer_Rank'] = range(1, len(pareto_data) + 1)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Market share bars
            fig.add_trace(
                go.Bar(
                    x=pareto_data['Manufacturer_Rank'], 
                    y=pareto_data['Market_Share_Pct'],
                    name='Market Share',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                secondary_y=False
            )
            
            # Cumulative line
            fig.add_trace(
                go.Scatter(
                    x=pareto_data['Manufacturer_Rank'], 
                    y=pareto_data['Cumulative_Share_Pct'],
                    name='Cumulative Share',
                    line=dict(color='red', width=3),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            # Add 80% line
            fig.add_hline(
                y=80, 
                line_dash="dash", 
                line_color="green", 
                secondary_y=True,
                annotation_text="80%",
                annotation_position="right"
            )
            
            fig.update_xaxes(title_text="Manufacturer Rank")
            fig.update_yaxes(title_text="Market Share (%)", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative Share (%)", secondary_y=True)
            
            fig.update_layout(
                height=500,
                title="Pareto Analysis: Market Concentration",
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio efficiency
        st.markdown('<div class="section-header">💼 Portfolio Efficiency Analysis</div>', 
                   unsafe_allow_html=True)
        
        latest_period = sorted(df['Period'].unique())[-1]
        latest_data = df[df['Period'] == latest_period]
        
        efficiency_data = latest_data.groupby('Manufacturer').agg({
            'International Product': 'nunique',
            metric: 'sum'
        }).reset_index()
        
        efficiency_data.columns = ['Manufacturer', 'Product_Count', 'Total_Value']
        
        efficiency_data['Efficiency'] = (
            efficiency_data['Total_Value'] / efficiency_data['Product_Count']
        ).fillna(0)
        
        efficiency_data = efficiency_data.sort_values('Total_Value', ascending=False).head(20)
        
        fig = px.scatter(
            efficiency_data,
            x='Product_Count',
            y='Efficiency',
            size='Total_Value',
            color='Efficiency',
            hover_name='Manufacturer',
            text='Manufacturer',
            color_continuous_scale='Viridis',
            title="Portfolio Efficiency: Revenue per Product",
            labels={
                'Product_Count': 'Number of Products',
                'Efficiency': f'Revenue per Product',
                'Total_Value': 'Total Revenue'
            },
            size_max=50
        )
        
        fig.update_traces(
            textposition='top center',
            textfont=dict(size=8)
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Data export
        st.markdown('<div class="section-header">💾 Export Analysis Results</div>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not manufacturer_metrics.empty:
                export_manufacturer = manufacturer_metrics.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Manufacturer Analysis",
                    data=export_manufacturer,
                    file_name="manufacturer_analysis.csv",
                    mime="text/csv",
                    help="Download comprehensive manufacturer metrics"
                )
        
        with col2:
            molecule_metrics = self.data_processor.calculate_market_metrics(
                df, 'Molecule', metric
            )
            if not molecule_metrics.empty:
                export_molecule = molecule_metrics.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Molecule Analysis",
                    data=export_molecule,
                    file_name="molecule_analysis.csv",
                    mime="text/csv",
                    help="Download detailed molecule performance metrics"
                )
        
        with col3:
            export_filtered = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Data",
                data=export_filtered,
                file_name="filtered_data.csv",
                mime="text/csv",
                help="Download the current filtered dataset"
            )


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        dashboard = PharmaAnalyticsDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or contact support if the issue persists.")