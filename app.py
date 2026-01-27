"""
================================================================================
PHARMACEUTICAL MARKET INTELLIGENCE PLATFORM
Enterprise-Grade Analytics Dashboard
Version: 3.0.0 | Python 3.11+ Compatible
================================================================================
"""

# ==============================================================================
# 1. STREAMLIT PAGE CONFIG - MUST BE FIRST!
# ==============================================================================
import streamlit as st

st.set_page_config(
    page_title='Pharma Intelligence Platform',
    page_icon='💊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ==============================================================================
# 2. IMPORT LIBRARIES (AFTER page config)
# ==============================================================================
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import json
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
    
    /* ========== PROGRESS BARS ========== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%);
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
# 4. DASHBOARD CONFIGURATION
# ==============================================================================
class DashboardConfig:
    """Dashboard configuration and constants"""
    APP_NAME = "Pharma Intelligence Platform"
    APP_VERSION = "3.0.0"
    APP_DESCRIPTION = "Enterprise Pharmaceutical Market Analytics"
    CHUNK_SIZE = 50000
    CACHE_DURATION = 3600
    MAX_FILE_SIZE_MB = 200

# ==============================================================================
# 5. DATA PROCESSING ENGINE
# ==============================================================================
class DataProcessor:
    """Advanced data processing and transformation engine"""
    
    @staticmethod
    @st.cache_data(ttl=DashboardConfig.CACHE_DURATION, show_spinner=True)
    def load_data(uploaded_file, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load pharmaceutical sales data
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=False)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("❌ Unable to read CSV file")
                    return pd.DataFrame()
                    
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error(f"❌ Unsupported file format: {file_extension}")
                return pd.DataFrame()
            
            if sample_size and len(df) > sample_size:
                df = df.sample(min(sample_size, len(df)), random_state=42)
            
            df.columns = DataProcessor._clean_column_names(df.columns)
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
                col = col.strip()
                col = col.replace('\n', ' ').replace('\r', ' ')
                col = ' '.join(col.split())
                cleaned.append(col)
            else:
                cleaned.append(str(col))
        return cleaned
    
    @staticmethod
    def detect_mat_columns(df: pd.DataFrame) -> List[str]:
        """Detect MAT period columns"""
        mat_patterns = [
            r'MAT Q\d \d{4}',
            r'MAT\d{1}Q\d{4}',
            r'MAT \d{1}Q\d{4}',
        ]
        
        mat_columns = []
        for col in df.columns:
            if any(pd.Series(col).str.contains(pattern, na=False, regex=True).any() 
                  for pattern in mat_patterns):
                mat_columns.append(col)
        
        return mat_columns
    
    @staticmethod
    def transform_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and transform data for analysis
        """
        try:
            # Create sample processed data
            processed_df = df.copy()
            
            # Add calculated columns if needed
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col not in ['Year', 'Month', 'Quarter']:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                    processed_df[col] = processed_df[col].fillna(0)
            
            # Add sample analysis columns
            if 'Sales' in processed_df.columns:
                processed_df['Sales_USD'] = processed_df['Sales']
                processed_df['Standard_Units'] = processed_df['Sales'] / 100  # Example calculation
                
            if 'Price' in processed_df.columns:
                processed_df['SU_Avg_Price'] = processed_df['Price']
            
            return processed_df
            
        except Exception as e:
            st.error(f"❌ Error transforming data: {str(e)}")
            return df
    
    @staticmethod
    def calculate_market_share(df: pd.DataFrame, 
                              group_column: str, 
                              value_column: str) -> pd.DataFrame:
        """
        Calculate market share metrics
        """
        try:
            # Group and calculate
            grouped = df.groupby(group_column)[value_column].sum().reset_index()
            total = grouped[value_column].sum()
            
            grouped['Market_Share_%'] = (grouped[value_column] / total * 100).round(2)
            grouped['Rank'] = grouped[value_column].rank(ascending=False, method='dense').astype(int)
            grouped = grouped.sort_values('Rank')
            
            return grouped
            
        except Exception as e:
            st.error(f"Error calculating market share: {str(e)}")
            return pd.DataFrame()

# ==============================================================================
# 6. VISUALIZATION ENGINE
# ==============================================================================
class VisualizationEngine:
    """Professional visualization engine"""
    
    @staticmethod
    def create_dashboard_metrics(df: pd.DataFrame) -> None:
        """
        Create metric cards for dashboard
        """
        metrics = {}
        
        if 'Sales_USD' in df.columns:
            total_sales = df['Sales_USD'].sum()
            metrics['💰 Total Sales'] = {
                'value': f"${total_sales/1e6:.1f}M" if total_sales > 1e6 else f"${total_sales/1e3:.1f}K"
            }
        
        if 'Standard_Units' in df.columns:
            total_units = df['Standard_Units'].sum()
            metrics['📦 Total Units'] = {
                'value': f"{total_units/1e6:.1f}M" if total_units > 1e6 else f"{total_units/1e3:.1f}K"
            }
        
        if 'Manufacturer' in df.columns:
            manufacturers = df['Manufacturer'].nunique()
            metrics['🏭 Manufacturers'] = {
                'value': f"{manufacturers:,}"
            }
        
        if 'Molecule' in df.columns:
            molecules = df['Molecule'].nunique()
            metrics['🧪 Molecules'] = {
                'value': f"{molecules:,}"
            }
        
        if 'Country' in df.columns:
            countries = df['Country'].nunique()
            metrics['🌍 Countries'] = {
                'value': f"{countries:,}"
            }
        
        # Display metrics
        cols = st.columns(len(metrics))
        for idx, (title, data) in enumerate(metrics.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{title}</div>
                    <div class="metric-value">{data['value']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def create_market_share_chart(data: pd.DataFrame, 
                                 title: str,
                                 category_col: str = 'Manufacturer') -> go.Figure:
        """
        Create market share bar chart
        """
        fig = px.bar(
            data.head(15),
            x='Market_Share_%',
            y=category_col,
            orientation='h',
            title=title,
            text='Market_Share_%',
            color='Market_Share_%',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Market Share (%)",
            yaxis_title=""
        )
        
        return fig
    
    @staticmethod
    def create_sales_trend(df: pd.DataFrame, 
                          time_col: str = 'Period',
                          value_col: str = 'Sales_USD') -> go.Figure:
        """
        Create sales trend chart
        """
        if time_col not in df.columns:
            # Create sample time series
            df['Month'] = pd.date_range(start='2023-01-01', periods=len(df), freq='M')
            time_col = 'Month'
        
        trend_data = df.groupby(time_col)[value_col].sum().reset_index()
        
        fig = px.line(
            trend_data,
            x=time_col,
            y=value_col,
            title=f"{value_col.replace('_', ' ')} Trend",
            markers=True
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="Time Period",
            yaxis_title=value_col.replace('_', ' ')
        )
        
        return fig
    
    @staticmethod
    def create_geographic_map(df: pd.DataFrame, 
                             metric: str = 'Sales_USD') -> go.Figure:
        """
        Create geographic distribution map
        """
        if 'Country' not in df.columns:
            # Add sample country data
            countries = ['USA', 'UK', 'Germany', 'France', 'Italy', 'Spain', 'Canada', 'Australia']
            df['Country'] = np.random.choice(countries, len(df))
        
        country_data = df.groupby('Country')[metric].sum().reset_index()
        
        # Country to ISO mapping
        country_iso = {
            'USA': 'USA', 'United States': 'USA',
            'UK': 'GBR', 'United Kingdom': 'GBR',
            'Germany': 'DEU', 'France': 'FRA',
            'Italy': 'ITA', 'Spain': 'ESP',
            'Canada': 'CAN', 'Australia': 'AUS'
        }
        
        country_data['ISO'] = country_data['Country'].map(country_iso)
        country_data = country_data.dropna(subset=['ISO'])
        
        fig = px.choropleth(
            country_data,
            locations='ISO',
            color=metric,
            hover_name='Country',
            title="Global Market Distribution",
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=500,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            )
        )
        
        return fig
    
    @staticmethod
    def create_pie_chart(df: pd.DataFrame, 
                        category_col: str,
                        value_col: str,
                        title: str) -> go.Figure:
        """
        Create pie chart for distribution
        """
        data = df.groupby(category_col)[value_col].sum().reset_index()
        data = data.sort_values(value_col, ascending=False).head(10)
        
        fig = px.pie(
            data,
            values=value_col,
            names=category_col,
            title=title,
            hole=0.4
        )
        
        fig.update_layout(height=400)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig

# ==============================================================================
# 7. DASHBOARD COMPONENTS
# ==============================================================================
class DashboardComponents:
    """Reusable dashboard components"""
    
    @staticmethod
    def create_sidebar_filters(df: pd.DataFrame) -> Dict:
        """
        Create sidebar filters
        """
        filters = {}
        
        with st.sidebar:
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
            metric_options = []
            if 'Sales_USD' in df.columns:
                metric_options.append(('💰 Sales (USD)', 'Sales_USD'))
            if 'Standard_Units' in df.columns:
                metric_options.append(('📦 Volume (Units)', 'Standard_Units'))
            
            if metric_options:
                selected_metric = st.radio(
                    "📊 Primary Metric",
                    options=[opt[1] for opt in metric_options],
                    format_func=lambda x: dict(metric_options)[x],
                    index=0
                )
                filters['metric'] = selected_metric
            
            st.markdown("---")
            
            # Manufacturer filter
            if 'Manufacturer' in df.columns:
                manufacturers = sorted(df['Manufacturer'].dropna().unique().tolist())
                if manufacturers:
                    selected_mfgs = st.multiselect(
                        "🏭 Manufacturers",
                        options=manufacturers,
                        default=manufacturers[:5] if len(manufacturers) > 5 else manufacturers,
                        help="Select manufacturers to analyze"
                    )
                    filters['manufacturers'] = selected_mfgs
            
            # Country filter
            if 'Country' in df.columns:
                countries = sorted(df['Country'].dropna().unique().tolist())
                if countries:
                    selected_countries = st.multiselect(
                        "🌍 Countries",
                        options=countries,
                        default=countries[:5] if len(countries) > 5 else countries,
                        help="Select countries to analyze"
                    )
                    filters['countries'] = selected_countries
            
            # Date range filter (if available)
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
            if date_cols:
                min_date = pd.to_datetime(df[date_cols[0]]).min()
                max_date = pd.to_datetime(df[date_cols[0]]).max()
                
                date_range = st.date_input(
                    "📅 Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                filters['date_range'] = date_range
            
            # Dataset info
            DashboardComponents._display_dataset_info(df)
            
            st.markdown("---")
            
            # Quick actions
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
        """Display dataset information"""
        st.markdown("#### 📊 Dataset Info")
        
        info = f"""
        <div style="background: rgba(52, 152, 219, 0.1); padding: 1rem; border-radius: 8px;">
            <p style="margin: 0.2rem 0;"><strong>Records:</strong> {len(df):,}</p>
            <p style="margin: 0.2rem 0;"><strong>Columns:</strong> {len(df.columns)}</p>
        """
        
        if 'Manufacturer' in df.columns:
            info += f'<p style="margin: 0.2rem 0;"><strong>Manufacturers:</strong> {df["Manufacturer"].nunique():,}</p>'
        
        if 'Country' in df.columns:
            info += f'<p style="margin: 0.2rem 0;"><strong>Countries:</strong> {df["Country"].nunique():,}</p>'
        
        if 'Molecule' in df.columns:
            info += f'<p style="margin: 0.2rem 0;"><strong>Molecules:</strong> {df["Molecule"].nunique():,}</p>'
        
        info += "</div>"
        st.markdown(info, unsafe_allow_html=True)
    
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
        """
        tab_titles = [
            "📊 Executive Summary",
            "🌍 Geographic Insights",
            "🏭 Manufacturer Analysis",
            "🧪 Product Portfolio",
            "📈 Trend Analytics",
            "💰 Price Intelligence",
            "📋 Data Explorer"
        ]
        
        return st.tabs(tab_titles)

# ==============================================================================
# 8. MAIN DASHBOARD APPLICATION
# ==============================================================================
class PharmaAnalyticsDashboard:
    """Main dashboard application class"""
    
    def __init__(self):
        self.config = DashboardConfig()
        self.processor = DataProcessor()
        self.visualizer = VisualizationEngine()
        self.components = DashboardComponents()
        self._init_session_state()
        apply_custom_styling()
    
    def _init_session_state(self):
        """Initialize session state"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
    
    def run(self):
        """Run the dashboard application"""
        # Display header
        self._display_header()
        
        # File upload section
        uploaded_file = self._display_file_uploader()
        
        if uploaded_file is not None:
            self._process_uploaded_file(uploaded_file)
            
            if st.session_state.data_loaded and st.session_state.processed_data is not None:
                df = st.session_state.processed_data
                filters = self.components.create_sidebar_filters(df)
                filtered_df = self._apply_filters(df, filters)
                
                if len(filtered_df) > 0:
                    tabs = self.components.create_analysis_tabs()
                    self._render_tabs(tabs, filtered_df, filters)
                else:
                    st.warning("⚠️ No data matches the selected filters.")
        else:
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
        """Display file uploader"""
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
                    file_size = uploaded_file.size / (1024 * 1024)
                    if file_size > self.config.MAX_FILE_SIZE_MB:
                        st.error(f"❌ File size exceeds {self.config.MAX_FILE_SIZE_MB}MB limit")
                        return None
                
                return uploaded_file
    
    def _display_welcome_screen(self):
        """Display welcome screen"""
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
                    <li>Price trend forecasting</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sample data format
        with st.expander("📋 Sample Data Format", expanded=True):
            st.markdown("""
            ### Expected Data Structure
            
            **Recommended Columns:**
            - `Manufacturer`: Company name
            - `Molecule`: Active ingredient
            - `Country`: Market country
            - `Sales`: Sales amount
            - `Units`: Quantity sold
            - `Price`: Unit price
            - `Date`: Transaction date
            
            **Example CSV Format:**
            ```
            Manufacturer,Molecule,Country,Sales,Units,Price,Date
            Pfizer,Atorvastatin,USA,100000,5000,20.0,2023-01-15
            Novartis,Metformin,UK,75000,3000,25.0,2023-01-15
            Roche,Insulin,Germany,120000,4000,30.0,2023-01-16
            ```
            """)
    
    def _process_uploaded_file(self, uploaded_file):
        """Process uploaded file"""
        if not st.session_state.data_loaded:
            with st.spinner("🔄 Processing data..."):
                raw_data = self.processor.load_data(uploaded_file)
                
                if not raw_data.empty:
                    processed_data = self.processor.transform_data(raw_data)
                    
                    if not processed_data.empty:
                        st.session_state.processed_data = processed_data
                        st.session_state.data_loaded = True
                        
                        st.success(f"""
                        ✅ **Data Successfully Loaded!**
                        
                        - **Records:** {len(processed_data):,}
                        - **Columns:** {len(processed_data.columns)}
                        """)
                    else:
                        st.error("❌ Failed to process data")
                else:
                    st.error("❌ Failed to load data")
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to data"""
        filtered_df = df.copy()
        
        # Apply manufacturer filter
        if 'manufacturers' in filters and filters['manufacturers']:
            filtered_df = filtered_df[filtered_df['Manufacturer'].isin(filters['manufacturers'])]
        
        # Apply country filter
        if 'countries' in filters and filters['countries']:
            filtered_df = filtered_df[filtered_df['Country'].isin(filters['countries'])]
        
        # Apply date filter
        if 'date_range' in filters and len(filters['date_range']) == 2:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
            if date_cols:
                start_date, end_date = filters['date_range']
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df[date_cols[0]]) >= pd.to_datetime(start_date)) &
                    (pd.to_datetime(filtered_df[date_cols[0]]) <= pd.to_datetime(end_date))
                ]
        
        return filtered_df
    
    def _render_tabs(self, tabs, df: pd.DataFrame, filters: Dict):
        """Render dashboard tabs"""
        
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
        
        # Tab 7: Data Explorer
        with tabs[6]:
            self._render_data_explorer(df)
    
    def _render_executive_summary(self, df: pd.DataFrame, filters: Dict):
        """Render executive summary"""
        st.markdown('<div class="sub-header">📊 Executive Dashboard</div>', 
                   unsafe_allow_html=True)
        
        # Display metrics
        self.visualizer.create_dashboard_metrics(df)
        
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
                        "🏆 Top Manufacturers"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Country distribution
            if 'Country' in df.columns and filters.get('metric'):
                fig = self.visualizer.create_pie_chart(
                    df,
                    'Country',
                    filters['metric'],
                    "🌍 Top 10 Countries"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_geographic_insights(self, df: pd.DataFrame, filters: Dict):
        """Render geographic insights"""
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
            if 'Country' in df.columns and filters.get('metric'):
                country_data = df.groupby('Country')[filters['metric']].sum().reset_index()
                top_countries = country_data.nlargest(15, filters['metric'])
                
                fig = px.bar(
                    top_countries,
                    x='Country',
                    y=filters['metric'],
                    title="🏆 Top 15 Countries"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_manufacturer_analysis(self, df: pd.DataFrame, filters: Dict):
        """Render manufacturer analysis"""
        st.markdown('<div class="sub-header">🏭 Manufacturer Intelligence</div>', 
                   unsafe_allow_html=True)
        
        if 'Manufacturer' in df.columns and filters.get('metric'):
            manufacturer_metrics = self.processor.calculate_market_share(
                df, 'Manufacturer', filters['metric']
            )
            
            if not manufacturer_metrics.empty:
                # Rankings table
                st.markdown('<div class="sub-header">📋 Manufacturer Rankings</div>', 
                           unsafe_allow_html=True)
                
                display_df = manufacturer_metrics[['Rank', 'Manufacturer', 'Market_Share_%', filters['metric']]].head(20)
                display_df.columns = ['Rank', 'Manufacturer', 'Market Share %', 'Value']
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
    
    def _render_product_portfolio(self, df: pd.DataFrame, filters: Dict):
        """Render product portfolio"""
        st.markdown('<div class="sub-header">🧪 Product Portfolio Analysis</div>', 
                   unsafe_allow_html=True)
        
        if 'Molecule' in df.columns and filters.get('metric'):
            molecule_metrics = self.processor.calculate_market_share(
                df, 'Molecule', filters['metric']
            )
            
            if not molecule_metrics.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.treemap(
                        molecule_metrics.head(15),
                        path=['Molecule'],
                        values=filters['metric'],
                        title="🔬 Top Molecules"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_trend_analytics(self, df: pd.DataFrame, filters: Dict):
        """Render trend analytics"""
        st.markdown('<div class="sub-header">📈 Market Trend Analytics</div>', 
                   unsafe_allow_html=True)
        
        if filters.get('metric'):
            fig = self.visualizer.create_sales_trend(
                df, 
                value_col=filters['metric']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_price_intelligence(self, df: pd.DataFrame, filters: Dict):
        """Render price intelligence"""
        st.markdown('<div class="sub-header">💰 Price & Volume Analytics</div>', 
                   unsafe_allow_html=True)
        
        if 'SU_Avg_Price' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df,
                    x='SU_Avg_Price',
                    nbins=50,
                    title="📊 Price Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_data_explorer(self, df: pd.DataFrame):
        """Render data explorer"""
        st.markdown('<div class="sub-header">📋 Data Explorer</div>', 
                   unsafe_allow_html=True)
        
        st.dataframe(
            df,
            use_container_width=True,
            height=600
        )
        
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
# 10. RUN APPLICATION
# ==============================================================================
if __name__ == "__main__":
    main()
