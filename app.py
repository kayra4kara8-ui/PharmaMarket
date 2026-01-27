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
import warnings
import io
import re
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

# ==============================================================================
# 3. CUSTOM CSS STYLING
# ==============================================================================
def apply_custom_styling():
    """Apply custom CSS styling for professional appearance"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1.5rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        position: relative;
        overflow: hidden;
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
    
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #4a5568;
        margin: 1.5rem 0 1rem 0;
        padding-left: 0.5rem;
        border-left: 4px solid #3498db;
    }
    
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
# 4. MAT DATA PROCESSING ENGINE
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
                # Read CSV with different encodings
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding, low_memory=False)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("❌ Unable to read CSV file with standard encodings")
                    return pd.DataFrame()
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
                if 'USD MNF' in col and 'Avg Price' not in col:
                    metrics['Sales_USD'] = col
                elif 'Standard Units' in col:
                    metrics['Standard_Units'] = col
                elif 'Units' in col and 'Standard' not in col:
                    metrics['Units'] = col
                elif 'SU Avg Price USD MNF' in col:
                    metrics['SU_Avg_Price'] = col
                elif 'Unit Avg Price USD MNF' in col:
                    metrics['Unit_Avg_Price'] = col
            
            # Create period DataFrame
            period_df = df[id_columns].copy()
            period_df['Period'] = period
            
            # Add metrics - FIXED: errors='coerce' yerine 'ignore' kullan
            for metric_name, col_name in metrics.items():
                # First convert to string and replace comma with dot
                if col_name in df.columns:
                    # Try to convert to numeric
                    try:
                        # Handle European decimal format (comma as decimal separator)
                        values = df[col_name].astype(str).str.replace(',', '.')
                        # Convert to numeric, coerce errors to NaN
                        period_df[metric_name] = pd.to_numeric(values, errors='coerce')
                    except Exception as e:
                        st.warning(f"Could not convert column {col_name}: {str(e)}")
                        period_df[metric_name] = 0
            
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
            
            # Fill NaN values
            numeric_cols = long_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                long_df[col] = long_df[col].fillna(0)
            
            return long_df
        else:
            return df
    
    @staticmethod
    def calculate_market_share(df: pd.DataFrame, 
                              group_column: str, 
                              period: str,
                              metric: str = 'Sales_USD') -> pd.DataFrame:
        """
        Calculate market share for specific period
        """
        period_data = df[df['Period'] == period]
        
        if len(period_data) == 0 or metric not in period_data.columns:
            return pd.DataFrame()
        
        # Group and calculate
        grouped = period_data.groupby(group_column)[metric].sum().reset_index()
        total = grouped[metric].sum()
        
        if total > 0:
            grouped['Market_Share_%'] = (grouped[metric] / total * 100).round(2)
        else:
            grouped['Market_Share_%'] = 0
        
        grouped['Rank'] = grouped[metric].rank(ascending=False, method='dense').astype(int)
        grouped = grouped.sort_values('Rank')
        
        return grouped

# ==============================================================================
# 5. VISUALIZATION ENGINE
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
        
        # Calculate metrics
        metrics_data = {}
        
        if 'Sales_USD' in latest_data.columns:
            total_sales = latest_data['Sales_USD'].sum()
            metrics_data['💰 MAT Sales'] = {
                'value': f"${total_sales/1e6:.1f}M" if total_sales > 1e6 else f"${total_sales/1e3:.1f}K",
                'period': latest_period
            }
        
        if 'Standard_Units' in latest_data.columns:
            total_units = latest_data['Standard_Units'].sum()
            metrics_data['📦 MAT Volume'] = {
                'value': f"{total_units/1e6:.1f}M" if total_units > 1e6 else f"{total_units/1e3:.1f}K",
                'period': latest_period
            }
        
        if 'SU_Avg_Price' in latest_data.columns:
            avg_price = latest_data['SU_Avg_Price'].mean()
            metrics_data['💵 Avg Price'] = {
                'value': f"${avg_price:.2f}",
                'period': latest_period
            }
        
        if 'Manufacturer' in latest_data.columns:
            manufacturers = latest_data['Manufacturer'].nunique()
            metrics_data['🏭 Manufacturers'] = {
                'value': f"{manufacturers:,}",
                'period': latest_period
            }
        
        # Display metrics
        cols = st.columns(len(metrics_data))
        for idx, (title, data) in enumerate(metrics_data.items()):
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
                              metric: str = 'Sales_USD') -> go.Figure:
        """
        Create MAT trend chart over periods
        """
        if metric not in df.columns:
            return None
        
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
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_title="MAT Period",
            yaxis_title=metric.replace('_', ' '),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_top_performers_chart(df: pd.DataFrame,
                                   period: str,
                                   metric: str = 'Sales_USD',
                                   group_by: str = 'Manufacturer',
                                   top_n: int = 10) -> go.Figure:
        """
        Create bar chart for top performers
        """
        if metric not in df.columns or group_by not in df.columns:
            return None
        
        period_data = df[df['Period'] == period]
        
        if len(period_data) == 0:
            return None
        
        # Get top performers
        top_data = period_data.groupby(group_by)[metric].sum().nlargest(top_n).reset_index()
        
        fig = px.bar(
            top_data,
            x=metric,
            y=group_by,
            orientation='h',
            title=f"Top {top_n} {group_by}s - {period}",
            color=metric,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title=metric.replace('_', ' '),
            yaxis_title=""
        )
        
        return fig

# ==============================================================================
# 6. DASHBOARD COMPONENTS
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
            </div>
            """, unsafe_allow_html=True)
            
            # Period selection
            if 'Period' in df.columns:
                periods = sorted(df['Period'].unique())
                default_period = periods[-1] if len(periods) > 0 else None
                
                st.markdown("#### 📅 MAT Period")
                selected_period = st.selectbox(
                    "Select MAT period",
                    options=periods,
                    index=len(periods)-1 if periods else 0,
                    label_visibility="collapsed"
                )
                filters['period'] = selected_period
            
            # Metric selection
            metric_options = []
            if 'Sales_USD' in df.columns:
                metric_options.append(('💰 Sales (USD MNF)', 'Sales_USD'))
            if 'Standard_Units' in df.columns:
                metric_options.append(('📦 Standard Units', 'Standard_Units'))
            
            if metric_options:
                st.markdown("#### 📊 Primary Metric")
                selected_metric = st.radio(
                    "",
                    options=[opt[1] for opt in metric_options],
                    format_func=lambda x: dict(metric_options)[x],
                    index=0,
                    label_visibility="collapsed"
                )
                filters['metric'] = selected_metric
            
            st.markdown("---")
            
            # Dataset info
            MATDashboardComponents._display_dataset_info(df)
        
        return filters
    
    @staticmethod
    def _display_dataset_info(df: pd.DataFrame):
        """Display dataset information"""
        st.markdown("#### 📊 Dataset Info")
        
        info_html = f"""
        <div style="background: rgba(52, 152, 219, 0.1); padding: 1rem; border-radius: 8px;">
            <p style="margin: 0.2rem 0;"><strong>Total Records:</strong> {len(df):,}</p>
            <p style="margin: 0.2rem 0;"><strong>MAT Periods:</strong> {df['Period'].nunique() if 'Period' in df.columns else 0}</p>
        """
        
        if 'Manufacturer' in df.columns:
            info_html += f'<p style="margin: 0.2rem 0;"><strong>Manufacturers:</strong> {df["Manufacturer"].nunique():,}</p>'
        
        if 'Molecule' in df.columns:
            info_html += f'<p style="margin: 0.2rem 0;"><strong>Molecules:</strong> {df["Molecule"].nunique():,}</p>'
        
        info_html += "</div>"
        st.markdown(info_html, unsafe_allow_html=True)
    
    @staticmethod
    def create_analysis_tabs() -> List:
        """
        Create analysis tabs
        """
        tab_titles = [
            "📊 Executive Summary",
            "🏭 Manufacturer Analysis",
            "🧪 Molecule Analysis",
            "📈 Trends",
            "📋 Data Explorer"
        ]
        
        return st.tabs(tab_titles)

# ==============================================================================
# 7. MAIN DASHBOARD APPLICATION
# ==============================================================================
class MATPharmaAnalyticsDashboard:
    """Main MAT dashboard application"""
    
    def __init__(self):
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
                
                if len(df) > 0:
                    tabs = self.components.create_analysis_tabs()
                    self._render_tabs(tabs, df, filters)
        else:
            self._display_welcome_screen()
    
    def _display_header(self):
        """Display header"""
        st.markdown("""
        <div class="main-header">
            Pharma MAT Intelligence Platform
        </div>
        <div style="text-align: center; color: #718096; margin-bottom: 2rem; font-size: 1.1rem;">
            MAT Data Analytics Dashboard • Version 3.0.0
        </div>
        """, unsafe_allow_html=True)
    
    def _display_file_uploader(self):
        """Display file uploader"""
        uploaded_file = st.file_uploader(
            "📁 Upload MAT Pharmaceutical Data",
            type=['xlsx', 'xls', 'csv'],
            help="Upload Excel or CSV files with MAT format data"
        )
        
        return uploaded_file
    
    def _display_welcome_screen(self):
        """Display welcome screen"""
        st.info("👆 Please upload a MAT format pharmaceutical data file to begin analysis")
        
        with st.expander("📋 MAT Data Format Guide"):
            st.markdown("""
            ### Expected MAT Data Format
            
            **Required MAT Period Columns:**
            ```
            "MAT Q3 2022 USD MNF"
            "MAT Q3 2022 Standard Units"
            "MAT Q3 2022 Units"
            "MAT Q3 2022 SU Avg Price USD MNF"
            "MAT Q3 2022 Unit Avg Price USD MNF"
            
            "MAT Q3 2023 USD MNF"
            "MAT Q3 2023 Standard Units"
            ... etc
            ```
            
            **Example Data:**
            ```
            Manufacturer: ABBOTT
            Molecule: PENICILLIN G
            MAT Q3 2022 USD MNF: 2265
            MAT Q3 2022 Standard Units: 7065
            MAT Q3 2023 USD MNF: 2270
            ```
            """)
    
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
                            
                            st.success(f"""
                            ✅ **MAT Data Successfully Processed!**
                            
                            - **Total Records:** {len(long_data):,}
                            - **MAT Periods:** {long_data['Period'].nunique()}
                            - **Latest Period:** {long_data['Period'].max()}
                            """)
                        else:
                            st.error("❌ Failed to transform MAT data")
                    else:
                        st.error("❌ No MAT period columns found")
                else:
                    st.error("❌ Failed to load data")
    
    def _render_tabs(self, tabs, df: pd.DataFrame, filters: Dict):
        """Render analysis tabs"""
        
        # Tab 1: Executive Summary
        with tabs[0]:
            self._render_executive_summary(df, filters)
        
        # Tab 2: Manufacturer Analysis
        with tabs[1]:
            self._render_manufacturer_analysis(df, filters)
        
        # Tab 3: Molecule Analysis
        with tabs[2]:
            self._render_molecule_analysis(df, filters)
        
        # Tab 4: Trends
        with tabs[3]:
            self._render_trend_analysis(df, filters)
        
        # Tab 5: Data Explorer
        with tabs[4]:
            self._render_data_explorer(df)
    
    def _render_executive_summary(self, df: pd.DataFrame, filters: Dict):
        """Render executive summary"""
        st.markdown('<div class="sub-header">📊 Executive Dashboard</div>', unsafe_allow_html=True)
        
        if 'period' in filters:
            latest_period = filters['period']
            
            # Display MAT metrics
            self.visualizer.create_mat_metrics(df, latest_period)
            
            st.markdown("---")
            
            # Top charts
            col1, col2 = st.columns(2)
            
            with col1:
                if 'metric' in filters:
                    # Top manufacturers
                    fig = self.visualizer.create_top_performers_chart(
                        df,
                        latest_period,
                        filters['metric'],
                        'Manufacturer',
                        10
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'metric' in filters:
                    # Top molecules
                    fig = self.visualizer.create_top_performers_chart(
                        df,
                        latest_period,
                        filters['metric'],
                        'Molecule',
                        10
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_manufacturer_analysis(self, df: pd.DataFrame, filters: Dict):
        """Render manufacturer analysis"""
        st.markdown('<div class="sub-header">🏭 Manufacturer Analysis</div>', unsafe_allow_html=True)
        
        if 'period' in filters and 'metric' in filters:
            period = filters['period']
            metric = filters['metric']
            
            # Calculate market share
            market_share = self.processor.calculate_market_share(
                df, 'Manufacturer', period, metric
            )
            
            if not market_share.empty:
                # Display top manufacturers table
                st.markdown(f"#### 📋 Manufacturer Rankings - {period}")
                
                display_df = market_share.head(20)[['Rank', 'Manufacturer', 'Market_Share_%', metric]]
                display_df.columns = ['Rank', 'Manufacturer', 'Market Share %', 'Value']
                
                # Format values
                display_df['Market Share %'] = display_df['Market Share %'].apply(lambda x: f'{x:.1f}%')
                display_df['Value'] = display_df['Value'].apply(lambda x: f'{x:,.0f}')
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
    
    def _render_molecule_analysis(self, df: pd.DataFrame, filters: Dict):
        """Render molecule analysis"""
        st.markdown('<div class="sub-header">🧪 Molecule Analysis</div>', unsafe_allow_html=True)
        
        if 'period' in filters and 'metric' in filters:
            period = filters['period']
            metric = filters['metric']
            
            # Calculate molecule performance
            molecule_data = self.processor.calculate_market_share(
                df, 'Molecule', period, metric
            )
            
            if not molecule_data.empty:
                # Display top molecules
                st.markdown(f"#### 🏆 Top Molecules - {period}")
                
                top_molecules = molecule_data.head(15)
                
                fig = px.bar(
                    top_molecules,
                    x='Molecule',
                    y=metric,
                    title=f"Top Molecules by {metric.replace('_', ' ')}",
                    color='Market_Share_%',
                    color_continuous_scale='Plasma'
                )
                
                fig.update_layout(height=500, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_trend_analysis(self, df: pd.DataFrame, filters: Dict):
        """Render trend analysis"""
        st.markdown('<div class="sub-header">📈 MAT Trend Analysis</div>', unsafe_allow_html=True)
        
        if 'metric' in filters:
            # Overall trend
            fig = self.visualizer.create_mat_trend_chart(df, filters['metric'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_data_explorer(self, df: pd.DataFrame):
        """Render data explorer"""
        st.markdown('<div class="sub-header">📋 Data Explorer</div>', unsafe_allow_html=True)
        
        # Data preview
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

# ==============================================================================
# 8. APPLICATION ENTRY POINT
# ==============================================================================
def main():
    """Main application entry point"""
    try:
        dashboard = MATPharmaAnalyticsDashboard()
        dashboard.run()
        
        # Add footer
        st.markdown("""
        <div class="footer">
            <p>MAT Pharma Intelligence Platform v3.0.0</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")

# ==============================================================================
# 9. RUN APPLICATION
# ==============================================================================
if __name__ == "__main__":
    main()
