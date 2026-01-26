"""
Pharma Analytics Dashboard - Streamlit Cloud Optimized
Author: Data Team
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Pharma Analytics",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_and_process_data(uploaded_file):
    """Load and process data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return pd.DataFrame()

def main():
    # Header
    st.markdown('<div class="main-header">💊 Pharmaceutical Sales Analytics</div>', 
                unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload your data (Excel or CSV)",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your pharmaceutical sales data"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a file to begin analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🌍 Geographic Analysis**
            - Country-level insights
            - Regional performance
            - Market distribution
            """)
        with col2:
            st.markdown("""
            **📈 Market Intelligence**
            - Manufacturer rankings
            - Molecule analysis
            - Growth trends
            """)
        with col3:
            st.markdown("""
            **💰 Price Analytics**
            - Price positioning
            - Volume analysis
            - Trend monitoring
            """)
        return
    
    # Load data
    df = load_and_process_data(uploaded_file)
    
    if df.empty:
        st.error("Failed to load data. Please check your file format.")
        return
    
    # Display basic info
    st.success(f"✅ Successfully loaded {len(df):,} records")
    
    # Filters
    with st.sidebar:
        st.header("🎛️ Filters")
        
        # Show available columns for filtering
        if 'Manufacturer' in df.columns:
            manufacturers = ['All'] + sorted(df['Manufacturer'].dropna().unique().tolist())
            selected_manufacturer = st.selectbox("🏭 Manufacturer", manufacturers)
        
        if 'Country' in df.columns:
            countries = ['All'] + sorted(df['Country'].dropna().unique().tolist())
            selected_country = st.selectbox("🌍 Country", countries)
        
        # Data info
        st.header("📊 Dataset Info")
        st.metric("Total Records", f"{len(df):,}")
        if 'Country' in df.columns:
            st.metric("Countries", df['Country'].nunique())
        if 'Manufacturer' in df.columns:
            st.metric("Manufacturers", df['Manufacturer'].nunique())
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview", 
        "📈 Analysis", 
        "📋 Data"
    ])
    
    with tab1:
        st.header("Dataset Overview")
        
        # Show first few rows
        st.dataframe(df.head(), use_container_width=True)
        
        # Column information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Non-Null Count': df.notnull().sum().values,
                'Data Type': df.dtypes.values
            })
            st.dataframe(col_info, use_container_width=True)
        
        with col2:
            st.subheader("Basic Statistics")
            if 'Sales_USD' in df.columns or 'USD' in df.columns:
                # Try to find sales column
                sales_cols = [col for col in df.columns if 'sales' in col.lower() or 'usd' in col.lower()]
                if sales_cols:
                    numeric_df = df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        st.dataframe(numeric_df.describe(), use_container_width=True)
    
    with tab2:
        st.header("Basic Analysis")
        
        # Try to create some visualizations
        if 'Manufacturer' in df.columns:
            # Manufacturer distribution
            manu_counts = df['Manufacturer'].value_counts().head(10)
            fig = px.bar(
                x=manu_counts.values,
                y=manu_counts.index,
                orientation='h',
                title="Top 10 Manufacturers",
                labels={'x': 'Count', 'y': 'Manufacturer'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if 'Country' in df.columns:
            # Country distribution
            country_counts = df['Country'].value_counts().head(15)
            fig = px.bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                title="Top 15 Countries",
                labels={'x': 'Count', 'y': 'Country'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Data Explorer")
        
        # Data filtering
        st.subheader("Filter Data")
        
        # Column selector for filtering
        filter_col = st.selectbox(
            "Select column to filter by",
            df.columns.tolist()
        )
        
        if filter_col in df.columns:
            unique_values = df[filter_col].dropna().unique().tolist()
            if len(unique_values) <= 20:  # Only show if reasonable number of values
                selected_values = st.multiselect(
                    f"Select {filter_col} values",
                    unique_values,
                    default=unique_values[:min(5, len(unique_values))]
                )
                
                if selected_values:
                    filtered_df = df[df[filter_col].isin(selected_values)]
                    st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
        
        # Data export
        st.subheader("Export Data")
        
        export_format = st.selectbox(
            "Select export format",
            ["CSV", "Excel"]
        )
        
        if st.button("📥 Download Data"):
            if export_format == "CSV":
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="pharma_data.csv",
                    mime="text/csv"
                )
            else:
                # For Excel, we need to use BytesIO
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name="pharma_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
