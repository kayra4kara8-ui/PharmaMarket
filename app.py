# app.py - 4000+ lines of production-grade pharmaceutical analytics application
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.colors import qualitative
import datetime
import io
import warnings
import re
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pharma Commercial Analytics Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3d8c;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c5282;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f7f9fc 0%, #edf2f7 100%);
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #3182ce;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .trend-up { color: #10b981; font-weight: bold; }
    .trend-down { color: #ef4444; font-weight: bold; }
    .trend-neutral { color: #6b7280; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f4f8;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e2e8f0;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3182ce;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class PharmaDataProcessor:
    """Professional data processor for pharmaceutical commercial data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.year_columns = {
            '2022': ['MAT Q3 2022 USD MNF', 'MAT Q3 2022 Standard Units', 
                    'MAT Q3 2022 Units', 'MAT Q3 2022 SU Avg Price USD MNF',
                    'MAT Q3 2022 Unit Avg Price USD MNF'],
            '2023': ['MAT Q3 2023 USD MNF', 'MAT Q3 2023 Standard Units',
                    'MAT Q3 2023 Units', 'MAT Q3 2023 SU Avg Price USD MNF',
                    'MAT Q3 2023 Unit Avg Price USD MNF'],
            '2024': ['MAT Q3 2024 USD MNF', 'MAT Q3 2024 Standard Units',
                    'MAT Q3 2024 Units', 'MAT Q3 2024 SU Avg Price USD MNF',
                    'MAT Q3 2024 Unit Avg Price USD MNF']
        }
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Clean and prepare data with robust error handling"""
        # Convert numeric columns
        for year in ['2022', '2023', '2024']:
            for col in self.year_columns[year]:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Fill missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
        
        # Clean string columns
        str_cols = self.df.select_dtypes(include=[object]).columns
        for col in str_cols:
            self.df[col] = self.df[col].fillna('Unknown').str.strip()
        
        # Add calculated columns
        self._add_calculated_metrics()
    
    def _add_calculated_metrics(self):
        """Add growth metrics and trend classifications"""
        # USD MNF Growth
        self.df['USD_MNF_Growth_22_23'] = self._safe_divide(
            self.df['MAT Q3 2023 USD MNF'] - self.df['MAT Q3 2022 USD MNF'],
            self.df['MAT Q3 2022 USD MNF']
        ) * 100
        
        self.df['USD_MNF_Growth_23_24'] = self._safe_divide(
            self.df['MAT Q3 2024 USD MNF'] - self.df['MAT Q3 2023 USD MNF'],
            self.df['MAT Q3 2023 USD MNF']
        ) * 100
        
        self.df['USD_MNF_Growth_22_24'] = self._safe_divide(
            self.df['MAT Q3 2024 USD MNF'] - self.df['MAT Q3 2022 USD MNF'],
            self.df['MAT Q3 2022 USD MNF']
        ) * 100
        
        # Units Growth
        self.df['Units_Growth_22_23'] = self._safe_divide(
            self.df['MAT Q3 2023 Units'] - self.df['MAT Q3 2022 Units'],
            self.df['MAT Q3 2022 Units']
        ) * 100
        
        self.df['Units_Growth_23_24'] = self._safe_divide(
            self.df['MAT Q3 2024 Units'] - self.df['MAT Q3 2023 Units'],
            self.df['MAT Q3 2023 Units']
        ) * 100
        
        self.df['Units_Growth_22_24'] = self._safe_divide(
            self.df['MAT Q3 2024 Units'] - self.df['MAT Q3 2022 Units'],
            self.df['MAT Q3 2022 Units']
        ) * 100
        
        # Standard Units Growth
        self.df['SU_Growth_22_23'] = self._safe_divide(
            self.df['MAT Q3 2023 Standard Units'] - self.df['MAT Q3 2022 Standard Units'],
            self.df['MAT Q3 2022 Standard Units']
        ) * 100
        
        self.df['SU_Growth_23_24'] = self._safe_divide(
            self.df['MAT Q3 2024 Standard Units'] - self.df['MAT Q3 2023 Standard Units'],
            self.df['MAT Q3 2023 Standard Units']
        ) * 100
        
        self.df['SU_Growth_22_24'] = self._safe_divide(
            self.df['MAT Q3 2024 Standard Units'] - self.df['MAT Q3 2022 Standard Units'],
            self.df['MAT Q3 2022 Standard Units']
        ) * 100
        
        # Price Change
        self.df['SU_Price_Change_22_23'] = self._safe_divide(
            self.df['MAT Q3 2023 SU Avg Price USD MNF'] - self.df['MAT Q3 2022 SU Avg Price USD MNF'],
            self.df['MAT Q3 2022 SU Avg Price USD MNF']
        ) * 100
        
        self.df['SU_Price_Change_23_24'] = self._safe_divide(
            self.df['MAT Q3 2024 SU Avg Price USD MNF'] - self.df['MAT Q3 2023 SU Avg Price USD MNF'],
            self.df['MAT Q3 2023 SU Avg Price USD MNF']
        ) * 100
        
        self.df['SU_Price_Change_22_24'] = self._safe_divide(
            self.df['MAT Q3 2024 SU Avg Price USD MNF'] - self.df['MAT Q3 2022 SU Avg Price USD MNF'],
            self.df['MAT Q3 2022 SU Avg Price USD MNF']
        ) * 100
        
        self.df['Unit_Price_Change_22_23'] = self._safe_divide(
            self.df['MAT Q3 2023 Unit Avg Price USD MNF'] - self.df['MAT Q3 2022 Unit Avg Price USD MNF'],
            self.df['MAT Q3 2022 Unit Avg Price USD MNF']
        ) * 100
        
        self.df['Unit_Price_Change_23_24'] = self._safe_divide(
            self.df['MAT Q3 2024 Unit Avg Price USD MNF'] - self.df['MAT Q3 2023 Unit Avg Price USD MNF'],
            self.df['MAT Q3 2023 Unit Avg Price USD MNF']
        ) * 100
        
        self.df['Unit_Price_Change_22_24'] = self._safe_divide(
            self.df['MAT Q3 2024 Unit Avg Price USD MNF'] - self.df['MAT Q3 2022 Unit Avg Price USD MNF'],
            self.df['MAT Q3 2022 Unit Avg Price USD MNF']
        ) * 100
        
        # Trend Classification
        self.df['USD_MNF_Trend'] = self.df.apply(
            lambda x: self._classify_trend(
                x['USD_MNF_Growth_22_23'],
                x['USD_MNF_Growth_23_24']
            ), axis=1
        )
        
        self.df['Units_Trend'] = self.df.apply(
            lambda x: self._classify_trend(
                x['Units_Growth_22_23'],
                x['Units_Growth_23_24']
            ), axis=1
        )
        
        self.df['Price_Trend'] = self.df.apply(
            lambda x: self._classify_trend(
                x['SU_Price_Change_22_23'],
                x['SU_Price_Change_23_24']
            ), axis=1
        )
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """Safe division with zero handling"""
        if denominator == 0:
            return 0
        return numerator / denominator
    
    def _classify_trend(self, growth_22_23: float, growth_23_24: float) -> str:
        """Classify 3-year trend"""
        if growth_22_23 > 5 and growth_23_24 > 5:
            return 'A Artan'
        elif growth_22_23 < -5 and growth_23_24 < -5:
            return 'B Azalan'
        elif abs(growth_22_23) < 5 and abs(growth_23_24) < 5:
            return 'D Stabil'
        else:
            return 'C Dalgalı'
    
    def get_filtered_data(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to data"""
        df_filtered = self.df.copy()
        
        if filters.get('country') and filters['country'] != 'Tümü':
            df_filtered = df_filtered[df_filtered['Country'] == filters['country']]
        
        if filters.get('corporation') and filters['corporation'] != 'Tümü':
            df_filtered = df_filtered[df_filtered['Corporation'] == filters['corporation']]
        
        if filters.get('molecule') and filters['molecule'] != 'Tümü':
            df_filtered = df_filtered[df_filtered['Molecule'] == filters['molecule']]
        
        if filters.get('sector') and filters['sector'] != 'Tümü':
            df_filtered = df_filtered[df_filtered['Sector'] == filters['sector']]
        
        if filters.get('panel') and filters['panel'] != 'Tümü':
            df_filtered = df_filtered[df_filtered['Panel'] == filters['panel']]
        
        if filters.get('specialty') and filters['specialty'] != 'Tümü':
            if filters['specialty'] == 'Specialty':
                df_filtered = df_filtered[df_filtered['Specialty Product'] == 'Yes']
            else:
                df_filtered = df_filtered[df_filtered['Specialty Product'] != 'Yes']
        
        return df_filtered

class AnalyticsVisualizer:
    """Professional visualization engine for pharmaceutical analytics"""
    
    def __init__(self, data_processor: PharmaDataProcessor):
        self.dp = data_processor
        self.colors = {
            '2022': '#3182ce',
            '2023': '#10b981',
            '2024': '#8b5cf6'
        }
    
    def create_executive_summary_metrics(self, df: pd.DataFrame) -> None:
        """Create executive summary metric cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_2022 = df['MAT Q3 2022 USD MNF'].sum()
            total_2023 = df['MAT Q3 2023 USD MNF'].sum()
            total_2024 = df['MAT Q3 2024 USD MNF'].sum()
            growth_22_24 = ((total_2024 - total_2022) / total_2022 * 100) if total_2022 != 0 else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Global USD MNF</h3>
                <div style="font-size: 2rem; font-weight: bold;">${total_2024:,.0f}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <span>2022: ${total_2022:,.0f}</span>
                    <span>2023: ${total_2023:,.0f}</span>
                </div>
                <div class="{'trend-up' if growth_22_24 > 0 else 'trend-down' if growth_22_24 < 0 else 'trend-neutral'}">
                    3-Yıl Trend: {growth_22_24:+.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            units_2022 = df['MAT Q3 2022 Units'].sum()
            units_2023 = df['MAT Q3 2023 Units'].sum()
            units_2024 = df['MAT Q3 2024 Units'].sum()
            units_growth_22_24 = ((units_2024 - units_2022) / units_2022 * 100) if units_2022 != 0 else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Units</h3>
                <div style="font-size: 2rem; font-weight: bold;">{units_2024:,.0f}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <span>2022: {units_2022:,.0f}</span>
                    <span>2023: {units_2023:,.0f}</span>
                </div>
                <div class="{'trend-up' if units_growth_22_24 > 0 else 'trend-down' if units_growth_22_24 < 0 else 'trend-neutral'}">
                    3-Yıl Trend: {units_growth_22_24:+.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            su_2022 = df['MAT Q3 2022 Standard Units'].sum()
            su_2023 = df['MAT Q3 2023 Standard Units'].sum()
            su_2024 = df['MAT Q3 2024 Standard Units'].sum()
            su_growth_22_24 = ((su_2024 - su_2022) / su_2022 * 100) if su_2022 != 0 else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Standard Units</h3>
                <div style="font-size: 2rem; font-weight: bold;">{su_2024:,.0f}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <span>2022: {su_2022:,.0f}</span>
                    <span>2023: {su_2023:,.0f}</span>
                </div>
                <div class="{'trend-up' if su_growth_22_24 > 0 else 'trend-down' if su_growth_22_24 < 0 else 'trend-neutral'}">
                    3-Yıl Trend: {su_growth_22_24:+.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_price_2022 = df['MAT Q3 2022 SU Avg Price USD MNF'].mean()
            avg_price_2023 = df['MAT Q3 2023 SU Avg Price USD MNF'].mean()
            avg_price_2024 = df['MAT Q3 2024 SU Avg Price USD MNF'].mean()
            price_change_22_24 = ((avg_price_2024 - avg_price_2022) / avg_price_2022 * 100) if avg_price_2022 != 0 else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Price Per SU</h3>
                <div style="font-size: 2rem; font-weight: bold;">${avg_price_2024:.2f}</div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <span>2022: ${avg_price_2022:.2f}</span>
                    <span>2023: ${avg_price_2023:.2f}</span>
                </div>
                <div class="{'trend-up' if price_change_22_24 > 0 else 'trend-down' if price_change_22_24 < 0 else 'trend-neutral'}">
                    3-Yıl Trend: {price_change_22_24:+.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def create_three_year_trend_chart(self, df: pd.DataFrame, metric: str, title: str) -> go.Figure:
        """Create 3-year trend chart"""
        years = ['2022', '2023', '2024']
        values = []
        
        for year in years:
            col_name = f'MAT Q3 {year} {metric}'
            if metric == 'USD MNF':
                col_name = f'MAT Q3 {year} USD MNF'
            elif metric == 'Standard Units':
                col_name = f'MAT Q3 {year} Standard Units'
            elif metric == 'Units':
                col_name = f'MAT Q3 {year} Units'
            
            if col_name in df.columns:
                values.append(df[col_name].sum())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers+text',
            line=dict(color='#3182ce', width=4),
            marker=dict(size=12, color='white', line=dict(width=2, color='#3182ce')),
            text=[f'${v:,.0f}' if 'USD' in metric else f'{v:,.0f}' for v in values],
            textposition='top center',
            textfont=dict(size=12, color='#1e3d8c'),
            name=metric
        ))
        
        # Add growth annotations
        if len(values) == 3:
            growth_22_23 = ((values[1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            growth_23_24 = ((values[2] - values[1]) / values[1] * 100) if values[1] != 0 else 0
            growth_22_24 = ((values[2] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            
            fig.add_annotation(
                x=1, y=values[1],
                text=f'22→23: {growth_22_23:+.1f}%',
                showarrow=False,
                yshift=20,
                font=dict(color='#10b981' if growth_22_23 > 0 else '#ef4444', size=10)
            )
            
            fig.add_annotation(
                x=2, y=values[2],
                text=f'23→24: {growth_23_24:+.1f}%',
                showarrow=False,
                yshift=20,
                font=dict(color='#10b981' if growth_23_24 > 0 else '#ef4444', size=10)
            )
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color='#1e3d8c'),
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title='Year',
                gridcolor='#e2e8f0',
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title=metric,
                gridcolor='#e2e8f0',
                tickfont=dict(size=12)
            ),
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_growth_comparison_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create growth comparison chart for 3-year periods"""
        # Calculate growth metrics by country
        country_growth = df.groupby('Country').agg({
            'USD_MNF_Growth_22_24': 'mean',
            'Units_Growth_22_24': 'mean',
            'MAT Q3 2024 USD MNF': 'sum'
        }).reset_index()
        
        country_growth = country_growth.sort_values('USD_MNF_Growth_22_24', ascending=False).head(10)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=country_growth['Country'],
            x=country_growth['USD_MNF_Growth_22_24'],
            orientation='h',
            marker=dict(
                color=country_growth['USD_MNF_Growth_22_24'].apply(
                    lambda x: '#10b981' if x > 0 else '#ef4444' if x < 0 else '#6b7280'
                ),
                line=dict(width=0)
            ),
            text=[f'{x:+.1f}%' for x in country_growth['USD_MNF_Growth_22_24']],
            textposition='outside',
            name='USD MNF Growth (22→24)'
        ))
        
        fig.update_layout(
            title=dict(
                text='Top 10 Countries by 3-Year USD MNF Growth',
                font=dict(size=18, color='#1e3d8c'),
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title='Growth % (2022→2024)',
                gridcolor='#e2e8f0'
            ),
            yaxis=dict(
                title='Country',
                categoryorder='total ascending'
            ),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False
        )
        
        return fig
    
    def create_molecule_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create molecule-level analysis with 3-year trends"""
        # Aggregate by molecule
        molecule_data = df.groupby('Molecule').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2023 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean',
            'USD_MNF_Trend': 'first'
        }).reset_index()
        
        # Calculate growth rates
        molecule_data['Growth_22_24'] = ((molecule_data['MAT Q3 2024 USD MNF'] - molecule_data['MAT Q3 2022 USD MNF']) / 
                                        molecule_data['MAT Q3 2022 USD MNF'].replace(0, np.nan)) * 100
        
        # Get top 15 molecules by 2024 sales
        top_molecules = molecule_data.nlargest(15, 'MAT Q3 2024 USD MNF')
        
        # Create grouped bar chart
        fig = go.Figure()
        
        years = ['2022', '2023', '2024']
        colors = ['#3182ce', '#10b981', '#8b5cf6']
        
        for year, color in zip(years, colors):
            fig.add_trace(go.Bar(
                x=top_molecules['Molecule'],
                y=top_molecules[f'MAT Q3 {year} USD MNF'],
                name=year,
                marker_color=color,
                text=[f'${x:,.0f}' for x in top_molecules[f'MAT Q3 {year} USD MNF']],
                textposition='auto',
                textfont=dict(size=9)
            ))
        
        fig.update_layout(
            title=dict(
                text='Top 15 Molecules - 3-Year USD MNF Trend',
                font=dict(size=18, color='#1e3d8c'),
                x=0.5
            ),
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title='Molecule',
                tickangle=45,
                gridcolor='#e2e8f0'
            ),
            yaxis=dict(
                title='USD MNF',
                gridcolor='#e2e8f0',
                tickprefix='$'
            ),
            height=500,
            margin=dict(l=50, r=50, t=80, b=150),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    def create_corporation_market_share(self, df: pd.DataFrame) -> go.Figure:
        """Create corporation market share analysis with 3-year trends"""
        # Calculate market share by year
        corporations = df.groupby('Corporation').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum'
        }).reset_index()
        
        # Calculate total sales each year
        total_2022 = corporations['MAT Q3 2022 USD MNF'].sum()
        total_2023 = corporations['MAT Q3 2023 USD MNF'].sum()
        total_2024 = corporations['MAT Q3 2024 USD MNF'].sum()
        
        # Calculate market shares
        corporations['Share_2022'] = (corporations['MAT Q3 2022 USD MNF'] / total_2022 * 100) if total_2022 != 0 else 0
        corporations['Share_2023'] = (corporations['MAT Q3 2023 USD MNF'] / total_2023 * 100) if total_2023 != 0 else 0
        corporations['Share_2024'] = (corporations['MAT Q3 2024 USD MNF'] / total_2024 * 100) if total_2024 != 0 else 0
        
        # Calculate share change
        corporations['Share_Change_22_24'] = corporations['Share_2024'] - corporations['Share_2022']
        
        # Get top 10 corporations by 2024 share
        top_corps = corporations.nlargest(10, 'Share_2024')
        
        # Create stacked area chart for share evolution
        fig = go.Figure()
        
        for _, row in top_corps.iterrows():
            fig.add_trace(go.Scatter(
                x=['2022', '2023', '2024'],
                y=[row['Share_2022'], row['Share_2023'], row['Share_2024']],
                mode='lines+markers',
                name=row['Corporation'],
                stackgroup='one',
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=dict(
                text='Top 10 Corporations - Market Share Evolution (2022-2024)',
                font=dict(size=18, color='#1e3d8c'),
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title='Year',
                gridcolor='#e2e8f0'
            ),
            yaxis=dict(
                title='Market Share %',
                gridcolor='#e2e8f0',
                ticksuffix='%'
            ),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='x unified'
        )
        
        return fig
    
    def create_specialty_vs_non_specialty(self, df: pd.DataFrame) -> go.Figure:
        """Create specialty vs non-specialty analysis"""
        # Classify products
        df['Product_Type'] = df['Specialty Product'].apply(
            lambda x: 'Specialty' if x == 'Yes' else 'Non-Specialty'
        )
        
        # Aggregate by product type and year
        specialty_data = df.groupby('Product_Type').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 Units': 'sum',
            'MAT Q3 2023 Units': 'sum',
            'MAT Q3 2024 Units': 'sum',
            'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2023 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
        }).reset_index()
        
        # Create subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('USD MNF Trend', 'Units Trend', 
                          'Avg Price Trend', 'Market Share Evolution'),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. USD MNF Trend
        for product_type in ['Specialty', 'Non-Specialty']:
            data = specialty_data[specialty_data['Product_Type'] == product_type]
            if not data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=['2022', '2023', '2024'],
                        y=[data['MAT Q3 2022 USD MNF'].iloc[0], 
                          data['MAT Q3 2023 USD MNF'].iloc[0],
                          data['MAT Q3 2024 USD MNF'].iloc[0]],
                        mode='lines+markers',
                        name=product_type,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )
        
        # 2. Units Trend
        for product_type in ['Specialty', 'Non-Specialty']:
            data = specialty_data[specialty_data['Product_Type'] == product_type]
            if not data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=['2022', '2023', '2024'],
                        y=[data['MAT Q3 2022 Units'].iloc[0], 
                          data['MAT Q3 2023 Units'].iloc[0],
                          data['MAT Q3 2024 Units'].iloc[0]],
                        mode='lines+markers',
                        name=product_type,
                        line=dict(width=3),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. Avg Price Trend
        for product_type in ['Specialty', 'Non-Specialty']:
            data = specialty_data[specialty_data['Product_Type'] == product_type]
            if not data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=['2022', '2023', '2024'],
                        y=[data['MAT Q3 2022 SU Avg Price USD MNF'].iloc[0], 
                          data['MAT Q3 2023 SU Avg Price USD MNF'].iloc[0],
                          data['MAT Q3 2024 SU Avg Price USD MNF'].iloc[0]],
                        mode='lines+markers',
                        name=product_type,
                        line=dict(width=3),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Market Share Evolution
        total_by_year = {
            '2022': specialty_data['MAT Q3 2022 USD MNF'].sum(),
            '2023': specialty_data['MAT Q3 2023 USD MNF'].sum(),
            '2024': specialty_data['MAT Q3 2024 USD MNF'].sum()
        }
        
        for product_type in ['Specialty', 'Non-Specialty']:
            data = specialty_data[specialty_data['Product_Type'] == product_type]
            if not data.empty:
                shares = [
                    (data['MAT Q3 2022 USD MNF'].iloc[0] / total_by_year['2022'] * 100) if total_by_year['2022'] != 0 else 0,
                    (data['MAT Q3 2023 USD MNF'].iloc[0] / total_by_year['2023'] * 100) if total_by_year['2023'] != 0 else 0,
                    (data['MAT Q3 2024 USD MNF'].iloc[0] / total_by_year['2024'] * 100) if total_by_year['2024'] != 0 else 0
                ]
                
                fig.add_trace(
                    go.Scatter(
                        x=['2022', '2023', '2024'],
                        y=shares,
                        mode='lines+markers',
                        name=product_type,
                        line=dict(width=3),
                        marker=dict(size=8),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=dict(
                text='Specialty vs Non-Specialty Products - 3-Year Analysis',
                font=dict(size=20, color='#1e3d8c'),
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=700,
            margin=dict(l=50, r=50, t=100, b=50),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(gridcolor='#e2e8f0')
        fig.update_yaxes(gridcolor='#e2e8f0')
        fig.update_yaxes(title_text="USD MNF", row=1, col=1)
        fig.update_yaxes(title_text="Units", row=1, col=2)
        fig.update_yaxes(title_text="Avg Price", row=2, col=1)
        fig.update_yaxes(title_text="Market Share %", row=2, col=2)
        
        return fig
    
    def create_price_volume_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create price-volume analysis with 3-year trends"""
        # Calculate year-over-year changes
        total_2022_usd = df['MAT Q3 2022 USD MNF'].sum()
        total_2023_usd = df['MAT Q3 2023 USD MNF'].sum()
        total_2024_usd = df['MAT Q3 2024 USD MNF'].sum()
        
        total_2022_units = df['MAT Q3 2022 Units'].sum()
        total_2023_units = df['MAT Q3 2023 Units'].sum()
        total_2024_units = df['MAT Q3 2024 Units'].sum()
        
        avg_price_2022 = df['MAT Q3 2022 Unit Avg Price USD MNF'].mean()
        avg_price_2023 = df['MAT Q3 2023 Unit Avg Price USD MNF'].mean()
        avg_price_2024 = df['MAT Q3 2024 Unit Avg Price USD MNF'].mean()
        
        # Calculate growth components
        usd_growth_22_23 = ((total_2023_usd - total_2022_usd) / total_2022_usd * 100) if total_2022_usd != 0 else 0
        usd_growth_23_24 = ((total_2024_usd - total_2023_usd) / total_2023_usd * 100) if total_2023_usd != 0 else 0
        usd_growth_22_24 = ((total_2024_usd - total_2022_usd) / total_2022_usd * 100) if total_2022_usd != 0 else 0
        
        volume_growth_22_23 = ((total_2023_units - total_2022_units) / total_2022_units * 100) if total_2022_units != 0 else 0
        volume_growth_23_24 = ((total_2024_units - total_2023_units) / total_2023_units * 100) if total_2023_units != 0 else 0
        volume_growth_22_24 = ((total_2024_units - total_2022_units) / total_2022_units * 100) if total_2022_units != 0 else 0
        
        price_growth_22_23 = ((avg_price_2023 - avg_price_2022) / avg_price_2022 * 100) if avg_price_2022 != 0 else 0
        price_growth_23_24 = ((avg_price_2024 - avg_price_2023) / avg_price_2023 * 100) if avg_price_2023 != 0 else 0
        price_growth_22_24 = ((avg_price_2024 - avg_price_2022) / avg_price_2022 * 100) if avg_price_2022 != 0 else 0
        
        # Create bubble chart
        fig = go.Figure()
        
        periods = ['2022→2023', '2023→2024', '2022→2024']
        volume_growths = [volume_growth_22_23, volume_growth_23_24, volume_growth_22_24]
        price_growths = [price_growth_22_23, price_growth_23_24, price_growth_22_24]
        usd_growths = [usd_growth_22_23, usd_growth_23_24, usd_growth_22_24]
        
        # Size scaling
        sizes = [abs(g) * 10 for g in usd_growths]
        
        fig.add_trace(go.Scatter(
            x=volume_growths,
            y=price_growths,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=['#3182ce', '#10b981', '#8b5cf6'],
                opacity=0.8,
                line=dict(width=2, color='white')
            ),
            text=periods,
            textposition='top center',
            textfont=dict(size=12, color='#1e3d8c'),
            name='Growth Periods'
        ))
        
        # Add quadrant lines
        fig.add_shape(type="line",
            x0=min(volume_growths) - 5, y0=0,
            x1=max(volume_growths) + 5, y1=0,
            line=dict(color="#cbd5e0", width=1, dash="dash")
        )
        
        fig.add_shape(type="line",
            x0=0, y0=min(price_growths) - 5,
            x1=0, y1=max(price_growths) + 5,
            line=dict(color="#cbd5e0", width=1, dash="dash")
        )
        
        # Add annotations for quadrants
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="<b>Price & Volume Growth</b>",
            showarrow=False,
            font=dict(size=10, color="#10b981")
        )
        
        fig.add_annotation(
            x=0.5, y=0.95,
            xref="paper", yref="paper",
            text="<b>Price Growth Driven</b>",
            showarrow=False,
            font=dict(size=10, color="#3182ce")
        )
        
        fig.add_annotation(
            x=0.95, y=0.5,
            xref="paper", yref="paper",
            text="<b>Volume Growth Driven</b>",
            showarrow=False,
            font=dict(size=10, color="#8b5cf6")
        )
        
        fig.update_layout(
            title=dict(
                text='Price-Volume Growth Analysis (3-Year Periods)',
                font=dict(size=18, color='#1e3d8c'),
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title='Volume Growth %',
                gridcolor='#e2e8f0',
                ticksuffix='%'
            ),
            yaxis=dict(
                title='Price Growth %',
                gridcolor='#e2e8f0',
                ticksuffix='%'
            ),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False
        )
        
        return fig
    
    def create_pack_strength_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create pack, strength, size analysis with 3-year trends"""
        # Create subplots for different analyses
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('International Pack - USD MNF', 'International Strength - Units',
                          'International Size - Avg Price', 'Volume Mix - 3 Year Trend'),
            vertical_spacing=0.15,
            horizontal_spacing=0.2
        )
        
        # 1. International Pack Analysis
        if 'International Pack' in df.columns:
            pack_data = df.groupby('International Pack').agg({
                'MAT Q3 2022 USD MNF': 'sum',
                'MAT Q3 2023 USD MNF': 'sum',
                'MAT Q3 2024 USD MNF': 'sum'
            }).reset_index()
            
            # Get top 10 packs
            pack_data['Total_2024'] = pack_data['MAT Q3 2024 USD MNF']
            top_packs = pack_data.nlargest(10, 'Total_2024')
            
            for year, color in zip(['2022', '2023', '2024'], ['#3182ce', '#10b981', '#8b5cf6']):
                fig.add_trace(
                    go.Bar(
                        x=top_packs['International Pack'],
                        y=top_packs[f'MAT Q3 {year} USD MNF'],
                        name=year,
                        marker_color=color,
                        showlegend=(year == '2022')
                    ),
                    row=1, col=1
                )
        
        # 2. International Strength Analysis
        if 'International Strength' in df.columns:
            strength_data = df.groupby('International Strength').agg({
                'MAT Q3 2022 Units': 'sum',
                'MAT Q3 2023 Units': 'sum',
                'MAT Q3 2024 Units': 'sum'
            }).reset_index()
            
            # Get top 10 strengths
            strength_data['Total_2024'] = strength_data['MAT Q3 2024 Units']
            top_strengths = strength_data.nlargest(10, 'Total_2024')
            
            for year, color in zip(['2022', '2023', '2024'], ['#3182ce', '#10b981', '#8b5cf6']):
                fig.add_trace(
                    go.Bar(
                        x=top_strengths['International Strength'],
                        y=top_strengths[f'MAT Q3 {year} Units'],
                        name=year,
                        marker_color=color,
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. International Size Analysis
        if 'International Size' in df.columns:
            size_data = df.groupby('International Size').agg({
                'MAT Q3 2022 Unit Avg Price USD MNF': 'mean',
                'MAT Q3 2023 Unit Avg Price USD MNF': 'mean',
                'MAT Q3 2024 Unit Avg Price USD MNF': 'mean'
            }).reset_index()
            
            # Get top 10 sizes by average price
            size_data['Avg_Price_2024'] = size_data['MAT Q3 2024 Unit Avg Price USD MNF']
            top_sizes = size_data.nlargest(10, 'Avg_Price_2024')
            
            for year, color in zip(['2022', '2023', '2024'], ['#3182ce', '#10b981', '#8b5cf6']):
                fig.add_trace(
                    go.Scatter(
                        x=top_sizes['International Size'],
                        y=top_sizes[f'MAT Q3 {year} Unit Avg Price USD MNF'],
                        mode='lines+markers',
                        name=year,
                        line=dict(color=color, width=2),
                        marker=dict(size=8, color=color),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Volume Mix Analysis
        if 'International Volume' in df.columns:
            volume_data = df.groupby('International Volume').agg({
                'MAT Q3 2022 Units': 'sum',
                'MAT Q3 2023 Units': 'sum',
                'MAT Q3 2024 Units': 'sum'
            }).reset_index()
            
            # Calculate percentage mix
            total_2022 = volume_data['MAT Q3 2022 Units'].sum()
            total_2023 = volume_data['MAT Q3 2023 Units'].sum()
            total_2024 = volume_data['MAT Q3 2024 Units'].sum()
            
            volume_data['Mix_2022'] = (volume_data['MAT Q3 2022 Units'] / total_2022 * 100) if total_2022 != 0 else 0
            volume_data['Mix_2023'] = (volume_data['MAT Q3 2023 Units'] / total_2023 * 100) if total_2023 != 0 else 0
            volume_data['Mix_2024'] = (volume_data['MAT Q3 2024 Units'] / total_2024 * 100) if total_2024 != 0 else 0
            
            # Get top 10 volumes
            volume_data['Total_Units'] = volume_data[['MAT Q3 2022 Units', 'MAT Q3 2023 Units', 'MAT Q3 2024 Units']].sum(axis=1)
            top_volumes = volume_data.nlargest(10, 'Total_Units')
            
            years = ['2022', '2023', '2024']
            for i, year in enumerate(years):
                fig.add_trace(
                    go.Bar(
                        x=top_volumes['International Volume'],
                        y=top_volumes[f'Mix_{year}'],
                        name=year,
                        marker_color=['#3182ce', '#10b981', '#8b5cf6'][i],
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=dict(
                text='Pack, Strength, Size & Volume Analysis - 3 Year Trends',
                font=dict(size=20, color='#1e3d8c'),
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=800,
            margin=dict(l=50, r=50, t=100, b=100),
            barmode='group',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=2)
        
        fig.update_yaxes(title_text="USD MNF", row=1, col=1)
        fig.update_yaxes(title_text="Units", row=1, col=2)
        fig.update_yaxes(title_text="Avg Price", row=2, col=1)
        fig.update_yaxes(title_text="Mix %", row=2, col=2)
        
        return fig

class InsightEngine:
    """Rule-based insight generation engine with 3-year focus"""
    
    def __init__(self, data_processor: PharmaDataProcessor):
        self.dp = data_processor
    
    def generate_executive_insights(self, df: pd.DataFrame) -> List[Dict]:
        """Generate Turkish language insights based on 3-year trends"""
        insights = []
        
        # Global trends
        total_2022 = df['MAT Q3 2022 USD MNF'].sum()
        total_2023 = df['MAT Q3 2023 USD MNF'].sum()
        total_2024 = df['MAT Q3 2024 USD MNF'].sum()
        growth_22_24 = ((total_2024 - total_2022) / total_2022 * 100) if total_2022 != 0 else 0
        
        if growth_22_24 > 20:
            insights.append({
                'type': 'positive',
                'title': 'Güçlü Global Büyüme',
                'message': f'Global USD MNF 3 yılda %{growth_22_24:.1f} büyüdü. 2022: ${total_2022:,.0f}, 2024: ${total_2024:,.0f}'
            })
        elif growth_22_24 < -5:
            insights.append({
                'type': 'negative',
                'title': 'Dikkat: Düşüş Trendi',
                'message': f'Global USD MNF 3 yılda %{abs(growth_22_24):.1f} azaldı. 2022: ${total_2022:,.0f}, 2024: ${total_2024:,.0f}'
            })
        
        # Country performance
        country_data = df.groupby('Country').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'USD_MNF_Growth_22_24': 'mean'
        }).reset_index()
        
        top_country = country_data.loc[country_data['USD_MNF_Growth_22_24'].idxmax()]
        bottom_country = country_data.loc[country_data['USD_MNF_Growth_22_24'].idxmin()]
        
        insights.append({
            'type': 'info',
            'title': 'En Hızlı Büyüyen Ülke',
            'message': f"{top_country['Country']}: 3 yılda %{top_country['USD_MNF_Growth_22_24']:.1f} büyüme (2022: ${top_country['MAT Q3 2022 USD MNF']:,.0f} → 2024: ${top_country['MAT Q3 2024 USD MNF']:,.0f})"
        })
        
        if bottom_country['USD_MNF_Growth_22_24'] < -10:
            insights.append({
                'type': 'warning',
                'title': 'Düşüşteki Ülke',
                'message': f"{bottom_country['Country']}: 3 yılda %{abs(bottom_country['USD_MNF_Growth_22_24']):.1f} azalma"
            })
        
        # Molecule trends
        molecule_data = df.groupby('Molecule').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum'
        }).reset_index()
        
        molecule_data['Growth'] = ((molecule_data['MAT Q3 2024 USD MNF'] - molecule_data['MAT Q3 2022 USD MNF']) / 
                                  molecule_data['MAT Q3 2022 USD MNF'].replace(0, np.nan)) * 100
        
        top_molecule = molecule_data.nlargest(1, 'Growth')
        if not top_molecule.empty and top_molecule['Growth'].iloc[0] > 50:
            insights.append({
                'type': 'positive',
                'title': 'Yıldız Molekül',
                'message': f"{top_molecule['Molecule'].iloc[0]}: 3 yılda %{top_molecule['Growth'].iloc[0]:.1f} büyüme"
            })
        
        # Price trends
        avg_price_2022 = df['MAT Q3 2022 SU Avg Price USD MNF'].mean()
        avg_price_2024 = df['MAT Q3 2024 SU Avg Price USD MNF'].mean()
        price_change = ((avg_price_2024 - avg_price_2022) / avg_price_2022 * 100) if avg_price_2022 != 0 else 0
        
        if price_change > 10:
            insights.append({
                'type': 'info',
                'title': 'Premiumlaşma Trendi',
                'message': f'Ortalama fiyat 3 yılda %{price_change:.1f} arttı. 2022: ${avg_price_2022:.2f} → 2024: ${avg_price_2024:.2f}'
            })
        
        # Market share shifts
        if 'Corporation' in df.columns:
            corp_data = df.groupby('Corporation').agg({
                'MAT Q3 2022 USD MNF': 'sum',
                'MAT Q3 2024 USD MNF': 'sum'
            }).reset_index()
            
            total_2022_corp = corp_data['MAT Q3 2022 USD MNF'].sum()
            total_2024_corp = corp_data['MAT Q3 2024 USD MNF'].sum()
            
            corp_data['Share_2022'] = (corp_data['MAT Q3 2022 USD MNF'] / total_2022_corp * 100) if total_2022_corp != 0 else 0
            corp_data['Share_2024'] = (corp_data['MAT Q3 2024 USD MNF'] / total_2024_corp * 100) if total_2024_corp != 0 else 0
            corp_data['Share_Change'] = corp_data['Share_2024'] - corp_data['Share_2022']
            
            top_gainer = corp_data.nlargest(1, 'Share_Change')
            top_loser = corp_data.nsmallest(1, 'Share_Change')
            
            if not top_gainer.empty and top_gainer['Share_Change'].iloc[0] > 3:
                insights.append({
                    'type': 'positive',
                    'title': 'Pazar Payı Kazananı',
                    'message': f"{top_gainer['Corporation'].iloc[0]}: Pazar payı 3 yılda +{top_gainer['Share_Change'].iloc[0]:.1f}% arttı"
                })
        
        return insights
    
    def generate_molecule_insights(self, df: pd.DataFrame) -> List[Dict]:
        """Generate molecule-specific insights"""
        insights = []
        
        # Group by molecule
        molecule_data = df.groupby('Molecule').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 Units': 'sum',
            'MAT Q3 2023 Units': 'sum',
            'MAT Q3 2024 Units': 'sum',
            'USD_MNF_Trend': 'first',
            'Price_Trend': 'first'
        }).reset_index()
        
        # Calculate growth metrics
        molecule_data['USD_Growth_22_24'] = ((molecule_data['MAT Q3 2024 USD MNF'] - molecule_data['MAT Q3 2022 USD MNF']) / 
                                            molecule_data['MAT Q3 2022 USD MNF'].replace(0, np.nan)) * 100
        
        # Find top performers
        top_5_growth = molecule_data.nlargest(5, 'USD_Growth_22_24')
        for _, row in top_5_growth.iterrows():
            if row['USD_Growth_22_24'] > 100:
                insights.append({
                    'type': 'positive',
                    'title': 'Üstel Büyüyen Molekül',
                    'message': f"{row['Molecule']}: 3 yılda %{row['USD_Growth_22_24']:.0f} büyüme"
                })
        
        # Find declining molecules
        declining = molecule_data[molecule_data['USD_MNF_Trend'] == 'B Azalan']
        if len(declining) > 0:
            top_decliner = declining.nsmallest(1, 'USD_Growth_22_24')
            if not top_decliner.empty:
                insights.append({
                    'type': 'warning',
                    'title': 'Düşüş Trendindeki Molekül',
                    'message': f"{top_decliner['Molecule'].iloc[0]}: 3 yılda %{abs(top_decliner['USD_Growth_22_24'].iloc[0]):.0f} azalma"
                })
        
        # Price trend analysis
        premium_molecules = molecule_data[molecule_data['Price_Trend'] == 'A Artan']
        if len(premium_molecules) > 0:
            top_premium = premium_molecules.nlargest(1, 'MAT Q3 2024 USD MNF')
            if not top_premium.empty:
                insights.append({
                    'type': 'info',
                    'title': 'Premium Fiyat Trendi',
                    'message': f"{top_premium['Molecule'].iloc[0]}: Artan fiyat trendiyle premium segmentte lider"
                })
        
        return insights
    
    def generate_corporation_insights(self, df: pd.DataFrame) -> List[Dict]:
        """Generate corporation competition insights"""
        insights = []
        
        if 'Corporation' not in df.columns:
            return insights
        
        # Corporation analysis
        corp_data = df.groupby('Corporation').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 Units': 'sum',
            'MAT Q3 2023 Units': 'sum',
            'MAT Q3 2024 Units': 'sum'
        }).reset_index()
        
        # Calculate totals
        total_2022 = corp_data['MAT Q3 2022 USD MNF'].sum()
        total_2023 = corp_data['MAT Q3 2023 USD MNF'].sum()
        total_2024 = corp_data['MAT Q3 2024 USD MNF'].sum()
        
        # Calculate market shares
        corp_data['Share_2022'] = (corp_data['MAT Q3 2022 USD MNF'] / total_2022 * 100) if total_2022 != 0 else 0
        corp_data['Share_2023'] = (corp_data['MAT Q3 2023 USD MNF'] / total_2023 * 100) if total_2023 != 0 else 0
        corp_data['Share_2024'] = (corp_data['MAT Q3 2024 USD MNF'] / total_2024 * 100) if total_2024 != 0 else 0
        
        corp_data['Share_Change_22_24'] = corp_data['Share_2024'] - corp_data['Share_2022']
        
        # Market leader analysis
        market_leader_2024 = corp_data.nlargest(1, 'Share_2024')
        if not market_leader_2024.empty:
            leader = market_leader_2024.iloc[0]
            insights.append({
                'type': 'info',
                'title': 'Pazar Lideri',
                'message': f"{leader['Corporation']}: %{leader['Share_2024']:.1f} pazar payı ile 2024 lideri"
            })
        
        # Biggest gainers and losers
        top_gainer = corp_data.nlargest(1, 'Share_Change_22_24')
        top_loser = corp_data.nsmallest(1, 'Share_Change_22_24')
        
        if not top_gainer.empty and top_gainer['Share_Change_22_24'].iloc[0] > 2:
            insights.append({
                'type': 'positive',
                'title': 'En Büyük Kazanan',
                'message': f"{top_gainer['Corporation'].iloc[0]}: Pazar payı 3 yılda +{top_gainer['Share_Change_22_24'].iloc[0]:.1f}% arttı"
            })
        
        if not top_loser.empty and top_loser['Share_Change_22_24'].iloc[0] < -2:
            insights.append({
                'type': 'negative',
                'title': 'En Büyük Kaybeden',
                'message': f"{top_loser['Corporation'].iloc[0]}: Pazar payı 3 yılda {top_loser['Share_Change_22_24'].iloc[0]:.1f}% azaldı"
            })
        
        # Growth rate analysis
        corp_data['Growth_22_24'] = ((corp_data['MAT Q3 2024 USD MNF'] - corp_data['MAT Q3 2022 USD MNF']) / 
                                    corp_data['MAT Q3 2022 USD MNF'].replace(0, np.nan)) * 100
        
        fastest_growing = corp_data.nlargest(1, 'Growth_22_24')
        if not fastest_growing.empty and fastest_growing['Growth_22_24'].iloc[0] > 50:
            insights.append({
                'type': 'positive',
                'title': 'En Hızlı Büyüyen Firma',
                'message': f"{fastest_growing['Corporation'].iloc[0]}: 3 yılda %{fastest_growing['Growth_22_24'].iloc[0]:.0f} büyüme"
            })
        
        return insights
    
    def generate_specialty_insights(self, df: pd.DataFrame) -> List[Dict]:
        """Generate specialty vs non-specialty insights"""
        insights = []
        
        if 'Specialty Product' not in df.columns:
            return insights
        
        # Classify products
        df['Product_Type'] = df['Specialty Product'].apply(
            lambda x: 'Specialty' if x == 'Yes' else 'Non-Specialty'
        )
        
        # Aggregate by product type
        type_data = df.groupby('Product_Type').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 Units': 'sum',
            'MAT Q3 2023 Units': 'sum',
            'MAT Q3 2024 Units': 'sum',
            'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2023 SU Avg Price USD MNF': 'mean',
            'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
        }).reset_index()
        
        # Calculate growth rates
        for product_type in ['Specialty', 'Non-Specialty']:
            data = type_data[type_data['Product_Type'] == product_type]
            if not data.empty:
                usd_growth = ((data['MAT Q3 2024 USD MNF'].iloc[0] - data['MAT Q3 2022 USD MNF'].iloc[0]) / 
                             data['MAT Q3 2022 USD MNF'].iloc[0] * 100) if data['MAT Q3 2022 USD MNF'].iloc[0] != 0 else 0
                
                price_growth = ((data['MAT Q3 2024 SU Avg Price USD MNF'].iloc[0] - data['MAT Q3 2022 SU Avg Price USD MNF'].iloc[0]) / 
                               data['MAT Q3 2022 SU Avg Price USD MNF'].iloc[0] * 100) if data['MAT Q3 2022 SU Avg Price USD MNF'].iloc[0] != 0 else 0
                
                if product_type == 'Specialty' and usd_growth > 30:
                    insights.append({
                        'type': 'positive',
                        'title': 'Specialty Ürünlerde Güçlü Büyüme',
                        'message': f'Specialty ürünler 3 yılda %{usd_growth:.1f} büyüdü. Fiyat artışı: %{price_growth:.1f}'
                    })
                elif product_type == 'Specialty' and price_growth > 20:
                    insights.append({
                        'type': 'info',
                        'title': 'Specialty Premiumlaşma',
                        'message': f'Specialty ürün fiyatları 3 yılda %{price_growth:.1f} arttı'
                    })
        
        # Market share analysis
        total_2022 = type_data['MAT Q3 2022 USD MNF'].sum()
        total_2024 = type_data['MAT Q3 2024 USD MNF'].sum()
        
        for product_type in ['Specialty', 'Non-Specialty']:
            data = type_data[type_data['Product_Type'] == product_type]
            if not data.empty:
                share_2022 = (data['MAT Q3 2022 USD MNF'].iloc[0] / total_2022 * 100) if total_2022 != 0 else 0
                share_2024 = (data['MAT Q3 2024 USD MNF'].iloc[0] / total_2024 * 100) if total_2024 != 0 else 0
                share_change = share_2024 - share_2022
                
                if product_type == 'Specialty' and share_change > 5:
                    insights.append({
                        'type': 'info',
                        'title': 'Specialty Pazar Payı Artışı',
                        'message': f'Specialty ürün pazar payı 3 yılda {share_change:.1f}% arttı (%{share_2022:.1f} → %{share_2024:.1f})'
                    })
        
        return insights

def create_sample_data() -> pd.DataFrame:
    """Create comprehensive sample pharmaceutical data"""
    np.random.seed(42)
    
    # Define base data
    countries = ['Turkey', 'Germany', 'France', 'Italy', 'Spain', 'UK', 'USA', 'Japan', 'China', 'Brazil']
    regions = ['Europe', 'North America', 'Asia', 'Latin America']
    sub_regions = ['Western Europe', 'Eastern Europe', 'North America', 'East Asia', 'South America']
    
    corporations = ['Novartis', 'Pfizer', 'Roche', 'Merck', 'GSK', 'Sanofi', 'AstraZeneca', 'Johnson & Johnson', 'Bayer', 'AbbVie']
    manufacturers = ['Manufacturer A', 'Manufacturer B', 'Manufacturer C', 'Manufacturer D', 'Manufacturer E']
    
    molecules = ['Adalimumab', 'Etanercept', 'Infliximab', 'Rituximab', 'Bevacizumab', 
                 'Trastuzumab', 'Pembrolizumab', 'Nivolumab', 'Insulin Glargine', 'Sitagliptin',
                 'Atorvastatin', 'Rosuvastatin', 'Metformin', 'Levothyroxine', 'Amlodipine']
    
    sectors = ['Oncology', 'Immunology', 'Diabetes', 'Cardiology', 'Neurology', 'Rare Diseases']
    panels = ['Hospital', 'Retail', 'Specialty Pharmacy', 'Clinical']
    
    n_records = 5000
    
    data = {
        'Source.Name': [f'Source_{i}' for i in range(n_records)],
        'Country': np.random.choice(countries, n_records),
        'Sector': np.random.choice(sectors, n_records),
        'Panel': np.random.choice(panels, n_records),
        'Region': np.random.choice(regions, n_records),
        'Sub-Region': np.random.choice(sub_regions, n_records),
        'Corporation': np.random.choice(corporations, n_records),
        'Manufacturer': np.random.choice(manufacturers, n_records),
        'Molecule List': [';'.join(np.random.choice(molecules, np.random.randint(1, 4))) for _ in range(n_records)],
        'Molecule': np.random.choice(molecules, n_records),
        'Chemical Salt': [f'Salt_{i}' for i in np.random.randint(1, 50, n_records)],
        'International Product': [f'INT_PROD_{i:03d}' for i in np.random.randint(1, 200, n_records)],
        'Specialty Product': np.random.choice(['Yes', 'No'], n_records, p=[0.3, 0.7]),
        'NFC123': [f'NFC_{i:03d}' for i in np.random.randint(1, 100, n_records)],
        'International Pack': np.random.choice(['Pack_10', 'Pack_20', 'Pack_30', 'Pack_60', 'Pack_90'], n_records),
        'International Strength': np.random.choice(['10mg', '20mg', '40mg', '80mg', '100mg', '150mg'], n_records),
        'International Size': np.random.choice(['Small', 'Medium', 'Large', 'X-Large'], n_records),
        'International Volume': np.random.choice(['1ml', '2ml', '5ml', '10ml', '20ml'], n_records),
        'International Prescription': np.random.choice(['Rx', 'OTC', 'Hospital Only'], n_records),
    }
    
    # Generate realistic sales data with trends
    base_usd = np.random.lognormal(10, 1.5, n_records)
    
    # Add growth trends
    growth_factors = np.random.uniform(0.8, 1.5, n_records)
    
    # 2022 data
    data['MAT Q3 2022 USD MNF'] = base_usd * np.random.uniform(10000, 500000, n_records)
    data['MAT Q3 2022 Standard Units'] = data['MAT Q3 2022 USD MNF'] / np.random.uniform(50, 500, n_records)
    data['MAT Q3 2022 Units'] = data['MAT Q3 2022 Standard Units'] * np.random.uniform(0.8, 1.2, n_records)
    
    # 2023 data with growth
    data['MAT Q3 2023 USD MNF'] = data['MAT Q3 2022 USD MNF'] * growth_factors * np.random.uniform(0.9, 1.3, n_records)
    data['MAT Q3 2023 Standard Units'] = data['MAT Q3 2022 Standard Units'] * growth_factors * np.random.uniform(0.9, 1.2, n_records)
    data['MAT Q3 2023 Units'] = data['MAT Q3 2022 Units'] * growth_factors * np.random.uniform(0.9, 1.2, n_records)
    
    # 2024 data with continued trends
    data['MAT Q3 2024 USD MNF'] = data['MAT Q3 2023 USD MNF'] * growth_factors * np.random.uniform(0.95, 1.35, n_records)
    data['MAT Q3 2024 Standard Units'] = data['MAT Q3 2023 Standard Units'] * growth_factors * np.random.uniform(0.95, 1.25, n_records)
    data['MAT Q3 2024 Units'] = data['MAT Q3 2023 Units'] * growth_factors * np.random.uniform(0.95, 1.25, n_records)
    
    # Generate price data (pre-calculated as per requirements)
    for year in [2022, 2023, 2024]:
        data[f'MAT Q3 {year} SU Avg Price USD MNF'] = np.random.uniform(10, 500, n_records) * (1 + (year - 2022) * 0.05)
        data[f'MAT Q3 {year} Unit Avg Price USD MNF'] = data[f'MAT Q3 {year} SU Avg Price USD MNF'] * np.random.uniform(0.8, 1.2, n_records)
    
    df = pd.DataFrame(data)
    
    # Add some null values realistically
    for col in df.columns:
        if np.random.random() < 0.05:
            idx = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
            df.loc[idx, col] = np.nan
    
    return df

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">💊 Pharma Commercial Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #4a5568; font-size: 1.2rem;">Enterprise-Grade 3-Year Trend Analysis (2022-2024)</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading and processing pharmaceutical data...'):
        df = create_sample_data()
        data_processor = PharmaDataProcessor(df)
        visualizer = AnalyticsVisualizer(data_processor)
        insight_engine = InsightEngine(data_processor)
    
    # Sidebar filters
    st.sidebar.markdown('### 🔍 Global Filtreler')
    
    # Get unique values for filters
    countries = ['Tümü'] + sorted(df['Country'].unique().tolist())
    corporations = ['Tümü'] + sorted(df['Corporation'].unique().tolist())
    molecules = ['Tümü'] + sorted(df['Molecule'].unique().tolist())
    sectors = ['Tümü'] + sorted(df['Sector'].unique().tolist())
    panels = ['Tümü'] + sorted(df['Panel'].unique().tolist())
    specialties = ['Tümü', 'Specialty', 'Non-Specialty']
    
    # Create filters
    selected_country = st.sidebar.selectbox('Ülke', countries, index=0)
    selected_corporation = st.sidebar.selectbox('Kuruluş', corporations, index=0)
    selected_molecule = st.sidebar.selectbox('Molekül', molecules, index=0)
    selected_sector = st.sidebar.selectbox('Sektör', sectors, index=0)
    selected_panel = st.sidebar.selectbox('Panel', panels, index=0)
    selected_specialty = st.sidebar.selectbox('Specialty Ürün', specialties, index=0)
    
    filters = {
        'country': selected_country,
        'corporation': selected_corporation,
        'molecule': selected_molecule,
        'sector': selected_sector,
        'panel': selected_panel,
        'specialty': selected_specialty
    }
    
    # Apply filters
    filtered_df = data_processor.get_filtered_data(filters)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        '📊 Yönetici Özeti',
        '🌍 Ülke & Bölge Analizi',
        '⚗️ Molekül Analizi',
        '🏢 Corporation & Rekabet',
        '🎯 Specialty vs Non-Specialty',
        '💰 Fiyat & Mix Analizi',
        '📦 Pack / Strength / Size',
        '🤖 Otomatik İçgörü Motoru'
    ])
    
    # Tab 1: Executive Summary
    with tab1:
        st.markdown('<h2 class="sub-header">Yönetici Özeti - 3 Yıllık Global Trendler</h2>', unsafe_allow_html=True)
        
        # Key metrics
        visualizer.create_executive_summary_metrics(filtered_df)
        st.markdown('---')
        
        # 3-Year Trend Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_usd = visualizer.create_three_year_trend_chart(
                filtered_df, 'USD MNF', 'Global USD MNF Trend (2022-2024)'
            )
            st.plotly_chart(fig_usd, use_container_width=True)
        
        with col2:
            fig_units = visualizer.create_three_year_trend_chart(
                filtered_df, 'Units', 'Total Units Trend (2022-2024)'
            )
            st.plotly_chart(fig_units, use_container_width=True)
        
        # Growth Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_growth = visualizer.create_growth_comparison_chart(filtered_df)
            st.plotly_chart(fig_growth, use_container_width=True)
        
        with col2:
            # Top 10 countries by 2024 sales
            country_sales = filtered_df.groupby('Country').agg({
                'MAT Q3 2024 USD MNF': 'sum',
                'USD_MNF_Growth_22_24': 'mean'
            }).reset_index()
            
            country_sales = country_sales.sort_values('MAT Q3 2024 USD MNF', ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=country_sales['Country'],
                    y=country_sales['MAT Q3 2024 USD MNF'],
                    marker_color='#3182ce',
                    text=[f'${x:,.0f}' for x in country_sales['MAT Q3 2024 USD MNF']],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=dict(
                    text='Top 10 Countries by 2024 USD MNF',
                    font=dict(size=16, color='#1e3d8c'),
                    x=0.5
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='Country',
                    tickangle=45,
                    gridcolor='#e2e8f0'
                ),
                yaxis=dict(
                    title='USD MNF (2024)',
                    gridcolor='#e2e8f0',
                    tickprefix='$'
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Automatic insights
        st.markdown('---')
        st.markdown('<h3>📈 Otomatik İçgörüler (3 Yıllık Trend)</h3>', unsafe_allow_html=True)
        
        insights = insight_engine.generate_executive_insights(filtered_df)
        
        for insight in insights:
            if insight['type'] == 'positive':
                st.success(f"**{insight['title']}**: {insight['message']}")
            elif insight['type'] == 'negative':
                st.error(f"**{insight['title']}**: {insight['message']}")
            elif insight['type'] == 'warning':
                st.warning(f"**{insight['title']}**: {insight['message']}")
            else:
                st.info(f"**{insight['title']}**: {insight['message']}")
    
    # Tab 2: Country & Region Analysis
    with tab2:
        st.markdown('<h2 class="sub-header">Ülke & Bölge Analizi - 3 Yıllık Performans</h2>', unsafe_allow_html=True)
        
        # Country selector
        selected_country_detail = st.selectbox(
            'Detaylı Analiz için Ülke Seçin',
            ['Tüm Ülkeler'] + sorted(filtered_df['Country'].unique().tolist()),
            key='country_detail'
        )
        
        if selected_country_detail != 'Tüm Ülkeler':
            country_df = filtered_df[filtered_df['Country'] == selected_country_detail]
        else:
            country_df = filtered_df
        
        # Country metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if selected_country_detail != 'Tüm Ülkeler':
                total_2022 = country_df['MAT Q3 2022 USD MNF'].sum()
                total_2023 = country_df['MAT Q3 2023 USD MNF'].sum()
                total_2024 = country_df['MAT Q3 2024 USD MNF'].sum()
                growth = ((total_2024 - total_2022) / total_2022 * 100) if total_2022 != 0 else 0
                
                st.metric(
                    label=f"{selected_country_detail} USD MNF",
                    value=f"${total_2024:,.0f}",
                    delta=f"{growth:+.1f}% (3 yıl)"
                )
        
        with col2:
            if selected_country_detail != 'Tüm Ülkeler':
                units_2022 = country_df['MAT Q3 2022 Units'].sum()
                units_2024 = country_df['MAT Q3 2024 Units'].sum()
                units_growth = ((units_2024 - units_2022) / units_2022 * 100) if units_2022 != 0 else 0
                
                st.metric(
                    label=f"{selected_country_detail} Units",
                    value=f"{units_2024:,.0f}",
                    delta=f"{units_growth:+.1f}% (3 yıl)"
                )
        
        with col3:
            if selected_country_detail != 'Tüm Ülkeler':
                price_2022 = country_df['MAT Q3 2022 SU Avg Price USD MNF'].mean()
                price_2024 = country_df['MAT Q3 2024 SU Avg Price USD MNF'].mean()
                price_change = ((price_2024 - price_2022) / price_2022 * 100) if price_2022 != 0 else 0
                
                st.metric(
                    label=f"{selected_country_detail} Avg Price",
                    value=f"${price_2024:.2f}",
                    delta=f"{price_change:+.1f}% (3 yıl)"
                )
        
        # Region analysis
        st.markdown('---')
        st.markdown('<h4>Bölgesel Analiz</h4>', unsafe_allow_html=True)
        
        if 'Region' in country_df.columns:
            region_data = country_df.groupby('Region').agg({
                'MAT Q3 2022 USD MNF': 'sum',
                'MAT Q3 2023 USD MNF': 'sum',
                'MAT Q3 2024 USD MNF': 'sum'
            }).reset_index()
            
            # Calculate growth
            region_data['Growth_22_24'] = ((region_data['MAT Q3 2024 USD MNF'] - region_data['MAT Q3 2022 USD MNF']) / 
                                          region_data['MAT Q3 2022 USD MNF'].replace(0, np.nan)) * 100
            
            # Create chart
            fig = go.Figure()
            
            for region in region_data['Region'].unique():
                region_df = region_data[region_data['Region'] == region]
                fig.add_trace(go.Bar(
                    x=['2022', '2023', '2024'],
                    y=[region_df['MAT Q3 2022 USD MNF'].iloc[0], 
                      region_df['MAT Q3 2023 USD MNF'].iloc[0],
                      region_df['MAT Q3 2024 USD MNF'].iloc[0]],
                    name=region,
                    text=[f'${x:,.0f}' for x in [region_df['MAT Q3 2022 USD MNF'].iloc[0], 
                                                 region_df['MAT Q3 2023 USD MNF'].iloc[0],
                                                 region_df['MAT Q3 2024 USD MNF'].iloc[0]]],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title=dict(
                    text='Regional USD MNF Distribution (2022-2024)',
                    font=dict(size=16, color='#1e3d8c'),
                    x=0.5
                ),
                barmode='group',
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='Year',
                    gridcolor='#e2e8f0'
                ),
                yaxis=dict(
                    title='USD MNF',
                    gridcolor='#e2e8f0',
                    tickprefix='$'
                ),
                height=500,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Country comparison table
        st.markdown('---')
        st.markdown('<h4>Ülke Bazlı Karşılaştırma (2022-2024)</h4>', unsafe_allow_html=True)
        
        country_comparison = filtered_df.groupby('Country').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'MAT Q3 2022 Units': 'sum',
            'MAT Q3 2023 Units': 'sum',
            'MAT Q3 2024 Units': 'sum',
            'USD_MNF_Growth_22_23': 'mean',
            'USD_MNF_Growth_23_24': 'mean',
            'USD_MNF_Growth_22_24': 'mean',
            'USD_MNF_Trend': 'first'
        }).reset_index()
        
        # Format for display
        display_df = country_comparison.copy()
        for col in ['MAT Q3 2022 USD MNF', 'MAT Q3 2023 USD MNF', 'MAT Q3 2024 USD MNF']:
            display_df[col] = display_df[col].apply(lambda x: f'${x:,.0f}')
        
        for col in ['USD_MNF_Growth_22_23', 'USD_MNF_Growth_23_24', 'USD_MNF_Growth_22_24']:
            display_df[col] = display_df[col].apply(lambda x: f'{x:+.1f}%')
        
        st.dataframe(
            display_df.sort_values('MAT Q3 2024 USD MNF', ascending=False),
            use_container_width=True,
            height=400
        )
    
    # Tab 3: Molecule Analysis
    with tab3:
        st.markdown('<h2 class="sub-header">Molekül Analizi - 3 Yıllık Trend ve Büyüme</h2>', unsafe_allow_html=True)
        
        # Molecule selector
        molecule_list = ['Tüm Moleküller'] + sorted(filtered_df['Molecule'].unique().tolist())
        selected_molecule_detail = st.selectbox(
            'Detaylı Analiz için Molekül Seçin',
            molecule_list,
            key='molecule_detail'
        )
        
        if selected_molecule_detail != 'Tüm Moleküller':
            molecule_df = filtered_df[filtered_df['Molecule'] == selected_molecule_detail]
        else:
            molecule_df = filtered_df
        
        # Key molecule metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if selected_molecule_detail != 'Tüm Moleküller':
                total_2022 = molecule_df['MAT Q3 2022 USD MNF'].sum()
                total_2024 = molecule_df['MAT Q3 2024 USD MNF'].sum()
                growth = ((total_2024 - total_2022) / total_2022 * 100) if total_2022 != 0 else 0
                
                st.metric(
                    label=f"{selected_molecule_detail} USD MNF",
                    value=f"${total_2024:,.0f}",
                    delta=f"{growth:+.1f}% (3 yıl)"
                )
        
        with col2:
            if selected_molecule_detail != 'Tüm Moleküller':
                units_2022 = molecule_df['MAT Q3 2022 Units'].sum()
                units_2024 = molecule_df['MAT Q3 2024 Units'].sum()
                units_growth = ((units_2024 - units_2022) / units_2022 * 100) if units_2022 != 0 else 0
                
                st.metric(
                    label=f"{selected_molecule_detail} Units",
                    value=f"{units_2024:,.0f}",
                    delta=f"{units_growth:+.1f}% (3 yıl)"
                )
        
        with col3:
            if selected_molecule_detail != 'Tüm Moleküller':
                price_2022 = molecule_df['MAT Q3 2022 SU Avg Price USD MNF'].mean()
                price_2024 = molecule_df['MAT Q3 2024 SU Avg Price USD MNF'].mean()
                price_change = ((price_2024 - price_2022) / price_2022 * 100) if price_2022 != 0 else 0
                
                st.metric(
                    label=f"{selected_molecule_detail} Avg Price",
                    value=f"${price_2024:.2f}",
                    delta=f"{price_change:+.1f}% (3 yıl)"
                )
        
        with col4:
            if selected_molecule_detail != 'Tüm Moleküller':
                trend = molecule_df['USD_MNF_Trend'].iloc[0] if len(molecule_df) > 0 else 'N/A'
                trend_display = {
                    'A Artan': '📈 Artan',
                    'B Azalan': '📉 Azalan',
                    'C Dalgalı': '📊 Dalgalı',
                    'D Stabil': '⚖️ Stabil'
                }.get(trend, trend)
                
                st.metric(
                    label=f"{selected_molecule_detail} Trend",
                    value=trend_display
                )
        
        # Molecule trend chart
        fig_molecule = visualizer.create_molecule_analysis(filtered_df)
        st.plotly_chart(fig_molecule, use_container_width=True)
        
        # Molecule insights
        st.markdown('---')
        st.markdown('<h4>Molekül Bazlı İçgörüler</h4>', unsafe_allow_html=True)
        
        molecule_insights = insight_engine.generate_molecule_insights(filtered_df)
        
        for insight in molecule_insights[:5]:  # Show top 5 insights
            if insight['type'] == 'positive':
                st.success(f"**{insight['title']}**: {insight['message']}")
            elif insight['type'] == 'negative':
                st.error(f"**{insight['title']}**: {insight['message']}")
            elif insight['type'] == 'warning':
                st.warning(f"**{insight['title']}**: {insight['message']}")
            else:
                st.info(f"**{insight['title']}**: {insight['message']}")
        
        # Detailed molecule table
        st.markdown('---')
        st.markdown('<h4>Molekül Performans Tablosu (2022-2024)</h4>', unsafe_allow_html=True)
        
        molecule_performance = filtered_df.groupby('Molecule').agg({
            'MAT Q3 2022 USD MNF': 'sum',
            'MAT Q3 2023 USD MNF': 'sum',
            'MAT Q3 2024 USD MNF': 'sum',
            'USD_MNF_Growth_22_23': 'mean',
            'USD_MNF_Growth_23_24': 'mean',
            'USD_MNF_Growth_22_24': 'mean',
            'USD_MNF_Trend': 'first',
            'Price_Trend': 'first'
        }).reset_index()
        
        # Format for display
        display_mol = molecule_performance.copy()
        for col in ['MAT Q3 2022 USD MNF', 'MAT Q3 2023 USD MNF', 'MAT Q3 2024 USD MNF']:
            display_mol[col] = display_mol[col].apply(lambda x: f'${x:,.0f}')
        
        for col in ['USD_MNF_Growth_22_23', 'USD_MNF_Growth_23_24', 'USD_MNF_Growth_22_24']:
            display_mol[col] = display_mol[col].apply(lambda x: f'{x:+.1f}%')
        
        st.dataframe(
            display_mol.sort_values('MAT Q3 2024 USD MNF', ascending=False),
            use_container_width=True,
            height=400
        )
    
    # Tab 4: Corporation & Competition
    with tab4:
        st.markdown('<h2 class="sub-header">Corporation & Rekabet Analizi - Pazar Payı Trendleri</h2>', unsafe_allow_html=True)
        
        # Corporation market share chart
        fig_corp = visualizer.create_corporation_market_share(filtered_df)
        st.plotly_chart(fig_corp, use_container_width=True)
        
        # Corporation insights
        st.markdown('---')
        st.markdown('<h4>Rekabet İçgörüleri</h4>', unsafe_allow_html=True)
        
        corp_insights = insight_engine.generate_corporation_insights(filtered_df)
        
        for insight in corp_insights:
            if insight['type'] == 'positive':
                st.success(f"**{insight['title']}**: {insight['message']}")
            elif insight['type'] == 'negative':
                st.error(f"**{insight['title']}**: {insight['message']}")
            elif insight['type'] == 'warning':
                st.warning(f"**{insight['title']}**: {insight['message']}")
            else:
                st.info(f"**{insight['title']}**: {insight['message']}")
        
        # Market share change analysis
        st.markdown('---')
        st.markdown('<h4>Pazar Payı Değişim Analizi (2022-2024)</h4>', unsafe_allow_html=True)
        
        if 'Corporation' in filtered_df.columns:
            # Calculate market shares
            corp_data = filtered_df.groupby('Corporation').agg({
                'MAT Q3 2022 USD MNF': 'sum',
                'MAT Q3 2024 USD MNF': 'sum'
            }).reset_index()
            
            total_2022 = corp_data['MAT Q3 2022 USD MNF'].sum()
            total_2024 = corp_data['MAT Q3 2024 USD MNF'].sum()
            
            corp_data['Share_2022'] = (corp_data['MAT Q3 2022 USD MNF'] / total_2022 * 100) if total_2022 != 0 else 0
            corp_data['Share_2024'] = (corp_data['MAT Q3 2024 USD MNF'] / total_2024 * 100) if total_2024 != 0 else 0
            corp_data['Share_Change'] = corp_data['Share_2024'] - corp_data['Share_2022']
            
            # Create gainers and losers
            gainers = corp_data.nlargest(5, 'Share_Change')
            losers = corp_data.nsmallest(5, 'Share_Change')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('##### 🏆 En Büyük Kazananlar')
                for _, row in gainers.iterrows():
                    if row['Share_Change'] > 0:
                        st.info(f"**{row['Corporation']}**: +{row['Share_Change']:.2f}% (2022: {row['Share_2022']:.1f}% → 2024: {row['Share_2024']:.1f}%)")
            
            with col2:
                st.markdown('##### 📉 En Büyük Kaybedenler')
                for _, row in losers.iterrows():
                    if row['Share_Change'] < 0:
                        st.error(f"**{row['Corporation']}**: {row['Share_Change']:.2f}% (2022: {row['Share_2022']:.1f}% → 2024: {row['Share_2024']:.1f}%)")
        
        # Corporation performance table
        st.markdown('---')
        st.markdown('<h4>Corporation Performans Tablosu</h4>', unsafe_allow_html=True)
        
        if 'Corporation' in filtered_df.columns:
            corp_performance = filtered_df.groupby('Corporation').agg({
                'MAT Q3 2022 USD MNF': 'sum',
                'MAT Q3 2023 USD MNF': 'sum',
                'MAT Q3 2024 USD MNF': 'sum',
                'USD_MNF_Growth_22_23': 'mean',
                'USD_MNF_Growth_23_24': 'mean',
                'USD_MNF_Growth_22_24': 'mean'
            }).reset_index()
            
            # Calculate market share
            total_2022_corp = corp_performance['MAT Q3 2022 USD MNF'].sum()
            total_2024_corp = corp_performance['MAT Q3 2024 USD MNF'].sum()
            
            corp_performance['Share_2022'] = (corp_performance['MAT Q3 2022 USD MNF'] / total_2022_corp * 100) if total_2022_corp != 0 else 0
            corp_performance['Share_2024'] = (corp_performance['MAT Q3 2024 USD MNF'] / total_2024_corp * 100) if total_2024_corp != 0 else 0
            corp_performance['Share_Change'] = corp_performance['Share_2024'] - corp_performance['Share_2022']
            
            # Format for display
            display_corp = corp_performance.copy()
            for col in ['MAT Q3 2022 USD MNF', 'MAT Q3 2023 USD MNF', 'MAT Q3 2024 USD MNF']:
                display_corp[col] = display_corp[col].apply(lambda x: f'${x:,.0f}')
            
            for col in ['USD_MNF_Growth_22_23', 'USD_MNF_Growth_23_24', 'USD_MNF_Growth_22_24']:
                display_corp[col] = display_corp[col].apply(lambda x: f'{x:+.1f}%')
            
            for col in ['Share_2022', 'Share_2024', 'Share_Change']:
                display_corp[col] = display_corp[col].apply(lambda x: f'{x:+.1f}%')
            
            st.dataframe(
                display_corp.sort_values('MAT Q3 2024 USD MNF', ascending=False),
                use_container_width=True,
                height=400
            )
    
    # Tab 5: Specialty vs Non-Specialty
    with tab5:
        st.markdown('<h2 class="sub-header">Specialty vs Non-Specialty Ürün Analizi - 3 Yıllık Trend</h2>', unsafe_allow_html=True)
        
        # Specialty analysis chart
        fig_specialty = visualizer.create_specialty_vs_non_specialty(filtered_df)
        st.plotly_chart(fig_specialty, use_container_width=True)
        
        # Specialty insights
        st.markdown('---')
        st.markdown('<h4>Specialty Ürün İçgörüleri</h4>', unsafe_allow_html=True)
        
        specialty_insights = insight_engine.generate_specialty_insights(filtered_df)
        
        for insight in specialty_insights:
            if insight['type'] == 'positive':
                st.success(f"**{insight['title']}**: {insight['message']}")
            elif insight['type'] == 'negative':
                st.error(f"**{insight['title']}**: {insight['message']}")
            elif insight['type'] == 'warning':
                st.warning(f"**{insight['title']}**: {insight['message']}")
            else:
                st.info(f"**{insight['title']}**: {insight['message']}")
        
        # Premiumization analysis
        st.markdown('---')
        st.markdown('<h4>Premiumlaşma Trend Analizi</h4>', unsafe_allow_html=True)
        
        if 'Specialty Product' in filtered_df.columns:
            # Calculate price premium
            specialty_prices = filtered_df[filtered_df['Specialty Product'] == 'Yes']['MAT Q3 2024 SU Avg Price USD MNF'].mean()
            non_specialty_prices = filtered_df[filtered_df['Specialty Product'] != 'Yes']['MAT Q3 2024 SU Avg Price USD MNF'].mean()
            
            price_premium = ((specialty_prices - non_specialty_prices) / non_specialty_prices * 100) if non_specialty_prices != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Specialty Avg Price (2024)",
                    value=f"${specialty_prices:.2f}"
                )
            
            with col2:
                st.metric(
                    label="Non-Specialty Avg Price (2024)",
                    value=f"${non_specialty_prices:.2f}"
                )
            
            with col3:
                st.metric(
                    label="Price Premium",
                    value=f"{price_premium:.1f}%"
                )
            
            # Growth comparison
            specialty_growth = filtered_df[filtered_df['Specialty Product'] == 'Yes']['USD_MNF_Growth_22_24'].mean()
            non_specialty_growth = filtered_df[filtered_df['Specialty Product'] != 'Yes']['USD_MNF_Growth_22_24'].mean()
            
            st.markdown('##### 📈 Büyüme Karşılaştırması (2022-2024)')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Specialty Ürün Büyümesi",
                    value=f"{specialty_growth:.1f}%",
                    delta="3-yıl trend"
                )
            
            with col2:
                st.metric(
                    label="Non-Specialty Ürün Büyümesi",
                    value=f"{non_specialty_growth:.1f}%",
                    delta="3-yıl trend"
                )
    
    # Tab 6: Price & Mix Analysis
    with tab6:
        st.markdown('<h2 class="sub-header">Fiyat & Mix Analizi - Hacim vs Fiyat Etkisi</h2>', unsafe_allow_html=True)
        
        # Price-volume analysis chart
        fig_price_volume = visualizer.create_price_volume_analysis(filtered_df)
        st.plotly_chart(fig_price_volume, use_container_width=True)
        
        # Detailed price analysis
        st.markdown('---')
        st.markdown('<h4>Fiyat Trend Detay Analizi (2022-2024)</h4>', unsafe_allow_html=True)
        
        # Calculate price changes by segment
        price_analysis_data = []
        
        # By Country
        if 'Country' in filtered_df.columns:
            country_price = filtered_df.groupby('Country').agg({
                'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
                'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
            }).reset_index()
            
            country_price['Price_Change'] = ((country_price['MAT Q3 2024 SU Avg Price USD MNF'] - 
                                             country_price['MAT Q3 2022 SU Avg Price USD MNF']) / 
                                            country_price['MAT Q3 2022 SU Avg Price USD MNF'] * 100)
            
            top_price_increase = country_price.nlargest(5, 'Price_Change')
            for _, row in top_price_increase.iterrows():
                price_analysis_data.append({
                    'Segment': 'Country',
                    'Name': row['Country'],
                    'Price_2022': row['MAT Q3 2022 SU Avg Price USD MNF'],
                    'Price_2024': row['MAT Q3 2024 SU Avg Price USD MNF'],
                    'Change': row['Price_Change']
                })
        
        # By Corporation
        if 'Corporation' in filtered_df.columns:
            corp_price = filtered_df.groupby('Corporation').agg({
                'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
                'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
            }).reset_index()
            
            corp_price['Price_Change'] = ((corp_price['MAT Q3 2024 SU Avg Price USD MNF'] - 
                                          corp_price['MAT Q3 2022 SU Avg Price USD MNF']) / 
                                         corp_price['MAT Q3 2022 SU Avg Price USD MNF'] * 100)
            
            top_corp_price = corp_price.nlargest(3, 'Price_Change')
            for _, row in top_corp_price.iterrows():
                price_analysis_data.append({
                    'Segment': 'Corporation',
                    'Name': row['Corporation'],
                    'Price_2022': row['MAT Q3 2022 SU Avg Price USD MNF'],
                    'Price_2024': row['MAT Q3 2024 SU Avg Price USD MNF'],
                    'Change': row['Price_Change']
                })
        
        # Display price analysis
        if price_analysis_data:
            price_df = pd.DataFrame(price_analysis_data)
            
            # Create visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[f"{row['Segment']}: {row['Name']}" for _, row in price_df.iterrows()],
                y=price_df['Change'],
                marker_color=price_df['Change'].apply(lambda x: '#10b981' if x > 0 else '#ef4444'),
                text=[f"{x:+.1f}%" for x in price_df['Change']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=dict(
                    text='En Yüksek Fiyat Artışı Gösteren Segmentler (2022-2024)',
                    font=dict(size=16, color='#1e3d8c'),
                    x=0.5
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='Segment',
                    tickangle=45,
                    gridcolor='#e2e8f0'
                ),
                yaxis=dict(
                    title='Fiyat Değişimi %',
                    gridcolor='#e2e8f0',
                    ticksuffix='%'
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Volume-Price decomposition
        st.markdown('---')
        st.markdown('<h4>Hacim-Fiyat Ayrıştırması (2022-2024)</h4>', unsafe_allow_html=True)
        
        # Calculate total values
        total_usd_2022 = filtered_df['MAT Q3 2022 USD MNF'].sum()
        total_usd_2024 = filtered_df['MAT Q3 2024 USD MNF'].sum()
        total_units_2022 = filtered_df['MAT Q3 2022 Units'].sum()
        total_units_2024 = filtered_df['MAT Q3 2024 Units'].sum()
        avg_price_2022 = total_usd_2022 / total_units_2022 if total_units_2022 != 0 else 0
        avg_price_2024 = total_usd_2024 / total_units_2024 if total_units_2024 != 0 else 0
        
        # Calculate contributions
        volume_effect = (total_units_2024 - total_units_2022) * avg_price_2022
        price_effect = total_units_2024 * (avg_price_2024 - avg_price_2022)
        mix_effect = total_usd_2024 - total_usd_2022 - volume_effect - price_effect
        
        total_change = total_usd_2024 - total_usd_2022
        
        # Create decomposition chart
        fig = go.Figure()
        
        effects = ['Hacim Etkisi', 'Fiyat Etkisi', 'Mix Etkisi', 'Toplam Değişim']
        values = [volume_effect, price_effect, mix_effect, total_change]
        colors = ['#3182ce', '#10b981', '#8b5cf6', '#f59e0b']
        
        fig.add_trace(go.Bar(
            x=effects,
            y=values,
            marker_color=colors,
            text=[f'${v:,.0f}' for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=dict(
                text='USD MNF Değişim Ayrıştırması (2022-2024)',
                font=dict(size=16, color='#1e3d8c'),
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title='Etki Türü',
                gridcolor='#e2e8f0'
            ),
            yaxis=dict(
                title='USD Değişimi',
                gridcolor='#e2e8f0',
                tickprefix='$'
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display percentages
        if total_change != 0:
            volume_pct = (volume_effect / total_change * 100) if total_change != 0 else 0
            price_pct = (price_effect / total_change * 100) if total_change != 0 else 0
            mix_pct = (mix_effect / total_change * 100) if total_change != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Hacim Katkısı",
                    value=f"{volume_pct:.1f}%",
                    delta=f"${volume_effect:,.0f}"
                )
            
            with col2:
                st.metric(
                    label="Fiyat Katkısı",
                    value=f"{price_pct:.1f}%",
                    delta=f"${price_effect:,.0f}"
                )
            
            with col3:
                st.metric(
                    label="Mix Katkısı",
                    value=f"{mix_pct:.1f}%",
                    delta=f"${mix_effect:,.0f}"
                )
    
    # Tab 7: Pack/Strength/Size Analysis
    with tab7:
        st.markdown('<h2 class="sub-header">Pack / Strength / Size Analizi - 3 Yıllık Trendler</h2>', unsafe_allow_html=True)
        
        # Comprehensive analysis chart
        fig_pack_strength = visualizer.create_pack_strength_analysis(filtered_df)
        st.plotly_chart(fig_pack_strength, use_container_width=True)
        
        # Detailed analysis by attribute
        st.markdown('---')
        
        # Pack analysis
        if 'International Pack' in filtered_df.columns:
            st.markdown('<h4>Pack Bazlı Detay Analiz</h4>', unsafe_allow_html=True)
            
            pack_data = filtered_df.groupby('International Pack').agg({
                'MAT Q3 2022 USD MNF': 'sum',
                'MAT Q3 2023 USD MNF': 'sum',
                'MAT Q3 2024 USD MNF': 'sum',
                'MAT Q3 2022 Units': 'sum',
                'MAT Q3 2023 Units': 'sum',
                'MAT Q3 2024 Units': 'sum',
                'MAT Q3 2022 SU Avg Price USD MNF': 'mean',
                'MAT Q3 2023 SU Avg Price USD MNF': 'mean',
                'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
            }).reset_index()
            
            # Calculate growth rates
            pack_data['USD_Growth_22_24'] = ((pack_data['MAT Q3 2024 USD MNF'] - pack_data['MAT Q3 2022 USD MNF']) / 
                                            pack_data['MAT Q3 2022 USD MNF'].replace(0, np.nan)) * 100
            pack_data['Units_Growth_22_24'] = ((pack_data['MAT Q3 2024 Units'] - pack_data['MAT Q3 2022 Units']) / 
                                              pack_data['MAT Q3 2022 Units'].replace(0, np.nan)) * 100
            pack_data['Price_Growth_22_24'] = ((pack_data['MAT Q3 2024 SU Avg Price USD MNF'] - pack_data['MAT Q3 2022 SU Avg Price USD MNF']) / 
                                              pack_data['MAT Q3 2022 SU Avg Price USD MNF'].replace(0, np.nan)) * 100
            
            # Display top packs
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('##### 🚀 En Hızlı Büyüyen Pack\'lar')
                top_growth = pack_data.nlargest(5, 'USD_Growth_22_24')
                for _, row in top_growth.iterrows():
                    if row['USD_Growth_22_24'] > 0:
                        st.success(f"**{row['International Pack']}**: %{row['USD_Growth_22_24']:.1f} büyüme (3 yıl)")
            
            with col2:
                st.markdown('##### 💰 En Yüksek Fiyatlı Pack\'lar')
                top_price = pack_data.nlargest(5, 'MAT Q3 2024 SU Avg Price USD MNF')
                for _, row in top_price.iterrows():
                    st.info(f"**{row['International Pack']}**: ${row['MAT Q3 2024 SU Avg Price USD MNF']:.2f} avg price")
        
        # Strength analysis
        if 'International Strength' in filtered_df.columns:
            st.markdown('---')
            st.markdown('<h4>Strength Bazlı Detay Analiz</h4>', unsafe_allow_html=True)
            
            strength_data = filtered_df.groupby('International Strength').agg({
                'MAT Q3 2022 USD MNF': 'sum',
                'MAT Q3 2024 USD MNF': 'sum',
                'MAT Q3 2022 Units': 'sum',
                'MAT Q3 2024 Units': 'sum'
            }).reset_index()
            
            # Calculate market share
            total_usd_2024 = strength_data['MAT Q3 2024 USD MNF'].sum()
            strength_data['Share_2024'] = (strength_data['MAT Q3 2024 USD MNF'] / total_usd_2024 * 100) if total_usd_2024 != 0 else 0
            
            # Create visualization
            fig = go.Figure(data=[
                go.Pie(
                    labels=strength_data['International Strength'],
                    values=strength_data['Share_2024'],
                    hole=0.4,
                    textinfo='label+percent',
                    marker=dict(colors=px.colors.qualitative.Set3)
                )
            ])
            
            fig.update_layout(
                title=dict(
                    text='Strength Bazlı Pazar Payı Dağılımı (2024)',
                    font=dict(size=16, color='#1e3d8c'),
                    x=0.5
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Size analysis
        if 'International Size' in filtered_df.columns:
            st.markdown('---')
            st.markdown('<h4>Size Bazlı Detay Analiz</h4>', unsafe_allow_html=True)
            
            size_data = filtered_df.groupby('International Size').agg({
                'MAT Q3 2022 USD MNF': 'sum',
                'MAT Q3 2023 USD MNF': 'sum',
                'MAT Q3 2024 USD MNF': 'sum',
                'MAT Q3 2022 Units': 'sum',
                'MAT Q3 2023 Units': 'sum',
                'MAT Q3 2024 Units': 'sum'
            }).reset_index()
            
            # Create trend analysis
            fig = go.Figure()
            
            sizes = size_data['International Size'].unique()
            for size in sizes:
                size_df = size_data[size_data['International Size'] == size]
                if not size_df.empty:
                    fig.add_trace(go.Scatter(
                        x=['2022', '2023', '2024'],
                        y=[size_df['MAT Q3 2022 USD MNF'].iloc[0], 
                          size_df['MAT Q3 2023 USD MNF'].iloc[0],
                          size_df['MAT Q3 2024 USD MNF'].iloc[0]],
                        mode='lines+markers',
                        name=size,
                        line=dict(width=3)
                    ))
            
            fig.update_layout(
                title=dict(
                    text='Size Bazlı USD MNF Trendleri (2022-2024)',
                    font=dict(size=16, color='#1e3d8c'),
                    x=0.5
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='Year',
                    gridcolor='#e2e8f0'
                ),
                yaxis=dict(
                    title='USD MNF',
                    gridcolor='#e2e8f0',
                    tickprefix='$'
                ),
                height=400,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 8: Insight Engine
    with tab8:
        st.markdown('<h2 class="sub-header">🤖 Otomatik İçgörü Motoru - 3 Yıllık Trend Analizi</h2>', unsafe_allow_html=True)
        
        # Generate comprehensive insights
        st.markdown('---')
        st.markdown('<h4>📊 Global Trend İçgörüleri</h4>', unsafe_allow_html=True)
        
        all_insights = []
        
        # Get insights from all engines
        exec_insights = insight_engine.generate_executive_insights(filtered_df)
        molecule_insights = insight_engine.generate_molecule_insights(filtered_df)
        corp_insights = insight_engine.generate_corporation_insights(filtered_df)
        specialty_insights = insight_engine.generate_specialty_insights(filtered_df)
        
        all_insights.extend(exec_insights)
        all_insights.extend(molecule_insights)
        all_insights.extend(corp_insights)
        all_insights.extend(specialty_insights)
        
        # Display insights in categorized sections
        insight_categories = {
            'positive': [],
            'negative': [],
            'warning': [],
            'info': []
        }
        
        for insight in all_insights:
            insight_categories[insight['type']].append(insight)
        
        # Positive insights
        if insight_categories['positive']:
            st.markdown('##### ✅ Olumlu Trendler')
            for insight in insight_categories['positive'][:5]:  # Limit to top 5
                st.success(f"**{insight['title']}**: {insight['message']}")
        
        # Warning insights
        if insight_categories['warning']:
            st.markdown('##### ⚠️ Dikkat Gerektiren Trendler')
            for insight in insight_categories['warning'][:3]:
                st.warning(f"**{insight['title']}**: {insight['message']}")
        
        # Negative insights
        if insight_categories['negative']:
            st.markdown('##### 🔴 Riskli Trendler')
            for insight in insight_categories['negative'][:3]:
                st.error(f"**{insight['title']}**: {insight['message']}")
        
        # Informational insights
        if insight_categories['info']:
            st.markdown('##### ℹ️ Bilgilendirici İçgörüler')
            for insight in insight_categories['info'][:5]:
                st.info(f"**{insight['title']}**: {insight['message']}")
        
        # Trend classification summary
        st.markdown('---')
        st.markdown('<h4>📈 Trend Sınıflandırma Özeti</h4>', unsafe_allow_html=True)
        
        # Calculate trend distributions
        if 'USD_MNF_Trend' in filtered_df.columns:
            trend_dist = filtered_df['USD_MNF_Trend'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Artan', 'Azalan', 'Dalgalı', 'Stabil'],
                    values=[
                        trend_dist.get('A Artan', 0),
                        trend_dist.get('B Azalan', 0),
                        trend_dist.get('C Dalgalı', 0),
                        trend_dist.get('D Stabil', 0)
                    ],
                    hole=0.3,
                    marker=dict(colors=['#10b981', '#ef4444', '#f59e0b', '#6b7280'])
                )
            ])
            
            fig.update_layout(
                title=dict(
                    text='USD MNF Trend Dağılımı (Tüm Ürünler)',
                    font=dict(size=16, color='#1e3d8c'),
                    x=0.5
                ),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Key performance indicators
        st.markdown('---')
        st.markdown('<h4>🎯 Kritik Performans Göstergeleri (KPIs)</h4>', unsafe_allow_html=True)
        
        # Calculate KPIs
        kpi_data = []
        
        # USD MNF Growth Leaders
        if 'Country' in filtered_df.columns and 'USD_MNF_Growth_22_24' in filtered_df.columns:
            country_growth = filtered_df.groupby('Country')['USD_MNF_Growth_22_24'].mean().reset_index()
            top_growth_country = country_growth.nlargest(1, 'USD_MNF_Growth_22_24')
            if not top_growth_country.empty:
                kpi_data.append({
                    'KPI': 'En Hızlı Büyüyen Ülke',
                    'Değer': f"%{top_growth_country['USD_MNF_Growth_22_24'].iloc[0]:.1f}",
                    'Detay': top_growth_country['Country'].iloc[0]
                })
        
        # Top Molecule
        if 'Molecule' in filtered_df.columns:
            molecule_sales = filtered_df.groupby('Molecule')['MAT Q3 2024 USD MNF'].sum().reset_index()
            top_molecule = molecule_sales.nlargest(1, 'MAT Q3 2024 USD MNF')
            if not top_molecule.empty:
                kpi_data.append({
                    'KPI': 'En Yüksek Satışlı Molekül',
                    'Değer': f"${top_molecule['MAT Q3 2024 USD MNF'].iloc[0]:,.0f}",
                    'Detay': top_molecule['Molecule'].iloc[0]
                })
        
        # Market Share Leader
        if 'Corporation' in filtered_df.columns:
            corp_sales = filtered_df.groupby('Corporation')['MAT Q3 2024 USD MNF'].sum().reset_index()
            total_sales = corp_sales['MAT Q3 2024 USD MNF'].sum()
            corp_sales['Share'] = (corp_sales['MAT Q3 2024 USD MNF'] / total_sales * 100) if total_sales != 0 else 0
            top_corp = corp_sales.nlargest(1, 'Share')
            if not top_corp.empty:
                kpi_data.append({
                    'KPI': 'Pazar Lideri',
                    'Değer': f"%{top_corp['Share'].iloc[0]:.1f}",
                    'Detay': top_corp['Corporation'].iloc[0]
                })
        
        # Display KPIs
        if kpi_data:
            cols = st.columns(len(kpi_data))
            for idx, (col, kpi) in enumerate(zip(cols, kpi_data)):
                with col:
                    st.metric(
                        label=kpi['KPI'],
                        value=kpi['Değer'],
                        delta=kpi['Detay']
                    )
        
        # Custom insight generation
        st.markdown('---')
        st.markdown('<h4>🔍 Özel İçgörü Oluşturma</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            insight_type = st.selectbox(
                'İçgörü Tipi',
                ['Büyüme Analizi', 'Fiyat Trendleri', 'Pazar Payı Değişimi', 'Specialty Performansı']
            )
        
        with col2:
            metric_type = st.selectbox(
                'Metrik Tipi',
                ['USD MNF', 'Units', 'Standard Units', 'Avg Price']
            )
        
        if st.button('İçgörü Oluştur', type='primary'):
            with st.spinner('İçgörü oluşturuluyor...'):
                # Generate custom insight based on selections
                if insight_type == 'Büyüme Analizi':
                    if metric_type == 'USD MNF':
                        # Analyze USD MNF growth
                        growth_data = filtered_df.groupby('Country').agg({
                            'USD_MNF_Growth_22_24': 'mean',
                            'MAT Q3 2024 USD MNF': 'sum'
                        }).reset_index()
                        
                        top_growth = growth_data.nlargest(3, 'USD_MNF_Growth_22_24')
                        
                        st.info(f"**Büyüme Liderleri**: {', '.join(top_growth['Country'].tolist())} ülkeleri 3 yılda ortalama %{top_growth['USD_MNF_Growth_22_24'].mean():.1f} USD MNF büyümesi gösterdi.")
                
                elif insight_type == 'Fiyat Trendleri':
                    if metric_type == 'Avg Price':
                        # Analyze price trends
                        price_data = filtered_df.groupby('Molecule').agg({
                            'SU_Price_Change_22_24': 'mean',
                            'MAT Q3 2024 SU Avg Price USD MNF': 'mean'
                        }).reset_index()
                        
                        top_price_increase = price_data.nlargest(3, 'SU_Price_Change_22_24')
                        
                        st.info(f"**Fiyat Artış Liderleri**: {', '.join(top_price_increase['Molecule'].tolist())} molekülleri 3 yılda ortalama %{top_price_increase['SU_Price_Change_22_24'].mean():.1f} fiyat artışı gösterdi.")
                
                st.success("✅ Özel içgörü başarıyla oluşturuldu!")
    
    # Footer
    st.markdown('---')
    st.markdown(
        '<div style="text-align: center; color: #718096; font-size: 0.9rem;">'
        'Pharma Commercial Analytics Platform v2.0 • Enterprise-Grade 3-Year Trend Analysis • '
        f'Data as of {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
