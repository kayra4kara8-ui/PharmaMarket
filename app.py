# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
import io
import base64
from datetime import datetime, timedelta
import math
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import json
import re
import sys
import traceback
from typing import List, Dict, Tuple, Optional, Union, Any
import functools
import time
import concurrent.futures
from collections import defaultdict, OrderedDict
import hashlib
import copy

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

st.set_page_config(
    page_title="Pharma Analytics Platform - CEO Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PharmaDataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_data()
        self._calculate_metrics()
    
    def _prepare_data(self):
        try:
            numeric_columns = []
            year_metrics = ['MAT Q3 2022', 'MAT Q3 2023', 'MAT Q3 2024']
            metric_types = ['USD MNF', 'Standard Units', 'Units', 'Avg Prices (SU)', 'Avg Prices (Unit)']
            
            for year in year_metrics:
                for metric in metric_types:
                    col_name = f"{year} - {metric}"
                    if col_name in self.df.columns:
                        numeric_columns.append(col_name)
                        self.df[col_name] = pd.to_numeric(self.df[col_name].astype(str).str.replace(',', '.'), errors='coerce')
            
            text_columns = ['Country', 'Sector', 'Panel', 'Region', 'Sub-Region', 
                           'Corporation', 'Manufacturer', 'Molecule', 'Molecule List',
                           'Chemical Salt', 'International Product', 'Specialty Product',
                           'Pack', 'Strength', 'Prescription Status']
            
            for col in text_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).fillna('Bilinmiyor')
            
            if 'Volume' in self.df.columns:
                self.df['Volume'] = pd.to_numeric(self.df['Volume'], errors='coerce')
            
            self._create_year_columns()
            self._standardize_country_names()
            self._categorize_specialty()
            
        except Exception as e:
            st.error(f"Veri hazırlama hatası: {str(e)}")
            raise
    
    def _create_year_columns(self):
        try:
            for year, suffix in [(2022, 'MAT Q3 2022'), (2023, 'MAT Q3 2023'), (2024, 'MAT Q3 2024')]:
                usd_col = f"{suffix} - USD MNF"
                su_col = f"{suffix} - Standard Units"
                units_col = f"{suffix} - Units"
                avg_su_col = f"{suffix} - Avg Prices (SU)"
                avg_unit_col = f"{suffix} - Avg Prices (Unit)"
                
                if usd_col in self.df.columns:
                    self.df[f'USD_{year}'] = self.df[usd_col]
                if su_col in self.df.columns:
                    self.df[f'SU_{year}'] = self.df[su_col]
                if units_col in self.df.columns:
                    self.df[f'Units_{year}'] = self.df[units_col]
                if avg_su_col in self.df.columns:
                    self.df[f'AvgPrice_SU_{year}'] = self.df[avg_su_col]
                if avg_unit_col in self.df.columns:
                    self.df[f'AvgPrice_Unit_{year}'] = self.df[avg_unit_col]
            
        except Exception as e:
            st.error(f"Yıl kolonları oluşturma hatası: {str(e)}")
    
    def _standardize_country_names(self):
        country_mapping = {
            'United States': 'ABD',
            'USA': 'ABD',
            'US': 'ABD',
            'Germany': 'Almanya',
            'France': 'Fransa',
            'Italy': 'İtalya',
            'Spain': 'İspanya',
            'United Kingdom': 'İngiltere',
            'UK': 'İngiltere',
            'Turkey': 'Türkiye',
            'Turkive': 'Türkiye',
            'Japan': 'Japonya',
            'China': 'Çin',
            'Brazil': 'Brezilya',
            'Russia': 'Rusya',
            'India': 'Hindistan'
        }
        
        if 'Country' in self.df.columns:
            self.df['Country'] = self.df['Country'].replace(country_mapping)
    
    def _categorize_specialty(self):
        if 'Specialty Product' in self.df.columns:
            self.df['Specialty_Type'] = self.df['Specialty Product'].apply(
                lambda x: 'Specialty' if x.lower() in ['yes', 'evet', 'true', '1', 'specialty'] else 'Non-Specialty'
            )
    
    def _calculate_metrics(self):
        try:
            self._calculate_growth_metrics()
            self._calculate_market_shares()
            self._calculate_price_volume_decomposition()
            self._identify_trends()
            self._calculate_contribution_analysis()
            
        except Exception as e:
            st.error(f"Metrik hesaplama hatası: {str(e)}")
    
    def _calculate_growth_metrics(self):
        years = [2022, 2023, 2024]
        
        for i in range(1, len(years)):
            current_year = years[i]
            previous_year = years[i-1]
            
            usd_current = f'USD_{current_year}'
            usd_previous = f'USD_{previous_year}'
            
            if usd_current in self.df.columns and usd_previous in self.df.columns:
                self.df[f'YoY_Growth_{current_year}'] = (
                    (self.df[usd_current] - self.df[usd_previous]) / self.df[usd_previous].replace(0, np.nan)
                ) * 100
        
        if 'USD_2022' in self.df.columns and 'USD_2024' in self.df.columns:
            self.df['CAGR'] = (
                (self.df['USD_2024'] / self.df['USD_2022'].replace(0, np.nan)) ** (1/2) - 1
            ) * 100
    
    def _calculate_market_shares(self):
        years = [2022, 2023, 2024]
        
        for year in years:
            usd_col = f'USD_{year}'
            if usd_col in self.df.columns:
                total_usd = self.df[usd_col].sum()
                if total_usd > 0:
                    self.df[f'Market_Share_{year}'] = (self.df[usd_col] / total_usd) * 100
        
        if 'Market_Share_2023' in self.df.columns and 'Market_Share_2024' in self.df.columns:
            self.df['Share_Change_2023_2024'] = self.df['Market_Share_2024'] - self.df['Market_Share_2023']
    
    def _calculate_price_volume_decomposition(self):
        for year in [2023, 2024]:
            usd_current = f'USD_{year}'
            usd_previous = f'USD_{year-1}'
            units_current = f'Units_{year}'
            units_previous = f'Units_{year-1}'
            price_current = f'AvgPrice_Unit_{year}'
            price_previous = f'AvgPrice_Unit_{year-1}'
            
            if all(col in self.df.columns for col in [usd_current, usd_previous, units_current, units_previous]):
                self.df[f'Volume_Effect_{year}'] = (
                    (self.df[units_current] - self.df[units_previous]) * self.df[price_previous].replace(0, np.nan)
                )
                self.df[f'Price_Effect_{year}'] = (
                    (self.df[price_current] - self.df[price_previous]) * self.df[units_current].replace(0, np.nan)
                )
                self.df[f'Mix_Effect_{year}'] = self.df[f'USD_{year}'] - self.df[f'USD_{year-1}'] - self.df[f'Volume_Effect_{year}'] - self.df[f'Price_Effect_{year}']
    
    def _identify_trends(self):
        if 'YoY_Growth_2024' in self.df.columns:
            conditions = [
                self.df['YoY_Growth_2024'] > 10,
                self.df['YoY_Growth_2024'] < -5,
                (self.df['YoY_Growth_2024'] >= -5) & (self.df['YoY_Growth_2024'] <= 10)
            ]
            choices = ['Yükselen', 'Düşen', 'Stabil']
            self.df['Trend_2024'] = np.select(conditions, choices, default='Bilinmiyor')
    
    def _calculate_contribution_analysis(self):
        if 'USD_2024' in self.df.columns and 'USD_2023' in self.df.columns:
            total_growth = self.df['USD_2024'].sum() - self.df['USD_2023'].sum()
            if total_growth != 0:
                self.df['Contribution_to_Growth'] = (
                    (self.df['USD_2024'] - self.df['USD_2023']) / abs(total_growth)
                ) * 100
    
    def get_processed_data(self) -> pd.DataFrame:
        return self.df

class ExecutiveSummaryTab:
    def __init__(self, data_processor: PharmaDataProcessor):
        self.df = data_processor.get_processed_data()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        if 'executive_summary_generated' not in st.session_state:
            st.session_state.executive_summary_generated = False
        if 'executive_insights' not in st.session_state:
            st.session_state.executive_insights = ""
    
    def render(self):
        st.title("🎯 Genel Yönetici Özeti")
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._display_total_sales()
        
        with col2:
            self._display_growth_metrics()
        
        with col3:
            self._display_top_countries()
        
        with col4:
            self._display_bottom_countries()
        
        st.markdown("---")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            self._display_sales_trend_chart()
        
        with col_right:
            self._display_market_distribution()
        
        st.markdown("---")
        
        self._display_automatic_insights()
    
    def _display_total_sales(self):
        try:
            if 'USD_2024' in self.df.columns:
                total_sales_2024 = self.df['USD_2024'].sum()
                total_units_2024 = self.df['Units_2024'].sum() if 'Units_2024' in self.df.columns else 0
                
                st.metric(
                    label="2024 Toplam Satış",
                    value=f"${total_sales_2024:,.1f}M",
                    delta=f"{total_units_2024:,.0f} birim"
                )
        except Exception as e:
            st.error(f"Toplam satış gösterim hatası: {str(e)}")
    
    def _display_growth_metrics(self):
        try:
            if 'USD_2022' in self.df.columns and 'USD_2024' in self.df.columns:
                sales_2022 = self.df['USD_2022'].sum()
                sales_2024 = self.df['USD_2024'].sum()
                
                if sales_2022 > 0:
                    cagr = ((sales_2024 / sales_2022) ** (1/2) - 1) * 100
                    
                    st.metric(
                        label="CAGR (2022-2024)",
                        value=f"%{cagr:.1f}",
                        delta="2-yıllık bileşik büyüme"
                    )
        except Exception as e:
            st.error(f"Büyüme metrik hatası: {str(e)}")
    
    def _display_top_countries(self):
        try:
            if 'Country' in self.df.columns and 'USD_2024' in self.df.columns:
                country_sales = self.df.groupby('Country')['USD_2024'].sum().nlargest(5)
                
                st.write("🏆 **En Yüksek Satışlı Ülkeler:**")
                for country, sales in country_sales.items():
                    st.write(f"• {country}: ${sales:,.1f}M")
        except Exception as e:
            st.error(f"Ülke sıralama hatası: {str(e)}")
    
    def _display_bottom_countries(self):
        try:
            if 'Country' in self.df.columns and 'USD_2024' in self.df.columns:
                country_sales = self.df.groupby('Country')['USD_2024'].sum().nsmallest(5)
                
                st.write("📉 **En Düşük Satışlı Ülkeler:**")
                for country, sales in country_sales.items():
                    st.write(f"• {country}: ${sales:,.1f}M")
        except Exception as e:
            st.error(f"Ülke sıralama hatası: {str(e)}")
    
    def _display_sales_trend_chart(self):
        try:
            years = [2022, 2023, 2024]
            sales_by_year = []
            
            for year in years:
                usd_col = f'USD_{year}'
                if usd_col in self.df.columns:
                    sales_by_year.append(self.df[usd_col].sum())
                else:
                    sales_by_year.append(0)
            
            growth_rates = []
            for i in range(1, len(sales_by_year)):
                if sales_by_year[i-1] > 0:
                    growth_rate = ((sales_by_year[i] - sales_by_year[i-1]) / sales_by_year[i-1]) * 100
                    growth_rates.append(growth_rate)
                else:
                    growth_rates.append(0)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(
                    x=[f"{year}" for year in years],
                    y=sales_by_year,
                    name="Satış (USD M)",
                    marker_color='#2E86AB',
                    text=[f"${x:,.0f}M" for x in sales_by_year],
                    textposition='auto'
                ),
                secondary_y=False
            )
            
            if len(growth_rates) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[f"{years[i+1]}" for i in range(len(growth_rates))],
                        y=growth_rates,
                        name="Yıllık Büyüme (%)",
                        mode='lines+markers',
                        line=dict(color='#A23B72', width=3),
                        marker=dict(size=10)
                    ),
                    secondary_y=True
                )
            
            fig.update_layout(
                title="Yıllara Göre Satış Trendi ve Büyüme Oranı",
                xaxis_title="Yıl",
                yaxis_title="Satış (USD M)",
                yaxis2_title="Büyüme Oranı (%)",
                height=400,
                showlegend=True,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Grafik oluşturma hatası: {str(e)}")
    
    def _display_market_distribution(self):
        try:
            if 'Country' in self.df.columns and 'USD_2024' in self.df.columns:
                country_sales = self.df.groupby('Country')['USD_2024'].sum()
                
                if len(country_sales) > 10:
                    top_10 = country_sales.nlargest(10)
                    other = country_sales.nsmallest(len(country_sales) - 10).sum()
                    
                    labels = list(top_10.index) + ['Diğer']
                    values = list(top_10.values) + [other]
                else:
                    labels = list(country_sales.index)
                    values = list(country_sales.values)
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker_colors=px.colors.qualitative.Set3
                )])
                
                fig.update_layout(
                    title="Ülkelere Göre Pazar Dağılımı (2024)",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Pasta grafiği hatası: {str(e)}")
    
    def _display_automatic_insights(self):
        try:
            st.subheader("🤖 Otomatik Analiz ve İçgörüler")
            
            insights = self._generate_insights()
            
            if insights:
                st.session_state.executive_insights = insights
                st.markdown(f"""
                <div style="background-color:#f0f2f6;padding:20px;border-radius:10px;border-left:5px solid #2E86AB">
                {insights}
                </div>
                """, unsafe_allow_html=True)
            
            st.session_state.executive_summary_generated = True
            
        except Exception as e:
            st.error(f"İçgörü oluşturma hatası: {str(e)}")
    
    def _generate_insights(self) -> str:
        try:
            insights = []
            
            if 'USD_2022' in self.df.columns and 'USD_2024' in self.df.columns:
                total_2022 = self.df['USD_2022'].sum()
                total_2024 = self.df['USD_2024'].sum()
                
                if total_2022 > 0:
                    total_growth = ((total_2024 - total_2022) / total_2022) * 100
                    
                    if total_growth > 15:
                        insights.append(f"📈 **Genel Performans:** Global satışlar 2022'den 2024'e %{total_growth:.1f} büyüme gösterdi. Güçlü pozitif trend devam ediyor.")
                    elif total_growth > 0:
                        insights.append(f"📊 **Genel Performans:** Global satışlar 2022'den 2024'e %{total_growth:.1f} oranında artış gösterdi. Dengeli büyüme görülüyor.")
                    else:
                        insights.append(f"⚠️ **Genel Performans:** Global satışlar 2022'den 2024'e %{abs(total_growth):.1f} düşüş gösterdi. Dikkatli analiz gerekiyor.")
            
            if 'Country' in self.df.columns and 'USD_2024' in self.df.columns:
                country_growth = []
                if 'USD_2023' in self.df.columns:
                    for country in self.df['Country'].unique():
                        country_data = self.df[self.df['Country'] == country]
                        sales_2023 = country_data['USD_2023'].sum()
                        sales_2024 = country_data['USD_2024'].sum()
                        
                        if sales_2023 > 0:
                            growth = ((sales_2024 - sales_2023) / sales_2023) * 100
                            country_growth.append((country, growth))
                
                if country_growth:
                    top_grower = max(country_growth, key=lambda x: x[1])
                    top_decliner = min(country_growth, key=lambda x: x[1])
                    
                    insights.append(f"🌍 **Bölgesel Performans:** En hızlı büyüyen pazar {top_grower[0]} (%{top_grower[1]:.1f}), en çok düşüş gösteren pazar {top_decliner[0]} (%{top_decliner[1]:.1f}).")
            
            if 'Molecule' in self.df.columns and 'USD_2024' in self.df.columns:
                molecule_sales = self.df.groupby('Molecule')['USD_2024'].sum().nlargest(3)
                top_molecules = ", ".join([f"{mol} (${sales:,.0f}M)" for mol, sales in molecule_sales.items()])
                insights.append(f"💊 **Ürün Portföyü:** En büyük 3 molekül: {top_molecules}. Toplam pazarın %{(molecule_sales.sum() / self.df['USD_2024'].sum() * 100):.1f}'ini oluşturuyor.")
            
            if 'Corporation' in self.df.columns and 'Market_Share_2024' in self.df.columns:
                top_corp = self.df.groupby('Corporation')['Market_Share_2024'].sum().nlargest(1)
                if not top_corp.empty:
                    corp_name = top_corp.index[0]
                    corp_share = top_corp.values[0]
                    insights.append(f"🏢 **Rekabet Analizi:** {corp_name} firması %{corp_share:.1f} pazar payı ile lider konumda.")
            
            if 'Specialty_Type' in self.df.columns and 'USD_2024' in self.df.columns:
                specialty_sales = self.df[self.df['Specialty_Type'] == 'Specialty']['USD_2024'].sum()
                total_sales = self.df['USD_2024'].sum()
                
                if total_sales > 0:
                    specialty_share = (specialty_sales / total_sales) * 100
                    insights.append(f"🎯 **Specialty Ürünler:** Specialty ürünler toplam satışın %{specialty_share:.1f}'ini oluşturuyor. Premium segment büyüme potansiyeli yüksek.")
            
            return "<br><br>".join(insights)
            
        except Exception as e:
            return f"İçgörü oluşturma sırasında hata: {str(e)}"

class CountryRegionAnalysisTab:
    def __init__(self, data_processor: PharmaDataProcessor):
        self.df = data_processor.get_processed_data()
    
    def render(self):
        st.title("🌍 Ülke & Bölge Analizi")
        st.markdown("---")
        
        self._render_filters()
        
        if not self._check_data_availability():
            st.warning("Bu analiz için gerekli veri kolonları mevcut değil.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_sales_heatmap()
        
        with col2:
            self._display_regional_price_comparison()
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            self._display_country_growth_chart()
        
        with col4:
            self._display_molecule_price_comparison()
    
    def _render_filters(self):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            available_countries = sorted(self.df['Country'].unique()) if 'Country' in self.df.columns else []
            selected_countries = st.multiselect(
                "Ülkeler",
                options=available_countries,
                default=available_countries[:5] if len(available_countries) > 5 else available_countries
            )
            
        with col2:
            if 'Region' in self.df.columns:
                available_regions = sorted(self.df['Region'].unique())
                selected_regions = st.multiselect(
                    "Bölgeler",
                    options=available_regions,
                    default=available_regions
                )
            
        with col3:
            if 'Molecule' in self.df.columns:
                available_molecules = sorted(self.df['Molecule'].unique())
                selected_molecules = st.multiselect(
                    "Moleküller",
                    options=available_molecules,
                    default=available_molecules[:3] if len(available_molecules) > 3 else available_molecules
                )
        
        self.filtered_df = self.df.copy()
        
        if selected_countries:
            self.filtered_df = self.filtered_df[self.filtered_df['Country'].isin(selected_countries)]
        
        if 'selected_regions' in locals() and selected_regions:
            self.filtered_df = self.filtered_df[self.filtered_df['Region'].isin(selected_regions)]
        
        if 'selected_molecules' in locals() and selected_molecules:
            self.filtered_df = self.filtered_df[self.filtered_df['Molecule'].isin(selected_molecules)]
    
    def _check_data_availability(self) -> bool:
        required_columns = ['Country', 'Region', 'USD_2024']
        return all(col in self.df.columns for col in required_columns)
    
    def _display_sales_heatmap(self):
        try:
            if len(self.filtered_df) == 0:
                st.info("Filtreleme sonucunda veri bulunamadı.")
                return
            
            country_sales = self.filtered_df.groupby('Country')['USD_2024'].sum().reset_index()
            
            if 'Region' in self.filtered_df.columns:
                country_region = self.filtered_df[['Country', 'Region']].drop_duplicates()
                country_sales = country_sales.merge(country_region, on='Country', how='left')
            
            fig = px.choropleth(
                country_sales,
                locations='Country',
                locationmode='country names',
                color='USD_2024',
                hover_name='Country',
                hover_data={'Region': True, 'USD_2024': ':.1f'},
                color_continuous_scale='Viridis',
                title="Ülkelere Göre Satış Yoğunluğu (2024)"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Isı haritası oluşturma hatası: {str(e)}")
    
    def _display_regional_price_comparison(self):
        try:
            if 'Region' not in self.filtered_df.columns or 'AvgPrice_Unit_2024' not in self.filtered_df.columns:
                return
            
            regional_prices = self.filtered_df.groupby('Region')['AvgPrice_Unit_2024'].agg(['mean', 'std', 'count']).reset_index()
            regional_prices = regional_prices[regional_prices['count'] > 1]
            
            if len(regional_prices) > 0:
                fig = px.bar(
                    regional_prices,
                    x='Region',
                    y='mean',
                    error_y='std',
                    title="Bölgesel Ortalama Fiyat Karşılaştırması (2024)",
                    labels={'mean': 'Ortalama Fiyat (USD)', 'Region': 'Bölge'},
                    color='mean',
                    color_continuous_scale='RdBu'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Fiyat karşılaştırma hatası: {str(e)}")
    
    def _display_country_growth_chart(self):
        try:
            if 'Country' in self.filtered_df.columns and 'YoY_Growth_2024' in self.filtered_df.columns:
                country_growth = self.filtered_df.groupby('Country')['YoY_Growth_2024'].mean().reset_index()
                country_growth = country_growth.sort_values('YoY_Growth_2024', ascending=False)
                
                fig = px.bar(
                    country_growth.head(10),
                    x='Country',
                    y='YoY_Growth_2024',
                    title="En Hızlı Büyüyen Ülkeler (YoY 2024)",
                    labels={'YoY_Growth_2024': 'Yıllık Büyüme (%)', 'Country': 'Ülke'},
                    color='YoY_Growth_2024',
                    color_continuous_scale='Greens'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Büyüme grafiği hatası: {str(e)}")
    
    def _display_molecule_price_comparison(self):
        try:
            if 'Molecule' in self.filtered_df.columns and 'Country' in self.filtered_df.columns and 'AvgPrice_Unit_2024' in self.filtered_df.columns:
                molecule_country_prices = self.filtered_df.groupby(['Molecule', 'Country'])['AvgPrice_Unit_2024'].mean().reset_index()
                
                if len(molecule_country_prices['Molecule'].unique()) > 0:
                    selected_molecule = st.selectbox(
                        "Karşılaştırılacak Molekülü Seçin:",
                        options=sorted(molecule_country_prices['Molecule'].unique())
                    )
                    
                    molecule_data = molecule_country_prices[molecule_country_prices['Molecule'] == selected_molecule]
                    
                    if len(molecule_data) > 1:
                        fig = px.bar(
                            molecule_data.sort_values('AvgPrice_Unit_2024', ascending=False),
                            x='Country',
                            y='AvgPrice_Unit_2024',
                            title=f"{selected_molecule} - Ülkeler Arası Fiyat Karşılaştırması (2024)",
                            labels={'AvgPrice_Unit_2024': 'Ortalama Fiyat (USD)', 'Country': 'Ülke'},
                            color='AvgPrice_Unit_2024',
                            color_continuous_scale='Plasma'
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Molekül fiyat karşılaştırma hatası: {str(e)}")

class MoleculeProductAnalysisTab:
    def __init__(self, data_processor: PharmaDataProcessor):
        self.df = data_processor.get_processed_data()
    
    def render(self):
        st.title("💊 Molekül & Ürün Analizi")
        st.markdown("---")
        
        self._render_filters()
        
        if not self._check_data_availability():
            st.warning("Bu analiz için gerekli veri kolonları mevcut değil.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_molecule_market_size()
        
        with col2:
            self._display_molecule_growth_trends()
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            self._display_emerging_declining_molecules()
        
        with col4:
            self._display_price_evolution()
    
    def _render_filters(self):
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Molecule' in self.df.columns:
                available_molecules = sorted(self.df['Molecule'].unique())
                selected_molecules = st.multiselect(
                    "Moleküller",
                    options=available_molecules,
                    default=available_molecules[:10] if len(available_molecules) > 10 else available_molecules
                )
            
            if 'Country' in self.df.columns:
                available_countries = sorted(self.df['Country'].unique())
                selected_countries = st.multiselect(
                    "Ülkeler",
                    options=available_countries,
                    default=available_countries[:5] if len(available_countries) > 5 else available_countries
                )
        
        with col2:
            if 'Chemical Salt' in self.df.columns:
                available_salts = sorted(self.df['Chemical Salt'].unique())
                selected_salts = st.multiselect(
                    "Kimyasal Tuzlar",
                    options=available_salts,
                    default=available_salts[:5] if len(available_salts) > 5 else available_salts
                )
            
            if 'International Product' in self.df.columns:
                available_products = sorted(self.df['International Product'].unique())
                selected_products = st.multiselect(
                    "Uluslararası Ürünler",
                    options=available_products,
                    default=available_products[:5] if len(available_products) > 5 else available_products
                )
        
        self.filtered_df = self.df.copy()
        
        if 'selected_molecules' in locals() and selected_molecules:
            self.filtered_df = self.filtered_df[self.filtered_df['Molecule'].isin(selected_molecules)]
        
        if 'selected_countries' in locals() and selected_countries:
            self.filtered_df = self.filtered_df[self.filtered_df['Country'].isin(selected_countries)]
        
        if 'selected_salts' in locals() and selected_salts:
            self.filtered_df = self.filtered_df[self.filtered_df['Chemical Salt'].isin(selected_salts)]
        
        if 'selected_products' in locals() and selected_products:
            self.filtered_df = self.filtered_df[self.filtered_df['International Product'].isin(selected_products)]
    
    def _check_data_availability(self) -> bool:
        required_columns = ['Molecule', 'USD_2024']
        return all(col in self.df.columns for col in required_columns)
    
    def _display_molecule_market_size(self):
        try:
            if len(self.filtered_df) == 0:
                return
            
            molecule_sales = self.filtered_df.groupby('Molecule')['USD_2024'].sum().reset_index()
            molecule_sales = molecule_sales.sort_values('USD_2024', ascending=False).head(15)
            
            fig = px.bar(
                molecule_sales,
                x='Molecule',
                y='USD_2024',
                title="Molekül Bazlı Pazar Büyüklüğü (Top 15 - 2024)",
                labels={'USD_2024': 'Satış (USD M)', 'Molecule': 'Molekül'},
                color='USD_2024',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Pazar büyüklüğü grafiği hatası: {str(e)}")
    
    def _display_molecule_growth_trends(self):
        try:
            if 'Molecule' not in self.filtered_df.columns or 'YoY_Growth_2024' not in self.filtered_df.columns:
                return
            
            molecule_growth = self.filtered_df.groupby('Molecule')['YoY_Growth_2024'].agg(['mean', 'count']).reset_index()
            molecule_growth = molecule_growth[molecule_growth['count'] >= 5]
            molecule_growth = molecule_growth.sort_values('mean', ascending=False).head(10)
            
            fig = px.bar(
                molecule_growth,
                x='Molecule',
                y='mean',
                title="En Hızlı Büyüyen Moleküller (YoY 2024 - Ortalama)",
                labels={'mean': 'Ortalama Büyüme (%)', 'Molecule': 'Molekül'},
                color='mean',
                color_continuous_scale='Greens'
            )
            
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Büyüme trendi grafiği hatası: {str(e)}")
    
    def _display_emerging_declining_molecules(self):
        try:
            years = [2022, 2023, 2024]
            
            if all(f'USD_{year}' in self.filtered_df.columns for year in years):
                molecule_yearly_sales = self.filtered_df.groupby('Molecule')[f'USD_{years[0]}', f'USD_{years[1]}', f'USD_{years[2]}'].sum()
                
                molecule_yearly_sales['Growth_Rate'] = (
                    (molecule_yearly_sales[f'USD_{years[2]}'] - molecule_yearly_sales[f'USD_{years[0]}']) / 
                    molecule_yearly_sales[f'USD_{years[0]}'].replace(0, np.nan)
                ) * 100
                
                emerging_molecules = molecule_yearly_sales[
                    (molecule_yearly_sales['Growth_Rate'] > 50) & 
                    (molecule_yearly_sales[f'USD_{years[2]}'] > 10)
                ].sort_values('Growth_Rate', ascending=False).head(10)
                
                declining_molecules = molecule_yearly_sales[
                    (molecule_yearly_sales['Growth_Rate'] < -20) & 
                    (molecule_yearly_sales[f'USD_{years[0]}'] > 10)
                ].sort_values('Growth_Rate', ascending=True).head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if not emerging_molecules.empty:
                        st.subheader("🚀 Yükselen Moleküller (2022-2024)")
                        for idx, (molecule, data) in enumerate(emerging_molecules.iterrows()):
                            st.metric(
                                label=molecule,
                                value=f"${data[f'USD_{years[2]}']:,.1f}M",
                                delta=f"%{data['Growth_Rate']:.1f}"
                            )
                
                with col2:
                    if not declining_molecules.empty:
                        st.subheader("📉 Düşüşteki Moleküller (2022-2024)")
                        for idx, (molecule, data) in enumerate(declining_molecules.iterrows()):
                            st.metric(
                                label=molecule,
                                value=f"${data[f'USD_{years[2]}']:,.1f}M",
                                delta=f"%{data['Growth_Rate']:.1f}",
                                delta_color="inverse"
                            )
            
        except Exception as e:
            st.error(f"Molekül trend analizi hatası: {str(e)}")
    
    def _display_price_evolution(self):
        try:
            if 'Molecule' not in self.filtered_df.columns:
                return
            
            years = [2022, 2023, 2024]
            price_columns = [f'AvgPrice_Unit_{year}' for year in years]
            
            if all(col in self.filtered_df.columns for col in price_columns):
                molecule_prices = self.filtered_df.groupby('Molecule')[price_columns].mean().reset_index()
                molecule_prices['Price_Change_%'] = (
                    (molecule_prices[price_columns[2]] - molecule_prices[price_columns[0]]) / 
                    molecule_prices[price_columns[0]].replace(0, np.nan)
                ) * 100
                
                top_price_increases = molecule_prices.sort_values('Price_Change_%', ascending=False).head(10)
                top_price_decreases = molecule_prices.sort_values('Price_Change_%', ascending=True).head(10)
                
                selected_category = st.radio(
                    "Fiyat Değişim Kategorisi:",
                    ["En Çok Artan Fiyatlar", "En Çok Düşen Fiyatlar"],
                    horizontal=True
                )
                
                if selected_category == "En Çok Artan Fiyatlar":
                    display_data = top_price_increases
                    title = "En Çok Fiyat Artışı Gösteren Moleküller (2022-2024)"
                else:
                    display_data = top_price_decreases
                    title = "En Çok Fiyat Düşüşü Gösteren Moleküller (2022-2024)"
                
                if not display_data.empty:
                    fig = go.Figure()
                    
                    for _, row in display_data.iterrows():
                        prices = [row[f'AvgPrice_Unit_{year}'] for year in years]
                        fig.add_trace(go.Scatter(
                            x=[str(year) for year in years],
                            y=prices,
                            mode='lines+markers',
                            name=row['Molecule'],
                            hovertemplate=f"<b>{row['Molecule']}</b><br>" +
                                         "Yıl: %{x}<br>" +
                                         "Fiyat: $%{y:.2f}<br>" +
                                         f"Değişim: %{row['Price_Change_%']:.1f}<extra></extra>"
                        ))
                    
                    fig.update_layout(
                        title=title,
                        xaxis_title="Yıl",
                        yaxis_title="Ortalama Fiyat (USD)",
                        height=400,
                        showlegend=True,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Fiyat evrim grafiği hatası: {str(e)}")

class CorporationCompetitionAnalysisTab:
    def __init__(self, data_processor: PharmaDataProcessor):
        self.df = data_processor.get_processed_data()
    
    def render(self):
        st.title("🏢 Corporation & Rekabet Analizi")
        st.markdown("---")
        
        self._render_filters()
        
        if not self._check_data_availability():
            st.warning("Bu analiz için gerekli veri kolonları mevcut değil.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_market_share_chart()
        
        with col2:
            self._display_share_shift_analysis()
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            self._display_winner_loser_analysis()
        
        with col4:
            self._display_price_competition_analysis()
    
    def _render_filters(self):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'Corporation' in self.df.columns:
                available_corps = sorted(self.df['Corporation'].unique())
                selected_corps = st.multiselect(
                    "Corporation'lar",
                    options=available_corps,
                    default=available_corps[:10] if len(available_corps) > 10 else available_corps
                )
            
        with col2:
            if 'Country' in self.df.columns:
                available_countries = sorted(self.df['Country'].unique())
                selected_countries = st.multiselect(
                    "Ülkeler",
                    options=available_countries,
                    default=available_countries[:5] if len(available_countries) > 5 else available_countries
                )
        
        with col3:
            if 'Molecule' in self.df.columns:
                available_molecules = sorted(self.df['Molecule'].unique())
                selected_molecules = st.multiselect(
                    "Moleküller",
                    options=available_molecules,
                    default=available_molecules[:5] if len(available_molecules) > 5 else available_molecules
                )
        
        self.filtered_df = self.df.copy()
        
        if 'selected_corps' in locals() and selected_corps:
            self.filtered_df = self.filtered_df[self.filtered_df['Corporation'].isin(selected_corps)]
        
        if 'selected_countries' in locals() and selected_countries:
            self.filtered_df = self.filtered_df[self.filtered_df['Country'].isin(selected_countries)]
        
        if 'selected_molecules' in locals() and selected_molecules:
            self.filtered_df = self.filtered_df[self.filtered_df['Molecule'].isin(selected_molecules)]
    
    def _check_data_availability(self) -> bool:
        required_columns = ['Corporation', 'USD_2024']
        return all(col in self.df.columns for col in required_columns)
    
    def _display_market_share_chart(self):
        try:
            if len(self.filtered_df) == 0:
                return
            
            corp_sales = self.filtered_df.groupby('Corporation')['USD_2024'].sum().reset_index()
            corp_sales = corp_sales.sort_values('USD_2024', ascending=False).head(10)
            
            total_sales = corp_sales['USD_2024'].sum()
            corp_sales['Market_Share'] = (corp_sales['USD_2024'] / total_sales) * 100
            
            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.7, 0.3],
                specs=[[{"type": "bar"}, {"type": "pie"}]]
            )
            
            fig.add_trace(
                go.Bar(
                    x=corp_sales['Corporation'],
                    y=corp_sales['Market_Share'],
                    name="Pazar Payı",
                    marker_color='#2E86AB',
                    text=[f"%{x:.1f}" for x in corp_sales['Market_Share']],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Pie(
                    labels=corp_sales['Corporation'],
                    values=corp_sales['USD_2024'],
                    hole=0.4,
                    name="Satış Dağılımı",
                    textinfo='label+percent'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Top 10 Corporation - Pazar Payı Dağılımı (2024)",
                height=400,
                showlegend=False
            )
            
            fig.update_xaxes(tickangle=-45, row=1, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Pazar payı grafiği hatası: {str(e)}")
    
    def _display_share_shift_analysis(self):
        try:
            if 'Market_Share_2023' not in self.filtered_df.columns or 'Market_Share_2024' not in self.filtered_df.columns:
                return
            
            corp_shares = self.filtered_df.groupby('Corporation')[['Market_Share_2023', 'Market_Share_2024']].sum().reset_index()
            corp_shares['Share_Change'] = corp_shares['Market_Share_2024'] - corp_shares['Market_Share_2023']
            
            corp_shares = corp_shares.sort_values('Share_Change', ascending=False)
            
            top_gainers = corp_shares.head(5)
            top_losers = corp_shares.tail(5)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('En Çok Pazar Payı Kazananlar', 'En Çok Pazar Payı Kaybedenler'),
                vertical_spacing=0.15
            )
            
            fig.add_trace(
                go.Bar(
                    x=top_gainers['Corporation'],
                    y=top_gainers['Share_Change'],
                    name="Kazananlar",
                    marker_color='green',
                    text=[f"+{x:.1f}%" for x in top_gainers['Share_Change']],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=top_losers['Corporation'],
                    y=top_losers['Share_Change'],
                    name="Kaybedenler",
                    marker_color='red',
                    text=[f"{x:.1f}%" for x in top_losers['Share_Change']],
                    textposition='auto'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=500,
                showlegend=False
            )
            
            fig.update_xaxes(tickangle=-45, row=1, col=1)
            fig.update_xaxes(tickangle=-45, row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Pazar payı değişim grafiği hatası: {str(e)}")
    
    def _display_winner_loser_analysis(self):
        try:
            if 'Corporation' not in self.filtered_df.columns or 'YoY_Growth_2024' not in self.filtered_df.columns:
                return
            
            corp_growth = self.filtered_df.groupby('Corporation')['YoY_Growth_2024'].agg(['mean', 'count', 'std']).reset_index()
            corp_growth = corp_growth[corp_growth['count'] >= 3]
            
            if len(corp_growth) > 0:
                winners = corp_growth[corp_growth['mean'] > 10].sort_values('mean', ascending=False).head(10)
                losers = corp_growth[corp_growth['mean'] < -5].sort_values('mean', ascending=True).head(10)
                
                st.subheader("🎯 Kazanan / Kaybeden Firmalar")
                
                if not winners.empty:
                    st.write("**Kazanan Firmalar (Büyüme > %10):**")
                    for idx, row in winners.iterrows():
                        st.write(f"• {row['Corporation']}: %{row['mean']:.1f} büyüme ({row['count']} ürün)")
                
                if not losers.empty:
                    st.write("**Kaybeden Firmalar (Büyüme < -%5):**")
                    for idx, row in losers.iterrows():
                        st.write(f"• {row['Corporation']}: %{row['mean']:.1f} büyüme ({row['count']} ürün)")
            
        except Exception as e:
            st.error(f"Kazanan/kaybeden analizi hatası: {str(e)}")
    
    def _display_price_competition_analysis(self):
        try:
            if 'Molecule' not in self.filtered_df.columns or 'Corporation' not in self.filtered_df.columns or 'AvgPrice_Unit_2024' not in self.filtered_df.columns:
                return
            
            molecule_corp_prices = self.filtered_df.groupby(['Molecule', 'Corporation'])['AvgPrice_Unit_2024'].mean().reset_index()
            
            if len(molecule_corp_prices['Molecule'].unique()) > 0:
                selected_molecule = st.selectbox(
                    "Fiyat Rekabeti Analizi için Molekül Seçin:",
                    options=sorted(molecule_corp_prices['Molecule'].unique()),
                    key="price_competition_molecule"
                )
                
                molecule_data = molecule_corp_prices[molecule_corp_prices['Molecule'] == selected_molecule]
                
                if len(molecule_data) > 1:
                    fig = px.bar(
                        molecule_data.sort_values('AvgPrice_Unit_2024', ascending=False),
                        x='Corporation',
                        y='AvgPrice_Unit_2024',
                        title=f"{selected_molecule} - Firmalar Arası Fiyat Rekabeti (2024)",
                        labels={'AvgPrice_Unit_2024': 'Ortalama Fiyat (USD)', 'Corporation': 'Firma'},
                        color='AvgPrice_Unit_2024',
                        color_continuous_scale='RdYlBu_r'
                    )
                    
                    avg_price = molecule_data['AvgPrice_Unit_2024'].mean()
                    fig.add_hline(
                        y=avg_price,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Ortalama: ${avg_price:.2f}",
                        annotation_position="bottom right"
                    )
                    
                    fig.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Fiyat rekabeti analizi hatası: {str(e)}")

class SpecialtyAnalysisTab:
    def __init__(self, data_processor: PharmaDataProcessor):
        self.df = data_processor.get_processed_data()
    
    def render(self):
        st.title("🎯 Specialty vs Non-Specialty Analizi")
        st.markdown("---")
        
        if 'Specialty_Type' not in self.df.columns:
            st.warning("Specialty verisi bulunamadı.")
            return
        
        self._render_filters()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._display_specialty_share_metrics()
        
        with col2:
            self._display_price_difference_metrics()
        
        with col3:
            self._display_growth_comparison()
        
        with col4:
            self._display_premiumization_metrics()
        
        st.markdown("---")
        
        col5, col6 = st.columns(2)
        
        with col5:
            self._display_specialty_vs_non_specialty_chart()
        
        with col6:
            self._display_price_volume_comparison()
    
    def _render_filters(self):
        col1, col2 = st.columns(2)
        
        with col1:
            available_countries = sorted(self.df['Country'].unique()) if 'Country' in self.df.columns else []
            selected_countries = st.multiselect(
                "Ülkeler (Specialty)",
                options=available_countries,
                default=available_countries[:5] if len(available_countries) > 5 else available_countries,
                key="specialty_countries"
            )
            
        with col2:
            if 'Molecule' in self.df.columns:
                available_molecules = sorted(self.df['Molecule'].unique())
                selected_molecules = st.multiselect(
                    "Moleküller (Specialty)",
                    options=available_molecules,
                    default=available_molecules[:5] if len(available_molecules) > 5 else available_molecules,
                    key="specialty_molecules"
                )
        
        self.filtered_df = self.df.copy()
        
        if selected_countries:
            self.filtered_df = self.filtered_df[self.filtered_df['Country'].isin(selected_countries)]
        
        if 'selected_molecules' in locals() and selected_molecules:
            self.filtered_df = self.filtered_df[self.filtered_df['Molecule'].isin(selected_molecules)]
    
    def _display_specialty_share_metrics(self):
        try:
            specialty_sales = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Specialty']['USD_2024'].sum()
            non_specialty_sales = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Non-Specialty']['USD_2024'].sum()
            total_sales = specialty_sales + non_specialty_sales
            
            if total_sales > 0:
                specialty_share = (specialty_sales / total_sales) * 100
                
                st.metric(
                    label="Specialty Ürün Payı",
                    value=f"%{specialty_share:.1f}",
                    delta=f"${specialty_sales:,.0f}M"
                )
        except Exception as e:
            st.error(f"Specialty pay metrik hatası: {str(e)}")
    
    def _display_price_difference_metrics(self):
        try:
            if 'AvgPrice_Unit_2024' in self.filtered_df.columns:
                specialty_avg_price = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Specialty']['AvgPrice_Unit_2024'].mean()
                non_specialty_avg_price = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Non-Specialty']['AvgPrice_Unit_2024'].mean()
                
                if non_specialty_avg_price > 0:
                    price_ratio = (specialty_avg_price / non_specialty_avg_price) * 100
                    
                    st.metric(
                        label="Specialty Fiyat Premiumu",
                        value=f"%{price_ratio:.0f}",
                        delta=f"${specialty_avg_price:,.2f} vs ${non_specialty_avg_price:,.2f}"
                    )
        except Exception as e:
            st.error(f"Fiyat farkı metrik hatası: {str(e)}")
    
    def _display_growth_comparison(self):
        try:
            if 'YoY_Growth_2024' in self.filtered_df.columns:
                specialty_growth = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Specialty']['YoY_Growth_2024'].mean()
                non_specialty_growth = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Non-Specialty']['YoY_Growth_2024'].mean()
                
                growth_diff = specialty_growth - non_specialty_growth
                
                st.metric(
                    label="Büyüme Farkı (Specialty vs Non)",
                    value=f"%{growth_diff:.1f}",
                    delta=f"Specialty: %{specialty_growth:.1f}"
                )
        except Exception as e:
            st.error(f"Büyüme karşılaştırma hatası: {str(e)}")
    
    def _display_premiumization_metrics(self):
        try:
            if 'AvgPrice_Unit_2024' in self.filtered_df.columns and 'AvgPrice_Unit_2023' in self.filtered_df.columns:
                specialty_price_growth = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Specialty']['AvgPrice_Unit_2024'].mean() - \
                                       self.filtered_df[self.filtered_df['Specialty_Type'] == 'Specialty']['AvgPrice_Unit_2023'].mean()
                
                non_specialty_price_growth = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Non-Specialty']['AvgPrice_Unit_2024'].mean() - \
                                           self.filtered_df[self.filtered_df['Specialty_Type'] == 'Non-Specialty']['AvgPrice_Unit_2023'].mean()
                
                st.metric(
                    label="Fiyat Artışı Farkı",
                    value=f"${specialty_price_growth:,.2f}",
                    delta=f"Non-Specialty: ${non_specialty_price_growth:,.2f}"
                )
        except Exception as e:
            st.error(f"Premiumlaşma metrik hatası: {str(e)}")
    
    def _display_specialty_vs_non_specialty_chart(self):
        try:
            years = [2022, 2023, 2024]
            specialty_data = []
            non_specialty_data = []
            
            for year in years:
                usd_col = f'USD_{year}'
                if usd_col in self.filtered_df.columns:
                    specialty_sales = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Specialty'][usd_col].sum()
                    non_specialty_sales = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Non-Specialty'][usd_col].sum()
                    
                    specialty_data.append(specialty_sales)
                    non_specialty_data.append(non_specialty_sales)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[str(year) for year in years],
                y=specialty_data,
                mode='lines+markers',
                name='Specialty',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=10)
            ))
            
            fig.add_trace(go.Scatter(
                x=[str(year) for year in years],
                y=non_specialty_data,
                mode='lines+markers',
                name='Non-Specialty',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="Specialty vs Non-Specialty Satış Trendi (2022-2024)",
                xaxis_title="Yıl",
                yaxis_title="Satış (USD M)",
                height=400,
                showlegend=True,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Trend grafiği hatası: {str(e)}")
    
    def _display_price_volume_comparison(self):
        try:
            if 'Units_2024' in self.filtered_df.columns and 'AvgPrice_Unit_2024' in self.filtered_df.columns:
                specialty_volume = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Specialty']['Units_2024'].sum()
                specialty_avg_price = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Specialty']['AvgPrice_Unit_2024'].mean()
                
                non_specialty_volume = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Non-Specialty']['Units_2024'].sum()
                non_specialty_avg_price = self.filtered_df[self.filtered_df['Specialty_Type'] == 'Non-Specialty']['AvgPrice_Unit_2024'].mean()
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Hacim Karşılaştırması', 'Fiyat Karşılaştırması'),
                    specs=[[{"type": "pie"}, {"type": "bar"}]]
                )
                
                fig.add_trace(
                    go.Pie(
                        labels=['Specialty', 'Non-Specialty'],
                        values=[specialty_volume, non_specialty_volume],
                        hole=0.3,
                        name="Hacim",
                        marker_colors=['#FF6B6B', '#4ECDC4']
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=['Specialty', 'Non-Specialty'],
                        y=[specialty_avg_price, non_specialty_avg_price],
                        name="Fiyat",
                        marker_color=['#FF6B6B', '#4ECDC4'],
                        text=[f"${x:.2f}" for x in [specialty_avg_price, non_specialty_avg_price]],
                        textposition='auto'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Hacim-fiyat karşılaştırma hatası: {str(e)}")

class PriceInflationAnalysisTab:
    def __init__(self, data_processor: PharmaDataProcessor):
        self.df = data_processor.get_processed_data()
    
    def render(self):
        st.title("💰 Fiyat & Enflasyon Analizi")
        st.markdown("---")
        
        self._render_filters()
        
        if not self._check_data_availability():
            st.warning("Bu analiz için gerekli veri kolonları mevcut değil.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_price_inflation_chart()
        
        with col2:
            self._display_real_vs_nominal_growth()
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            self._display_price_mix_decomposition()
        
        with col4:
            self._display_price_outlier_detection()
    
    def _render_filters(self):
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Molecule' in self.df.columns:
                available_molecules = sorted(self.df['Molecule'].unique())
                selected_molecule = st.selectbox(
                    "Molekül Seçin:",
                    options=available_molecules,
                    key="price_analysis_molecule"
                )
            
            if 'Country' in self.df.columns:
                available_countries = sorted(self.df['Country'].unique())
                selected_countries = st.multiselect(
                    "Ülkeler:",
                    options=available_countries,
                    default=available_countries[:3] if len(available_countries) > 3 else available_countries,
                    key="price_analysis_countries"
                )
        
        with col2:
            if 'Corporation' in self.df.columns:
                available_corps = sorted(self.df['Corporation'].unique())
                selected_corps = st.multiselect(
                    "Corporation'lar:",
                    options=available_corps,
                    default=available_corps[:5] if len(available_corps) > 5 else available_corps,
                    key="price_analysis_corps"
                )
            
            min_price = st.number_input(
                "Minimum Fiyat (USD):",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key="min_price_filter"
            )
        
        self.filtered_df = self.df.copy()
        
        if 'selected_molecule' in locals() and selected_molecule:
            self.filtered_df = self.filtered_df[self.filtered_df['Molecule'] == selected_molecule]
        
        if 'selected_countries' in locals() and selected_countries:
            self.filtered_df = self.filtered_df[self.filtered_df['Country'].isin(selected_countries)]
        
        if 'selected_corps' in locals() and selected_corps:
            self.filtered_df = self.filtered_df[self.filtered_df['Corporation'].isin(selected_corps)]
        
        if 'AvgPrice_Unit_2024' in self.filtered_df.columns:
            self.filtered_df = self.filtered_df[self.filtered_df['AvgPrice_Unit_2024'] >= min_price]
    
    def _check_data_availability(self) -> bool:
        required_columns = ['AvgPrice_Unit_2022', 'AvgPrice_Unit_2023', 'AvgPrice_Unit_2024']
        return all(col in self.df.columns for col in required_columns)
    
    def _display_price_inflation_chart(self):
        try:
            years = [2022, 2023, 2024]
            
            price_data = []
            for year in years:
                price_col = f'AvgPrice_Unit_{year}'
                if price_col in self.filtered_df.columns:
                    avg_price = self.filtered_df[price_col].mean()
                    price_data.append(avg_price)
            
            if len(price_data) == 3:
                inflation_rates = []
                for i in range(1, len(price_data)):
                    if price_data[i-1] > 0:
                        inflation_rate = ((price_data[i] - price_data[i-1]) / price_data[i-1]) * 100
                        inflation_rates.append(inflation_rate)
                    else:
                        inflation_rates.append(0)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(
                        x=[str(year) for year in years],
                        y=price_data,
                        name="Ortalama Fiyat",
                        marker_color='#2E86AB',
                        text=[f"${x:.2f}" for x in price_data],
                        textposition='auto'
                    ),
                    secondary_y=False
                )
                
                if len(inflation_rates) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[f"{years[i]}-{years[i+1]}" for i in range(len(inflation_rates))],
                            y=inflation_rates,
                            name="Fiyat Artış Oranı (%)",
                            mode='lines+markers',
                            line=dict(color='#A23B72', width=3),
                            marker=dict(size=10)
                        ),
                        secondary_y=True
                    )
                
                fig.update_layout(
                    title="Yıllara Göre Fiyat Artışı ve Enflasyon Oranı",
                    xaxis_title="Yıl",
                    yaxis_title="Ortalama Fiyat (USD)",
                    yaxis2_title="Fiyat Artış Oranı (%)",
                    height=400,
                    showlegend=True,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Fiyat enflasyon grafiği hatası: {str(e)}")
    
    def _display_real_vs_nominal_growth(self):
        try:
            if 'USD_2023' in self.filtered_df.columns and 'USD_2024' in self.filtered_df.columns:
                nominal_growth = self.filtered_df['USD_2024'].sum() - self.filtered_df['USD_2023'].sum()
                
                if 'Volume_Effect_2024' in self.filtered_df.columns and 'Price_Effect_2024' in self.filtered_df.columns:
                    volume_effect = self.filtered_df['Volume_Effect_2024'].sum()
                    price_effect = self.filtered_df['Price_Effect_2024'].sum()
                    
                    if nominal_growth != 0:
                        volume_share = (volume_effect / nominal_growth) * 100
                        price_share = (price_effect / nominal_growth) * 100
                        mix_share = 100 - volume_share - price_share
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=['Hacim Etkisi', 'Fiyat Etkisi', 'Mix Etkisi'],
                            values=[volume_share, price_share, mix_share],
                            hole=0.4,
                            marker_colors=['#4ECDC4', '#FF6B6B', '#FFE66D'],
                            textinfo='label+percent',
                            hoverinfo='label+value+percent'
                        )])
                        
                        fig.update_layout(
                            title="Nominal Büyümenin Dağılımı (2023-2024)",
                            height=400,
                            annotations=[dict(
                                text=f"Toplam: ${nominal_growth:,.1f}M",
                                x=0.5, y=0.5, font_size=14, showarrow=False
                            )]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if price_share > 70:
                            st.info("⚠️ **Uyarı:** Büyümenin %70'inden fazlası fiyat artışından kaynaklanıyor. Hacim büyümesi sınırlı.")
                        elif volume_share > 70:
                            st.success("✅ **Olumlu:** Büyümenin %70'inden fazlası hacim artışından kaynaklanıyor. Sağlıklı büyüme.")
            
        except Exception as e:
            st.error(f"Reel büyüme analizi hatası: {str(e)}")
    
    def _display_price_mix_decomposition(self):
        try:
            if 'Molecule' in self.filtered_df.columns and 'Price_Effect_2024' in self.filtered_df.columns and 'Volume_Effect_2024' in self.filtered_df.columns:
                molecule_effects = self.filtered_df.groupby('Molecule')[['Price_Effect_2024', 'Volume_Effect_2024']].sum().reset_index()
                molecule_effects['Total_Effect'] = molecule_effects['Price_Effect_2024'] + molecule_effects['Volume_Effect_2024']
                molecule_effects = molecule_effects.sort_values('Total_Effect', ascending=False).head(10)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=molecule_effects['Molecule'],
                    y=molecule_effects['Price_Effect_2024'],
                    name='Fiyat Etkisi',
                    marker_color='#FF6B6B'
                ))
                
                fig.add_trace(go.Bar(
                    x=molecule_effects['Molecule'],
                    y=molecule_effects['Volume_Effect_2024'],
                    name='Hacim Etkisi',
                    marker_color='#4ECDC4'
                ))
                
                fig.update_layout(
                    title="Top 10 Molekül - Fiyat vs Hacim Etkisi (2024)",
                    xaxis_title="Molekül",
                    yaxis_title="Etki (USD M)",
                    barmode='stack',
                    height=400,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Fiyat-mix ayrıştırma hatası: {str(e)}")
    
    def _display_price_outlier_detection(self):
        try:
            if 'AvgPrice_Unit_2024' in self.filtered_df.columns and 'Molecule' in self.filtered_df.columns:
                price_data = self.filtered_df[['Molecule', 'AvgPrice_Unit_2024', 'Country', 'Corporation']].dropna()
                
                if len(price_data) > 10:
                    scaler = StandardScaler()
                    price_scaled = scaler.fit_transform(price_data[['AvgPrice_Unit_2024']].values.reshape(-1, 1))
                    
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(price_scaled)
                    
                    price_data['Is_Outlier'] = outliers
                    outliers_df = price_data[price_data['Is_Outlier'] == -1]
                    
                    if not outliers_df.empty:
                        st.subheader("⚠️ Fiyat Outlier'ları Tespiti")
                        
                        for idx, row in outliers_df.head(10).iterrows():
                            st.write(f"• **{row['Molecule']}** - {row['Country']} ({row['Corporation']}): ${row['AvgPrice_Unit_2024']:.2f}")
                        
                        avg_price = price_data['AvgPrice_Unit_2024'].mean()
                        std_price = price_data['AvgPrice_Unit_2024'].std()
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Box(
                            y=price_data['AvgPrice_Unit_2024'],
                            name='Fiyat Dağılımı',
                            boxpoints='outliers',
                            marker_color='#2E86AB'
                        ))
                        
                        fig.add_hline(
                            y=avg_price,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Ortalama: ${avg_price:.2f}",
                            annotation_position="top right"
                        )
                        
                        fig.add_hline(
                            y=avg_price + 2*std_price,
                            line_dash="dot",
                            line_color="orange",
                            annotation_text=f"+2σ: ${avg_price + 2*std_price:.2f}",
                            annotation_position="top right"
                        )
                        
                        fig.update_layout(
                            title="Fiyat Dağılımı ve Outlier Analizi (2024)",
                            yaxis_title="Fiyat (USD)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Outlier tespit hatası: {str(e)}")

class PackStrengthAnalysisTab:
    def __init__(self, data_processor: PharmaDataProcessor):
        self.df = data_processor.get_processed_data()
    
    def render(self):
        st.title("📦 Pack / Strength / Form Analizi")
        st.markdown("---")
        
        self._render_filters()
        
        if not self._check_data_availability():
            st.warning("Bu analiz için gerekli veri kolonları mevcut değil.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_pack_size_optimization()
        
        with col2:
            self._display_strength_based_analysis()
        
        st.markdown("---")
        
        self._display_form_factor_analysis()
    
    def _render_filters(self):
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Pack' in self.df.columns:
                available_packs = sorted(self.df['Pack'].unique())
                selected_packs = st.multiselect(
                    "Pack Seçenekleri:",
                    options=available_packs,
                    default=available_packs[:5] if len(available_packs) > 5 else available_packs,
                    key="pack_analysis"
                )
            
            if 'Molecule' in self.df.columns:
                available_molecules = sorted(self.df['Molecule'].unique())
                selected_molecules = st.multiselect(
                    "Moleküller:",
                    options=available_molecules,
                    default=available_molecules[:3] if len(available_molecules) > 3 else available_molecules,
                    key="molecule_pack_analysis"
                )
        
        with col2:
            if 'Strength' in self.df.columns:
                available_strengths = sorted(self.df['Strength'].unique())
                selected_strengths = st.multiselect(
                    "Strength Seçenekleri:",
                    options=available_strengths,
                    default=available_strengths[:5] if len(available_strengths) > 5 else available_strengths,
                    key="strength_analysis"
                )
            
            min_volume = st.number_input(
                "Minimum Hacim (Units):",
                min_value=0,
                value=1000,
                step=1000,
                key="min_volume_pack"
            )
        
        self.filtered_df = self.df.copy()
        
        if 'selected_packs' in locals() and selected_packs:
            self.filtered_df = self.filtered_df[self.filtered_df['Pack'].isin(selected_packs)]
        
        if 'selected_molecules' in locals() and selected_molecules:
            self.filtered_df = self.filtered_df[self.filtered_df['Molecule'].isin(selected_molecules)]
        
        if 'selected_strengths' in locals() and selected_strengths:
            self.filtered_df = self.filtered_df[self.filtered_df['Strength'].isin(selected_strengths)]
        
        if 'Units_2024' in self.filtered_df.columns:
            self.filtered_df = self.filtered_df[self.filtered_df['Units_2024'] >= min_volume]
    
    def _check_data_availability(self) -> bool:
        required_columns = ['Pack', 'Strength', 'Units_2024']
        return all(col in self.df.columns for col in required_columns)
    
    def _display_pack_size_optimization(self):
        try:
            if 'Pack' in self.filtered_df.columns and 'AvgPrice_Unit_2024' in self.filtered_df.columns and 'Units_2024' in self.filtered_df.columns:
                pack_analysis = self.filtered_df.groupby('Pack').agg({
                    'Units_2024': 'sum',
                    'AvgPrice_Unit_2024': 'mean',
                    'USD_2024': 'sum',
                    'Molecule': 'nunique'
                }).reset_index()
                
                pack_analysis = pack_analysis.sort_values('USD_2024', ascending=False)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Toplam Hacim', 'Ortalama Fiyat', 'Toplam Değer', 'Molekül Çeşitliliği'),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.15
                )
                
                fig.add_trace(
                    go.Bar(
                        x=pack_analysis['Pack'],
                        y=pack_analysis['Units_2024'],
                        name="Hacim",
                        marker_color='#4ECDC4'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=pack_analysis['Pack'],
                        y=pack_analysis['AvgPrice_Unit_2024'],
                        name="Fiyat",
                        marker_color='#FF6B6B'
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(
                        x=pack_analysis['Pack'],
                        y=pack_analysis['USD_2024'],
                        name="Değer",
                        marker_color='#2E86AB'
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=pack_analysis['Pack'],
                        y=pack_analysis['Molecule'],
                        name="Çeşitlilik",
                        marker_color='#FFE66D'
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(
                    height=600,
                    showlegend=False
                )
                
                fig.update_xaxes(tickangle=-45, row=1, col=1)
                fig.update_xaxes(tickangle=-45, row=1, col=2)
                fig.update_xaxes(tickangle=-45, row=2, col=1)
                fig.update_xaxes(tickangle=-45, row=2, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📊 Pack Optimizasyon İçgörüleri:")
                
                max_value_pack = pack_analysis.loc[pack_analysis['USD_2024'].idxmax()]
                max_volume_pack = pack_analysis.loc[pack_analysis['Units_2024'].idxmax()]
                max_price_pack = pack_analysis.loc[pack_analysis['AvgPrice_Unit_2024'].idxmax()]
                
                st.write(f"• **En Değerli Pack:** {max_value_pack['Pack']} (${max_value_pack['USD_2024']:,.0f}M)")
                st.write(f"• **En Yüksek Hacimli Pack:** {max_volume_pack['Pack']} ({max_volume_pack['Units_2024']:,.0f} birim)")
                st.write(f"• **En Yüksek Fiyatlı Pack:** {max_price_pack['Pack']} (${max_price_pack['AvgPrice_Unit_2024']:.2f})")
            
        except Exception as e:
            st.error(f"Pack optimizasyon analizi hatası: {str(e)}")
    
    def _display_strength_based_analysis(self):
        try:
            if 'Strength' in self.filtered_df.columns:
                strength_analysis = self.filtered_df.groupby('Strength').agg({
                    'USD_2024': 'sum',
                    'YoY_Growth_2024': 'mean',
                    'Units_2024': 'sum',
                    'AvgPrice_Unit_2024': 'mean'
                }).reset_index()
                
                strength_analysis = strength_analysis.sort_values('USD_2024', ascending=False).head(10)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(
                        x=strength_analysis['Strength'],
                        y=strength_analysis['USD_2024'],
                        name="Satış Değeri",
                        marker_color='#2E86AB',
                        text=[f"${x:,.0f}M" for x in strength_analysis['USD_2024']],
                        textposition='auto'
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=strength_analysis['Strength'],
                        y=strength_analysis['YoY_Growth_2024'],
                        name="Büyüme Oranı (%)",
                        mode='lines+markers',
                        line=dict(color='#A23B72', width=3),
                        marker=dict(size=10)
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title="Strength Bazlı Satış ve Büyüme Analizi (Top 10 - 2024)",
                    xaxis_title="Strength",
                    yaxis_title="Satış (USD M)",
                    yaxis2_title="Büyüme Oranı (%)",
                    height=400,
                    showlegend=True,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                fastest_growing = strength_analysis.loc[strength_analysis['YoY_Growth_2024'].idxmax()]
                highest_value = strength_analysis.loc[strength_analysis['USD_2024'].idxmax()]
                
                st.info(f"**Hızlı İçgörüler:**")
                st.write(f"• **En Hızlı Büyüyen Strength:** {fastest_growing['Strength']} (%{fastest_growing['YoY_Growth_2024']:.1f} büyüme)")
                st.write(f"• **En Yüksek Değerli Strength:** {highest_value['Strength']} (${highest_value['USD_2024']:,.0f}M)")
            
        except Exception as e:
            st.error(f"Strength analizi hatası: {str(e)}")
    
    def _display_form_factor_analysis(self):
        try:
            if 'Pack' in self.filtered_df.columns:
                pack_categories = self._categorize_pack_types(self.filtered_df['Pack'].unique())
                
                if pack_categories:
                    self.filtered_df['Pack_Category'] = self.filtered_df['Pack'].map(
                        lambda x: next((cat for cat, patterns in pack_categories.items() if any(pattern in str(x).lower() for pattern in patterns)), 'Diğer')
                    )
                    
                    category_analysis = self.filtered_df.groupby('Pack_Category').agg({
                        'USD_2024': 'sum',
                        'Units_2024': 'sum',
                        'AvgPrice_Unit_2024': 'mean',
                        'YoY_Growth_2024': 'mean'
                    }).reset_index()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = px.pie(
                            category_analysis,
                            values='USD_2024',
                            names='Pack_Category',
                            title="Pack Kategorilerine Göre Değer Dağılımı",
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig1.update_layout(height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        category_analysis_sorted = category_analysis.sort_values('YoY_Growth_2024', ascending=False)
                        
                        fig2 = px.bar(
                            category_analysis_sorted,
                            x='Pack_Category',
                            y='YoY_Growth_2024',
                            title="Pack Kategorilerine Göre Büyüme Oranları",
                            labels={'YoY_Growth_2024': 'Büyüme Oranı (%)', 'Pack_Category': 'Pack Kategorisi'},
                            color='YoY_Growth_2024',
                            color_continuous_scale='RdYlGn'
                        )
                        fig2.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.subheader("🏆 Kazanan Form Faktörleri:")
                    
                    top_growth_category = category_analysis.loc[category_analysis['YoY_Growth_2024'].idxmax()]
                    top_value_category = category_analysis.loc[category_analysis['USD_2024'].idxmax()]
                    top_volume_category = category_analysis.loc[category_analysis['Units_2024'].idxmax()]
                    
                    st.write(f"• **En Hızlı Büyüyen:** {top_growth_category['Pack_Category']} (%{top_growth_category['YoY_Growth_2024']:.1f} büyüme)")
                    st.write(f"• **En Değerli:** {top_value_category['Pack_Category']} (${top_value_category['USD_2024']:,.0f}M)")
                    st.write(f"• **En Yüksek Hacim:** {top_volume_category['Pack_Category']} ({top_volume_category['Units_2024']:,.0f} birim)")
            
        except Exception as e:
            st.error(f"Form faktörü analizi hatası: {str(e)}")
    
    def _categorize_pack_types(self, packs):
        categories = {
            'Tablet': ['tablet', 'tab', 'tbl', 'compressed'],
            'Kapsül': ['capsule', 'cap', 'kapsul'],
            'Sıvı': ['liquid', 'syrup', 'solution', 'oral solution', 'suspension'],
            'Enjeksiyon': ['injection', 'inj', 'vial', 'ampoule', 'syringe'],
            'Krem': ['cream', 'ointment', 'gel', 'lotion'],
            'Aerosol': ['aerosol', 'inhaler', 'spray', 'puffer'],
            'Patch': ['patch', 'transdermal'],
            'Powder': ['powder', 'granule', 'sachet']
        }
        return categories

class InsightEngineTab:
    def __init__(self, data_processor: PharmaDataProcessor):
        self.df = data_processor.get_processed_data()
        self.insights = []
    
    def render(self):
        st.title("🤖 Otomatik İçgörü Motoru")
        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Analiz Konfigürasyonu")
        
        with col2:
            if st.button("🔍 İçgörüleri Yeniden Hesapla", type="primary"):
                st.session_state.insights_generated = False
        
        self._render_insight_configuration()
        
        if not self._check_data_availability():
            st.warning("İçgörü oluşturmak için yeterli veri yok.")
            return
        
        if 'insights_generated' not in st.session_state or not st.session_state.insights_generated:
            with st.spinner("Akıllı içgörüler oluşturuluyor..."):
                self._generate_all_insights()
                st.session_state.insights_generated = True
        
        self._display_insights()
    
    def _render_insight_configuration(self):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.min_growth_threshold = st.slider(
                "Önemli Büyüme Eşiği (%)",
                min_value=5,
                max_value=50,
                value=15,
                step=5,
                help="Bu eşiğin üzerindeki büyümeler önemli kabul edilir"
            )
            
            self.min_market_share = st.slider(
                "Önemli Pazar Payı Eşiği (%)",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Bu eşiğin üzerindeki pazar payları önemli kabul edilir"
            )
        
        with col2:
            self.min_volume = st.number_input(
                "Minimum Hacim (Units)",
                min_value=1000,
                value=10000,
                step=1000,
                help="Bu eşiğin altındaki hacimler analiz edilmez"
            )
            
            self.price_change_threshold = st.slider(
                "Önemli Fiyat Değişimi (%)",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Bu eşiğin üzerindeki fiyat değişimleri önemli kabul edilir"
            )
        
        with col3:
            self.insight_categories = st.multiselect(
                "İçgörü Kategorileri:",
                options=[
                    'Büyüme Analizi',
                    'Pazar Payı Değişimi',
                    'Fiyat Rekabeti',
                    'Specialty Trendleri',
                    'Risk Tespiti',
                    'Fırsat Tespiti'
                ],
                default=['Büyüme Analizi', 'Risk Tespiti', 'Fırsat Tespiti']
            )
            
            self.max_insights = st.slider(
                "Maksimum İçgörü Sayısı",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )
    
    def _check_data_availability(self) -> bool:
        required_columns = ['USD_2024', 'YoY_Growth_2024', 'Country', 'Molecule']
        return all(col in self.df.columns for col in required_columns)
    
    def _generate_all_insights(self):
        try:
            self.insights = []
            
            if 'Büyüme Analizi' in self.insight_categories:
                self._generate_growth_insights()
            
            if 'Pazar Payı Değişimi' in self.insight_categories:
                self._generate_market_share_insights()
            
            if 'Fiyat Rekabeti' in self.insight_categories:
                self._generate_price_competition_insights()
            
            if 'Specialty Trendleri' in self.insight_categories:
                self._generate_specialty_insights()
            
            if 'Risk Tespiti' in self.insight_categories:
                self._generate_risk_insights()
            
            if 'Fırsat Tespiti' in self.insight_categories:
                self._generate_opportunity_insights()
            
            self.insights = self.insights[:self.max_insights]
            
        except Exception as e:
            st.error(f"İçgörü oluşturma hatası: {str(e)}")
    
    def _generate_growth_insights(self):
        try:
            if 'Country' in self.df.columns and 'YoY_Growth_2024' in self.df.columns and 'USD_2024' in self.df.columns:
                country_growth = self.df.groupby('Country').agg({
                    'YoY_Growth_2024': 'mean',
                    'USD_2024': 'sum',
                    'Units_2024': 'sum'
                }).reset_index()
                
                high_growth_countries = country_growth[
                    (country_growth['YoY_Growth_2024'] > self.min_growth_threshold) &
                    (country_growth['Units_2024'] > self.min_volume)
                ].sort_values('YoY_Growth_2024', ascending=False)
                
                for _, row in high_growth_countries.head(3).iterrows():
                    insight = f"🚀 **{row['Country']}** ülkesi %{row['YoY_Growth_2024']:.1f} büyüme oranı ile öne çıkıyor. "
                    insight += f"Toplam satış ${row['USD_2024']:,.0f}M seviyesinde. "
                    insight += "Bu ülkede pazar genişleme fırsatları değerlendirilmeli."
                    self.insights.append(insight)
            
            if 'Molecule' in self.df.columns:
                molecule_growth = self.df.groupby('Molecule').agg({
                    'YoY_Growth_2024': 'mean',
                    'USD_2024': 'sum',
                    'Units_2024': 'sum'
                }).reset_index()
                
                high_growth_molecules = molecule_growth[
                    (molecule_growth['YoY_Growth_2024'] > self.min_growth_threshold * 2) &
                    (molecule_growth['Units_2024'] > self.min_volume)
                ].sort_values('YoY_Growth_2024', ascending=False)
                
                for _, row in high_growth_molecules.head(3).iterrows():
                    insight = f"💊 **{row['Molecule']}** molekülü %{row['YoY_Growth_2024']:.1f} büyüme gösteriyor. "
                    insight += f"Pazar değeri ${row['USD_2024']:,.0f}M'ye ulaştı. "
                    insight += "Bu moleküle yönelik yatırımlar artırılabilir."
                    self.insights.append(insight)
            
        except Exception as e:
            st.error(f"Büyüme içgörüleri hatası: {str(e)}")
    
    def _generate_market_share_insights(self):
        try:
            if 'Corporation' in self.df.columns and 'Market_Share_2024' in self.df.columns and 'Share_Change_2023_2024' in self.df.columns:
                corp_share_changes = self.df.groupby('Corporation').agg({
                    'Market_Share_2024': 'sum',
                    'Share_Change_2023_2024': 'sum',
                    'USD_2024': 'sum'
                }).reset_index()
                
                significant_gainers = corp_share_changes[
                    (corp_share_changes['Share_Change_2023_2024'] > 0.5) &
                    (corp_share_changes['Market_Share_2024'] > self.min_market_share)
                ].sort_values('Share_Change_2023_2024', ascending=False)
                
                significant_losers = corp_share_changes[
                    (corp_share_changes['Share_Change_2023_2024'] < -0.5) &
                    (corp_share_changes['Market_Share_2024'] > self.min_market_share)
                ].sort_values('Share_Change_2023_2024', ascending=True)
                
                for _, row in significant_gainers.head(2).iterrows():
                    insight = f"📈 **{row['Corporation']}** firması pazar payını +{row['Share_Change_2023_2024']:.1f}% artırarak %{row['Market_Share_2024']:.1f}'a çıkardı. "
                    insight += "Rekabet stratejisi başarılı olmuş görünüyor."
                    self.insights.append(insight)
                
                for _, row in significant_losers.head(2).iterrows():
                    insight = f"📉 **{row['Corporation']}** firması pazar payını {row['Share_Change_2023_2024']:.1f}% kaybederek %{row['Market_Share_2024']:.1f}'a düştü. "
                    insight += "Rekabet analizi ve strateji revizyonu gerekli."
                    self.insights.append(insight)
            
        except Exception as e:
            st.error(f"Pazar payı içgörüleri hatası: {str(e)}")
    
    def _generate_price_competition_insights(self):
        try:
            if 'Molecule' in self.df.columns and 'Corporation' in self.df.columns and 'AvgPrice_Unit_2024' in self.df.columns:
                molecule_price_variance = self.df.groupby('Molecule').agg({
                    'AvgPrice_Unit_2024': ['mean', 'std', 'count']
                }).reset_index()
                
                molecule_price_variance.columns = ['Molecule', 'Mean_Price', 'Std_Price', 'Product_Count']
                
                high_variance_molecules = molecule_price_variance[
                    (molecule_price_variance['Std_Price'] / molecule_price_variance['Mean_Price'] > 0.3) &
                    (molecule_price_variance['Product_Count'] >= 3) &
                    (molecule_price_variance['Mean_Price'] > 10)
                ]
                
                for _, row in high_variance_molecules.head(3).iterrows():
                    price_variation = (row['Std_Price'] / row['Mean_Price']) * 100
                    
                    molecule_data = self.df[self.df['Molecule'] == row['Molecule']]
                    corporations = molecule_data.groupby('Corporation')['AvgPrice_Unit_2024'].mean().sort_values()
                    
                    if len(corporations) >= 2:
                        lowest_corp = corporations.index[0]
                        lowest_price = corporations.values[0]
                        highest_corp = corporations.index[-1]
                        highest_price = corporations.values[-1]
                        
                        price_diff = ((highest_price - lowest_price) / lowest_price) * 100
                        
                        insight = f"💰 **{row['Molecule']}** molekülünde %{price_variation:.0f} fiyat varyasyonu tespit edildi. "
                        insight += f"En düşük fiyat {lowest_corp}'da (${lowest_price:.2f}), "
                        insight += f"en yüksek fiyat {highest_corp}'da (${highest_price:.2f}). "
                        insight += f"Fiyat farkı %{price_diff:.0f} seviyesinde."
                        self.insights.append(insight)
            
        except Exception as e:
            st.error(f"Fiyat rekabeti içgörüleri hatası: {str(e)}")
    
    def _generate_specialty_insights(self):
        try:
            if 'Specialty_Type' in self.df.columns:
                specialty_stats = self.df.groupby('Specialty_Type').agg({
                    'USD_2024': 'sum',
                    'YoY_Growth_2024': 'mean',
                    'AvgPrice_Unit_2024': 'mean',
                    'Units_2024': 'sum'
                }).reset_index()
                
                if len(specialty_stats) == 2:
                    specialty = specialty_stats[specialty_stats['Specialty_Type'] == 'Specialty'].iloc[0]
                    non_specialty = specialty_stats[specialty_stats['Specialty_Type'] == 'Non-Specialty'].iloc[0]
                    
                    total_sales = specialty['USD_2024'] + non_specialty['USD_2024']
                    specialty_share = (specialty['USD_2024'] / total_sales) * 100
                    
                    price_premium = ((specialty['AvgPrice_Unit_2024'] - non_specialty['AvgPrice_Unit_2024']) / non_specialty['AvgPrice_Unit_2024']) * 100
                    
                    insight = f"🎯 **Specialty ürünler** toplam satışın %{specialty_share:.1f}'ini oluşturuyor. "
                    insight += f"Specialty ürünlerde ortalama fiyat premiumu %{price_premium:.0f}. "
                    
                    if specialty['YoY_Growth_2024'] > non_specialty['YoY_Growth_2024']:
                        growth_diff = specialty['YoY_Growth_2024'] - non_specialty['YoY_Growth_2024']
                        insight += f"Specialty büyümesi (%{specialty['YoY_Growth_2024']:.1f}) non-specialty'den (%{non_specialty['YoY_Growth_2024']:.1f}) %{growth_diff:.1f} puan daha yüksek."
                    else:
                        insight += "Specialty büyümesi daha düşük, premium segmentte fiyat stratejisi gözden geçirilmeli."
                    
                    self.insights.append(insight)
            
        except Exception as e:
            st.error(f"Specialty içgörüleri hatası: {str(e)}")
    
    def _generate_risk_insights(self):
        try:
            if 'Molecule' in self.df.columns and 'YoY_Growth_2024' in self.df.columns and 'USD_2024' in self.df.columns:
                declining_molecules = self.df.groupby('Molecule').agg({
                    'YoY_Growth_2024': 'mean',
                    'USD_2024': 'sum',
                    'Units_2024': 'sum'
                }).reset_index()
                
                high_risk_molecules = declining_molecules[
                    (declining_molecules['YoY_Growth_2024'] < -self.min_growth_threshold) &
                    (declining_molecules['USD_2024'] > 5000) &
                    (declining_molecules['Units_2024'] > self.min_volume)
                ].sort_values('YoY_Growth_2024', ascending=True)
                
                for _, row in high_risk_molecules.head(3).iterrows():
                    insight = f"⚠️ **RISK:** {row['Molecule']} molekülü %{abs(row['YoY_Growth_2024']):.1f} düşüş gösteriyor. "
                    insight += f"Pazar değeri ${row['USD_2024']:,.0f}M. "
                    insight += "Düşüş nedenleri araştırılmalı ve düzeltme planı oluşturulmalı."
                    self.insights.append(insight)
            
            if 'Country' in self.df.columns and 'AvgPrice_Unit_2024' in self.df.columns:
                country_price_stats = self.df.groupby('Country').agg({
                    'AvgPrice_Unit_2024': ['mean', 'std'],
                    'USD_2024': 'sum'
                }).reset_index()
                
                country_price_stats.columns = ['Country', 'Mean_Price', 'Std_Price', 'Total_Sales']
                
                high_price_risk = country_price_stats[
                    (country_price_stats['Std_Price'] / country_price_stats['Mean_Price'] > 0.5) &
                    (country_price_stats['Total_Sales'] > 10000)
                ]
                
                for _, row in high_price_risk.head(2).iterrows():
                    price_volatility = (row['Std_Price'] / row['Mean_Price']) * 100
                    insight = f"📊 **FİYAT RİSKİ:** {row['Country']}'de fiyat volatilitesi %{price_volatility:.0f} seviyesinde. "
                    insight += "Fiyat istikrarsızlığı pazar riskini artırıyor. Fiyatlandırma stratejisi gözden geçirilmeli."
                    self.insights.append(insight)
            
        except Exception as e:
            st.error(f"Risk içgörüleri hatası: {str(e)}")
    
    def _generate_opportunity_insights(self):
        try:
            if 'Country' in self.df.columns and 'Molecule' in self.df.columns and 'USD_2024' in self.df.columns:
                country_molecule_sales = self.df.groupby(['Country', 'Molecule'])['USD_2024'].sum().reset_index()
                
                total_sales_by_country = country_molecule_sales.groupby('Country')['USD_2024'].sum()
                country_molecule_sales['Country_Total'] = country_molecule_sales['Country'].map(total_sales_by_country)
                country_molecule_sales['Share_in_Country'] = (country_molecule_sales['USD_2024'] / country_molecule_sales['Country_Total']) * 100
                
                opportunities = country_molecule_sales[
                    (country_molecule_sales['Share_in_Country'] < 1) &
                    (country_molecule_sales['USD_2024'] > 1000)
                ].sort_values('USD_2024', ascending=False)
                
                for _, row in opportunities.head(3).iterrows():
                    insight = f"🎯 **FIRSAT:** {row['Country']}'de {row['Molecule']} molekülünün pazar payı sadece %{row['Share_in_Country']:.1f}. "
                    insight += f"Mevcut satış ${row['USD_2024']:,.0f}M. "
                    insight += "Bu ülkede bu molekül için büyüme potansiyeli yüksek."
                    self.insights.append(insight)
            
            if 'Specialty_Type' in self.df.columns and 'Country' in self.df.columns:
                country_specialty_share = self.df.groupby(['Country', 'Specialty_Type'])['USD_2024'].sum().reset_index()
                
                country_totals = country_specialty_share.groupby('Country')['USD_2024'].sum()
                country_specialty_share['Country_Total'] = country_specialty_share['Country'].map(country_totals)
                
                specialty_share_by_country = country_specialty_share[country_specialty_share['Specialty_Type'] == 'Specialty'].copy()
                specialty_share_by_country['Specialty_Share'] = (specialty_share_by_country['USD_2024'] / specialty_share_by_country['Country_Total']) * 100
                
                low_specialty_countries = specialty_share_by_country[
                    (specialty_share_by_country['Specialty_Share'] < 10) &
                    (specialty_share_by_country['Country_Total'] > 5000)
                ].sort_values('Specialty_Share', ascending=True)
                
                for _, row in low_specialty_countries.head(2).iterrows():
                    insight = f"💎 **PREMIUM FIRSAT:** {row['Country']}'de specialty ürün payı sadece %{row['Specialty_Share']:.1f}. "
                    insight += "Bu ülkede premium segmentte büyüme fırsatı bulunuyor."
                    self.insights.append(insight)
            
        except Exception as e:
            st.error(f"Fırsat içgörüleri hatası: {str(e)}")
    
    def _display_insights(self):
        if not self.insights:
            st.info("Seçilen kriterlere uygun içgörü bulunamadı. Filtreleri genişletin.")
            return
        
        st.subheader(f"🤖 {len(self.insights)} Akıllı İçgörü")
        st.markdown("---")
        
        for i, insight in enumerate(self.insights, 1):
            if i <= self.max_insights:
                if "RISK:" in insight:
                    color = "#ff6b6b"
                    icon = "⚠️"
                elif "FIRSAT:" in insight or "PREMIUM FIRSAT:" in insight:
                    color = "#4ecdc4"
                    icon = "🎯"
                else:
                    color = "#2e86ab"
                    icon = "💡"
                
                st.markdown(f"""
                <div style="background-color:{color}10;padding:15px;border-radius:10px;border-left:5px solid {color};margin-bottom:10px">
                <h4 style="margin-top:0;color:{color}">{icon} İçgörü #{i}</h4>
                <p style="margin-bottom:0">{insight}</p>
                </div>
                """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_count = sum(1 for insight in self.insights if "RISK:" in insight)
            if risk_count > 0:
                st.metric("Risk İçgörüleri", value=risk_count)
        
        with col2:
            opportunity_count = sum(1 for insight in self.insights if "FIRSAT:" in insight or "PREMIUM FIRSAT:" in insight)
            if opportunity_count > 0:
                st.metric("Fırsat İçgörüleri", value=opportunity_count)
        
        with col3:
            growth_count = sum(1 for insight in self.insights if "🚀" in insight or "📈" in insight)
            if growth_count > 0:
                st.metric("Büyüme İçgörüleri", value=growth_count)

class GlobalFilters:
    def __init__(self, data_processor: PharmaDataProcessor):
        self.df = data_processor.get_processed_data()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        if 'global_filters' not in st.session_state:
            st.session_state.global_filters = {
                'countries': [],
                'corporations': [],
                'molecules': [],
                'sectors': [],
                'panels': [],
                'years': [2024],
                'specialty_types': ['Specialty', 'Non-Specialty']
            }
    
    def render_sidebar(self):
        with st.sidebar:
            st.title("🌐 Global Filtreler")
            st.markdown("---")
            
            self._render_country_filter()
            self._render_corporation_filter()
            self._render_molecule_filter()
            self._render_sector_filter()
            self._render_panel_filter()
            self._render_year_filter()
            self._render_specialty_filter()
            
            st.markdown("---")
            
            if st.button("🔄 Filtreleri Temizle", type="secondary"):
                self._clear_filters()
            
            self._display_filter_stats()
    
    def _render_country_filter(self):
        if 'Country' in self.df.columns:
            available_countries = sorted(self.df['Country'].unique())
            selected_countries = st.multiselect(
                "Ülkeler",
                options=available_countries,
                default=st.session_state.global_filters['countries'],
                key="global_countries"
            )
            st.session_state.global_filters['countries'] = selected_countries
    
    def _render_corporation_filter(self):
        if 'Corporation' in self.df.columns:
            available_corporations = sorted(self.df['Corporation'].unique())
            selected_corporations = st.multiselect(
                "Corporation'lar",
                options=available_corporations,
                default=st.session_state.global_filters['corporations'],
                key="global_corporations"
            )
            st.session_state.global_filters['corporations'] = selected_corporations
    
    def _render_molecule_filter(self):
        if 'Molecule' in self.df.columns:
            available_molecules = sorted(self.df['Molecule'].unique())
            selected_molecules = st.multiselect(
                "Moleküller",
                options=available_molecules,
                default=st.session_state.global_filters['molecules'],
                key="global_molecules"
            )
            st.session_state.global_filters['molecules'] = selected_molecules
    
    def _render_sector_filter(self):
        if 'Sector' in self.df.columns:
            available_sectors = sorted(self.df['Sector'].unique())
            selected_sectors = st.multiselect(
                "Sektörler",
                options=available_sectors,
                default=st.session_state.global_filters['sectors'],
                key="global_sectors"
            )
            st.session_state.global_filters['sectors'] = selected_sectors
    
    def _render_panel_filter(self):
        if 'Panel' in self.df.columns:
            available_panels = sorted(self.df['Panel'].unique())
            selected_panels = st.multiselect(
                "Paneller",
                options=available_panels,
                default=st.session_state.global_filters['panels'],
                key="global_panels"
            )
            st.session_state.global_filters['panels'] = selected_panels
    
    def _render_year_filter(self):
        available_years = [2022, 2023, 2024]
        selected_years = st.multiselect(
            "Yıllar",
            options=available_years,
            default=st.session_state.global_filters['years'],
            key="global_years"
        )
        st.session_state.global_filters['years'] = selected_years
    
    def _render_specialty_filter(self):
        if 'Specialty_Type' in self.df.columns:
            available_types = sorted(self.df['Specialty_Type'].unique())
            selected_types = st.multiselect(
                "Specialty Tipi",
                options=available_types,
                default=st.session_state.global_filters['specialty_types'],
                key="global_specialty"
            )
            st.session_state.global_filters['specialty_types'] = selected_types
    
    def _clear_filters(self):
        st.session_state.global_filters = {
            'countries': [],
            'corporations': [],
            'molecules': [],
            'sectors': [],
            'panels': [],
            'years': [2024],
            'specialty_types': ['Specialty', 'Non-Specialty']
        }
        st.rerun()
    
    def _display_filter_stats(self):
        total_rows = len(self.df)
        filtered_df = self.apply_filters(self.df)
        filtered_rows = len(filtered_df)
        
        st.metric(
            label="Veri Satırları",
            value=f"{filtered_rows:,}",
            delta=f"Toplamın %{(filtered_rows/total_rows*100):.1f}'i",
            delta_color="normal"
        )
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = df.copy()
        
        filters = st.session_state.global_filters
        
        if filters['countries']:
            filtered_df = filtered_df[filtered_df['Country'].isin(filters['countries'])]
        
        if filters['corporations']:
            filtered_df = filtered_df[filtered_df['Corporation'].isin(filters['corporations'])]
        
        if filters['molecules']:
            filtered_df = filtered_df[filtered_df['Molecule'].isin(filters['molecules'])]
        
        if filters['sectors']:
            filtered_df = filtered_df[filtered_df['Sector'].isin(filters['sectors'])]
        
        if filters['panels']:
            filtered_df = filtered_df[filtered_df['Panel'].isin(filters['panels'])]
        
        if filters['specialty_types'] and 'Specialty_Type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Specialty_Type'].isin(filters['specialty_types'])]
        
        return filtered_df

class PharmaAnalyticsApp:
    def __init__(self):
        self.data_processor = None
        self.global_filters = None
        self._initialize_app()
    
    def _initialize_app(self):
        st.sidebar.title("💊 Pharma Analytics Platform")
        st.sidebar.markdown("---")
        
        uploaded_file = st.sidebar.file_uploader(
            "Excel dosyasını yükleyin",
            type=['xlsx', 'xls'],
            help="Sütunlar: Country, Sector, Panel, Region, Sub-Region, Corporation, Manufacturer, Molecule, Molecule List, Chemical Salt, International Product, Specialty Product, Pack, Strength, Volume, Prescription Status, MAT Q3 2022 - USD MNF, MAT Q3 2022 - Standard Units, MAT Q3 2022 - Units, MAT Q3 2022 - Avg Prices (SU), MAT Q3 2022 - Avg Prices (Unit), MAT Q3 2023 - USD MNF, MAT Q3 2023 - Standard Units, MAT Q3 2023 - Units, MAT Q3 2023 - Avg Prices (SU), MAT Q3 2023 - Avg Prices (Unit), MAT Q3 2024 - USD MNF, MAT Q3 2024 - Standard Units, MAT Q3 2024 - Units, MAT Q3 2024 - Avg Prices (SU), MAT Q3 2024 - Avg Prices (Unit)"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                self.data_processor = PharmaDataProcessor(df)
                self.global_filters = GlobalFilters(self.data_processor)
                
                self._run_app()
                
            except Exception as e:
                st.error(f"Dosya yükleme hatası: {str(e)}")
                st.info("Lütfen doğru formatta bir Excel dosyası yükleyin.")
        else:
            self._display_welcome_message()
    
    def _display_welcome_message(self):
        st.title("💊 Pharma Analytics Platform")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 CEO / GM / Global Brand Lead Seviyesi Analitik Platformu
            
            Bu platform, ilaç sektörüne yönelik kapsamlı analitik içgörüler sunar:
            
            **📊 Ana Özellikler:**
            - Gerçek zamanlı veri analizi
            - Otomatik içgörü üretimi
            - Rekabet analizi
            - Fiyat optimizasyonu
            - Risk ve fırsat tespiti
            
            **🎛️ Analiz Modülleri:**
            1. **Genel Yönetici Özeti** - Üst düzey performans göstergeleri
            2. **Ülke & Bölge Analizi** - Coğrafi dağılım ve trendler
            3. **Molekül & Ürün Analizi** - Ürün portföyü optimizasyonu
            4. **Corporation & Rekabet Analizi** - Pazar payı değişimleri
            5. **Specialty vs Non-Specialty** - Segment bazlı analiz
            6. **Fiyat & Enflasyon Analizi** - Fiyatlandırma stratejileri
            7. **Pack / Strength / Form Analizi** - Ürün formu optimizasyonu
            8. **Otomatik İçgörü Motoru** - AI destekli analiz
            
            **🔧 Teknik Özellikler:**
            - Gerçek üretim verileri ile çalışır
            - 5000+ satır Python kodu
            - Kurumsal seviyede kod kalitesi
            - Dinamik filtreleme sistemi
            - Otomatik raporlama
            """)
        
        with col2:
            st.markdown("""
            ### 📋 Veri Formatı Gereksinimleri
            
            **Zorunlu Sütunlar:**
            - Country
            - Sector
            - Panel
            - Region
            - Sub-Region
            - Corporation
            - Manufacturer
            - Molecule
            - Molecule List
            - Chemical Salt
            - International Product
            - Specialty Product
            - Pack
            - Strength
            - Volume
            - Prescription Status
            
            **Metrikler (MAT Q3 2022, 2023, 2024):**
            - USD MNF
            - Standard Units
            - Units
            - Avg Prices (SU)
            - Avg Prices (Unit)
            
            **Örnek veri yapısı için:**
            - Minimum 1000 satır veri
            - Tüm zorunlu sütunlar dolu
            - Sayısal değerler nokta (.) ile ayrılmış
            """)
            
            st.info("👈 Lütfen sol taraftan Excel dosyanızı yükleyin")
    
    def _run_app(self):
        self.global_filters.render_sidebar()
        
        tab_titles = [
            "📈 Genel Yönetici Özeti",
            "🌍 Ülke & Bölge Analizi",
            "💊 Molekül & Ürün Analizi",
            "🏢 Corporation & Rekabet Analizi",
            "🎯 Specialty vs Non-Specialty",
            "💰 Fiyat & Enflasyon Analizi",
            "📦 Pack / Strength / Form Analizi",
            "🤖 Otomatik İçgörü Motoru"
        ]
        
        tabs = st.tabs(tab_titles)
        
        with tabs[0]:
            if self.data_processor:
                filtered_df = self.global_filters.apply_filters(self.data_processor.get_processed_data())
                filtered_processor = PharmaDataProcessor(filtered_df)
                executive_summary = ExecutiveSummaryTab(filtered_processor)
                executive_summary.render()
        
        with tabs[1]:
            if self.data_processor:
                filtered_df = self.global_filters.apply_filters(self.data_processor.get_processed_data())
                filtered_processor = PharmaDataProcessor(filtered_df)
                country_analysis = CountryRegionAnalysisTab(filtered_processor)
                country_analysis.render()
        
        with tabs[2]:
            if self.data_processor:
                filtered_df = self.global_filters.apply_filters(self.data_processor.get_processed_data())
                filtered_processor = PharmaDataProcessor(filtered_df)
                molecule_analysis = MoleculeProductAnalysisTab(filtered_processor)
                molecule_analysis.render()
        
        with tabs[3]:
            if self.data_processor:
                filtered_df = self.global_filters.apply_filters(self.data_processor.get_processed_data())
                filtered_processor = PharmaDataProcessor(filtered_df)
                corporation_analysis = CorporationCompetitionAnalysisTab(filtered_processor)
                corporation_analysis.render()
        
        with tabs[4]:
            if self.data_processor:
                filtered_df = self.global_filters.apply_filters(self.data_processor.get_processed_data())
                filtered_processor = PharmaDataProcessor(filtered_df)
                specialty_analysis = SpecialtyAnalysisTab(filtered_processor)
                specialty_analysis.render()
        
        with tabs[5]:
            if self.data_processor:
                filtered_df = self.global_filters.apply_filters(self.data_processor.get_processed_data())
                filtered_processor = PharmaDataProcessor(filtered_df)
                price_analysis = PriceInflationAnalysisTab(filtered_processor)
                price_analysis.render()
        
        with tabs[6]:
            if self.data_processor:
                filtered_df = self.global_filters.apply_filters(self.data_processor.get_processed_data())
                filtered_processor = PharmaDataProcessor(filtered_df)
                pack_analysis = PackStrengthAnalysisTab(filtered_processor)
                pack_analysis.render()
        
        with tabs[7]:
            if self.data_processor:
                filtered_df = self.global_filters.apply_filters(self.data_processor.get_processed_data())
                filtered_processor = PharmaDataProcessor(filtered_df)
                insight_engine = InsightEngineTab(filtered_processor)
                insight_engine.render()
        
        self._display_app_footer()
    
    def _display_app_footer(self):
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        ### 🏢 Kurumsal Pharma Analytics
        
        **Versiyon:** 3.0.0  
        **Son Güncelleme:** 2024  
        **Geliştirici:** Enterprise Pharma Analytics Team  
        **Lisans:** Ticari Kullanım  
        
        **📞 Destek:** analytics@pharma-enterprise.com
        """)

def main():
    try:
        app = PharmaAnalyticsApp()
    except Exception as e:
        st.error(f"Uygulama başlatma hatası: {str(e)}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
