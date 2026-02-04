import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import pycountry
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime
import math
import itertools
import re
from scipy import stats
import hashlib
import time

# ============================================
# KONFİGÜRASYON VE STİL
# ============================================
st.set_page_config(
    page_title="Pharma Commercial Analytics Suite",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid #10B981;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .insight-card {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #D97706;
    }
    .warning-card {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #DC2626;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F8FAFC;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #E2E8F0;
        border-radius: 8px 8px 0 0;
        gap: 8px;
        padding: 12px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .stButton button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        color: #6B7280;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ============================================
# VERİ YÜKLEME VE VALİDASYON
# ============================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_and_validate_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        required_columns = [
            'Source.Name', 'Country', 'Sector', 'Panel', 'Region', 'Sub-Region',
            'Corporation', 'Manufacturer', 'Molecule List', 'Molecule', 'Chemical Salt',
            'International Product', 'Specialty Product', 'NFC123', 'International Pack',
            'International Strength', 'International Size', 'International Volume',
            'International Prescription',
            'MAT Q3 2022\nUSD MNF', 'MAT Q3 2022\nStandard Units', 'MAT Q3 2022\nUnits',
            'MAT Q3 2022\nSU Avg Price USD MNF', 'MAT Q3 2022\nUnit Avg Price USD MNF',
            'MAT Q3 2023\nUSD MNF', 'MAT Q3 2023\nStandard Units', 'MAT Q3 2023\nUnits',
            'MAT Q3 2023\nSU Avg Price USD MNF', 'MAT Q3 2023\nUnit Avg Price USD MNF',
            'MAT Q3 2024\nUSD MNF', 'MAT Q3 2024\nStandard Units', 'MAT Q3 2024\nUnits',
            'MAT Q3 2024\nSU Avg Price USD MNF', 'MAT Q3 2024\nUnit Avg Price USD MNF'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"EKSİK KOLONLAR: {missing_cols}")
            st.stop()
        
        extra_cols = [col for col in df.columns if col not in required_columns]
        if extra_cols:
            st.warning(f"EKSTRA KOLONLAR (göz ardı edilecek): {extra_cols}")
        
        df = df[required_columns]
        
        numeric_columns = [col for col in df.columns if any(x in col for x in ['USD', 'Units', 'Price'])]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        st.stop()

# ============================================
# GLOBAL FİLTRELER
# ============================================
class GlobalFilters:
    def __init__(self, df):
        self.df = df
        self.initialize_filters()
    
    def initialize_filters(self):
        with st.sidebar:
            st.markdown("### 🌍 GLOBAL FİLTRELER")
            
            # Country (çoklu seçim)
            all_countries = sorted(self.df['Country'].dropna().unique())
            self.selected_countries = st.multiselect(
                "Ülke (Çoklu Seçim)",
                options=all_countries,
                default=all_countries[:5] if len(all_countries) > 5 else all_countries,
                key="country_filter"
            )
            
            # Region
            all_regions = sorted(self.df['Region'].dropna().unique())
            self.selected_region = st.selectbox(
                "Bölge",
                options=["Tümü"] + all_regions,
                key="region_filter"
            )
            
            # Sub-Region
            all_subregions = sorted(self.df['Sub-Region'].dropna().unique())
            self.selected_subregion = st.selectbox(
                "Alt Bölge",
                options=["Tümü"] + all_subregions,
                key="subregion_filter"
            )
            
            # Sector
            all_sectors = sorted(self.df['Sector'].dropna().unique())
            self.selected_sector = st.selectbox(
                "Sektör",
                options=["Tümü"] + all_sectors,
                key="sector_filter"
            )
            
            # Panel
            all_panels = sorted(self.df['Panel'].dropna().unique())
            self.selected_panel = st.selectbox(
                "Panel",
                options=["Tümü"] + all_panels,
                key="panel_filter"
            )
            
            # Corporation (çoklu)
            all_corps = sorted(self.df['Corporation'].dropna().unique())
            self.selected_corps = st.multiselect(
                "Kuruluş (Çoklu)",
                options=all_corps,
                default=all_corps[:3] if len(all_corps) > 3 else all_corps,
                key="corp_filter"
            )
            
            # Manufacturer
            all_manufacturers = sorted(self.df['Manufacturer'].dropna().unique())
            self.selected_manufacturer = st.selectbox(
                "Üretici",
                options=["Tümü"] + all_manufacturers,
                key="manufacturer_filter"
            )
            
            # Molecule (çoklu)
            all_molecules = sorted(self.df['Molecule'].dropna().unique())
            self.selected_molecules = st.multiselect(
                "Molekül (Çoklu)",
                options=all_molecules,
                default=all_molecules[:5] if len(all_molecules) > 5 else all_molecules,
                key="molecule_filter"
            )
            
            # Specialty Product
            all_specialty = sorted(self.df['Specialty Product'].dropna().unique())
            self.selected_specialty = st.selectbox(
                "Özel Ürün",
                options=["Tümü"] + all_specialty,
                key="specialty_filter"
            )
            
            # International Product
            all_intl = sorted(self.df['International Product'].dropna().unique())
            self.selected_intl = st.selectbox(
                "Uluslararası Ürün",
                options=["Tümü"] + all_intl,
                key="intl_filter"
            )
    
    def apply_filters(self, df):
        filtered_df = df.copy()
        
        if self.selected_countries:
            filtered_df = filtered_df[filtered_df['Country'].isin(self.selected_countries)]
        
        if self.selected_region != "Tümü":
            filtered_df = filtered_df[filtered_df['Region'] == self.selected_region]
        
        if self.selected_subregion != "Tümü":
            filtered_df = filtered_df[filtered_df['Sub-Region'] == self.selected_subregion]
        
        if self.selected_sector != "Tümü":
            filtered_df = filtered_df[filtered_df['Sector'] == self.selected_sector]
        
        if self.selected_panel != "Tümü":
            filtered_df = filtered_df[filtered_df['Panel'] == self.selected_panel]
        
        if self.selected_corps:
            filtered_df = filtered_df[filtered_df['Corporation'].isin(self.selected_corps)]
        
        if self.selected_manufacturer != "Tümü":
            filtered_df = filtered_df[filtered_df['Manufacturer'] == self.selected_manufacturer]
        
        if self.selected_molecules:
            filtered_df = filtered_df[filtered_df['Molecule'].isin(self.selected_molecules)]
        
        if self.selected_specialty != "Tümü":
            filtered_df = filtered_df[filtered_df['Specialty Product'] == self.selected_specialty]
        
        if self.selected_intl != "Tümü":
            filtered_df = filtered_df[filtered_df['International Product'] == self.selected_intl]
        
        return filtered_df

# ============================================
# HARİTA İŞLEMLERİ
# ============================================
class WorldMapHandler:
    def __init__(self):
        self.world = self._load_world_geojson()
    
    def _load_world_geojson(self):
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            return world
        except:
            return None
    
    def get_country_code(self, country_name):
        try:
            country = pycountry.countries.get(name=country_name)
            if country:
                return country.alpha_3
            
            name_mapping = {
                'USA': 'United States',
                'UK': 'United Kingdom',
                'UAE': 'United Arab Emirates',
                'Russia': 'Russian Federation',
                'Iran': 'Iran, Islamic Republic of',
                'South Korea': 'Korea, Republic of',
                'North Korea': "Korea, Democratic People's Republic of",
                'Vietnam': 'Viet Nam',
                'Bolivia': 'Bolivia, Plurinational State of',
                'Venezuela': 'Venezuela, Bolivarian Republic of',
                'Syria': 'Syrian Arab Republic',
                'Tanzania': 'Tanzania, United Republic of',
                'Laos': "Lao People's Democratic Republic",
                'Brunei': 'Brunei Darussalam',
                'Cape Verde': 'Cabo Verde',
                'Congo': 'Congo',
                'Congo DR': 'Congo, The Democratic Republic of the',
                'Ivory Coast': "Côte d'Ivoire",
                'East Timor': 'Timor-Leste',
                'Macedonia': 'North Macedonia',
                'Moldova': 'Moldova, Republic of',
                'Palestine': 'Palestine, State of',
                'Taiwan': 'Taiwan, Province of China',
                'Turkey': 'Türkiye'
            }
            
            if country_name in name_mapping:
                country = pycountry.countries.get(name=name_mapping[country_name])
                if country:
                    return country.alpha_3
            
            return None
        except:
            return None
    
    def prepare_map_data(self, df, year):
        if year == 2022:
            usd_col = 'MAT Q3 2022\nUSD MNF'
            units_col = 'MAT Q3 2022\nUnits'
            su_col = 'MAT Q3 2022\nStandard Units'
        elif year == 2023:
            usd_col = 'MAT Q3 2023\nUSD MNF'
            units_col = 'MAT Q3 2023\nUnits'
            su_col = 'MAT Q3 2023\nStandard Units'
        elif year == 2024:
            usd_col = 'MAT Q3 2024\nUSD MNF'
            units_col = 'MAT Q3 2024\nUnits'
            su_col = 'MAT Q3 2024\nStandard Units'
        else:
            return pd.DataFrame()
        
        country_data = df.groupby('Country').agg({
            usd_col: 'sum',
            units_col: 'sum',
            su_col: 'sum'
        }).reset_index()
        
        total_usd = country_data[usd_col].sum()
        if total_usd > 0:
            country_data['Global_Pay_Pct'] = (country_data[usd_col] / total_usd) * 100
        else:
            country_data['Global_Pay_Pct'] = 0
        
        country_data['ISO_A3'] = country_data['Country'].apply(self.get_country_code)
        country_data = country_data.dropna(subset=['ISO_A3'])
        
        return country_data

# ============================================
# ANALİTİK MOTORLARI
# ============================================
class AnalyticsEngine:
    @staticmethod
    def calculate_growth(df, start_year, end_year):
        if start_year == 2022 and end_year == 2023:
            start_col = 'MAT Q3 2022\nUSD MNF'
            end_col = 'MAT Q3 2023\nUSD MNF'
        elif start_year == 2023 and end_year == 2024:
            start_col = 'MAT Q3 2023\nUSD MNF'
            end_col = 'MAT Q3 2024\nUSD MNF'
        elif start_year == 2022 and end_year == 2024:
            start_col = 'MAT Q3 2022\nUSD MNF'
            end_col = 'MAT Q3 2024\nUSD MNF'
        else:
            return 0, 0
        
        start_total = df[start_col].sum()
        end_total = df[end_col].sum()
        
        if start_total == 0:
            return 0, end_total
        
        growth_pct = ((end_total - start_total) / start_total) * 100
        growth_abs = end_total - start_total
        
        return growth_pct, growth_abs
    
    @staticmethod
    def price_volume_mix_analysis(df, start_year, end_year):
        if start_year == 2022 and end_year == 2023:
            usd_start = 'MAT Q3 2022\nUSD MNF'
            usd_end = 'MAT Q3 2023\nUSD MNF'
            units_start = 'MAT Q3 2022\nUnits'
            units_end = 'MAT Q3 2023\nUnits'
            price_start = 'MAT Q3 2022\nUnit Avg Price USD MNF'
            price_end = 'MAT Q3 2023\nUnit Avg Price USD MNF'
        elif start_year == 2023 and end_year == 2024:
            usd_start = 'MAT Q3 2023\nUSD MNF'
            usd_end = 'MAT Q3 2024\nUSD MNF'
            units_start = 'MAT Q3 2023\nUnits'
            units_end = 'MAT Q3 2024\nUnits'
            price_start = 'MAT Q3 2023\nUnit Avg Price USD MNF'
            price_end = 'MAT Q3 2024\nUnit Avg Price USD MNF'
        else:
            return {}
        
        total_start_usd = df[usd_start].sum()
        total_end_usd = df[usd_end].sum()
        total_start_units = df[units_start].sum()
        total_end_units = df[units_end].sum()
        
        if total_start_usd == 0 or total_start_units == 0:
            return {
                'price_effect': 0,
                'volume_effect': 0,
                'mix_effect': 0,
                'total_growth': 0
            }
        
        weighted_price_start = total_start_usd / total_start_units
        weighted_price_end = total_end_usd / total_end_units
        
        price_effect = (weighted_price_end - weighted_price_start) * total_start_units
        volume_effect = weighted_price_start * (total_end_units - total_start_units)
        mix_effect = total_end_usd - (weighted_price_start * total_end_units)
        
        total_growth = total_end_usd - total_start_usd
        
        price_effect_pct = (price_effect / total_start_usd) * 100 if total_start_usd != 0 else 0
        volume_effect_pct = (volume_effect / total_start_usd) * 100 if total_start_usd != 0 else 0
        mix_effect_pct = (mix_effect / total_start_usd) * 100 if total_start_usd != 0 else 0
        
        return {
            'price_effect': price_effect,
            'volume_effect': volume_effect,
            'mix_effect': mix_effect,
            'total_growth': total_growth,
            'price_effect_pct': price_effect_pct,
            'volume_effect_pct': volume_effect_pct,
            'mix_effect_pct': mix_effect_pct,
            'weighted_price_start': weighted_price_start,
            'weighted_price_end': weighted_price_end,
            'unit_growth_pct': ((total_end_units - total_start_units) / total_start_units * 100) if total_start_units != 0 else 0
        }
    
    @staticmethod
    def calculate_market_share(df, year, group_by='Corporation'):
        if year == 2022:
            usd_col = 'MAT Q3 2022\nUSD MNF'
        elif year == 2023:
            usd_col = 'MAT Q3 2023\nUSD MNF'
        elif year == 2024:
            usd_col = 'MAT Q3 2024\nUSD MNF'
        else:
            return pd.DataFrame()
        
        share_df = df.groupby(group_by).agg({
            usd_col: 'sum'
        }).reset_index()
        
        total_usd = share_df[usd_col].sum()
        if total_usd > 0:
            share_df['Market_Share_Pct'] = (share_df[usd_col] / total_usd) * 100
        else:
            share_df['Market_Share_Pct'] = 0
        
        share_df = share_df.sort_values('Market_Share_Pct', ascending=False)
        share_df['Cumulative_Share'] = share_df['Market_Share_Pct'].cumsum()
        
        return share_df
    
    @staticmethod
    def calculate_specialty_metrics(df):
        metrics = {}
        
        for year in [2022, 2023, 2024]:
            if year == 2022:
                usd_col = 'MAT Q3 2022\nUSD MNF'
            elif year == 2023:
                usd_col = 'MAT Q3 2023\nUSD MNF'
            else:
                usd_col = 'MAT Q3 2024\nUSD MNF'
            
            specialty_total = df[df['Specialty Product'] == 'Specialty'][usd_col].sum()
            non_specialty_total = df[df['Specialty Product'] != 'Specialty'][usd_col].sum()
            total_usd = specialty_total + non_specialty_total
            
            if total_usd > 0:
                specialty_pct = (specialty_total / total_usd) * 100
            else:
                specialty_pct = 0
            
            metrics[f'specialty_total_{year}'] = specialty_total
            metrics[f'non_specialty_total_{year}'] = non_specialty_total
            metrics[f'specialty_pct_{year}'] = specialty_pct
        
        return metrics

# ============================================
# OTOMATİK İÇGÖRÜ MOTORU
# ============================================
class InsightEngine:
    def __init__(self, df):
        self.df = df
    
    def generate_country_insights(self, country):
        country_df = self.df[self.df['Country'] == country]
        
        if len(country_df) == 0:
            return []
        
        insights = []
        
        # Büyüme analizi
        growth_22_23, _ = AnalyticsEngine.calculate_growth(country_df, 2022, 2023)
        growth_23_24, _ = AnalyticsEngine.calculate_growth(country_df, 2023, 2024)
        growth_22_24, _ = AnalyticsEngine.calculate_growth(country_df, 2022, 2024)
        
        if growth_22_23 > 20:
            insights.append(f"🇹🇷 **{country}**, 2022'den 2023'e **%{growth_22_23:.1f}** büyüme ile çok güçlü performans sergiledi.")
        elif growth_22_23 < -10:
            insights.append(f"⚠️ **{country}**, 2022'den 2023'e **%{growth_22_23:.1f}** küçülme yaşadı. Dikkat gerektiriyor.")
        
        if growth_23_24 > growth_22_23:
            insights.append(f"📈 **{country}**, büyüme hızını artırdı: 2023-2024 (%{growth_23_24:.1f}) > 2022-2023 (%{growth_22_23:.1f})")
        
        # Global pay analizi
        global_share_2024 = (country_df['MAT Q3 2024\nUSD MNF'].sum() / self.df['MAT Q3 2024\nUSD MNF'].sum() * 100)
        
        if global_share_2024 > 5:
            insights.append(f"🌍 **{country}**, %{global_share_2024:.2f} global pay ile kilit pazarlardan biri.")
        elif global_share_2024 < 0.5:
            insights.append(f"🔍 **{country}**, sadece %{global_share_2024:.2f} global paya sahip. Büyüme potansiyeli incelenmeli.")
        
        # Molekül bazlı analiz
        mol_share = country_df.groupby('Molecule').agg({
            'MAT Q3 2024\nUSD MNF': 'sum'
        }).reset_index()
        mol_share['Share'] = (mol_share['MAT Q3 2024\nUSD MNF'] / mol_share['MAT Q3 2024\nUSD MNF'].sum() * 100)
        top_molecule = mol_share.nlargest(1, 'Share')
        
        if not top_molecule.empty:
            mol_name = top_molecule.iloc[0]['Molecule']
            mol_pct = top_molecule.iloc[0]['Share']
            insights.append(f"💊 En büyük molekül: **{mol_name}** (%{mol_pct:.1f} pay)")
        
        # Fiyat-Volume ayrıştırma
        pvm_22_23 = AnalyticsEngine.price_volume_mix_analysis(country_df, 2022, 2023)
        price_effect_pct = pvm_22_23.get('price_effect_pct', 0)
        
        if price_effect_pct > 10:
            insights.append(f"💰 Fiyat etkisi baskın: 2022-2023 büyümesinin %{price_effect_pct:.1f}'i fiyattan geldi.")
        elif price_effect_pct < -5:
            insights.append(f"📉 Fiyat erozyonu: 2022-2023'te fiyatlar %{abs(price_effect_pct):.1f} düştü.")
        
        return insights[:5]
    
    def generate_molecule_insights(self, molecule):
        mol_df = self.df[self.df['Molecule'] == molecule]
        
        if len(mol_df) == 0:
            return []
        
        insights = []
        
        # Global büyüme
        growth_22_23, _ = AnalyticsEngine.calculate_growth(mol_df, 2022, 2023)
        growth_23_24, _ = AnalyticsEngine.calculate_growth(mol_df, 2023, 2024)
        
        if growth_22_23 > 0 and growth_23_24 > 0:
            insights.append(f"🚀 **{molecule}** molekülü iki yıl üst üste büyüdü: %{growth_22_23:.1f} → %{growth_23_24:.1f}")
        
        # Ülke dağılımı
        country_share = mol_df.groupby('Country').agg({
            'MAT Q3 2024\nUSD MNF': 'sum'
        }).reset_index()
        country_share['Share'] = (country_share['MAT Q3 2024\nUSD MNF'] / country_share['MAT Q3 2024\nUSD MNF'].sum() * 100)
        top_countries = country_share.nlargest(3, 'Share')
        
        if len(top_countries) > 0:
            country_list = ", ".join([f"{row['Country']} (%{row['Share']:.1f})" for _, row in top_countries.iterrows()])
            insights.append(f"📍 En büyük pazarlar: {country_list}")
        
        # Üretici konsantrasyonu
        mfg_share = mol_df.groupby('Manufacturer').agg({
            'MAT Q3 2024\nUSD MNF': 'sum'
        }).reset_index()
        mfg_share['Share'] = (mfg_share['MAT Q3 2024\nUSD MNF'] / mfg_share['MAT Q3 2024\nUSD MNF'].sum() * 100)
        top_mfg = mfg_share.nlargest(1, 'Share')
        
        if not top_mfg.empty:
            mfg_name = top_mfg.iloc[0]['Manufacturer']
            mfg_pct = top_mfg.iloc[0]['Share']
            if mfg_pct > 50:
                insights.append(f"🏭 **{mfg_name}**, %{mfg_pct:.1f} pay ile pazara hakim.")
        
        return insights[:5]
    
    def generate_corporation_insights(self, corporation):
        corp_df = self.df[self.df['Corporation'] == corporation]
        
        if len(corp_df) == 0:
            return []
        
        insights = []
        
        # Pazar payı trendi
        share_2022 = (corp_df['MAT Q3 2022\nUSD MNF'].sum() / self.df['MAT Q3 2022\nUSD MNF'].sum() * 100)
        share_2023 = (corp_df['MAT Q3 2023\nUSD MNF'].sum() / self.df['MAT Q3 2023\nUSD MNF'].sum() * 100)
        share_2024 = (corp_df['MAT Q3 2024\nUSD MNF'].sum() / self.df['MAT Q3 2024\nUSD MNF'].sum() * 100)
        
        share_change_22_24 = share_2024 - share_2022
        
        if share_change_22_24 > 1:
            insights.append(f"📊 **{corporation}**, pazar payını %{share_change_22_24:.2f} artırdı (2022: %{share_2022:.2f} → 2024: %{share_2024:.2f})")
        elif share_change_22_24 < -1:
            insights.append(f"⚠️ **{corporation}**, pazar payını %{abs(share_change_22_24):.2f} kaybetti (2022: %{share_2022:.2f} → 2024: %{share_2024:.2f})")
        
        # Coğrafi çeşitlilik
        country_count = corp_df['Country'].nunique()
        if country_count > 20:
            insights.append(f"🌐 **{country_count} ülkede** varlık gösteriyor. Yüksek coğrafi çeşitlilik.")
        
        # Molekül konsantrasyonu
        top_mol = corp_df.groupby('Molecule').agg({
            'MAT Q3 2024\nUSD MNF': 'sum'
        }).reset_index().nlargest(1, 'MAT Q3 2024\nUSD MNF')
        
        if not top_mol.empty:
            mol_name = top_mol.iloc[0]['Molecule']
            mol_value = top_mol.iloc[0]['MAT Q3 2024\nUSD MNF']
            total_value = corp_df['MAT Q3 2024\nUSD MNF'].sum()
            mol_pct = (mol_value / total_value * 100) if total_value > 0 else 0
            
            if mol_pct > 30:
                insights.append(f"💡 En büyük molekül **{mol_name}**, toplamın %{mol_pct:.1f}'ini oluşturuyor.")
        
        return insights[:5]

# ============================================
# ANA UYGULAMA
# ============================================
def main():
    st.markdown('<h1 class="main-header">💊 Pharma Commercial Analytics Suite</h1>', unsafe_allow_html=True)
    st.markdown("### Üç Yıllık Zincir Analiz ve Global Ticari İstihbarat Platformu")
    
    # Veri yükleme
    uploaded_file = st.file_uploader("📂 Excel dosyasını yükleyin (.xlsx formatında)", type=["xlsx"])
    
    if not uploaded_file:
        st.warning("🔍 Lütfen analiz için Excel dosyasını yükleyin.")
        st.stop()
    
    # Veri yükleme ve validasyon
    with st.spinner("📊 Veri yükleniyor ve doğrulanıyor..."):
        df = load_and_validate_data(uploaded_file)
    
    st.success(f"✅ Veri başarıyla yüklendi: {len(df)} satır, {len(df.columns)} sütun")
    
    # Global filtreleri başlat
    filters = GlobalFilters(df)
    filtered_df = filters.apply_filters(df)
    
    st.sidebar.info(f"🔍 {len(filtered_df):,} satır filtrelendi")
    
    # Analytics Engine
    analytics = AnalyticsEngine()
    insight_engine = InsightEngine(filtered_df)
    map_handler = WorldMapHandler()
    
    # Sekmeler
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Yönetici Özeti",
        "🌍 Global Harita Analizi",
        "🇹🇷 Ülke Derinlemesine",
        "💊 Molekül & Ürün",
        "🏢 Corporation & Rekabet",
        "⭐ Specialty vs Non-Specialty",
        "📈 Fiyat – Volume – Mix",
        "🤖 Otomatik İçgörü Motoru"
    ])
    
    # ============================================
    # TAB 1: Yönetici Özeti
    # ============================================
    with tab1:
        st.markdown('<h2 class="sub-header">📊 Yönetici Özeti</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_2022 = filtered_df['MAT Q3 2022\nUSD MNF'].sum()
            total_2023 = filtered_df['MAT Q3 2023\nUSD MNF'].sum()
            total_2024 = filtered_df['MAT Q3 2024\nUSD MNF'].sum()
            
            growth_22_23, abs_22_23 = analytics.calculate_growth(filtered_df, 2022, 2023)
            growth_23_24, abs_23_24 = analytics.calculate_growth(filtered_df, 2023, 2024)
            
            st.markdown(f'''
            <div class="metric-card">
                <h3>🌍 Global USD MNF</h3>
                <p style="font-size: 2rem; font-weight: 800; color: #1E3A8A;">${total_2024:,.0f}M</p>
                <p>2022: ${total_2022:,.0f}M</p>
                <p>2023: ${total_2023:,.0f}M</p>
                <p style="color: {'#10B981' if growth_23_24 > 0 else '#DC2626'}; font-weight: 600;">
                    2023→2024: {'+' if growth_23_24 >= 0 else ''}{growth_23_24:.1f}%
                </p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            units_2024 = filtered_df['MAT Q3 2024\nUnits'].sum()
            su_2024 = filtered_df['MAT Q3 2024\nStandard Units'].sum()
            
            units_growth = ((filtered_df['MAT Q3 2024\nUnits'].sum() - filtered_df['MAT Q3 2023\nUnits'].sum()) / 
                           filtered_df['MAT Q3 2023\nUnits'].sum() * 100) if filtered_df['MAT Q3 2023\nUnits'].sum() > 0 else 0
            
            st.markdown(f'''
            <div class="metric-card">
                <h3>📦 Volume Metrikleri</h3>
                <p style="font-size: 1.5rem; font-weight: 700; color: #1E3A8A;">{units_2024:,.0f}</p>
                <p>Units (2024)</p>
                <p style="font-size: 1.5rem; font-weight: 700; color: #1E3A8A;">{su_2024:,.0f}</p>
                <p>Standard Units (2024)</p>
                <p style="color: {'#10B981' if units_growth > 0 else '#DC2626'}; font-weight: 600;">
                    Units Büyüme: {'+' if units_growth >= 0 else ''}{units_growth:.1f}%
                </p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            top_countries = filtered_df.groupby('Country').agg({
                'MAT Q3 2024\nUSD MNF': 'sum'
            }).reset_index().nlargest(3, 'MAT Q3 2024\nUSD MNF')
            
            country_list = ""
            for idx, row in top_countries.iterrows():
                share = (row['MAT Q3 2024\nUSD MNF'] / total_2024 * 100) if total_2024 > 0 else 0
                country_list += f"• {row['Country']}: %{share:.1f}<br>"
            
            st.markdown(f'''
            <div class="metric-card">
                <h3>🏆 Top 3 Ülke</h3>
                <div style="font-size: 1.1rem;">
                    {country_list}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Detaylı analiz
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📈 Üç Yıllık Trend")
            
            trend_data = pd.DataFrame({
                'Yıl': [2022, 2023, 2024],
                'USD MNF': [total_2022, total_2023, total_2024],
                'Units': [
                    filtered_df['MAT Q3 2022\nUnits'].sum(),
                    filtered_df['MAT Q3 2023\nUnits'].sum(),
                    filtered_df['MAT Q3 2024\nUnits'].sum()
                ]
            })
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=trend_data['Yıl'], y=trend_data['USD MNF'],
                      name='USD MNF', marker_color='#3B82F6'),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(x=trend_data['Yıl'], y=trend_data['Units'],
                          name='Units', mode='lines+markers',
                          line=dict(color='#10B981', width=3)),
                secondary_y=True,
            )
            
            fig.update_layout(
                title="USD MNF ve Units Trendi (2022-2024)",
                height=400,
                showlegend=True,
                template="plotly_white"
            )
            
            fig.update_yaxes(title_text="USD MNF", secondary_y=False)
            fig.update_yaxes(title_text="Units", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### 📊 Büyüme Zinciri")
            
            growth_data = pd.DataFrame({
                'Dönem': ['2022→2023', '2023→2024'],
                'Büyüme (%)': [growth_22_23, growth_23_24],
                'Mutlak Büyüme ($M)': [abs_22_23, abs_23_24]
            })
            
            fig = go.Figure(data=[
                go.Bar(name='Büyüme %', x=growth_data['Dönem'], y=growth_data['Büyüme (%)'],
                      marker_color=['#60A5FA', '#3B82F6']),
                go.Scatter(name='Mutlak Büyüme', x=growth_data['Dönem'], y=growth_data['Mutlak Büyüme ($M)'],
                          mode='lines+markers', line=dict(color='#EF4444', width=3),
                          yaxis='y2')
            ])
            
            fig.update_layout(
                title="Zincir Büyüme Analizi",
                height=400,
                yaxis=dict(title="Büyüme (%)"),
                yaxis2=dict(title="Mutlak Büyüme ($M)", overlaying='y', side='right'),
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Otomatik içgörüler
        st.markdown("---")
        st.markdown("##### 🧠 Otomatik İçgörüler")
        
        # Global içgörüler
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if growth_22_23 > 0 and growth_23_24 > 0:
                st.markdown('<div class="insight-card">✅ İki yıl üst üste pozitif büyüme</div>', unsafe_allow_html=True)
            elif growth_23_24 < 0:
                st.markdown(f'<div class="warning-card">⚠️ Son dönem küçülme: %{growth_23_24:.1f}</div>', unsafe_allow_html=True)
        
        with col2:
            avg_price_2024 = total_2024 / filtered_df['MAT Q3 2024\nUnits'].sum() if filtered_df['MAT Q3 2024\nUnits'].sum() > 0 else 0
            avg_price_2023 = total_2023 / filtered_df['MAT Q3 2023\nUnits'].sum() if filtered_df['MAT Q3 2023\nUnits'].sum() > 0 else 0
            
            price_change = ((avg_price_2024 - avg_price_2023) / avg_price_2023 * 100) if avg_price_2023 > 0 else 0
            
            if price_change > 5:
                st.markdown(f'<div class="insight-card">💰 Ortalama fiyat %{price_change:.1f} arttı</div>', unsafe_allow_html=True)
            elif price_change < -5:
                st.markdown(f'<div class="warning-card">📉 Ortalama fiyat %{abs(price_change):.1f} düştü</div>', unsafe_allow_html=True)
        
        with col3:
            country_concentration = (top_countries['MAT Q3 2024\nUSD MNF'].sum() / total_2024 * 100) if total_2024 > 0 else 0
            
            if country_concentration > 60:
                st.markdown(f'<div class="warning-card">🌍 Yüksek konsantrasyon: Top 3 ülke %{country_concentration:.1f} pay</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-card">🌐 Sağlıklı dağılım: Top 3 ülke %{country_concentration:.1f} pay</div>', unsafe_allow_html=True)
    
    # ============================================
    # TAB 2: Global Harita Analizi
    # ============================================
    with tab2:
        st.markdown('<h2 class="sub-header">🌍 Global Harita Analizi</h2>', unsafe_allow_html=True)
        
        map_tab1, map_tab2, map_tab3 = st.tabs([
            "🗺️ USD MNF Dağılımı",
            "📊 Global Pay (%)",
            "📈 Büyüme Analizi"
        ])
        
        with map_tab1:
            year_select = st.selectbox("Harita Yılı", [2024, 2023, 2022], key="map_usd_year")
            
            map_data = map_handler.prepare_map_data(filtered_df, year_select)
            
            if not map_data.empty and map_handler.world is not None:
                merged_data = map_handler.world.merge(map_data, how='left', left_on='iso_a3', right_on='ISO_A3')
                
                fig = px.choropleth(
                    merged_data,
                    geojson=merged_data.geometry,
                    locations=merged_data.index,
                    color=f'MAT Q3 {year_select}\nUSD MNF',
                    hover_name='name',
                    hover_data={
                        f'MAT Q3 {year_select}\nUSD MNF': ':.2f',
                        f'MAT Q3 {year_select}\nUnits': True,
                        f'MAT Q3 {year_select}\nStandard Units': True,
                        'Global_Pay_Pct': ':.2f%'
                    },
                    color_continuous_scale="Viridis",
                    labels={f'MAT Q3 {year_select}\nUSD MNF': f'USD MNF ({year_select})'},
                    title=f"Dünya Haritası - USD MNF Dağılımı ({year_select})"
                )
                
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Ülke sıralaması
                st.markdown("##### 🏆 Ülke Sıralaması")
                top_countries_map = map_data.nlargest(10, f'MAT Q3 {year_select}\nUSD MNF')
                
                fig_bar = go.Figure(data=[
                    go.Bar(x=top_countries_map['Country'],
                          y=top_countries_map[f'MAT Q3 {year_select}\nUSD MNF'],
                          marker_color='#3B82F6',
                          text=top_countries_map[f'MAT Q3 {year_select}\nUSD MNF'].apply(lambda x: f'${x:,.0f}M'),
                          textposition='auto')
                ])
                
                fig_bar.update_layout(
                    title=f"Top 10 Ülke - USD MNF ({year_select})",
                    height=400,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Harita verisi bulunamadı veya ülke kodları eşleştirilemedi.")
        
        with map_tab2:
            year_select_share = st.selectbox("Harita Yılı", [2024, 2023, 2022], key="map_share_year")
            
            share_data = map_handler.prepare_map_data(filtered_df, year_select_share)
            
            if not share_data.empty and map_handler.world is not None:
                merged_share = map_handler.world.merge(share_data, how='left', left_on='iso_a3', right_on='ISO_A3')
                
                fig = px.choropleth(
                    merged_share,
                    geojson=merged_share.geometry,
                    locations=merged_share.index,
                    color='Global_Pay_Pct',
                    hover_name='name',
                    hover_data={
                        f'MAT Q3 {year_select_share}\nUSD MNF': ':.2f',
                        'Global_Pay_Pct': ':.2f%'
                    },
                    color_continuous_scale="Plasma",
                    labels={'Global_Pay_Pct': 'Global Pay (%)'},
                    title=f"Dünya Haritası - Global Pay Dağılımı ({year_select_share})"
                )
                
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Pay değişimi analizi
                st.markdown("##### 📊 Global Pay Değişimi (2022 → 2024)")
                
                share_2022 = map_handler.prepare_map_data(filtered_df, 2022)
                share_2024 = map_handler.prepare_map_data(filtered_df, 2024)
                
                if not share_2022.empty and not share_2024.empty:
                    share_comparison = pd.merge(
                        share_2022[['Country', 'Global_Pay_Pct']],
                        share_2024[['Country', 'Global_Pay_Pct']],
                        on='Country',
                        suffixes=('_2022', '_2024')
                    )
                    
                    share_comparison['Change'] = share_comparison['Global_Pay_Pct_2024'] - share_comparison['Global_Pay_Pct_2022']
                    top_gainers = share_comparison.nlargest(5, 'Change')
                    top_losers = share_comparison.nsmallest(5, 'Change')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**⬆️ En Çok Kazananlar**")
                        for _, row in top_gainers.iterrows():
                            st.write(f"{row['Country']}: +{row['Change']:.2f}%")
                    
                    with col2:
                        st.markdown("**⬇️ En Çok Kaybedenler**")
                        for _, row in top_losers.iterrows():
                            st.write(f"{row['Country']}: {row['Change']:.2f}%")
        
        with map_tab3:
            growth_type = st.radio("Büyüme Dönemi", ["2022 → 2023", "2023 → 2024"], horizontal=True)
            
            if growth_type == "2022 → 2023":
                start_year, end_year = 2022, 2023
            else:
                start_year, end_year = 2023, 2024
            
            # Ülke bazlı büyüme hesapla
            country_growth = []
            
            for country in filtered_df['Country'].unique():
                country_df = filtered_df[filtered_df['Country'] == country]
                growth_pct, _ = analytics.calculate_growth(country_df, start_year, end_year)
                country_growth.append({
                    'Country': country,
                    'Growth_Pct': growth_pct
                })
            
            growth_df = pd.DataFrame(country_growth)
            
            if not growth_df.empty and map_handler.world is not None:
                growth_df['ISO_A3'] = growth_df['Country'].apply(map_handler.get_country_code)
                growth_df = growth_df.dropna(subset=['ISO_A3'])
                
                merged_growth = map_handler.world.merge(growth_df, how='left', left_on='iso_a3', right_on='ISO_A3')
                
                # Büyüme kategorileri
                def growth_category(x):
                    if pd.isna(x):
                        return 'Veri Yok'
                    elif x > 20:
                        return 'Çok Yüksek Büyüme (>20%)'
                    elif x > 10:
                        return 'Yüksek Büyüme (10-20%)'
                    elif x > 0:
                        return 'Orta Büyüme (0-10%)'
                    elif x > -10:
                        return 'Hafif Düşüş (0- -10%)'
                    else:
                        return 'Keskin Düşüş (<-10%)'
                
                merged_growth['Growth_Category'] = merged_growth['Growth_Pct'].apply(growth_category)
                
                color_discrete_map = {
                    'Çok Yüksek Büyüme (>20%)': '#10B981',
                    'Yüksek Büyüme (10-20%)': '#34D399',
                    'Orta Büyüme (0-10%)': '#60A5FA',
                    'Hafif Düşüş (0- -10%)': '#FBBF24',
                    'Keskin Düşüş (<-10%)': '#DC2626',
                    'Veri Yok': '#9CA3AF'
                }
                
                fig = px.choropleth(
                    merged_growth,
                    geojson=merged_growth.geometry,
                    locations=merged_growth.index,
                    color='Growth_Category',
                    hover_name='name',
                    hover_data={
                        'Growth_Pct': ':.1f%',
                        'Country': True
                    },
                    color_discrete_map=color_discrete_map,
                    category_orders={
                        'Growth_Category': [
                            'Çok Yüksek Büyüme (>20%)',
                            'Yüksek Büyüme (10-20%)',
                            'Orta Büyüme (0-10%)',
                            'Hafif Düşüş (0- -10%)',
                            'Keskin Düşüş (<-10%)',
                            'Veri Yok'
                        ]
                    },
                    title=f"Dünya Haritası - Büyüme Oranları ({start_year} → {end_year})"
                )
                
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Büyüme istatistikleri
                st.markdown("##### 📈 Büyüme Dağılımı")
                
                growth_stats = growth_df['Growth_Pct'].describe()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Ortalama Büyüme", f"{growth_stats['mean']:.1f}%")
                with col2:
                    st.metric("Medyan Büyüme", f"{growth_stats['50%']:.1f}%")
                with col3:
                    st.metric("Maksimum", f"{growth_stats['max']:.1f}%")
                with col4:
                    st.metric("Minimum", f"{growth_stats['min']:.1f}%")
    
    # ============================================
    # TAB 3: Ülke Derinlemesine
    # ============================================
    with tab3:
        st.markdown('<h2 class="sub-header">🇹🇷 Ülke Derinlemesine Analiz</h2>', unsafe_allow_html=True)
        
        available_countries = sorted(filtered_df['Country'].unique())
        selected_country = st.selectbox("Analiz Edilecek Ülke Seçin", available_countries)
        
        if selected_country:
            country_df = filtered_df[filtered_df['Country'] == selected_country]
            
            if len(country_df) > 0:
                # Özet metrikler
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    usd_2024 = country_df['MAT Q3 2024\nUSD MNF'].sum()
                    st.metric("2024 USD MNF", f"${usd_2024:,.0f}M")
                
                with col2:
                    global_share = (usd_2024 / filtered_df['MAT Q3 2024\nUSD MNF'].sum() * 100) if filtered_df['MAT Q3 2024\nUSD MNF'].sum() > 0 else 0
                    st.metric("Global Pay", f"{global_share:.2f}%")
                
                with col3:
                    growth_22_23, _ = analytics.calculate_growth(country_df, 2022, 2023)
                    st.metric("2022→2023 Büyüme", f"{growth_22_23:.1f}%")
                
                with col4:
                    growth_23_24, _ = analytics.calculate_growth(country_df, 2023, 2024)
                    st.metric("2023→2024 Büyüme", f"{growth_23_24:.1f}%")
                
                st.markdown("---")
                
                # Trend analizi
                st.markdown("##### 📈 Üç Yıllık Trend")
                
                trend_country = pd.DataFrame({
                    'Yıl': [2022, 2023, 2024],
                    'USD MNF': [
                        country_df['MAT Q3 2022\nUSD MNF'].sum(),
                        country_df['MAT Q3 2023\nUSD MNF'].sum(),
                        country_df['MAT Q3 2024\nUSD MNF'].sum()
                    ],
                    'Units': [
                        country_df['MAT Q3 2022\nUnits'].sum(),
                        country_df['MAT Q3 2023\nUnits'].sum(),
                        country_df['MAT Q3 2024\nUnits'].sum()
                    ],
                    'Standard Units': [
                        country_df['MAT Q3 2022\nStandard Units'].sum(),
                        country_df['MAT Q3 2023\nStandard Units'].sum(),
                        country_df['MAT Q3 2024\nStandard Units'].sum()
                    ]
                })
                
                fig = make_subplots(rows=2, cols=2, subplot_titles=("USD MNF Trendi", "Units Trendi", "Standard Units Trendi", "Büyüme Zinciri"))
                
                # USD MNF
                fig.add_trace(
                    go.Bar(x=trend_country['Yıl'], y=trend_country['USD MNF'],
                          name='USD MNF', marker_color='#3B82F6'),
                    row=1, col=1
                )
                
                # Units
                fig.add_trace(
                    go.Bar(x=trend_country['Yıl'], y=trend_country['Units'],
                          name='Units', marker_color='#10B981'),
                    row=1, col=2
                )
                
                # Standard Units
                fig.add_trace(
                    go.Bar(x=trend_country['Yıl'], y=trend_country['Standard Units'],
                          name='Standard Units', marker_color='#F59E0B'),
                    row=2, col=1
                )
                
                # Büyüme zinciri
                growth_values = [
                    ((trend_country.loc[1, 'USD MNF'] - trend_country.loc[0, 'USD MNF']) / trend_country.loc[0, 'USD MNF'] * 100) if trend_country.loc[0, 'USD MNF'] > 0 else 0,
                    ((trend_country.loc[2, 'USD MNF'] - trend_country.loc[1, 'USD MNF']) / trend_country.loc[1, 'USD MNF'] * 100) if trend_country.loc[1, 'USD MNF'] > 0 else 0
                ]
                
                fig.add_trace(
                    go.Scatter(x=['2022→2023', '2023→2024'], y=growth_values,
                              mode='lines+markers', name='Büyüme %',
                              line=dict(color='#EF4444', width=3)),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                # Price-Volume-Mix analizi
                st.markdown("---")
                st.markdown("##### 📊 Price-Volume-Mix Ayrıştırması")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**2022 → 2023**")
                    pvm_22_23 = analytics.price_volume_mix_analysis(country_df, 2022, 2023)
                    
                    if pvm_22_23:
                        fig_pvm1 = go.Figure(data=[
                            go.Bar(name='Fiyat Etkisi', x=['Fiyat'], y=[pvm_22_23['price_effect_pct']],
                                  marker_color='#3B82F6'),
                            go.Bar(name='Volume Etkisi', x=['Volume'], y=[pvm_22_23['volume_effect_pct']],
                                  marker_color='#10B981'),
                            go.Bar(name='Mix Etkisi', x=['Mix'], y=[pvm_22_23['mix_effect_pct']],
                                  marker_color='#F59E0B')
                        ])
                        
                        fig_pvm1.update_layout(
                            title="Ayrıştırma Katkıları (%)",
                            yaxis_title="Katkı (%)",
                            barmode='group',
                            height=400
                        )
                        
                        st.plotly_chart(fig_pvm1, use_container_width=True)
                        
                        # Detaylı bilgiler
                        st.write(f"**Toplam Büyüme:** %{((pvm_22_23['total_growth'] / country_df['MAT Q3 2022\nUSD MNF'].sum()) * 100) if country_df['MAT Q3 2022\nUSD MNF'].sum() > 0 else 0:.1f}")
                        st.write(f"**Ortalama Fiyat Değişimi:** %{((pvm_22_23['weighted_price_end'] - pvm_22_23['weighted_price_start']) / pvm_22_23['weighted_price_start'] * 100) if pvm_22_23['weighted_price_start'] > 0 else 0:.1f}")
                        st.write(f"**Volume Büyümesi:** %{pvm_22_23['unit_growth_pct']:.1f}")
                
                with col2:
                    st.markdown("**2023 → 2024**")
                    pvm_23_24 = analytics.price_volume_mix_analysis(country_df, 2023, 2024)
                    
                    if pvm_23_24:
                        fig_pvm2 = go.Figure(data=[
                            go.Bar(name='Fiyat Etkisi', x=['Fiyat'], y=[pvm_23_24['price_effect_pct']],
                                  marker_color='#3B82F6'),
                            go.Bar(name='Volume Etkisi', x=['Volume'], y=[pvm_23_24['volume_effect_pct']],
                                  marker_color='#10B981'),
                            go.Bar(name='Mix Etkisi', x=['Mix'], y=[pvm_23_24['mix_effect_pct']],
                                  marker_color='#F59E0B')
                        ])
                        
                        fig_pvm2.update_layout(
                            title="Ayrıştırma Katkıları (%)",
                            yaxis_title="Katkı (%)",
                            barmode='group',
                            height=400
                        )
                        
                        st.plotly_chart(fig_pvm2, use_container_width=True)
                        
                        # Detaylı bilgiler
                        st.write(f"**Toplam Büyüme:** %{((pvm_23_24['total_growth'] / country_df['MAT Q3 2023\nUSD MNF'].sum()) * 100) if country_df['MAT Q3 2023\nUSD MNF'].sum() > 0 else 0:.1f}")
                        st.write(f"**Ortalama Fiyat Değişimi:** %{((pvm_23_24['weighted_price_end'] - pvm_23_24['weighted_price_start']) / pvm_23_24['weighted_price_start'] * 100) if pvm_23_24['weighted_price_start'] > 0 else 0:.1f}")
                        st.write(f"**Volume Büyümesi:** %{pvm_23_24['unit_growth_pct']:.1f}")
                
                # Molekül analizi
                st.markdown("---")
                st.markdown("##### 💊 Molekül Bazlı Analiz")
                
                mol_analysis = country_df.groupby('Molecule').agg({
                    'MAT Q3 2024\nUSD MNF': 'sum',
                    'MAT Q3 2023\nUSD MNF': 'sum',
                    'MAT Q3 2022\nUSD MNF': 'sum'
                }).reset_index()
                
                mol_analysis['Share_2024'] = (mol_analysis['MAT Q3 2024\nUSD MNF'] / mol_analysis['MAT Q3 2024\nUSD MNF'].sum() * 100)
                mol_analysis = mol_analysis.sort_values('Share_2024', ascending=False).head(10)
                
                fig_mol = go.Figure()
                
                fig_mol.add_trace(go.Bar(
                    x=mol_analysis['Molecule'],
                    y=mol_analysis['MAT Q3 2024\nUSD MNF'],
                    name='2024',
                    marker_color='#3B82F6'
                ))
                
                fig_mol.add_trace(go.Bar(
                    x=mol_analysis['Molecule'],
                    y=mol_analysis['MAT Q3 2023\nUSD MNF'],
                    name='2023',
                    marker_color='#60A5FA'
                ))
                
                fig_mol.add_trace(go.Bar(
                    x=mol_analysis['Molecule'],
                    y=mol_analysis['MAT Q3 2022\nUSD MNF'],
                    name='2022',
                    marker_color='#93C5FD'
                ))
                
                fig_mol.update_layout(
                    title="Top 10 Molekül - Üç Yıllık Karşılaştırma",
                    barmode='group',
                    height=500,
                    xaxis_tickangle=-45,
                    yaxis_title="USD MNF"
                )
                
                st.plotly_chart(fig_mol, use_container_width=True)
                
                # Üretici analizi
                st.markdown("##### 🏭 Üretici Dağılımı")
                
                mfg_analysis = country_df.groupby('Manufacturer').agg({
                    'MAT Q3 2024\nUSD MNF': 'sum'
                }).reset_index()
                
                mfg_analysis['Share'] = (mfg_analysis['MAT Q3 2024\nUSD MNF'] / mfg_analysis['MAT Q3 2024\nUSD MNF'].sum() * 100)
                mfg_analysis = mfg_analysis.sort_values('Share', ascending=False)
                
                fig_mfg = px.pie(
                    mfg_analysis.head(8),
                    values='MAT Q3 2024\nUSD MNF',
                    names='Manufacturer',
                    title="Üretici Pay Dağılımı (2024)",
                    hole=0.4
                )
                
                fig_mfg.update_traces(textposition='inside', textinfo='percent+label')
                fig_mfg.update_layout(height=500)
                
                st.plotly_chart(fig_mfg, use_container_width=True)
    
    # ============================================
    # TAB 4: Molekül & Ürün
    # ============================================
    with tab4:
        st.markdown('<h2 class="sub-header">💊 Molekül & Ürün Analizi</h2>', unsafe_allow_html=True)
        
        mol_tab1, mol_tab2, mol_tab3 = st.tabs([
            "📊 Molekül Performansı",
            "🌍 Coğrafi Dağılım",
            "📈 Trend Analizi"
        ])
        
        with mol_tab1:
            # Molekül seçimi
            available_molecules = sorted(filtered_df['Molecule'].unique())
            selected_molecules = st.multiselect(
                "Analiz Edilecek Moleküller (Çoklu Seçim)",
                options=available_molecules,
                default=available_molecules[:3] if len(available_molecules) >= 3 else available_molecules
            )
            
            if selected_molecules:
                # Performans özeti
                st.markdown("##### 📈 Performans Özeti")
                
                perf_data = []
                
                for molecule in selected_molecules:
                    mol_df = filtered_df[filtered_df['Molecule'] == molecule]
                    
                    if len(mol_df) > 0:
                        usd_2022 = mol_df['MAT Q3 2022\nUSD MNF'].sum()
                        usd_2023 = mol_df['MAT Q3 2023\nUSD MNF'].sum()
                        usd_2024 = mol_df['MAT Q3 2024\nUSD MNF'].sum()
                        
                        growth_22_23 = ((usd_2023 - usd_2022) / usd_2022 * 100) if usd_2022 > 0 else 0
                        growth_23_24 = ((usd_2024 - usd_2023) / usd_2023 * 100) if usd_2023 > 0 else 0
                        
                        global_share_2024 = (usd_2024 / filtered_df['MAT Q3 2024\nUSD MNF'].sum() * 100) if filtered_df['MAT Q3 2024\nUSD MNF'].sum() > 0 else 0
                        
                        perf_data.append({
                            'Molecule': molecule,
                            'USD_2022': usd_2022,
                            'USD_2023': usd_2023,
                            'USD_2024': usd_2024,
                            'Growth_22_23': growth_22_23,
                            'Growth_23_24': growth_23_24,
                            'Global_Share_2024': global_share_2024
                        })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    
                    # Metrikler
                    cols = st.columns(len(selected_molecules))
                    
                    for idx, molecule in enumerate(selected_molecules):
                        with cols[idx]:
                            mol_row = perf_df[perf_df['Molecule'] == molecule].iloc[0]
                            st.metric(
                                molecule,
                                f"${mol_row['USD_2024']:,.0f}M",
                                f"{mol_row['Growth_23_24']:.1f}%"
                            )
                    
                    # Detaylı tablo
                    st.markdown("##### 📋 Detaylı Performans Tablosu")
                    
                    display_df = perf_df.copy()
                    display_df['USD_2022'] = display_df['USD_2022'].apply(lambda x: f"${x:,.0f}M")
                    display_df['USD_2023'] = display_df['USD_2023'].apply(lambda x: f"${x:,.0f}M")
                    display_df['USD_2024'] = display_df['USD_2024'].apply(lambda x: f"${x:,.0f}M")
                    display_df['Growth_22_23'] = display_df['Growth_22_23'].apply(lambda x: f"{x:.1f}%")
                    display_df['Growth_23_24'] = display_df['Growth_23_24'].apply(lambda x: f"{x:.1f}%")
                    display_df['Global_Share_2024'] = display_df['Global_Share_2024'].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Trend grafiği
                    st.markdown("##### 📊 Üç Yıllık Trend")
                    
                    trend_mol = pd.melt(
                        perf_df[['Molecule', 'USD_2022', 'USD_2023', 'USD_2024']],
                        id_vars=['Molecule'],
                        value_vars=['USD_2022', 'USD_2023', 'USD_2024'],
                        var_name='Year',
                        value_name='USD_MNF'
                    )
                    
                    trend_mol['Year'] = trend_mol['Year'].str.replace('USD_', '')
                    
                    fig = px.line(
                        trend_mol,
                        x='Year',
                        y='USD_MNF',
                        color='Molecule',
                        markers=True,
                        title="Moleküller - Üç Yıllık Trend"
                    )
                    
                    fig.update_layout(height=500, yaxis_title="USD MNF")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Büyüme karşılaştırması
                    st.markdown("##### ⚡ Büyüme Karşılaştırması")
                    
                    growth_comparison = perf_df[['Molecule', 'Growth_22_23', 'Growth_23_24']].copy()
                    growth_comparison['Growth_22_24'] = growth_comparison.apply(
                        lambda row: ((row['Growth_22_23']/100 + 1) * (row['Growth_23_24']/100 + 1) - 1) * 100,
                        axis=1
                    )
                    
                    fig_growth = go.Figure()
                    
                    for molecule in growth_comparison['Molecule']:
                        row = growth_comparison[growth_comparison['Molecule'] == molecule].iloc[0]
                        fig_growth.add_trace(go.Scatter(
                            x=['2022→2023', '2023→2024', '2022→2024'],
                            y=[row['Growth_22_23'], row['Growth_23_24'], row['Growth_22_24']],
                            mode='lines+markers',
                            name=molecule
                        ))
                    
                    fig_growth.update_layout(
                        title="Büyüme Oranları Karşılaştırması",
                        height=500,
                        yaxis_title="Büyüme (%)",
                        xaxis_title="Dönem"
                    )
                    
                    st.plotly_chart(fig_growth, use_container_width=True)
        
        with mol_tab2:
            # Molekül seçimi
            mol_for_map = st.selectbox(
                "Haritalandırılacak Molekül",
                options=sorted(filtered_df['Molecule'].unique())
            )
            
            if mol_for_map:
                mol_map_df = filtered_df[filtered_df['Molecule'] == mol_for_map]
                
                if not mol_map_df.empty:
                    # Coğrafi dağılım
                    country_dist = mol_map_df.groupby('Country').agg({
                        'MAT Q3 2024\nUSD MNF': 'sum',
                        'MAT Q3 2023\nUSD MNF': 'sum',
                        'MAT Q3 2022\nUSD MNF': 'sum'
                    }).reset_index()
                    
                    total_mol_2024 = country_dist['MAT Q3 2024\nUSD MNF'].sum()
                    country_dist['Share_2024'] = (country_dist['MAT Q3 2024\nUSD MNF'] / total_mol_2024 * 100) if total_mol_2024 > 0 else 0
                    
                    # Harita hazırlığı
                    country_dist['ISO_A3'] = country_dist['Country'].apply(map_handler.get_country_code)
                    country_dist = country_dist.dropna(subset=['ISO_A3'])
                    
                    if not country_dist.empty and map_handler.world is not None:
                        merged_mol = map_handler.world.merge(country_dist, how='left', left_on='iso_a3', right_on='ISO_A3')
                        
                        fig = px.choropleth(
                            merged_mol,
                            geojson=merged_mol.geometry,
                            locations=merged_mol.index,
                            color='MAT Q3 2024\nUSD MNF',
                            hover_name='name',
                            hover_data={
                                'MAT Q3 2024\nUSD MNF': ':.2f',
                                'Share_2024': ':.2f%',
                                'MAT Q3 2023\nUSD MNF': ':.2f',
                                'MAT Q3 2022\nUSD MNF': ':.2f'
                            },
                            color_continuous_scale="Viridis",
                            title=f"{mol_for_map} - Coğrafi Dağılım (2024)"
                        )
                        
                        fig.update_geos(fitbounds="locations", visible=False)
                        fig.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Ülke sıralaması
                    st.markdown("##### 🏆 Ülke Sıralaması")
                    
                    top_countries_mol = country_dist.nlargest(10, 'MAT Q3 2024\nUSD MNF')
                    
                    fig_bar = go.Figure()
                    
                    fig_bar.add_trace(go.Bar(
                        x=top_countries_mol['Country'],
                        y=top_countries_mol['MAT Q3 2024\nUSD MNF'],
                        name='2024',
                        marker_color='#3B82F6',
                        text=top_countries_mol['Share_2024'].apply(lambda x: f'{x:.1f}%'),
                        textposition='auto'
                    ))
                    
                    fig_bar.add_trace(go.Bar(
                        x=top_countries_mol['Country'],
                        y=top_countries_mol['MAT Q3 2023\nUSD MNF'],
                        name='2023',
                        marker_color='#60A5FA'
                    ))
                    
                    fig_bar.update_layout(
                        title=f"{mol_for_map} - Top 10 Ülke",
                        barmode='group',
                        height=500,
                        xaxis_tickangle=-45,
                        yaxis_title="USD MNF"
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Konsantrasyon analizi
                    st.markdown("##### 🎯 Pazar Konsantrasyonu")
                    
                    top5_share = top_countries_mol.head(5)['MAT Q3 2024\nUSD MNF'].sum() / total_mol_2024 * 100 if total_mol_2024 > 0 else 0
                    top3_share = top_countries_mol.head(3)['MAT Q3 2024\nUSD MNF'].sum() / total_mol_2024 * 100 if total_mol_2024 > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Top 5 Ülke Payı", f"{top5_share:.1f}%")
                    
                    with col2:
                        st.metric("Top 3 Ülke Payı", f"{top3_share:.1f}%")
                    
                    with col3:
                        country_count = len(country_dist)
                        st.metric("Toplam Ülke Sayısı", country_count)
        
        with mol_tab3:
            # Trend analizi için molekül seçimi
            trend_molecules = st.multiselect(
                "Trend Analizi için Moleküller",
                options=sorted(filtered_df['Molecule'].unique()),
                default=sorted(filtered_df['Molecule'].unique())[:5] if len(filtered_df['Molecule'].unique()) >= 5 else sorted(filtered_df['Molecule'].unique())
            )
            
            if trend_molecules:
                # Trend verisi hazırlama
                trend_data_all = []
                
                for molecule in trend_molecules:
                    mol_trend_df = filtered_df[filtered_df['Molecule'] == molecule]
                    
                    if len(mol_trend_df) > 0:
                        for year in [2022, 2023, 2024]:
                            if year == 2022:
                                usd_col = 'MAT Q3 2022\nUSD MNF'
                            elif year == 2023:
                                usd_col = 'MAT Q3 2023\nUSD MNF'
                            else:
                                usd_col = 'MAT Q3 2024\nUSD MNF'
                            
                            total_usd = mol_trend_df[usd_col].sum()
                            global_share = (total_usd / filtered_df[usd_col].sum() * 100) if filtered_df[usd_col].sum() > 0 else 0
                            
                            trend_data_all.append({
                                'Molecule': molecule,
                                'Year': year,
                                'USD_MNF': total_usd,
                                'Global_Share': global_share
                            })
                
                if trend_data_all:
                    trend_df = pd.DataFrame(trend_data_all)
                    
                    # USD MNF trendi
                    st.markdown("##### 📈 USD MNF Trendi")
                    
                    fig_trend = px.line(
                        trend_df,
                        x='Year',
                        y='USD_MNF',
                        color='Molecule',
                        markers=True,
                        title="Moleküller - USD MNF Trendi (2022-2024)"
                    )
                    
                    fig_trend.update_layout(height=500, yaxis_title="USD MNF")
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Global pay trendi
                    st.markdown("##### 📊 Global Pay Trendi")
                    
                    fig_share = px.line(
                        trend_df,
                        x='Year',
                        y='Global_Share',
                        color='Molecule',
                        markers=True,
                        title="Moleküller - Global Pay Trendi (2022-2024)"
                    )
                    
                    fig_share.update_layout(height=500, yaxis_title="Global Pay (%)")
                    st.plotly_chart(fig_share, use_container_width=True)
                    
                    # Büyüme matrisi
                    st.markdown("##### ⚡ Büyüme Matrisi")
                    
                    growth_matrix = []
                    
                    for molecule in trend_molecules:
                        mol_data = trend_df[trend_df['Molecule'] == molecule]
                        
                        if len(mol_data) == 3:
                            usd_2022 = mol_data[mol_data['Year'] == 2022]['USD_MNF'].values[0]
                            usd_2023 = mol_data[mol_data['Year'] == 2023]['USD_MNF'].values[0]
                            usd_2024 = mol_data[mol_data['Year'] == 2024]['USD_MNF'].values[0]
                            
                            growth_22_23 = ((usd_2023 - usd_2022) / usd_2022 * 100) if usd_2022 > 0 else 0
                            growth_23_24 = ((usd_2024 - usd_2023) / usd_2023 * 100) if usd_2023 > 0 else 0
                            
                            growth_matrix.append({
                                'Molecule': molecule,
                                'USD_2022': usd_2022,
                                'USD_2023': usd_2023,
                                'USD_2024': usd_2024,
                                'Growth_22_23': growth_22_23,
                                'Growth_23_24': growth_23_24,
                                'Growth_22_24': ((usd_2024 - usd_2022) / usd_2022 * 100) if usd_2022 > 0 else 0
                            })
                    
                    if growth_matrix:
                        growth_df = pd.DataFrame(growth_matrix)
                        
                        # Heatmap için veri hazırlama
                        heatmap_data = growth_df[['Molecule', 'Growth_22_23', 'Growth_23_24']].copy()
                        heatmap_data = heatmap_data.set_index('Molecule')
                        
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=heatmap_data.values,
                            x=['2022→2023', '2023→2024'],
                            y=heatmap_data.index,
                            colorscale='RdYlGn',
                            zmid=0,
                            text=[[f'{val:.1f}%' for val in row] for row in heatmap_data.values],
                            texttemplate='%{text}',
                            textfont={"size": 12}
                        ))
                        
                        fig_heatmap.update_layout(
                            title="Büyüme Oranları Heatmap",
                            height=400,
                            xaxis_title="Dönem",
                            yaxis_title="Molekül"
                        )
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # ============================================
    # TAB 5: Corporation & Rekabet
    # ============================================
    with tab5:
        st.markdown('<h2 class="sub-header">🏢 Corporation & Rekabet Analizi</h2>', unsafe_allow_html=True)
        
        corp_tab1, corp_tab2, corp_tab3 = st.tabs([
            "📊 Pazar Payı Analizi",
            "📈 Pay Değişimi",
            "🌍 Coğrafi Varlık"
        ])
        
        with corp_tab1:
            # Pazar payı analizi
            year_for_share = st.selectbox("Pazar Payı Yılı", [2024, 2023, 2022], key="corp_share_year")
            
            share_df = analytics.calculate_market_share(filtered_df, year_for_share, 'Corporation')
            
            if not share_df.empty:
                # Top 10 corporation
                top_corps = share_df.head(10)
                
                st.markdown(f"##### 🏆 Top 10 Corporation - {year_for_share}")
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=top_corps['Corporation'],
                    y=top_corps['Market_Share_Pct'],
                    marker_color='#3B82F6',
                    text=top_corps['Market_Share_Pct'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f"Pazar Payı Dağılımı ({year_for_share})",
                    height=500,
                    xaxis_tickangle=-45,
                    yaxis_title="Pazar Payı (%)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cumulative share
                st.markdown("##### 📈 Kümülatif Pazar Payı")
                
                fig_cum = go.Figure()
                
                fig_cum.add_trace(go.Scatter(
                    x=top_corps['Corporation'],
                    y=top_corps['Cumulative_Share'],
                    mode='lines+markers',
                    line=dict(color='#10B981', width=3),
                    marker=dict(size=10),
                    name='Kümülatif Pay'
                ))
                
                fig_cum.add_trace(go.Bar(
                    x=top_corps['Corporation'],
                    y=top_corps['Market_Share_Pct'],
                    name='Bireysel Pay',
                    marker_color='rgba(59, 130, 246, 0.5)'
                ))
                
                fig_cum.update_layout(
                    title="Kümülatif Pazar Payı",
                    height=500,
                    xaxis_tickangle=-45,
                    yaxis_title="Pay (%)",
                    barmode='overlay'
                )
                
                st.plotly_chart(fig_cum, use_container_width=True)
                
                # Pazar konsantrasyonu
                st.markdown("##### 🎯 Pazar Konsantrasyonu")
                
                top3_share = top_corps.head(3)['Market_Share_Pct'].sum()
                top5_share = top_corps.head(5)['Market_Share_Pct'].sum()
                top10_share = top_corps['Market_Share_Pct'].sum()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Top 3 Payı", f"{top3_share:.1f}%")
                
                with col2:
                    st.metric("Top 5 Payı", f"{top5_share:.1f}%")
                
                with col3:
                    st.metric("Top 10 Payı", f"{top10_share:.1f}%")
                
                # Herfindahl-Hirschman Index (HHI)
                hhi = (share_df['Market_Share_Pct'] ** 2).sum()
                
                if hhi < 1500:
                    concentration_level = "Düşük Konsantrasyon"
                    concentration_color = "#10B981"
                elif hhi < 2500:
                    concentration_level = "Orta Konsantrasyon"
                    concentration_color = "#F59E0B"
                else:
                    concentration_level = "Yüksek Konsantrasyon"
                    concentration_color = "#DC2626"
                
                st.markdown(f'''
                <div class="metric-card">
                    <h3>📊 Herfindahl-Hirschman Index (HHI)</h3>
                    <p style="font-size: 2rem; font-weight: 800; color: {concentration_color};">{hhi:.0f}</p>
                    <p>{concentration_level}</p>
                    <p style="font-size: 0.9rem; color: #6B7280;">
                        HHI < 1,500: Düşük konsantrasyon<br>
                        HHI 1,500-2,500: Orta konsantrasyon<br>
                        HHI > 2,500: Yüksek konsantrasyon
                    </p>
                </div>
                ''', unsafe_allow_html=True)
        
        with corp_tab2:
            # Pay değişimi analizi
            st.markdown("##### 📈 Pazar Payı Değişimi (2022 → 2024)")
            
            share_2022 = analytics.calculate_market_share(filtered_df, 2022, 'Corporation')
            share_2023 = analytics.calculate_market_share(filtered_df, 2023, 'Corporation')
            share_2024 = analytics.calculate_market_share(filtered_df, 2024, 'Corporation')
            
            if not share_2022.empty and not share_2023.empty and not share_2024.empty:
                # Pay değişimi tablosu
                share_comparison = pd.merge(
                    share_2022[['Corporation', 'Market_Share_Pct']],
                    share_2023[['Corporation', 'Market_Share_Pct']],
                    on='Corporation',
                    suffixes=('_2022', '_2023')
                )
                
                share_comparison = pd.merge(
                    share_comparison,
                    share_2024[['Corporation', 'Market_Share_Pct']],
                    on='Corporation'
                )
                
                share_comparison = share_comparison.rename(columns={'Market_Share_Pct': 'Market_Share_Pct_2024'})
                
                share_comparison['Change_22_23'] = share_comparison['Market_Share_Pct_2023'] - share_comparison['Market_Share_Pct_2022']
                share_comparison['Change_23_24'] = share_comparison['Market_Share_Pct_2024'] - share_comparison['Market_Share_Pct_2023']
                share_comparison['Change_22_24'] = share_comparison['Market_Share_Pct_2024'] - share_comparison['Market_Share_Pct_2022']
                
                # Top 20 corporation filtrele
                top_corps_all = share_comparison.nlargest(20, 'Market_Share_Pct_2024')
                
                # Pay trend grafiği
                trend_corp_data = []
                
                for _, row in top_corps_all.iterrows():
                    trend_corp_data.append({
                        'Corporation': row['Corporation'],
                        'Year': 2022,
                        'Market_Share': row['Market_Share_Pct_2022']
                    })
                    trend_corp_data.append({
                        'Corporation': row['Corporation'],
                        'Year': 2023,
                        'Market_Share': row['Market_Share_Pct_2023']
                    })
                    trend_corp_data.append({
                        'Corporation': row['Corporation'],
                        'Year': 2024,
                        'Market_Share': row['Market_Share_Pct_2024']
                    })
                
                trend_corp_df = pd.DataFrame(trend_corp_data)
                
                fig_trend = px.line(
                    trend_corp_df,
                    x='Year',
                    y='Market_Share',
                    color='Corporation',
                    markers=True,
                    title="Top 20 Corporation - Pazar Payı Trendi"
                )
                
                fig_trend.update_layout(height=600, yaxis_title="Pazar Payı (%)")
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # En çok kazanan ve kaybedenler
                st.markdown("##### 🏆 En Çok Kazananlar & Kaybedenler")
                
                top_gainers_corp = share_comparison.nlargest(5, 'Change_22_24')
                top_losers_corp = share_comparison.nsmallest(5, 'Change_22_24')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**⬆️ En Çok Kazananlar (2022→2024)**")
                    
                    for _, row in top_gainers_corp.iterrows():
                        st.write(f"**{row['Corporation']}**: +{row['Change_22_24']:.2f}%")
                        st.progress(min(row['Change_22_24'] / 10, 1.0))
                
                with col2:
                    st.markdown("**⬇️ En Çok Kaybedenler (2022→2024)**")
                    
                    for _, row in top_losers_corp.iterrows():
                        st.write(f"**{row['Corporation']}**: {row['Change_22_24']:.2f}%")
                        st.progress(min(abs(row['Change_22_24']) / 10, 1.0))
                
                # Pay değişimi heatmap
                st.markdown("##### 🔥 Pay Değişimi Heatmap")
                
                heatmap_corp_data = top_corps_all[['Corporation', 'Change_22_23', 'Change_23_24']].copy()
                heatmap_corp_data = heatmap_corp_data.set_index('Corporation')
                
                fig_heatmap_corp = go.Figure(data=go.Heatmap(
                    z=heatmap_corp_data.values,
                    x=['2022→2023', '2023→2024'],
                    y=heatmap_corp_data.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=[[f'{val:+.2f}%' for val in row] for row in heatmap_corp_data.values],
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig_heatmap_corp.update_layout(
                    title="Pazar Payı Değişimi Heatmap",
                    height=500,
                    xaxis_title="Dönem",
                    yaxis_title="Corporation"
                )
                
                st.plotly_chart(fig_heatmap_corp, use_container_width=True)
        
        with corp_tab3:
            # Corporation seçimi
            selected_corp = st.selectbox(
                "Analiz Edilecek Corporation",
                options=sorted(filtered_df['Corporation'].unique())
            )
            
            if selected_corp:
                corp_df = filtered_df[filtered_df['Corporation'] == selected_corp]
                
                if len(corp_df) > 0:
                    # Coğrafi dağılım
                    country_dist_corp = corp_df.groupby('Country').agg({
                        'MAT Q3 2024\nUSD MNF': 'sum',
                        'MAT Q3 2023\nUSD MNF': 'sum',
                        'MAT Q3 2022\nUSD MNF': 'sum'
                    }).reset_index()
                    
                    total_corp_2024 = country_dist_corp['MAT Q3 2024\nUSD MNF'].sum()
                    country_dist_corp['Share_2024'] = (country_dist_corp['MAT Q3 2024\nUSD MNF'] / total_corp_2024 * 100) if total_corp_2024 > 0 else 0
                    
                    # Harita
                    country_dist_corp['ISO_A3'] = country_dist_corp['Country'].apply(map_handler.get_country_code)
                    country_dist_corp = country_dist_corp.dropna(subset=['ISO_A3'])
                    
                    if not country_dist_corp.empty and map_handler.world is not None:
                        merged_corp = map_handler.world.merge(country_dist_corp, how='left', left_on='iso_a3', right_on='ISO_A3')
                        
                        fig = px.choropleth(
                            merged_corp,
                            geojson=merged_corp.geometry,
                            locations=merged_corp.index,
                            color='MAT Q3 2024\nUSD MNF',
                            hover_name='name',
                            hover_data={
                                'MAT Q3 2024\nUSD MNF': ':.2f',
                                'Share_2024': ':.2f%',
                                'MAT Q3 2023\nUSD MNF': ':.2f'
                            },
                            color_continuous_scale="Viridis",
                            title=f"{selected_corp} - Coğrafi Dağılım (2024)"
                        )
                        
                        fig.update_geos(fitbounds="locations", visible=False)
                        fig.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Ülke sıralaması
                    st.markdown("##### 🌍 Ülke Dağılımı")
                    
                    top_countries_corp = country_dist_corp.nlargest(10, 'MAT Q3 2024\nUSD MNF')
                    
                    fig_bar_corp = go.Figure()
                    
                    fig_bar_corp.add_trace(go.Bar(
                        x=top_countries_corp['Country'],
                        y=top_countries_corp['MAT Q3 2024\nUSD MNF'],
                        name='2024',
                        marker_color='#3B82F6',
                        text=top_countries_corp['Share_2024'].apply(lambda x: f'{x:.1f}%'),
                        textposition='auto'
                    ))
                    
                    fig_bar_corp.add_trace(go.Bar(
                        x=top_countries_corp['Country'],
                        y=top_countries_corp['MAT Q3 2023\nUSD MNF'],
                        name='2023',
                        marker_color='#60A5FA'
                    ))
                    
                    fig_bar_corp.add_trace(go.Bar(
                        x=top_countries_corp['Country'],
                        y=top_countries_corp['MAT Q3 2022\nUSD MNF'],
                        name='2022',
                        marker_color='#93C5FD'
                    ))
                    
                    fig_bar_corp.update_layout(
                        title=f"{selected_corp} - Top 10 Ülke",
                        barmode='group',
                        height=500,
                        xaxis_tickangle=-45,
                        yaxis_title="USD MNF"
                    )
                    
                    st.plotly_chart(fig_bar_corp, use_container_width=True)
                    
                    # Coğrafi çeşitlilik metrikleri
                    st.markdown("##### 📊 Coğrafi Çeşitlilik Metrikleri")
                    
                    total_countries = len(country_dist_corp)
                    top5_share_corp = top_countries_corp.head(5)['MAT Q3 2024\nUSD MNF'].sum() / total_corp_2024 * 100 if total_corp_2024 > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Toplam Ülke Sayısı", total_countries)
                    
                    with col2:
                        st.metric("Top 5 Ülke Payı", f"{top5_share_corp:.1f}%")
                    
                    with col3:
                        region_count = corp_df['Region'].nunique()
                        st.metric("Bölge Sayısı", region_count)
                    
                    # Molekül dağılımı
                    st.markdown("##### 💊 Molekül Dağılımı")
                    
                    mol_dist_corp = corp_df.groupby('Molecule').agg({
                        'MAT Q3 2024\nUSD MNF': 'sum'
                    }).reset_index()
                    
                    mol_dist_corp['Share'] = (mol_dist_corp['MAT Q3 2024\nUSD MNF'] / mol_dist_corp['MAT Q3 2024\nUSD MNF'].sum() * 100)
                    mol_dist_corp = mol_dist_corp.sort_values('Share', ascending=False).head(10)
                    
                    fig_pie_corp = px.pie(
                        mol_dist_corp,
                        values='MAT Q3 2024\nUSD MNF',
                        names='Molecule',
                        title=f"{selected_corp} - Molekül Dağılımı (2024)",
                        hole=0.4
                    )
                    
                    fig_pie_corp.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie_corp.update_layout(height=500)
                    
                    st.plotly_chart(fig_pie_corp, use_container_width=True)
    
    # ============================================
    # TAB 6: Specialty vs Non-Specialty
    # ============================================
    with tab6:
        st.markdown('<h2 class="sub-header">⭐ Specialty vs Non-Specialty Analizi</h2>', unsafe_allow_html=True)
        
        spec_tab1, spec_tab2, spec_tab3 = st.tabs([
            "📊 Premium Pay Analizi",
            "📈 Premiumlaşma Trendi",
            "🌍 Coğrafi Dağılım"
        ])
        
        with spec_tab1:
            # Specialty metrikleri
            spec_metrics = analytics.calculate_specialty_metrics(filtered_df)
            
            if spec_metrics:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    specialty_pct_2024 = spec_metrics.get('specialty_pct_2024', 0)
                    st.metric(
                        "2024 Specialty Payı",
                        f"{specialty_pct_2024:.1f}%",
                        delta=f"{specialty_pct_2024 - spec_metrics.get('specialty_pct_2023', 0):.1f}%"
                    )
                
                with col2:
                    specialty_total_2024 = spec_metrics.get('specialty_total_2024', 0)
                    st.metric(
                        "2024 Specialty USD MNF",
                        f"${specialty_total_2024:,.0f}M"
                    )
                
                with col3:
                    non_specialty_total_2024 = spec_metrics.get('non_specialty_total_2024', 0)
                    st.metric(
                        "2024 Non-Specialty USD MNF",
                        f"${non_specialty_total_2024:,.0f}M"
                    )
                
                # Pay dağılımı
                st.markdown("##### 📊 Pay Dağılımı (2024)")
                
                fig_pie = px.pie(
                    values=[spec_metrics['specialty_total_2024'], spec_metrics['non_specialty_total_2024']],
                    names=['Specialty', 'Non-Specialty'],
                    title="Specialty vs Non-Specialty Pay Dağılımı (2024)",
                    color=['Specialty', 'Non-Specialty'],
                    color_discrete_map={'Specialty': '#3B82F6', 'Non-Specialty': '#10B981'}
                )
                
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=400)
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Üç yıllık karşılaştırma
                st.markdown("##### 📈 Üç Yıllık Karşılaştırma")
                
                years = [2022, 2023, 2024]
                specialty_values = [spec_metrics[f'specialty_total_{year}'] for year in years]
                non_specialty_values = [spec_metrics[f'non_specialty_total_{year}'] for year in years]
                specialty_pcts = [spec_metrics[f'specialty_pct_{year}'] for year in years]
                
                fig_bar = go.Figure()
                
                fig_bar.add_trace(go.Bar(
                    x=years,
                    y=specialty_values,
                    name='Specialty',
                    marker_color='#3B82F6',
                    text=[f'{pct:.1f}%' for pct in specialty_pcts],
                    textposition='auto'
                ))
                
                fig_bar.add_trace(go.Bar(
                    x=years,
                    y=non_specialty_values,
                    name='Non-Specialty',
                    marker_color='#10B981'
                ))
                
                fig_bar.update_layout(
                    title="Specialty vs Non-Specialty - Üç Yıllık Trend",
                    barmode='stack',
                    height=500,
                    xaxis_title="Yıl",
                    yaxis_title="USD MNF"
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Specialty büyüme analizi
                st.markdown("##### ⚡ Specialty Büyüme Analizi")
                
                specialty_growth_22_23 = ((spec_metrics['specialty_total_2023'] - spec_metrics['specialty_total_2022']) / 
                                         spec_metrics['specialty_total_2022'] * 100) if spec_metrics['specialty_total_2022'] > 0 else 0
                
                specialty_growth_23_24 = ((spec_metrics['specialty_total_2024'] - spec_metrics['specialty_total_2023']) / 
                                         spec_metrics['specialty_total_2023'] * 100) if spec_metrics['specialty_total_2023'] > 0 else 0
                
                non_specialty_growth_22_23 = ((spec_metrics['non_specialty_total_2023'] - spec_metrics['non_specialty_total_2022']) / 
                                             spec_metrics['non_specialty_total_2022'] * 100) if spec_metrics['non_specialty_total_2022'] > 0 else 0
                
                non_specialty_growth_23_24 = ((spec_metrics['non_specialty_total_2024'] - spec_metrics['non_specialty_total_2023']) / 
                                             spec_metrics['non_specialty_total_2023'] * 100) if spec_metrics['non_specialty_total_2023'] > 0 else 0
                
                growth_data = pd.DataFrame({
                    'Segment': ['Specialty', 'Specialty', 'Non-Specialty', 'Non-Specialty'],
                    'Dönem': ['2022→2023', '2023→2024', '2022→2023', '2023→2024'],
                    'Büyüme (%)': [specialty_growth_22_23, specialty_growth_23_24, 
                                  non_specialty_growth_22_23, non_specialty_growth_23_24]
                })
                
                fig_growth = px.bar(
                    growth_data,
                    x='Segment',
                    y='Büyüme (%)',
                    color='Dönem',
                    barmode='group',
                    title="Segment Büyüme Karşılaştırması",
                    color_discrete_sequence=['#3B82F6', '#10B981']
                )
                
                fig_growth.update_layout(height=500)
                st.plotly_chart(fig_growth, use_container_width=True)
        
        with spec_tab2:
            # Premiumlaşma trendi
            st.markdown("##### 📈 Premiumlaşma Trendi (2022 → 2024)")
            
            # Ülke bazlı premiumlaşma analizi
            country_premium = []
            
            for country in filtered_df['Country'].unique():
                country_df_spec = filtered_df[filtered_df['Country'] == country]
                
                spec_metrics_country = analytics.calculate_specialty_metrics(country_df_spec)
                
                if spec_metrics_country:
                    premium_change = spec_metrics_country.get('specialty_pct_2024', 0) - spec_metrics_country.get('specialty_pct_2022', 0)
                    
                    country_premium.append({
                        'Country': country,
                        'Specialty_Pct_2022': spec_metrics_country.get('specialty_pct_2022', 0),
                        'Specialty_Pct_2023': spec_metrics_country.get('specialty_pct_2023', 0),
                        'Specialty_Pct_2024': spec_metrics_country.get('specialty_pct_2024', 0),
                        'Premium_Change_22_24': premium_change
                    })
            
            if country_premium:
                premium_df = pd.DataFrame(country_premium)
                
                # En çok premiumlaşan ülkeler
                top_premium_gainers = premium_df.nlargest(10, 'Premium_Change_22_24')
                top_premium_losers = premium_df.nsmallest(10, 'Premium_Change_22_24')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**⬆️ En Çok Premiumlaşan Ülkeler**")
                    
                    for _, row in top_premium_gainers.iterrows():
                        st.write(f"**{row['Country']}**: +{row['Premium_Change_22_24']:.1f}% (2022: {row['Specialty_Pct_2022']:.1f}% → 2024: {row['Specialty_Pct_2024']:.1f}%)")
                
                with col2:
                    st.markdown("**⬇️ En Çok Premium Kaybeden Ülkeler**")
                    
                    for _, row in top_premium_losers.iterrows():
                        st.write(f"**{row['Country']}**: {row['Premium_Change_22_24']:.1f}% (2022: {row['Specialty_Pct_2022']:.1f}% → 2024: {row['Specialty_Pct_2024']:.1f}%)")
                
                # Premiumlaşma haritası
                st.markdown("##### 🌍 Premiumlaşma Haritası (2022 → 2024)")
                
                premium_df['ISO_A3'] = premium_df['Country'].apply(map_handler.get_country_code)
                premium_df = premium_df.dropna(subset=['ISO_A3'])
                
                if not premium_df.empty and map_handler.world is not None:
                    merged_premium = map_handler.world.merge(premium_df, how='left', left_on='iso_a3', right_on='ISO_A3')
                    
                    def premium_category(x):
                        if pd.isna(x):
                            return 'Veri Yok'
                        elif x > 10:
                            return 'Çok Yüksek Artış (>10%)'
                        elif x > 5:
                            return 'Yüksek Artış (5-10%)'
                        elif x > 0:
                            return 'Orta Artış (0-5%)'
                        elif x > -5:
                            return 'Hafif Düşüş (0- -5%)'
                        else:
                            return 'Keskin Düşüş (<-5%)'
                    
                    merged_premium['Premium_Category'] = merged_premium['Premium_Change_22_24'].apply(premium_category)
                    
                    color_discrete_map_premium = {
                        'Çok Yüksek Artış (>10%)': '#10B981',
                        'Yüksek Artış (5-10%)': '#34D399',
                        'Orta Artış (0-5%)': '#60A5FA',
                        'Hafif Düşüş (0- -5%)': '#FBBF24',
                        'Keskin Düşüş (<-5%)': '#DC2626',
                        'Veri Yok': '#9CA3AF'
                    }
                    
                    fig = px.choropleth(
                        merged_premium,
                        geojson=merged_premium.geometry,
                        locations=merged_premium.index,
                        color='Premium_Category',
                        hover_name='name',
                        hover_data={
                            'Premium_Change_22_24': ':.1f%',
                            'Specialty_Pct_2022': ':.1f%',
                            'Specialty_Pct_2024': ':.1f%'
                        },
                        color_discrete_map=color_discrete_map_premium,
                        category_orders={
                            'Premium_Category': [
                                'Çok Yüksek Artış (>10%)',
                                'Yüksek Artış (5-10%)',
                                'Orta Artış (0-5%)',
                                'Hafif Düşüş (0- -5%)',
                                'Keskin Düşüş (<-5%)',
                                'Veri Yok'
                            ]
                        },
                        title="Premiumlaşma Değişimi (2022 → 2024)"
                    )
                    
                    fig.update_geos(fitbounds="locations", visible=False)
                    fig.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Premiumlaşma trend grafiği
                st.markdown("##### 📊 Global Premiumlaşma Trendi")
                
                global_premium_trend = pd.DataFrame({
                    'Yıl': [2022, 2023, 2024],
                    'Specialty_Payı': [
                        spec_metrics['specialty_pct_2022'],
                        spec_metrics['specialty_pct_2023'],
                        spec_metrics['specialty_pct_2024']
                    ]
                })
                
                fig_trend = px.line(
                    global_premium_trend,
                    x='Yıl',
                    y='Specialty_Payı',
                    markers=True,
                    title="Global Specialty Payı Trendi"
                )
                
                fig_trend.update_layout(
                    height=400,
                    yaxis_title="Specialty Payı (%)",
                    yaxis_range=[0, max(global_premium_trend['Specialty_Payı']) * 1.2]
                )
                
                # Trend çizgisi ekle
                fig_trend.add_trace(go.Scatter(
                    x=global_premium_trend['Yıl'],
                    y=global_premium_trend['Specialty_Payı'],
                    mode='lines',
                    line=dict(color='#EF4444', width=3, dash='dash'),
                    showlegend=False
                ))
                
                st.plotly_chart(fig_trend, use_container_width=True)
        
        with spec_tab3:
            # Coğrafi dağılım
            year_for_spec_map = st.selectbox("Harita Yılı", [2024, 2023, 2022], key="spec_map_year")
            
            # Ülke bazlı specialty payı
            country_spec_data = []
            
            for country in filtered_df['Country'].unique():
                country_df_spec_map = filtered_df[filtered_df['Country'] == country]
                
                spec_metrics_country_map = analytics.calculate_specialty_metrics(country_df_spec_map)
                
                if spec_metrics_country_map:
                    specialty_pct = spec_metrics_country_map.get(f'specialty_pct_{year_for_spec_map}', 0)
                    
                    country_spec_data.append({
                        'Country': country,
                        'Specialty_Pct': specialty_pct,
                        'Specialty_USD': spec_metrics_country_map.get(f'specialty_total_{year_for_spec_map}', 0),
                        'Non_Specialty_USD': spec_metrics_country_map.get(f'non_specialty_total_{year_for_spec_map}', 0)
                    })
            
            if country_spec_data:
                spec_map_df = pd.DataFrame(country_spec_data)
                
                # Harita
                spec_map_df['ISO_A3'] = spec_map_df['Country'].apply(map_handler.get_country_code)
                spec_map_df = spec_map_df.dropna(subset=['ISO_A3'])
                
                if not spec_map_df.empty and map_handler.world is not None:
                    merged_spec_map = map_handler.world.merge(spec_map_df, how='left', left_on='iso_a3', right_on='ISO_A3')
                    
                    def spec_category(x):
                        if pd.isna(x):
                            return 'Veri Yok'
                        elif x > 30:
                            return 'Çok Yüksek (>30%)'
                        elif x > 20:
                            return 'Yüksek (20-30%)'
                        elif x > 10:
                            return 'Orta (10-20%)'
                        elif x > 0:
                            return 'Düşük (0-10%)'
                        else:
                            return 'Yok'
                    
                    merged_spec_map['Spec_Category'] = merged_spec_map['Specialty_Pct'].apply(spec_category)
                    
                    color_discrete_map_spec = {
                        'Çok Yüksek (>30%)': '#1E3A8A',
                        'Yüksek (20-30%)': '#3B82F6',
                        'Orta (10-20%)': '#60A5FA',
                        'Düşük (0-10%)': '#93C5FD',
                        'Yok': '#E5E7EB',
                        'Veri Yok': '#9CA3AF'
                    }
                    
                    fig = px.choropleth(
                        merged_spec_map,
                        geojson=merged_spec_map.geometry,
                        locations=merged_spec_map.index,
                        color='Spec_Category',
                        hover_name='name',
                        hover_data={
                            'Specialty_Pct': ':.1f%',
                            'Specialty_USD': ':.2f',
                            'Non_Specialty_USD': ':.2f'
                        },
                        color_discrete_map=color_discrete_map_spec,
                        category_orders={
                            'Spec_Category': [
                                'Çok Yüksek (>30%)',
                                'Yüksek (20-30%)',
                                'Orta (10-20%)',
                                'Düşük (0-10%)',
                                'Yok',
                                'Veri Yok'
                            ]
                        },
                        title=f"Specialty Payı Coğrafi Dağılımı ({year_for_spec_map})"
                    )
                    
                    fig.update_geos(fitbounds="locations", visible=False)
                    fig.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Ülke sıralaması
                st.markdown(f"##### 🏆 En Yüksek Specialty Payı - {year_for_spec_map}")
                
                top_spec_countries = spec_map_df.nlargest(10, 'Specialty_Pct')
                
                fig_bar_spec = go.Figure()
                
                fig_bar_spec.add_trace(go.Bar(
                    x=top_spec_countries['Country'],
                    y=top_spec_countries['Specialty_Pct'],
                    marker_color='#3B82F6',
                    text=top_spec_countries['Specialty_Pct'].apply(lambda x: f'{x:.1f}%'),
                    textposition='auto'
                ))
                
                fig_bar_spec.update_layout(
                    title=f"Top 10 Ülke - Specialty Payı ({year_for_spec_map})",
                    height=500,
                    xaxis_tickangle=-45,
                    yaxis_title="Specialty Payı (%)"
                )
                
                st.plotly_chart(fig_bar_spec, use_container_width=True)
                
                # Molekül bazlı specialty analizi
                st.markdown("##### 💊 Molekül Bazlı Specialty Analizi")
                
                # Specialty molekülleri
                specialty_molecules = filtered_df[filtered_df['Specialty Product'] == 'Specialty']['Molecule'].unique()
                
                if len(specialty_molecules) > 0:
                    mol_spec_data = []
                    
                    for molecule in specialty_molecules[:10]:  # İlk 10 molekül
                        mol_df_spec = filtered_df[filtered_df['Molecule'] == molecule]
                        
                        if len(mol_df_spec) > 0:
                            if year_for_spec_map == 2022:
                                usd_col = 'MAT Q3 2022\nUSD MNF'
                            elif year_for_spec_map == 2023:
                                usd_col = 'MAT Q3 2023\nUSD MNF'
                            else:
                                usd_col = 'MAT Q3 2024\nUSD MNF'
                            
                            total_usd = mol_df_spec[usd_col].sum()
                            
                            mol_spec_data.append({
                                'Molecule': molecule,
                                'USD_MNF': total_usd
                            })
                    
                    if mol_spec_data:
                        mol_spec_df = pd.DataFrame(mol_spec_data)
                        mol_spec_df = mol_spec_df.sort_values('USD_MNF', ascending=False)
                        
                        fig_mol_spec = px.bar(
                            mol_spec_df,
                            x='Molecule',
                            y='USD_MNF',
                            title=f"Top Specialty Moleküller ({year_for_spec_map})",
                            color_discrete_sequence=['#3B82F6']
                        )
                        
                        fig_mol_spec.update_layout(
                            height=500,
                            xaxis_tickangle=-45,
                            yaxis_title="USD MNF"
                        )
                        
                        st.plotly_chart(fig_mol_spec, use_container_width=True)
    
    # ============================================
    # TAB 7: Fiyat – Volume – Mix
    # ============================================
    with tab7:
        st.markdown('<h2 class="sub-header">📈 Fiyat – Volume – Mix Ayrıştırması</h2>', unsafe_allow_html=True)
        
        pvm_tab1, pvm_tab2, pvm_tab3 = st.tabs([
            "2022 → 2023 Ayrıştırma",
            "2023 → 2024 Ayrıştırma",
            "Zincir Özet"
        ])
        
        with pvm_tab1:
            st.markdown("##### 📊 2022 → 2023 Price-Volume-Mix Ayrıştırması")
            
            pvm_22_23 = analytics.price_volume_mix_analysis(filtered_df, 2022, 2023)
            
            if pvm_22_23:
                # Ayrıştırma grafiği
                effects = ['Fiyat Etkisi', 'Volume Etkisi', 'Mix Etkisi']
                values_pct = [pvm_22_23['price_effect_pct'], pvm_22_23['volume_effect_pct'], pvm_22_23['mix_effect_pct']]
                values_abs = [pvm_22_23['price_effect'], pvm_22_23['volume_effect'], pvm_22_23['mix_effect']]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pie = px.pie(
                        values=values_pct,
                        names=effects,
                        title="Katkı Payları (%)",
                        color=effects,
                        color_discrete_map={
                            'Fiyat Etkisi': '#3B82F6',
                            'Volume Etkisi': '#10B981',
                            'Mix Etkisi': '#F59E0B'
                        }
                    )
                    
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_bar = go.Figure()
                    
                    colors = ['#3B82F6', '#10B981', '#F59E0B']
                    
                    for i, (effect, value_pct, value_abs, color) in enumerate(zip(effects, values_pct, values_abs, colors)):
                        fig_bar.add_trace(go.Bar(
                            x=[effect],
                            y=[value_pct],
                            name=effect,
                            marker_color=color,
                            text=[f'{value_pct:.1f}%<br>${value_abs:,.0f}M'],
                            textposition='auto'
                        ))
                    
                    fig_bar.update_layout(
                        title="Ayrıştırma Katkıları",
                        height=400,
                        yaxis_title="Katkı (%)",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Detaylı metrikler
                st.markdown("##### 📈 Detaylı Metrikler")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_growth_pct = ((pvm_22_23['total_growth'] / filtered_df['MAT Q3 2022\nUSD MNF'].sum()) * 100) if filtered_df['MAT Q3 2022\nUSD MNF'].sum() > 0 else 0
                    st.metric("Toplam Büyüme", f"{total_growth_pct:.1f}%", f"${pvm_22_23['total_growth']:,.0f}M")
                
                with col2:
                    st.metric("Fiyat Etkisi", f"{pvm_22_23['price_effect_pct']:.1f}%", f"${pvm_22_23['price_effect']:,.0f}M")
                
                with col3:
                    st.metric("Volume Etkisi", f"{pvm_22_23['volume_effect_pct']:.1f}%", f"${pvm_22_23['volume_effect']:,.0f}M")
                
                with col4:
                    st.metric("Mix Etkisi", f"{pvm_22_23['mix_effect_pct']:.1f}%", f"${pvm_22_23['mix_effect']:,.0f}M")
                
                # Fiyat ve volume trendi
                st.markdown("##### 📊 Fiyat ve Volume Trendi")
                
                price_change = ((pvm_22_23['weighted_price_end'] - pvm_22_23['weighted_price_start']) / 
                               pvm_22_23['weighted_price_start'] * 100) if pvm_22_23['weighted_price_start'] > 0 else 0
                
                volume_change = pvm_22_23['unit_growth_pct']
                
                fig_trend = go.Figure()
                
                fig_trend.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=price_change,
                    title={'text': "Ortalama Fiyat Değişimi"},
                    delta={'reference': 0},
                    gauge={
                        'axis': {'range': [min(price_change, 0) - 10, max(price_change, 0) + 10]},
                        'bar': {'color': "#3B82F6"},
                        'steps': [
                            {'range': [-100, 0], 'color': "#FEE2E2"},
                            {'range': [0, 100], 'color': "#DCFCE7"}
                        ]
                    },
                    domain={'row': 0, 'column': 0}
                ))
                
                fig_trend.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=volume_change,
                    title={'text': "Volume Değişimi"},
                    delta={'reference': 0},
                    gauge={
                        'axis': {'range': [min(volume_change, 0) - 10, max(volume_change, 0) + 10]},
                        'bar': {'color': "#10B981"},
                        'steps': [
                            {'range': [-100, 0], 'color': "#FEE2E2"},
                            {'range': [0, 100], 'color': "#DCFCE7"}
                        ]
                    },
                    domain={'row': 0, 'column': 1}
                ))
                
                fig_trend.update_layout(
                    grid={'rows': 1, 'columns': 2, 'pattern': "independent"},
                    height=300
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Ülke bazlı ayrıştırma
                st.markdown("##### 🌍 Ülke Bazlı Ayrıştırma")
                
                country_pvm_data = []
                
                for country in filtered_df['Country'].unique()[:10]:  # İlk 10 ülke
                    country_df_pvm = filtered_df[filtered_df['Country'] == country]
                    pvm_country = analytics.price_volume_mix_analysis(country_df_pvm, 2022, 2023)
                    
                    if pvm_country:
                        country_pvm_data.append({
                            'Country': country,
                            'Price_Effect': pvm_country.get('price_effect_pct', 0),
                            'Volume_Effect': pvm_country.get('volume_effect_pct', 0),
                            'Mix_Effect': pvm_country.get('mix_effect_pct', 0),
                            'Total_Growth': ((pvm_country.get('total_growth', 0) / country_df_pvm['MAT Q3 2022\nUSD MNF'].sum()) * 100) if country_df_pvm['MAT Q3 2022\nUSD MNF'].sum() > 0 else 0
                        })
                
                if country_pvm_data:
                    country_pvm_df = pd.DataFrame(country_pvm_data)
                    
                    fig_country = go.Figure()
                    
                    fig_country.add_trace(go.Bar(
                        name='Fiyat',
                        x=country_pvm_df['Country'],
                        y=country_pvm_df['Price_Effect'],
                        marker_color='#3B82F6'
                    ))
                    
                    fig_country.add_trace(go.Bar(
                        name='Volume',
                        x=country_pvm_df['Country'],
                        y=country_pvm_df['Volume_Effect'],
                        marker_color='#10B981'
                    ))
                    
                    fig_country.add_trace(go.Bar(
                        name='Mix',
                        x=country_pvm_df['Country'],
                        y=country_pvm_df['Mix_Effect'],
                        marker_color='#F59E0B'
                    ))
                    
                    fig_country.add_trace(go.Scatter(
                        name='Toplam Büyüme',
                        x=country_pvm_df['Country'],
                        y=country_pvm_df['Total_Growth'],
                        mode='lines+markers',
                        line=dict(color='#EF4444', width=3),
                        yaxis='y2'
                    ))
                    
                    fig_country.update_layout(
                        title="Ülke Bazlı Ayrıştırma (2022 → 2023)",
                        barmode='relative',
                        height=600,
                        xaxis_tickangle=-45,
                        yaxis_title="Katkı (%)",
                        yaxis2=dict(
                            title="Toplam Büyüme (%)",
                            overlaying='y',
                            side='right'
                        )
                    )
                    
                    st.plotly_chart(fig_country, use_container_width=True)
        
        with pvm_tab2:
            st.markdown("##### 📊 2023 → 2024 Price-Volume-Mix Ayrıştırması")
            
            pvm_23_24 = analytics.price_volume_mix_analysis(filtered_df, 2023, 2024)
            
            if pvm_23_24:
                # Ayrıştırma grafiği
                effects = ['Fiyat Etkisi', 'Volume Etkisi', 'Mix Etkisi']
                values_pct = [pvm_23_24['price_effect_pct'], pvm_23_24['volume_effect_pct'], pvm_23_24['mix_effect_pct']]
                values_abs = [pvm_23_24['price_effect'], pvm_23_24['volume_effect'], pvm_23_24['mix_effect']]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pie = px.pie(
                        values=values_pct,
                        names=effects,
                        title="Katkı Payları (%)",
                        color=effects,
                        color_discrete_map={
                            'Fiyat Etkisi': '#3B82F6',
                            'Volume Etkisi': '#10B981',
                            'Mix Etkisi': '#F59E0B'
                        }
                    )
                    
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=400)
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_bar = go.Figure()
                    
                    colors = ['#3B82F6', '#10B981', '#F59E0B']
                    
                    for i, (effect, value_pct, value_abs, color) in enumerate(zip(effects, values_pct, values_abs, colors)):
                        fig_bar.add_trace(go.Bar(
                            x=[effect],
                            y=[value_pct],
                            name=effect,
                            marker_color=color,
                            text=[f'{value_pct:.1f}%<br>${value_abs:,.0f}M'],
                            textposition='auto'
                        ))
                    
                    fig_bar.update_layout(
                        title="Ayrıştırma Katkıları",
                        height=400,
                        yaxis_title="Katkı (%)",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Detaylı metrikler
                st.markdown("##### 📈 Detaylı Metrikler")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_growth_pct = ((pvm_23_24['total_growth'] / filtered_df['MAT Q3 2023\nUSD MNF'].sum()) * 100) if filtered_df['MAT Q3 2023\nUSD MNF'].sum() > 0 else 0
                    st.metric("Toplam Büyüme", f"{total_growth_pct:.1f}%", f"${pvm_23_24['total_growth']:,.0f}M")
                
                with col2:
                    st.metric("Fiyat Etkisi", f"{pvm_23_24['price_effect_pct']:.1f}%", f"${pvm_23_24['price_effect']:,.0f}M")
                
                with col3:
                    st.metric("Volume Etkisi", f"{pvm_23_24['volume_effect_pct']:.1f}%", f"${pvm_23_24['volume_effect']:,.0f}M")
                
                with col4:
                    st.metric("Mix Etkisi", f"{pvm_23_24['mix_effect_pct']:.1f}%", f"${pvm_23_24['mix_effect']:,.0f}M")
                
                # Trend karşılaştırması
                st.markdown("##### 📊 2022→2023 vs 2023→2024 Karşılaştırması")
                
                comparison_data = pd.DataFrame({
                    'Dönem': ['2022→2023', '2023→2024'],
                    'Fiyat Etkisi': [pvm_22_23.get('price_effect_pct', 0), pvm_23_24.get('price_effect_pct', 0)],
                    'Volume Etkisi': [pvm_22_23.get('volume_effect_pct', 0), pvm_23_24.get('volume_effect_pct', 0)],
                    'Mix Etkisi': [pvm_22_23.get('mix_effect_pct', 0), pvm_23_24.get('mix_effect_pct', 0)]
                })
                
                fig_comp = go.Figure()
                
                fig_comp.add_trace(go.Bar(
                    name='Fiyat Etkisi',
                    x=comparison_data['Dönem'],
                    y=comparison_data['Fiyat Etkisi'],
                    marker_color='#3B82F6'
                ))
                
                fig_comp.add_trace(go.Bar(
                    name='Volume Etkisi',
                    x=comparison_data['Dönem'],
                    y=comparison_data['Volume Etkisi'],
                    marker_color='#10B981'
                ))
                
                fig_comp.add_trace(go.Bar(
                    name='Mix Etkisi',
                    x=comparison_data['Dönem'],
                    y=comparison_data['Mix Etkisi'],
                    marker_color='#F59E0B'
                ))
                
                fig_comp.update_layout(
                    title="Ayrıştırma Karşılaştırması",
                    barmode='group',
                    height=500,
                    yaxis_title="Katkı (%)"
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Büyüme kaynakları analizi
                st.markdown("##### 🔍 Büyüme Kaynakları Analizi")
                
                if pvm_22_23 and pvm_23_24:
                    price_contribution_change = pvm_23_24['price_effect_pct'] - pvm_22_23['price_effect_pct']
                    volume_contribution_change = pvm_23_24['volume_effect_pct'] - pvm_22_23['volume_effect_pct']
                    mix_contribution_change = pvm_23_24['mix_effect_pct'] - pvm_22_23['mix_effect_pct']
                    
                    insights = []
                    
                    if price_contribution_change > 5:
                        insights.append("💰 **Fiyat katkısı arttı:** Son dönemde fiyat etkisi daha belirleyici oldu.")
                    elif price_contribution_change < -5:
                        insights.append("📉 **Fiyat katkısı azaldı:** Fiyat artışları yavaşladı.")
                    
                    if volume_contribution_change > 5:
                        insights.append("📦 **Volume katkısı arttı:** Satış hacmi büyümede daha etkili.")
                    elif volume_contribution_change < -5:
                        insights.append("🚫 **Volume katkısı azaldı:** Hacim büyümesi yavaşladı.")
                    
                    if mix_contribution_change > 5:
                        insights.append("🔄 **Mix katkısı arttı:** Ürün mix değişiklikleri büyümeye daha çok katkı sağladı.")
                    elif mix_contribution_change < -5:
                        insights.append("⚖️ **Mix katkısı azaldı:** Ürün mix stabil hale geldi.")
                    
                    if insights:
                        for insight in insights:
                            st.info(insight)
        
        with pvm_tab3:
            st.markdown("##### 📊 Zincir Özet (2022 → 2024)")
            
            # Zincir büyüme hesaplama
            growth_22_24_pct, growth_22_24_abs = analytics.calculate_growth(filtered_df, 2022, 2024)
            
            # Zincir ayrıştırma (22→23 + 23→24)
            if pvm_22_23 and pvm_23_24:
                chain_price = pvm_22_23['price_effect'] + pvm_23_24['price_effect']
                chain_volume = pvm_22_23['volume_effect'] + pvm_23_24['volume_effect']
                chain_mix = pvm_22_23['mix_effect'] + pvm_23_24['mix_effect']
                
                total_start = filtered_df['MAT Q3 2022\nUSD MNF'].sum()
                
                chain_price_pct = (chain_price / total_start * 100) if total_start > 0 else 0
                chain_volume_pct = (chain_volume / total_start * 100) if total_start > 0 else 0
                chain_mix_pct = (chain_mix / total_start * 100) if total_start > 0 else 0
            
            st.markdown("##### 📈 Zincir Büyüme Özeti")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "2022→2024 Büyüme",
                    f"{growth_22_24_pct:.1f}%",
                    f"${growth_22_24_abs:,.0f}M"
                )
            
            with col2:
                st.metric(
                    "Zincir Fiyat Etkisi",
                    f"{chain_price_pct:.1f}%",
                    f"${chain_price:,.0f}M"
                )
            
            with col3:
                st.metric(
                    "Zincir Volume Etkisi",
                    f"{chain_volume_pct:.1f}%",
                    f"${chain_volume:,.0f}M"
                )
            
            with col4:
                st.metric(
                    "Zincir Mix Etkisi",
                    f"{chain_mix_pct:.1f}%",
                    f"${chain_mix:,.0f}M"
                )
            
            # Zincir ayrıştırma grafiği
            st.markdown("##### 📊 Zincir Ayrıştırma")
            
            chain_effects = ['Fiyat Etkisi', 'Volume Etkisi', 'Mix Etkisi']
            chain_values_pct = [chain_price_pct, chain_volume_pct, chain_mix_pct]
            chain_values_abs = [chain_price, chain_volume, chain_mix]
            
            fig_chain = go.Figure()
            
            fig_chain.add_trace(go.Bar(
                x=chain_effects,
                y=chain_values_pct,
                marker_color=['#3B82F6', '#10B981', '#F59E0B'],
                text=[f'{pct:.1f}%<br>${abs:,.0f}M' for pct, abs in zip(chain_values_pct, chain_values_abs)],
                textposition='auto'
            ))
            
            fig_chain.update_layout(
                title="Zincir Ayrıştırma Katkıları (2022 → 2024)",
                height=500,
                yaxis_title="Katkı (%)"
            )
            
            st.plotly_chart(fig_chain, use_container_width=True)
            
            # Dönemsel katkılar
            st.markdown("##### 📅 Dönemsel Katkılar")
            
            period_data = pd.DataFrame({
                'Dönem': ['2022→2023', '2023→2024'],
                'Fiyat Katkısı': [pvm_22_23.get('price_effect_pct', 0), pvm_23_24.get('price_effect_pct', 0)],
                'Volume Katkısı': [pvm_22_23.get('volume_effect_pct', 0), pvm_23_24.get('volume_effect_pct', 0)],
                'Mix Katkısı': [pvm_22_23.get('mix_effect_pct', 0), pvm_23_24.get('mix_effect_pct', 0)]
            })
            
            fig_period = go.Figure()
            
            for col in ['Fiyat Katkısı', 'Volume Katkısı', 'Mix Katkısı']:
                fig_period.add_trace(go.Scatter(
                    x=period_data['Dönem'],
                    y=period_data[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(width=3)
                ))
            
            fig_period.update_layout(
                title="Dönemsel Katkı Trendleri",
                height=500,
                yaxis_title="Katkı (%)",
                xaxis_title="Dönem"
            )
            
            st.plotly_chart(fig_period, use_container_width=True)
            
            # Zincir içgörüler
            st.markdown("##### 🧠 Zincir İçgörüler")
            
            insights_chain = []
            
            if chain_price_pct > chain_volume_pct and chain_price_pct > chain_mix_pct:
                insights_chain.append("💰 **Fiyat odaklı büyüme:** İki yıllık dönemde büyümenin ana kaynağı fiyat artışları oldu.")
            
            if chain_volume_pct > chain_price_pct and chain_volume_pct > chain_mix_pct:
                insights_chain.append("📦 **Volume odaklı büyüme:** İki yıllık dönemde büyümenin ana kaynağı satış hacmi artışı oldu.")
            
            if chain_mix_pct > chain_price_pct and chain_mix_pct > chain_volume_pct:
                insights_chain.append("🔄 **Mix odaklı büyüme:** İki yıllık dönemde büyümenin ana kaynağı ürün mix değişiklikleri oldu.")
            
            if chain_price_pct < 0:
                insights_chain.append("⚠️ **Fiyat erozyonu:** İki yıllık dönemde fiyatlar genel olarak düştü.")
            
            if chain_volume_pct < 0:
                insights_chain.append("⚠️ **Volume kaybı:** İki yıllık dönemde satış hacmi azaldı.")
            
            if growth_22_24_pct > 20:
                insights_chain.append(f"🚀 **Güçlü büyüme:** İki yılda %{growth_22_24_pct:.1f} büyüme kaydedildi.")
            elif growth_22_24_pct < 0:
                insights_chain.append(f"📉 **Küçülme:** İki yılda %{abs(growth_22_24_pct):.1f} küçülme yaşandı.")
            
            if insights_chain:
                for insight in insights_chain:
                    st.info(insight)
    
    # ============================================
    # TAB 8: Otomatik İçgörü Motoru
    # ============================================
    with tab8:
        st.markdown('<h2 class="sub-header">🤖 Otomatik İçgörü Motoru</h2>', unsafe_allow_html=True)
        
        insight_tab1, insight_tab2, insight_tab3 = st.tabs([
            "🇹🇷 Ülke İçgörüleri",
            "💊 Molekül İçgörüleri",
            "🏢 Corporation İçgörüleri"
        ])
        
        with insight_tab1:
            st.markdown("##### 🌍 Ülke Bazlı Otomatik İçgörüler")
            
            country_for_insights = st.selectbox(
                "İçgörü Alınacak Ülke",
                options=sorted(filtered_df['Country'].unique())
            )
            
            if country_for_insights:
                insights = insight_engine.generate_country_insights(country_for_insights)
                
                if insights:
                    st.markdown(f"### {country_for_insights} - Ana İçgörüler")
                    
                    for i, insight in enumerate(insights):
                        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
                    
                    # Ek metrikler
                    st.markdown("##### 📊 Ek Metrikler")
                    
                    country_insight_df = filtered_df[filtered_df['Country'] == country_for_insights]
                    
                    if len(country_insight_df) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_2024 = country_insight_df['MAT Q3 2024\nUSD MNF'].sum()
                            st.metric("2024 Toplam", f"${total_2024:,.0f}M")
                        
                        with col2:
                            global_share = (total_2024 / filtered_df['MAT Q3 2024\nUSD MNF'].sum() * 100) if filtered_df['MAT Q3 2024\nUSD MNF'].sum() > 0 else 0
                            st.metric("Global Pay", f"{global_share:.2f}%")
                        
                        with col3:
                            molecule_count = country_insight_df['Molecule'].nunique()
                            st.metric("Molekül Çeşidi", molecule_count)
                        
                        with col4:
                            manufacturer_count = country_insight_df['Manufacturer'].nunique()
                            st.metric("Üretici Sayısı", manufacturer_count)
                        
                        # Trend özeti
                        st.markdown("##### 📈 Trend Özeti")
                        
                        trend_summary = pd.DataFrame({
                            'Yıl': [2022, 2023, 2024],
                            'USD MNF': [
                                country_insight_df['MAT Q3 2022\nUSD MNF'].sum(),
                                country_insight_df['MAT Q3 2023\nUSD MNF'].sum(),
                                country_insight_df['MAT Q3 2024\nUSD MNF'].sum()
                            ],
                            'Units': [
                                country_insight_df['MAT Q3 2022\nUnits'].sum(),
                                country_insight_df['MAT Q3 2023\nUnits'].sum(),
                                country_insight_df['MAT Q3 2024\nUnits'].sum()
                            ]
                        })
                        
                        fig_trend_summary = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig_trend_summary.add_trace(
                            go.Bar(x=trend_summary['Yıl'], y=trend_summary['USD MNF'],
                                  name='USD MNF', marker_color='#3B82F6'),
                            secondary_y=False,
                        )
                        
                        fig_trend_summary.add_trace(
                            go.Scatter(x=trend_summary['Yıl'], y=trend_summary['Units'],
                                      name='Units', mode='lines+markers',
                                      line=dict(color='#10B981', width=3)),
                            secondary_y=True,
                        )
                        
                        fig_trend_summary.update_layout(
                            title=f"{country_for_insights} - Üç Yıllık Trend",
                            height=400,
                            showlegend=True
                        )
                        
                        fig_trend_summary.update_yaxes(title_text="USD MNF", secondary_y=False)
                        fig_trend_summary.update_yaxes(title_text="Units", secondary_y=True)
                        
                        st.plotly_chart(fig_trend_summary, use_container_width=True)
                else:
                    st.warning(f"{country_for_insights} için yeterli veri bulunamadı.")
        
        with insight_tab2:
            st.markdown("##### 💊 Molekül Bazlı Otomatik İçgörüler")
            
            molecule_for_insights = st.selectbox(
                "İçgörü Alınacak Molekül",
                options=sorted(filtered_df['Molecule'].unique())
            )
            
            if molecule_for_insights:
                insights = insight_engine.generate_molecule_insights(molecule_for_insights)
                
                if insights:
                    st.markdown(f"### {molecule_for_insights} - Ana İçgörüler")
                    
                    for i, insight in enumerate(insights):
                        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
                    
                    # Ek metrikler
                    st.markdown("##### 📊 Ek Metrikler")
                    
                    mol_insight_df = filtered_df[filtered_df['Molecule'] == molecule_for_insights]
                    
                    if len(mol_insight_df) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_2024 = mol_insight_df['MAT Q3 2024\nUSD MNF'].sum()
                            st.metric("2024 Toplam", f"${total_2024:,.0f}M")
                        
                        with col2:
                            global_share = (total_2024 / filtered_df['MAT Q3 2024\nUSD MNF'].sum() * 100) if filtered_df['MAT Q3 2024\nUSD MNF'].sum() > 0 else 0
                            st.metric("Global Pay", f"{global_share:.2f}%")
                        
                        with col3:
                            country_count = mol_insight_df['Country'].nunique()
                            st.metric("Ülke Sayısı", country_count)
                        
                        with col4:
                            manufacturer_count = mol_insight_df['Manufacturer'].nunique()
                            st.metric("Üretici Sayısı", manufacturer_count)
                        
                        # Coğrafi dağılım
                        st.markdown("##### 🌍 Coğrafi Dağılım")
                        
                        country_dist_insight = mol_insight_df.groupby('Country').agg({
                            'MAT Q3 2024\nUSD MNF': 'sum'
                        }).reset_index()
                        
                        country_dist_insight = country_dist_insight.sort_values('MAT Q3 2024\nUSD MNF', ascending=False).head(10)
                        
                        fig_country_insight = px.bar(
                            country_dist_insight,
                            x='Country',
                            y='MAT Q3 2024\nUSD MNF',
                            title=f"{molecule_for_insights} - Top 10 Ülke (2024)",
                            color_discrete_sequence=['#3B82F6']
                        )
                        
                        fig_country_insight.update_layout(
                            height=400,
                            xaxis_tickangle=-45,
                            yaxis_title="USD MNF"
                        )
                        
                        st.plotly_chart(fig_country_insight, use_container_width=True)
                else:
                    st.warning(f"{molecule_for_insights} için yeterli veri bulunamadı.")
        
        with insight_tab3:
            st.markdown("##### 🏢 Corporation Bazlı Otomatik İçgörüler")
            
            corp_for_insights = st.selectbox(
                "İçgörü Alınacak Corporation",
                options=sorted(filtered_df['Corporation'].unique())
            )
            
            if corp_for_insights:
                insights = insight_engine.generate_corporation_insights(corp_for_insights)
                
                if insights:
                    st.markdown(f"### {corp_for_insights} - Ana İçgörüler")
                    
                    for i, insight in enumerate(insights):
                        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
                    
                    # Ek metrikler
                    st.markdown("##### 📊 Ek Metrikler")
                    
                    corp_insight_df = filtered_df[filtered_df['Corporation'] == corp_for_insights]
                    
                    if len(corp_insight_df) > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_2024 = corp_insight_df['MAT Q3 2024\nUSD MNF'].sum()
                            st.metric("2024 Toplam", f"${total_2024:,.0f}M")
                        
                        with col2:
                            global_share = (total_2024 / filtered_df['MAT Q3 2024\nUSD MNF'].sum() * 100) if filtered_df['MAT Q3 2024\nUSD MNF'].sum() > 0 else 0
                            st.metric("Global Pay", f"{global_share:.2f}%")
                        
                        with col3:
                            country_count = corp_insight_df['Country'].nunique()
                            st.metric("Ülke Sayısı", country_count)
                        
                        with col4:
                            molecule_count = corp_insight_df['Molecule'].nunique()
                            st.metric("Molekül Çeşidi", molecule_count)
                        
                        # Pazar payı trendi
                        st.markdown("##### 📈 Pazar Payı Trendi")
                        
                        share_trend = []
                        for year in [2022, 2023, 2024]:
                            if year == 2022:
                                usd_col = 'MAT Q3 2022\nUSD MNF'
                            elif year == 2023:
                                usd_col = 'MAT Q3 2023\nUSD MNF'
                            else:
                                usd_col = 'MAT Q3 2024\nUSD MNF'
                            
                            corp_total = corp_insight_df[usd_col].sum()
                            global_total = filtered_df[usd_col].sum()
                            share = (corp_total / global_total * 100) if global_total > 0 else 0
                            
                            share_trend.append({
                                'Yıl': year,
                                'Pazar Payı': share
                            })
                        
                        share_trend_df = pd.DataFrame(share_trend)
                        
                        fig_share_trend = px.line(
                            share_trend_df,
                            x='Yıl',
                            y='Pazar Payı',
                            markers=True,
                            title=f"{corp_for_insights} - Pazar Payı Trendi"
                        )
                        
                        fig_share_trend.update_layout(
                            height=400,
                            yaxis_title="Pazar Payı (%)",
                            yaxis_range=[0, max(share_trend_df['Pazar Payı']) * 1.5]
                        )
                        
                        st.plotly_chart(fig_share_trend, use_container_width=True)
                else:
                    st.warning(f"{corp_for_insights} için yeterli veri bulunamadı.")
    
    # ============================================
    # FOOTER
    # ============================================
    st.markdown("---")
    st.markdown('<div class="footer">© 2024 Pharma Commercial Analytics Suite | Enterprise Streamlit Application</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
