# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Pharma Commercial Analytics Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 1.2rem;
    }
    .dataframe {
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED COLUMNS DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLUMNS = [
    "Source.Name",
    "Country",
    "Sector",
    "Panel",
    "Region",
    "Sub-Region",
    "Corporation",
    "Manufacturer",
    "Molecule List",
    "Molecule",
    "Chemical Salt",
    "International Product",
    "Specialty Product",
    "NFC123",
    "International Pack",
    "International Strength",
    "International Size",
    "International Volume",
    "International Prescription",
    "MAT Q3 2022\nUSD MNF",
    "MAT Q3 2022\nStandard Units",
    "MAT Q3 2022\nUnits",
    "MAT Q3 2022\nSU Avg Price USD MNF",
    "MAT Q3 2022\nUnit Avg Price USD MNF",
    "MAT Q3 2023\nUSD MNF",
    "MAT Q3 2023\nStandard Units",
    "MAT Q3 2023\nUnits",
    "MAT Q3 2023\nSU Avg Price USD MNF",
    "MAT Q3 2023\nUnit Avg Price USD MNF",
    "MAT Q3 2024\nUSD MNF",
    "MAT Q3 2024\nStandard Units",
    "MAT Q3 2024\nUnits",
    "MAT Q3 2024\nSU Avg Price USD MNF",
    "MAT Q3 2024\nUnit Avg Price USD MNF"
]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_validate_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        
        missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_columns:
            st.error(f"❌ Eksik kolonlar tespit edildi: {missing_columns}")
            st.stop()
        
        extra_columns = set(df.columns) - set(REQUIRED_COLUMNS)
        if extra_columns:
            st.warning(f"⚠️ Fazladan kolonlar bulundu ve görmezden geliniyor: {extra_columns}")
        
        df = df[REQUIRED_COLUMNS].copy()
        
        numeric_columns = [col for col in df.columns if 'MAT Q3' in col]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.fillna(0, inplace=True)
        
        return df
    
    except Exception as e:
        st.error(f"❌ Dosya yükleme hatası: {str(e)}")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL FILTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_global_filters(df):
    filtered_df = df.copy()
    
    st.sidebar.markdown("## 🎛️ Global Filtreler")
    st.sidebar.markdown("---")
    
    countries = sorted(df['Country'].unique())
    selected_countries = st.sidebar.multiselect(
        "🌍 Ülke Seçimi",
        options=countries,
        default=countries,
        key="country_filter"
    )
    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
    
    regions = sorted(filtered_df['Region'].unique())
    selected_regions = st.sidebar.multiselect(
        "🗺️ Bölge Seçimi",
        options=regions,
        default=regions,
        key="region_filter"
    )
    if selected_regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_regions)]
    
    sub_regions = sorted(filtered_df['Sub-Region'].unique())
    selected_sub_regions = st.sidebar.multiselect(
        "📍 Alt Bölge Seçimi",
        options=sub_regions,
        default=sub_regions,
        key="sub_region_filter"
    )
    if selected_sub_regions:
        filtered_df = filtered_df[filtered_df['Sub-Region'].isin(selected_sub_regions)]
    
    sectors = sorted(filtered_df['Sector'].unique())
    selected_sectors = st.sidebar.multiselect(
        "🏢 Sektör Seçimi",
        options=sectors,
        default=sectors,
        key="sector_filter"
    )
    if selected_sectors:
        filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]
    
    panels = sorted(filtered_df['Panel'].unique())
    selected_panels = st.sidebar.multiselect(
        "📊 Panel Seçimi",
        options=panels,
        default=panels,
        key="panel_filter"
    )
    if selected_panels:
        filtered_df = filtered_df[filtered_df['Panel'].isin(selected_panels)]
    
    corporations = sorted(filtered_df['Corporation'].unique())
    selected_corporations = st.sidebar.multiselect(
        "🏛️ Corporation Seçimi",
        options=corporations,
        default=corporations,
        key="corporation_filter"
    )
    if selected_corporations:
        filtered_df = filtered_df[filtered_df['Corporation'].isin(selected_corporations)]
    
    manufacturers = sorted(filtered_df['Manufacturer'].unique())
    selected_manufacturers = st.sidebar.multiselect(
        "🏭 Üretici Seçimi",
        options=manufacturers,
        default=manufacturers,
        key="manufacturer_filter"
    )
    if selected_manufacturers:
        filtered_df = filtered_df[filtered_df['Manufacturer'].isin(selected_manufacturers)]
    
    molecules = sorted(filtered_df['Molecule'].unique())
    selected_molecules = st.sidebar.multiselect(
        "⚗️ Molekül Seçimi",
        options=molecules,
        default=molecules,
        key="molecule_filter"
    )
    if selected_molecules:
        filtered_df = filtered_df[filtered_df['Molecule'].isin(selected_molecules)]
    
    specialty_products = sorted(filtered_df['Specialty Product'].unique())
    selected_specialty = st.sidebar.multiselect(
        "💎 Specialty Product Seçimi",
        options=specialty_products,
        default=specialty_products,
        key="specialty_filter"
    )
    if selected_specialty:
        filtered_df = filtered_df[filtered_df['Specialty Product'].isin(selected_specialty)]
    
    intl_products = sorted(filtered_df['International Product'].unique())
    selected_intl_products = st.sidebar.multiselect(
        "🌐 International Product Seçimi",
        options=intl_products,
        default=intl_products,
        key="intl_product_filter"
    )
    if selected_intl_products:
        filtered_df = filtered_df[filtered_df['International Product'].isin(selected_intl_products)]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📋 Filtrelenmiş Kayıt Sayısı: {len(filtered_df):,}")
    
    return filtered_df

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_growth_rate(current, previous):
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def calculate_contribution(current, previous):
    return current - previous

def safe_division(numerator, denominator):
    if denominator == 0:
        return 0
    return numerator / denominator

def format_number(num):
    if abs(num) >= 1_000_000_000:
        return f"${num/1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"${num/1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"${num/1_000:.2f}K"
    else:
        return f"${num:.2f}"

def format_percentage(pct):
    return f"{pct:.2f}%"

def get_trend_indicator(value):
    if value > 0:
        return "📈"
    elif value < 0:
        return "📉"
    else:
        return "➡️"

# ═══════════════════════════════════════════════════════════════════════════════
# PRICE-VOLUME-MIX DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

def price_volume_mix_decomposition(df, year_from, year_to, groupby_cols=None):
    if groupby_cols is None:
        groupby_cols = []
    
    usd_col_from = f"MAT Q3 {year_from}\nUSD MNF"
    usd_col_to = f"MAT Q3 {year_to}\nUSD MNF"
    units_col_from = f"MAT Q3 {year_from}\nStandard Units"
    units_col_to = f"MAT Q3 {year_to}\nStandard Units"
    price_col_from = f"MAT Q3 {year_from}\nSU Avg Price USD MNF"
    price_col_to = f"MAT Q3 {year_to}\nSU Avg Price USD MNF"
    
    if groupby_cols:
        agg_dict = {
            usd_col_from: 'sum',
            usd_col_to: 'sum',
            units_col_from: 'sum',
            units_col_to: 'sum'
        }
        grouped = df.groupby(groupby_cols).agg(agg_dict).reset_index()
        
        grouped['avg_price_from'] = safe_division(
            grouped[usd_col_from], 
            grouped[units_col_from]
        )
        grouped['avg_price_to'] = safe_division(
            grouped[usd_col_to], 
            grouped[units_col_to]
        )
        
        grouped['total_change'] = grouped[usd_col_to] - grouped[usd_col_from]
        grouped['volume_effect'] = (grouped[units_col_to] - grouped[units_col_from]) * grouped['avg_price_from']
        grouped['price_effect'] = (grouped['avg_price_to'] - grouped['avg_price_from']) * grouped[units_col_to]
        grouped['mix_effect'] = grouped['total_change'] - grouped['volume_effect'] - grouped['price_effect']
        
        return grouped
    else:
        total_usd_from = df[usd_col_from].sum()
        total_usd_to = df[usd_col_to].sum()
        total_units_from = df[units_col_from].sum()
        total_units_to = df[units_col_to].sum()
        
        avg_price_from = safe_division(total_usd_from, total_units_from)
        avg_price_to = safe_division(total_usd_to, total_units_to)
        
        total_change = total_usd_to - total_usd_from
        volume_effect = (total_units_to - total_units_from) * avg_price_from
        price_effect = (avg_price_to - avg_price_from) * total_units_to
        mix_effect = total_change - volume_effect - price_effect
        
        return {
            'total_change': total_change,
            'volume_effect': volume_effect,
            'price_effect': price_effect,
            'mix_effect': mix_effect,
            'total_usd_from': total_usd_from,
            'total_usd_to': total_usd_to
        }

# ═══════════════════════════════════════════════════════════════════════════════
# COUNTRY MAP GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_country_map(df, metric_col, title, color_scale='Blues'):
    country_data = df.groupby('Country')[metric_col].sum().reset_index()
    country_data.columns = ['Country', 'Value']
    
    fig = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color='Value',
        hover_name='Country',
        hover_data={'Value': ':,.2f'},
        color_continuous_scale=color_scale,
        title=title
    )
    
    fig.update_layout(
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        )
    )
    
    return fig

def generate_global_share_map(df, year):
    usd_col = f"MAT Q3 {year}\nUSD MNF"
    units_col = f"MAT Q3 {year}\nUnits"
    su_col = f"MAT Q3 {year}\nStandard Units"
    
    country_agg = df.groupby('Country').agg({
        usd_col: 'sum',
        units_col: 'sum',
        su_col: 'sum'
    }).reset_index()
    
    total_global = country_agg[usd_col].sum()
    country_agg['Global_Share_Pct'] = (country_agg[usd_col] / total_global) * 100
    
    hover_text = []
    for idx, row in country_agg.iterrows():
        text = f"<b>{row['Country']}</b><br>"
        text += f"USD MNF: {format_number(row[usd_col])}<br>"
        text += f"Units: {row[units_col]:,.0f}<br>"
        text += f"Standard Units: {row[su_col]:,.0f}<br>"
        text += f"Global Pay: {row['Global_Share_Pct']:.2f}%"
        hover_text.append(text)
    
    country_agg['hover_text'] = hover_text
    
    fig = px.choropleth(
        country_agg,
        locations='Country',
        locationmode='country names',
        color='Global_Share_Pct',
        hover_name='Country',
        hover_data={'hover_text': True, 'Global_Share_Pct': False},
        color_continuous_scale='YlOrRd',
        title=f'Global Pay (%) - {year}'
    )
    
    fig.update_traces(hovertemplate='%{customdata[0]}')
    
    fig.update_layout(
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        )
    )
    
    return fig

def generate_growth_map(df, year_from, year_to):
    usd_col_from = f"MAT Q3 {year_from}\nUSD MNF"
    usd_col_to = f"MAT Q3 {year_to}\nUSD MNF"
    
    country_agg = df.groupby('Country').agg({
        usd_col_from: 'sum',
        usd_col_to: 'sum'
    }).reset_index()
    
    country_agg['Growth_Pct'] = country_agg.apply(
        lambda row: calculate_growth_rate(row[usd_col_to], row[usd_col_from]),
        axis=1
    )
    
    hover_text = []
    for idx, row in country_agg.iterrows():
        text = f"<b>{row['Country']}</b><br>"
        text += f"{year_from}: {format_number(row[usd_col_from])}<br>"
        text += f"{year_to}: {format_number(row[usd_col_to])}<br>"
        text += f"Büyüme: {row['Growth_Pct']:.2f}%"
        hover_text.append(text)
    
    country_agg['hover_text'] = hover_text
    
    fig = px.choropleth(
        country_agg,
        locations='Country',
        locationmode='country names',
        color='Growth_Pct',
        hover_name='Country',
        hover_data={'hover_text': True, 'Growth_Pct': False},
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title=f'Büyüme Oranı (%) - {year_from} → {year_to}'
    )
    
    fig.update_traces(hovertemplate='%{customdata[0]}')
    
    fig.update_layout(
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        )
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# AUTOMATIC INSIGHTS GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_executive_insights(df):
    insights = []
    
    usd_2022 = df["MAT Q3 2022\nUSD MNF"].sum()
    usd_2023 = df["MAT Q3 2023\nUSD MNF"].sum()
    usd_2024 = df["MAT Q3 2024\nUSD MNF"].sum()
    
    growth_22_23 = calculate_growth_rate(usd_2023, usd_2022)
    growth_23_24 = calculate_growth_rate(usd_2024, usd_2023)
    
    insights.append(f"Global satışlar 2022'den 2023'e {get_trend_indicator(growth_22_23)} {abs(growth_22_23):.2f}% değişim gösterdi.")
    insights.append(f"2023'den 2024'e satışlarda {get_trend_indicator(growth_23_24)} {abs(growth_23_24):.2f}% değişim gözlemlendi.")
    
    country_2024 = df.groupby('Country')["MAT Q3 2024\nUSD MNF"].sum().sort_values(ascending=False)
    top_country = country_2024.index[0]
    top_country_value = country_2024.iloc[0]
    top_country_share = (top_country_value / usd_2024) * 100
    
    insights.append(f"En büyük pazar {top_country} olup, toplam satışların %{top_country_share:.2f}'sini oluşturuyor.")
    
    country_growth = df.groupby('Country').agg({
        "MAT Q3 2023\nUSD MNF": 'sum',
        "MAT Q3 2024\nUSD MNF": 'sum'
    })
    country_growth['Growth'] = country_growth.apply(
        lambda row: calculate_growth_rate(row["MAT Q3 2024\nUSD MNF"], row["MAT Q3 2023\nUSD MNF"]),
        axis=1
    )
    country_growth = country_growth.sort_values('Growth', ascending=False)
    
    if len(country_growth) > 0:
        fastest_growing = country_growth.index[0]
        fastest_growth_rate = country_growth['Growth'].iloc[0]
        insights.append(f"En hızlı büyüyen pazar {fastest_growing} olup, %{fastest_growth_rate:.2f} büyüme gösterdi.")
    
    molecule_2024 = df.groupby('Molecule')["MAT Q3 2024\nUSD MNF"].sum().sort_values(ascending=False)
    if len(molecule_2024) > 0:
        top_molecule = molecule_2024.index[0]
        top_molecule_value = molecule_2024.iloc[0]
        top_molecule_share = (top_molecule_value / usd_2024) * 100
        insights.append(f"En değerli molekül {top_molecule} olup, toplam satışların %{top_molecule_share:.2f}'sini temsil ediyor.")
    
    return insights

def generate_country_insights(df, country):
    insights = []
    
    country_df = df[df['Country'] == country]
    
    if len(country_df) == 0:
        return ["Seçilen ülke için yeterli veri bulunmuyor."]
    
    usd_2022 = country_df["MAT Q3 2022\nUSD MNF"].sum()
    usd_2023 = country_df["MAT Q3 2023\nUSD MNF"].sum()
    usd_2024 = country_df["MAT Q3 2024\nUSD MNF"].sum()
    
    growth_22_23 = calculate_growth_rate(usd_2023, usd_2022)
    growth_23_24 = calculate_growth_rate(usd_2024, usd_2023)
    
    insights.append(f"{country} pazarı 2022'den 2023'e {get_trend_indicator(growth_22_23)} {abs(growth_22_23):.2f}% değişim gösterdi.")
    insights.append(f"{country} pazarı 2023'den 2024'e {get_trend_indicator(growth_23_24)} {abs(growth_23_24):.2f}% değişim gösterdi.")
    
    global_usd_2024 = df["MAT Q3 2024\nUSD MNF"].sum()
    country_share = (usd_2024 / global_usd_2024) * 100
    insights.append(f"{country}'in global pazardaki payı %{country_share:.2f} seviyesinde.")
    
    top_molecules = country_df.groupby('Molecule')["MAT Q3 2024\nUSD MNF"].sum().sort_values(ascending=False).head(3)
    if len(top_molecules) > 0:
        mol_list = ", ".join([f"{mol} ({format_number(val)})" for mol, val in top_molecules.items()])
        insights.append(f"{country}'deki en büyük moleküller: {mol_list}")
    
    return insights

def generate_molecule_insights(df, molecule):
    insights = []
    
    molecule_df = df[df['Molecule'] == molecule]
    
    if len(molecule_df) == 0:
        return ["Seçilen molekül için yeterli veri bulunmuyor."]
    
    usd_2022 = molecule_df["MAT Q3 2022\nUSD MNF"].sum()
    usd_2023 = molecule_df["MAT Q3 2023\nUSD MNF"].sum()
    usd_2024 = molecule_df["MAT Q3 2024\nUSD MNF"].sum()
    
    growth_22_23 = calculate_growth_rate(usd_2023, usd_2022)
    growth_23_24 = calculate_growth_rate(usd_2024, usd_2023)
    
    insights.append(f"{molecule} satışları 2022'den 2023'e {get_trend_indicator(growth_22_23)} {abs(growth_22_23):.2f}% değişim gösterdi.")
    insights.append(f"{molecule} satışları 2023'den 2024'e {get_trend_indicator(growth_23_24)} {abs(growth_23_24):.2f}% değişim gösterdi.")
    
    global_usd_2024 = df["MAT Q3 2024\nUSD MNF"].sum()
    molecule_share = (usd_2024 / global_usd_2024) * 100
    insights.append(f"{molecule}'ün global pazardaki payı %{molecule_share:.2f} seviyesinde.")
    
    country_sales = molecule_df.groupby('Country')["MAT Q3 2024\nUSD MNF"].sum().sort_values(ascending=False).head(5)
    if len(country_sales) > 0:
        country_list = ", ".join([f"{country} ({format_number(val)})" for country, val in country_sales.items()])
        insights.append(f"{molecule} için en büyük pazarlar: {country_list}")
    
    return insights

def generate_corporation_insights(df, corporation):
    insights = []
    
    corp_df = df[df['Corporation'] == corporation]
    
    if len(corp_df) == 0:
        return ["Seçilen corporation için yeterli veri bulunmuyor."]
    
    usd_2022 = corp_df["MAT Q3 2022\nUSD MNF"].sum()
    usd_2023 = corp_df["MAT Q3 2023\nUSD MNF"].sum()
    usd_2024 = corp_df["MAT Q3 2024\nUSD MNF"].sum()
    
    growth_22_23 = calculate_growth_rate(usd_2023, usd_2022)
    growth_23_24 = calculate_growth_rate(usd_2024, usd_2023)
    
    insights.append(f"{corporation} satışları 2022'den 2023'e {get_trend_indicator(growth_22_23)} {abs(growth_22_23):.2f}% değişim gösterdi.")
    insights.append(f"{corporation} satışları 2023'den 2024'e {get_trend_indicator(growth_23_24)} {abs(growth_23_24):.2f}% değişim gösterdi.")
    
    global_usd_2024 = df["MAT Q3 2024\nUSD MNF"].sum()
    corp_share = (usd_2024 / global_usd_2024) * 100
    insights.append(f"{corporation}'ın global pazardaki payı %{corp_share:.2f} seviyesinde.")
    
    top_countries = corp_df.groupby('Country')["MAT Q3 2024\nUSD MNF"].sum().sort_values(ascending=False).head(3)
    if len(top_countries) > 0:
        country_list = ", ".join([f"{country} ({format_number(val)})" for country, val in top_countries.items()])
        insights.append(f"{corporation} için en büyük pazarlar: {country_list}")
    
    return insights

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("<h1 class='main-header'>💊 Pharma Commercial Analytics Platform</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📂 Excel Dosyası Yükleyin (.xlsx)",
        type=['xlsx'],
        help="MAT Q3 2022, 2023, 2024 verilerini içeren Excel dosyasını yükleyin."
    )
    
    if uploaded_file is None:
        st.info("👆 Lütfen analiz için Excel dosyasını yükleyin.")
        st.markdown("### 📋 Beklenen Veri Formatı")
        st.markdown("""
        Excel dosyanız aşağıdaki kolonları içermelidir:
        - Tanımlayıcı Kolonlar: Source.Name, Country, Sector, Panel, Region, Sub-Region, Corporation, Manufacturer, vb.
        - Zaman Serisi Kolonları: MAT Q3 2022/2023/2024 için USD MNF, Standard Units, Units, Avg Price kolonları
        """)
        st.stop()
    
    with st.spinner("🔄 Veriler yükleniyor ve doğrulanıyor..."):
        df = load_and_validate_data(uploaded_file)
    
    st.success(f"✅ Veri başarıyla yüklendi! Toplam {len(df):,} kayıt.")
    
    filtered_df = apply_global_filters(df)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Platform Bilgileri")
    st.sidebar.info("""
    **Enterprise Pharma Analytics**
    
    - 3 Yıllık Trend Analizi
    - Global Harita Görselleştirme
    - Price-Volume-Mix Ayrıştırma
    - Otomatik İçgörü Motoru
    """)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Yönetici Özeti",
        "🌍 Global Harita Analizi",
        "🏳️ Ülke Derinlemesine",
        "⚗️ Molekül & Ürün",
        "🏛️ Corporation & Rekabet",
        "💎 Specialty vs Non-Specialty",
        "📈 Fiyat-Volume-Mix",
        "🧠 Otomatik İçgörü Motoru"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: YÖNETİCİ ÖZETİ
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.markdown("<h2 class='sub-header'>📊 Yönetici Özeti - Global Performans</h2>", unsafe_allow_html=True)
        
        usd_2022 = filtered_df["MAT Q3 2022\nUSD MNF"].sum()
        usd_2023 = filtered_df["MAT Q3 2023\nUSD MNF"].sum()
        usd_2024 = filtered_df["MAT Q3 2024\nUSD MNF"].sum()
        
        units_2022 = filtered_df["MAT Q3 2022\nUnits"].sum()
        units_2023 = filtered_df["MAT Q3 2023\nUnits"].sum()
        units_2024 = filtered_df["MAT Q3 2024\nUnits"].sum()
        
        su_2022 = filtered_df["MAT Q3 2022\nStandard Units"].sum()
        su_2023 = filtered_df["MAT Q3 2023\nStandard Units"].sum()
        su_2024 = filtered_df["MAT Q3 2024\nStandard Units"].sum()
        
        growth_usd_22_23 = calculate_growth_rate(usd_2023, usd_2022)
        growth_usd_23_24 = calculate_growth_rate(usd_2024, usd_2023)
        
        growth_units_22_23 = calculate_growth_rate(units_2023, units_2022)
        growth_units_23_24 = calculate_growth_rate(units_2024, units_2023)
        
        growth_su_22_23 = calculate_growth_rate(su_2023, su_2022)
        growth_su_23_24 = calculate_growth_rate(su_2024, su_2023)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="💰 2022 USD MNF",
                value=format_number(usd_2022),
                delta=None
            )
            st.metric(
                label="💰 2023 USD MNF",
                value=format_number(usd_2023),
                delta=f"{growth_usd_22_23:.2f}%"
            )
            st.metric(
                label="💰 2024 USD MNF",
                value=format_number(usd_2024),
                delta=f"{growth_usd_23_24:.2f}%"
            )
        
        with col2:
            st.metric(
                label="📦 2022 Units",
                value=f"{units_2022:,.0f}",
                delta=None
            )
            st.metric(
                label="📦 2023 Units",
                value=f"{units_2023:,.0f}",
                delta=f"{growth_units_22_23:.2f}%"
            )
            st.metric(
                label="📦 2024 Units",
                value=f"{units_2024:,.0f}",
                delta=f"{growth_units_23_24:.2f}%"
            )
        
        with col3:
            st.metric(
                label="📊 2022 Standard Units",
                value=f"{su_2022:,.0f}",
                delta=None
            )
            st.metric(
                label="📊 2023 Standard Units",
                value=f"{su_2023:,.0f}",
                delta=f"{growth_su_22_23:.2f}%"
            )
            st.metric(
                label="📊 2024 Standard Units",
                value=f"{su_2024:,.0f}",
                delta=f"{growth_su_23_24:.2f}%"
            )
        
        st.markdown("---")
        
        st.markdown("### 📈 3 Yıllık USD MNF Trendi")
        
        trend_data = pd.DataFrame({
            'Yıl': ['2022', '2023', '2024'],
            'USD MNF': [usd_2022, usd_2023, usd_2024]
        })
        
        fig_trend = px.line(
            trend_data,
            x='Yıl',
            y='USD MNF',
            markers=True,
            title='Global USD MNF Trendi (2022-2024)',
            text='USD MNF'
        )
        
        fig_trend.update_traces(
            texttemplate='%{text:.2s}',
            textposition='top center',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=12)
        )
        
        fig_trend.update_layout(
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🏆 Top 10 Ülkeler - 2024 USD MNF")
        
        country_2024 = filtered_df.groupby('Country').agg({
            "MAT Q3 2022\nUSD MNF": 'sum',
            "MAT Q3 2023\nUSD MNF": 'sum',
            "MAT Q3 2024\nUSD MNF": 'sum'
        }).reset_index()
        
        country_2024['Growth_22_23'] = country_2024.apply(
            lambda row: calculate_growth_rate(row["MAT Q3 2023\nUSD MNF"], row["MAT Q3 2022\nUSD MNF"]),
            axis=1
        )
        
        country_2024['Growth_23_24'] = country_2024.apply(
            lambda row: calculate_growth_rate(row["MAT Q3 2024\nUSD MNF"], row["MAT Q3 2023\nUSD MNF"]),
            axis=1
        )
        
        country_2024['Contribution_22_23'] = country_2024["MAT Q3 2023\nUSD MNF"] - country_2024["MAT Q3 2022\nUSD MNF"]
        country_2024['Contribution_23_24'] = country_2024["MAT Q3 2024\nUSD MNF"] - country_2024["MAT Q3 2023\nUSD MNF"]
        
        country_2024['Global_Share_2024'] = (country_2024["MAT Q3 2024\nUSD MNF"] / usd_2024) * 100
        
        top_countries = country_2024.nlargest(10, "MAT Q3 2024\nUSD MNF")
        
        fig_top_countries = px.bar(
            top_countries,
            x='Country',
            y="MAT Q3 2024\nUSD MNF",
            title='Top 10 Ülkeler - 2024 USD MNF',
            text="MAT Q3 2024\nUSD MNF",
            color='Global_Share_2024',
            color_continuous_scale='Blues'
        )
        
        fig_top_countries.update_traces(
            texttemplate='%{text:.2s}',
            textposition='outside'
        )
        
        fig_top_countries.update_layout(height=500)
        
        st.plotly_chart(fig_top_countries, use_container_width=True)
        
        st.markdown("### 📋 Top 10 Ülkeler - Detaylı Tablo")
        
        display_top_countries = top_countries[[
            'Country',
            'MAT Q3 2024\nUSD MNF',
            'Global_Share_2024',
            'Growth_22_23',
            'Growth_23_24',
            'Contribution_22_23',
            'Contribution_23_24'
        ]].copy()
        
        display_top_countries.columns = [
            'Ülke',
            '2024 USD MNF',
            'Global Pay (%)',
            'Büyüme 22→23 (%)',
            'Büyüme 23→24 (%)',
            'Katkı 22→23',
            'Katkı 23→24'
        ]
        
        st.dataframe(
            display_top_countries.style.format({
                '2024 USD MNF': '{:,.0f}',
                'Global Pay (%)': '{:.2f}',
                'Büyüme 22→23 (%)': '{:.2f}',
                'Büyüme 23→24 (%)': '{:.2f}',
                'Katkı 22→23': '{:,.0f}',
                'Katkı 23→24': '{:,.0f}'
            }),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        st.markdown("### 🧠 Otomatik İçgörüler")
        
        insights = generate_executive_insights(filtered_df)
        
        for insight in insights:
            st.markdown(f"<div class='insight-box'>💡 {insight}</div>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: GLOBAL HARİTA ANALİZİ
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.markdown("<h2 class='sub-header'>🌍 Global Harita Analizi</h2>", unsafe_allow_html=True)
        
        st.markdown("### 💰 Dünya Haritası - USD MNF")
        
        map_year = st.selectbox(
            "Yıl Seçimi",
            options=[2022, 2023, 2024],
            index=2,
            key="map_year_usd"
        )
        
        usd_col = f"MAT Q3 {map_year}\nUSD MNF"
        fig_map_usd = generate_country_map(filtered_df, usd_col, f"Global USD MNF Dağılımı - {map_year}", 'Blues')
        st.plotly_chart(fig_map_usd, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Dünya Haritası - Global Pay (%)")
        
        map_year_share = st.selectbox(
            "Yıl Seçimi",
            options=[2022, 2023, 2024],
            index=2,
            key="map_year_share"
        )
        
        fig_map_share = generate_global_share_map(filtered_df, map_year_share)
        st.plotly_chart(fig_map_share, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Dünya Haritası - Büyüme Oranı (%)")
        
        growth_period = st.selectbox(
            "Büyüme Dönemi Seçimi",
            options=["2022 → 2023", "2023 → 2024"],
            index=1,
            key="growth_period"
        )
        
        if growth_period == "2022 → 2023":
            fig_map_growth = generate_growth_map(filtered_df, 2022, 2023)
        else:
            fig_map_growth = generate_growth_map(filtered_df, 2023, 2024)
        
        st.plotly_chart(fig_map_growth, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Global Pay Trend - Top 10 Ülkeler")
        
        share_data = []
        for year in [2022, 2023, 2024]:
            usd_col = f"MAT Q3 {year}\nUSD MNF"
            country_sum = filtered_df.groupby('Country')[usd_col].sum().reset_index()
            total = country_sum[usd_col].sum()
            country_sum['Share'] = (country_sum[usd_col] / total) * 100
            country_sum['Year'] = year
            share_data.append(country_sum[['Country', 'Share', 'Year']])
        
        share_df = pd.concat(share_data, ignore_index=True)
        
        top_10_countries_2024 = filtered_df.groupby('Country')["MAT Q3 2024\nUSD MNF"].sum().nlargest(10).index.tolist()
        
        share_df_top10 = share_df[share_df['Country'].isin(top_10_countries_2024)]
        
        fig_share_trend = px.line(
            share_df_top10,
            x='Year',
            y='Share',
            color='Country',
            markers=True,
            title='Top 10 Ülkeler - Global Pay Trendi (%)',
            labels={'Share': 'Global Pay (%)', 'Year': 'Yıl'}
        )
        
        fig_share_trend.update_layout(height=500, hovermode='x unified')
        
        st.plotly_chart(fig_share_trend, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: ÜLKE DERİNLEMESİNE
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab3:
        st.markdown("<h2 class='sub-header'>🏳️ Ülke Derinlemesine Analiz</h2>", unsafe_allow_html=True)
        
        available_countries = sorted(filtered_df['Country'].unique())
        
        selected_country = st.selectbox(
            "Ülke Seçin",
            options=available_countries,
            key="country_deep_dive"
        )
        
        country_df = filtered_df[filtered_df['Country'] == selected_country]
        
        if len(country_df) == 0:
            st.warning("Seçilen ülke için veri bulunmuyor.")
        else:
            usd_2022_country = country_df["MAT Q3 2022\nUSD MNF"].sum()
            usd_2023_country = country_df["MAT Q3 2023\nUSD MNF"].sum()
            usd_2024_country = country_df["MAT Q3 2024\nUSD MNF"].sum()
            
            growth_22_23_country = calculate_growth_rate(usd_2023_country, usd_2022_country)
            growth_23_24_country = calculate_growth_rate(usd_2024_country, usd_2023_country)
            
            global_share_2022 = (usd_2022_country / usd_2022) * 100
            global_share_2023 = (usd_2023_country / usd_2023) * 100
            global_share_2024 = (usd_2024_country / usd_2024) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=f"💰 {selected_country} - 2022 USD MNF",
                    value=format_number(usd_2022_country),
                    delta=None
                )
                st.metric(
                    label=f"💰 {selected_country} - 2023 USD MNF",
                    value=format_number(usd_2023_country),
                    delta=f"{growth_22_23_country:.2f}%"
                )
                st.metric(
                    label=f"💰 {selected_country} - 2024 USD MNF",
                    value=format_number(usd_2024_country),
                    delta=f"{growth_23_24_country:.2f}%"
                )
            
            with col2:
                st.metric(
                    label="🎯 Global Pay - 2022",
                    value=f"{global_share_2022:.2f}%",
                    delta=None
                )
                st.metric(
                    label="🎯 Global Pay - 2023",
                    value=f"{global_share_2023:.2f}%",
                    delta=f"{global_share_2023 - global_share_2022:.2f} pp"
                )
                st.metric(
                    label="🎯 Global Pay - 2024",
                    value=f"{global_share_2024:.2f}%",
                    delta=f"{global_share_2024 - global_share_2023:.2f} pp"
                )
            
            with col3:
                contribution_22_23 = usd_2023_country - usd_2022_country
                contribution_23_24 = usd_2024_country - usd_2023_country
                
                st.metric(
                    label="📊 Katkı 22→23",
                    value=format_number(contribution_22_23),
                    delta=None
                )
                st.metric(
                    label="📊 Katkı 23→24",
                    value=format_number(contribution_23_24),
                    delta=None
                )
                
                total_contribution = contribution_22_23 + contribution_23_24
                st.metric(
                    label="📊 Toplam Katkı (22→24)",
                    value=format_number(total_contribution),
                    delta=None
                )
            
            st.markdown("---")
            
            st.markdown(f"### 📈 {selected_country} - 3 Yıllık Satış Trendi")
            
            country_trend = pd.DataFrame({
                'Yıl': ['2022', '2023', '2024'],
                'USD MNF': [usd_2022_country, usd_2023_country, usd_2024_country]
            })
            
            fig_country_trend = px.line(
                country_trend,
                x='Yıl',
                y='USD MNF',
                markers=True,
                title=f'{selected_country} - USD MNF Trendi',
                text='USD MNF'
            )
            
            fig_country_trend.update_traces(
                texttemplate='%{text:.2s}',
                textposition='top center',
                line=dict(width=3, color='#ff7f0e'),
                marker=dict(size=12)
            )
            
            fig_country_trend.update_layout(height=400, hovermode='x unified')
            
            st.plotly_chart(fig_country_trend, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown(f"### 💹 {selected_country} - Price-Volume-Mix Ayrıştırması")
            
            st.markdown("#### 2022 → 2023")
            
            pvm_22_23 = price_volume_mix_decomposition(country_df, 2022, 2023)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam Değişim", format_number(pvm_22_23['total_change']))
            with col2:
                st.metric("Volume Etkisi", format_number(pvm_22_23['volume_effect']))
            with col3:
                st.metric("Price Etkisi", format_number(pvm_22_23['price_effect']))
            with col4:
                st.metric("Mix Etkisi", format_number(pvm_22_23['mix_effect']))
            
            pvm_22_23_df = pd.DataFrame({
                'Bileşen': ['Volume', 'Price', 'Mix'],
                'Katkı': [pvm_22_23['volume_effect'], pvm_22_23['price_effect'], pvm_22_23['mix_effect']]
            })
            
            fig_pvm_22_23 = px.bar(
                pvm_22_23_df,
                x='Bileşen',
                y='Katkı',
                title=f'{selected_country} - PVM Ayrıştırması (2022 → 2023)',
                text='Katkı',
                color='Bileşen',
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
            
            fig_pvm_22_23.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig_pvm_22_23.update_layout(height=400, showlegend=False)
            
            st.plotly_chart(fig_pvm_22_23, use_container_width=True)
            
            st.markdown("#### 2023 → 2024")
            
            pvm_23_24 = price_volume_mix_decomposition(country_df, 2023, 2024)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam Değişim", format_number(pvm_23_24['total_change']))
            with col2:
                st.metric("Volume Etkisi", format_number(pvm_23_24['volume_effect']))
            with col3:
                st.metric("Price Etkisi", format_number(pvm_23_24['price_effect']))
            with col4:
                st.metric("Mix Etkisi", format_number(pvm_23_24['mix_effect']))
            
            pvm_23_24_df = pd.DataFrame({
                'Bileşen': ['Volume', 'Price', 'Mix'],
                'Katkı': [pvm_23_24['volume_effect'], pvm_23_24['price_effect'], pvm_23_24['mix_effect']]
            })
            
            fig_pvm_23_24 = px.bar(
                pvm_23_24_df,
                x='Bileşen',
                y='Katkı',
                title=f'{selected_country} - PVM Ayrıştırması (2023 → 2024)',
                text='Katkı',
                color='Bileşen',
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
            
            fig_pvm_23_24.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig_pvm_23_24.update_layout(height=400, showlegend=False)
            
            st.plotly_chart(fig_pvm_23_24, use_container_width=True)
            
            st.markdown("#### Zincir Özet (2022 → 2024)")
            
            total_change_chain = pvm_22_23['total_change'] + pvm_23_24['total_change']
            volume_chain = pvm_22_23['volume_effect'] + pvm_23_24['volume_effect']
            price_chain = pvm_22_23['price_effect'] + pvm_23_24['price_effect']
            mix_chain = pvm_22_23['mix_effect'] + pvm_23_24['mix_effect']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam Değişim (Zincir)", format_number(total_change_chain))
            with col2:
                st.metric("Volume Etkisi (Zincir)", format_number(volume_chain))
            with col3:
                st.metric("Price Etkisi (Zincir)", format_number(price_chain))
            with col4:
                st.metric("Mix Etkisi (Zincir)", format_number(mix_chain))
            
            st.markdown("---")
            
            st.markdown(f"### ⚗️ {selected_country} - Top 10 Moleküller")
            
            molecule_country = country_df.groupby('Molecule').agg({
                "MAT Q3 2022\nUSD MNF": 'sum',
                "MAT Q3 2023\nUSD MNF": 'sum',
                "MAT Q3 2024\nUSD MNF": 'sum'
            }).reset_index()
            
            molecule_country['Growth_22_23'] = molecule_country.apply(
                lambda row: calculate_growth_rate(row["MAT Q3 2023\nUSD MNF"], row["MAT Q3 2022\nUSD MNF"]),
                axis=1
            )
            
            molecule_country['Growth_23_24'] = molecule_country.apply(
                lambda row: calculate_growth_rate(row["MAT Q3 2024\nUSD MNF"], row["MAT Q3 2023\nUSD MNF"]),
                axis=1
            )
            
            top_molecules_country = molecule_country.nlargest(10, "MAT Q3 2024\nUSD MNF")
            
            fig_mol_country = px.bar(
                top_molecules_country,
                x='Molecule',
                y="MAT Q3 2024\nUSD MNF",
                title=f'{selected_country} - Top 10 Moleküller (2024)',
                text="MAT Q3 2024\nUSD MNF",
                color='Growth_23_24',
                color_continuous_scale='RdYlGn'
            )
            
            fig_mol_country.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig_mol_country.update_layout(height=500)
            
            st.plotly_chart(fig_mol_country, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown(f"### 🧠 {selected_country} - Otomatik İçgörüler")
            
            country_insights = generate_country_insights(filtered_df, selected_country)
            
            for insight in country_insights:
                st.markdown(f"<div class='insight-box'>💡 {insight}</div>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: MOLEKÜL & ÜRÜN
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab4:
        st.markdown("<h2 class='sub-header'>⚗️ Molekül & Ürün Analizi</h2>", unsafe_allow_html=True)
        
        available_molecules = sorted(filtered_df['Molecule'].unique())
        
        selected_molecule = st.selectbox(
            "Molekül Seçin",
            options=available_molecules,
            key="molecule_analysis"
        )
        
        molecule_df = filtered_df[filtered_df['Molecule'] == selected_molecule]
        
        if len(molecule_df) == 0:
            st.warning("Seçilen molekül için veri bulunmuyor.")
        else:
            usd_2022_mol = molecule_df["MAT Q3 2022\nUSD MNF"].sum()
            usd_2023_mol = molecule_df["MAT Q3 2023\nUSD MNF"].sum()
            usd_2024_mol = molecule_df["MAT Q3 2024\nUSD MNF"].sum()
            
            growth_22_23_mol = calculate_growth_rate(usd_2023_mol, usd_2022_mol)
            growth_23_24_mol = calculate_growth_rate(usd_2024_mol, usd_2023_mol)
            
            global_share_2024_mol = (usd_2024_mol / usd_2024) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=f"💰 {selected_molecule} - 2022 USD MNF",
                    value=format_number(usd_2022_mol),
                    delta=None
                )
                st.metric(
                    label=f"💰 {selected_molecule} - 2023 USD MNF",
                    value=format_number(usd_2023_mol),
                    delta=f"{growth_22_23_mol:.2f}%"
                )
                st.metric(
                    label=f"💰 {selected_molecule} - 2024 USD MNF",
                    value=format_number(usd_2024_mol),
                    delta=f"{growth_23_24_mol:.2f}%"
                )
            
            with col2:
                units_2022_mol = molecule_df["MAT Q3 2022\nUnits"].sum()
                units_2023_mol = molecule_df["MAT Q3 2023\nUnits"].sum()
                units_2024_mol = molecule_df["MAT Q3 2024\nUnits"].sum()
                
                growth_units_22_23_mol = calculate_growth_rate(units_2023_mol, units_2022_mol)
                growth_units_23_24_mol = calculate_growth_rate(units_2024_mol, units_2023_mol)
                
                st.metric(
                    label="📦 Units - 2022",
                    value=f"{units_2022_mol:,.0f}",
                    delta=None
                )
                st.metric(
                    label="📦 Units - 2023",
                    value=f"{units_2023_mol:,.0f}",
                    delta=f"{growth_units_22_23_mol:.2f}%"
                )
                st.metric(
                    label="📦 Units - 2024",
                    value=f"{units_2024_mol:,.0f}",
                    delta=f"{growth_units_23_24_mol:.2f}%"
                )
            
            with col3:
                st.metric(
                    label="🎯 Global Pay - 2024",
                    value=f"{global_share_2024_mol:.2f}%",
                    delta=None
                )
                
                avg_price_2022_mol = safe_division(usd_2022_mol, molecule_df["MAT Q3 2022\nStandard Units"].sum())
                avg_price_2023_mol = safe_division(usd_2023_mol, molecule_df["MAT Q3 2023\nStandard Units"].sum())
                avg_price_2024_mol = safe_division(usd_2024_mol, molecule_df["MAT Q3 2024\nStandard Units"].sum())
                
                price_growth_22_23 = calculate_growth_rate(avg_price_2023_mol, avg_price_2022_mol)
                price_growth_23_24 = calculate_growth_rate(avg_price_2024_mol, avg_price_2023_mol)
                
                st.metric(
                    label="💵 Avg Price - 2023",
                    value=f"${avg_price_2023_mol:.2f}",
                    delta=f"{price_growth_22_23:.2f}%"
                )
                st.metric(
                    label="💵 Avg Price - 2024",
                    value=f"${avg_price_2024_mol:.2f}",
                    delta=f"{price_growth_23_24:.2f}%"
                )
            
            st.markdown("---")
            
            st.markdown(f"### 📈 {selected_molecule} - 3 Yıllık Satış Trendi")
            
            mol_trend = pd.DataFrame({
                'Yıl': ['2022', '2023', '2024'],
                'USD MNF': [usd_2022_mol, usd_2023_mol, usd_2024_mol]
            })
            
            fig_mol_trend = px.line(
                mol_trend,
                x='Yıl',
                y='USD MNF',
                markers=True,
                title=f'{selected_molecule} - USD MNF Trendi',
                text='USD MNF'
            )
            
            fig_mol_trend.update_traces(
                texttemplate='%{text:.2s}',
                textposition='top center',
                line=dict(width=3, color='#2ca02c'),
                marker=dict(size=12)
            )
            
            fig_mol_trend.update_layout(height=400, hovermode='x unified')
            
            st.plotly_chart(fig_mol_trend, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown(f"### 🌍 {selected_molecule} - Ülke Bazlı Dağılım (2024)")
            
            country_mol = molecule_df.groupby('Country').agg({
                "MAT Q3 2022\nUSD MNF": 'sum',
                "MAT Q3 2023\nUSD MNF": 'sum',
                "MAT Q3 2024\nUSD MNF": 'sum'
            }).reset_index()
            
            country_mol['Growth_23_24'] = country_mol.apply(
                lambda row: calculate_growth_rate(row["MAT Q3 2024\nUSD MNF"], row["MAT Q3 2023\nUSD MNF"]),
                axis=1
            )
            
            country_mol['Share_2024'] = (country_mol["MAT Q3 2024\nUSD MNF"] / usd_2024_mol) * 100
            
            top_countries_mol = country_mol.nlargest(10, "MAT Q3 2024\nUSD MNF")
            
            fig_country_mol = px.bar(
                top_countries_mol,
                x='Country',
                y="MAT Q3 2024\nUSD MNF",
                title=f'{selected_molecule} - Top 10 Ülkeler (2024)',
                text="MAT Q3 2024\nUSD MNF",
                color='Growth_23_24',
                color_continuous_scale='RdYlGn'
            )
            
            fig_country_mol.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig_country_mol.update_layout(height=500)
            
            st.plotly_chart(fig_country_mol, use_container_width=True)
            
            st.markdown("### 📋 Ülke Detayları")
            
            display_country_mol = top_countries_mol[[
                'Country',
                'MAT Q3 2024\nUSD MNF',
                'Share_2024',
                'Growth_23_24'
            ]].copy()
            
            display_country_mol.columns = ['Ülke', '2024 USD MNF', 'Molekül İçi Pay (%)', 'Büyüme 23→24 (%)']
            
            st.dataframe(
                display_country_mol.style.format({
                    '2024 USD MNF': '{:,.0f}',
                    'Molekül İçi Pay (%)': '{:.2f}',
                    'Büyüme 23→24 (%)': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
            
            st.markdown("---")
            
            st.markdown(f"### 🏛️ {selected_molecule} - Corporation Dağılımı")
            
            corp_mol = molecule_df.groupby('Corporation').agg({
                "MAT Q3 2022\nUSD MNF": 'sum',
                "MAT Q3 2023\nUSD MNF": 'sum',
                "MAT Q3 2024\nUSD MNF": 'sum'
            }).reset_index()
            
            corp_mol['Share_2024'] = (corp_mol["MAT Q3 2024\nUSD MNF"] / usd_2024_mol) * 100
            corp_mol = corp_mol.sort_values("MAT Q3 2024\nUSD MNF", ascending=False).head(10)
            
            fig_corp_mol = px.pie(
                corp_mol,
                names='Corporation',
                values="MAT Q3 2024\nUSD MNF",
                title=f'{selected_molecule} - Corporation Pazar Payı (2024)',
                hole=0.4
            )
            
            fig_corp_mol.update_traces(textposition='inside', textinfo='percent+label')
            fig_corp_mol.update_layout(height=500)
            
            st.plotly_chart(fig_corp_mol, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown(f"### 🧠 {selected_molecule} - Otomatik İçgörüler")
            
            molecule_insights = generate_molecule_insights(filtered_df, selected_molecule)
            
            for insight in molecule_insights:
                st.markdown(f"<div class='insight-box'>💡 {insight}</div>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5: CORPORATION & REKABET
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab5:
        st.markdown("<h2 class='sub-header'>🏛️ Corporation & Rekabet Analizi</h2>", unsafe_allow_html=True)
        
        st.markdown("### 📊 Global Corporation Pazar Payları")
        
        corp_global = filtered_df.groupby('Corporation').agg({
            "MAT Q3 2022\nUSD MNF": 'sum',
            "MAT Q3 2023\nUSD MNF": 'sum',
            "MAT Q3 2024\nUSD MNF": 'sum'
        }).reset_index()
        
        corp_global['Share_2022'] = (corp_global["MAT Q3 2022\nUSD MNF"] / usd_2022) * 100
        corp_global['Share_2023'] = (corp_global["MAT Q3 2023\nUSD MNF"] / usd_2023) * 100
        corp_global['Share_2024'] = (corp_global["MAT Q3 2024\nUSD MNF"] / usd_2024) * 100
        
        corp_global['Share_Change_22_23'] = corp_global['Share_2023'] - corp_global['Share_2022']
        corp_global['Share_Change_23_24'] = corp_global['Share_2024'] - corp_global['Share_2023']
        
        corp_global_sorted = corp_global.sort_values("MAT Q3 2024\nUSD MNF", ascending=False).head(15)
        
        fig_corp_share = px.bar(
            corp_global_sorted,
            x='Corporation',
            y='Share_2024',
            title='Top 15 Corporation - Pazar Payı (2024)',
            text='Share_2024',
            color='Share_Change_23_24',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0
        )
        
        fig_corp_share.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_corp_share.update_layout(height=500)
        
        st.plotly_chart(fig_corp_share, use_container_width=True)
        
        st.markdown("### 📋 Corporation Detayları")
        
        display_corp = corp_global_sorted[[
            'Corporation',
            'Share_2022',
            'Share_2023',
            'Share_2024',
            'Share_Change_22_23',
            'Share_Change_23_24'
        ]].copy()
        
        display_corp.columns = [
            'Corporation',
            'Pay 2022 (%)',
            'Pay 2023 (%)',
            'Pay 2024 (%)',
            'Pay Değişimi 22→23 (pp)',
            'Pay Değişimi 23→24 (pp)'
        ]
        
        st.dataframe(
            display_corp.style.format({
                'Pay 2022 (%)': '{:.2f}',
                'Pay 2023 (%)': '{:.2f}',
                'Pay 2024 (%)': '{:.2f}',
                'Pay Değişimi 22→23 (pp)': '{:.2f}',
                'Pay Değişimi 23→24 (pp)': '{:.2f}'
            }),
            use_container_width=True,
            height=500
        )
        
        st.markdown("---")
        
        st.markdown("### 📈 Corporation Pazar Payı Trendi")
        
        corp_trend_data = []
        for _, row in corp_global_sorted.iterrows():
            corp_trend_data.append({'Corporation': row['Corporation'], 'Yıl': '2022', 'Pay (%)': row['Share_2022']})
            corp_trend_data.append({'Corporation': row['Corporation'], 'Yıl': '2023', 'Pay (%)': row['Share_2023']})
            corp_trend_data.append({'Corporation': row['Corporation'], 'Yıl': '2024', 'Pay (%)': row['Share_2024']})
        
        corp_trend_df = pd.DataFrame(corp_trend_data)
        
        fig_corp_trend = px.line(
            corp_trend_df,
            x='Yıl',
            y='Pay (%)',
            color='Corporation',
            markers=True,
            title='Top 15 Corporation - Pazar Payı Trendi'
        )
        
        fig_corp_trend.update_layout(height=600, hovermode='x unified')
        
        st.plotly_chart(fig_corp_trend, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🔍 Corporation Derinlemesine Analiz")
        
        available_corporations = sorted(filtered_df['Corporation'].unique())
        
        selected_corporation = st.selectbox(
            "Corporation Seçin",
            options=available_corporations,
            key="corporation_analysis"
        )
        
        corp_df = filtered_df[filtered_df['Corporation'] == selected_corporation]
        
        if len(corp_df) == 0:
            st.warning("Seçilen corporation için veri bulunmuyor.")
        else:
            usd_2022_corp = corp_df["MAT Q3 2022\nUSD MNF"].sum()
            usd_2023_corp = corp_df["MAT Q3 2023\nUSD MNF"].sum()
            usd_2024_corp = corp_df["MAT Q3 2024\nUSD MNF"].sum()
            
            growth_22_23_corp = calculate_growth_rate(usd_2023_corp, usd_2022_corp)
            growth_23_24_corp = calculate_growth_rate(usd_2024_corp, usd_2023_corp)
            
            share_2022_corp = (usd_2022_corp / usd_2022) * 100
            share_2023_corp = (usd_2023_corp / usd_2023) * 100
            share_2024_corp = (usd_2024_corp / usd_2024) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label=f"💰 {selected_corporation} - 2024 USD MNF",
                    value=format_number(usd_2024_corp),
                    delta=f"{growth_23_24_corp:.2f}%"
                )
                st.metric(
                    label="🎯 Pazar Payı - 2024",
                    value=f"{share_2024_corp:.2f}%",
                    delta=f"{share_2024_corp - share_2023_corp:.2f} pp"
                )
            
            with col2:
                st.metric(
                    label="📊 Büyüme 22→23",
                    value=f"{growth_22_23_corp:.2f}%",
                    delta=None
                )
                st.metric(
                    label="📊 Büyüme 23→24",
                    value=f"{growth_23_24_corp:.2f}%",
                    delta=None
                )
            
            st.markdown(f"#### {selected_corporation} - Top 10 Ülkeler")
            
            country_corp = corp_df.groupby('Country')["MAT Q3 2024\nUSD MNF"].sum().sort_values(ascending=False).head(10)
            
            fig_country_corp = px.bar(
                x=country_corp.index,
                y=country_corp.values,
                title=f'{selected_corporation} - Top 10 Ülkeler (2024)',
                labels={'x': 'Ülke', 'y': 'USD MNF'},
                text=country_corp.values
            )
            
            fig_country_corp.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig_country_corp.update_layout(height=400)
            
            st.plotly_chart(fig_country_corp, use_container_width=True)
            
            st.markdown(f"#### {selected_corporation} - Top 10 Moleküller")
            
            molecule_corp = corp_df.groupby('Molecule')["MAT Q3 2024\nUSD MNF"].sum().sort_values(ascending=False).head(10)
            
            fig_molecule_corp = px.bar(
                x=molecule_corp.index,
                y=molecule_corp.values,
                title=f'{selected_corporation} - Top 10 Moleküller (2024)',
                labels={'x': 'Molekül', 'y': 'USD MNF'},
                text=molecule_corp.values
            )
            
            fig_molecule_corp.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig_molecule_corp.update_layout(height=400)
            
            st.plotly_chart(fig_molecule_corp, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown(f"### 🧠 {selected_corporation} - Otomatik İçgörüler")
            
            corp_insights = generate_corporation_insights(filtered_df, selected_corporation)
            
            for insight in corp_insights:
                st.markdown(f"<div class='insight-box'>💡 {insight}</div>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 6: SPECIALTY VS NON-SPECIALTY
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab6:
        st.markdown("<h2 class='sub-header'>💎 Specialty vs Non-Specialty Analizi</h2>", unsafe_allow_html=True)
        
        specialty_agg = filtered_df.groupby('Specialty Product').agg({
            "MAT Q3 2022\nUSD MNF": 'sum',
            "MAT Q3 2023\nUSD MNF": 'sum',
            "MAT Q3 2024\nUSD MNF": 'sum',
            "MAT Q3 2022\nStandard Units": 'sum',
            "MAT Q3 2023\nStandard Units": 'sum',
            "MAT Q3 2024\nStandard Units": 'sum'
        }).reset_index()
        
        specialty_agg['Share_2022'] = (specialty_agg["MAT Q3 2022\nUSD MNF"] / usd_2022) * 100
        specialty_agg['Share_2023'] = (specialty_agg["MAT Q3 2023\nUSD MNF"] / usd_2023) * 100
        specialty_agg['Share_2024'] = (specialty_agg["MAT Q3 2024\nUSD MNF"] / usd_2024) * 100
        
        specialty_agg['Growth_22_23'] = specialty_agg.apply(
            lambda row: calculate_growth_rate(row["MAT Q3 2023\nUSD MNF"], row["MAT Q3 2022\nUSD MNF"]),
            axis=1
        )
        
        specialty_agg['Growth_23_24'] = specialty_agg.apply(
            lambda row: calculate_growth_rate(row["MAT Q3 2024\nUSD MNF"], row["MAT Q3 2023\nUSD MNF"]),
            axis=1
        )
        
        specialty_agg['Avg_Price_2022'] = specialty_agg.apply(
            lambda row: safe_division(row["MAT Q3 2022\nUSD MNF"], row["MAT Q3 2022\nStandard Units"]),
            axis=1
        )
        
        specialty_agg['Avg_Price_2023'] = specialty_agg.apply(
            lambda row: safe_division(row["MAT Q3 2023\nUSD MNF"], row["MAT Q3 2023\nStandard Units"]),
            axis=1
        )
        
        specialty_agg['Avg_Price_2024'] = specialty_agg.apply(
            lambda row: safe_division(row["MAT Q3 2024\nUSD MNF"], row["MAT Q3 2024\nStandard Units"]),
            axis=1
        )
        
        st.markdown("### 📊 Specialty vs Non-Specialty - USD MNF Karşılaştırması")
        
        specialty_comparison = specialty_agg.melt(
            id_vars=['Specialty Product'],
            value_vars=["MAT Q3 2022\nUSD MNF", "MAT Q3 2023\nUSD MNF", "MAT Q3 2024\nUSD MNF"],
            var_name='Yıl',
            value_name='USD MNF'
        )
        
        specialty_comparison['Yıl'] = specialty_comparison['Yıl'].str.replace('MAT Q3 ', '').str.replace('\nUSD MNF', '')
        
        fig_specialty_comp = px.bar(
            specialty_comparison,
            x='Yıl',
            y='USD MNF',
            color='Specialty Product',
            barmode='group',
            title='Specialty vs Non-Specialty - USD MNF Karşılaştırması',
            text='USD MNF'
        )
        
        fig_specialty_comp.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig_specialty_comp.update_layout(height=500)
        
        st.plotly_chart(fig_specialty_comp, use_container_width=True)
        
        st.markdown("### 📋 Specialty vs Non-Specialty - Detaylı Tablo")
        
        display_specialty = specialty_agg[[
            'Specialty Product',
            'MAT Q3 2024\nUSD MNF',
            'Share_2024',
            'Growth_22_23',
            'Growth_23_24',
            'Avg_Price_2024'
        ]].copy()
        
        display_specialty.columns = [
            'Kategori',
            '2024 USD MNF',
            'Pazar Payı (%)',
            'Büyüme 22→23 (%)',
            'Büyüme 23→24 (%)',
            'Ortalama Fiyat'
        ]
        
        st.dataframe(
            display_specialty.style.format({
                '2024 USD MNF': '{:,.0f}',
                'Pazar Payı (%)': '{:.2f}',
                'Büyüme 22→23 (%)': '{:.2f}',
                'Büyüme 23→24 (%)': '{:.2f}',
                'Ortalama Fiyat': '{:.2f}'
            }),
            use_container_width=True,
            height=300
        )
        
        st.markdown("---")
        
        st.markdown("### 📈 Specialty Pay Trendi")
        
        specialty_trend_data = []
        for _, row in specialty_agg.iterrows():
            specialty_trend_data.append({'Kategori': row['Specialty Product'], 'Yıl': '2022', 'Pay (%)': row['Share_2022']})
            specialty_trend_data.append({'Kategori': row['Specialty Product'], 'Yıl': '2023', 'Pay (%)': row['Share_2023']})
            specialty_trend_data.append({'Kategori': row['Specialty Product'], 'Yıl': '2024', 'Pay (%)': row['Share_2024']})
        
        specialty_trend_df = pd.DataFrame(specialty_trend_data)
        
        fig_specialty_trend = px.line(
            specialty_trend_df,
            x='Yıl',
            y='Pay (%)',
            color='Kategori',
            markers=True,
            title='Specialty vs Non-Specialty - Pazar Payı Trendi'
        )
        
        fig_specialty_trend.update_layout(height=400, hovermode='x unified')
        
        st.plotly_chart(fig_specialty_trend, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 💵 Ortalama Fiyat Karşılaştırması")
        
        price_comparison = specialty_agg.melt(
            id_vars=['Specialty Product'],
            value_vars=['Avg_Price_2022', 'Avg_Price_2023', 'Avg_Price_2024'],
            var_name='Yıl',
            value_name='Avg Price'
        )
        
        price_comparison['Yıl'] = price_comparison['Yıl'].str.replace('Avg_Price_', '')
        
        fig_price_comp = px.bar(
            price_comparison,
            x='Yıl',
            y='Avg Price',
            color='Specialty Product',
            barmode='group',
            title='Specialty vs Non-Specialty - Ortalama Fiyat Karşılaştırması',
            text='Avg Price'
        )
        
        fig_price_comp.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig_price_comp.update_layout(height=500)
        
        st.plotly_chart(fig_price_comp, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🧠 Otomatik İçgörüler")
        
        specialty_insights = []
        
        if len(specialty_agg) >= 2:
            specialty_row = specialty_agg[specialty_agg['Specialty Product'].str.contains('Specialty', case=False, na=False)]
            if len(specialty_row) > 0:
                specialty_row = specialty_row.iloc[0]
                specialty_insights.append(
                    f"Specialty ürünlerin pazar payı 2024'te %{specialty_row['Share_2024']:.2f} seviyesinde."
                )
                specialty_insights.append(
                    f"Specialty ürünler 2023'ten 2024'e {get_trend_indicator(specialty_row['Growth_23_24'])} %{abs(specialty_row['Growth_23_24']):.2f} büyüdü."
                )
        
        for insight in specialty_insights:
            st.markdown(f"<div class='insight-box'>💡 {insight}</div>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 7: FİYAT-VOLUME-MIX
    # ═══════════════════════════════════════════════════════════════════════════
    
    with tab7:
        st.markdown("<h2 class='sub-header'>📈 Fiyat-Volume-Mix Ayrıştırması</h2>", unsafe_allow_html=True)
        
        st.markdown("### 🔍 Global Price-Volume-Mix Analizi")
        
        st.markdown("#### 2022 → 2023")
        
        pvm_global_22_23 = price_volume_mix_decomposition(filtered_df, 2022, 2023)
        
        col1, col2, col3,col4 = st.columns(4)
        with col1:
        st.metric("Toplam Değişim", format_number(pvm_global_22_23['total_change']))
    with col2:
        st.metric("Volume Etkisi", format_number(pvm_global_22_23['volume_effect']))
    with col3:
        st.metric("Price Etkisi", format_number(pvm_global_22_23['price_effect']))
    with col4:
        st.metric("Mix Etkisi", format_number(pvm_global_22_23['mix_effect']))
    
    pvm_global_22_23_df = pd.DataFrame({
        'Bileşen': ['Volume', 'Price', 'Mix'],
        'Katkı': [
            pvm_global_22_23['volume_effect'],
            pvm_global_22_23['price_effect'],
            pvm_global_22_23['mix_effect']
        ]
    })
    
    fig_pvm_global_22_23 = px.bar(
        pvm_global_22_23_df,
        x='Bileşen',
        y='Katkı',
        title='Global PVM Ayrıştırması (2022 → 2023)',
        text='Katkı',
        color='Bileşen',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    
    fig_pvm_global_22_23.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig_pvm_global_22_23.update_layout(height=450, showlegend=False)
    
    st.plotly_chart(fig_pvm_global_22_23, use_container_width=True)
    
    st.markdown("#### 2023 → 2024")
    
    pvm_global_23_24 = price_volume_mix_decomposition(filtered_df, 2023, 2024)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Toplam Değişim", format_number(pvm_global_23_24['total_change']))
    with col2:
        st.metric("Volume Etkisi", format_number(pvm_global_23_24['volume_effect']))
    with col3:
        st.metric("Price Etkisi", format_number(pvm_global_23_24['price_effect']))
    with col4:
        st.metric("Mix Etkisi", format_number(pvm_global_23_24['mix_effect']))
    
    pvm_global_23_24_df = pd.DataFrame({
        'Bileşen': ['Volume', 'Price', 'Mix'],
        'Katkı': [
            pvm_global_23_24['volume_effect'],
            pvm_global_23_24['price_effect'],
            pvm_global_23_24['mix_effect']
        ]
    })
    
    fig_pvm_global_23_24 = px.bar(
        pvm_global_23_24_df,
        x='Bileşen',
        y='Katkı',
        title='Global PVM Ayrıştırması (2023 → 2024)',
        text='Katkı',
        color='Bileşen',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    
    fig_pvm_global_23_24.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig_pvm_global_23_24.update_layout(height=450, showlegend=False)
    
    st.plotly_chart(fig_pvm_global_23_24, use_container_width=True)
    
    st.markdown("#### Zincir Özet (2022 → 2024)")
    
    total_change_global_chain = pvm_global_22_23['total_change'] + pvm_global_23_24['total_change']
    volume_global_chain = pvm_global_22_23['volume_effect'] + pvm_global_23_24['volume_effect']
    price_global_chain = pvm_global_22_23['price_effect'] + pvm_global_23_24['price_effect']
    mix_global_chain = pvm_global_22_23['mix_effect'] + pvm_global_23_24['mix_effect']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Toplam Değişim (Zincir)", format_number(total_change_global_chain))
    with col2:
        st.metric("Volume Etkisi (Zincir)", format_number(volume_global_chain))
    with col3:
        st.metric("Price Etkisi (Zincir)", format_number(price_global_chain))
    with col4:
        st.metric("Mix Etkisi (Zincir)", format_number(mix_global_chain))
    
    pvm_chain_df = pd.DataFrame({
        'Bileşen': ['Volume', 'Price', 'Mix'],
        'Katkı': [volume_global_chain, price_global_chain, mix_global_chain]
    })
    
    fig_pvm_chain = px.bar(
        pvm_chain_df,
        x='Bileşen',
        y='Katkı',
        title='Global PVM Zincir Özet (2022 → 2024)',
        text='Katkı',
        color='Bileşen',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
    )
    
    fig_pvm_chain.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig_pvm_chain.update_layout(height=450, showlegend=False)
    
    st.plotly_chart(fig_pvm_chain, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🌍 Ülke Bazlı PVM Analizi")
    
    pvm_country_22_23 = price_volume_mix_decomposition(filtered_df, 2022, 2023, groupby_cols=['Country'])
    pvm_country_23_24 = price_volume_mix_decomposition(filtered_df, 2023, 2024, groupby_cols=['Country'])
    
    pvm_country_22_23 = pvm_country_22_23.sort_values('total_change', ascending=False).head(10)
    pvm_country_23_24 = pvm_country_23_24.sort_values('total_change', ascending=False).head(10)
    
    st.markdown("#### Top 10 Ülkeler - PVM Ayrıştırması (2022 → 2023)")
    
    fig_pvm_country_22_23 = go.Figure()
    
    fig_pvm_country_22_23.add_trace(go.Bar(
        name='Volume',
        x=pvm_country_22_23['Country'],
        y=pvm_country_22_23['volume_effect'],
        marker_color='#1f77b4'
    ))
    
    fig_pvm_country_22_23.add_trace(go.Bar(
        name='Price',
        x=pvm_country_22_23['Country'],
        y=pvm_country_22_23['price_effect'],
        marker_color='#ff7f0e'
    ))
    
    fig_pvm_country_22_23.add_trace(go.Bar(
        name='Mix',
        x=pvm_country_22_23['Country'],
        y=pvm_country_22_23['mix_effect'],
        marker_color='#2ca02c'
    ))
    
    fig_pvm_country_22_23.update_layout(
        barmode='stack',
        title='Top 10 Ülkeler - PVM Ayrıştırması (2022 → 2023)',
        xaxis_title='Ülke',
        yaxis_title='Katkı (USD MNF)',
        height=500
    )
    
    st.plotly_chart(fig_pvm_country_22_23, use_container_width=True)
    
    st.markdown("#### Top 10 Ülkeler - PVM Ayrıştırması (2023 → 2024)")
    
    fig_pvm_country_23_24 = go.Figure()
    
    fig_pvm_country_23_24.add_trace(go.Bar(
        name='Volume',
        x=pvm_country_23_24['Country'],
        y=pvm_country_23_24['volume_effect'],
        marker_color='#1f77b4'
    ))
    
    fig_pvm_country_23_24.add_trace(go.Bar(
        name='Price',
        x=pvm_country_23_24['Country'],
        y=pvm_country_23_24['price_effect'],
        marker_color='#ff7f0e'
    ))
    
    fig_pvm_country_23_24.add_trace(go.Bar(
        name='Mix',
        x=pvm_country_23_24['Country'],
        y=pvm_country_23_24['mix_effect'],
        marker_color='#2ca02c'
    ))
    
    fig_pvm_country_23_24.update_layout(
        barmode='stack',
        title='Top 10 Ülkeler - PVM Ayrıştırması (2023 → 2024)',
        xaxis_title='Ülke',
        yaxis_title='Katkı (USD MNF)',
        height=500
    )
    
    st.plotly_chart(fig_pvm_country_23_24, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### ⚗️ Molekül Bazlı PVM Analizi")
    
    pvm_molecule_22_23 = price_volume_mix_decomposition(filtered_df, 2022, 2023, groupby_cols=['Molecule'])
    pvm_molecule_23_24 = price_volume_mix_decomposition(filtered_df, 2023, 2024, groupby_cols=['Molecule'])
    
    pvm_molecule_22_23 = pvm_molecule_22_23.sort_values('total_change', ascending=False).head(10)
    pvm_molecule_23_24 = pvm_molecule_23_24.sort_values('total_change', ascending=False).head(10)
    
    st.markdown("#### Top 10 Moleküller - PVM Ayrıştırması (2022 → 2023)")
    
    fig_pvm_molecule_22_23 = go.Figure()
    
    fig_pvm_molecule_22_23.add_trace(go.Bar(
        name='Volume',
        x=pvm_molecule_22_23['Molecule'],
        y=pvm_molecule_22_23['volume_effect'],
        marker_color='#1f77b4'
    ))
    
    fig_pvm_molecule_22_23.add_trace(go.Bar(
        name='Price',
        x=pvm_molecule_22_23['Molecule'],
        y=pvm_molecule_22_23['price_effect'],
        marker_color='#ff7f0e'
    ))
    
    fig_pvm_molecule_22_23.add_trace(go.Bar(
        name='Mix',
        x=pvm_molecule_22_23['Molecule'],
        y=pvm_molecule_22_23['mix_effect'],
        marker_color='#2ca02c'
    ))
    
    fig_pvm_molecule_22_23.update_layout(
        barmode='stack',
        title='Top 10 Moleküller - PVM Ayrıştırması (2022 → 2023)',
        xaxis_title='Molekül',
        yaxis_title='Katkı (USD MNF)',
        height=500
    )
    
    st.plotly_chart(fig_pvm_molecule_22_23, use_container_width=True)
    
    st.markdown("#### Top 10 Moleküller - PVM Ayrıştırması (2023 → 2024)")
    
    fig_pvm_molecule_23_24 = go.Figure()
    
    fig_pvm_molecule_23_24.add_trace(go.Bar(
        name='Volume',
        x=pvm_molecule_23_24['Molecule'],
        y=pvm_molecule_23_24['volume_effect'],
        marker_color='#1f77b4'
    ))
    
    fig_pvm_molecule_23_24.add_trace(go.Bar(
        name='Price',
        x=pvm_molecule_23_24['Molecule'],
        y=pvm_molecule_23_24['price_effect'],
        marker_color='#ff7f0e'
    ))
    
    fig_pvm_molecule_23_24.add_trace(go.Bar(
        name='Mix',
        x=pvm_molecule_23_24['Molecule'],
        y=pvm_molecule_23_24['mix_effect'],
        marker_color='#2ca02c'
    ))
    
    fig_pvm_molecule_23_24.update_layout(
        barmode='stack',
        title='Top 10 Moleküller - PVM Ayrıştırması (2023 → 2024)',
        xaxis_title='Molekül',
        yaxis_title='Katkı (USD MNF)',
        height=500
    )
    
    st.plotly_chart(fig_pvm_molecule_23_24, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 8: OTOMATİK İÇGÖRÜ MOTORU
# ═══════════════════════════════════════════════════════════════════════════

with tab8:
    st.markdown("<h2 class='sub-header'>🧠 Otomatik İçgörü Motoru</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Bu bölüm, verilerinizden otomatik olarak anlamlı içgörüler üretir.
    Ülke, molekül veya corporation bazlı analizler yaparak stratejik kararlarınızı destekler.
    """)
    
    st.markdown("---")
    
    insight_type = st.selectbox(
        "İçgörü Tipi Seçin",
        options=["Global Özet", "Ülke Bazlı", "Molekül Bazlı", "Corporation Bazlı"],
        key="insight_type"
    )
    
    if insight_type == "Global Özet":
        st.markdown("### 🌍 Global Pazar İçgörüleri")
        
        global_insights = generate_executive_insights(filtered_df)
        
        for insight in global_insights:
            st.markdown(f"<div class='success-box'>✅ {insight}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Ek Stratejik İçgörüler")
        
        units_2022 = filtered_df["MAT Q3 2022\nUnits"].sum()
        units_2023 = filtered_df["MAT Q3 2023\nUnits"].sum()
        units_2024 = filtered_df["MAT Q3 2024\nUnits"].sum()
        
        volume_growth_22_23 = calculate_growth_rate(units_2023, units_2022)
        volume_growth_23_24 = calculate_growth_rate(units_2024, units_2023)
        
        avg_price_2022_global = safe_division(usd_2022, filtered_df["MAT Q3 2022\nStandard Units"].sum())
        avg_price_2023_global = safe_division(usd_2023, filtered_df["MAT Q3 2023\nStandard Units"].sum())
        avg_price_2024_global = safe_division(usd_2024, filtered_df["MAT Q3 2024\nStandard Units"].sum())
        
        price_growth_22_23 = calculate_growth_rate(avg_price_2023_global, avg_price_2022_global)
        price_growth_23_24 = calculate_growth_rate(avg_price_2024_global, avg_price_2023_global)
        
        st.markdown(f"<div class='info-box'>📦 Global hacim 2022'den 2023'e {get_trend_indicator(volume_growth_22_23)} %{abs(volume_growth_22_23):.2f} değişti.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-box'>📦 Global hacim 2023'ten 2024'e {get_trend_indicator(volume_growth_23_24)} %{abs(volume_growth_23_24):.2f} değişti.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-box'>💵 Ortalama fiyat 2022'den 2023'e {get_trend_indicator(price_growth_22_23)} %{abs(price_growth_22_23):.2f} değişti.</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-box'>💵 Ortalama fiyat 2023'ten 2024'e {get_trend_indicator(price_growth_23_24)} %{abs(price_growth_23_24):.2f} değişti.</div>", unsafe_allow_html=True)
        
        region_2024 = filtered_df.groupby('Region')["MAT Q3 2024\nUSD MNF"].sum().sort_values(ascending=False)
        if len(region_2024) > 0:
            top_region = region_2024.index[0]
            top_region_share = (region_2024.iloc[0] / usd_2024) * 100
            st.markdown(f"<div class='info-box'>🗺️ En büyük bölge {top_region} olup, global satışların %{top_region_share:.2f}'sini oluşturuyor.</div>", unsafe_allow_html=True)
    
    elif insight_type == "Ülke Bazlı":
        st.markdown("### 🏳️ Ülke Bazlı İçgörüler")
        
        available_countries_insight = sorted(filtered_df['Country'].unique())
        
        selected_country_insight = st.selectbox(
            "Ülke Seçin",
            options=available_countries_insight,
            key="country_insight"
        )
        
        country_insights = generate_country_insights(filtered_df, selected_country_insight)
        
        for insight in country_insights:
            st.markdown(f"<div class='success-box'>✅ {insight}</div>", unsafe_allow_html=True)
    
    elif insight_type == "Molekül Bazlı":
        st.markdown("### ⚗️ Molekül Bazlı İçgörüler")
        
        available_molecules_insight = sorted(filtered_df['Molecule'].unique())
        
        selected_molecule_insight = st.selectbox(
            "Molekül Seçin",
            options=available_molecules_insight,
            key="molecule_insight"
        )
        
        molecule_insights = generate_molecule_insights(filtered_df, selected_molecule_insight)
        
        for insight in molecule_insights:
            st.markdown(f"<div class='success-box'>✅ {insight}</div>", unsafe_allow_html=True)
    
    elif insight_type == "Corporation Bazlı":
        st.markdown("### 🏛️ Corporation Bazlı İçgörüler")
        
        available_corporations_insight = sorted(filtered_df['Corporation'].unique())
        
        selected_corporation_insight = st.selectbox(
            "Corporation Seçin",
            options=available_corporations_insight,
            key="corporation_insight"
        )
        
        corp_insights = generate_corporation_insights(filtered_df, selected_corporation_insight)
        
        for insight in corp_insights:
            st.markdown(f"<div class='success-box'>✅ {insight}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 İçgörü Özeti")
    
    st.info("""
    **Otomatik İçgörü Motoru Özellikleri:**
    
    - 🔍 Verinizden otomatik olarak anlamlı pattern'ler keşfeder
    - 📊 3 yıllık trend analizleri yapar
    - 🎯 Büyüme ve pazar payı değişimlerini raporlar
    - 🌍 Global ve lokal karşılaştırmalar sunar
    - 💬 Doğal Türkçe dilde içgörüler üretir
    """)
    if name == "main":
main()
